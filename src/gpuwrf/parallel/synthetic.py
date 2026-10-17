"""Synthetic-from-real large domains for decomposition/scaling tests: mirror-tile a real loaded case.

``mirror_tile_case(carry, namelist, mx, my)`` repeats the real domain mx x my times, mirroring every odd block
(mass cell j -> n-1-j, face f -> n-f): every staggering stays continuous across the seams (no sign flips, so the
flow is smooth but not a physical mirror image), and the lateral-boundary strips stay consistent with the new
edges (the east edge of a mirrored block IS the original west edge). Work per column equals the real case.
Not a forecast: a numerically well-posed, real-structured workload for timing and decomposition checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gpuwrf.parallel import decomposition as dc


def _np_or_jnp(a):
    return np if isinstance(a, np.ndarray) else __import__("jax.numpy", fromlist=["x"])


def _tile_axis(a, axis: int, n: int, reps: int, face: bool):
    """Concatenate reps blocks of length n (n+1 for faces) along axis, odd blocks mirrored; seams shared."""
    xp = _np_or_jnp(a)
    blocks = []
    for b in range(reps):
        blk = xp.flip(a, axis=axis) if b % 2 else a
        if face and b < reps - 1:
            blk = xp.take(blk, xp.arange(n), axis=axis) if xp is not np else np.take(blk, np.arange(n), axis=axis)
        blocks.append(blk)
    return xp.concatenate(blocks, axis=axis)


def _tile_h(a, spec: dc.DecompositionSpec, mx: int, my: int, sy: int, sx: int):
    a = _tile_axis(a, a.ndim - 1, spec.nx, mx, bool(sx))
    return _tile_axis(a, a.ndim - 2, spec.ny, my, bool(sy))


def _tile_flat(a, spec: dc.DecompositionSpec, mx: int, my: int, axis: int):
    shp = tuple(a.shape)
    b = a.reshape(shp[:axis] + (spec.ny, spec.nx) + shp[axis + 1:])
    b = _tile_axis(b, axis + 1, spec.nx, mx, False)
    b = _tile_axis(b, axis, spec.ny, my, False)
    return b.reshape(shp[:axis] + (spec.ny * my * spec.nx * mx,) + shp[axis + 1:])


def _tile_bdy(a, spec: dc.DecompositionSpec, mx: int, my: int, name: str):
    """(time, 4 W/E/S/N, bw, z, side_len) strips of the big domain. Along-edge staggering: v on W/E, u on S/N."""
    xp = _np_or_jnp(a)
    nx2, ny2 = spec.nx * mx, spec.ny * my
    side2 = max(nx2, ny2) + 1
    out = []
    for side in range(4):
        along_y = side < 2
        n, reps = (spec.ny, my) if along_y else (spec.nx, mx)
        face = (name == "v_bdy") if along_y else (name == "u_bdy")
        # the big domain's E (N) edge is the original E (N) edge if its last block is unmirrored, else the W (S) one
        src_side = side
        if side == 1 and (mx - 1) % 2:
            src_side = 0
        if side == 3 and (my - 1) % 2:
            src_side = 2
        strip = a[:, src_side, ..., : n + (1 if face else 0)]
        strip = _tile_axis(strip, strip.ndim - 1, n, reps, face)
        pad = side2 - int(strip.shape[-1])
        if pad > 0:
            strip = xp.concatenate([strip, xp.repeat(strip[..., -1:], pad, axis=-1)], axis=-1)
        out.append(strip)
    return xp.stack(out, axis=1)


def _grid(grid, fn, mx, my, memo):
    from gpuwrf.contracts.grid import GridSpec, Projection
    import dataclasses

    p = grid.projection
    nx2, ny2 = p.nx * mx, p.ny * my
    projection = Projection(p.kind, p.lat_0, p.lon_0, p.dx_m, p.dy_m, nx2, ny2)
    terrain = dataclasses.replace(grid.terrain, shape=(ny2, nx2), source_path=f"{grid.terrain.source_path}:mirror{mx}x{my}")
    return GridSpec(projection, terrain, grid.vertical, grid.bc, grid.eta_levels, fn(grid.terrain_height),
                    metrics=dc._map_obj(grid.metrics, fn, lambda g, m: _grid(g, fn, mx, my, m), memo),
                    halo_width=grid.halo_width, staggering=grid.staggering)


def mirror_tile_case(carry, namelist, mx: int, my: int) -> tuple[Any, Any]:
    """(carry, namelist) of the mx x my mirror-tiled domain (host/device arrays follow the inputs' type)."""
    import jax

    nx, ny = int(namelist.grid.nx), int(namelist.grid.ny)
    spec = dc.DecompositionSpec.build(nx, ny, 1, 1, 0)
    state = getattr(carry, "state", None)
    names = {}
    if state is not None:
        for n in type(state).__slots__:
            v = getattr(state, n, None)
            if n.endswith("_bdy") and v is not None:
                names[id(v)] = n

    def fn(a):
        kind = dc.classify(np.shape(a), spec)
        if kind[0] == "h":
            return _tile_h(a, spec, mx, my, kind[1], kind[2])
        if kind[0] == "flat":
            return _tile_flat(a, spec, mx, my, kind[1])
        if kind[0] == "bdy":
            return _tile_bdy(a, spec, mx, my, names.get(id(a), ""))
        return a

    big_carry = jax.tree_util.tree_map(lambda x: fn(x) if dc._is_array(x) else x, carry)
    memo: dict = {}
    big_nl = dc._map_obj(namelist, fn, lambda g, m: _grid(g, fn, mx, my, m), memo)
    return big_carry, big_nl


__all__ = ["mirror_tile_case"]
