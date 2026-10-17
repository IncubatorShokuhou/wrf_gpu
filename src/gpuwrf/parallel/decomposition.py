"""Deep-halo horizontal decomposition of ONE domain: partition spec, tile slicing, exchange plan, gather.

Design (ADR-O2-MULTIGPU-DECOMPOSITION): every tile runs the UNMODIFIED single-domain step program on its
owned columns plus a halo of width ``halo`` on each internal side (columns stay intact). The step treats
every array edge as a domain edge (specified/relaxation zones, edge padding, degraded advection), so the
outer rows of an internal halo are contaminated each step. Admission needs the dependency cone of ALL
enabled operators, including radiation before dynamics (tile_driver.configuration_halo). Numerical
partition consistency must be checked on each selected path; GPU release kernels remain unqualified.

Leaf classes (by shape, global ``nx``/``ny``; pytree paths carry no field names):
  ``h``    trailing dims (ny + sy, nx + sx), sy/sx in {0, 1}: mass, x-face, y-face (and corner) fields
  ``flat`` one axis of length ny * nx (C order, e.g. GWDO statics): reshaped, sliced, flattened back
  ``bdy``  lateral-boundary strips (time, 4 sides W/E/S/N, bdy_width, z, side_len): the along-edge axis is
           re-sliced by the tile's offset (y for W/E, x for S/N); every tile gets all four sides, so true
           edges are exact and internal edges only feed the (overwritten) halo
  ``rep``  everything else (scalars, vertical profiles, tables): replicated unchanged
Ownership: mass cells [x0, x1) per tile; x faces [x0, x1) plus face nx for the last tile column (same in y),
so owned regions partition every staggering. The halo of a tile is exactly the union of its range with the
other tiles' owned regions, which gives the pairwise exchange plan below (corners and multi-hop included).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

SIDES = ("W", "E", "S", "N")  # boundary-strip side order (coupling/boundary_apply.SIDES)


@dataclass(frozen=True)
class Tile:
    """One rank's piece of the domain. Ranges are global mass-cell index ranges [start, stop)."""

    rank: int
    ix: int
    iy: int
    own_x: tuple[int, int]
    own_y: tuple[int, int]
    ext_x: tuple[int, int]
    ext_y: tuple[int, int]

    @property
    def nx(self) -> int:
        return self.ext_x[1] - self.ext_x[0]

    @property
    def ny(self) -> int:
        return self.ext_y[1] - self.ext_y[0]


def _split(n: int, parts: int) -> list[tuple[int, int]]:
    """Even split like WRF's patch decomposition: the first n % parts patches get one extra cell."""
    base, extra = divmod(int(n), int(parts))
    out, start = [], 0
    for i in range(parts):
        stop = start + base + (1 if i < extra else 0)
        out.append((start, stop))
        start = stop
    return out


@dataclass(frozen=True)
class DecompositionSpec:
    """px x py tiles of an nx x ny (mass) domain with a deep halo of ``halo`` cells on internal sides."""

    nx: int
    ny: int
    px: int
    py: int
    halo: int
    bdy_width: int = 5
    tiles: tuple[Tile, ...] = ()

    @classmethod
    def build(cls, nx: int, ny: int, px: int, py: int, halo: int, *, bdy_width: int = 5) -> "DecompositionSpec":
        if px < 1 or py < 1 or halo < 0:
            raise ValueError("px, py must be >= 1 and halo >= 0")
        if px > nx or py > ny:
            raise ValueError(f"cannot split {nx} x {ny} columns into {px} x {py} tiles")
        tiles = []
        for iy, (y0, y1) in enumerate(_split(ny, py)):
            for ix, (x0, x1) in enumerate(_split(nx, px)):
                ext_x = (max(0, x0 - halo), min(nx, x1 + halo))
                ext_y = (max(0, y0 - halo), min(ny, y1 + halo))
                tiles.append(Tile(len(tiles), ix, iy, (x0, x1), (y0, y1), ext_x, ext_y))
        return cls(int(nx), int(ny), int(px), int(py), int(halo), int(bdy_width), tuple(tiles))

    @property
    def side_len(self) -> int:
        return max(self.nx, self.ny) + 1

    # --- per-staggering ranges (s = 0 mass, 1 face) -------------------------------------------------
    def owned_x(self, t: Tile, s: int) -> tuple[int, int]:
        return t.own_x[0], t.own_x[1] + (1 if s and t.own_x[1] == self.nx else 0)

    def owned_y(self, t: Tile, s: int) -> tuple[int, int]:
        return t.own_y[0], t.own_y[1] + (1 if s and t.own_y[1] == self.ny else 0)

    @staticmethod
    def range_x(t: Tile, s: int) -> tuple[int, int]:
        return t.ext_x[0], t.ext_x[1] + s

    @staticmethod
    def range_y(t: Tile, s: int) -> tuple[int, int]:
        return t.ext_y[0], t.ext_y[1] + s

    def describe(self) -> dict:
        return {"nx": self.nx, "ny": self.ny, "px": self.px, "py": self.py, "halo": self.halo,
                "tiles": [{"rank": t.rank, "own_x": t.own_x, "own_y": t.own_y, "ext_x": t.ext_x, "ext_y": t.ext_y,
                           "redundancy": round(t.nx * t.ny / ((t.own_x[1] - t.own_x[0]) * (t.own_y[1] - t.own_y[0])), 4)}
                          for t in self.tiles]}


# --- leaf classification -----------------------------------------------------------------------------

def classify(shape: tuple[int, ...], spec: DecompositionSpec) -> tuple:
    shape = tuple(int(d) for d in shape)
    nx, ny = spec.nx, spec.ny
    if (len(shape) == 5 and shape[1] == 4 and shape[2] == spec.bdy_width and shape[-1] == spec.side_len):
        return ("bdy",)
    if len(shape) >= 2:
        sy, sx = shape[-2] - ny, shape[-1] - nx
        if sy in (0, 1) and sx in (0, 1):
            return ("h", sy, sx)
    flat = [i for i, d in enumerate(shape) if d == nx * ny]
    if len(flat) == 1 and nx * ny > max(nx, ny) + 1:
        return ("flat", flat[0])
    return ("rep",)


def _slice_h(a, t: Tile, spec: DecompositionSpec, sy: int, sx: int):
    y0, y1 = spec.range_y(t, sy)
    x0, x1 = spec.range_x(t, sx)
    return a[..., y0:y1, x0:x1]


def _slice_flat(a, t: Tile, spec: DecompositionSpec, axis: int):
    shp = tuple(a.shape)
    b = a.reshape(shp[:axis] + (spec.ny, spec.nx) + shp[axis + 1:])
    b = b[(slice(None),) * axis + (slice(t.ext_y[0], t.ext_y[1]), slice(t.ext_x[0], t.ext_x[1]))]
    return b.reshape(shp[:axis] + (t.ny * t.nx,) + shp[axis + 1:])


def _slice_bdy(a, t: Tile, spec: DecompositionSpec):
    """(time, 4, bw, z, side_len) -> the tile's strips: along-edge offset y0 (W/E) / x0 (S/N), padded with the
    last global value (finite, plausible) where the tile's padded side reaches past the global strip."""
    is_jax = not isinstance(a, np.ndarray)
    xp = __import__("jax.numpy", fromlist=["x"]) if is_jax else np
    lt = max(t.nx, t.ny) + 1
    sides = []
    for side, off in ((0, t.ext_y[0]), (1, t.ext_y[0]), (2, t.ext_x[0]), (3, t.ext_x[0])):
        strip = a[:, side, ..., off:off + lt]
        short = lt - int(strip.shape[-1])
        if short > 0:
            strip = xp.concatenate([strip, xp.repeat(strip[..., -1:], short, axis=-1)], axis=-1)
        sides.append(strip)
    return xp.stack(sides, axis=1)


def slice_array(a, t: Tile, spec: DecompositionSpec):
    kind = classify(np.shape(a), spec)
    if kind[0] == "h":
        return _slice_h(a, t, spec, kind[1], kind[2])
    if kind[0] == "flat":
        return _slice_flat(a, t, spec, kind[1])
    if kind[0] == "bdy":
        return _slice_bdy(a, t, spec)
    return a


def _is_array(x) -> bool:
    if isinstance(x, np.ndarray):
        return True
    try:
        import jax
    except ImportError:  # pragma: no cover
        return False
    return isinstance(x, jax.Array)


def slice_tree(tree, t: Tile, spec: DecompositionSpec):
    """Slice every array leaf of a registered pytree (the carry, namelist children)."""
    import jax

    return jax.tree_util.tree_map(lambda leaf: slice_array(leaf, t, spec) if _is_array(leaf) else leaf, tree)


def _tile_grid(grid, t: Tile, spec: DecompositionSpec, memo: dict):
    from gpuwrf.contracts.grid import GridSpec, Projection

    proj = grid.projection
    projection = Projection(proj.kind, proj.lat_0, proj.lon_0, proj.dx_m, proj.dy_m, t.nx, t.ny)
    terrain = dataclasses.replace(grid.terrain, shape=(t.ny, t.nx),
                                  source_path=f"{grid.terrain.source_path}:tile{t.rank}")
    fn = lambda a: slice_array(a, t, spec)  # noqa: E731
    gfn = lambda g, m: _tile_grid(g, t, spec, m)  # noqa: E731
    return GridSpec(projection, terrain, grid.vertical, grid.bc, grid.eta_levels,
                    _map_obj(grid.terrain_height, fn, gfn, memo), metrics=_map_obj(grid.metrics, fn, gfn, memo),
                    halo_width=grid.halo_width, staggering=grid.staggering)


def _slice_obj(x: Any, t: Tile, spec: DecompositionSpec, memo: dict) -> Any:
    return _map_obj(x, lambda a: slice_array(a, t, spec), lambda g, m: _tile_grid(g, t, spec, m), memo)


def _map_obj(x: Any, fn, grid_fn, memo: dict) -> Any:
    """Generic recursive array transform for the namelist (dataclasses, registered pytrees, identity holders,
    containers); GridSpec is rebuilt by ``grid_fn(grid, memo)`` (its nx/ny/terrain shape change). Shared
    sub-objects stay shared (memo by identity); unchanged sub-objects are returned as is."""
    key = id(x)
    if key in memo:
        return memo[key][1]
    out = _map_obj_uncached(x, fn, grid_fn, memo)
    memo[key] = (x, out)  # keep x alive so its id is not reused during this walk
    return out


def _map_obj_uncached(x: Any, fn, grid_fn, memo: dict) -> Any:
    import jax

    if x is None or isinstance(x, (bool, int, float, complex, str, bytes, np.generic)):
        return x
    if _is_array(x):
        return fn(x)
    from gpuwrf.contracts.grid import GridSpec

    if isinstance(x, GridSpec):
        return grid_fn(x, memo)
    rec = lambda v: _map_obj(v, fn, grid_fn, memo)  # noqa: E731
    cls = type(x)
    if cls.__name__ == "_StaticHolder" and hasattr(x, "value"):
        new = rec(x.value)
        return x if new is x.value else cls(new)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        changes = {}
        for f in dataclasses.fields(x):
            if not f.init:
                continue
            old = getattr(x, f.name)
            new = rec(old)
            if new is not old:
                changes[f.name] = new
        return dataclasses.replace(x, **changes) if changes else x
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        vals = [rec(v) for v in x]
        return x if all(a is b for a, b in zip(vals, x)) else cls(*vals)
    if isinstance(x, (tuple, list)):
        vals = [rec(v) for v in x]
        return x if all(a is b for a, b in zip(vals, x)) else cls(vals)
    if isinstance(x, dict):
        return {k: rec(v) for k, v in x.items()}
    leaves, treedef = jax.tree_util.tree_flatten(x)
    if treedef.num_leaves and not (len(leaves) == 1 and leaves[0] is x):
        new = [rec(v) for v in leaves]
        if all(a is b for a, b in zip(new, leaves)):
            return x
        return jax.tree_util.tree_unflatten(treedef, new)
    return x


def slice_namelist(namelist, t: Tile, spec: DecompositionSpec):
    """The tile's OperationalNamelist: every horizontal array (children AND static aux: GridSpec terrain/
    metrics, Noah-MP static/params, radiation/GWDO statics, lower boundary) sliced; scalars unchanged."""
    out = _slice_obj(namelist, t, spec, {})
    leftover = audit_global_arrays(out, spec, tile=t)
    if leftover:
        raise ValueError(f"tile namelist still holds global-size horizontal arrays: {leftover[:8]}")
    return out


def audit_global_arrays(obj: Any, spec: DecompositionSpec, *, tile: Tile | None = None, limit: int = 64) -> list[str]:
    """Paths of arrays that still look like GLOBAL horizontal fields (h/flat class at global size)."""
    import jax

    found: list[str] = []
    seen: set[int] = set()

    def walk(x, path):
        if len(found) >= limit or id(x) in seen:
            return
        seen.add(id(x))
        if _is_array(x):
            kind = classify(np.shape(x), spec)
            # Full-extent tiles legitimately retain global-size arrays. Check the expected per-tile
            # shape, not just whether the array happens to match the original domain.
            expected = tuple(np.shape(x))
            if tile is not None and kind[0] == "h":
                expected = expected[:-2] + (tile.ny + kind[1], tile.nx + kind[2])
            elif tile is not None and kind[0] == "flat":
                expected = expected[:kind[1]] + (tile.ny * tile.nx,) + expected[kind[1] + 1:]
            if (kind[0] in ("h", "flat") and not (spec.px == 1 and spec.py == 1)
                    and (tile is None or tuple(np.shape(x)) != expected)):
                found.append(f"{path} {tuple(np.shape(x))}")
            return
        if x is None or isinstance(x, (bool, int, float, complex, str, bytes, np.generic)):
            return
        if type(x).__name__ == "_StaticHolder" and hasattr(x, "value"):
            walk(x.value, path + ".value")
        elif dataclasses.is_dataclass(x) and not isinstance(x, type):
            for f in dataclasses.fields(x):
                walk(getattr(x, f.name), f"{path}.{f.name}")
        elif isinstance(x, (tuple, list)):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}[{k!r}]")
        else:
            leaves = jax.tree_util.tree_leaves(x)
            if not (len(leaves) == 1 and leaves[0] is x):
                for i, v in enumerate(leaves):
                    walk(v, f"{path}<{i}>")

    walk(obj, "nl")
    return found


# --- exchange plan --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Transfer:
    """One leaf's rectangle copied from rank ``src`` (owner) into rank ``dst``'s halo; local index ranges."""

    leaf: int
    src_y: tuple[int, int]
    src_x: tuple[int, int]
    dst_y: tuple[int, int]
    dst_x: tuple[int, int]


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int] | None:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None


def exchange_plan(spec: DecompositionSpec, kinds: Iterable[tuple]) -> dict[tuple[int, int], list[Transfer]]:
    """{(src, dst): [Transfer]} for every ``h`` leaf: dst.range ∩ src.owned per staggering (src != dst)."""
    kinds = list(kinds)
    plan: dict[tuple[int, int], list[Transfer]] = {}
    for dst in spec.tiles:
        for src in spec.tiles:
            if src.rank == dst.rank:
                continue
            moves = []
            for leaf, kind in enumerate(kinds):
                if kind[0] != "h":
                    continue
                sy, sx = kind[1], kind[2]
                oy = _overlap(spec.range_y(dst, sy), spec.owned_y(src, sy))
                ox = _overlap(spec.range_x(dst, sx), spec.owned_x(src, sx))
                if oy is None or ox is None:
                    continue
                moves.append(Transfer(leaf,
                                      (oy[0] - src.ext_y[0], oy[1] - src.ext_y[0]), (ox[0] - src.ext_x[0], ox[1] - src.ext_x[0]),
                                      (oy[0] - dst.ext_y[0], oy[1] - dst.ext_y[0]), (ox[0] - dst.ext_x[0], ox[1] - dst.ext_x[0])))
            if moves:
                plan[(src.rank, dst.rank)] = moves
    return plan


def plan_bytes(plan: dict, leaves: list) -> dict:
    """Bytes per (src, dst) message for leaf arrays ``leaves`` (global or any tile leaves: only the
    leading dims and itemsize are read)."""
    out = {}
    for pair, moves in plan.items():
        total = 0
        for m in moves:
            leaf = leaves[m.leaf]
            lead = int(np.prod(np.shape(leaf)[:-2], dtype=np.int64))
            total += lead * (m.src_y[1] - m.src_y[0]) * (m.src_x[1] - m.src_x[0]) * np.dtype(leaf.dtype).itemsize
        out[pair] = int(total)
    return out


# --- gather -----------------------------------------------------------------------------------------------

def gather_leaves(tile_leaves: list[list], template: list, kinds: list[tuple], spec: DecompositionSpec) -> list:
    """Global host leaves from per-tile leaves: owned regions of every ``h`` leaf; ``rep``/``bdy``/``flat``
    leaves are taken from the template (``bdy``/``flat`` are static) or from tile 0 (``rep``)."""
    out = []
    for i, kind in enumerate(kinds):
        if kind[0] != "h":
            out.append(np.asarray(tile_leaves[0][i]) if kind[0] == "rep" else np.asarray(template[i]))
            continue
        sy, sx = kind[1], kind[2]
        g = np.empty(np.shape(template[i]), dtype=np.asarray(tile_leaves[0][i]).dtype)
        for t in spec.tiles:
            y0, y1 = spec.owned_y(t, sy)
            x0, x1 = spec.owned_x(t, sx)
            local = np.asarray(tile_leaves[t.rank][i])
            g[..., y0:y1, x0:x1] = local[..., y0 - t.ext_y[0]:y1 - t.ext_y[0], x0 - t.ext_x[0]:x1 - t.ext_x[0]]
        out.append(g)
    return out


__all__ = [
    "DecompositionSpec", "SIDES", "Tile", "Transfer", "audit_global_arrays", "classify", "exchange_plan",
    "gather_leaves", "plan_bytes", "slice_array", "slice_namelist", "slice_tree",
]
