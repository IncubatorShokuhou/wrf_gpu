"""Deep-halo decomposition bookkeeping (lane o2-multigpu M2): partition, halo exchange plan, boundary strips, gather.

Pure NumPy except the jitted pack/unpack round trip; the real-operator virtual-rank check is
proofs/o2_multigpu/virtual_rank_check.py (minutes of CPU per arm, not a unit test)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwrf.parallel import decomposition as dc


def _fields(nx, ny, rng):
    return [rng.standard_normal((3, ny, nx)), rng.standard_normal((2, ny, nx + 1)), rng.standard_normal((ny + 1, nx)),
            rng.standard_normal((ny + 1, nx + 1)).astype(np.float32), rng.integers(0, 9, (4, ny, nx)).astype(np.int32)]


def _simulate_exchange(spec, plan, tiles_leaves):
    """Apply the plan with NumPy: copy each owner's rectangle into the receiver's halo."""
    out = [[x.copy() for x in leaves] for leaves in tiles_leaves]
    for (src, dst), moves in plan.items():
        for m in moves:
            out[dst][m.leaf][..., m.dst_y[0]:m.dst_y[1], m.dst_x[0]:m.dst_x[1]] = \
                tiles_leaves[src][m.leaf][..., m.src_y[0]:m.src_y[1], m.src_x[0]:m.src_x[1]]
    return out


def _halo_mask(spec, t, sy, sx, shape):
    """True on the tile's halo (cells not owned by the tile) for one staggering."""
    m = np.ones(shape[-2:], dtype=bool)
    y0, y1 = spec.owned_y(t, sy)
    x0, x1 = spec.owned_x(t, sx)
    m[y0 - t.ext_y[0]:y1 - t.ext_y[0], x0 - t.ext_x[0]:x1 - t.ext_x[0]] = False
    return m


@pytest.mark.parametrize("nx,ny,px,py,halo", [(41, 29, 2, 1, 7), (41, 29, 1, 3, 5), (41, 29, 3, 2, 6), (40, 30, 4, 3, 13),
                                              (17, 11, 4, 2, 9)])
def test_exchange_refills_every_halo_cell_exactly_including_corners_and_multihop(nx, ny, px, py, halo):
    rng = np.random.default_rng(nx * 100 + px * 10 + py)
    spec = dc.DecompositionSpec.build(nx, ny, px, py, halo)
    glob = _fields(nx, ny, rng)
    kinds = [dc.classify(x.shape, spec) for x in glob]
    assert [k[0] for k in kinds] == ["h"] * 5 and [k[1:] for k in kinds] == [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0)]
    exact = [[dc.slice_array(x, t, spec) for x in glob] for t in spec.tiles]
    poisoned = []
    for t, leaves in zip(spec.tiles, exact):
        bad = []
        for x, k in zip(leaves, kinds):
            y = x.copy()
            y[..., _halo_mask(spec, t, k[1], k[2], x.shape)] = -999
            bad.append(y)
        poisoned.append(bad)
    plan = dc.exchange_plan(spec, kinds)
    refilled = _simulate_exchange(spec, plan, poisoned)
    for t in spec.tiles:
        for a, b in zip(refilled[t.rank], exact[t.rank]):
            np.testing.assert_array_equal(a, b)
    # owned regions partition every staggering -> gather reproduces the global fields
    gathered = dc.gather_leaves(exact, glob, kinds, spec)
    for a, b in zip(gathered, glob):
        np.testing.assert_array_equal(a, b)


def test_dropping_corner_or_a_strip_breaks_the_refill():
    """Mutants: a 2x2 split without diagonal (corner) messages, and a halo one cell short, must not refill."""
    rng = np.random.default_rng(3)
    spec = dc.DecompositionSpec.build(30, 24, 2, 2, 5)
    glob = _fields(30, 24, rng)
    kinds = [dc.classify(x.shape, spec) for x in glob]
    exact = [[dc.slice_array(x, t, spec) for x in glob] for t in spec.tiles]
    poisoned = [[np.where(_halo_mask(spec, t, k[1], k[2], x.shape), -999, x).astype(x.dtype) for x, k in zip(lv, kinds)]
                for t, lv in zip(spec.tiles, exact)]
    plan = dc.exchange_plan(spec, kinds)
    diag = {(0, 3), (3, 0), (1, 2), (2, 1)}
    assert diag <= set(plan)  # corner neighbours exchange directly
    no_corner = {k: v for k, v in plan.items() if k not in diag}
    refilled = _simulate_exchange(spec, no_corner, poisoned)
    assert any((refilled[t.rank][0] == -999).any() for t in spec.tiles)
    short = {k: [dc.Transfer(m.leaf, m.src_y, (m.src_x[0], m.src_x[1] - 1), m.dst_y, (m.dst_x[0], m.dst_x[1] - 1))
                 if m.src_x[1] - m.src_x[0] > 1 else m for m in v] for k, v in plan.items()}
    refilled = _simulate_exchange(spec, short, poisoned)
    assert any((refilled[t.rank][0] == -999).any() for t in spec.tiles)


def test_spec_ownership_and_redundancy():
    spec = dc.DecompositionSpec.build(120, 70, 2, 1, 37)
    a, b = spec.tiles
    assert (a.own_x, b.own_x) == ((0, 60), (60, 120)) and (a.ext_x, b.ext_x) == ((0, 97), (23, 120))
    assert a.ext_y == b.ext_y == (0, 70)
    assert spec.owned_x(a, 1) == (0, 60) and spec.owned_x(b, 1) == (60, 121)  # the last tile owns face nx
    assert [d["redundancy"] for d in spec.describe()["tiles"]] == [round(97 / 60, 4)] * 2
    with pytest.raises(ValueError):
        dc.DecompositionSpec.build(3, 3, 4, 1, 1)


@pytest.mark.parametrize("bw", [5, 8])
def test_boundary_strips_are_resliced_by_the_tile_offset(bw):
    nx, ny, nz, nt = 30, 20, 3, 2
    spec = dc.DecompositionSpec.build(nx, ny, 2, 2, 4, bdy_width=bw)
    side = spec.side_len
    rng = np.random.default_rng(0)
    leaf = rng.standard_normal((nt, 4, bw, nz, side))
    assert dc.classify(leaf.shape, spec) == ("bdy",)
    for t in spec.tiles:
        strips = dc.slice_array(leaf, t, spec)
        lt = max(t.nx, t.ny) + 1
        assert strips.shape == (nt, 4, bw, nz, lt)
        for s, off, n in ((0, t.ext_y[0], t.ny + 1), (1, t.ext_y[0], t.ny + 1), (2, t.ext_x[0], t.nx + 1), (3, t.ext_x[0], t.nx + 1)):
            valid = min(n, side - off)
            np.testing.assert_array_equal(strips[:, s, ..., :valid], leaf[:, s, ..., off:off + valid])
            assert np.isfinite(strips).all()


def test_flat_and_replicated_leaves():
    spec = dc.DecompositionSpec.build(12, 7, 3, 1, 2)
    flat = np.arange(84.0)
    t = spec.tiles[1]
    np.testing.assert_array_equal(dc.slice_array(flat, t, spec), flat.reshape(7, 12)[:, t.ext_x[0]:t.ext_x[1]].reshape(-1))
    for shape in [(44,), (45,), (), (4, 5)]:
        assert dc.classify(shape, spec) == ("rep",)


def test_jitted_byte_messages_round_trip_every_dtype():
    jax = pytest.importorskip("jax")
    from gpuwrf.parallel import tile_driver as td

    rng = np.random.default_rng(5)
    spec = dc.DecompositionSpec.build(26, 18, 2, 2, 4)
    glob = _fields(26, 18, rng) + [rng.random((18, 26)) > 0.5]
    kinds = [dc.classify(x.shape, spec) for x in glob]
    plan = dc.exchange_plan(spec, kinds)
    exact = [[dc.slice_array(x, t, spec) for x in glob] for t in spec.tiles]
    for (src, dst), moves in plan.items():
        nbytes = sum(td._nbytes(m, glob[m.leaf]) for m in moves)
        msg = jax.jit(lambda lv: td._pack(lv, moves, nbytes + 7))([jax.numpy.asarray(x) for x in exact[src]])
        assert msg.dtype == np.uint8 and msg.shape == (nbytes + 7,)
        target = [jax.numpy.asarray(np.zeros_like(x)) for x in exact[dst]]
        got = jax.jit(lambda lv, b: td._unpack(lv, b, moves))(target, msg)
        for m in moves:
            np.testing.assert_array_equal(np.asarray(got[m.leaf])[..., m.dst_y[0]:m.dst_y[1], m.dst_x[0]:m.dst_x[1]],
                                          exact[src][m.leaf][..., m.src_y[0]:m.src_y[1], m.src_x[0]:m.src_x[1]])


def test_message_rounds_are_permutations():
    from gpuwrf.parallel.tile_driver import message_rounds

    spec = dc.DecompositionSpec.build(40, 30, 3, 3, 4)
    plan = dc.exchange_plan(spec, [("h", 0, 0)])
    rounds = message_rounds(spec, plan)
    assert sum(len(r) for r in rounds) == len(plan) and len(rounds) == 8  # 8 neighbour offsets
    for perm in rounds:
        assert len({s for s, _ in perm}) == len(perm) == len({d for _, d in perm})


TOY = r"""
import json, sys
import numpy as np
import jax, jax.numpy as jnp
from gpuwrf.parallel import decomposition as dc
from gpuwrf.parallel.tile_driver import TileRuntime, live_input_leaves

px, py, halos, k, transport, steps = int(sys.argv[1]), int(sys.argv[2]), [int(h) for h in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
use_live = len(sys.argv) > 7 and sys.argv[7] == "live"
NX, NY, NZ, ZONE = 37, 29, 3, 2   # toy step: radius 3 per step (2 Laplacians + mass->face->mass), ZONE-wide edge write

def lap(a):
    p = jnp.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    return p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:] - 4 * a

def one_step(c, nl, si):
        a, u, v = c["a"], c["u"], c["v"]
        ny, nx = a.shape[-2:]
        yy, xx = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
        edge = (yy < ZONE) | (yy >= ny - ZONE) | (xx < ZONE) | (xx >= nx - ZONE)
        a = jnp.where(edge, nl["f"], a)               # boundary zone at EVERY array edge (as the real step)
        ux = 0.5 * (u[..., :-1] + u[..., 1:])        # face -> mass
        vy = 0.5 * (v[:-1, :] + v[1:, :])
        a = a + 0.05 * lap(a) + 0.01 * ux * vy + nl["f"] * 1e-3 * si.astype(a.dtype)
        a = a + 0.05 * lap(a)                         # second Laplacian
        u = u.at[..., 1:-1].add(0.1 * (a[..., 1:] - a[..., :-1]))
        v = v.at[1:-1, :].add(0.1 * (a[0, 1:, :] - a[0, :-1, :]))
        w = lap(a[0]) * 2.0                            # scratch: written every step, never read
        return {"a": a, "u": u, "v": v, "s": c["s"], "w": w}

def advance(c, nl, start, clk, *, n_steps, cadence):
    return jax.lax.fori_loop(0, n_steps, lambda i, c: one_step(c, nl, start + i), c)

rng = np.random.default_rng(1)
carry = {"a": jnp.asarray(rng.standard_normal((NZ, NY, NX))), "u": jnp.asarray(rng.standard_normal((NZ, NY, NX + 1))),
         "v": jnp.asarray(rng.standard_normal((NY + 1, NX))), "s": jnp.asarray(3.0),
         "w": jnp.asarray(rng.standard_normal((NY, NX)))}
nl = {"f": jnp.asarray(rng.standard_normal((NY, NX)))}
ref = carry
for i in range(0, steps, k):
    ref = jax.jit(advance)(ref, nl, jnp.int32(1 + i), None, n_steps=jnp.int32(k), cadence=jnp.int32(1))  # traced, as the tiles
live = live_input_leaves(lambda c, si: one_step(c, nl, si), carry, jnp.int32(1))
out = {"live": live, "same": {}, "bytes": {}}
for halo in halos:
    spec = dc.DecompositionSpec.build(NX, NY, px, py, halo)
    rt = TileRuntime.create(carry, nl, spec, jax.devices()[:px * py], cadence=1, exchange_every=k, clock_base=jnp.zeros(()),
                            advance=advance, transport=transport, live=live if use_live else None)
    rt.advance(1, steps)
    got = rt.gather()
    out["same"][halo] = {name: bool(np.array_equal(np.asarray(got[name]), np.asarray(ref[name]))) for name in ref}
    out["bytes"][halo] = {f"{s}->{d}": b for (s, d), b in rt.message_bytes().items()}
print(json.dumps(out))
"""


def _toy(tmp_path, *argv):
    import json
    import os
    import subprocess
    import sys

    script = tmp_path / "toy.py"
    script.write_text(TOY)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=6", JAX_ENABLE_X64="1",
               PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    res = subprocess.run([sys.executable, str(script), *map(str, argv)], env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    out["exact"] = {int(h): all(v.values()) for h, v in out["same"].items()}
    return out


@pytest.mark.parametrize("px,py,k,transport", [(2, 1, 1, "device_put"), (3, 2, 1, "ppermute"), (2, 3, 2, "device_put"),
                                               (3, 2, 2, "ppermute")])
def test_tile_runtime_is_exact_from_the_derived_minimal_halo_and_not_one_cell_below(tmp_path, px, py, k, transport):
    """Toy step: edge zone 2 at every array edge, then a += lap(a) + u.v coupling, a += lap(a), u_face(i) from
    a(i-1), a(i). From an exact halo, step 1 corrupts mass 0..3 (zone 2 + two Laplacians) and faces 0..4 (face i
    reads mass i-1); every further step adds 2 (face->mass coupling, second Laplacian). A tile's lowest OWNED face
    reads the halo mass cell below it, so the minimal exact halo is the corrupted face count: 2k + 3. Verified
    tight: exact at 2k+3 (and above), wrong at 2k+2."""
    hmin = 2 * k + 3
    out = _toy(tmp_path, px, py, f"{hmin - 1},{hmin},{hmin + 3}", k, transport, 4)
    assert out["exact"] == {hmin - 1: False, hmin: True, hmin + 3: True}, out


def test_tile_runtime_zone_only_halo_breaks_the_interior(tmp_path):
    out = _toy(tmp_path, 3, 2, "2", 1, "device_put", 4)
    assert not out["same"]["2"]["a"], out


def test_live_leaf_exchange_skips_write_before_read_scratch_and_stays_exact(tmp_path):
    full = _toy(tmp_path, 3, 2, "5", 1, "device_put", 3)
    live = _toy(tmp_path, 3, 2, "5", 1, "device_put", 3, "live")
    assert live["live"] == [True, True, True, True, False]  # leaves a, s, u, v, w: the scratch w is never read
    assert live["exact"] == {5: True} and full["exact"] == {5: True}
    assert all(live["bytes"]["5"][p] < full["bytes"]["5"][p] for p in full["bytes"]["5"])


def test_required_halo_formula():
    from gpuwrf.parallel.tile_driver import required_halo

    assert required_halo(4) == 38 and required_halo(10) == 47 and required_halo(4, 2) == 70


CLI_TOY = r"""
import json, os, sys
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import jax, jax.numpy as jnp
from gpuwrf.parallel import cli_runtime

class G(NamedTuple):
    nx: int
    ny: int

@dataclass(frozen=True)
class NL:
    f: jax.Array
    grid: G
    radiation_cadence_steps: int = 1
    acoustic_substeps: int = 4

NL = jax.tree_util.register_dataclass(NL, data_fields=["f"], meta_fields=["grid", "radiation_cadence_steps", "acoustic_substeps"])
NX, NY = 31, 23

def advance(c, nl, start, clk, *, n_steps, cadence):
    def body(i, c):
        a = c["a"]
        p = jnp.pad(a, ((1, 1), (1, 1)), mode="edge")
        a = a + 0.1 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * a) + 1e-3 * nl.f * (start + i)
        ny, nx = a.shape
        yy, xx = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing="ij")
        a = jnp.where((yy < 1) | (yy >= ny - 1) | (xx < 1) | (xx >= nx - 1), nl.f, a)
        return {"a": a}
    return jax.lax.fori_loop(0, n_steps, body, c)

rng = np.random.default_rng(2)
nl = NL(f=jnp.asarray(rng.standard_normal((NY, NX))), grid=G(NX, NY))
carry = {"a": jnp.asarray(rng.standard_normal((NY, NX)))}
ref = jax.jit(advance)(carry, nl, jnp.int32(1), None, n_steps=jnp.int32(8), cadence=jnp.int32(1))
tree = type("T", (), {})()
tree.domains = {"d01": type("B", (), {"namelist": nl})()}
tree.edges = {}
os.environ["GPUWRF_DECOMPOSE"] = sys.argv[1]
c = cli_runtime.decomposed_advance(tree, "d01", carry, 1, 4, jnp.zeros(()), advance=advance)
c = {"a": c["a"] + 0.0}                       # host-side callback changes nothing numerically, new buffers
c = cli_runtime.decomposed_advance(tree, "d01", c, 5, 4, jnp.zeros(()), advance=advance)   # scatter path
print(json.dumps({"same": bool(np.array_equal(np.asarray(c["a"]), np.asarray(ref["a"]))),
                  "report": {k: v["config"] for k, v in cli_runtime.REPORT.items()}}))
"""


def test_cli_runtime_parse():
    from gpuwrf.parallel.cli_runtime import parse

    assert parse("2x1") == {"px": 2, "py": 1, "halo": None, "k": 1, "transport": "device_put", "exchange": "live"}
    assert parse("2x2,halo=40,k=2,transport=ppermute,exchange=all")["halo"] == 40
    for bad in ("", "2", "1x1", "2x1,foo=1", "2x1,transport=mpi", "0x2", "2x1,k=0"):
        with pytest.raises(ValueError):
            parse(bad)


@pytest.mark.parametrize("cfg", ["2x1,halo=3,exchange=all", "2x2,halo=3,transport=ppermute,exchange=all"])
def test_cli_runtime_scatter_advance_gather_matches_the_single_device_run(tmp_path, cfg):
    import json
    import os
    import subprocess
    import sys

    script = tmp_path / "cli_toy.py"
    script.write_text(CLI_TOY)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=4", JAX_ENABLE_X64="1",
               PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    res = subprocess.run([sys.executable, str(script), cfg], env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["same"], out


def test_tile_runtime_unaliases_shared_carry_buffers_before_donation():
    """The product carry holds some arrays at two positions; a 1x1 tile (full-extent slice) must not hand the
    donating tile program the same buffer twice (PjRt INVALID_ARGUMENT, found on the real PROD carry)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from gpuwrf.parallel.tile_driver import TileRuntime

    shared = jnp.ones((3, 9, 11))
    carry = {"a": shared, "b": shared, "s": jnp.asarray(1.0)}

    def advance(c, nl, start, clk, *, n_steps, cadence):
        return jax.lax.fori_loop(0, n_steps, lambda i, c: {"a": c["a"] * 2.0, "b": c["b"] + 1.0, "s": c["s"]}, c)

    spec = dc.DecompositionSpec.build(11, 9, 1, 1, 0)
    rt = TileRuntime.create(carry, {}, spec, jax.devices()[:1], cadence=1, clock_base=jnp.zeros(()), advance=advance)
    rt.advance(1, 2)
    got = rt.gather()
    np.testing.assert_array_equal(got["a"], np.full((3, 9, 11), 4.0))
    np.testing.assert_array_equal(got["b"], np.full((3, 9, 11), 3.0))
