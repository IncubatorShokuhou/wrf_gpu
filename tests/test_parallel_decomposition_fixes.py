"""Deletion-sensitive checks for the seven rv-multigpu findings; CPU only."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gpuwrf.parallel import decomposition as dc
from gpuwrf.parallel.harness import compare, finite_summary
from gpuwrf.parallel.tile_driver import configuration_halo, required_halo

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["candidate", "reference", "both"])
def test_nonfinite_is_never_bitwise(value, side):
    got, ref = [np.ones(3)], [np.ones(3)]
    if side in ("candidate", "both"):
        got[0][1] = value
    if side in ("reference", "both"):
        ref[0][1] = value
    report = compare(got, ref, [("rep",)], ref)
    assert report["verdict"] == "INVALID", report
    assert report["non_finite"] + report["reference_non_finite"] > 0


@pytest.mark.parametrize("got,ref", [([np.ones(1)], [np.ones(1), np.ones(1)]),
                                     ([np.ones(2)], [np.ones((1, 2))]),
                                     ([np.ones(1, np.float32)], [np.ones(1, np.float64)]),
                                     ({"a": np.ones(1)}, {"b": np.ones(1)}), ([], [])])
def test_schema_mismatch_is_invalid(got, ref):
    import jax

    leaves = jax.tree_util.tree_leaves(ref)
    report = compare(got, ref, [("rep",)] * len(leaves), leaves)
    assert report["verdict"] == "INVALID" and report["schema_errors"], report


def test_signed_zero_and_equal_bytes():
    ref = [np.array([0.0])]
    assert compare([np.array([-0.0])], ref, [("rep",)], ref)["verdict"] == "MISMATCH"
    assert compare(ref, ref, [("rep",)], ref)["verdict"] == "BITWISE"
    assert finite_summary([np.array([np.inf])])["verdict"] == "INVALID"


def _nl(**changes):
    values = dict(acoustic_substeps=4, topo_shading=1, slope_rad=1, topo_shadow_length_m=25000.0,
                  grid=NS(projection=NS(dx_m=100.0)), disable_guards=True)
    values.update(changes)
    return NS(**values)


def test_shadow_then_dynamics_cones_add_and_compose_across_exchange_interval():
    assert configuration_halo(_nl()) == required_halo(4, shadow_radius=252) == 290
    assert configuration_halo(_nl(), 2) == 574
    assert configuration_halo(_nl(slope_rad=0)) == 38
    for kwargs in ({"km_opt": 2}, {"km_opt": 5}, {"const_nu_m2_s": 1.0}, {"disable_guards": False},
                   {"ra_lw_physics": 3}, {"mp_physics": 18}, {"radiation_static": None},
                   {"specified_adv_degrade": False}):
        with pytest.raises(ValueError, match="decomposition"):
            configuration_halo(_nl(**kwargs))


def test_actual_shadow_operator_ridge_outside_old_halo_is_retained():
    import jax.numpy as jnp
    from gpuwrf.coupling.physics_couplers import _wrf_topographic_shadow_mask

    terrain = np.zeros((12, 200), np.float64)
    terrain[:, 160] = 10000.0

    def shadow(h):
        z = jnp.zeros(h.shape)
        return np.asarray(_wrf_topographic_shadow_mask(jnp.asarray(h), latitude_deg=z,
            coszen=jnp.full(h.shape, .1), declination_rad=0., hour_angle_rad=-np.pi / 2,
            sina=z, cosa=jnp.ones(h.shape), dx_m=100., dy_m=100., shadow_length_m=25000.))

    full = shadow(terrain)
    old = dc.DecompositionSpec.build(200, 12, 2, 1, 38)
    new = dc.DecompositionSpec.build(200, 12, 2, 1, configuration_halo(_nl()))
    assert full[5, 99] == 1
    assert shadow(dc.slice_array(terrain, old.tiles[0], old))[5, 99] == 0
    np.testing.assert_array_equal(shadow(dc.slice_array(terrain, new.tiles[0], new))[:, :100], full[:, :100])


def test_full_extent_tile_statics_are_valid_and_unsliced_partial_tile_is_not():
    spec = dc.DecompositionSpec.build(42, 42, 2, 1, 38)
    arrays = {"terrain": np.zeros((42, 42)), "flat": np.zeros(42 * 42)}
    assert not dc.audit_global_arrays(dc.slice_namelist(arrays, spec.tiles[0], spec), spec, tile=spec.tiles[0])
    partial = dc.DecompositionSpec.build(42, 42, 2, 1, 3)
    assert dc.audit_global_arrays(arrays, partial, tile=partial.tiles[0])


def test_census_is_explicitly_refused_even_when_environment_is_off(monkeypatch):
    import jax
    import jax.numpy as jnp
    from gpuwrf.diagnostics.census import initial_census
    from gpuwrf.parallel.tile_driver import TileRuntime

    monkeypatch.setenv("GPUWRF_CENSUS", "0")
    c = {"a": jnp.ones((12, 20)), "census": initial_census()}
    with pytest.raises(ValueError, match="owned-cell guard counts"):
        TileRuntime.create(c, {}, dc.DecompositionSpec.build(20, 12, 1, 1, 0), jax.devices(), cadence=1,
                           clock_base=jnp.zeros(()), advance=lambda *a, **kw: None)
    monkeypatch.setenv("GPUWRF_CENSUS", "1")
    with pytest.raises(ValueError, match="F2 qualification is unavailable"):
        TileRuntime.create({"a": c["a"]}, {}, dc.DecompositionSpec.build(20, 12, 1, 1, 0), jax.devices(),
                           cadence=1, clock_base=jnp.zeros(()), advance=lambda *a, **kw: None)


RUNTIME_CHECK = r'''
import dataclasses, gc, json, os, weakref
from typing import NamedTuple
import numpy as np
import jax, jax.numpy as jnp
from gpuwrf.parallel import cli_runtime as cli, decomposition as dc
from gpuwrf.parallel.tile_driver import TileRuntime
from gpuwrf.runtime.domain_tree import DomainTree
class Grid(NamedTuple):
    nx: int
    ny: int
class Boundary(NamedTuple):
    spec_bdy_width: int = 8
@dataclasses.dataclass(frozen=True)
class NL:
    f: jax.Array
    grid: Grid
    radiation_cadence_steps: int = 1
    acoustic_substeps: int = 4
    boundary_config: Boundary = Boundary()
NL = jax.tree_util.register_dataclass(NL, data_fields=['f'], meta_fields=['grid','radiation_cadence_steps','acoustic_substeps','boundary_config'])
def advance(c, nl, start, clock, *, n_steps, cadence):
    return jax.lax.fori_loop(0, n_steps, lambda i,c: {'a': c['a'] + nl.f + (start+i)*.001 + clock}, c)
def tree(f):
    return DomainTree(hierarchy=None,
        domains={'d01': type('Bundle', (), {'namelist': NL(jnp.full((12,80), f), Grid(80,12))})()}, edges={})
os.environ['GPUWRF_DECOMPOSE'] = '2x1,halo=4,exchange=all'
carry = {'a': jnp.zeros((12,80))}
clock = jnp.zeros(())
t1,t2 = tree(1.),tree(7.)
cli.decomposed_advance(t1,'d01',carry,1,1,clock,advance=advance)
assert cli._RUNS[id(t1)][1]['d01'][1].spec.bdy_width == 8, 'configured boundary width was not forwarded to slicing'
got = cli.decomposed_advance(t2,'d01',carry,1,1,clock,advance=advance)
np.testing.assert_array_equal(np.asarray(got['a']), np.full((12,80),7.001))
old = cli._RUNS[id(t2)][1]['d01'][1]
t2.domains['d01'].namelist = tree(3.).domains['d01'].namelist
got = cli.decomposed_advance(t2,'d01',carry,1,1,clock,advance=advance)
np.testing.assert_array_equal(np.asarray(got['a']), np.full((12,80),3.001))
assert not old.carries and not old.steps
os.environ['GPUWRF_DECOMPOSE'] = '2x1,halo=5,exchange=all'
got = cli.decomposed_advance(t2,'d01',carry,1,1,clock,advance=advance)
assert cli._RUNS[id(t2)][1]['d01'][1].spec.halo == 5
os.environ['GPUWRF_DECOMPOSE'] = '2x1,halo=4,exchange=all'
rt = TileRuntime.create(carry,t1.domains['d01'].namelist,dc.DecompositionSpec.build(80,12,2,1,4),
                       jax.devices(),cadence=1,clock_base=clock,advance=advance)
packers = tuple(rt.packers)
rt.scatter(carry, start_step=1)
assert tuple(rt.packers) == packers and all(p._cache_size() == 1 for p in rt.packers), 'output-boundary scatter retraced halo packers'
rt.advance(1,1); rt.block()   # compile all programs before the transfer check
with jax.transfer_guard('disallow'):
    rt.advance(2,4); rt.block()
assert rt.next_step == 6 and all(int(np.asarray(s)) == 6 for s in rt.starts)
assert all(step._cache_size() == 1 for step in rt.steps), 'clock must not cause per-step recompiles'
weak_rt = weakref.ref(cli._RUNS[id(t2)][1]['d01'][1])
del t2,got
gc.collect()
assert weak_rt() is None, 'global cache retained completed run buffers'
cli.reset(t1)
assert id(t1) not in cli._RUNS
print(json.dumps({'run_isolation':True,'configuration_invalidation':True,'transfer_guard':True,'clock':6,'cache_count_per_tile':1,'gc_release':True}))
'''


def test_run_scope_configuration_gc_and_device_control_scalars(tmp_path):
    script = tmp_path / "runtime.py"
    script.write_text(RUNTIME_CHECK)
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", GPUWRF_CENSUS="0", JAX_ENABLE_X64="1",
               XLA_FLAGS="--xla_force_host_platform_device_count=2", PYTHONPATH=str(REPO / "src"))
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1])["transfer_guard"]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "runpod" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ensemble_acceptance_requires_every_completed_member_and_nonempty_equal_frames(tmp_path):
    validate = _module("validate_ensemble").validate
    assert validate(tmp_path)["verdict"] == "FAIL"
    for i in range(1, 5):
        member = tmp_path / f"member{i:02d}"
        (member / "wrfout").mkdir(parents=True)
        (member / "receipt.json").write_text(json.dumps({"rc": 0}))
        (member / "wrfout" / "wrfout_d01_01").write_bytes(b"identical frames")
    assert validate(tmp_path, min_frames=1)["verdict"] == "PASS"
    frame = tmp_path / "member04" / "wrfout" / "wrfout_d01_01"
    frame.write_bytes(b"")
    assert validate(tmp_path, min_frames=1)["verdict"] == "FAIL"
    frame.write_bytes(b"different")
    assert validate(tmp_path, min_frames=1)["verdict"] == "FAIL"
    frame.write_bytes(b"identical frames")
    (tmp_path / "member04" / "receipt.json").write_text('{"rc":1}')
    assert validate(tmp_path, min_frames=1)["verdict"] == "FAIL"


@pytest.mark.parametrize("warmup,k", [(False, 1), (True, 1), (True, 2)])
def test_run_single_entry_aliases_carry_and_resident_step_clock(monkeypatch, warmup, k):
    import jax
    from gpuwrf.parallel import harness
    from gpuwrf.runtime import operational_mode as om

    jit = jax.jit

    def advance(c, nl, si, clock, *, n_steps, cadence):
        return jax.lax.fori_loop(0, n_steps, lambda i, c: {"a": c["a"] + si + i}, c)

    inner = jit(advance, donate_argnums=(0,))
    monkeypatch.setattr(om, "_advance_chunk_fori", inner)
    entries = []

    def capture(fun, *args, **kwargs):
        entry = jit(fun, *args, **kwargs)
        entries.append(entry)
        return entry

    # Observe the executable created by the public benchmark entry, rather than a lookalike wrapper.
    monkeypatch.setattr(jax, "jit", capture)
    device = jax.devices()[0]
    carry = {"a": jax.device_put(np.ones((1024, 1024), np.float32), device)}
    clock = jax.device_put(np.int32(0), device)
    result, timing = harness.run_single(carry, {}, start_step=7, steps=4, k=k, clock=clock,
                                       cadence=3, warmup=warmup)
    np.testing.assert_array_equal(np.asarray(result["a"]), np.full((1024, 1024), 35, np.float32))
    assert not carry["a"].is_deleted() and not clock.is_deleted()
    np.testing.assert_array_equal(np.asarray(carry["a"]), np.ones((1024, 1024), np.float32))
    assert timing["start_step"] == 7 and timing["stop_step_exclusive"] == 11
    assert len(entries) == 1 and entries[0]._cache_size() == 1
    start, trips, cadence = (jax.device_put(np.int32(x), device) for x in (7, k, 3))
    compiled = entries[0].lower(carry, {}, start, clock, trips, cadence).compile()
    # Inner donation alone yields zero outer aliases; donating only carry still loses the 4-byte clock.
    expected_alias_bytes = carry["a"].nbytes + start.nbytes
    stats = compiled.memory_analysis()
    assert stats.alias_size_in_bytes == expected_alias_bytes, stats


@pytest.mark.parametrize("start,steps,k,no_single,nonfinite", [(1, 3, 1, False, False),
    (7, 4, 2, False, False), (1, 1, 1, True, True)])
def test_benchmark_windows_and_post_timing_acceptance(tmp_path, monkeypatch, start, steps, k, no_single, nonfinite):
    import jax
    import jax.numpy as jnp
    from gpuwrf.parallel import harness
    from gpuwrf.parallel.tile_driver import TileRuntime
    from gpuwrf.runtime import operational_mode as om

    bench = _module("decomposed_bench")
    carry = {"a": jnp.full((12, 20), np.nan if nonfinite else 1.0)}

    def advance(c, nl, si, clock, *, n_steps, cadence):
        return jax.lax.fori_loop(0, n_steps, lambda i,c: {"a": c["a"] + si+i +
            jnp.where((si+i-1) % cadence == 0, 100., 0.)}, c)

    monkeypatch.setattr(om, "_advance_chunk_fori", advance)
    monkeypatch.setattr(om, "build_clock_base", lambda _: jnp.zeros(()))
    # Static configuration remains a registered pytree, as in the product.
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Config:
        grid: object
        acoustic_substeps: int = 4
        radiation_cadence_steps: int = 3
    Config = jax.tree_util.register_dataclass(Config, data_fields=[], meta_fields=["grid", "acoustic_substeps", "radiation_cadence_steps"])
    from typing import NamedTuple
    class Grid(NamedTuple):
        nx: int
        ny: int
    nl = Config(Grid(20, 12))
    monkeypatch.setattr(harness, "load_case", lambda *a, **kw: (carry, nl, {"nx":20,"ny":12}))
    calls = []
    original = TileRuntime.advance
    def record(rt, si, count):
        calls.append((si, count))
        return original(rt, si, count)
    monkeypatch.setattr(TileRuntime, "advance", record)
    argv = ["decomposed_bench.py", "--allow-cpu", "--out", str(tmp_path), "--px", "1", "--py", "1",
            "--halo", "0", "--steps", str(steps), "--start-step", str(start), "--exchange-every", str(k), "--exchange", "all"]
    if no_single:
        argv.append("--no-single")
    monkeypatch.setattr(sys, "argv", argv)
    rc = bench.main()
    report = json.loads(next(tmp_path.glob("bench_*.json")).read_text())
    assert calls == [(start, k), (start, steps)]
    assert (rc, report["acceptance"]) == ((1, "FAIL") if nonfinite else (0, "PASS"))
    if not no_single:
        for key in ("start_step", "steps", "stop_step_exclusive", "cadence"):
            assert report["single_gpu"][key] == report["decomposed"][key]
        assert report["compare"]["verdict"] == "BITWISE"


def test_eight_rank_ppermute_many_async_exchanges(tmp_path):
    # Exercise the original failure mode: no per-interval block, many per-rank producers and offset
    # rendezvous, 8 virtual ranks competing on the lane's two CPU cores. One fused transport call/interval.
    spec = importlib.util.spec_from_file_location("decomp_toy", REPO / "tests" / "test_parallel_decomposition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    script = tmp_path / "stress.py"
    script.write_text(module.TOY)
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", GPUWRF_CENSUS="0", JAX_ENABLE_X64="1",
               XLA_FLAGS="--xla_force_host_platform_device_count=8", PYTHONPATH=str(REPO / "src"))
    result = subprocess.run([sys.executable, str(script), "4", "2", "5,8", "1", "ppermute", "32"],
                            env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr[-4000:]
    report = json.loads(result.stdout.splitlines()[-1])
    assert all(all(leaves.values()) for leaves in report["same"].values()), report


def test_full_extent_tile_donation_preserves_the_callers_global_carry():
    import jax
    import jax.numpy as jnp
    from gpuwrf.parallel.tile_driver import TileRuntime

    original = jnp.ones((12, 20))
    carry = {"a": original}
    def advance(c, nl, si, clock, *, n_steps, cadence):
        return {"a": c["a"] + 1.0}
    rt = TileRuntime.create(carry, {}, dc.DecompositionSpec.build(20, 12, 1, 1, 0), jax.devices(),
                            cadence=1, clock_base=jnp.zeros(()), advance=advance)
    rt.advance(1, 1)
    rt.block()
    assert not original.is_deleted(), "tile donation deleted the caller's global carry"
    np.testing.assert_array_equal(np.asarray(original), np.ones((12, 20)))
    rt.close()


def test_matrix_records_failure_and_runs_later_independent_arms(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python3"
    python.write_text(f'''#!/usr/bin/env bash
if [ "$1" = -m ]; then
  case "$*" in *T1_cold*) exit 7;; esac
  exit 0
fi
exec '{sys.executable}' "$@"
''')
    python.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], WRFGPU_REPO=str(REPO))
    results = tmp_path / "results"
    result = subprocess.run(["bash", str(REPO / "scripts/runpod/test_matrix.sh"), str(results), "T1"],
                            env=env, capture_output=True, text=True, timeout=30)
    report = json.loads((results / "matrix_status.json").read_text())
    assert result.returncode == 1 and report["verdict"] == "FAIL"
    assert report["failed_arms"] == ["T1_cold"]
    assert {r["name"] for r in report["arms"]} >= {"T1_cold", "T1_warm", "cost_model"}


def test_runpod_table_preparation_materializes_noahmp_and_checks_all_tables(tmp_path):
    wrf, bin_dir = tmp_path / "WRF", tmp_path / "bin"
    (wrf / "run").mkdir(parents=True)
    bin_dir.mkdir()
    for name in ("SOILPARM.TBL", "GENPARM.TBL", "VEGPARM.TBL", "RRTMG_LW_DATA", "RRTMG_SW_DATA"):
        (wrf / "run" / name).write_text("tracked table\n")
    # Isolated clone/submodule substitute: source actions and linkage are under test, no network/build/GPU.
    git = bin_dir / "git"
    git.write_text('''#!/usr/bin/env bash
if [ "$3" = rev-parse ]; then echo f52c197ed39d12e087d02c50f412d90d418f6186; exit 0; fi
if [ "$3" = submodule ] && [ "$4" = update ] && [ "$5" = --init ] && [ "$6" = --depth ] && [ "$7" = 1 ] && [ "$8" = phys/noahmp ]; then
  mkdir -p "$2/phys/noahmp/parameters"
  echo 'pinned Noah-MP parameters' > "$2/phys/noahmp/parameters/MPTABLE.TBL"
  exit 0
fi
exit 1
''')
    git.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    cmd = ["bash", str(REPO / "scripts/runpod/prepare_wrf_tables.sh"), str(wrf)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert (wrf / "run/MPTABLE.TBL").is_symlink()
    assert (wrf / "run/MPTABLE.TBL").read_text() == "pinned Noah-MP parameters\n"
    (wrf / "run/SOILPARM.TBL").unlink()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "missing/empty WRF runtime table" in result.stderr
