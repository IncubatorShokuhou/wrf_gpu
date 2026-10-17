"""Extra-species nest plumbing: pristine REAL routines, not coupled qualification.

All ten fields are active with signed spatial structure. This isolates forcing,
RK1 tendency ownership and final pinning; full 2-domain forecasts and nest-start
interpolation are separate gates and the CLI refusal remains in force.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.precision import EXTRA_SCALAR_BOUNDARY_LEAVES, STATE_FIELD_ORDER
from gpuwrf.contracts.state import EXTRA_SCALAR_BOUNDARY_SPECIES, State
from gpuwrf.coupling import boundary_apply as bd
from gpuwrf.nesting.boundary_construction import (
    build_child_boundary_package, build_nest_force_weights, initialize_child_scalar_boundaries,
)
from gpuwrf.nesting.scalar_species import extra_boundary_species
from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/host_sync/b36"), str(ROOT / "tests/v025/b_diff")]
F32 = np.float32
SCALES = dict(qh=1.e-3, Nc=1.e6, Nn=1.e8, Ns=1.e3, Ng=1.e3, Nh=1.e3,
              qvolg=1.e-6, qvolh=1.e-6, nwfa=1.e8, nifa=1.e5)
# This oracle cohort covers the ten microphysics scalar species. Epsilon-PBL
# scalars have their separate selection/admission cohort in test_v040_epsilon_wiring.
EXTRA = tuple(SCALES)


def assert_real(got, want, *, ulps=32):
    got, want = np.asarray(got, F32), np.asarray(want, F32)
    assert np.isfinite(got).all() and np.isfinite(want).all()
    scale = F32(np.abs(want).max())
    tolerance = float(np.spacing(scale)) * ulps
    error = float(np.abs(got.astype(np.float64) - want).max())
    assert error <= tolerance, (error, tolerance, scale)
    return error / float(np.spacing(scale)) if scale else 0.


def mass(state, metrics):
    return (np.asarray(metrics.c1h, F32)[:, None, None] * np.asarray(state.mu_total, F32)[None]
            + np.asarray(metrics.c2h, F32)[:, None, None])


def weather(grid, phase):
    """Active signed operator operands; no claim these are a forecast fixture."""
    state = build_neutral_state(grid)
    z, y, x = np.indices(state.qv.shape, dtype=F32)
    pattern = (np.sin((x + F32(phase)) / F32(2.7)) + np.cos(y / F32(3.1))
               + F32(.08) * z).astype(F32)
    updates = {n: jnp.asarray(F32(SCALES[n]) * pattern) for n in EXTRA}
    mu = (F32(80000.) + F32(100.) * y[0] + F32(51.) * x[0]).astype(F32)
    updates.update(mu_total=jnp.asarray(mu), mu_perturbation=jnp.asarray(mu - F32(80000.)))
    return state.replace(**updates)


@pytest.fixture(scope="module")
def operands():
    parent_grid = build_flat_grid(nx=32, ny=28, nz=3, dx_m=9000.)
    child_grid = build_flat_grid(nx=22, ny=19, nz=3, dx_m=3000.)
    pm = replace(parent_grid.metrics, c1h=jnp.asarray([.8, .5, .2], jnp.float32),
                 c2h=jnp.asarray([101., 202., 303.], jnp.float32))
    cm = replace(child_grid.metrics, c1h=jnp.asarray([.7, .4, .1], jnp.float32),
                 c2h=jnp.asarray([113., 227., 331.], jnp.float32))
    parent, child = weather(parent_grid, 3), weather(child_grid, 1)
    # Materialize a union solely for the operator gate. Operational selection is tested below.
    child = initialize_child_scalar_boundaries(child)
    child = child.replace(**{f"{n}_bdy": jnp.zeros_like(child.qc_bdy) for n in EXTRA})
    weights = build_nest_force_weights(parent_grid_ratio=3, i_parent_start=6, j_parent_start=7,
                                      parent_grid=parent_grid, child_grid=child_grid)
    return parent, child, pm, cm, weights


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    # Each builder stages unchanged source routines in the lane's basetemp, never the shared WRF build.
    from pristine_forcedown import build as build_force
    from pristine_boundary import build as build_boundary
    return (build_force(tmp_path_factory.mktemp("force")),
            build_boundary(tmp_path_factory.mktemp("boundary")))


@pytest.fixture(scope="module")
def forced(operands):
    parent, child, pm, cm, weights = operands
    # CPU path uses the staged JAX source SINT, not the GPU-only kernel.
    return build_child_boundary_package(child, parent, weights, bdy_width=5,
                                       parent_metrics=pm, child_metrics=cm, coupled_forcedown=True,
                                       parent_grid_ratio=3)


@pytest.mark.parametrize("mp,expected", [(8, ()), (18, ("qh", "Nc", "Ns", "Ng", "Nh", "Nn", "qvolg", "qvolh")),
    (24, ("qh",)), (26, ("qh", "Nc", "Nn")), (28, ("Nc", "nwfa", "nifa")), (40, ("Nc", "Ns", "Ng"))])
def test_only_selected_records_materialized(operands, mp, expected):
    child = operands[1].replace(**{f"{n}_bdy": None for n in EXTRA})
    state = initialize_child_scalar_boundaries(child, mp_physics=mp)
    assert extra_boundary_species(mp) == expected
    for name in EXTRA:
        rec = getattr(state, f"{name}_bdy")
        assert (rec is not None) == (name in expected)
        if rec is not None:
            assert rec.shape == state.qc_bdy.shape and rec.dtype == jnp.float32
    assert initialize_child_scalar_boundaries(state, mp_physics=mp) is state
    assert state.bytes() - child.bytes() == len(expected) * child.qc_bdy.size * 4


def test_missing_selected_prognostic_refused(operands):
    with pytest.raises(ValueError, match="'qh'.*absent"):
        initialize_child_scalar_boundaries(operands[1].replace(qh=None), mp_physics=24)


@pytest.mark.parametrize("name", EXTRA)
def test_forcedown_pristine_real4(operands, pristine, forced, name):
    from pristine_forcedown import forcedown
    parent, child, pm, cm, _ = operands
    coarse = np.asarray(getattr(parent, name), F32) * mass(parent, pm)
    fine = np.asarray(getattr(child, name), F32) * mass(child, cm)
    assert np.count_nonzero(coarse) == coarse.size
    assert np.any(coarse < 0) and np.any(coarse > 0)
    old, rate = forcedown(pristine[0], coarse, fine, start=(6, 7), ratio=3,
                          cadence=18., side_len=child.qv_bdy.shape[-1])
    records = np.asarray(getattr(forced, f"{name}_bdy"), F32)
    assert records.dtype == F32
    assert_real(records[0], old, ulps=1)
    got_rate = ((records[1] - records[0]).astype(np.float64) / 18.).astype(F32)
    ulps = assert_real(got_rate, rate)
    # Force-down may change ONLY boundary leaves.
    assert getattr(forced, name) is getattr(child, name)
    print(f"forcedown {name}: max_error={ulps:.3f} ulp(max_reference)")


@pytest.mark.parametrize("lead", [0., 6., 9., 18.])
def test_consumer_pristine_all_extra_species(operands, pristine, forced, lead):
    from pristine_boundary import Oracle
    cm = operands[3]
    shape = forced.qv.shape
    config = bd.BoundaryConfig(update_cadence_s=18., nested_frozen_wrf_boundary_bundle=True)
    names = bd.nested_boundary_scalar_species(forced)
    assert names == (*bd.NESTED_BOUNDARY_SCALAR_SPECIES, *EXTRA)
    tendencies = bd.nested_scalar_boundary_tendencies(forced, lead, cm, 6., config)
    oracle = Oracle(pristine[1], shape, nested=True, dt=6., dtbc=lead, dts=6.)
    oracle.set("mu", forced.mu_total)
    oracle.set("c1h", cm.c1h); oracle.set("c2h", cm.c2h)
    for name in EXTRA:
        rec = np.asarray(getattr(forced, f"{name}_bdy"), F32)
        rate = ((rec[1] - rec[0]).astype(np.float64) / 18.).astype(F32)
        oracle.set("scalar", getattr(forced, name))
        oracle.set("scalar_tend", np.zeros(shape, F32))
        oracle.boundary("scalar", rec[0], rate)
        oracle.run("relax_bdy_scalar")
        oracle.set("field_tend", oracle.field("scalar_tend", shape))
        oracle.run("spec_bdytend", tag=6)
        ulps = assert_real(tendencies[names.index(name)], oracle.field("field_tend", shape))
        print(f"tendency {name} lead={lead}: max_error={ulps:.3f} ulp(max_reference)")


def test_stage_excludes_spec_and_adds_extra_boundary_tendency(operands, forced):
    from gpuwrf.runtime.operational_mode import _nested_scalar_stage_tendencies
    cm = operands[3]
    config = bd.BoundaryConfig(update_cadence_s=18., nested_frozen_wrf_boundary_bundle=True)
    names = bd.nested_boundary_scalar_species(forced)
    frozen = bd.nested_scalar_boundary_tendencies(forced, 6., cm, 6., config)
    adv = tuple(jnp.full_like(forced.qv, i + 1) for i in range(len(names)))
    got_names, tendencies = _nested_scalar_stage_tendencies(adv, names, frozen, config,
                                  jnp.full_like(forced.mu_total, 1.25), boundary_species=names)
    assert got_names == names
    for i, n in enumerate(names):
        expected = np.array(frozen[i], dtype=np.result_type(adv[i].dtype, frozen[i].dtype))
        expected[:, 1:-1, 1:-1] += F32((i + 1) * 1.25)
        np.testing.assert_array_equal(tendencies[i], expected, err_msg=n)


@pytest.mark.parametrize("lead", [0., 6., 18.])
def test_spec_final_pristine_all_extra_species(operands, pristine, forced, lead):
    from pristine_boundary import Oracle
    cm, shape = operands[3], forced.qv.shape
    config = bd.BoundaryConfig(update_cadence_s=18., nested_frozen_wrf_boundary_bundle=True)
    # End-of-step dry mass differs from both force-down records.
    state = forced.replace(mu_total=forced.mu_total * F32(1.001))
    result = bd.nested_scalar_spec_final(state, lead, cm, config)
    oracle = Oracle(pristine[1], shape, nested=True, dt=6., dtbc=lead, dts=6.)
    oracle.set("mu", state.mu_total)
    oracle.set("c1h", cm.c1h); oracle.set("c2h", cm.c2h)
    for name in EXTRA:
        rec = np.asarray(getattr(state, f"{name}_bdy"), F32)
        rate = ((rec[1] - rec[0]).astype(np.float64) / 18.).astype(F32)
        oracle.set("field", getattr(state, name))
        oracle.boundary("scalar", rec[0], rate)
        oracle.run("spec_bdy_final", tag=4)
        assert_real(getattr(result, name), oracle.field("field", shape))
        np.testing.assert_array_equal(np.asarray(getattr(result, name))[:, 1:-1, 1:-1],
                                      np.asarray(getattr(state, name))[:, 1:-1, 1:-1])


def test_compiled_forcedown_keeps_all_extra_records(operands, forced):
    parent, child, pm, cm, weights = operands
    compiled = build_child_boundary_package(child, parent, weights, bdy_width=5,
                         parent_metrics=pm, child_metrics=cm, coupled_forcedown=True,
                         parent_grid_ratio=3, _compiled_producers=True)
    for name in EXTRA_SCALAR_BOUNDARY_LEAVES:
        # Regression of the compiled producer, not a second WRF fidelity claim.
        np.testing.assert_array_equal(getattr(compiled, name), getattr(forced, name))


def test_extra_records_restart_roundtrip_and_missing_refused(operands, forced, tmp_path):
    from gpuwrf.io.restart_schema import validate_held_state
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state, read_wrfrst_state
    from gpuwrf.validation.moving_nest_testbed import build_domain_namelist
    from datetime import datetime
    grid = build_flat_grid(nx=22, ny=19, nz=3, dx_m=3000.)
    namelist = build_domain_namelist(grid, dt_s=6., is_child=False)
    assert set(State.__slots__) == set(STATE_FIELD_ORDER)
    assert tuple(State.__slots__[89:99]) == tuple(f"{name}_bdy" for name in EXTRA)
    assert tuple(name for name in State.__slots__ if name in EXTRA_SCALAR_BOUNDARY_LEAVES) == EXTRA_SCALAR_BOUNDARY_LEAVES
    path = tmp_path / "restart.pkl"
    write_checkpoint(forced, namelist, grid, 3, path)
    restored, *_ = read_checkpoint(path)
    nc = tmp_path / "wrfrst"
    write_wrfrst_state(state=forced, grid=grid, namelist=namelist, path=nc,
                      valid_time=datetime(2026, 1, 1), run_start=datetime(2026, 1, 1), step_index=3)
    netcdf, _ = read_wrfrst_state(nc)
    for name in EXTRA_SCALAR_BOUNDARY_LEAVES:
        np.testing.assert_array_equal(getattr(restored, name), getattr(forced, name))
        np.testing.assert_array_equal(getattr(netcdf, name), getattr(forced, name))
    fields = {name: getattr(forced, name) for name in forced.active_field_names()}
    active = replace(namelist, mp_physics=24, boundary_config=bd.BoundaryConfig(nested_frozen_wrf_boundary_bundle=True))
    fields.pop("qh_bdy")
    with pytest.raises(ValueError, match="extra scalar boundary.*qh_bdy"):
        validate_held_state(fields, active)
    # Tampered field manifest must not silently disable a record without an explicit namelist.
    from netCDF4 import Dataset
    with Dataset(nc, "a") as ds:
        order = json.loads(ds.GPUWRF_STATE_FIELD_ORDER)
        order.remove("qh_bdy")
        ds.GPUWRF_STATE_FIELD_ORDER = json.dumps(order)
        ds.GPUWRF_STATE_FIELD_COUNT = len(order)
    with pytest.raises(ValueError, match="extra scalar boundary"):
        read_wrfrst_state(nc)


@pytest.mark.parametrize("mp", [18, 24, 26, 28, 40])
def test_unqualified_nested_run_still_refused(mp):
    from gpuwrf.io.namelist_check import refuse_single_domain_mp_nested, UnsupportedSchemeError
    with pytest.raises(UnsupportedSchemeError, match="parent->child"):
        refuse_single_domain_mp_nested({"physics": {"mp_physics": [mp, mp]}}, 2)


@pytest.mark.parametrize("key,code", [("sf_surface_physics", 3), ("sf_urban_physics", 1), ("km_opt", 2)])
def test_nested_options_refused_with_precise_port_reason(key, code):
    from gpuwrf.io.namelist_check import refuse_unqualified_nested_options, UnsupportedSchemeError
    refuse_unqualified_nested_options({key: [code]}, 1)
    with pytest.raises(UnsupportedSchemeError, match="d02:.*two-domain CPU-WRF comparison"):
        refuse_unqualified_nested_options({key: [0, code]}, 2)
    # Inactive d03 is not part of the requested two-domain run.
    refuse_unqualified_nested_options({key: [0, 0, code]}, 2)


@pytest.mark.parametrize("option", ["ruc", "urban", "km2"])
def test_cli_refuses_unqualified_nest_before_model_import(tmp_path, option):
    nml = tmp_path / "namelist.input"
    lines = dict(ruc="sf_surface_physics=3,3,\n num_soil_layers=6,",
                 urban="sf_surface_physics=4,4,\n sf_urban_physics=1,1,",
                 km2="sf_surface_physics=0,0,")
    nml.write_text("&domains\n max_dom=2,\n/\n&physics\n mp_physics=8,8,\n "
                   "sf_sfclay_physics=5,5,\n bl_pbl_physics=5,5,\n " + lines[option] + "\n/\n" +
                   ("&dynamics\n diff_opt=2,2,\n km_opt=2,2,\n moist_adv_opt=1,1,\n/\n" if option == "km2" else ""))
    code = '''
import importlib.abc
import sys
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("gpuwrf.integration", "gpuwrf.runtime.operational_mode", "gpuwrf.physics")):
            raise AssertionError("model imported before nested refusal: " + fullname)
sys.meta_path.insert(0, NoModel())
from gpuwrf import cli
# Allocator re-exec is outside this preflight test and must not query a GPU.
cli._maybe_reexec_for_nested_allocator = lambda args: None
assert cli.main(["run", "--namelist", sys.argv[1], "--input-dir", sys.argv[2],
                 "--output-dir", sys.argv[3], "--hours", "1", "--max-dom", "2"]) == 2
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, "-c", code, str(nml), str(tmp_path), str(tmp_path / "out")],
                            capture_output=True, text=True, env=env, cwd=ROOT, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "two-domain CPU-WRF comparison" in result.stderr
    assert not (tmp_path / "out").exists()


def test_nested_api_refuses_before_case_build(tmp_path, monkeypatch):
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.io.namelist_check import UnsupportedSchemeError
    (tmp_path / "namelist.input").write_text("&physics\n sf_surface_physics=0,3,\n/\n")
    def no_case(*args, **kwargs):
        raise AssertionError("case builder ran before nested qualification refusal")
    monkeypatch.setattr(pipeline, "build_replay_case", no_case)
    cfg = pipeline.NestedPipelineConfig(tmp_path, tmp_path / "out", tmp_path / "proof", hours=1, max_dom=2)
    with pytest.raises(UnsupportedSchemeError, match="d02:.*RUC"):
        pipeline._load_domains(cfg, ("d01", "d02"), _init_kernel_call=lambda *a: None)


def test_grid_fdda_domain_files_coefficients_clocks_and_defaults(tmp_path):
    from datetime import datetime, timedelta
    from netCDF4 import Dataset
    from gpuwrf.io.wrffdda import load_grid_fdda
    from gpuwrf.io.namelist_check import refuse_unqualified_nested_options
    sys.path.insert(0, str(ROOT / "tests/v040/fdda"))
    from test_o2_fdda_namelist_io import _write_wrffdda
    start = datetime(2023, 1, 15)
    _write_wrffdda(tmp_path / "wrffdda_d01", [start + timedelta(hours=k) for k in range(3)],
                   nz=3, ny=4, nx=5)
    _write_wrffdda(tmp_path / "wrffdda_d02", [start + timedelta(minutes=30*k) for k in range(5)],
                   nz=3, ny=5, nx=7)
    with Dataset(tmp_path / "wrffdda_d02", "a") as ds:
        ds["U_NDG_NEW"][:] += F32(100.)
    nml = {"domains": {"max_dom": 3}, "fdda": {"grid_fdda": [1, 1],
        "gfdda_interval_m": [60, 30], "gfdda_end_h": [2, 2], "guv": [3.e-4, 6.e-4]}}
    refuse_unqualified_nested_options(nml, 3)
    parent = load_grid_fdda(tmp_path, nml, "d01", run_start=start, dt_s=54.,
                            nz=3, ny=4, nx=5, run_seconds=7200.)
    child = load_grid_fdda(tmp_path, nml, "d02", run_start=start, dt_s=18.,
                           nz=3, ny=5, nx=7, run_seconds=7200.)
    assert parent.options.guv == 3.e-4 and child.options.guv == 6.e-4
    assert parent.record_steps == (0, 67) and child.record_steps == (0, 100, 200, 300)
    assert float(parent.u_new[0, 0, 0, 0]) == 1. and float(child.u_new[0, 0, 0, 0]) == 101.
    assert parent.u_new.shape == (2, 3, 4, 6) and child.u_new.shape == (4, 3, 5, 8)
    # WRF Registry max_domains default: omitted grid_fdda(3) = 0, not the child's 1.
    assert load_grid_fdda(tmp_path, nml, "d03", run_start=start, dt_s=6., nz=3, ny=6, nx=9) is None


def test_dfi_nested_reason_is_port_limitation():
    from gpuwrf.dfi.control import dfi_refusals
    text = (ROOT / "tests/fixtures/o2_dfi_swiss_tdfi.namelist").read_text()
    from gpuwrf.io.namelist_check import _coerce_config
    reasons = dfi_refusals(_coerce_config(text), max_dom=2)
    assert any("WRF supports simultaneous-start nests" in reason and "recursive filter phases" in reason
               for reason in reasons)
