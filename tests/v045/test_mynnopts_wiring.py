"""Opt-in MYNN moment lifecycle and the actual column/RK caller seams."""
from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from test_v022_camuw_pbl import _grid, _state, _namelist
from gpuwrf.contracts.precision import MYNN_CLOUDPDF_LEAVES, mynn_cloudpdf_leaves
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.runtime import operational_mode as O
from gpuwrf.io.namelist_binding import bound_options


@pytest.mark.parametrize("option", [0, 1, 2])
def test_real_binding_static_roundtrip_and_selected_allocation(option):
    grid = _grid()
    bound = bound_options({"physics": {"bl_mynn_cloudpdf": option}}, "d02")
    assert bound["bl_mynn_cloudpdf"] == option
    nl = _namelist(grid, **bound)
    children, aux = nl.tree_flatten()
    restored = nl.tree_unflatten(aux, children)
    assert restored.bl_mynn_cloudpdf == option
    default = replace(nl, bl_mynn_cloudpdf=2)
    assert (jax.tree.structure(nl) == jax.tree.structure(default)) == (option == 2)
    state = O._initial_carry_for_run(_state(grid), nl).state
    assert {name for name in MYNN_CLOUDPDF_LEAVES if getattr(state, name) is not None} == set(mynn_cloudpdf_leaves(nl))
    for name in mynn_cloudpdf_leaves(nl):
        assert getattr(state, name).dtype == jnp.float32
        assert getattr(state, name).shape == state.theta.shape


@pytest.mark.parametrize("option", [0, 1])
@pytest.mark.parametrize("adapter", [C.mynn_adapter, C.mynn_adapter_with_source_leaves])
def test_prepared_column_operands_and_return_reach_rk(monkeypatch, option, adapter):
    grid = _grid()
    names = mynn_cloudpdf_leaves({"bl_mynn_cloudpdf": option})
    state = _state(grid)
    state = state.replace(**{name: jnp.full_like(state.qsq, i + .25) for i, name in enumerate(names)})
    seen = []
    def step(column, dt, **kw):
        assert kw["cloudpdf"] == option
        for name in names:
            np.testing.assert_array_equal(getattr(column, name), C._to_columns(getattr(state, name)).reshape(-1, grid.nz))
        seen.append(1)
        return column.replace(**{name: getattr(column, name) + 3 for name in names}), jnp.zeros((grid.nx * grid.ny,))
    monkeypatch.setattr(C, "step_mynn_pbl_column_with_pblh", step)
    result = adapter(state, 18., grid, cloudpdf=option)
    physical = result.state if hasattr(result, "state") else result
    applied = O._apply_physics_non_dry_updates(state, state, physical)
    assert seen == [1]
    for name in names:
        np.testing.assert_array_equal(getattr(applied, name), getattr(state, name) + 3)


@pytest.mark.parametrize("option", [0, 1])
@pytest.mark.parametrize("traced", [False, True])
def test_fresh_selected_moments_seed_before_condensation(monkeypatch, option, traced):
    grid = _grid()
    names = mynn_cloudpdf_leaves({"bl_mynn_cloudpdf": option})
    state = _state(grid)
    state = state.replace(**{name: jnp.zeros_like(state.qsq) for name in names})
    def seed(s, g, *, return_moments, return_cloudpdf_moments=False):
        assert return_moments
        values = (7., 8., 9., 10., 11., 12.) if return_cloudpdf_moments else (7., 8., 9.)
        return tuple(jnp.full_like(s.qsq, value) for value in values)
    monkeypatch.setattr(C, "mynn_coldstart_qke_from_state", seed)
    selected = ("qke", "el_pbl", "qsq", *names)
    values = dict(zip(("qke", "el_pbl", "qsq", "tsq", "cov", "sh3d"), (7., 8., 9., 10., 11., 12.)))
    out = C._mynn_state_with_first_call_qke(state, grid, jnp.asarray(True) if traced else True)
    for name in selected:
        np.testing.assert_array_equal(getattr(out, name), jnp.full_like(getattr(state, name), values[name]))
    assert C._mynn_state_with_first_call_qke(state, grid, True, restart=True) is state
    later = C._mynn_state_with_first_call_qke(state, grid, jnp.asarray(False) if traced else False)
    for name in selected:
        np.testing.assert_array_equal(getattr(later, name), getattr(state, name))


@pytest.mark.parametrize("overrides", [{"bl_pbl_physics": 1}, {"cycling": True}, {"tke_budget": 1},
                                      {"bl_mynn_output": 1}, {"bl_mynn_cloudpdf": -1}, {"bl_mynn_cloudpdf": 3}])
def test_direct_namelist_refuses_unqualified_combinations(overrides):
    with pytest.raises(ValueError, match="bl_mynn_cloudpdf"):
        _namelist(_grid(), **({"bl_mynn_cloudpdf": 0} | overrides))


def test_binding_refuses_per_domain_pdf_list():
    with pytest.raises(ValueError, match="one scalar"):
        bound_options({"physics": {"bl_mynn_cloudpdf": [0, 1]}}, "d02")


def test_selected_initial_tsq_cov_against_original_wrf_mym_initialize(tmp_path):
    """Same release/no-FMA condition as the existing original-WRF init gate."""
    import hashlib
    import json
    import os
    import subprocess
    import gpuwrf
    root = Path(__file__).parents[2]
    directory = root / "tests/data/w44_qsqseed"
    path = directory / "original.npz"
    provenance = json.loads((directory / "provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["original_sha256"]
    env = {key: value for key, value in os.environ.items() if not key.startswith("GPUWRF_") and key != "XLA_FLAGS"}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_FAST_DEFAULTS="1", GPUWRF_JAX_CACHE="0",
               PYTHONPATH=str(Path(gpuwrf.__file__).resolve().parents[1]), XLA_FLAGS="--xla_cpu_max_isa=AVX")
    report = tmp_path / "cloudpdf_init.json"
    run = subprocess.run([sys.executable, str(root / "proofs/v45_mynnopts/check_init.py"), str(path), str(report)],
                         env=env, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stdout[-1000:] + run.stderr[-3000:]
    result = json.loads(report.read_text())
    assert result["passed"] and result["fast_defaults"] and result["backend"] == "cpu"
    assert {row["field"] for row in result["rows"]} == {"QKE", "EL", "QSQ", "TSQ", "COV"}
    assert all(row["truth_nonzero"] > 0 for row in result["rows"])


@pytest.mark.parametrize("site", ["allocation", "rk", "seed"])
def test_lifecycle_guards_kill_semantic_deletions(monkeypatch, site):
    """Same healthy assertion gates, with numeric/source decisions removed."""
    if site == "allocation":
        monkeypatch.setattr(O, "mynn_cloudpdf_leaves", lambda nl: ())
        gate = lambda: test_real_binding_static_roundtrip_and_selected_allocation(0)
    elif site == "rk":
        monkeypatch.setattr(O, "_PHYSICS_NON_DRY_REPLACE_FIELDS", tuple(
            name for name in O._PHYSICS_NON_DRY_REPLACE_FIELDS if name not in MYNN_CLOUDPDF_LEAVES))
        gate = lambda: test_prepared_column_operands_and_return_reach_rk(monkeypatch, 0, C.mynn_adapter)
    else:
        import inspect
        source = inspect.getsource(C._mynn_state_with_first_call_qke)
        needle = 'cloudpdf_names = tuple(name for name in ("tsq", "cov", "sh3d") if getattr(state, name, None) is not None)'
        assert source.count(needle) == 1
        scope = {}
        exec(compile(source.replace(needle, "cloudpdf_names = ()"), "<no-cloudpdf-seed>", "exec"), C.__dict__, scope)
        monkeypatch.setattr(C, "_mynn_state_with_first_call_qke", scope["_mynn_state_with_first_call_qke"])
        gate = lambda: test_fresh_selected_moments_seed_before_condensation(monkeypatch, 0, False)
    with pytest.raises(AssertionError):
        gate()


@pytest.mark.parametrize("option", [0, 1, 2])
@pytest.mark.parametrize("format", ["checkpoint", "pickle", "netcdf"])
def test_restart_preserves_selected_moments_and_refuses_selection_change(tmp_path, option, format):
    from gpuwrf.io import restart as R, wrfrst_netcdf as N
    from gpuwrf.runtime import checkpoint as K
    grid = _grid()
    nl = _namelist(grid, bl_mynn_cloudpdf=option)
    carry = O._initial_carry_for_run(_state(grid), nl)
    names = mynn_cloudpdf_leaves(nl)
    state = carry.state.replace(**{name: jnp.full_like(carry.state.qsq, 2. + i) for i, name in enumerate(names)})
    carry = carry.replace(state=state)
    path = tmp_path / "restart"
    if format == "checkpoint":
        K.write_checkpoint(state, nl, grid, 2, path, runtime_state=carry)
        read = lambda requested: K.read_checkpoint(path, namelist=requested)[0]
    elif format == "pickle":
        R.write_restart(carry, nl, grid, 2, path)
        read = lambda requested: R.read_restart(path, namelist=requested)[0].state
    else:
        N.write_wrfrst_carry(carry, grid, nl, path, valid_time=nl.time_utc, run_start=nl.time_utc, step_index=2)
        read = lambda requested: N.read_wrfrst_carry(path, namelist=requested)[0].state
    restored = read(nl)
    for name in names:
        np.testing.assert_array_equal(getattr(restored, name), getattr(state, name))
    with pytest.raises(ValueError, match="bl_mynn_cloudpdf.*differs"):
        read(replace(nl, bl_mynn_cloudpdf=(1 if option == 0 else 0)))


def test_restart_refuses_present_but_empty_selected_moment():
    from gpuwrf.io.mynn_cloudpdf_schema import validate
    with pytest.raises(ValueError, match="missing held"):
        validate({"tsq": None, "cov": jnp.zeros((2,))}, {"bl_mynn_cloudpdf": 0})


@pytest.mark.parametrize("option", [0, 1])
@pytest.mark.parametrize("route", ["source", "legacy", "edmf_off"])
def test_actual_runtime_forwards_pdf_and_prepared_state(monkeypatch, option, route):
    """E11: reach the actual runtime branch before stopping at its MYNN call."""
    grid = _grid()
    bound = bound_options({"physics": {"bl_mynn_cloudpdf": option}}, "d01")
    settings = dict(mp_physics=0, cu_physics=0, use_noahmp=False, sf_surface_physics=0,
                    rad_rk_tendf=0 if route == "legacy" else 1)
    if route == "edmf_off":
        settings["bl_mynn_edmf"] = 0
        monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", "1")
    nl = _namelist(grid, **(bound | settings))
    carry = O._initial_carry_for_run(_state(grid), nl)
    class Reached(RuntimeError):
        pass
    def entry(state, *args, **kwargs):
        assert kwargs.get("cloudpdf") == option
        for name in mynn_cloudpdf_leaves(nl):
            assert getattr(state, name) is not None
        if route == "edmf_off":
            assert kwargs["budget_config"]["bl_mynn_edmf"] == 0
        raise Reached()
    if route == "edmf_off":
        from gpuwrf.coupling import mynn_budget_adapter
        monkeypatch.setattr(mynn_budget_adapter, "mynn_sources_with_history", entry)
    else:
        monkeypatch.setattr(O, "mynn_adapter" if route == "legacy" else "mynn_adapter_with_source_leaves", entry)
    with pytest.raises(Reached):
        O._physics_step_forcing(carry, nl, 0., run_radiation=False)
