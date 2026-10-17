"""MP adapter seams against frozen pristine-WRF precipitation and caller math.

The column solver outputs here come from WRF fixtures, independently of JAX.
Thermodynamic boundary tests isolate the prep/finish operator from the solver;
the existing per-scheme savepoint suites continue to gate the column solvers.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling import scan_adapters as adapters
from gpuwrf.io.wrfout_writer import _build_subset_output_fields, _dimension_sizes

ROOT = Path(__file__).resolve().parents[1]
ORACLES = (
    ("proofs/v060_wdm6/savepoints", "wdm6_case_*.json", "RAIN"),
    ("proofs/v013/savepoints_wsm7", "wsm7_case_*.json", "RAIN"),
    ("proofs/v034/f2_oracles/morrison_aero_opt0", "morr_aero_opt0_case_*.json", "RAINNC"),
)
CASES = [(p, total) for folder, pattern, total in ORACLES for p in sorted((ROOT / folder).glob(pattern))]
assert len(CASES) == 19, "Required original-WRF fixtures must be present, not silently skipped"


def state_for(option=26, nz=40, dtype=jnp.float32):
    grid = SimpleNamespace(nz=nz, ny=1, nx=2)
    state = State(**{n: jnp.zeros(shape, DEFAULT_DTYPES.dtype_for(n))
                     for n, shape in _state_field_shapes(grid, mp_physics=option).items()})
    return state.replace(_cast=False, **{n: jnp.asarray(getattr(state, n), dtype) for n in
                            ("theta", "qv", "rain_acc", "snow_acc", "graupel_acc", "ice_acc", "hail_acc")
                            if getattr(state, n) is not None})


@pytest.mark.parametrize("fixture,total_key", CASES, ids=[p.stem for p, _ in CASES])
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_original_wrf_total_is_accumulated_once(fixture, total_key, dtype):
    jax.config.update("jax_enable_x64", True)
    scalars = json.loads(fixture.read_text())["scalars"]
    mapping = {"rain_acc": "RAINNCV", "snow_acc": "SNOWNCV", "graupel_acc": "GRAUPELNCV"}
    if "HAILNCV" in scalars:
        mapping["hail_acc"] = "HAILNCV"
    inc = {leaf: jnp.full((2,), scalars[key], dtype) for leaf, key in mapping.items()}
    tendency = PhysicsTendency(accumulator_increments=inc)
    state = state_for(dtype=dtype)
    # Nonzero previous precipitation and two calls exercise accumulation,
    # rather than just re-labelling a single scheme's output.
    state = state.replace(rain_acc=jnp.full((1, 2), .5, dtype),
                          snow_acc=jnp.full((1, 2), .25, dtype))
    run = jax.jit(lambda s: adapters._apply_mp_replacements(s, tendency, ny=1, nx=2, nz=40))
    state = run(run(state))
    assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree.leaves(state))
    dimensions = _dimension_sizes(nx=2, ny=1, nz=40, namelist={})
    fields = _build_subset_output_fields(state, None, {}, dimensions, requested_names=frozenset({"RAINNC"}))
    expected = np.float32(.75 + 2 * scalars[total_key])
    np.testing.assert_allclose(fields["RAINNC"], expected, rtol=4*np.finfo(np.float32).eps, atol=0)
    for leaf, key in mapping.items():
        if leaf != "rain_acc":
            initial = .25 if leaf == "snow_acc" else 0.
            np.testing.assert_allclose(np.asarray(getattr(state, leaf)), initial + 2*scalars[key],
                                       rtol=4*np.finfo(np.dtype(dtype)).eps, atol=1.e-18)
    assert state.rain_acc.dtype == dtype


@pytest.mark.parametrize("option", (16, 26))
def test_wdm_caller_uses_pristine_dry_theta_and_moist_finish(monkeypatch, option):
    """Exact WRF prep/finish input/output, including both land and water."""
    data = json.loads((ROOT / "proofs/t_mprain/thermo_oracle.json").read_text())
    state = state_for(option=option, nz=len(data["theta_m_in"]))
    col = lambda key: jnp.asarray(data[key], jnp.float32)[:, None, None].repeat(2, axis=2)
    state = state.replace(theta=col("theta_m_in"), qv=col("qv_in"),
                          p_total=jnp.full_like(state.theta, 90000.),
                          ph_total=jnp.arange(state.theta.shape[0]+1, dtype=jnp.float32)[:,None,None].repeat(2,axis=2)*1000.,
                          xland=jnp.array([[1., 2.]], jnp.float32))
    calls = []

    def column_solver(*args):
        calls.append(True)
        np.testing.assert_allclose(np.asarray(args[0])[0], data["dry_theta_in"], rtol=0, atol=4.e-5)
        np.testing.assert_array_equal(np.asarray(args[-1]), [1., 2.])
        shape = (2, state.theta.shape[0])
        return PhysicsTendency(state_replacements={
            "theta": jnp.broadcast_to(jnp.asarray(data["dry_theta_out"], jnp.float32), shape),
            "qv": jnp.broadcast_to(jnp.asarray(data["qv_out"], jnp.float32), shape)},
            diagnostics={"Nn": jnp.full(shape, 1.e8, jnp.float32)})

    monkeypatch.setattr(adapters, f"wdm{6 if option == 16 else 7}_physics_tendency", column_solver)
    result = getattr(adapters, f"wdm{6 if option == 16 else 7}_adapter")(state, 18.)
    assert calls == [True], "The intended WDM adapter must run"
    np.testing.assert_allclose(np.asarray(result.theta)[:,0,0], data["theta_m_out"], rtol=0, atol=6.e-5)
    assert result.theta.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(result.qv), np.asarray(col("qv_out")))
    np.testing.assert_array_equal(np.asarray(result.Nn), 1.e8)


def test_liquid_only_precipitation_is_unchanged():
    state = state_for()
    result = adapters._apply_mp_replacements(state, PhysicsTendency(
        accumulator_increments={"rain_acc": jnp.array([.125, .25], jnp.float32)}), ny=1, nx=2, nz=40)
    np.testing.assert_array_equal(np.asarray(result.rain_acc), [[.125, .25]])
