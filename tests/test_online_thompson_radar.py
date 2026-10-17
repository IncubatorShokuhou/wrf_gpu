"""Online Thompson diagnostics against independently compiled pristine WRF."""
from pathlib import Path
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.diagnostics import thompson_radar as radar


FIXTURE = Path(__file__).resolve().parents[1] / "proofs/u2_diag/radar_oracle.npz"


@pytest.fixture(scope="module")
def reference():
    manifest = json.loads(FIXTURE.with_name("radar_oracle_manifest.json").read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    assert manifest["real_history_field_columns"] == 24
    assert all(manifest["census"][f"{species}_levels"] > 0 for species in ("rain", "snow", "graupel"))
    with np.load(FIXTURE) as data:
        yield dict(data), manifest


def _inputs(arrays):
    return tuple(jnp.asarray(arrays[:, i]) for i in (0, 2, 3, 4, 5, 6, 8, 9))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bottom_only", [False, True])
def test_pristine_active_radar_operator(reference, compiled, bottom_only):
    data, manifest = reference
    args = _inputs(data["inputs"])
    function = lambda *a: radar.thompson_reflectivity(*a, ke_diag=1 if bottom_only else None)
    if compiled:
        function = jax.jit(function)
    actual = np.asarray(function(*args))
    expected = data["bottom" if bottom_only else "dbz"]
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=manifest["tolerance"]["max_absolute_dbz"], rtol=0)
    if bottom_only:
        np.testing.assert_array_equal(actual[:, 1:], -35.)


def test_final_ng_pristine_private_block(reference):
    data, manifest = reference
    actual = np.asarray(jax.jit(radar.final_graupel_number)(*map(jnp.asarray, np.moveaxis(data["ng_inputs"], 1, 0))))
    expected = data["balanced_ng"]
    np.testing.assert_array_max_ulp(actual, expected, maxulp=manifest["tolerance"]["ng_max_ulp"])
    # This detects using evolved-but-unbalanced ng or an entry reconstruction.
    assert not np.allclose(data["ng_inputs"][:, 1], expected, rtol=1e-3, atol=1e-6)
    np.testing.assert_array_equal(actual[data["ng_inputs"][:, 0] <= 1e-12], 0.)


def test_pallas_column_register_path(reference):
    data, manifest = reference
    # Constructed mixed warm/rain/snow/graupel column exercises all operator branches.
    actual = np.asarray(radar.thompson_reflectivity(*_inputs(data["inputs"][31:32]), interpret=True))
    np.testing.assert_allclose(actual, data["dbz"][31:32], atol=manifest["tolerance"]["max_absolute_dbz"], rtol=0)


def test_explicit_ng_controls_active_graupel(reference):
    data, _ = reference
    args = list(_inputs(data["inputs"][29:30]))
    baseline = np.asarray(radar.thompson_reflectivity(*args))
    args[5] = args[5] * 10.
    changed = np.asarray(radar.thompson_reflectivity(*args))
    assert np.max(np.abs(changed - baseline)) > 1.


def test_wet_snow_mutant_is_detected(reference, monkeypatch):
    data, manifest = reference
    original = radar._wet_particle_z
    monkeypatch.setattr(radar, "_wet_particle_z", lambda d, fm: original(d, fm) * .5)
    actual = np.asarray(radar.thompson_reflectivity(*_inputs(data["inputs"][27:28])))
    assert np.max(np.abs(actual - data["dbz"][27:28])) > manifest["tolerance"]["max_absolute_dbz"] * 100


def test_adapter_context_has_raw_vapor_and_private_ng(monkeypatch):
    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.physics.thompson_column import ThompsonColumnState
    q = jnp.asarray([[1e-3, 1e-4, 1e-5]], jnp.float32)
    z = jnp.zeros_like(q)
    column = ThompsonColumnState(q, z, q, z, q, q, z, q, q + 270., q + 90000., q + 1.)
    raw_qv, balanced_ng = -q, q + 123.
    output = column.replace(qv=q + .01, Ng=q + 10.)
    precipitation = {"rain": jnp.asarray([1.], jnp.float32)}
    state, next_state = object(), object()
    calls = []
    monkeypatch.setattr(pc, "_thompson_column_from_state", lambda state, grid: column)
    def producer(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("return_radar_context", False):
            return output, precipitation, (raw_qv, balanced_ng)
        return output, precipitation
    monkeypatch.setattr(pc, "step_thompson_column_with_precip", producer)
    monkeypatch.setattr(pc, "_state_from_thompson_output", lambda *a: next_state)
    result = pc.thompson_adapter(state, 18., return_radar_context=True)
    assert len(result) == 3 and result[0] is next_state and result[1] is precipitation
    context = result[2]
    assert context.qv is raw_qv and context.ng is balanced_ng
    assert context.qr is output.qr and context.nr is output.Nr and context.temperature is output.T
    assert pc.thompson_adapter(state, 18.) is next_state
    assert "return_radar_context" not in calls[-1]


def test_input_shape_and_ke_diag_refusal():
    args = [jnp.ones((2, 3), jnp.float32)] * 8
    with pytest.raises(ValueError, match="ke_diag"):
        radar.thompson_reflectivity(*args, ke_diag=0)
    with pytest.raises(ValueError, match="identical"):
        radar.thompson_reflectivity(*(args[:-1] + [jnp.ones((3,))]))


def test_pristine_bottom_active_melting_and_exact_full_projection():
    path = FIXTURE.with_name("radar_bottom_oracle.npz")
    manifest = json.loads(path.with_name("radar_bottom_oracle_manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    assert manifest["census"]["bottom_active_snow_columns"] == 8
    with np.load(path) as data:
        args = _inputs(data["inputs"])
        expected = data["bottom"]
    actual = np.asarray(jax.jit(lambda *a: radar.thompson_reflectivity(*a, ke_diag=1))(*args))
    np.testing.assert_allclose(actual, expected, atol=manifest["tolerance"]["max_absolute_dbz"], rtol=0)
    full = np.asarray(jax.jit(radar.thompson_reflectivity)(*args))
    np.testing.assert_array_equal(actual[:, 0], full[:, 0])
    # The GPU register implementation also uses a scalar DOUBLE bin carry.
    interpreted = np.asarray(radar.thompson_reflectivity(*(x[:1] for x in args), ke_diag=1, interpret=True))
    np.testing.assert_allclose(interpreted, expected[:1], atol=manifest["tolerance"]["max_absolute_dbz"], rtol=0)
