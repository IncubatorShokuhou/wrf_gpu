"""Lifecycle/routing regression tests; scientific truth is the pristine oracle."""
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.coupling.topographic_shadow import cold_kwargs, held_kwargs, seed
from gpuwrf.io.restart import _carry_to_payload, _validate_radiation_diagnostics_schema
from gpuwrf.io.topographic_shadow_schema import marker, validate
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry


def _namelist(on=True):
    return SimpleNamespace(topo_shading=int(on), slope_rad=1,
                           ra_sw_physics=4, ra_lw_physics=4,
                           boundary_config=SimpleNamespace(force_geopotential=False))


def _diag(height=None):
    z = jnp.zeros((4, 4), jnp.float32)
    values = [z] * 15
    values[10] = z.astype(jnp.int32)
    return pc.RRTMGRadiationDiagnostics(*values, shadow_height_m=height)


@dataclass(frozen=True)
class _Carry:
    state: object
    radiation_diagnostics: object

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def _fixtures():
    path = Path(__file__).resolve().parents[2] / "test_rrtmg_topographic_coupling.py"
    spec = importlib.util.spec_from_file_location("topo_grid_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grid = module._grid()
    # Production cold-init seam: under release defaults (MP_RE=1) RRTMG needs
    # the held microphysics radii, so the fixture runs with both flag sets.
    state = module._state(grid).ensure_conditional_leaves(
        mp_physics=8, use_mp_re=1, ra_lw_physics=4, ra_sw_physics=4)
    return grid, state


def test_cold_seed_is_real_zero_and_off_has_no_leaf_or_allocation():
    state = SimpleNamespace(t_skin=jnp.ones((4, 4), jnp.float64))
    original = _Carry(state, _diag(jnp.ones((4, 4), jnp.float32)))
    assert seed(original, _namelist(False)) is original
    assert cold_kwargs(_namelist(False), state) == {}
    assert held_kwargs(_namelist(False), _diag()) == {}
    started = seed(original, _namelist())
    assert started.radiation_diagnostics.shadow_height_m.dtype == jnp.float32
    np.testing.assert_array_equal(started.radiation_diagnostics.shadow_height_m, 0)
    assert cold_kwargs(_namelist(), state)["nested_shadow"] is True
    assert held_kwargs(_namelist(), started.radiation_diagnostics)["previous_shadow_height_m"] is started.radiation_diagnostics.shadow_height_m
    with pytest.raises(ValueError, match="HGT_SHAD"):
        held_kwargs(_namelist(), _diag())


def test_goddard_does_not_materialize_optional_shadow_height():
    from gpuwrf.coupling.goddard_radiation import initial_diagnostics
    state = SimpleNamespace(t_skin=jnp.zeros((3, 5), jnp.float32), qv=jnp.zeros((8, 3, 5), jnp.float32))
    diagnostics = initial_diagnostics(state)
    assert diagnostics.shadow_height_m is None
    assert marker(diagnostics) is None
    from gpuwrf.io.restart_schema import validate_goddard_radiation, GODDARD_RADIATION_SCHEMA_VERSION
    validate_goddard_radiation(diagnostics, schema_version=GODDARD_RADIATION_SCHEMA_VERSION,
                               state_shape=state.qv.shape)
    with pytest.raises(ValueError, match="Goddard.*shadow"):
        validate_goddard_radiation(diagnostics._replace(shadow_height_m=state.t_skin),
            schema_version=GODDARD_RADIATION_SCHEMA_VERSION, state_shape=state.qv.shape)


def test_legacy_radiation_keeps_shadow_height_absent_at_restart():
    from gpuwrf.coupling.flg_gfdl_radiation import initial_diagnostics
    from gpuwrf.io.flg_gfdl_schema import marker as legacy_marker, validate as validate_legacy
    state = SimpleNamespace(t_skin=jnp.zeros((3, 5), jnp.float32), qv=jnp.zeros((8, 3, 5), jnp.float32))
    diagnostics = initial_diagnostics(state)
    assert diagnostics.shadow_height_m is None
    options = SimpleNamespace(ra_lw_physics=7, ra_sw_physics=7)
    schema = legacy_marker(diagnostics, None)
    validate_legacy(diagnostics, None, options, schema, state_shape=state.qv.shape)
    with pytest.raises(ValueError, match="foreign.*shadow"):
        validate_legacy(diagnostics._replace(shadow_height_m=state.t_skin), None,
                         options, schema, state_shape=state.qv.shape)


def test_rrtmg_and_slope_keep_eot_while_shadow_uses_its_own_clock(monkeypatch):
    grid, state = _fixtures()
    seen = {}
    original = pc._rrtmg_topography_state
    def watch(static, grid, geometry, **kwargs):
        seen.update(geometry=geometry, **kwargs)
        return original(static, grid, geometry, **kwargs)
    monkeypatch.setattr(pc, "_rrtmg_topography_state", watch)
    static = pc.build_radiation_static_from_wrf_fields(
        jnp.full((4, 4), 28.3), jnp.full((4, 4), -16.4), grid.terrain_height,
        dx_m=3000., dy_m=3000.)
    columns = pc._rrtmg_column_inputs(state, grid, time_utc=datetime(2026, 2, 28, tzinfo=timezone.utc),
        lead_seconds=7.5 * 3600, solar_lead_seconds=7.75 * 3600,
        radiation_static=static, topo_shading=1, slope_rad=1,
        previous_shadow_height_m=jnp.zeros((4, 4), jnp.float32), nested_shadow=True)
    geometry, topography = columns[4:6]
    np.testing.assert_array_equal(columns[0].coszen, geometry.coszen)
    np.testing.assert_array_equal(topography.hour_angle_rad, geometry.hour_angle_rad)
    assert np.max(np.abs(np.asarray(seen["shadow_geometry"].hour_angle_rad - geometry.hour_angle_rad))) > .01
    assert topography.shadow_height_m.dtype == jnp.float32


def test_compiled_refresh_holds_height_and_passes_own_previous_field(monkeypatch):
    grid, state = _fixtures()
    nml = op.OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics,
                               dt_s=18., cu_physics=0, topo_shading=1, slope_rad=1)
    carry = initial_operational_carry(state).replace(radiation_diagnostics=_diag(jnp.zeros((4, 4), jnp.float32)))
    def fake_radiation(state, grid, **kwargs):
        assert "previous_shadow_height_m" in kwargs
        old = kwargs["previous_shadow_height_m"]
        return jnp.zeros_like(state.theta), _diag(old + jnp.float32(1))
    monkeypatch.setattr(op, "rrtmg_theta_tendency", fake_radiation)
    run = jax.jit(lambda c, do_rad: op._refresh_rrtmg_driver(c, nml, jnp.float32(0), do_rad, None))
    updated = run(carry, True)
    held = run(updated, False)
    again = run(held, True)
    np.testing.assert_array_equal(updated.radiation_diagnostics.shadow_height_m, 1)
    np.testing.assert_array_equal(held.radiation_diagnostics.shadow_height_m, 1)
    np.testing.assert_array_equal(again.radiation_diagnostics.shadow_height_m, 2)
    run.clear_cache()


def test_restart_real_payload_marks_height_and_refuses_missing_leaf_or_marker():
    grid, state = _fixtures()
    carry = initial_operational_carry(state).replace(radiation_diagnostics=_diag(jnp.zeros((4, 4), jnp.float32)))
    payload = _carry_to_payload(carry)
    assert payload["topographic_shadow_schema_version"] == 1
    _validate_radiation_diagnostics_schema(payload, namelist=_namelist())
    without_marker = dict(payload)
    without_marker.pop("topographic_shadow_schema_version")
    with pytest.raises(ValueError, match="HGT_SHAD"):
        _validate_radiation_diagnostics_schema(without_marker, namelist=_namelist())
    without_height = dict(payload, extra_fields=dict(payload["extra_fields"], radiation_diagnostics=_diag()))
    with pytest.raises(ValueError, match="HGT_SHAD"):
        _validate_radiation_diagnostics_schema(without_height, namelist=_namelist())


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_restart_rejects_nonfinite_height(value):
    array = np.zeros((4, 4), np.float32)
    array[1, 1] = value
    with pytest.raises(ValueError, match="nonfinite"):
        validate(_diag(array), 1, (4, 4), namelist=_namelist())


def test_restart_off_legacy_is_accepted_but_bad_shape_dtype_and_version_are_not():
    validate(None, None, None, namelist=_namelist(False))
    for array, version in ((np.zeros((4, 3), np.float32), 1),
                           (np.zeros((4, 4), np.float64), 1),
                           (np.zeros((4, 4), np.float32), 0)):
        with pytest.raises(ValueError, match="HGT_SHAD"):
            validate(_diag(array), version, (4, 4), namelist=_namelist())
    assert marker(_diag()) is None
