"""t-cfix-sfc F21: sf_sfclay_physics=0 runs no surface layer and no land model.

module_surface_driver.F:1490 ``if (sf_sfclay_physics .eq. 0) return`` precedes PSFC
(:1988), every surface-layer call and every LSM/urban/lake call, so UST, HFX, QFX,
QSFC, ZNT, MOL, TSK and the land state keep their values; the disabled-surface history
holds PSFC/T2/Q2/U10/V10/TH2 (io/disabled_surface_history.py). The same return also
precedes the SST_UPDATE block (:1638), the SST skin (:1795) and ocean_driver (:4472).
"""
from __future__ import annotations

from types import SimpleNamespace

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

HELD = ('ustar', 'hfx', 'qfx', 'qsfc', 'roughness_m', 'mol', 't_skin', 'theta_flux', 'qv_flux', 'tau_u', 'tau_v',
        'fltv', 'rhosfc')


def _case(*, noahmp):
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    grid = _grid(nz=12, ny=3, nx=3)
    state = _base_state(grid)
    state = state.replace(ustar=jnp.full((3, 3), .37), hfx=jnp.full((3, 3), 55.), qfx=jnp.full((3, 3), 3.e-5),
                          qsfc=jnp.full((3, 3), .009), mol=jnp.full((3, 3), -.2), theta_flux=jnp.full((3, 3), .04),
                          qv_flux=jnp.full((3, 3), 2.e-5))
    nl = _namelist(grid, mp_physics=0, cu_physics=0, bl_pbl_physics=0, sf_sfclay_physics=0, use_noahmp=noahmp,
                   sf_surface_physics=4 if noahmp else 0, ra_lw_physics=0, ra_sw_physics=0)
    return state, nl


def _forbid_surface(monkeypatch, om):
    from gpuwrf.coupling import sfclay_selected

    def forbidden(*args, **kwargs):
        raise AssertionError('sf_sfclay_physics=0 ran a surface-layer or land call')
    monkeypatch.setattr(om, 'surface_adapter', forbidden)
    monkeypatch.setattr(om, 'noahmp_surface_step', forbidden)
    monkeypatch.setattr(sfclay_selected, 'selected_surface_step', forbidden)


@pytest.mark.parametrize('noahmp', (False, True))
def test_sf0_runs_no_surface_layer_and_no_land_model(monkeypatch, noahmp):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    state, nl = _case(noahmp=noahmp)
    _forbid_surface(monkeypatch, om)
    carry = initial_operational_carry(state)
    if noahmp:
        zero = jnp.zeros(state.t_skin.shape, jnp.float32)
        land = SimpleNamespace(ch=zero + .01, cm=zero + .02)
        carry = carry.replace(noahmp_land=land, radiation_diagnostics=SimpleNamespace(swnorm=zero, glw=zero,
                                                                                       coszen=zero))
    result = om._physics_step_forcing(carry, nl, 0., run_radiation=False, first_timestep=True)
    for name in HELD:
        np.testing.assert_array_equal(np.asarray(getattr(result.state, name)), np.asarray(getattr(state, name)),
                                      err_msg=name)
    if noahmp:
        assert result.carry.noahmp_land is carry.noahmp_land


def test_sf0_also_skips_ocean_mixed_layer_sst_update_and_sst_skin(monkeypatch):
    """Admitted sf0/bl0/Noah-MP/OML (rv-t-cfix SS05) plus aux4 SST and SST-skin forcing."""
    from gpuwrf.coupling import ocean_surface_hook, sst_skin_hook
    from gpuwrf.io import lower_boundary
    from gpuwrf.physics.ocean_mixed_layer import initialize
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry, seed_history_diagnostics
    state, nl = _case(noahmp=True)
    nl = dataclasses.replace(nl, sf_ocean_physics=1)
    om._resolve_operational_suite(nl)  # the combination is admitted
    _forbid_surface(monkeypatch, om)
    calls = []

    def record(name):
        def call(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f'sf_sfclay_physics=0 ran {name}')
        return call
    monkeypatch.setattr(ocean_surface_hook, 'ocean_surface_step', record('ocean_driver'))
    monkeypatch.setattr(lower_boundary, 'apply_lower_boundary_fields', record('SST_UPDATE'))
    monkeypatch.setattr(sst_skin_hook, 'apply_sst_skin', record('SST skin'))
    zero = jnp.zeros(state.t_skin.shape, jnp.float32)
    ocean = initialize(state.t_skin)
    carry = initial_operational_carry(state).replace(
        noahmp_land=SimpleNamespace(ch=zero + .01, cm=zero + .02), ocean=ocean,
        sst_skin=SimpleNamespace(bulk_sst=zero + 290.), history_diagnostics=seed_history_diagnostics(state),
        radiation_diagnostics=SimpleNamespace(swnorm=zero, glw=zero, coszen=zero))
    sst = {'SST': jnp.full(state.t_skin.shape, 301.25, jnp.float32)}
    result = om._physics_step_forcing(carry, nl, 0., run_radiation=False, first_timestep=True, lower_fields=sst)
    assert calls == []
    import jax
    held = jax.tree_util.tree_leaves(ocean)
    assert held and len(jax.tree_util.tree_leaves(result.carry.ocean)) == len(held)
    for after, before in zip(jax.tree_util.tree_leaves(result.carry.ocean), held):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    np.testing.assert_array_equal(np.asarray(result.state.t_skin), np.asarray(state.t_skin))
