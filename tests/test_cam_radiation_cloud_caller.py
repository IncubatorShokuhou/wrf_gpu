"""CAM must honour pristine WRF's non-MYNN icloud_bl=0 in both consumers.

The saturated resolved-cloud expectation is the original cal_cldfra1 branch 2
already acquired in proofs/t_camuw/cloud_oracle.json. CAM's unchanged spectral
solvers have separate pristine fresh/held LW and SW gates.
"""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import cam_radiation as cam
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_rrtm_lw_operational_wiring import _grid, _namelist, _state


def cloudy_state():
    grid = _grid(nz=4)
    state = _state(grid)
    state = state.replace(qv=jnp.full_like(state.qv, .03), qc=jnp.full_like(state.qc, 1e-4),
                          cldfra_bl=jnp.zeros_like(state.qv), qc_bl=jnp.zeros_like(state.qv),
                          qi_bl=jnp.zeros_like(state.qv))
    return grid, state


def mock_spectral_calls(monkeypatch, seen):
    def lw(op, held=None, doabsems=True):
        seen.append(op)
        return SimpleNamespace(rthratenlw=jnp.ones_like(op.t_phy), glw=op.cldfra.mean(axis=1), held=held)

    def sw(op, grid):
        seen.append(op)
        return SimpleNamespace(rthratensw=jnp.ones_like(op.t_phy), swdown=op.cldfra.mean(axis=1))

    monkeypatch.setattr(cam, "_lw_held_call", lw)
    monkeypatch.setattr(cam, "_lw_call", lw)
    monkeypatch.setattr(cam, "_sw_call", sw)


def test_cam_operands_preserve_original_non_mynn_clouds(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_SGS_CLOUD", "1")
    oracle = json.loads((Path(__file__).resolve().parents[1] / "proofs/t_camuw/cloud_oracle.json").read_text())
    saturated = [c for c in oracle["cases"] if c["input"][0] == 8 and c["branch"] == 2]
    assert saturated and all(c["fraction"] == 1 for c in saturated)
    grid, state = cloudy_state()
    # Original BL9 check_a_mundo forces resolved clouds; its unused MYNN work
    # arrays remain zero. Explicit MYNN retains the existing replacement policy.
    resolved = cam.cam_operands(state, grid, time_utc="2023-01-15T00:00:00Z", lead_seconds=3600.,
                               use_mynn_sgs_cloud=False, cloud_mp_physics=8)
    mynn = cam.cam_operands(state, grid, time_utc="2023-01-15T00:00:00Z", lead_seconds=3600.)
    np.testing.assert_array_equal(resolved.cldfra, 1.)
    np.testing.assert_array_equal(mynn.cldfra, 0.)
    np.testing.assert_array_equal(resolved.qc, mynn.qc)


@pytest.mark.parametrize("consumer", ["cam_lw_theta_tendency", "cam_sw_theta_tendency", "cam_surface_lw", "cam_surface_forcing"])
def test_each_cam_consumer_passes_original_cloud_selection(monkeypatch, consumer):
    monkeypatch.setenv("GPUWRF_MYNN_SGS_CLOUD", "1")
    grid, state = cloudy_state()
    seen = []
    mock_spectral_calls(monkeypatch, seen)
    kw = dict(time_utc="2023-01-15T00:00:00Z", lead_seconds=3600.,
              use_mynn_sgs_cloud=False, cloud_mp_physics=8)
    if consumer == "cam_surface_forcing":
        cam.cam_surface_forcing(state, grid, jnp.zeros_like(state.t_skin), jnp.zeros_like(state.t_skin),
                                ra_lw_physics=3, ra_sw_physics=3, **kw)
    else:
        getattr(cam, consumer)(state, grid, **kw)
    assert seen
    for op in seen:
        np.testing.assert_array_equal(op.cldfra, 1.)


@pytest.mark.parametrize("pbl,expected", [(9, 1.), (5, 0.)])
def test_operational_cam_heating_and_land_receive_original_clouds(monkeypatch, pbl, expected):
    monkeypatch.setenv("GPUWRF_MYNN_SGS_CLOUD", "1")
    grid, state = cloudy_state()
    nml = dataclasses.replace(_namelist(grid, ra_sw_physics=3, ra_lw_physics=3), bl_pbl_physics=pbl)
    seen = []
    mock_spectral_calls(monkeypatch, seen)
    # Use the actual operational dispatch and public couplers, replacing only
    # the independently qualified expensive spectral leaf computations.
    forcing = om._physics_step_forcing(initial_operational_carry(state), nml, 3600., run_radiation=True)
    assert np.isfinite(np.asarray(forcing.carry.rthraten)).all()
    np.testing.assert_array_equal(forcing.carry.rthraten, 2.)
    assert len(seen) == 2
    for op in seen:
        # MYNN actually updates its BL-cloud carry during this forcing pass.
        # Its radiation view must use that produced field, rather than the
        # initial zeros used in the separate direct-assembly control above.
        selected = expected if pbl == 9 else np.moveaxis(
            np.asarray(forcing.state.cldfra_bl), 0, -1).reshape(op.cldfra.shape)
        np.testing.assert_array_equal(op.cldfra, selected)
    seen.clear()
    # Surface refresh precedes the heating refresh and must get the same policy.
    monkeypatch.setattr(om, "rrtmg_radiation_diagnostics", lambda *a, **kw: SimpleNamespace(
        swnorm=jnp.zeros_like(state.t_skin), glw=jnp.zeros_like(state.t_skin), coszen=jnp.zeros_like(state.t_skin)))
    soldn, lwdn, _ = om._refresh_noahmp_rad(state, nml, 3600., True, None)
    np.testing.assert_array_equal(soldn, expected)
    np.testing.assert_array_equal(lwdn, expected)
    assert len(seen) == 2
    for op in seen:
        np.testing.assert_array_equal(op.cldfra, expected)
