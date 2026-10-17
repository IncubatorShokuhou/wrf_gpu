"""c40 release blocker: SSiB must receive WRF's SWDOWN (lane t-ysuqc, 2026-10-16).

surface_driver.F:1918-1936: only slope_rad=1 runs TOPO_RAD_ADJ_DRVR, which
rewrites SWDOWN with the topographic flux before the LSMs. The SSiB radiation
(``ssib_radiation``: Dudhia SW + RRTMG LW) has no topographic pair and leaves
``swnorm`` = 0 -- like WRF's SWNORM history at slope_rad=0 -- so feeding SSiB
``swnorm`` removed all daytime shortwave (c40 GPU: SWDOWN 0, T2 -20 K at noon).
The test drives the real producer (no synthetic diagnostics) at a daytime clock.
"""
from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class _Stop(Exception):
    pass


def _daytime_case(tmp_path):
    from test_s2_ssib_wiring import _case
    from gpuwrf.coupling.physics_couplers import RRTMGRadiationStatic
    from gpuwrf.coupling.ssib_surface_hook import ssib_radiation
    from gpuwrf.runtime import operational_mode as om
    _grid, state, nml, _carry = _case(tmp_path)
    z = np.zeros((2, 3), np.float32)
    static = RRTMGRadiationStatic(xlat_deg=z + 42., xlong_deg=z, terrain_height_m=z, slope_rad=z,
                                  slope_azimuth_rad=z, sina=z, cosa=z + 1.)   # c40 latitude, flat
    nml = dataclasses.replace(nml, time_utc=datetime(2026, 10, 13, 12, tzinfo=timezone.utc), radiation_static=static)
    seeded = om._initial_carry_for_run(state, nml)
    _rate, diag = ssib_radiation(state, seeded.ssib_land, nml.ssib_static, nml, 0., None)
    return om, nml, seeded.replace(radiation_diagnostics=diag), diag


def test_ssib_surface_step_receives_real_daytime_swdown(cpu_pallas_interpret, tmp_path, monkeypatch):
    from gpuwrf.coupling import ssib_surface_hook as hook
    om, nml, carry, diag = _daytime_case(tmp_path)
    swdown = np.asarray(diag.swdown)
    assert swdown.min() > 500. and np.all(np.asarray(diag.swnorm) == 0.)   # the producer contract
    seen = {}

    def step(*args, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(hook, "ssib_surface_step", step)
    with pytest.raises(_Stop):
        om._physics_step_forcing(carry, nml, 0., run_radiation=False, first_timestep=True)
    np.testing.assert_array_equal(np.asarray(seen["radiation"][0]), swdown.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(seen["radiation"][1]), np.asarray(diag.glw, np.float32))


def test_ssib_refuses_slope_rad_without_a_topographic_pair(cpu_pallas_interpret, tmp_path):
    om, nml, carry, _diag = _daytime_case(tmp_path)
    with pytest.raises(NotImplementedError, match="slope_rad=1 with SSiB"):
        om._physics_step_forcing(carry, dataclasses.replace(nml, slope_rad=1), 0.,
                                 run_radiation=False, first_timestep=True)
