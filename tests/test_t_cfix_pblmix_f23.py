"""t-cfix-pblmix F23 (class B): LSM caller operands WRF hands over in surface_driver.

* noahmpdrv.F:817-818 passes CM=CMXY, CH=CHXY (Noah-MP's own previous values);
  with opt_sfc=2 (SFCDIF2 iterates from them) a selected sfclay CH/CM such as
  QNSESFC's must not replace them.
* surface_driver.F:1918-1936: slope_rad=1 runs TOPO_RAD_ADJ_DRVR, which rewrites
  SWDOWN and GSW before every LSM; RUC consumes GSW, SSiB SWDOWN.  The port holds
  that topographic pair as radiation_diagnostics.swnorm / swup_topographic (RRTMG:
  == swdown/swup at slope_rad=0; SSiB's own radiation leaves swnorm = 0, so SSiB
  reads swdown at slope_rad=0 and refuses slope_rad=1 -- c40 fix, lane t-ysuqc).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class _Stop(Exception):
    pass


@pytest.mark.parametrize("opt_sfc", [1, 2])
def test_noahmp_sfcdif_seed_follows_wrf_chxy(monkeypatch, opt_sfc):
    from test_noahmp_coupler import _build, HAVE_TABLES
    from gpuwrf.physics import noahmp_coupler as nc
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.coupling import temf
    if not HAVE_TABLES:
        pytest.skip("WRF parameter tables unavailable")
    state, land, static, rad, clock = _build()
    shape = state.t_skin.shape
    a = lambda v: jnp.full(shape, v, jnp.float32)  # noqa: E731
    flux = SurfaceFluxes(ustar=a(.3), theta_flux=a(.01), qv_flux=a(1e-5), tau_u=a(-.09), tau_v=a(0.),
                         rhosfc=a(1.2), fltv=a(.01), xland=state.xland, wspd=None)
    selected_ch, selected_cm = a(0.0123), a(0.0456)   # a selected-sfclay CH/CM (QNSESFC-like)
    diag = temf.SurfaceDiagnostics(flux, a(30.), a(25.), a(2.), a(1.), a(295.), a(300.), a(.008), a(.011),
                                   a(0.), state.roughness_m, a(.02), a(.015), a(.015), selected_ch, selected_cm)
    land = land.replace(ch=a(0.0071), cm=a(0.0082))
    seen = {}

    def firewall(values, shape_):
        seen["ch"], seen["cm"] = values[0].ch, values[0].cm
        raise _Stop

    monkeypatch.setattr(nc, "_small_grid_firewall", firewall)
    with pytest.raises(_Stop):
        nc.noahmp_surface_adapter(state, land, static, radiation=rad, clock=clock, dt=18.,
                                  surface_diagnostics=diag, options=NoahMPOptions(opt_sfc=opt_sfc))
    own = opt_sfc == 2
    np.testing.assert_array_equal(np.asarray(seen["ch"]), np.asarray(land.ch if own else selected_ch))
    np.testing.assert_array_equal(np.asarray(seen["cm"]), np.asarray(land.cm if own else selected_cm))


def _topographic(rdiag):
    f = lambda x, s: (jnp.asarray(x) * s).astype(jnp.asarray(x).dtype)  # noqa: E731
    return rdiag._replace(swnorm=f(rdiag.swdown, 0.0) + 640.0, swdown=f(rdiag.swdown, 0.0) + 700.0,
                          swup=f(rdiag.swdown, 0.0) + 120.0, swup_topographic=f(rdiag.swdown, 0.0) + 110.0)


def test_ssib_receives_unadjusted_swdown_at_slope_rad_0(cpu_pallas_interpret, tmp_path, monkeypatch):
    from test_s2_ssib_wiring import _case
    from gpuwrf.coupling import ssib_surface_hook as hook
    from gpuwrf.runtime import operational_mode as om
    _grid, state, nml, carry = _case(tmp_path)
    seeded = om._initial_carry_for_run(state, nml)
    seeded = seeded.replace(radiation_diagnostics=_topographic(seeded.radiation_diagnostics))
    seen = {}

    def step(*args, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(hook, "ssib_surface_step", step)
    with pytest.raises(_Stop):
        om._physics_step_forcing(seeded, nml, 0., run_radiation=False, first_timestep=True)
    # slope_rad=0: TOPO_RAD_ADJ_DRVR does not run, SSiB gets the unadjusted SWDOWN (700), never
    # swnorm (0 from ssib_radiation; c40 blocker, tests/test_ssib_swdown_caller.py). slope_rad=1
    # with SSiB is refused: ssib_radiation has no topographic pair.
    np.testing.assert_array_equal(np.asarray(seen["radiation"][0]), np.full((2, 3), 700., np.float32))


def test_ruc_receives_topographic_gsw(cpu_pallas_interpret, tmp_path, monkeypatch):
    import dataclasses
    from test_s2_ssib_wiring import _case
    from gpuwrf.coupling import ruc_surface_hook as hook
    from gpuwrf.runtime import operational_mode as om
    _grid, state, nml, carry = _case(tmp_path)
    seeded = om._initial_carry_for_run(state, nml)
    from types import SimpleNamespace
    z = jnp.zeros((2, 3), jnp.float32)
    land = SimpleNamespace(alb=z + 0.2, emiss=z + 0.97)   # read only for radiation surface properties
    seeded = seeded.replace(radiation_diagnostics=_topographic(seeded.radiation_diagnostics),
                            ruc_land=land, ruc_rad=(z, z))
    ruc = dataclasses.replace(nml, sf_surface_physics=3, ssib_static=None, ssib_land=None, ruc_static=object())
    seen = {}

    def step(*args, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(hook, "ruc_surface_step", step)
    with pytest.raises(_Stop):
        om._physics_step_forcing(seeded, ruc, 0., run_radiation=False, first_timestep=True)
    # GSW = topographic SWDOWN - topographic SWUP (640 - 110), not 700 - 120.
    np.testing.assert_array_equal(np.asarray(seen["radiation"].gsw), np.full((2, 3), 530., np.float32))
