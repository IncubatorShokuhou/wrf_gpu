"""t-cfix-cu F18: the operational Grell-Freitas (cu=3) caller hands GFDRV WRF's inputs.

WRF GFDRV (module_cu_gf_wrfdrv.F) reads: the held RTHFTEN/RQVFTEN, RTHRATEN and this
step's RTHBLTEN/RQVBLTEN in the deep forced sounding (:412-417), PSUR = p8w(1)*.01 with
P8W = p_hyd_w (:385), omega from the STAGGERED w (:471), grid%kpbl of the active PBL
(first_rk_step_part1.F:1397) and writes only ids+4..ide-5 / jds+4..jde-5 unless
periodic (:264-277).  Kernel-level parity of these inputs against pristine GFDRV is in
test_t_cfix_cu_gf_ishallow.py; here the operational slot -> adapter -> kernel seam.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _case():
    from gpuwrf.coupling.cu_forcing import initial_cu_forcing
    from gpuwrf.runtime.operational_state import initial_operational_carry

    spec = importlib.util.spec_from_file_location("rk_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(ny=11, nx=12, nz=8)
    state = f._state(grid).replace(pblh=jnp.full((11, 12), 900.0))
    nml = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0), run_physics=True,
                              use_noahmp=False, mp_physics=8, cu_physics=3, bl_pbl_physics=0,
                              sf_sfclay_physics=0, rad_rk_tendf=1, use_flux_advection=True)
    th = jnp.full_like(state.theta, 3.0e-5)
    qv = jnp.full_like(state.theta, 2.0e-8)
    ra = jnp.full_like(state.theta, -1.2e-5)
    carry = initial_operational_carry(state).replace(cu_forcing=(th, qv), rthraten=ra)
    return grid, state, carry, nml, (th, qv, ra)


def _spy(monkeypatch):
    from gpuwrf.coupling import scan_adapters

    seen = {}
    real = scan_adapters.gfdrv_batched

    def spy(*args, **kw):
        seen["args"], seen["kw"] = args, kw
        return real(*args, **kw)

    monkeypatch.setattr(scan_adapters, "gfdrv_batched", spy)
    return seen


def test_operational_gf_slot_hands_wrf_inputs_to_gfdrv(monkeypatch):
    from gpuwrf.coupling.cuplus import mynn_kpbl
    from gpuwrf.coupling.physics_couplers import _wrf_hydrostatic_pressure_profiles_from_state
    from gpuwrf.runtime import operational_mode as om

    grid, state, carry, nml, (th, qv, ra) = _case()
    seen = _spy(monkeypatch)
    om._physics_step_forcing(carry, nml, jnp.asarray(0.0), run_radiation=False)
    nz, ny, nx = state.theta.shape
    cols1 = lambda a: np.concatenate([np.zeros((ny * nx, 1)), np.moveaxis(np.asarray(a, np.float64), 0, -1)  # noqa: E731
                                      .reshape(ny * nx, nz)], axis=1)
    kw, args = seen["kw"], seen["args"]
    np.testing.assert_allclose(np.asarray(kw["rthften1"]), cols1(th), rtol=0)
    np.testing.assert_allclose(np.asarray(kw["rqvften1"]), cols1(qv), rtol=0)
    np.testing.assert_allclose(np.asarray(kw["rthraten1"]), cols1(ra), rtol=0)
    _p, _w, psfc = _wrf_hydrostatic_pressure_profiles_from_state(state, grid.metrics, output_dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(kw["psur_pa"]), np.asarray(psfc).reshape(-1), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(args[8]), cols1(np.asarray(state.w)[:nz]))  # staggered w
    candidates = []
    for g in (9.81, 9.80665):  # phy_prep z_at_w uses 9.81; the retained GF view used 9.80665
        zw = np.asarray(state.ph, np.float64) / g
        candidates.append(np.asarray(mynn_kpbl(jnp.asarray(zw - zw[0][None]), state.pblh,
                                               mass_levels=nz).reshape(-1)))
    assert any(np.array_equal(np.asarray(args[16]), c) for c in candidates)
    assert not np.array_equal(np.asarray(args[16]), np.full(ny * nx, 2))  # an actual PBLH level, not the fallback


def test_gf_writes_only_wrf_interior_columns(monkeypatch):
    from gpuwrf.coupling import scan_adapters

    grid, state, _carry, _nml, _forcing = _case()

    def all_active(*args, **kw):
        ncol, k1 = args[0].shape
        one = jnp.ones((ncol, k1))
        return {"RTHCUTEN": one, "RQVCUTEN": one, "RQCCUTEN": one, "RQICUTEN": one,
                "RAINCV": jnp.ones((ncol,))}

    monkeypatch.setattr(scan_adapters, "gfdrv_batched", all_active)
    rates, rain = scan_adapters.gf_adapter(state, 60.0, grid, return_rates=True)
    ny, nx = rain.shape
    expect = np.zeros((ny, nx))
    expect[4:ny - 4, 4:nx - 4] = 1.0  # ids+4..ide-5 / jds+4..jde-5 (0-based 4..n-5)
    np.testing.assert_array_equal(np.asarray(rain), expect)
    np.testing.assert_array_equal(np.asarray(rates["theta"])[0], expect)
    rates_p, rain_p = scan_adapters.gf_adapter(state, 60.0, grid, return_rates=True,
                                               periodic_x=True, periodic_y=True)
    assert np.all(np.asarray(rain_p) == 1.0)


def test_gf_reads_the_kpbl_published_by_the_gfs_pbl(monkeypatch):
    """grid%kpbl = BL_GFS KPBL2D (pbl_driver.F:1418), read by GFDRV (first_rk_step_part1.F:1397):
    the GF slot gets the KPBL the GFS kernel computed this step, not a PBLH-derived stand-in."""
    from gpuwrf.coupling import scan_adapters
    from gpuwrf.runtime import operational_mode as om

    grid, state, carry, nml, _forcing = _case()
    nml = dataclasses.replace(nml, bl_pbl_physics=3, sf_sfclay_physics=3)  # WRF: GFS PBL pairs with sf 3
    monkeypatch.setattr(om, "_phys_tend_rk_wrf_enabled", lambda: True)
    ncol = state.theta.shape[1] * state.theta.shape[2]
    sentinel = jnp.asarray(3 + (np.arange(ncol) % 2), jnp.int32)
    produced = []

    def with_sentinel(real):
        def gfs(*a, **k):
            out = dict(real(*a, **k))
            assert out["kpbl"].shape == (ncol,)
            produced.append(real.__name__)
            out["kpbl"] = sentinel
            return out
        return gfs

    # sf=3 feeds the BL_GFS REAL wrapper (t-cfix-sfc F13); other surface paths the plain kernel.
    for name in ("gfs_columns", "gfs_bl_real_double"):
        monkeypatch.setattr(scan_adapters, name, with_sentinel(getattr(scan_adapters, name)))
    seen = _spy(monkeypatch)
    om._physics_step_forcing(carry, nml, jnp.asarray(0.0), run_radiation=False)
    assert produced  # the GFS PBL actually ran and published its KPBL2D
    np.testing.assert_array_equal(np.asarray(seen["args"][16]), np.asarray(sentinel))
