"""t-cfix-pblmix F6: PBL cloud-water/ice mixing (RQCBLTEN/RQIBLTEN) for MRF/GFS/MYJ.

Oracle: the UNMODIFIED pristine WRF 4.7.1 scheme modules driven through their
public entry (proofs/t_cfix_pblmix/build_*_moist_oracle.sh) on REAL cloudy
columns of the original CPU-WRF composite run (real_columns.py) -- not a port
self-compare (E98).  DOUBLE builds gate the transcription; the REAL build gates
the operational fp32 path against WRF's own REAL-vs-DOUBLE spread.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "proofs" / "t_cfix_pblmix"
TEN = (("u", "RUBLTEN"), ("v", "RVBLTEN"), ("theta", "RTHBLTEN"),
       ("qv", "RQVBLTEN"), ("qc", "RQCBLTEN"), ("qi", "RQIBLTEN"))
WRF_KEYS = {"u": "rublten", "v": "rvblten", "theta": "rthblten", "qv": "rqvblten",
            "qc": "rqcblten", "qi": "rqiblten"}


def _fixture(name):
    return json.loads((FIX / name).read_text())


def _misses(out, ref, *, rel=1.0e-9, atol_scale=1.0e-10, tol=None):
    # atol_scale: per-field absolute floor (x field max) for (theta_new-theta)/dt cancellation.
    """Field -> max abs error for fields outside tolerance (non-finite always misses)."""
    bad = {}
    for key, wrf in TEN:
        got = np.asarray(out[WRF_KEYS[key]], np.float64)
        want = np.asarray(ref[wrf], np.float64)
        if not (np.isfinite(got).all() and np.isfinite(want).all()):
            bad[wrf] = float("nan")
            continue
        err = np.abs(got - want)
        limit = (tol[wrf] if tol is not None else
                 np.maximum(rel * np.abs(want), atol_scale * max(np.abs(want).max(), 1.0e-30)))
        if np.any(err > limit):
            bad[wrf] = float(err.max())
    return bad


# ----------------------------------------------------------------------------- MRF
def _mrf(fx, *, real=False, qi=True, qc=True):
    from gpuwrf.physics.bl_mrf import mrf_columns
    c, s = fx["inputs"]["columns"], fx["inputs"]["scalars"]
    dtype = jnp.float32 if real else jnp.float64
    with jax.enable_x64(not real):
        col = lambda n: jnp.asarray(np.asarray(c[n]), dtype)  # noqa: E731
        one = lambda n: jnp.asarray(np.asarray(s[n]), dtype)  # noqa: E731
        out = jax.jit(lambda: mrf_columns(
            col("U"), col("V"), col("T"), col("QV"), col("QC") if qc else jnp.zeros_like(col("QC")),
            col("P"), col("PI"), col("DZ"), col("Z"), psfc=one("PSFC"), znt=one("ZNT"), ust=one("UST"),
            hfx=one("HFX"), qfx=one("QFX"), tsk=one("TSK"), gz1oz0=one("GZ1OZ0"), wspd=one("WSPD"),
            br=one("BR"), psim=one("PSIM"), psih=one("PSIH"), xland=one("XLAND"), dt=54.0,
            qi=col("QI") if qi else None))()
        return {k: np.asarray(v, np.float64) for k, v in out.items()}


def _wrf_spread_tol(fx):
    """REAL gate: 2x WRF's own REAL-vs-DOUBLE build spread per field (independent of the port)."""
    r4, r8 = fx["outputs"]["r4_qi"], fx["outputs"]["r8_qi"]
    return {wrf: 2.0 * float(np.abs(np.asarray(r4[wrf]) - np.asarray(r8[wrf])).max()) + 1.0e-30
            for _, wrf in TEN}


def test_mrf_oracle_census_has_active_condensate_mixing():
    fx = _fixture("mrf_moist_fixture.json")
    ref = fx["outputs"]["r8_qi"]
    qc, qi = np.asarray(fx["inputs"]["columns"]["QC"]), np.asarray(fx["inputs"]["columns"]["QI"])
    assert len(fx["inputs"]["j"]) >= 16 and (qi.max(1) > 1.0e-6).sum() >= 8   # ice-bearing real columns
    assert (np.abs(np.asarray(ref["RQCBLTEN"])) > 1.0e-10).sum() >= 50
    assert (np.abs(np.asarray(ref["RQIBLTEN"])) > 1.0e-12).sum() >= 50
    # both regimes: stable night (06Z) and convective day (13Z) columns
    assert min(fx["inputs"]["scalars"]["BR"]) < 0.0 < max(fx["inputs"]["scalars"]["BR"])
    assert np.count_nonzero(fx["outputs"]["r8_noqi"]["RQIBLTEN"]) == 0       # FLAG_QI=.false.


def test_mrf_double_matches_pristine_with_and_without_ice_slot():
    fx = _fixture("mrf_moist_fixture.json")
    for label, qi in (("r8_qi", True), ("r8_noqi", False)):
        out = _mrf(fx, qi=qi)
        ref = fx["outputs"][label]
        assert _misses(out, ref) == {}, (label, _misses(out, ref))
        np.testing.assert_allclose(out["pblh"], ref["PBLH"], rtol=1.0e-9, atol=1.0e-6)


def test_mrf_real_operational_path_inside_wrf_real_double_spread():
    fx = _fixture("mrf_moist_fixture.json")
    out = _mrf(fx, real=True)
    assert _misses(out, fx["outputs"]["r4_qi"], tol=_wrf_spread_tol(fx)) == {}


def test_mrf_pre_fix_caller_without_condensate_misses_pristine():
    # Deletion discrimination: the old caller (QC3D zeros, no QI3D) cannot
    # reproduce WRF's condensate mixing nor its in-cloud Ri diffusivity.
    fx = _fixture("mrf_moist_fixture.json")
    bad = _misses(_mrf(fx, qc=False, qi=False), fx["outputs"]["r8_qi"])
    assert {"RQCBLTEN", "RQIBLTEN", "RTHBLTEN"} <= set(bad), bad


def _column_state(nz=6):
    from test_v013_operational_smoke import _grid, _base_state
    grid = _grid(nz=nz, ny=1, nx=2)
    state = _base_state(grid)
    k = jnp.arange(nz).reshape(nz, 1, 1)
    qc = jnp.broadcast_to(jnp.where((k >= 1) & (k <= 2), 2.0e-4, 0.0), (nz, 1, 2)).astype(state.qc.dtype)
    qi = jnp.broadcast_to(jnp.where((k >= 3) & (k <= 4), 6.0e-5, 0.0), (nz, 1, 2)).astype(state.qi.dtype)
    return grid, state.replace(_cast=False, qc=qc, qi=qi)


@pytest.mark.parametrize("scheme,kernel", [(99, "mrf_columns"), (3, "gfs_columns")])
def test_adapter_passes_state_condensate_and_applies_its_rates(monkeypatch, scheme, kernel):
    """pbl_driver.F:1358-1359 (MRF) / :1408 (GFS): QC3D=qc_curr, QI3D=qi_curr reach the kernel."""
    from gpuwrf.coupling import scan_adapters as sa
    grid, state = _column_state()
    nz, ncol = grid.nz, 2
    seen = {}
    rate = {"rqcblten": 1.0e-6, "rqiblten": -2.0e-6}

    def spy(u, v, t, qv, qc, *args, **kw):
        seen.update(qc=np.asarray(qc), qi=np.asarray(kw.get("qi")))
        z = jnp.zeros_like(u)
        out = {k: z for k in ("rublten", "rvblten", "rthblten", "rqvblten")}
        out.update({k: z + r for k, r in rate.items()})
        out.update(pblh=jnp.zeros(ncol, u.dtype), kpbl=jnp.zeros(ncol, jnp.int32), ust=jnp.full(ncol, 0.3, u.dtype))
        return out

    monkeypatch.setattr(sa, kernel, spy)
    out, tend = sa.PBL_SCAN_ADAPTERS[scheme](state, 18.0, grid, _with_sources=True)
    cols = lambda a: np.moveaxis(np.asarray(a), 0, -1).reshape(ncol, nz)  # noqa: E731
    np.testing.assert_array_equal(seen["qc"], cols(state.qc).astype(seen["qc"].dtype))
    np.testing.assert_array_equal(seen["qi"], cols(state.qi).astype(seen["qi"].dtype))
    for key, wrf in (("qc", "rqcblten"), ("qi", "rqiblten")):
        np.testing.assert_allclose(np.asarray(tend[key]), np.full((ncol, nz), rate[wrf]), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(getattr(out, key)),
                                   np.asarray(getattr(state, key)) + 18.0 * rate[wrf], rtol=1e-6, atol=1e-12)


def test_mrf_adapter_writes_back_mrf2d_ust():
    """module_bl_mrf.F:774-781: UST is INOUT, rebuilt as KARMAN*WSPD/(GZ1OZ0-PSIM)."""
    from gpuwrf.coupling import scan_adapters as sa
    grid, state = _column_state()
    out, _ = sa.mrf_pbl_adapter(state, 18.0, grid, _with_sources=True)
    assert not np.allclose(np.asarray(out.ustar), np.asarray(state.ustar))
    assert np.isfinite(np.asarray(out.ustar)).all() and (np.asarray(out.ustar) >= 0.1).all()  # land floor


@pytest.mark.parametrize("scheme,f_qi", [(99, True), (99, False)])
def test_sf1_condensate_rates_enter_rk_once(monkeypatch, scheme, f_qi):
    """Real adapter + kernel on the operational sf1 branch: RQC/RQIBLTEN become
    moist_tendf (coupled with c1h*mut+c2h, calculate_phy_tend) and qc/qi stay at
    the step-entry state (no second Euler increment)."""
    from gpuwrf.coupling import scan_adapters as sa
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    grid = _grid(nz=12, ny=3, nx=3)
    state = _base_state(grid)
    k = jnp.arange(12).reshape(12, 1, 1)
    state = state.replace(_cast=False,
                          qc=jnp.broadcast_to(jnp.where((k >= 2) & (k <= 4), 3.0e-4, 0.0), state.qc.shape).astype(state.qc.dtype),
                          qi=jnp.broadcast_to(jnp.where((k >= 5) & (k <= 7), 8.0e-5, 0.0), state.qi.shape).astype(state.qi.dtype))
    ns = _namelist(grid, dt_s=18., mp_physics=0, cu_physics=0, bl_pbl_physics=scheme, sf_sfclay_physics=1,
                   sf_surface_physics=0, use_noahmp=False, ra_lw_physics=0, ra_sw_physics=0, rad_rk_tendf=1)
    captured = {}
    real_adapter = om.PBL_SCAN_ADAPTERS[scheme]

    def spy(*args, **kwargs):
        result = real_adapter(*args, **kwargs)
        captured["rates"] = result[1]
        return result

    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, scheme, spy)
    monkeypatch.setattr(om, "_phys_tend_rk_wrf_enabled", lambda: True)
    monkeypatch.setattr(om, "_pbl_f_qi", lambda mp: f_qi)   # Registry table tested separately
    forcing = om._physics_step_forcing(initial_operational_carry(state), ns, 0., run_radiation=False,
                                       first_timestep=True)
    rates = captured["rates"]
    assert np.abs(np.asarray(rates["qc"])).max() > 0
    if not f_qi:   # FLAG_QI=.false.: no RQIBLTEN, no qi source in the RK
        assert "qi" not in rates and "qi" not in forcing.dry_tendencies.moist_tendf
        np.testing.assert_array_equal(np.asarray(forcing.state.qi), np.asarray(state.qi))
        return
    assert np.abs(np.asarray(rates["qi"])).max() > 0
    mass = (np.asarray(ns.metrics.c1h)[:, None, None] * np.asarray(state.mu_total)[None]
            + np.asarray(ns.metrics.c2h)[:, None, None])
    back = lambda r: np.moveaxis(np.asarray(r, np.float64).reshape(3, 3, 12), -1, 0)  # noqa: E731
    for key in ("qc", "qi"):
        np.testing.assert_array_equal(np.asarray(getattr(forcing.state, key)), np.asarray(getattr(state, key)))
        np.testing.assert_allclose(np.asarray(forcing.dry_tendencies.moist_tendf[key]), mass * back(rates[key]),
                                   rtol=8 * np.finfo(np.float32).eps, atol=1e-30)


# ----------------------------------------------------------------------------- GFS
def _gfs(fx, *, qi=True, qc=True):
    # GFS MONINP runs at kind_phys=real*8 in every WRF build; the port's WRF-legal
    # GFS branch (sf 3/91, legacy forcing) runs the kernel in fp64 too.
    from gpuwrf.physics.bl_gfs import gfs_columns
    c, s = fx["inputs"]["columns"], fx["inputs"]["scalars"]
    with jax.enable_x64(True):
        col = lambda n: jnp.asarray(np.asarray(c[n]), jnp.float64)  # noqa: E731
        one = lambda n: jnp.asarray(np.asarray(s[n]), jnp.float64)  # noqa: E731
        out = jax.jit(lambda: gfs_columns(
            col("U"), col("V"), col("T"), col("QV"), col("QC") if qc else jnp.zeros_like(col("QC")),
            col("P"), col("PI"), col("DZ"), col("Z"), psfc=one("PSFC"), ust=one("UST"), hfx=one("HFX"),
            qfx=one("QFX"), tsk=one("TSK"), gz1oz0=one("GZ1OZ0"), psim=one("PSIM"), psih=one("PSIH"),
            wspd=one("WSPD"), br=one("BR"), dt=54.0, qi=col("QI") if qi else None))()
        return {k: np.asarray(v, np.float64) for k, v in out.items()}


def test_gfs_oracle_census_has_active_condensate_and_countergradient_columns():
    fx = _fixture("gfs_moist_fixture.json")
    ref, inp = fx["outputs"]["r8_qi"], fx["inputs"]
    assert (np.asarray(inp["columns"]["QI"]).max(1) > 1.0e-6).sum() >= 4
    assert (np.abs(np.asarray(ref["RQCBLTEN"])) > 1.0e-10).sum() >= 50
    assert (np.abs(np.asarray(ref["RQIBLTEN"])) > 1.0e-12).sum() >= 20
    # BR<0 with HFX>0 and a WRF KPBL of 2: the KPBL<=1 PBLFLG edge (module_bl_gfs.F:935)
    br, hfx, kpbl = (np.asarray(inp["scalars"]["BR"]), np.asarray(inp["scalars"]["HFX"]), np.asarray(ref["KPBL"]))
    assert np.any((br < 0) & (hfx > 0) & (kpbl == 2))
    assert np.count_nonzero(fx["outputs"]["r8_noqi"]["RQIBLTEN"]) == 0       # P_QI absent: NTRAC=2


def test_gfs_double_matches_pristine_with_and_without_ice_tracer():
    fx = _fixture("gfs_moist_fixture.json")
    for label, qi in (("r8_qi", True), ("r8_noqi", False)):
        out = _gfs(fx, qi=qi)
        ref = fx["outputs"][label]
        assert _misses(out, ref) == {}, (label, _misses(out, ref))
        np.testing.assert_allclose(out["pblh"], ref["PBLH"], rtol=1.0e-9, atol=1.0e-6)
        np.testing.assert_array_equal(out["kpbl"], ref["KPBL"])
        np.testing.assert_allclose(out["ust"], ref["UST"], rtol=1.0e-12)   # INOUT UST=SQRT(STRESS)


def test_gfs_operational_kernel_inside_wrf_real_build_spread():
    fx = _fixture("gfs_moist_fixture.json")
    assert _misses(_gfs(fx), fx["outputs"]["r4_qi"], tol=_wrf_spread_tol(fx)) == {}


def test_gfs_pre_fix_caller_without_condensate_misses_pristine():
    fx = _fixture("gfs_moist_fixture.json")
    bad = _misses(_gfs(fx, qc=False, qi=False), fx["outputs"]["r8_qi"])
    assert {"RQCBLTEN", "RQIBLTEN"} <= set(bad), bad


@pytest.mark.parametrize("f_qi", [True, False])
def test_gfs_wrf_legal_branch_applies_condensate_rates_once(monkeypatch, f_qi):
    """bl=3 with sf=3 (the WRF-legal GFS pairing) takes the legacy State branch:
    the qc/qi increments of the real kernel's RQC/RQIBLTEN are in the physics
    state exactly once (carried post-RK by _PHYSICS_NON_DRY_INCREMENT_FIELDS)."""
    from gpuwrf.coupling import scan_adapters as sa
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    grid = _grid(nz=12, ny=3, nx=3)
    state = _base_state(grid)
    k = jnp.arange(12).reshape(12, 1, 1)
    state = state.replace(_cast=False,
                          qc=jnp.broadcast_to(jnp.where((k >= 2) & (k <= 4), 3.0e-4, 0.0), state.qc.shape).astype(state.qc.dtype),
                          qi=jnp.broadcast_to(jnp.where((k >= 5) & (k <= 7), 8.0e-5, 0.0), state.qi.shape).astype(state.qi.dtype))
    ns = _namelist(grid, dt_s=18., mp_physics=0, cu_physics=0, bl_pbl_physics=3, sf_sfclay_physics=3,
                   sf_surface_physics=0, use_noahmp=False, ra_lw_physics=0, ra_sw_physics=0)
    captured = {}
    real_kernel = sa.bl_gfs_wrf_columns  # sf3/bl3 operational path: WRF BL_GFS call (t-cfix-sfc F13)

    def spy(*args, **kwargs):
        out = real_kernel(*args, **kwargs)
        captured.update(out)
        return out

    monkeypatch.setattr(sa, "bl_gfs_wrf_columns", spy)
    monkeypatch.setattr(om, "_pbl_f_qi", lambda mp: f_qi)   # Registry table tested separately
    forcing = om._physics_step_forcing(initial_operational_carry(state), ns, 0., run_radiation=False,
                                       first_timestep=True)
    back = lambda r: np.moveaxis(np.asarray(r, np.float64).reshape(3, 3, 12), -1, 0)  # noqa: E731
    if not f_qi:   # P_QI absent: NTRAC=2, State qi untouched by the PBL
        np.testing.assert_array_equal(np.abs(captured["rqiblten"]), 0.0)
        np.testing.assert_array_equal(np.asarray(forcing.state.qi), np.asarray(state.qi))
    for key, wrf in (("qc", "rqcblten"), ("qi", "rqiblten"))[: 2 if f_qi else 1]:
        rate = back(captured[wrf])
        assert np.abs(rate).max() > 0, key
        np.testing.assert_allclose(np.asarray(getattr(forcing.state, key), np.float64),
                                   np.asarray(getattr(state, key), np.float64) + 18.0 * rate, rtol=1e-6, atol=1e-12)


# ----------------------------------------------------------------------------- MYJ
def _myj(fx, *, qi=True):
    from gpuwrf.physics.bl_myj import myj_columns
    c, s = fx["inputs"]["columns"], fx["inputs"]["scalars"]
    with jax.enable_x64(True):
        col = lambda n: jnp.asarray(np.asarray(c[n]), jnp.float64)  # noqa: E731
        one = lambda n: jnp.asarray(np.asarray(s[n]), jnp.float64)  # noqa: E731
        out = jax.jit(lambda: myj_columns(
            col("U"), col("V"), col("T"), col("TH"), col("QV"), col("QC"), col("P"), col("PINT"), col("PI"),
            col("DZ"), col("TKE"), tsk=one("TSK"), xland=one("XLAND"), ustar=one("UST"), akhs=one("AKHS"),
            akms=one("AKMS"), chklowq=one("CHKLOWQ"), elflx=one("LH"), thz0=one("THZ0"), qz0=one("QZ0"),
            uz0=one("UZ0"), vz0=one("VZ0"), qsfc=one("QSFC"), ct=one("CT"), snow=one("SNOW"), sice=one("SICE"),
            dt=54.0, stepbl=1, ht=one("HT"), qi=col("QI") if qi else None))()
        out = {k: np.asarray(v, np.float64) for k, v in out.items()}
    return {**{WRF_KEYS[k]: out[w] for k, w in TEN if w in out}, "pblh": out["PBLH"], "kpbl": out["KPBL"]}


def test_myj_oracle_census_has_active_ice_and_cloud_water_mixing():
    fx = _fixture("myj_moist_fixture.json")
    ref = fx["outputs"]["r8_qi"]
    assert (np.asarray(fx["inputs"]["columns"]["QI"]).max(1) > 1.0e-6).sum() >= 4
    assert (np.abs(np.asarray(ref["RQCBLTEN"])) > 1.0e-10).sum() >= 30
    assert (np.abs(np.asarray(ref["RQIBLTEN"])) > 1.0e-13).sum() >= 30
    assert np.asarray(fx["inputs"]["columns"]["TKE"]).max() > 0.1          # real TKE_PBL of the c03 run


def test_myj_double_matches_pristine_with_and_without_qci():
    fx = _fixture("myj_moist_fixture.json")
    for label, qi in (("r8_qi", True), ("r8_noqi", False)):
        out = _myj(fx, qi=qi)
        ref = fx["outputs"][label]
        assert _misses(out, ref) == {}, (label, _misses(out, ref))
        np.testing.assert_allclose(out["pblh"], ref["PBLH"], rtol=1.0e-9, atol=1.0e-6)
        np.testing.assert_array_equal(out["kpbl"], ref["KPBL"])


def test_myj_operational_kernel_inside_wrf_real_build_spread():
    fx = _fixture("myj_moist_fixture.json")
    assert _misses(_myj(fx), fx["outputs"]["r4_qi"], tol=_wrf_spread_tol(fx)) == {}


def test_myj_pre_fix_kernel_without_qci_misses_pristine():
    # flQI is always .TRUE. in the WRF caller (QI_CURR always passed); the old
    # kernel (QCI row zero, CWM=QCW) misses the ice mixing and its latent heat.
    fx = _fixture("myj_moist_fixture.json")
    bad = _misses(_myj(fx, qi=False), fx["outputs"]["r8_qi"])
    assert {"RQIBLTEN", "RTHBLTEN"} <= set(bad), bad


@pytest.mark.parametrize("f_qi", [True, False])
def test_myj_adapter_passes_raw_condensate_and_applies_rates(monkeypatch, f_qi):
    from gpuwrf.physics import myj_adapters as ma
    grid, state = _column_state()
    state = state.replace(_cast=False, qc=state.qc.at[0].set(-1.0e-9))   # raw, not clipped (pbl_driver.F:1448)
    nz, ncol = grid.nz, 2
    seen = {}
    rate = {"RQCBLTEN": 1.0e-6, "RQIBLTEN": -2.0e-6}
    real_kernel = ma.myj_columns

    def spy(*args, **kw):
        seen.update(qc=np.asarray(args[5]), qi=None if kw.get("qi") is None else np.asarray(kw["qi"]))
        out = dict(real_kernel(*args, **kw))
        out.update({k: jnp.zeros_like(out["RTHBLTEN"]) + r for k, r in rate.items()})
        return out

    monkeypatch.setattr(ma, "myj_columns", spy)
    out = ma.myj_pbl_adapter(state, 18.0, grid, f_qi=f_qi)
    cols = lambda a: np.moveaxis(np.asarray(a, np.float64), 0, -1).reshape(ncol, nz)  # noqa: E731
    np.testing.assert_array_equal(seen["qc"], cols(state.qc))
    if f_qi:
        np.testing.assert_array_equal(seen["qi"], cols(state.qi))
        np.testing.assert_allclose(np.asarray(out.qi), np.asarray(state.qi) + 18.0 * rate["RQIBLTEN"], rtol=1e-6, atol=1e-12)
    else:
        assert seen["qi"] is None
        assert out.qi is state.qi or np.array_equal(np.asarray(out.qi), np.asarray(state.qi))
    np.testing.assert_allclose(np.asarray(out.qc), np.asarray(state.qc) + 18.0 * rate["RQCBLTEN"], rtol=1e-6, atol=1e-12)
    _, rates = ma.myj_pbl_adapter(state, 18.0, grid, return_tendencies=True, f_qi=f_qi)
    assert {"rqcblten", "rqiblten"} <= set(rates)


@pytest.mark.parametrize("mp,expect", [(0, False), (1, False), (3, False), (2, True), (8, True), (13, True), (97, True)])
def test_pbl_f_qi_follows_wrf_registry_qi_slot(mp, expect):
    from gpuwrf.runtime.operational_mode import _pbl_f_qi
    assert _pbl_f_qi(mp) is expect
