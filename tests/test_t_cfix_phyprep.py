"""Legacy surface/PBL/cumulus callers hand WRF ``phy_prep`` staging to their kernels (CALLER_AUDIT F4/F5/F15/F16).

Independent reference (E98): ``proofs/t_cfix_phyprep/fixtures/phyprep_c03_oracle.npz`` = the UNTOUCHED pristine
WRF 4.7.1 ``SUBROUTINE phy_prep`` + ``conv_t_tendf_to_moist`` + calc_p_rho_phi ``al`` statements compiled as
REAL*4 and run on the whole real c03 CPU-WRF frame (Swiss 2023-01-15 12 UTC, cloud/ice columns);
``build_phyprep_oracle.py`` rebuilds it. Each test observes what the actual adapter hands its column kernel
(mocked boundary, never installed in a forecast program) and compares with the oracle on 12 real columns.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import BCMetadata, DycoreMetrics, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State
from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.physics import myj_adapters as myj

FIXTURE = Path(__file__).resolve().parents[1] / "proofs/t_cfix_phyprep/fixtures/phyprep_c03_oracle.npz"
DT = 60.0
F32 = np.float32


def _ulps(x, n):
    return n * np.spacing(np.abs(np.asarray(x, F32))).astype(np.float64)


@pytest.fixture(scope="module")
def case():
    manifest = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    f = dict(np.load(FIXTURE))
    ncol, nz = f["THM"].shape[:2]
    # 1 x ncol strip of the oracle column centres; every compared phy_prep field is column-local.
    strip = lambda name: np.transpose(f[name])[:, None, :].astype(F32)  # (nz|nz+1, 1, ncol)
    flat = lambda name: f[name].reshape(1, ncol).astype(F32)
    eta = jnp.linspace(1.0, 0.0, nz + 1)
    metrics = dataclasses.replace(
        DycoreMetrics.flat(ny=1, nx=ncol, nz=nz, eta_levels=eta, top_pressure_pa=float(f["P_TOP"])),
        **{k.lower(): jnp.asarray(f[k]) for k in ("C1H", "C2H", "C3H", "C4H", "C1F", "C2F", "C3F", "C4F",
                                                    "DNW", "FNM", "FNP")}, p_top=jnp.asarray(f["P_TOP"]))
    grid = GridSpec(Projection("lambert", 46.8, 8.2, 3000.0, 3000.0, ncol, 1),
                    TerrainProvenance("c03-oracle", "c03-oracle", (1, ncol), "m", "native", 0.0, True),
                    VerticalCoord("hybrid_eta", nz, float(f["P_TOP"]), eta), BCMetadata("ideal", (), 1, "linear", True),
                    eta, jnp.zeros((1, ncol)), metrics=metrics)
    u = np.concatenate([strip("U"), strip("U")[..., -1:]], axis=-1)
    v = np.concatenate([strip("V")] * 2, axis=1)
    state = State.zeros(grid).replace(
        theta=strip("THM") + F32(300.0), qv=strip("QVAPOR"), qc=strip("QCLOUD"), qr=strip("QRAIN"),
        qi=strip("QICE"), qs=strip("QSNOW"), qg=strip("QGRAUP"), p=strip("P") + strip("PB"),
        ph=strip("PH") + strip("PHB"), mu=flat("MUT"), u=u, v=v,
        qke=np.full((nz, 1, ncol), 0.2, F32), t_skin=np.full((1, ncol), 275.0, F32),
        xland=np.ones((1, ncol), F32), roughness_m=np.full((1, ncol), 0.1, F32),
        ustar=np.full((1, ncol), 0.3, F32), rhosfc=np.full((1, ncol), 1.1, F32), mavail=np.full((1, ncol), 0.5, F32))
    assert state.theta.dtype == jnp.float32
    wrf = {k[6:]: np.transpose(v) for k, v in f.items() if k.startswith("wrf32_")}  # (nz|nz+1, ncol)
    return SimpleNamespace(state=state, grid=grid, wrf=wrf, f=f, ncol=ncol, nz=nz,
                           rth=np.transpose(f["RTH"]), rqv=np.transpose(f["RQV"]))


def _assert_staging(case, got, fields):
    """got[name] -> (ncol, nlev) arrays the kernel received; compare with the REAL*4 pristine phy_prep."""
    wrf = case.wrf
    for name, (key, ulps) in fields.items():
        actual = np.asarray(got[name], np.float64)
        expected = np.asarray(wrf[key], np.float64).T[:, :actual.shape[1]]
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
        bound = _ulps(expected, ulps)
        if key == "dz8w":
            # dz8w = z_at_w(k+1)-z_at_w(k): XLA rewrites /g into *(1/g) under jit (E130), so each operand may
            # sit 1 ULP of z_at_w off the literal division; bound = 2 operand ULPs + the difference's rounding.
            zw = np.asarray(wrf["z_at_w"], np.float64).T
            bound = (_ulps(zw[:, 1:], 2) + _ulps(zw[:, :-1], 2))[:, :actual.shape[1]] + _ulps(expected, 1)
        np.testing.assert_array_less(np.abs(actual - expected), bound + 1e-300,
                                     err_msg=f"{name} must be WRF phy_prep {key}")


# Literal REAL4 chains: th_phy 1 division; t_phy one pow + one product; heights one division + difference;
# hydrostatic faces accumulate 44 levels of (1+qtot)*mass*dnw in REAL (XLA:CPU may contract FMAs, E107)
# -> 64 ULPs of the face pressure; total density one log/divide chain on a rounded geopotential total (E133).
STAGING = {"T": ("t_phy", 4), "theta": ("th_phy", 1), "p": ("p_hyd", 64), "dz": ("dz8w", 0)}


def _assert_rho(case, rho, what):
    """phy_prep RHO=(1+qv)/ALT, ALT from the literal calc_p_rho_phi statement (oracle).

    The State carries only the rounded total geopotential, so the shared REAL producer differences
    ph_total(k+1)-ph_total(k) where WRF differences ph and phb separately (E133 class, pre-existing, shared with
    KF/Tiedtke/ACM2/MYNN), and WRF's own LOG(pfd/pfu) is ill-conditioned in thin layers (1 ULP of the REAL ratio
    -> eps/|ln(pfd/pfu)| relative). Derived bound = (ULP(ph_total(k)) + ULP(ph_total(k+1)))/|dph|
    + 4 eps32/|ln(pfd/pfu)| + 8 eps32, relative.
    """
    f, eps = case.f, np.finfo(F32).eps
    phtot = np.asarray(case.state.ph, np.float32)[:, 0, :].T
    dph = np.abs(np.diff(phtot.astype(np.float64), axis=1))
    mut = f["MUT"].astype(np.float64)[:, None]
    pf = f["C3F"][None, :] * mut + f["C4F"][None, :] + float(f["P_TOP"])
    lnr = np.abs(np.log(pf[:, :-1] / pf[:, 1:]))
    rel = (np.spacing(np.abs(phtot[:, 1:])) + np.spacing(np.abs(phtot[:, :-1]))) / dph + 4 * eps / lnr + 8 * eps
    expected = (1.0 + np.asarray(case.state.qv, np.float64)[:, 0, :].T) / case.wrf["alt"].T
    np.testing.assert_array_less(np.abs(np.asarray(rho, np.float64) - expected), rel * np.abs(expected),
                                 err_msg=f"{what}: RHO must be phy_prep total density (1+qv)/alt")


def test_wrf_phy_prep_helper_matches_pristine(case):
    from gpuwrf.coupling.wrf_phy_prep import wrf_phy_prep
    prep = jax.jit(lambda s: wrf_phy_prep(s, case.grid.metrics))(case.state)
    col = lambda a: np.asarray(a)[:, 0, :].T
    _assert_staging(case, {k: col(prep[k]) for k in ("th_phy", "t_phy", "pi_phy", "p_hyd", "p_hyd_w", "dz8w", "z", "z_at_w")},
                    {"th_phy": ("th_phy", 1), "t_phy": ("t_phy", 4), "pi_phy": ("pi_phy", 2), "p_hyd": ("p_hyd", 64),
                     "p_hyd_w": ("p_hyd_w", 64), "dz8w": ("dz8w", 0), "z": ("z", 4), "z_at_w": ("z_at_w", 2)})
    _assert_rho(case, col(prep["rho"]), "wrf_phy_prep")
    np.testing.assert_array_equal(np.asarray(prep["psfc"])[0], np.asarray(prep["p_hyd_w"])[0, 0])


def _observe_myj(case, monkeypatch, rates=True):
    seen = {}
    real_sfc = myj.myjsfc_columns

    def sfc(*args, **kw):
        seen["sfc_T"], seen["sfc_theta"], seen["sfc_p"], seen["sfc_dz"] = args[2], args[3], args[6], args[7]
        seen["sfc_psfc"] = kw["psfc"]
        return real_sfc(*args, **kw)

    def pbl(u, v, T, theta, qv, qc, p, p_int, pii, dz, tke, **kw):
        seen.update(T=T, theta=theta, p=p, p_int=p_int, dz=dz)
        n, nz = T.shape
        zero = jnp.zeros((n, nz), jnp.float64)
        out = dict(RUBLTEN=zero, RVBLTEN=zero, TKE_MYJ=jnp.full((n, nz), 0.1), PBLH=jnp.full((n,), 500.0),
                   KPBL=jnp.full((n,), 3, jnp.int32), EXCH_H=zero, RQCBLTEN=zero, RQIBLTEN=zero,
                   # MYJPBL INOUT surface memory (t-cfix-sfc F12): pass the caller's values through.
                   QSFC=kw["qsfc"], THZ0=kw["thz0"], QZ0=kw["qz0"])
        out["RTHBLTEN"] = jnp.asarray(case.rth.T if rates else zero)
        out["RQVBLTEN"] = jnp.asarray(case.rqv.T if rates else zero)
        return out

    monkeypatch.setattr(myj, "myjsfc_columns", sfc)
    monkeypatch.setattr(myj, "myj_columns", pbl)
    return seen


def test_myj_pbl_and_janjic_receive_phy_prep_staging(case, monkeypatch):
    seen = _observe_myj(case, monkeypatch)
    myj.myj_pbl_adapter(case.state, DT, case.grid)
    _assert_staging(case, seen, STAGING)
    _assert_staging(case, {"T": seen["sfc_T"], "theta": seen["sfc_theta"], "p": seen["sfc_p"], "dz": seen["sfc_dz"]}, STAGING)
    # MYJPBL/MYJSFC PINT and PSFC=PINT(LOWLYR) are p_hyd_w (first_rk_step_part1.F:1130, myjpbl.F:476).
    _assert_staging(case, {"p_int": seen["p_int"]}, {"p_int": ("p_hyd_w", 64)})
    np.testing.assert_array_less(np.abs(np.asarray(seen["sfc_psfc"]) - case.wrf["p_hyd_w"][0]),
                                 _ulps(case.wrf["p_hyd_w"][0], 64))


def _expected_theta(case, state, rate=None):
    r = case.wrf["theta_m_rate"] if rate is None else rate
    return (np.asarray(state.theta, np.float64)[:, 0, :] + DT * r).astype(F32)


def _assert_theta(case, updated, expected, what):
    got = np.asarray(updated.theta)[:, 0, :]
    np.testing.assert_array_less(np.abs(got.astype(np.float64) - expected), _ulps(expected, 2) + 1e-300,
                                 err_msg=f"{what}: dry rate must enter theta_m via conv_t_tendf_to_moist")


def test_myj_pbl_converts_dry_rthblten_to_theta_m(case, monkeypatch):
    _observe_myj(case, monkeypatch)
    updated = myj.myj_pbl_adapter(case.state, DT, case.grid)
    _assert_theta(case, updated, _expected_theta(case, case.state), "MYJ")


def test_gfs_legacy_pbl_receives_phy_prep_and_converts(case, monkeypatch):
    seen = {}
    nz, ncol = case.nz, case.ncol
    flat = jnp.zeros((ncol,), jnp.float64)
    monkeypatch.setattr(sa, "surface_layer_with_diagnostics", lambda *a, **k: SimpleNamespace(
        hfx=jnp.zeros((1, ncol)), lh=jnp.zeros((1, ncol)), psim=jnp.zeros((1, ncol)), psih=jnp.zeros((1, ncol)),
        br=jnp.zeros((1, ncol)), u10=jnp.zeros((1, ncol)), v10=jnp.zeros((1, ncol))))

    def gfs(u, v, T, qv, qc, p, pii, dz, z, **kw):
        seen.update(T=T, p=p, dz=dz, z=z, psfc=kw["psfc"])
        zero = jnp.zeros((ncol, nz), jnp.float64)
        return dict(rublten=zero, rvblten=zero, rthblten=jnp.asarray(case.rth.T), rqvblten=jnp.asarray(case.rqv.T),
                    rqcblten=zero, rqiblten=zero, pblh=flat)

    monkeypatch.setattr(sa, "gfs_columns", gfs)
    updated = sa.gfs_pbl_adapter(case.state, DT, case.grid)
    _assert_staging(case, seen, {"T": ("t_phy", 4), "p": ("p_hyd", 64), "dz": ("dz8w", 0), "z": ("z", 4)})
    np.testing.assert_array_less(np.abs(np.asarray(seen["psfc"]) - case.wrf["p_hyd_w"][0]), _ulps(case.wrf["p_hyd_w"][0], 64))
    _assert_theta(case, updated, _expected_theta(case, case.state), "GFS PBL")


def _vmap_observer(store, *names):
    """Host observation of vmapped column-kernel inputs (test-only, mocked boundary)."""

    def record(*values):
        for name, value in zip(names, values):
            store.setdefault(name, []).append(np.atleast_2d(np.asarray(value)))

    return lambda *values: jax.debug.callback(record, *values)


def _stack(store):
    return {k: np.concatenate(v, axis=0) for k, v in store.items()}


def _column_rates(case):
    """Per-column fixture rates selected inside a vmapped mock by the column's own lowest-level T_PHY."""
    t0 = jnp.asarray(case.wrf["t_phy"][0], jnp.float32)
    rth, rqv = jnp.asarray(case.rth.T), jnp.asarray(case.rqv.T)

    def pick(T0):
        i = jnp.argmin(jnp.abs(jnp.asarray(T0, jnp.float32)[0] - t0))
        return rth[i], rqv[i]

    return pick


def _assert_cumulus(case, got, with_p8w=False):
    _assert_staging(case, got, {"T": ("t_phy", 4), "p": ("p_hyd", 64), "dz": ("dz8w", 0)})
    if with_p8w:
        _assert_staging(case, {"p8w": got["p8w"]}, {"p8w": ("p_hyd_w", 64)})
    if "rho" in got:
        _assert_rho(case, got["rho"], "cumulus")


def test_bmj_receives_phy_prep_and_converts(case, monkeypatch):
    store = {}
    observe = _vmap_observer(store, "T", "p", "dz", "rho", "psfc")
    pick = _column_rates(case)

    def column(T0, QV0, P0, DZQ, RHOE, PII, dt, **kw):
        observe(T0, P0, DZQ, RHOE, kw["psfc"])
        rth, rqv = pick(T0)
        return SimpleNamespace(tendency=SimpleNamespace(state_tendencies={"theta": rth, "qv": rqv},
                               accumulator_increments={"rainc_acc": jnp.zeros(())}),
                               carry=SimpleNamespace(cumulus={"cldefi": kw["cldefi"]}))

    monkeypatch.setattr(sa, "step_bmj_column", column)
    updated, _ = sa.bmj_adapter(case.state, DT, jnp.ones((1, case.ncol)), grid=case.grid)
    got = _stack(store)
    _assert_cumulus(case, got)
    # bmj.F:193 PSFC=PINT(LOWLYR)=p_hyd_w(1)
    np.testing.assert_array_less(np.abs(got["psfc"].reshape(-1) - case.wrf["p_hyd_w"][0]), _ulps(case.wrf["p_hyd_w"][0], 64))
    _assert_theta(case, updated, _expected_theta(case, case.state), "BMJ")


def test_ntiedtke_receives_phy_prep_and_converts(case, monkeypatch):
    store = {}
    observe = _vmap_observer(store, "T", "p", "p8w", "dz", "rho")
    pick = _column_rates(case)

    def column(T0, QV0, QC0, QI0, P0, P8W0, DZ0, RHO0, *args, **kw):
        observe(T0, P0, P8W0, DZ0, RHO0)
        rth, rqv = pick(T0)
        zero = jnp.zeros_like(T0)
        return dict(RTHCUTEN=rth, RQVCUTEN=rqv, RQCCUTEN=zero, RQICUTEN=zero, RUCUTEN=zero, RVCUTEN=zero,
                    RAINCV=jnp.zeros(()))

    monkeypatch.setattr(sa, "ntiedtke_column_jax", column)
    updated = sa.ntiedtke_adapter(case.state, DT, case.grid)
    _assert_cumulus(case, _stack(store), with_p8w=True)
    _assert_theta(case, updated, _expected_theta(case, case.state), "NTiedtke")


def test_tiedtke_receives_hydrostatic_pressure_and_wrf_dz(case, monkeypatch):
    store = {}
    observe = _vmap_observer(store, "T", "p", "p8w", "dz")

    def column(T0, QV0, QC0, QI0, P0, P8W0, DZ0, *args, **kw):
        observe(T0, P0, P8W0, DZ0)
        zero = jnp.zeros_like(T0)
        return {k: zero for k in ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQRCUTEN", "RQICUTEN", "RQSCUTEN",
                                  "RUCUTEN", "RVCUTEN")} | {"RAINCV": jnp.zeros(()), "ICUM1": jnp.bool_(True)}

    monkeypatch.setattr(sa, "tiedtke_column_jax", column)
    sa.tiedtke_adapter(case.state, DT, case.grid)
    _assert_cumulus(case, _stack(store), with_p8w=True)


def test_gf_receives_phy_prep_and_converts(case, monkeypatch):
    seen = {}
    nz, ncol = case.nz, case.ncol

    def gf(T_c, qv_c, p_c, pii_c, dz_c, rho_c, u_c, v_c, w_c, f1, f2, kx, dt_b, dx_b, hfx_b, qfx_b, kpbl_c,
           xland_b, ht_b, **kw):
        seen.update(T=T_c[:, 1:], p=p_c[:, 1:], dz=dz_c[:, 1:], rho=rho_c[:, 1:], ht=ht_b)
        pad = lambda a: jnp.concatenate([jnp.zeros((ncol, 1), jnp.float64), jnp.asarray(a, jnp.float64)], axis=1)
        zero = pad(jnp.zeros((ncol, nz)))
        return dict(RTHCUTEN=pad(case.rth.T), RQVCUTEN=pad(case.rqv.T), RQCCUTEN=zero, RQICUTEN=zero,
                    RAINCV=jnp.zeros((ncol,)))

    monkeypatch.setattr(sa, "gfdrv_batched", gf)
    # 1 x ncol strip: WRF's GF lateral band (ids+4..ide-5 / jds+4..jde-5, gf_wrfdrv:264-277, t-cfix-cu F18)
    # leaves no interior column unless periodic; this test checks the phy_prep staging of every column.
    updated = sa.gf_adapter(case.state, DT, case.grid, periodic_x=True, periodic_y=True)
    _assert_cumulus(case, seen)
    # GFDRV HT=grid%ht; real.exe/phy_prep heights use g=9.81 (z_at_w(kts)).
    np.testing.assert_array_less(np.abs(np.asarray(seen["ht"]) - case.wrf["z_at_w"][0]), _ulps(case.wrf["z_at_w"][0], 2))
    _assert_theta(case, updated, _expected_theta(case, case.state), "GF")


# ---------------------------------------------------------------------------------------------------------------
# Caller path: the actual operational dispatch (_physics_step_forcing) twice, the second call on the State the
# first produced (E85). Expectations are a NumPy REAL4 transcription of the pristine statements (phy_prep :4834,
# :4851-4853, :4935-4968; conv_t_tendf_to_moist :6699-6703), independent of the port helpers.

R32 = F32(461.6) / F32(287.0)


def _literal_prep(state, metrics):
    th = np.asarray(state.theta, F32) / (F32(1.0) + R32 * np.asarray(state.qv, F32))
    pi = (np.asarray(state.p, F32) / F32(100000.0)) ** (F32(287.0) / F32(1004.5))
    mut = np.asarray(state.mu_total, F32)
    c1h, c2h, dnw = (np.asarray(getattr(metrics, n), F32) for n in ("c1h", "c2h", "dnw"))
    qtot = sum(np.asarray(getattr(state, n), F32) for n in ("qv", "qc", "qr", "qi", "qs", "qg"))
    nz = th.shape[0]
    faces = [np.full(mut.shape, np.asarray(metrics.p_top, F32), F32)]
    for k in range(nz - 1, -1, -1):
        faces.append(faces[-1] - (F32(1.0) + qtot[k]) * (c1h[k] * mut + c2h[k]) * dnw[k])
    pw = np.stack(faces[::-1])
    return dict(th=th, t=th * pi, p_hyd=F32(0.5) * (pw[:-1] + pw[1:]), p_hyd_w=pw)


def _literal_theta(entry, rth, rqv, dt):
    th, q = np.asarray(entry.theta, F32), np.asarray(entry.qv, F32)
    factor = F32(1.0) + R32 * q
    return th + F32(dt) * (factor * rth + R32 * th / factor * rqv)


SCHEMES_2STEP = {
    "myj": dict(bl_pbl_physics=2, sf_sfclay_physics=2, cu_physics=0),
    "gfs": dict(bl_pbl_physics=3, sf_sfclay_physics=3, cu_physics=0),
    "bmj": dict(bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=2),
    "gf": dict(bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=3),
    "ntiedtke": dict(bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=16, use_flux_advection=True, moist_adv_opt=2),
}


def _install_mock(name, case, monkeypatch, calls):
    """Mock the scheme's column boundary: record (T, P) inputs, return the fixture's dry rates."""
    nz, ncol = case.nz, case.ncol
    rth, rqv = jnp.asarray(case.rth.T, jnp.float32), jnp.asarray(case.rqv.T, jnp.float32)
    t0 = jnp.asarray(case.wrf["t_phy"][0], jnp.float32)

    def record(T, p):
        calls.append((np.atleast_2d(np.asarray(T)), np.atleast_2d(np.asarray(p))))

    def pick(T0):
        i = jnp.argmin(jnp.abs(jnp.asarray(T0, jnp.float32)[0] - t0))
        return rth[i], rqv[i]

    if name == "myj":
        def kernel(u, v, T, theta, qv, qc, p, p_int, pii, dz, tke, **kw):
            jax.debug.callback(record, T, p)
            zero = jnp.zeros((ncol, nz), jnp.float64)
            return dict(RUBLTEN=zero, RVBLTEN=zero, RTHBLTEN=rth, RQVBLTEN=rqv, TKE_MYJ=jnp.full((ncol, nz), .1),
                        PBLH=jnp.full((ncol,), 500.), KPBL=jnp.full((ncol,), 3, jnp.int32), EXCH_H=zero, RQCBLTEN=zero, RQIBLTEN=zero,
                        QSFC=kw["qsfc"], THZ0=kw["thz0"], QZ0=kw["qz0"])  # MYJPBL INOUT memory (t-cfix-sfc F12)
        monkeypatch.setattr(myj, "myj_columns", kernel)
    elif name == "gfs":
        def kernel(u, v, T, qv, qc, p, pii, dz, z, **kw):
            jax.debug.callback(record, T, p)
            zero = jnp.zeros((ncol, nz), jnp.float64)
            return dict(rublten=zero, rvblten=zero, rthblten=rth, rqvblten=rqv, rqcblten=zero, rqiblten=zero, pblh=jnp.zeros((ncol,)),
                        ust=kw["ust"])  # BL_GFS UST INOUT (module_bl_gfs.F:464; t-cfix-sfc F13)
        monkeypatch.setattr(sa, "gfs_columns", kernel)
        monkeypatch.setattr(sa, "bl_gfs_wrf_columns", kernel)  # sf3/bl3 WRF BL_GFS call (t-cfix-sfc F13)
    elif name == "bmj":
        def kernel(T0, QV0, P0, DZQ, RHOE, PII, dt, **kw):
            jax.debug.callback(record, T0, P0)
            a, b = pick(T0)
            return SimpleNamespace(tendency=SimpleNamespace(state_tendencies={"theta": a, "qv": b},
                                   accumulator_increments={"rainc_acc": jnp.zeros(())}),
                                   carry=SimpleNamespace(cumulus={"cldefi": kw["cldefi"]}))
        monkeypatch.setattr(sa, "step_bmj_column", kernel)
    elif name == "gf":
        def kernel(T_c, qv_c, p_c, *args, **kw):
            jax.debug.callback(record, T_c[:, 1:], p_c[:, 1:])
            pad = lambda a: jnp.concatenate([jnp.zeros((ncol, 1), a.dtype), a], axis=1)
            zero = pad(jnp.zeros((ncol, nz), jnp.float32))
            return dict(RTHCUTEN=pad(rth), RQVCUTEN=pad(rqv), RQCCUTEN=zero, RQICUTEN=zero, RAINCV=jnp.zeros((ncol,)))
        monkeypatch.setattr(sa, "gfdrv_batched", kernel)
    else:
        def kernel(T0, QV0, QC0, QI0, P0, *args, **kw):
            jax.debug.callback(record, T0, P0)
            a, b = pick(T0)
            zero = jnp.zeros_like(T0)
            return dict(RTHCUTEN=a, RQVCUTEN=b, RQCCUTEN=zero, RQICUTEN=zero, RUCUTEN=zero, RVCUTEN=zero,
                        RAINCV=jnp.zeros(()))
        monkeypatch.setattr(sa, "ntiedtke_column_jax", kernel)


@pytest.mark.parametrize("scheme", SCHEMES_2STEP)
def test_operational_two_step_caller_path(case, monkeypatch, scheme):
    from gpuwrf.runtime import operational_mode as om
    from test_v013_operational_smoke import TIME_UTC, _cpu_tendencies
    dt = 18.0
    namelist = dataclasses.replace(
        om.OperationalNamelist.from_grid(case.grid, dt_s=dt, tendencies=_cpu_tendencies(case.grid)),
        time_utc=TIME_UTC, run_physics=True, radiation_cadence_steps=1, mp_physics=0, ra_sw_physics=0,
        ra_lw_physics=0, sf_surface_physics=0, use_noahmp=False, **SCHEMES_2STEP[scheme])
    om._resolve_operational_suite(namelist)
    calls = []
    _install_mock(scheme, case, monkeypatch, calls)
    carry = om._initial_carry_for_run(case.state, namelist)
    for step in range(2):
        entry, start = carry.state, len(calls)
        forcing = om._physics_step_forcing(carry, namelist, step * dt, run_radiation=False)
        jax.block_until_ready(forcing.state.theta)
        new = calls[start:]
        # One batched call, or one call per column under the adapter's vmap.
        assert len(new) in (1, case.ncol), f"{scheme}: kernel must run once per step ({len(new)})"
        T, p = (np.concatenate([c[i] for c in new], axis=0) for i in (0, 1))
        lit = _literal_prep(entry, case.grid.metrics)
        col = lambda a: np.asarray(a)[:, 0, :].T
        np.testing.assert_array_less(np.abs(T - col(lit["t"])), _ulps(col(lit["t"]), 4),
                                     err_msg=f"{scheme} step{step + 1}: dry T_PHY")
        np.testing.assert_array_less(np.abs(p - col(lit["p_hyd"])), _ulps(col(lit["p_hyd"]), 64),
                                     err_msg=f"{scheme} step{step + 1}: hydrostatic P_PHY=p_hyd")
        expected = _literal_theta(entry, case.rth[:, None, :], case.rqv[:, None, :], dt).astype(np.float64)
        np.testing.assert_array_less(np.abs(np.asarray(forcing.state.theta, np.float64) - expected),
                                     _ulps(expected, 2) + 1e-300, err_msg=f"{scheme} step{step + 1}: theta_m form")
        carry = forcing.carry.replace(state=forcing.state)


# ---------------------------------------------------------------------------------------------------------------
# Series 2 (F15, class B): every PBL/cumulus driver gets P_PHY=grid%p_hyd and P8W=grid%p_hyd_w
# (first_rk_step_part1.F:633/637 surface, :1130/1132 PBL, :1380/1394/1400 cumulus); PI_PHY stays non-hydrostatic.

class _Captured(Exception):
    pass


def _assert_hydrostatic(case, p_mid, p_faces, what):
    """p_mid (ncol, nz) / p_faces (ncol, nz|nz+1) must be the pristine REAL4 p_hyd / p_hyd_w."""
    _assert_staging(case, {"p": p_mid}, {"p": ("p_hyd", 64)})
    _assert_staging(case, {"p8w": p_faces}, {"p8w": ("p_hyd_w", 64)})


def test_qnse_geometry_uses_hydrostatic_pressure(case):
    from gpuwrf.coupling.qnse_adapter import _geometry
    profiles, *_ = _geometry(case.state, case.grid)
    _assert_hydrostatic(case, np.asarray(profiles["p_mid"]).T, np.asarray(profiles["p_int"]).T, "QNSE")


def test_temf_pbl_inputs_use_hydrostatic_pressure(case):
    from gpuwrf.coupling.temf import phy_inputs
    f = phy_inputs(case.state, case.grid)
    _assert_hydrostatic(case, np.asarray(f["pressure"]), np.asarray(f["p8w"]), "TEMF")
    np.testing.assert_array_less(np.abs(np.asarray(f["psfc"]) - case.wrf["p_hyd_w"][0]), _ulps(case.wrf["p_hyd_w"][0], 64))


@pytest.mark.parametrize("cu", (5, 93))
def test_grell_receives_hydrostatic_pressure(case, monkeypatch, cu):
    from gpuwrf.physics import _grell_cup_jax as grell
    seen = {}

    def capture(**kw):
        seen.update(p=np.asarray(kw["p"])[:, 0, :].T, p8w=np.asarray(kw["p8w"])[:, 0, :].T)
        raise _Captured

    monkeypatch.setattr(grell, "g3drv_tile" if cu == 5 else "grelldrv_tile", capture)
    with pytest.raises(_Captured):
        sa.grell_cumulus_adapter(case.state, DT, case.grid, cu_physics=cu)
    # g3.F:342 / gd.F:233 PSUR = P8W(1)*.01 = p_hyd_w(1).
    _assert_hydrostatic(case, seen["p"], seen["p8w"], f"cu={cu}")


class _VmapCapture:
    """Proxy for the module-global ``jax`` that captures the operand dict handed to ``jax.vmap(column)``."""

    def __init__(self, seen):
        self._seen = seen

    def __getattr__(self, name):
        return getattr(jax, name)

    def vmap(self, fun, *a, **k):
        def run(common):
            self._seen.update({key: np.asarray(v) for key, v in common.items() if key in ("pressure", "p8w")})
            raise _Captured
        return run


@pytest.mark.parametrize("scheme", (16, 17))
def test_epsilon_pbl_receives_hydrostatic_pressure(case, monkeypatch, scheme):
    from gpuwrf.coupling import epsilon_pbl
    seen = {}
    monkeypatch.setattr(epsilon_pbl, "jax", _VmapCapture(seen))
    from gpuwrf.contracts.precision import EPSILON_PBL_DIAGNOSTICS
    from gpuwrf.nesting.scalar_species import epsilon_pbl_scalars
    zeros3 = jnp.zeros_like(case.state.qv)
    # Every carry leaf the E78 schema requires for this scheme (io/pbl_epsilon_schema.py), seeded as REAL zeros.
    leaves = {name: zeros3 for name in ("epsilon_pbl", *EPSILON_PBL_DIAGNOSTICS, *epsilon_pbl_scalars(scheme),
                                        "pek_adv", "pep_adv", "tke_adv", "diss_adv", "tpe_adv", "tpe_pbl", "pr_pbl")}
    state = case.state.replace(_cast=False, **leaves, qke=jnp.full_like(zeros3, 0.2), epsilon_pbl_step=jnp.int32(3))
    flat = jnp.zeros((1, case.ncol), jnp.float32)
    forcing = SimpleNamespace(rmol=flat, wspd=flat + 1.0, br=flat, fm=flat + 5.0, fhh=flat + 5.0)
    with pytest.raises(_Captured):
        epsilon_pbl.epsilon_pbl_adapter(state, DT, case.grid, scheme=scheme, step=jnp.int32(4),
                                        rthraten=zeros3, surface_forcing=forcing)
    if scheme == 16:  # EEPS P3D=p_phy (eepsilon.F:277)
        _assert_staging(case, {"p": seen["pressure"]}, {"p": ("p_hyd", 64)})
    else:  # KEPS P8W=p8w (keps.F:223 radiative flux layer weights)
        _assert_staging(case, {"p8w": seen["p8w"]}, {"p8w": ("p_hyd_w", 64)})


# ---------------------------------------------------------------------------------------------------------------
# Series 3 (F4/F15, API-only LSMs): slab / Noah-classic / PX get t_phy, th_phy, p_hyd(1), p_hyd_w(1), dz8w (g 9.81).

class _Bundle:
    """Minimal land/static stand-in: named fields, every other attribute a zero surface field."""

    def __init__(self, shape, **known):
        self._shape = shape
        self.__dict__.update(known)

    def __getattr__(self, name):
        return jnp.zeros(self._shape, jnp.float64)


def _lowest(case, key):
    return np.asarray(case.wrf[key], np.float64)[0]


def test_noahclassic_forcing_is_wrf_phy_prep(case):
    from gpuwrf.coupling.noahclassic_surface_hook import assemble_noahclassic_forcing
    shape = (1, case.ncol)
    land = _Bundle(shape, t1=jnp.full(shape, 275.0))
    static = _Bundle(shape, params=_Bundle(shape, alb=jnp.full(shape, .2), embrd=jnp.full(shape, .95)),
                     solnet_albedo=None, lwdn_emissivity=None)
    from gpuwrf.runtime.noahmp_precipitation import seed_precipitation
    f = assemble_noahclassic_forcing(case.state, land, static, None,
                                     precipitation=seed_precipitation(case.state, with_sr=True), grid=case.grid)
    t1, ph, pw = _lowest(case, "t_phy"), _lowest(case, "p_hyd"), _lowest(case, "p_hyd_w")
    np.testing.assert_array_less(np.abs(np.asarray(f.sfctmp)[0] - t1), _ulps(t1, 4), err_msg="SFCTMP=T3D(1)=t_phy")
    np.testing.assert_array_less(np.abs(np.asarray(f.sfcprs)[0] - ph), _ulps(ph, 64), err_msg="SFCPRS=p_hyd(1)")
    # noahdrv :822-825 TH2 = SFCTMP*(1E5/SFCPRS)**CAPA / (1E5/PSFC)**CAPA, PSFC=p_hyd_w(1).
    capa = 287.0 / 1004.5
    th2 = t1 * (1.0e5 / ph) ** capa / (1.0e5 / pw) ** capa
    np.testing.assert_allclose(np.asarray(f.th2)[0], th2, rtol=8 * np.finfo(F32).eps, err_msg="Noah TH2")
    dz = np.asarray(case.wrf["dz8w"], np.float64)[0]
    np.testing.assert_allclose(np.asarray(f.zlvl)[0], 0.5 * dz, rtol=1e-5, err_msg="ZLVL=0.5*DZ8W(1), g=9.81")


def test_slab_receives_wrf_phy_prep(case, monkeypatch):
    from gpuwrf.coupling import slab_surface_hook as hook
    seen = {}

    def capture(tb, qv, p_bottom, flhc, flqc, psfc, *a, **k):
        seen.update(t=np.asarray(tb), p=np.asarray(p_bottom), psfc=np.asarray(psfc))
        raise _Captured

    monkeypatch.setattr(hook, "slab_columns", capture)
    shape = (1, case.ncol)
    land = _Bundle(shape, tslb=jnp.full(shape + (hook.NUM_SOIL_LAYERS,), 275.0))
    static = _Bundle(shape, dzs=jnp.full((hook.NUM_SOIL_LAYERS,), .1), zs=jnp.linspace(.05, .9, hook.NUM_SOIL_LAYERS),
                     ifsnow=0)
    with pytest.raises(_Captured):
        hook.slab_surface_step(case.state, land, static, DT, grid=case.grid)
    for key, name, n in (("t", "t_phy", 4), ("p", "p_hyd", 64)):
        ref = _lowest(case, name)
        np.testing.assert_array_less(np.abs(seen[key] - ref), _ulps(ref, n), err_msg=f"SLAB {name}")
    ref = _lowest(case, "p_hyd_w")
    np.testing.assert_array_less(np.abs(seen["psfc"] - ref), _ulps(ref, 64), err_msg="SLAB PSFC=p_hyd_w(1)")


def test_pleim_xiu_receives_wrf_phy_prep(case, monkeypatch):
    from gpuwrf.coupling import pleim_xiu_surface_hook as hook
    seen = {}

    def capture(soldn, gsw, lwdn, z1, rmol, ust, psurf_cb, dens1, qv1, ta1, theta1, *a, **k):
        seen.update(z1=np.asarray(z1), psfc=1000.0 * np.asarray(psurf_cb), t=np.asarray(ta1), th=np.asarray(theta1))
        raise _Captured

    monkeypatch.setattr(hook, "pxlsm_columns", capture)
    shape = (1, case.ncol)
    static = _Bundle(shape, params=_Bundle(shape), ifsnow=0)
    with pytest.raises(_Captured):
        from gpuwrf.runtime.noahmp_precipitation import seed_precipitation
        hook.pleim_xiu_surface_step(case.state, _Bundle(shape), static, DT,
                                   precipitation=seed_precipitation(case.state), grid=case.grid)
    for key, name, n in (("t", "t_phy", 4), ("th", "th_phy", 1)):
        ref = _lowest(case, name)
        np.testing.assert_array_less(np.abs(seen[key] - ref), _ulps(ref, n), err_msg=f"PX {name}")
    ref = _lowest(case, "p_hyd_w")
    np.testing.assert_array_less(np.abs(seen["psfc"] - ref), _ulps(ref, 64) + 1e-9 * ref, err_msg="PX PSFC=p_hyd_w(1)")
    np.testing.assert_allclose(seen["z1"], 0.5 * np.asarray(case.wrf["dz8w"], np.float64)[0], rtol=1e-5)
