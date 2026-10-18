"""state_io: incremental hydrostatic rebalance, boundary taper, analysis wrfinput (lane a1-io).

CPU/NumPy only. Synthetic geometry is built from the WRF base-state formulas (start_em :600-640);
real-file tests read the CPU-WRF WN3 0502 run read-only and skip when it is absent. Every check
is a function that a deletion/sign mutant must FAIL (E39); mutant tests assert AssertionError.
The rebalance oracle (``hand_hydrostatic``) is an independent dry-mass hypsometric integration
(dφ = −α_d dp_d, ideal-gas α_d, moist pressure summed from the top), not the real.exe code path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import state_io
from gpuwrf.assimilation.letkf.state_io import (
    FRAME_VARIABLES,
    boundary_taper,
    diagnose_al,
    balance_state,
    hydrostatic_residual,
    read_frame,
    read_geometry,
    rebalance_increment,
    write_analysis_wrfinput,
)
from gpuwrf.init.real_init.types import CP, CVPM, G, P1000MB, R_D, RVOVRD, T0
from gpuwrf.io.netcdf_lock import Dataset

RUN = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260502_18z_a1/run/run")
T_A = "2026-05-04_12:00:00"
WRFOUT = RUN / f"wrfout_d01_{T_A}"
TEMPLATE = RUN / "wrfinput_d01"
real = pytest.mark.usefixtures("_alisios_guard")
RUN_C = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260503_18z_a1/run/run")


def _alisios_reads():
    """Every ALISIOS file this module opens (ALISIOS rule: guarded_session only)."""
    from gpuwrf.assimilation.letkf.ensemble import run_days_from_name
    reads = {}
    for run in (RUN, RUN_C):
        if not run.is_dir():
            continue
        days = run_days_from_name(run)
        for path, d in ((run / "wrfinput_d01", days[:1]), (run / "wrfinput_d02", days[:1]),
                        (run / "namelist.input", days), (run / f"wrfout_d01_{T_A}", ["20260504"]),
                        (run / f"wrfout_d02_{T_A}", ["20260504"])):
            if path.is_file():
                reads[path.resolve()] = d
    return reads


from _a1io_guard import module_guard  # noqa: E402

_alisios_guard = module_guard(_alisios_reads, autouse=False)


# --------------------------------------------------------------------------- synthetic geometry

def synthetic_geo(nz=20, ny=12, nx=14, p_top=5000.0, dtype=np.float32):
    znw = (np.linspace(1.0, 0.0, nz + 1) ** 1.3).astype(np.float64)
    znw[0], znw[-1] = 1.0, 0.0
    znu = 0.5 * (znw[:-1] + znw[1:])
    dnw = znw[1:] - znw[:-1]
    dn = np.concatenate(([dnw[0]], 0.5 * (dnw[1:] + dnw[:-1])))
    jj, ii = np.mgrid[0:ny, 0:nx]
    ht = 1500.0 * np.exp(-(((ii - nx / 2) / 3.0) ** 2 + ((jj - ny / 2) / 2.5) ** 2))
    p00, t00, a, tiso = 1.0e5, 290.0, 50.0, 200.0
    p_surf = p00 * np.exp(-t00 / a + np.sqrt((t00 / a) ** 2 - 2.0 * G * ht / a / R_D))
    c3h, c3f = znu[:, None, None], znw[:, None, None]
    pb = c3h * (p_surf - p_top) + p_top
    temp = np.maximum(tiso, t00 + a * np.log(pb / p00))
    t_init = temp * (p00 / pb) ** (R_D / CP) - T0
    alb = (R_D / P1000MB) * (t_init + T0) * (pb / P1000MB) ** CVPM
    mub = p_surf - p_top
    phb = np.empty((nz + 1, ny, nx))
    phb[0] = ht * G
    for k in range(1, nz + 1):
        pfu, pfd = c3f[k, 0, 0] * mub + p_top, c3f[k - 1, 0, 0] * mub + p_top
        phm = c3h[k - 1, 0, 0] * mub + p_top
        phb[k] = phb[k - 1] + alb[k - 1] * phm * np.log(pfd / pfu)
    ones_f, zeros_f = np.ones(nz + 1), np.zeros(nz + 1)
    geo = {"ZNW": znw, "ZNU": znu, "DNW": dnw, "RDNW": 1.0 / dnw, "DN": dn, "RDN": 1.0 / dn,
           "FNP": np.zeros(nz), "FNM": np.zeros(nz), "C1F": ones_f, "C2F": zeros_f, "C3F": znw, "C4F": zeros_f,
           "C1H": np.ones(nz), "C2H": np.zeros(nz), "C3H": znu, "C4H": np.zeros(nz),
           "MUB": mub, "PB": pb, "PHB": phb, "ALB": alb, "HGT": ht}
    geo = {k: np.asarray(v, dtype) for k, v in geo.items()}
    geo.update({"P_TOP": p_top, "CF1": 0.0, "CF2": 0.0, "CF3": 0.0, "CFN": 0.0, "CFN1": 0.0,
                "HYBRID_OPT": 2, "HYPSOMETRIC_OPT": 2, "t_init": t_init})
    return geo


def synthetic_background(geo, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    nz, (ny, nx) = geo["ZNU"].size, geo["MUB"].shape
    theta = geo["t_init"] + 1.5 * rng.standard_normal((nz, ny, nx)) * 0.2
    qv = 0.012 * np.asarray(geo["ZNU"], np.float64)[:, None, None] ** 3 * (1 + 0.1 * rng.random((nz, ny, nx)))
    mu = 300.0 * np.sin(np.arange(nx) / 3.0)[None, :] * np.ones((ny, 1))
    bal = balance_state(theta, qv, mu, geo)
    # The background is NOT exactly B(bg) (the model state carries a non-hydrostatic part).
    p = bal["P"] + 3.0 * rng.standard_normal(bal["P"].shape)
    ph = bal["PH"] + 2.0 * rng.standard_normal(bal["PH"].shape)
    ph[0] = 0.0
    bg = {"T": theta, "QVAPOR": qv, "MU": mu, "P": p, "PH": ph, "P_HYD": p + geo["PB"]}
    return {k: np.asarray(v, dtype) for k, v in bg.items()}


def blob(shape2d, centre=None, width=2.5):
    ny, nx = shape2d
    cy, cx = centre or (ny / 2, nx / 2)
    jj, ii = np.mgrid[0:ny, 0:nx]
    return np.exp(-(((ii - cx) / width) ** 2 + ((jj - cy) / width) ** 2))


# --------------------------------------------------------------------------- independent oracle

def hand_hydrostatic(geo, theta_pert, qv, mu):
    """Independent column integration: dry face pressures, moist pressure from the top, ideal-gas
    dry inverse density, dφ = α_d (p_d,down − p_d,up). Returns (φ − φ_sfc, p_full_mass)."""
    f = lambda n: np.asarray(geo[n], np.float64).reshape(-1)[:, None, None]  # noqa: E731
    mut = np.asarray(geo["MUB"], np.float64) + np.asarray(mu, np.float64)
    p_top = float(geo["P_TOP"])
    pf = f("C3F") * mut + f("C4F") + p_top
    dpd = pf[:-1] - pf[1:]
    qv = np.asarray(qv, np.float64)
    moist_dp = dpd * (1.0 + qv)
    p_face = p_top + np.concatenate((np.cumsum(moist_dp[::-1], axis=0)[::-1], np.zeros((1,) + mut.shape)), axis=0)
    p_mass = 0.5 * (p_face[:-1] + p_face[1:])
    temp = (np.asarray(theta_pert, np.float64) + T0) * (p_mass / P1000MB) ** (R_D / CP)
    alpha_d = R_D * temp * (1.0 + RVOVRD * qv) / p_mass
    phi = np.concatenate((np.zeros((1,) + mut.shape), np.cumsum(alpha_d * dpd, axis=0)), axis=0)
    return phi, p_mass


# --------------------------------------------------------------------------- checkers (mutants must fail)

def check_zero_increment(fn, geo, bg):
    out = fn(bg, bg, geo)
    for name in ("P", "PH"):
        assert out[name].dtype == bg[name].dtype, name
        assert np.array_equal(out[name], bg[name]), f"zero increment changed {name}"


def check_theta_increment(fn, geo, bg, rel_tol=0.02):
    """+1 K blob in the lowest third: ΔPH matches the hand integration, ΔPH[0] = 0, ΔPH ≥ 0 and
    non-decreasing upward, columns without increment untouched, |ΔP| ≪ ρgΔz scale."""
    nz = bg["T"].shape[0]
    shape2d = bg["MU"].shape
    bump = blob(shape2d)
    bump[bump < 1e-3] = 0.0
    an = dict(bg)
    levels = np.arange(nz)[:, None, None] < max(nz // 3, 2)
    an["T"] = (bg["T"].astype(np.float64) + 1.0 * levels * bump).astype(bg["T"].dtype)
    out = fn(bg, an, geo)
    dph = out["PH"].astype(np.float64) - bg["PH"].astype(np.float64)
    phi_b, _ = hand_hydrostatic(geo, bg["T"], bg["QVAPOR"], bg["MU"])
    phi_a, _ = hand_hydrostatic(geo, an["T"], an["QVAPOR"], an["MU"])
    dhand = phi_a - phi_b
    scale = np.abs(dhand).max()
    assert scale > 1.0, "oracle increment too small to be a test"
    assert np.all(dph[0] == 0.0), "surface geopotential must not change"
    err = np.abs(dph - dhand).max()
    assert err <= rel_tol * scale, f"ΔPH vs hand integration {err:.4g} > {rel_tol}·{scale:.4g}"
    active = bump > 0
    assert np.all(dph[:, ~active] == 0.0), "columns without increment changed"
    col = dph[:, active]
    assert np.all(col >= -1e-3 * scale) and np.all(np.diff(col, axis=0) >= -1e-3 * scale), "warming must raise PH"
    dp = out["P"].astype(np.float64) - bg["P"].astype(np.float64)
    return {"dph_max": float(dph.max()), "hand_max": float(scale), "max_abs_err": float(err),
            "rel_err": float(err / scale), "dp_max_abs": float(np.abs(dp).max())}


def _qv_analysis(bg, factor=0.10):
    bump = blob(bg["MU"].shape)
    bump[bump < 1e-3] = 0.0
    nz = bg["T"].shape[0]
    levels = np.arange(nz)[:, None, None] < max(nz // 3, 2)
    an = dict(bg)
    an["QVAPOR"] = (bg["QVAPOR"].astype(np.float64) * (1.0 + factor * levels * bump)).astype(bg["QVAPOR"].dtype)
    return an, bump > 0


def check_qv_increment(fn, geo, bg, rel_tol=0.02):
    """+10 % qv blob in the lowest third at fixed T, MU: ΔP = moisture loading (hand), ΔPH = hand thickness."""
    an, active = _qv_analysis(bg)
    out = fn(bg, an, geo)
    phi_b, p_b = hand_hydrostatic(geo, bg["T"], bg["QVAPOR"], bg["MU"])
    phi_a, p_a = hand_hydrostatic(geo, an["T"], an["QVAPOR"], an["MU"])
    dp = out["P"].astype(np.float64) - bg["P"].astype(np.float64)
    dph = out["PH"].astype(np.float64) - bg["PH"].astype(np.float64)
    dp_hand, dph_hand = p_a - p_b, phi_a - phi_b
    sp, sph = np.abs(dp_hand).max(), np.abs(dph_hand).max()
    assert sp > 1.0 and sph > 0.5, "oracle increment too small to be a test"
    err_p, err_ph = np.abs(dp - dp_hand).max(), np.abs(dph - dph_hand).max()
    assert err_p <= rel_tol * sp, f"ΔP vs hand {err_p:.4g} > {rel_tol}·{sp:.4g}"
    assert err_ph <= rel_tol * sph, f"ΔPH vs hand {err_ph:.4g} > {rel_tol}·{sph:.4g}"
    assert np.all(dph[0] == 0.0) and np.all(dp[:, ~active] == 0.0) and np.all(dph[:, ~active] == 0.0)
    assert dp[:, active].min() >= -1e-3 * sp, "more moisture loading cannot lower p"
    return {"dp_max": float(dp.max()), "dp_hand_max": float(sp), "dp_rel_err": float(err_p / sp),
            "dph_max": float(dph.max()), "dph_hand_max": float(sph), "dph_rel_err": float(err_ph / sph)}


def check_residual_change(fn, geo, bg, an, tol=1e-6):
    """The model-form hydrostatic residual of the stored analysis differs from the background's by
    round-off only (relative to the layer thickness)."""
    out = fn(bg, an, geo)
    r_b = hydrostatic_residual(bg["T"], bg["QVAPOR"], bg["P"], bg["PH"], bg["MU"], geo)
    r_a = hydrostatic_residual(an["T"], an["QVAPOR"], out["P"], out["PH"], an["MU"], geo)
    phb = np.asarray(geo["PHB"], np.float64)
    thick = np.abs(np.diff(np.asarray(out["PH"], np.float64) + phb, axis=0))
    rel = float((np.abs(r_a - r_b) / thick).max())
    assert rel <= tol, f"hydrostatic residual changed by {rel:.3g} (relative)"
    return rel


def check_balance_oracle(geo, bg, an, rel_tol=0.01):
    """Cross-check vs the port's real.exe balance (fixed MU): B(an) − B(bg) for P/PH."""
    out = rebalance_increment(bg, an, geo, return_increments=True)["increments"]
    b = balance_state(bg["T"], bg["QVAPOR"], bg["MU"], geo)
    a = balance_state(an["T"], an["QVAPOR"], an["MU"], geo)
    res = {}
    for name in ("P", "PH"):
        ref = a[name] - b[name]
        scale = np.abs(ref).max()
        err = np.abs(out[name] - ref).max()
        res[name] = {"oracle_max": float(scale), "max_abs_diff": float(err), "rel": float(err / scale) if scale else 0.0}
        if scale > 0:
            assert err <= rel_tol * scale, f"{name} vs balance oracle {err:.4g} > {rel_tol}·{scale:.4g}"
    return res


def _mut_no_increment(bg, an, geo):
    return {"P": np.array(bg["P"]), "PH": np.array(bg["PH"])}


def _mut_full_replacement(bg, an, geo):  # non-incremental: drops the background non-hydrostatic part
    a = balance_state(an["T"], an["QVAPOR"], an["MU"], geo)
    return {n: a[n].astype(bg[n].dtype) for n in ("P", "PH")}


def _mut_no_loading(bg, an, geo):  # moisture-loading Δp dropped
    out = rebalance_increment(bg, an, geo)
    out["P"] = np.array(bg["P"])
    return out


def _mut_sign(bg, an, geo):  # increment applied with the wrong sign
    inc = rebalance_increment(bg, an, geo, return_increments=True)["increments"]
    return {n: (bg[n].astype(np.float64) - inc[n]).astype(bg[n].dtype) for n in ("P", "PH")}


def _mut_eos_at_hydrostatic_seed(bg, an, geo):  # α evaluated at p′ = 0 instead of the model p′
    zero = {**bg, "P": np.zeros_like(bg["P"])}
    out = rebalance_increment(zero, an, geo, return_increments=True)["increments"]
    return {n: (bg[n].astype(np.float64) + out[n]).astype(bg[n].dtype) for n in ("P", "PH")}


# --------------------------------------------------------------------------- synthetic tests

@pytest.fixture(scope="module")
def syn():
    geo = synthetic_geo()
    return geo, synthetic_background(geo)


def test_zero_increment_bitwise_synthetic(syn):
    geo, bg = syn
    check_zero_increment(rebalance_increment, geo, bg)
    out = rebalance_increment(bg, bg, geo)
    assert np.array_equal(out["P_HYD"], bg["P_HYD"])


def test_theta_increment_vs_hand_integration_synthetic(syn):
    geo, bg = syn
    res = check_theta_increment(rebalance_increment, geo, bg)
    assert res["rel_err"] < 0.02


def test_qv_increment_vs_hand_integration_synthetic(syn):
    geo, bg = syn
    check_qv_increment(rebalance_increment, geo, bg)


def test_residual_and_balance_oracle_synthetic(syn):
    geo, bg = syn
    an, _ = _qv_analysis(bg)
    an["T"] = (bg["T"].astype(np.float64) + 0.7 * blob(bg["MU"].shape)).astype(bg["T"].dtype)
    check_residual_change(rebalance_increment, geo, bg, an)
    check_balance_oracle(geo, bg, an)


def test_hypsometric_opt1_branch_synthetic(syn):
    """opt-1 (linear dφ = −dnw·(c1h·μ_d+c2h)·α) increments vs the hand integration and residual."""
    geo, bg = syn
    g1 = dict(geo, HYPSOMETRIC_OPT=1, HYBRID_OPT=0)
    check_theta_increment(rebalance_increment, g1, bg)
    check_qv_increment(rebalance_increment, g1, bg)
    an, _ = _qv_analysis(bg)
    an["T"] = (bg["T"].astype(np.float64) + 0.7 * blob(bg["MU"].shape)).astype(bg["T"].dtype)
    check_residual_change(rebalance_increment, g1, bg, an)
    r2 = rebalance_increment(bg, an, geo, return_increments=True)["increments"]["PH"]
    r1 = rebalance_increment(bg, an, g1, return_increments=True)["increments"]["PH"]
    assert 0 < np.abs(r1 - r2).max() <= 0.01 * np.abs(r2).max()  # forms differ, both hydrostatic


def test_al_increment_branch_consistent_with_start_em(syn):
    """bg with AL → AL_a = AL_b + Δα, and Δα equals the change of the start_em AL diagnosed from PH."""
    geo, bg = syn
    an, _ = _qv_analysis(bg)
    an["T"] = (bg["T"].astype(np.float64) + 0.7 * blob(bg["MU"].shape)).astype(bg["T"].dtype)
    bg_al = dict(bg, AL=diagnose_al(bg["PH"], bg["MU"], geo).astype(np.float32))
    out = rebalance_increment(bg_al, an, geo, return_increments=True)
    inc = out["increments"]
    np.testing.assert_allclose(out["AL"].astype(np.float64) - bg_al["AL"], inc["ALPHA"], rtol=0, atol=1e-7)
    dal = diagnose_al(bg["PH"].astype(np.float64) + inc["PH"], bg["MU"], geo) - diagnose_al(bg["PH"], bg["MU"], geo)
    np.testing.assert_allclose(dal, inc["ALPHA"], rtol=0, atol=1e-12)
    without = rebalance_increment(bg, an, geo)  # no AL in bg → start_em diagnosis of PH_a
    np.testing.assert_allclose(without["AL"], diagnose_al(without["PH"], bg["MU"], geo), rtol=0, atol=1e-7)


def test_rebalance_report(syn):
    geo, bg = syn
    an, _ = _qv_analysis(bg)
    out = rebalance_increment(bg, an, geo)
    rep = state_io.rebalance_report(bg, an, out, geo)
    assert rep["residual_change_rel_max"] <= 1e-6 and rep["dPH_surface_max_abs"] == 0.0
    assert rep["min_full_pressure_pa"] > 0 and rep["min_alpha"] > 0 and rep["dP_min_max"][1] > 0
    bad = dict(out, PH=np.asarray(out["PH"]) * np.float32(1.01))
    assert state_io.rebalance_report(bg, an, bad, geo)["residual_change_rel_max"] > 1e-4


def test_analysis_mean_fields(syn):
    """Spec zone stays the control bitwise although the float32 member mean is not the control; P/PH come from a
    rebalance of the mean (residual gate), not from averaging member P/PH."""
    geo, ctrl = syn
    rng = np.random.default_rng(3)
    n = 7
    pert = rng.standard_normal((n,) + ctrl["T"].shape)
    pert -= pert.mean(axis=0)
    bg_t = (ctrl["T"].astype(np.float64) + 0.3 * pert).astype(np.float32)  # float32 mean != control
    taper = boundary_taper(*ctrl["MU"].shape, 3, 2)
    an_t = (bg_t.astype(np.float64) + taper * 0.4).astype(np.float32)
    an_t = np.where(taper == 0, bg_t, an_t)
    background = {"T": bg_t, "QVAPOR": np.repeat(ctrl["QVAPOR"][None], n, axis=0)}
    analysis = {"T": an_t, "QVAPOR": background["QVAPOR"]}
    assert not np.array_equal(bg_t.astype(np.float64).mean(axis=0).astype(np.float32)[:, taper == 0],
                              ctrl["T"][:, taper == 0])
    out = state_io.analysis_mean_fields(ctrl, background, analysis, geo, names=("T", "QVAPOR"))
    assert np.array_equal(out["T"][:, taper == 0], ctrl["T"][:, taper == 0])
    assert np.array_equal(out["QVAPOR"], ctrl["QVAPOR"])
    rep = state_io.rebalance_report(ctrl, {**ctrl, **out}, out, geo)
    assert rep["residual_change_rel_max"] <= 1e-6 and rep["dPH_min_max"][1] > 0


def test_wrf_time_forms():
    from datetime import datetime, timedelta, timezone
    want = "2026-05-04_12:00:00"
    for v in (want, "2026-05-04T12:00:00", "2026-05-04T12:00:00Z", "2026-05-04T12:00:00+00:00",
              "2026-05-04T13:00:00+01:00", datetime(2026, 5, 4, 12), datetime(2026, 5, 4, 12, tzinfo=timezone.utc),
              datetime(2026, 5, 4, 10, tzinfo=timezone(timedelta(hours=-2)))):
        assert state_io.wrf_time(v) == want, v
    with pytest.raises(ValueError):
        state_io.wrf_time("2026-05-04 12h")


def test_mu_increment_refused(syn):
    geo, bg = syn
    an = dict(bg, MU=bg["MU"] + np.float32(1.0))
    with pytest.raises(ValueError, match="MU increment"):
        rebalance_increment(bg, an, geo)
    neg = dict(bg, QVAPOR=bg["QVAPOR"] - np.float32(1.0))
    with pytest.raises(ValueError, match="negative"):
        rebalance_increment(bg, neg, geo)


def test_al_diagnosis_matches_balance_synthetic(syn):
    geo, bg = syn
    bal = balance_state(bg["T"], bg["QVAPOR"], bg["MU"], geo)
    al = diagnose_al(bal["PH"], bg["MU"], geo)
    np.testing.assert_allclose(al, bal["AL"], rtol=0, atol=1e-10)


@pytest.mark.parametrize("mutant", [_mut_no_increment, _mut_sign])
def test_rebalance_mutants_caught_by_theta_check(syn, mutant):
    geo, bg = syn
    with pytest.raises(AssertionError):
        check_theta_increment(mutant, geo, bg)


@pytest.mark.parametrize("mutant", [_mut_no_increment, _mut_no_loading, _mut_sign])
def test_rebalance_mutants_caught_by_qv_check(syn, mutant):
    geo, bg = syn
    with pytest.raises(AssertionError):
        check_qv_increment(mutant, geo, bg)


def test_eos_seed_mutant_caught_by_residual(syn):
    geo, bg = syn
    big = dict(bg, P=(bg["P"].astype(np.float64) + 4000.0).astype(np.float32))  # strongly non-zero p′
    an = dict(big, T=(big["T"].astype(np.float64) + 1.0).astype(np.float32))
    check_residual_change(rebalance_increment, geo, big, an)
    with pytest.raises(AssertionError):
        check_residual_change(_mut_eos_at_hydrostatic_seed, geo, big, an)


def test_full_replacement_mutant_caught_by_identity(syn):
    geo, bg = syn
    with pytest.raises(AssertionError):
        check_zero_increment(_mut_full_replacement, geo, bg)


def test_balance_oracle_refuses_inconsistent_hypsometric(syn):
    geo, bg = syn
    with pytest.raises(ValueError):
        balance_state(bg["T"], bg["QVAPOR"], bg["MU"], dict(geo, HYPSOMETRIC_OPT=1))


# --------------------------------------------------------------------------- taper

def check_taper(fn, ny=30, nx=40, sbw=5, ramp=5):
    for stagger, shape in ((None, (ny, nx)), ("x", (ny, nx + 1)), ("y", (ny + 1, nx))):
        w = fn(ny, nx, spec_bdy_width=sbw, ramp=ramp, stagger=stagger)
        assert w.shape == shape, (stagger, w.shape)
        assert np.array_equal(w, w[::-1, :]) and np.array_equal(w, w[:, ::-1]), "taper must be symmetric"
        assert np.all(w[:sbw, :] == 0) and np.all(w[:, :sbw] == 0), "spec/relax zone must be 0"
        mid_r, mid_c = shape[0] // 2, shape[1] // 2
        row = w[:, mid_c]
        ramp_vals = row[sbw:sbw + ramp]
        assert np.all((ramp_vals > 0) & (ramp_vals < 1)), "ramp rows strictly inside (0, 1)"
        assert np.all(np.diff(ramp_vals) > 0), "ramp must increase inward"
        assert np.all(row[sbw + ramp:shape[0] - sbw - ramp] == 1.0), "interior must be 1"
        assert np.all(w[sbw + ramp:-(sbw + ramp), sbw + ramp:-(sbw + ramp)] == 1.0)
        np.testing.assert_allclose(ramp_vals, 0.5 * (1 - np.cos(np.pi * np.arange(1, ramp + 1) / (ramp + 1))))
        assert w[mid_r, sbw - 1] == 0 and w[mid_r, sbw] > 0


def test_taper_shape_and_staggers():
    check_taper(boundary_taper)
    check_taper(boundary_taper, ny=20, nx=25, sbw=3, ramp=2)
    assert np.all(boundary_taper(12, 12, 5, 0) == np.where(
        np.minimum.outer(np.minimum(np.arange(12), 11 - np.arange(12)),
                         np.minimum(np.arange(12), 11 - np.arange(12))) >= 5, 1.0, 0.0))


def _taper_step(ny, nx, spec_bdy_width=5, ramp=5, stagger=None):
    w = boundary_taper(ny, nx, spec_bdy_width, ramp, stagger)
    return np.where(w > 0, 1.0, 0.0)


def _taper_off_by_one(ny, nx, spec_bdy_width=5, ramp=5, stagger=None):
    return boundary_taper(ny, nx, spec_bdy_width - 1, ramp, stagger)


def _taper_unstaggered(ny, nx, spec_bdy_width=5, ramp=5, stagger=None):
    w = boundary_taper(ny, nx, spec_bdy_width, ramp, None)
    if stagger == "x":
        return np.concatenate((w, w[:, -1:]), axis=1)
    if stagger == "y":
        return np.concatenate((w, w[-1:, :]), axis=0)
    return w


@pytest.mark.parametrize("mutant", [_taper_step, _taper_off_by_one, _taper_unstaggered])
def test_taper_mutants_caught(mutant):
    with pytest.raises(AssertionError):
        check_taper(mutant)


# --------------------------------------------------------------------------- real CPU-WRF files

EXTRA = ("P_HYD",)


@pytest.fixture(scope="module")
def real_case():
    if not (WRFOUT.is_file() and TEMPLATE.is_file()):
        pytest.skip("CPU-WRF 0502 run absent")
    geo = read_geometry(WRFOUT, base_wrfinput=TEMPLATE)
    bg = read_frame(WRFOUT, T_A, FRAME_VARIABLES + EXTRA)
    return geo, bg


@real
def test_real_geometry(real_case):
    geo, bg = real_case
    assert geo["HYBRID_OPT"] == 2 and geo["HYPSOMETRIC_OPT"] == 2 and geo["USE_THETA_M"] == 1
    assert geo["SPEC_BDY_WIDTH"] == 5 and geo["nz"] == 44
    assert geo["ALB_source"] == str(TEMPLATE)
    assert bg["T"].shape == (44, 70, 120) and bg["U"].shape == (44, 70, 121) and bg["V"].shape == (44, 71, 120)


@real
def test_real_read_frame_refuses_inexact_time():
    with pytest.raises(ValueError):
        read_frame(WRFOUT, "2026-05-04_12:00:18", ("T",))


@real
def test_real_zero_increment_bitwise(real_case):
    geo, bg = real_case
    check_zero_increment(rebalance_increment, geo, bg)
    assert np.array_equal(rebalance_increment(bg, bg, geo)["P_HYD"], bg["P_HYD"])


@real
def test_real_theta_and_qv_increment(real_case):
    geo, bg = real_case
    sub = {k: (v[..., 20:50, 40:80] if np.ndim(v) >= 2 else v) for k, v in bg.items() if k in ("T", "QVAPOR", "MU", "P", "PH")}
    g = {k: (v[..., 20:50, 40:80] if np.ndim(v) >= 2 and k not in state_io._VCOORD_1D else v) for k, v in geo.items()}
    check_theta_increment(rebalance_increment, g, sub)
    check_qv_increment(rebalance_increment, g, sub)
    an, _ = _qv_analysis(sub)
    an["T"] = (sub["T"].astype(np.float64) + 0.7 * blob(sub["MU"].shape)).astype(np.float32)
    check_residual_change(rebalance_increment, g, sub, an)
    check_balance_oracle(g, sub, an)


def _al_start_em(ph, mu, geo):
    """Independent restatement of WRF start_em :803-811 (hypsometric_opt 2) in float64."""
    nz = geo["nz"]
    p_top = float(geo["P_TOP"])
    mut = geo["MUB"].astype(np.float64) + mu.astype(np.float64)
    al = np.empty((nz,) + mut.shape)
    for k in range(nz):
        pfu = float(geo["C3F"][k + 1]) * mut + float(geo["C4F"][k + 1]) + p_top
        pfd = float(geo["C3F"][k]) * mut + float(geo["C4F"][k]) + p_top
        phm = float(geo["C3H"][k]) * mut + float(geo["C4H"][k]) + p_top
        al[k] = ((ph[k + 1].astype(np.float64) - ph[k] + geo["PHB"][k + 1].astype(np.float64) - geo["PHB"][k])
                 / phm / np.log(pfd / pfu) - geo["ALB"][k])
    return al


def check_written_analysis(out, geo, frame_path, an_fields, t_a, *, land=True):
    """Read back an analysis wrfinput: retimed, copied/analysed variables, derived THM/P_HYD/AL."""
    with Dataset(str(out)) as ds, Dataset(str(frame_path)) as fr:
        ds.set_auto_mask(False)
        fr.set_auto_mask(False)
        stamp = np.asarray(ds["Times"][0]).tobytes().decode()
        assert stamp == t_a, f"Times {stamp}"
        assert ds.START_DATE == t_a and ds.SIMULATION_START_DATE == t_a
        hour = int(t_a[11:13])
        assert float(ds.GMT) == float(hour) and int(ds.JULDAY) == 124 and int(ds.JULYR) == 2026
        prov = json.loads(ds.getncattr(state_io.PROVENANCE_ATTRIBUTE))
        assert prov["valid_time"] == t_a and len(prov["template_sha256"]) == 64
        assert ("GPUWRF_LAND_CYCLE" in ds.ncattrs()) == land
        changed = set(an_fields) | {"THM", "P_HYD", "AL"}
        compared = 0
        for name in prov["copied_from_frame"]:
            if name in changed:
                continue
            assert np.array_equal(ds[name][0], fr[name][0]), f"copied {name} differs from frame"
            compared += 1
        for name, value in an_fields.items():
            assert np.array_equal(ds[name][0], np.asarray(value).astype(ds[name].dtype)), f"analysed {name}"
        t_a_, q_a = ds["T"][0].astype(np.float64), ds["QVAPOR"][0].astype(np.float64)
        t_b, q_b = fr["T"][0].astype(np.float64), fr["QVAPOR"][0].astype(np.float64)
        thm_inc = ((t_a_ + T0) * (1 + RVOVRD * q_a)) - ((t_b + T0) * (1 + RVOVRD * q_b))
        np.testing.assert_allclose(ds["THM"][0], fr["THM"][0].astype(np.float64) + thm_inc, rtol=0, atol=3e-5)
        p_inc = ds["P"][0].astype(np.float64) - fr["P"][0]
        np.testing.assert_allclose(ds["P_HYD"][0], fr["P_HYD"][0].astype(np.float64) + p_inc, rtol=0, atol=0.02)
        al_ref = _al_start_em(ds["PH"][0], ds["MU"][0], geo)
        np.testing.assert_allclose(ds["AL"][0], al_ref, rtol=2e-6, atol=1e-7)
        return {"copied_compared_bitwise": compared, "analysed": sorted(an_fields)}


@pytest.fixture
def scratch(tmp_path):
    return tmp_path


@real
def test_real_writer_zero_increment_identity(real_case, scratch):
    geo, bg = real_case
    reb = rebalance_increment(bg, bg, geo)
    an = {n: bg[n] for n in ("U", "V", "T", "QVAPOR", "MU")} | {"P": reb["P"], "PH": reb["PH"]}
    out = scratch / "wrfinput_d01_zero"
    rec = write_analysis_wrfinput(TEMPLATE, WRFOUT, T_A, an, out)
    assert all(rec["analysis_fields"][n]["n_changed"] == 0 for n in an)
    res = check_written_analysis(out, geo, WRFOUT, an, T_A)
    assert res["copied_compared_bitwise"] > 150
    with Dataset(str(out)) as ds, Dataset(str(WRFOUT)) as fr:  # THM / P_HYD bitwise on zero increment
        for name in ("THM", "P_HYD", "T", "QVAPOR", "U", "V", "MU", "P", "PH", "W", "QCLOUD", "TSLB", "SMOIS"):
            assert np.array_equal(ds[name][0], fr[name][0]), name
    with pytest.raises(ValueError):  # never overwrite
        write_analysis_wrfinput(TEMPLATE, WRFOUT, T_A, an, out)


@real
def test_real_writer_land_marker_digests(real_case, scratch):
    from gpuwrf.cycling import core as cyc
    geo, bg = real_case
    an = {n: bg[n] for n in ("T", "QVAPOR")}
    out = scratch / "wrfinput_d01_land"
    write_analysis_wrfinput(TEMPLATE, WRFOUT, T_A, an, out)
    with Dataset(str(out)) as ds:
        report = json.loads(ds.getncattr(cyc.MARKER))
        assert report["valid_time"] == T_A
        land = cyc._read(ds, "LANDMASK", 0)[0] == 1
        if "LAKEMASK" in ds.variables:
            land &= cyc._read(ds, "LAKEMASK", 0)[0] == 0
        for name, rec in report["fields"].items():  # what restore_land_state verifies at Noah-MP init
            data, mask = cyc._field(ds, name, 0, land)
            assert cyc._digest(data, mask) == rec["sha256"], name


def _nonzero_analysis(geo, bg):
    taper = boundary_taper(70, 120, 5, 5)
    bump = blob((70, 120), centre=(35, 70), width=8.0) * taper
    an = {n: np.array(bg[n]) for n in ("U", "V", "T", "QVAPOR", "MU")}
    low = np.arange(44)[:, None, None] < 12
    an["T"] = (bg["T"].astype(np.float64) + 0.5 * bump * low).astype(np.float32)
    an["QVAPOR"] = (bg["QVAPOR"].astype(np.float64) * (1.0 + 0.05 * bump * low)).astype(np.float32)
    reb = rebalance_increment(bg, an, geo)
    an.update(P=reb["P"], PH=reb["PH"])
    return an


@real
def test_real_writer_nonzero_increment(real_case, scratch):
    geo, bg = real_case
    an = _nonzero_analysis(geo, bg)
    out = scratch / "wrfinput_d01_inc"
    rec = write_analysis_wrfinput(TEMPLATE, WRFOUT, T_A, an, out)
    assert rec["analysis_fields"]["T"]["n_changed"] > 1000 and rec["analysis_fields"]["U"]["n_changed"] == 0
    check_written_analysis(out, geo, WRFOUT, an, T_A)
    with Dataset(str(out)) as ds, Dataset(str(WRFOUT)) as fr:
        assert not np.array_equal(ds["THM"][0], fr["THM"][0]) and not np.array_equal(ds["PH"][0], fr["PH"][0])
        assert np.array_equal(ds["W"][0], fr["W"][0]) and np.array_equal(ds["QCLOUD"][0], fr["QCLOUD"][0])


@real
@pytest.mark.parametrize("mutant", ["no_retime", "no_thm"])
def test_real_writer_mutants_caught(real_case, scratch, monkeypatch, mutant):
    geo, bg = real_case
    an = _nonzero_analysis(geo, bg)
    if mutant == "no_retime":
        monkeypatch.setattr(state_io, "_set_times", lambda ds, stamp: None)
    else:
        monkeypatch.setattr(state_io, "_theta_m_pert", lambda t, q, u: np.zeros(np.shape(t)))
    out = scratch / f"wrfinput_d01_{mutant}"
    if mutant == "no_retime":  # the land cycle alone refuses it (Times mismatch) ...
        with pytest.raises(ValueError):
            write_analysis_wrfinput(TEMPLATE, WRFOUT, T_A, an, out)
        out = scratch / "wrfinput_d01_no_retime_noland"  # ... and the checker kills it without the cycle
        write_analysis_wrfinput(TEMPLATE, WRFOUT, T_A, an, out, land_cycle=False)
        with pytest.raises(AssertionError):
            check_written_analysis(out, geo, WRFOUT, an, T_A, land=False)
        return
    write_analysis_wrfinput(TEMPLATE, WRFOUT, T_A, an, out)
    with pytest.raises(AssertionError):
        check_written_analysis(out, geo, WRFOUT, an, T_A)


NEST = RUN / f"wrfout_d02_{T_A}"


@real
def test_real_nest_base_state_mismatch_refused(scratch):
    if not NEST.is_file():
        pytest.skip("CPU-WRF 0502 d02 absent")
    bg = read_frame(NEST, T_A, ("U", "V", "T", "QVAPOR", "MU", "P", "PH"))
    out = scratch / "wrfinput_d02_refused"
    with pytest.raises(ValueError, match="nest init blends terrain"):
        write_analysis_wrfinput(RUN / "wrfinput_d02", NEST, T_A, bg, out)
    assert not out.exists() and not list(scratch.glob(".a1io-analysis-*"))


ORACLE = Path(__file__).resolve().parent / "oracles" / "pristine_hydro" / "run_oracle.py"
FC = Path("<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran")


def _oracle_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("a1io_pristine_oracle", ORACLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_pristine(res):
    """REAL*8 verbatim real.exe block (6 moist species): ΔP exact; ΔPH/ΔAL within the α-at-model-p formulation
    difference; REAL*4 build: adapter inside WRF's own REAL*4 noise; hydrometeors present (non-vacuous)."""
    assert res["hydrometeor_columns_in_box"] > 20 and res["hydrometeor_max_in_box"] > 1e-5
    for name, case in res["cases"].items():
        r8, r4 = case["real8"], case["real4"]
        if name != "theta_only":
            assert r8["P"]["rel"] <= 1e-9, (name, r8["P"])
        else:
            assert r8["P"]["max_abs_diff"] <= 1e-6 and r8["P"]["oracle_max_abs"] <= 1e-6
        assert r8["PH"]["rel"] <= 5e-4 and r8["AL"]["rel"] <= 5e-4, (name, r8)
        assert r4["P"]["max_abs_diff"] <= 0.5 and r4["PH"]["max_abs_diff"] <= 0.1 and r4["AL"]["max_abs_diff"] <= 1e-4


@real
def test_real_pristine_fortran_oracle(tmp_path):
    if not (FC.is_file() and (RUN_C / f"wrfout_d01_{T_A}").is_file()):
        pytest.skip("gfortran or 0503 run absent")
    mod = _oracle_module()
    mod.WORK = tmp_path
    import sys
    check_pristine(mod.evaluate(t=sys.modules[__name__]))

    def no_loading(bg, an, geo, return_increments=True):
        out = rebalance_increment(bg, an, geo, return_increments=True)
        out["increments"]["P"] = np.zeros_like(out["increments"]["P"])
        return out

    with pytest.raises(AssertionError):
        check_pristine(mod.evaluate(adapter=no_loading, t=sys.modules[__name__]))
