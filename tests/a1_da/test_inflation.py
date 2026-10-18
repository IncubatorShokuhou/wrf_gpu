"""inflation (lane a1-io, task BA): debiased innovation inflation + per-band state inflation. Synthetic, CPU."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import inflation as inf


def _obs(kind, var, value, p=None):
    n = len(value)
    return SimpleNamespace(kind=np.asarray(kind), var=np.asarray(var), value=np.asarray(value, float),
                           p_pa=np.full(n, np.nan) if p is None else np.asarray(p, float),
                           lat=np.zeros(n), lon=np.zeros(n))


def test_group_rho_exact_and_bias_removed(monkeypatch):
    rng = np.random.default_rng(0)
    n, N = 4000, 20
    so, sb, rho_true, bias = 1.0, 0.5, 3.0, 2.0
    truth_spread = np.sqrt(rho_true) * sb
    hx = rng.normal(0, sb, (N, n))
    hx -= hx.mean(axis=0)  # member deviations with spread ≈ sb; ensemble mean 0
    value = bias + rng.normal(0, truth_spread, n) + rng.normal(0, so, n)
    obs = _obs(np.full(n, 2), np.full(n, 11), value)
    monkeypatch.setattr("gpuwrf.assimilation.letkf.analysis.obs_sigma", lambda o: np.full(len(o.value), so))
    t = inf.debiased_table(obs, hx, np.ones(n, bool))
    g = t["groups"]["surface:T2"]
    assert abs(g["bias"] - bias) < 0.1
    assert abs(g["rho_raw"] - rho_true) < 0.4, g
    assert g["rho_with_bias_raw"] > g["rho_raw"] + 10, "the bias inflates the E[d²] estimate"
    assert t["pooled"]["rho"] == pytest.approx(g["rho"])


def test_clamp_and_pooled_weighting(monkeypatch):
    monkeypatch.setattr("gpuwrf.assimilation.letkf.analysis.obs_sigma", lambda o: np.ones(len(o.value)))
    hx = np.tile(np.array([[-1.0], [1.0]]), (1, 6)) * 0.1  # tiny spread
    obs = _obs([1] * 3 + [2] * 3, [1] * 3 + [11] * 3, [0.0] * 6)
    t = inf.debiased_table(obs, hx, np.ones(6, bool), rho_min=1.0, rho_max=10.0, n_min=3)
    for g in t["groups"].values():
        assert 1.0 <= g["rho"] <= 10.0
    assert t["pooled"]["rho"] >= 1.0  # clamped groups pooled: never dragged below 1 by an over-estimated σo


def test_small_groups_not_pooled(monkeypatch):
    monkeypatch.setattr("gpuwrf.assimilation.letkf.analysis.obs_sigma", lambda o: np.ones(len(o.value)))
    rng = np.random.default_rng(3)
    hx = rng.normal(0, 0.5, (10, 22))
    hx -= hx.mean(axis=0)
    value = np.concatenate((rng.normal(0, 1.2, 20), [40.0, -40.0]))  # 20 sonde T + 2 wild surface V10
    obs = _obs([1] * 20 + [2] * 2, [1] * 20 + [14] * 2, value)
    t = inf.debiased_table(obs, hx, np.ones(22, bool), n_min=6)
    assert not t["groups"]["surface:V10"]["reliable"] and t["groups"]["surface:V10"]["rho"] == 10.0
    assert t["pooled"]["groups_used"] == ["sonde:T"]
    assert t["pooled"]["rho"] == pytest.approx(t["groups"]["sonde:T"]["rho"])
    assert t["by_kind"]["surface"]["n"] == 0 and t["by_kind"]["surface"]["rho"] == 1.0


def _frames(N=6, nz=6, ny=26, nx=28, seed=1):
    rng = np.random.default_rng(seed)
    f = {n: rng.normal(0, 1, (N, nz, ny, nx)).astype(np.float32) for n in ("T", "QVAPOR")}
    f["QVAPOR"] = (np.abs(f["QVAPOR"]) * 1e-3 + 2e-3).astype(np.float32)
    f["U"] = rng.normal(0, 1, (N, nz, ny, nx + 1)).astype(np.float32)
    f["V"] = rng.normal(0, 1, (N, nz, ny + 1, nx)).astype(np.float32)
    for n in ("T2", "Q2", "U10", "V10"):
        f[n] = rng.normal(0, 1, (N, ny, nx)).astype(np.float32)
    f["Q2"] = (np.abs(f["Q2"]) * 1e-3 + 1e-3).astype(np.float32)
    z = np.linspace(0, 6000, nz + 1)[:, None, None] * np.ones((1, ny, nx))
    f["PH"] = np.zeros((N, nz + 1, ny, nx), np.float32)
    f["PHB"] = np.broadcast_to((z * inf.G).astype(np.float32), (N, nz + 1, ny, nx)).copy()
    return f, {"HGT": np.zeros((ny, nx))}


def _table(rho):
    return {f"{v}|{lo/1000:.0f}-{'inf' if not np.isfinite(hi) else f'{hi/1000:.0f}'}km": {"rho": rho.get((v, b), 1.0)}
            for v in ("T", "QVAPOR", "U", "V") for b, (lo, hi) in enumerate(inf.BANDS)}


def test_inflate_state_bands_taper_mean():
    f, geo = _frames()
    out, cnt = inf.inflate_state(f, geo, _table({("T", 1): 4.0, ("U", 0): 9.0}))
    for n in ("T", "QVAPOR", "U", "V", "T2", "U10"):  # specified zone (5 rows/columns from each edge) stays bitwise
        a, b = out[n], f[n]
        for sl in ((Ellipsis, slice(0, 5), slice(None)), (Ellipsis, slice(-5, None), slice(None)),
                   (Ellipsis, slice(None), slice(0, 5)), (Ellipsis, slice(None), slice(-5, None))):
            assert np.array_equal(a[sl], b[sl]), n
    zm = 0.5 * (np.linspace(0, 6000, 7)[1:] + np.linspace(0, 6000, 7)[:-1])  # 500, 1500, ... m
    k1 = int(np.flatnonzero((zm >= 1000) & (zm < 2000))[0])
    k0 = int(np.flatnonzero(zm < 1000)[0])
    jj, ii = 13, 14  # interior (taper 1: ≥ 10 rows from every edge)
    dev = lambda a, k: a[:, k, jj, ii].astype(np.float64) - a[:, k, jj, ii].astype(np.float64).mean()  # noqa: E731
    np.testing.assert_allclose(dev(out["T"], k1), 2.0 * dev(f["T"], k1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dev(out["T"], k0), dev(f["T"], k0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out["T"].astype(np.float64).mean(0), f["T"].astype(np.float64).mean(0), atol=1e-5)
    np.testing.assert_allclose(out["U10"][:, jj, ii] - out["U10"][:, jj, ii].mean(),
                               3.0 * (f["U10"][:, jj, ii] - f["U10"][:, jj, ii].mean()), rtol=1e-4, atol=1e-4)
    assert cnt["QVAPOR_negative_after_inflation"] == 0


def test_inflate_state_qv_log_space_positive():
    f, geo = _frames()
    f["QVAPOR"][0, :, 13, 14] = 1e-6  # one member near zero: additive inflation would drive it negative
    out, cnt = inf.inflate_state(f, geo, _table({("QVAPOR", b): 9.0 for b in range(4)}))
    assert cnt["QVAPOR_negative_after_inflation"] == 0 and (out["QVAPOR"] > 0).all()
    np.testing.assert_allclose(out["QVAPOR"].astype(np.float64).mean(0), f["QVAPOR"].astype(np.float64).mean(0),
                               rtol=1e-5)
    assert out["QVAPOR"][:, :, 13, 14].std(0).mean() > 1.5 * f["QVAPOR"][:, :, 13, 14].std(0).mean()


def test_cli_dispatch_only_for_new_option(monkeypatch):
    from gpuwrf.assimilation.letkf import api as cli   # calibrate_scale moved from the probe cli to api
    calls = []
    monkeypatch.setattr(inf, "calibrate_debiased", lambda fr, g, o, s: calls.append(s) or ("F", {"inflation_rho": 2.0}))
    frames = {"T": np.zeros((2, 1, 1, 1))}
    assert cli.calibrate_scale(frames, {}, None, {"kind": "members", "scale": "innovation_debiased"})[0] == "F"
    out, info = cli.calibrate_scale(frames, {}, None, {"kind": "members", "scale": 1.0})
    assert out is frames and info["inflation_rho"] == 1.0 and len(calls) == 1
