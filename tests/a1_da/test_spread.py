"""Spread maintenance tools (lane w41-da-ens): RTPS toward the inflated prior, RTPP, additive inflation, diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import core, spread


def _ens(seed=0, n=12, shape=(4, 9, 11)):
    rng = np.random.default_rng(seed)
    xb = 5.0 + rng.normal(size=(n,) + shape)
    xa = xb.mean(0) + 0.3 * (xb - xb.mean(0)) + 0.1 * rng.normal(size=(n,) + shape)
    return xb, xa


def test_rtps_rho1_is_bitwise_the_probe_rtps():
    xb, xa = _ens()
    for alpha in (0.0, 0.5, 0.8, 1.0):
        np.testing.assert_array_equal(spread.rtps(xb, xa, alpha), core.rtps(xb, xa, alpha))


def test_rtps_alpha1_hits_the_inflated_prior_spread_and_keeps_the_mean():
    xb, xa = _ens(1)
    rho = np.linspace(1.0, 6.0, xb.shape[1])[:, None, None]       # per-level target inflation
    out = spread.rtps(xb, xa, 1.0, rho=rho)
    np.testing.assert_allclose(out.std(0, ddof=1), np.sqrt(rho) * xb.std(0, ddof=1), rtol=1e-12)
    np.testing.assert_allclose(out.mean(0), xa.mean(0), rtol=0, atol=1e-12)


def test_r3_mechanism_numbers():
    """R3 (3): prior 0.065, posterior 0.140 (ρ 6.15 in the transform). Raw RTPS 0.8 → 0.080 (R3: 0.081),
    inflated target → 0.140 + 0.8 (√6.15·0.065 − 0.140) = 0.157 > prior: the inflation survives."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(10, 1))
    base = (base - base.mean(0)) / base.std(0, ddof=1)
    xb = 0.065 * base
    xa = 0.140 * base
    raw = spread.rtps(xb, xa, 0.8).std(0, ddof=1)[0]
    inflated = spread.rtps(xb, xa, 0.8, rho=6.15).std(0, ddof=1)[0]
    assert abs(raw - (0.140 + 0.8 * (0.065 - 0.140))) < 1e-12
    assert abs(inflated - (0.140 + 0.8 * (np.sqrt(6.15) * 0.065 - 0.140))) < 1e-12
    assert inflated > 0.065 and inflated > raw


def test_rtps_refuses_bad_target():
    xb, xa = _ens()
    with pytest.raises(ValueError):
        spread.rtps(xb, xa, 0.5, rho=-1.0)
    with pytest.raises(ValueError):
        spread.rtps(xb, xa, 1.5)


def test_rtpp_limits():
    xb, xa = _ens(2)
    np.testing.assert_allclose(spread.rtpp(xb, xa, 0.0), xa, atol=1e-12)
    out = spread.rtpp(xb, xa, 1.0)
    np.testing.assert_allclose(out.mean(0), xa.mean(0), atol=1e-12)
    np.testing.assert_allclose(out - out.mean(0), xb - xb.mean(0), atol=1e-12)


def _members_and_pool(seed=4, n=20, m=30, nz=3, ny=16, nx=18):
    rng = np.random.default_rng(seed)
    members = {"T": (280 + rng.normal(size=(n, nz, ny, nx))).astype(np.float32),
               "QVAPOR": (0.008 * np.exp(0.2 * rng.normal(size=(n, nz, ny, nx)))).astype(np.float32),
               "U": rng.normal(size=(n, nz, ny, nx + 1)).astype(np.float32)}
    states = {"T": 280 + 2 * rng.normal(size=(m, nz, ny, nx)),
              "QVAPOR": 0.008 * np.exp(0.3 * rng.normal(size=(m, nz, ny, nx))),
              "U": 3 * rng.normal(size=(m, nz, ny, nx + 1))}
    return members, spread.pool_from_states(states)


def test_additive_inflation_centred_positive_tapered_reproducible():
    members, pool = _members_and_pool()
    ny, nx = members["T"].shape[-2:]
    taper = spread.taper_fields(("T", "QVAPOR", "U"), ny, nx, spec_bdy_width=3, ramp=3)
    out, receipt = spread.additive_inflation(members, pool, 0.3, seed=7, taper=taper)
    again, _ = spread.additive_inflation(members, pool, 0.3, seed=7, taper=taper)
    other, _ = spread.additive_inflation(members, pool, 0.3, seed=8, taper=taper)
    for name in members:
        np.testing.assert_array_equal(out[name], again[name])
        assert not np.array_equal(out[name], other[name])
        assert out[name].dtype == members[name].dtype
        zone = taper[name] == 0
        np.testing.assert_array_equal(out[name][..., zone], members[name][..., zone])  # specified zone: bytes kept
        dm = np.abs(out[name].astype(np.float64).mean(0) - members[name].astype(np.float64).mean(0))
        assert dm.max() < 1e-5 * max(1.0, float(np.abs(members[name]).max()))
        assert receipt["fields"][name]["spread_rms_after"] > receipt["fields"][name]["spread_rms_before"]
    assert (out["QVAPOR"] > 0).all() and receipt["fields"]["QVAPOR"]["negative_after"] == 0


def test_additive_covariance_matches_pool_in_expectation():
    rng = np.random.default_rng(5)
    m, n = 8, 4000
    pool = rng.normal(size=(m, 3)) @ np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]])
    pool = {"X": pool - pool.mean(0)}
    members = {"X": np.zeros((n, 3))}
    out, _ = spread.additive_inflation(members, pool, 1.0, seed=11)
    emp = np.cov(out["X"].T)
    ref = np.cov(pool["X"].T)
    np.testing.assert_allclose(emp, ref, atol=0.08 * np.abs(ref).max())


def test_spread_by_band_and_survival():
    rng = np.random.default_rng(6)
    n, nz, ny, nx = 8, 10, 40, 42
    z = np.linspace(0, 9000, nz + 1)[:, None, None] * np.ones((nz + 1, ny, nx))
    frames = {"PH": np.zeros((n, nz + 1, ny, nx)), "PHB": np.broadcast_to(z * 9.81, (n, nz + 1, ny, nx)).copy(),
              "T": rng.normal(size=(n, nz, ny, nx)), "U": 2 * rng.normal(size=(n, nz, ny, nx + 1)),
              "T2": 0.5 * rng.normal(size=(n, ny, nx))}
    geo = {"HGT": np.zeros((ny, nx))}
    s = spread.spread_by_band(frames, geo, names=("T", "U"), interior=10, surface=("T2",))
    assert s["interior_cells"] == (ny - 20) * (nx - 20)
    assert 0.8 < s["T"]["0-1km"] < 1.2 and 1.25 < s["U"]["2-5km"] < 1.6  # U face-averaged: 2/√2
    half = {**frames, "T": 0.5 * frames["T"]}
    r = spread.survival(s, spread.spread_by_band(half, geo, names=("T", "U"), interior=10, surface=("T2",)))
    assert abs(r["T"]["1-2km"] - 0.5) < 1e-12 and abs(r["U"]["1-2km"] - 1.0) < 1e-12


def test_maintenance_log_space_qv_and_methods():
    rng = np.random.default_rng(12)
    n = 10
    q = 0.006 * np.exp(0.2 * rng.normal(size=(n, 2, 6, 7)))
    pool = spread.pool_from_states({"QVAPOR": 0.006 * np.exp(0.3 * rng.normal(size=(8, 2, 6, 7)))})
    hook = spread.make_maintenance({"method": "none"}, pool=pool, amplitude=0.5, seed=1)
    # physical space: multiplicative, positive, mean kept
    phys = hook("QVAPOR", q, q.copy())
    assert (phys > 0).all()
    np.testing.assert_allclose(phys.mean(0), q.mean(0), rtol=1e-12)
    # log space (moisture policy 'log'): linear in log q, log-mean kept; same weights -> same log perturbation
    hook2 = spread.make_maintenance({"method": "none"}, pool=pool, amplitude=0.5, seed=1)
    lq = np.log(q)
    out = hook2("QVAPOR", lq, lq.copy())
    np.testing.assert_allclose(out.mean(0), lq.mean(0), atol=1e-12)
    # the physical (multiplicative, re-centred) and log-space paths apply the same per-member log perturbation up to a
    # per-point normalisation constant: their difference has zero spread over the members
    diff = (np.log(phys) - np.log(q)) - (out - lq)
    assert diff.std(0).max() < 1e-12
    xb, xa = _ens(13)
    np.testing.assert_allclose(spread.make_maintenance({"method": "rtpp", "alpha": 1.0})("T", xb, xa) - xa.mean(0),
                               xb - xb.mean(0), atol=1e-12)
    np.testing.assert_allclose(spread.make_maintenance({"method": "multiplicative", "factor": 2.0})("T", xb, xa)
                               .std(0, ddof=1), 2 * xa.std(0, ddof=1), rtol=1e-12)
