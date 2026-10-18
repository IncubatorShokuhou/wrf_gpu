"""LETKF algebra and localisation tables (lane a1-da, DESIGN V0). CPU, NumPy fp64.

Oracles are independent of the LETKF algebra: the state-space Kalman update
with the ensemble covariance (``kalman_reference``) and brute-force pair
distances. Each test has a mutant that it must catch (E39), see test bodies.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import core, grid


def _ensemble(seed=0, n_members=12, n_state=7, n_obs=5):
    rng = np.random.default_rng(seed)
    xb = rng.normal(size=(n_members, n_state)) + rng.normal(size=n_state)
    h = rng.normal(size=(n_obs, n_state))
    yo = h @ rng.normal(size=n_state) + rng.normal(size=n_obs)
    sigma = 0.5 + rng.random(n_obs)
    return xb, h, yo, sigma


def test_gaspari_cohn_shape():
    r = np.linspace(0.0, 2.5, 2501)
    g = core.gaspari_cohn(r)
    assert g[0] == 1.0 and np.all(g[r >= 2.0] == 0.0)
    assert np.all(np.diff(g) <= 1e-15)
    left, right = core.gaspari_cohn([1.0 - 1e-12, 1.0 + 1e-12])
    assert abs(left - right) < 1e-9 and abs(left - 5.0 / 24.0) < 1e-9


@pytest.mark.parametrize("seed", range(4))
def test_global_etkf_equals_ensemble_kalman(seed):
    xb, h, yo, sigma = _ensemble(seed)
    xa = core.etkf_global(xb, xb @ h.T, yo, sigma)
    mean_ref, cov_ref = core.kalman_reference(xb, h, yo, sigma)
    pert = xa - xa.mean(axis=0)
    np.testing.assert_allclose(xa.mean(axis=0), mean_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(pert.T @ pert / (len(xa) - 1), cov_ref, rtol=0, atol=1e-10)


def test_single_observation_gain():
    xb, h, yo, sigma = _ensemble(5, n_obs=1)
    xa = core.etkf_global(xb, xb @ h.T, yo, sigma)
    x = xb - xb.mean(axis=0)
    hx = x @ h[0]
    gain = (x.T @ hx / (len(xb) - 1)) / (hx @ hx / (len(xb) - 1) + sigma[0] ** 2)
    np.testing.assert_allclose(xa.mean(axis=0), xb.mean(axis=0) + gain * (yo[0] - h[0] @ xb.mean(axis=0)), atol=1e-12)


def test_no_observation_is_identity_and_rho_inflates():
    xb, h, yo, sigma = _ensemble(1)
    ypert = (xb @ h.T - (xb @ h.T).mean(axis=0)).T[None]
    _, w, t = core.letkf_transform(ypert, np.zeros((1, len(yo))), np.zeros((1, len(yo))))
    np.testing.assert_allclose(t[0], np.eye(len(xb)), atol=1e-13)
    _, _, t = core.letkf_transform(ypert, np.zeros((1, len(yo))), np.zeros((1, len(yo))), rho=1.21)
    xa = core.apply_transform(xb, t[0])
    np.testing.assert_allclose(xa - xa.mean(0), 1.1 * (xb - xb.mean(0)), atol=1e-12)


def test_localised_rinv_equals_kalman_with_scaled_errors():
    xb, h, yo, sigma = _ensemble(2)
    rho = np.array([1.0, 0.5, 0.25, 0.1, 0.0])
    yb = xb @ h.T
    ybar = yb.mean(0)
    _, _, t = core.letkf_transform((yb - ybar).T[None], (yo - ybar)[None], (rho / sigma ** 2)[None])
    xa = core.apply_transform(xb, t[0])
    keep = rho > 0
    mean_ref, cov_ref = core.kalman_reference(xb, h[keep], yo[keep], sigma[keep] / np.sqrt(rho[keep]))
    pert = xa - xa.mean(0)
    np.testing.assert_allclose(xa.mean(0), mean_ref, atol=1e-10)
    np.testing.assert_allclose(pert.T @ pert / (len(xa) - 1), cov_ref, atol=1e-10)


def test_member_permutation():
    xb, h, yo, sigma = _ensemble(3)
    xa = core.etkf_global(xb, xb @ h.T, yo, sigma)
    perm = np.random.default_rng(9).permutation(len(xb))
    xa_p = core.etkf_global(xb[perm], (xb @ h.T)[perm], yo, sigma)
    np.testing.assert_allclose(xa_p, xa[perm], atol=1e-12)


def test_rtps_limits():
    xb, h, yo, sigma = _ensemble(4)
    xa = core.etkf_global(xb, xb @ h.T, yo, sigma)
    np.testing.assert_allclose(core.rtps(xb, xa, 0.0), xa, atol=1e-13)
    full = core.rtps(xb, xa, 1.0)
    np.testing.assert_allclose(full.std(0, ddof=1), xb.std(0, ddof=1), atol=1e-12)
    np.testing.assert_allclose(full.mean(0), xa.mean(0), atol=1e-13)


def test_jax_path_matches_numpy():
    jnp = pytest.importorskip("jax.numpy")
    xb, h, yo, sigma = _ensemble(6)
    yb = xb @ h.T
    ybar = yb.mean(0)
    args = ((yb - ybar).T[None], (yo - ybar)[None], (1.0 / sigma ** 2)[None])
    ref = core.letkf_transform(*args)[2]
    got = np.asarray(core.letkf_transform(*(jnp.asarray(a, dtype=jnp.float32) for a in args), xp=jnp)[2])
    np.testing.assert_allclose(got, ref, rtol=0, atol=2e-4)


def _brute_rho(pxyz, plnp, oxyz, olnp, ch, cv):
    dh = np.linalg.norm(pxyz[:, None] - oxyz[None], axis=-1)
    rho = core.gaspari_cohn(dh / ch[None])
    vert = np.isfinite(cv)
    dv = np.abs(plnp[:, None] - np.where(vert, olnp, 0.0)[None])
    return rho * np.where(vert[None], core.gaspari_cohn(dv / np.where(vert, cv, 1.0)[None]), 1.0)


def test_local_obs_table_matches_brute_force():
    rng = np.random.default_rng(11)
    lat_p, lon_p = 27 + 2 * rng.random(300), -18 + 3 * rng.random(300)
    lat_o, lon_o = 27 + 2 * rng.random(120), -18 + 3 * rng.random(120)
    plnp, olnp = np.log(1e5) - 2 * rng.random(300), np.log(1e5) - 2 * rng.random(120)
    ch = np.full(120, 60.0) * (1 + rng.random(120))
    cv = np.where(rng.random(120) < 0.2, np.inf, 0.3 + 0.2 * rng.random(120))
    pxyz, oxyz = grid.lonlat_to_xyz_km(lat_p, lon_p), grid.lonlat_to_xyz_km(lat_o, lon_o)
    ref = _brute_rho(pxyz, plnp, oxyz, olnp, ch, cv)
    idx, rho, ntr = grid.local_obs_table(pxyz, plnp, oxyz, olnp, ch, cv, pmax=120, chunk=64)
    assert ntr == 0
    got = np.zeros_like(ref)
    for p in range(len(pxyz)):
        np.add.at(got[p], idx[p], rho[p])
    np.testing.assert_allclose(got, ref, atol=1e-12)
    # pmax truncation keeps the largest weights
    idx5, rho5, _ = grid.local_obs_table(pxyz, plnp, oxyz, olnp, ch, cv, pmax=5, kquery=120)
    expect = -np.sort(-ref, axis=1)[:, :5]
    np.testing.assert_allclose(-np.sort(-rho5, axis=1), expect, atol=1e-12)


def test_transform_field_identity_interpolation_and_staggers():
    rng = np.random.default_rng(12)
    nz, ny, nx, n = 5, 7, 9, 6
    lat, lon = np.meshgrid(28 + 0.1 * np.arange(ny), -17 + 0.1 * np.arange(nx), indexing="ij")
    lnp = np.log(1e5) - 0.2 * np.arange(nz)[:, None, None] + np.zeros((nz, ny, nx))
    g1 = grid.AnalysisGrid.from_mass_grid(lat, lon, lnp, stride_h=1, stride_v=1)
    t = rng.normal(size=g1.shape + (n, n))
    xb = rng.normal(size=(n, nz, ny, nx))
    got = grid.transform_field(xb, t, g1)
    ref = np.stack([core.apply_transform(xb[:, k], t[k]) for k in range(nz)], axis=1)
    np.testing.assert_allclose(got, ref, atol=1e-12)
    # identity transforms leave every grid (mass, U, V, 2-D) unchanged at any stride
    g3 = grid.AnalysisGrid.from_mass_grid(lat, lon, lnp, stride_h=3, stride_v=2)
    eye = np.broadcast_to(np.eye(n), g3.shape + (n, n))
    for field, st in ((rng.normal(size=(n, nz, ny, nx + 1)), "x"), (rng.normal(size=(n, nz, ny + 1, nx)), "y"),
                      (rng.normal(size=(n, ny, nx)), None)):
        np.testing.assert_allclose(grid.transform_field(field, eye, g3, stagger=st), field, atol=1e-12)
    # interpolation is linear in T: midpoint of two coarse columns gets the average transform
    t3 = rng.normal(size=g3.shape + (n, n))
    out = grid.transform_field(xb, t3, g3)
    k, j = 0, 0
    i = 1  # between ic[0] = 0 and ic[1] = 3 → frac 1/3
    tmix = (2 / 3) * t3[0, 0, 0] + (1 / 3) * t3[0, 0, 1]
    np.testing.assert_allclose(out[:, k, j, i], core.apply_transform(xb[:, k, j, i], tmix), atol=1e-12)
