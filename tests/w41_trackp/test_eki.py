"""EKI scaffold: analytic Kalman identities and a linear-Gaussian OSSE (independent closed-form reference)."""
from __future__ import annotations

import numpy as np
import pytest

from gpuwrf.surface_params import eki


def _linear_problem(p=3, m=12, seed=0, tikhonov=False):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(m, p))
    truth = np.array([0.3, -0.2, 0.1])[:p]
    gamma = np.diag(np.full(m, 0.05 ** 2))
    y = a @ truth + rng.normal(scale=0.05, size=m)
    prob = eki.EKIProblem(names=tuple(f"p{i}" for i in range(p)), prior_mean=np.zeros(p),
                          prior_cov=np.eye(p) * 0.25, obs=y, obs_cov=gamma, tikhonov=tikhonov)
    return prob, a, truth


def test_deterministic_step_is_kalman_update_with_sample_covariances():
    prob, a, _truth = _linear_problem()
    state = eki.initial_ensemble(prob, 40, seed=3)
    g = state.theta @ a.T
    new = eki.update(prob, state, g, perturb=False)
    th, gm = state.theta, g
    c_tg = (th - th.mean(0)).T @ (gm - gm.mean(0)) / (len(th) - 1)
    c_gg = (gm - gm.mean(0)).T @ (gm - gm.mean(0)) / (len(th) - 1)
    k = c_tg @ np.linalg.inv(c_gg + prob.gamma)
    want = th + (k @ (prob.obs[None, :] - gm).T).T
    np.testing.assert_allclose(new.theta, want, rtol=1e-12, atol=1e-12)


def test_large_ensemble_one_step_matches_linear_gaussian_posterior_mean():
    prob, a, _truth = _linear_problem()
    state = eki.initial_ensemble(prob, 20000, seed=5)
    new = eki.update(prob, state, state.theta @ a.T, seed=11)
    c0, gamma = prob.prior_cov, prob.gamma
    post_cov = np.linalg.inv(np.linalg.inv(c0) + a.T @ np.linalg.solve(gamma, a))
    post_mean = post_cov @ (a.T @ np.linalg.solve(gamma, prob.obs) + np.linalg.solve(c0, prob.prior_mean))
    np.testing.assert_allclose(new.theta.mean(0), post_mean, atol=0.01)
    np.testing.assert_allclose(np.cov(new.theta.T), post_cov, atol=0.002)


def test_tempered_teki_recovers_truth_and_reduces_misfit():
    prob, a, truth = _linear_problem(tikhonov=True)
    out = eki.run(prob, lambda th: th @ a.T, n_members=60, iterations=8, seed=7)
    assert np.max(np.abs(out.theta.mean(0) - truth)) < 0.05
    misfits = [h["normalised_misfit"] for h in out.history]
    assert misfits[-1] < 0.25 * misfits[0]


def test_bounds_clip_and_refusals():
    prob, a, _ = _linear_problem()
    bounded = eki.EKIProblem(**{**prob.__dict__, "lower": np.full(3, -0.05), "upper": np.full(3, 0.05)})
    state = eki.initial_ensemble(bounded, 30, seed=1)
    assert np.all(np.abs(state.theta) <= 0.05 + 1e-15)
    new = eki.update(bounded, state, state.theta @ a.T, seed=2)
    assert np.all(np.abs(new.theta) <= 0.05 + 1e-15)
    g = state.theta @ a.T
    g[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        eki.update(bounded, state, g)
    with pytest.raises(ValueError, match="do not match"):
        eki.update(bounded, state, (state.theta @ a.T)[:, :3])
    with pytest.raises(ValueError, match="non-finite"):
        eki.EKIProblem(names=("x",), prior_mean=np.zeros(1), prior_cov=np.eye(1), obs=np.array([np.inf]),
                       obs_cov=np.ones(1))
