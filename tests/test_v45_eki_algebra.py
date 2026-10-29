"""Independent dense algebra, finite/support refusals and allocation guards for host EKI."""
from dataclasses import replace

import numpy as np
import pytest

from gpuwrf.surface_params import eki


def case(n=7, m=13, p=3, *, kind="diagonal", tikhonov=True, rank_deficient=False):
    rng = np.random.default_rng(48)
    theta = rng.normal(size=(n, p))
    g = rng.normal(size=(n, m))
    if rank_deficient:
        theta[:] = theta[0] + np.arange(n)[:, None] * .1
        g[:] = rng.normal(size=m) + np.arange(n)[:, None] * .04
    variances = np.linspace(.3, 1.5, m)
    gamma = np.diag(variances)
    if kind != "diagonal":
        b = rng.normal(size=(min(m, 4), min(m, 4)))
        gamma[:len(b), :len(b)] = b @ b.T + np.eye(len(b)) * .4
    if kind == "blocks":
        k = min(m, 4)
        blocks = (gamma[:k, :k],) + ((variances[k:],) if k < m else ())
        cov = eki.BlockDiagonalCovariance(blocks)
    else:
        cov = variances if kind == "diagonal" else gamma
    prior = np.array([[1.3, .1, .2], [.1, .8, -.1], [.2, -.1, 1.7]])[:p, :p]
    problem = eki.EKIProblem(tuple(f"p{k}" for k in range(p)), np.arange(p)*.1, prior,
                             rng.normal(size=m), cov, tikhonov=tikhonov)
    return problem, eki.EKIState(theta), g, gamma


def dense_reference(problem, state, g, gamma, dt, perturb):
    """Original sample-covariance update; independent construction, identical Gaussian draws."""
    theta = state.theta
    obs = problem.obs
    if problem.tikhonov:
        m, p = len(obs), theta.shape[1]
        augmented = np.zeros((m+p, m+p))
        augmented[:m, :m], augmented[m:, m:] = gamma, problem.prior_cov
        gamma = augmented
        g = np.column_stack((g, theta))
        obs = np.r_[obs, problem.prior_mean]
    noise = np.zeros_like(g)
    if perturb:
        z = np.random.default_rng(91).standard_normal(g.shape)
        noise = z @ np.linalg.cholesky(gamma).T / np.sqrt(dt)
    dg, dtheta = g - g.mean(0), theta - theta.mean(0)
    cgg, ctg = dg.T @ dg / (len(theta)-1), dtheta.T @ dg / (len(theta)-1)
    result = theta + (ctg @ np.linalg.solve(cgg + gamma/dt, (obs + noise - g).T)).T
    return np.clip(result, -np.inf if problem.lower is None else problem.lower,
                   np.inf if problem.upper is None else problem.upper)


@pytest.mark.parametrize("n,m", [(2, 1), (4, 13), (19, 5), (8, 8)])
@pytest.mark.parametrize("kind", ["diagonal", "dense", "blocks"])
@pytest.mark.parametrize("tikhonov,perturb,rank_deficient", [
    (False, False, False), (True, False, True), (True, True, False), (False, True, True)])
def test_v45_ensemble_matches_independent_dense(n, m, kind, tikhonov, perturb, rank_deficient):
    problem, state, g, gamma = case(n, m, kind=kind, tikhonov=tikhonov, rank_deficient=rank_deficient)
    dt = .17
    expected = dense_reference(problem, state, g, gamma, dt, perturb)
    actual = eki.update(problem, state, g, dt=dt, seed=91, perturb=perturb)
    reference_path = eki.update(problem, state, g, dt=dt, seed=91, perturb=perturb, method="dense")
    assert np.all(np.isfinite(actual.theta))
    np.testing.assert_allclose(actual.theta, expected, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(reference_path.theta, expected, rtol=2e-11, atol=2e-12)
    residual = problem.obs - g.mean(0)
    expected_misfit = residual @ np.linalg.solve(gamma, residual) / m
    assert actual.history[-1]["normalised_misfit"] == pytest.approx(expected_misfit, rel=2e-12)
    assert actual.iteration == 1 and state.iteration == 0 and state.history == []


def test_v45_covariance_whitening_and_sampling_orientation():
    problem, _, _, gamma = case(kind="blocks")
    cov = problem.obs_cov
    eye = np.eye(len(problem.obs))
    root = cov.color(eye)
    np.testing.assert_allclose(root.T @ root, gamma, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(cov.whiten(root), eye, rtol=1e-13, atol=1e-13)
    assert all(not x.flags.writeable for x in cov.blocks)


def test_v45_large_support_never_uses_dense_view(monkeypatch):
    n, m, p = 6, 1025, 3
    problem, state, g, _ = case(n, m, p)
    monkeypatch.setattr(eki.EKIProblem, "gamma", property(lambda _: pytest.fail("dense gamma used")))
    solve = np.linalg.solve
    shapes = []

    def bounded_solve(matrix, rhs):
        shapes.append(matrix.shape)
        assert matrix.shape[0] <= max(n, p), "observation-sized solve"
        return solve(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", bounded_solve)
    assert np.all(np.isfinite(eki.update(problem, state, g, seed=91).theta))
    assert (n, n) in shapes


def test_v45_large_ensemble_uses_thin_svd(monkeypatch):
    problem, state, g, gamma = case(1024, 4)
    svd = np.linalg.svd

    def thin(a, **kwargs):
        assert kwargs.get("full_matrices") is False
        return svd(a, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", thin)
    new = eki.update(problem, state, g, perturb=False)
    expected = dense_reference(problem, state, g, gamma, 1, False)
    np.testing.assert_allclose(new.theta, expected, rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("target", ["obs", "theta", "predictions", "prior_mean", "prior_cov"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_v45_missing_values_refused_before_reduction(target, bad):
    problem, state, g, _ = case()
    item = state.theta if target == "theta" else g if target == "predictions" else getattr(problem, target)
    item.flat[-1] = bad
    with pytest.raises(ValueError, match="non-finite"):
        eki.update(problem, state, g)
    with pytest.raises(ValueError, match="non-finite"):
        eki.diagnostics(problem, state.theta, g)


def test_v45_finite_guard_precedes_covariance_and_reduction(monkeypatch):
    problem, state, g, _ = case()
    g[0, 0] = np.nan

    def forbidden(*args):
        raise AssertionError("covariance/reductions reached with a missing member prediction")

    monkeypatch.setattr(eki, "_observation_covariance", forbidden)
    with pytest.raises(ValueError, match="non-finite"):
        eki.update(problem, state, g)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0., -1.])
def test_v45_invalid_covariances_refused(bad):
    problem, _, _, _ = case()
    variance = np.ones(len(problem.obs))
    variance[-1] = bad
    with pytest.raises(ValueError, match="non-finite|positive"):
        replace(problem, obs_cov=variance)


@pytest.mark.parametrize("bad", [np.array([[1., .4], [0., 1.]]), np.array([[1., 2.], [2., 1.]]),
                                 np.zeros((2, 2)), np.ones((2, 3)), np.ones((2, 2, 2)), np.array([])])
def test_v45_invalid_blocks_refused(bad):
    with pytest.raises(ValueError):
        eki.BlockDiagonalCovariance((bad,))


def test_v45_empty_or_wrong_support_is_not_silently_dropped():
    problem, state, g, _ = case()
    for bad_obs in (np.empty(0), problem.obs[:, None]):
        with pytest.raises(ValueError, match="1D observation"):
            replace(problem, obs=bad_obs)
    with pytest.raises(ValueError, match="block sizes"):
        replace(problem, obs_cov=eki.BlockDiagonalCovariance((np.ones(2),)))
    for bad in (g[:, :-1], g[0], np.empty((0, g.shape[1]))):
        with pytest.raises(ValueError, match="do not match"):
            eki.update(problem, state, bad)
    with pytest.raises(ValueError, match="at least 2"):
        eki.update(problem, eki.EKIState(state.theta[:1]), g[:1])


def test_v45_dense_guard_precedes_allocation(monkeypatch):
    m = eki.DENSE_OBSERVATION_LIMIT + 1
    problem, state, g, _ = case(4, m)

    def forbidden(*a, **kw):
        raise AssertionError("attempted dense allocation")

    monkeypatch.setattr(np, "diag", forbidden)
    monkeypatch.setattr(np, "zeros", forbidden)
    with pytest.raises(ValueError, match="limited"):
        _ = problem.gamma
    with pytest.raises(ValueError, match="limited"):
        eki.update(problem, state, g, method="dense")


def test_v45_dense_inputs_and_augmentation_are_bounded():
    problem, state, g, _ = case(4, eki.DENSE_OBSERVATION_LIMIT)
    with pytest.raises(ValueError, match="limited"):
        eki.update(problem, state, g, method="dense")  # prior pushes the augmented size over cap
    with pytest.raises(ValueError, match="limited"):
        eki.BlockDiagonalCovariance((np.eye(eki.DENSE_OBSERVATION_LIMIT + 1),))
    too_large = np.eye(eki.DENSE_OBSERVATION_LIMIT + 1)
    with pytest.raises(ValueError, match="limited"):
        replace(problem, obs=np.ones(len(too_large)), obs_cov=too_large)


@pytest.mark.parametrize("dt", [0, -1, 1.01, np.nan, np.inf])
def test_v45_invalid_tempering_refused(dt):
    problem, state, g, _ = case()
    with pytest.raises(ValueError, match="dt"):
        eki.update(problem, state, g, dt=dt)


def test_v45_bounds_seed_and_history():
    problem, state, g, _ = case()
    problem = replace(problem, lower=np.full(3, -.02), upper=np.full(3, .02))
    a = eki.update(problem, state, g, seed=4)
    b = eki.update(problem, state, g, seed=4)
    np.testing.assert_array_equal(a.theta, b.theta)
    assert np.max(np.abs(a.theta)) <= .02
    assert a.history is not state.history
