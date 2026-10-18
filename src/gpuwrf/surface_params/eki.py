"""Ensemble Kalman inversion (EKI / Tikhonov-EKI) for low-dimensional surface-parameter calibration.

Host-only float64 NumPy (no JAX, no model imports): the ensemble matrices are N x p and N x m with
N (members) << 1e3, p (parameters) ~ 1e0..1e2, m (observations) ~ 1e2..1e4, so the update costs
milliseconds on the CPU between forecast windows (Astra critique: small ensemble matrices in fp64 on the CPU,
forecasts keep WRF REAL). The forward model is a callable ``theta (N, p) -> predictions (N, m)``; for the
model it runs one forecast window per member (``gpuwrf run --surface-controls``) and samples stations.

Update (Iglesias, Law & Stuart 2013, perturbed observations, step ``dt``):
    theta_j <- theta_j + C_tg (C_gg + Gamma/dt)^-1 (y + eta_j - g_j),   eta_j ~ N(0, Gamma/dt)
Tikhonov EKI (Chada, Stuart & Tong 2020) augments g with theta and y with the prior mean m0 (covariance C0),
which regularises toward the table prior and keeps the iteration from over-fitting repeated observations.
Parameters are log multipliers of the table values (``theta = log m``), clipped to physical bounds.
Limits (stated, not hidden): updates stay in the span of the initial ensemble (<= N-1 directions); spread is
not a posterior; repeated hourly observations are not independent — use tempering (dt < 1 per iteration with
sum dt = 1) or an inflated Gamma for temporal correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EKIProblem:
    names: tuple[str, ...]            # parameter labels, e.g. ("z0mvt:forest", "z0mvt:shrub")
    prior_mean: np.ndarray            # (p,) in theta space (log multiplier; 0 = table value)
    prior_cov: np.ndarray             # (p, p)
    obs: np.ndarray                   # (m,) observations y
    obs_cov: np.ndarray               # (m, m) Gamma (or (m,) diagonal)
    lower: np.ndarray | None = None   # (p,) theta bounds (log of the multiplier bounds)
    upper: np.ndarray | None = None
    tikhonov: bool = True

    def __post_init__(self):
        p, m = len(self.names), np.asarray(self.obs).size
        if np.shape(self.prior_mean) != (p,) or np.shape(self.prior_cov) != (p, p):
            raise ValueError("prior_mean/prior_cov must be (p,)/(p, p)")
        if np.shape(self.obs_cov) not in ((m,), (m, m)):
            raise ValueError("obs_cov must be (m,) or (m, m)")
        for name, arr in (("prior_mean", self.prior_mean), ("prior_cov", self.prior_cov),
                          ("obs", self.obs), ("obs_cov", self.obs_cov)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")

    @property
    def gamma(self) -> np.ndarray:
        g = np.asarray(self.obs_cov, dtype=np.float64)
        return np.diag(g) if g.ndim == 1 else g


@dataclass
class EKIState:
    theta: np.ndarray                 # (N, p)
    iteration: int = 0
    history: list = field(default_factory=list)


def initial_ensemble(problem: EKIProblem, n_members: int, *, seed: int) -> EKIState:
    if n_members < 2:
        raise ValueError("EKI needs at least 2 members")
    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal(np.asarray(problem.prior_mean, np.float64),
                                    np.asarray(problem.prior_cov, np.float64), size=n_members)
    return EKIState(theta=_clip(problem, theta))


def _clip(problem: EKIProblem, theta: np.ndarray) -> np.ndarray:
    lo = -np.inf if problem.lower is None else problem.lower
    hi = np.inf if problem.upper is None else problem.upper
    return np.clip(theta, lo, hi)


def _cov(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    da, db = a - a.mean(axis=0), b - b.mean(axis=0)
    return da.T @ db / (a.shape[0] - 1)


def update(problem: EKIProblem, state: EKIState, predictions: np.ndarray, *, dt: float = 1.0,
           seed: int | None = None, perturb: bool = True) -> EKIState:
    """One EKI/TEKI step from the members' predictions ``g (N, m)``; returns the new state."""
    theta = np.asarray(state.theta, dtype=np.float64)
    g = np.asarray(predictions, dtype=np.float64)
    n, m = g.shape[0], np.asarray(problem.obs).size
    if theta.shape[0] != n or g.shape != (n, m):
        raise ValueError(f"predictions {g.shape} do not match ensemble {theta.shape} and obs ({m},)")
    if not np.all(np.isfinite(g)):
        raise ValueError("non-finite member predictions (refusing the update)")
    if not 0.0 < dt <= 1.0:
        raise ValueError("dt must be in (0, 1]")
    y = np.asarray(problem.obs, dtype=np.float64)
    gamma = problem.gamma
    if problem.tikhonov:  # augment with the prior as pseudo-observations of theta
        g = np.concatenate([g, theta], axis=1)
        y = np.concatenate([y, np.asarray(problem.prior_mean, np.float64)])
        gamma = np.block([[gamma, np.zeros((m, theta.shape[1]))],
                          [np.zeros((theta.shape[1], m)), np.asarray(problem.prior_cov, np.float64)]])
    c_tg = _cov(theta, g)
    c_gg = _cov(g, g)
    noise = gamma / dt
    eta = (np.random.default_rng(seed).multivariate_normal(np.zeros(y.size), noise, size=n)
           if perturb else np.zeros((n, y.size)))
    innovation = y[None, :] + eta - g                      # (N, m')
    gain_rhs = np.linalg.solve(c_gg + noise, innovation.T)  # (m', N)
    new = theta + (c_tg @ gain_rhs).T
    new = _clip(problem, new)
    record = diagnostics(problem, theta, predictions)
    out = EKIState(theta=new, iteration=state.iteration + 1, history=state.history + [record])
    return out


def diagnostics(problem: EKIProblem, theta: np.ndarray, predictions: np.ndarray) -> dict:
    g = np.asarray(predictions, dtype=np.float64)
    resid = np.asarray(problem.obs, np.float64) - g.mean(axis=0)
    gamma = problem.gamma
    misfit = float(resid @ np.linalg.solve(gamma, resid)) / resid.size
    return {"theta_mean": theta.mean(axis=0).tolist(), "theta_std": theta.std(axis=0, ddof=1).tolist(),
            "normalised_misfit": misfit, "prediction_spread": float(g.std(axis=0, ddof=1).mean())}


def run(problem: EKIProblem, forward: Callable[[np.ndarray], np.ndarray], *, n_members: int, iterations: int,
        seed: int, tempering: bool = True, stop_misfit: float | None = None) -> EKIState:
    """Iterate EKI; with ``tempering`` the per-step dt = 1/iterations (sum of dt = 1: the data are used once)."""
    state = initial_ensemble(problem, n_members, seed=seed)
    dt = 1.0 / iterations if tempering else 1.0
    for k in range(iterations):
        g = forward(state.theta)
        if stop_misfit is not None and diagnostics(problem, state.theta, g)["normalised_misfit"] <= stop_misfit:
            state.history.append({**diagnostics(problem, state.theta, g), "stopped": True})
            return state
        state = update(problem, state, g, dt=dt, seed=seed + 1 + k)
    g = forward(state.theta)
    state.history.append(diagnostics(problem, state.theta, g))
    return state


def multipliers(theta: np.ndarray) -> np.ndarray:
    """Table multipliers from log-space parameters."""
    return np.exp(np.asarray(theta, dtype=np.float64))
