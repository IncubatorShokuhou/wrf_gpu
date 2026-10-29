"""Ensemble Kalman inversion (EKI / Tikhonov-EKI) for low-dimensional surface-parameter calibration.

Host-only float64 NumPy (no JAX, no model imports). Diagonal or bounded block-diagonal observation
errors are whitened without an m x m matrix. The solve is N x N (members), or a thin SVD when there
are more members than augmented observations. Storage is O(N*(m+p) + N*p + sum(block_size**2));
the normal N <= m+p path costs O((m+p)*N**2 + N**3). The forward model is a callable
``theta (N, p) -> predictions (N, m)``; for the
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


# Dense inputs/reference solves are for small problems only, including TEKI augmentation.
DENSE_OBSERVATION_LIMIT = 512
# Capability consumed by cloudscore admission; only enable after this implementation is reviewed/merged.
ENSEMBLE_SPACE_UPDATE = True


def _dense_guard(size: int):
    if size > DENSE_OBSERVATION_LIMIT:
        raise ValueError(f"dense observation covariance/solve limited to {DENSE_OBSERVATION_LIMIT} observations; "
                         "use a variance vector or bounded BlockDiagonalCovariance")


@dataclass(frozen=True)
class BlockDiagonalCovariance:
    """Contiguous observation groups, in the exact order of predictions/observations.

    A block is a positive variance vector or a symmetric positive-definite matrix (<=512 rows).
    Permute keys AND all member vectors together upstream for noncontiguous correlation groups.
    This interface supplies a covariance, not an automatic correlation-length/independence claim.
    """

    blocks: tuple[np.ndarray, ...]
    _factors: tuple[np.ndarray, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        blocks, factors = [], []
        if not self.blocks:
            raise ValueError("observation covariance needs nonempty blocks")
        for raw in self.blocks:
            block = np.array(raw, dtype=np.float64, copy=True)
            if block.ndim not in (1, 2) or not block.size:
                raise ValueError("covariance block must be a nonempty variance vector or square matrix")
            if not np.all(np.isfinite(block)):
                raise ValueError("obs_cov contains non-finite values")
            if block.ndim == 1:
                if np.any(block <= 0):
                    raise ValueError("observation variances must be positive")
                factor = np.sqrt(block)
            else:
                if block.shape[0] != block.shape[1]:
                    raise ValueError("covariance block must be square")
                _dense_guard(block.shape[0])
                if not np.array_equal(block, block.T):
                    raise ValueError("covariance block must be symmetric")
                try:
                    factor = np.linalg.cholesky(block)
                except np.linalg.LinAlgError as exc:
                    raise ValueError("covariance block must be positive definite") from exc
            block.flags.writeable = False
            factor.flags.writeable = False
            blocks.append(block)
            factors.append(factor)
        object.__setattr__(self, "blocks", tuple(blocks))
        object.__setattr__(self, "_factors", tuple(factors))

    @property
    def size(self):
        return sum(block.shape[0] for block in self.blocks)

    def whiten(self, rows):
        """Apply L^-1 to each column-vector (rows store its transpose), Gamma = L L.T."""
        out = np.empty_like(rows, dtype=np.float64)
        start = 0
        for factor in self._factors:
            stop = start + factor.shape[0]
            part = rows[..., start:stop]
            out[..., start:stop] = (part / factor if factor.ndim == 1
                                    else np.linalg.solve(factor, part.T).T)
            start = stop
        return out

    def color(self, rows):
        out = np.empty_like(rows, dtype=np.float64)
        start = 0
        for factor in self._factors:
            stop = start + factor.shape[0]
            part = rows[..., start:stop]
            out[..., start:stop] = part * factor if factor.ndim == 1 else part @ factor.T
            start = stop
        return out

    def dense(self):
        _dense_guard(self.size)  # check BEFORE allocating, also for a diagonal input
        out = np.zeros((self.size, self.size), dtype=np.float64)
        start = 0
        for block in self.blocks:
            stop = start + block.shape[0]
            out[start:stop, start:stop] = np.diag(block) if block.ndim == 1 else block
            start = stop
        return out


def _observation_covariance(value, m):
    if isinstance(value, BlockDiagonalCovariance):
        cov = value
    else:
        if np.shape(value) not in ((m,), (m, m)):
            raise ValueError("obs_cov must be (m,), small (m, m), or BlockDiagonalCovariance")
        if np.ndim(value) == 2:
            _dense_guard(m)
        cov = BlockDiagonalCovariance((value,))
    if cov.size != m:
        raise ValueError("obs_cov block sizes do not match observations")
    return cov


@dataclass(frozen=True)
class EKIProblem:
    names: tuple[str, ...]            # parameter labels, e.g. ("z0mvt:forest", "z0mvt:shrub")
    prior_mean: np.ndarray            # (p,) in theta space (log multiplier; 0 = table value)
    prior_cov: np.ndarray             # (p, p)
    obs: np.ndarray                   # (m,) observations y
    obs_cov: np.ndarray | BlockDiagonalCovariance  # variances, bounded blocks, or small dense Gamma
    lower: np.ndarray | None = None   # (p,) theta bounds (log of the multiplier bounds)
    upper: np.ndarray | None = None
    tikhonov: bool = True

    def __post_init__(self):
        p, m = len(self.names), np.asarray(self.obs).size
        if p < 1 or np.shape(self.obs) != (m,) or m == 0:
            raise ValueError("EKI needs parameters and a nonempty 1D observation vector with fixed support")
        if np.shape(self.prior_mean) != (p,) or np.shape(self.prior_cov) != (p, p):
            raise ValueError("prior_mean/prior_cov must be (p,)/(p, p)")
        for name, arr in (("prior_mean", self.prior_mean), ("prior_cov", self.prior_cov),
                          ("obs", self.obs)):
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
        _observation_covariance(self.obs_cov, m)
        # Prior is a small parameter-space covariance, also used as a TEKI observation block.
        BlockDiagonalCovariance((self.prior_cov,))
        lo = np.full(p, -np.inf) if self.lower is None else np.asarray(self.lower, np.float64)
        hi = np.full(p, np.inf) if self.upper is None else np.asarray(self.upper, np.float64)
        if (lo.shape != (p,) or hi.shape != (p,) or np.any(np.isnan(lo)) or np.any(np.isnan(hi))
                or np.any(np.isposinf(lo)) or np.any(np.isneginf(hi)) or np.any(lo > hi)):
            raise ValueError("bounds must be ordered (p,) vectors without NaN")

    @property
    def gamma(self) -> np.ndarray:
        """Explicit, guarded dense reference view; production updates never use this property."""
        return _observation_covariance(self.obs_cov, np.asarray(self.obs).size).dense()


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
           seed: int | None = None, perturb: bool = True, method: str = "ensemble") -> EKIState:
    """One EKI/TEKI step; ``method='dense'`` is an explicit small-m reference only.

    Missing observations/member predictions are refused, never silently dropped or filled. Freeze
    common support in the observation operator. Seeded perturbations use a block Cholesky root;
    their distribution is unchanged, their bytes differ from the former dense-SVD RNG path.
    """
    theta, g = _validated_members(problem, state.theta, predictions)
    n, m = g.shape
    if not np.isfinite(dt) or not 0.0 < dt <= 1.0:
        raise ValueError("dt must be in (0, 1]")
    if method not in ("ensemble", "dense"):
        raise ValueError("EKI method must be ensemble or dense")
    if method == "dense":
        _dense_guard(m + (theta.shape[1] if problem.tikhonov else 0))
    y = np.asarray(problem.obs, dtype=np.float64)
    cov = _observation_covariance(problem.obs_cov, m)
    if problem.tikhonov:  # augment with the prior as pseudo-observations of theta
        g = np.concatenate([g, theta], axis=1)
        y = np.concatenate([y, np.asarray(problem.prior_mean, np.float64)])
        cov = BlockDiagonalCovariance((*cov.blocks, np.asarray(problem.prior_cov, np.float64)))
    white_noise = np.random.default_rng(seed).standard_normal(g.shape) if perturb else None
    if method == "dense":
        noise = cov.dense() / dt
        eta = 0 if white_noise is None else cov.color(white_noise) / np.sqrt(dt)
        innovation = y[None, :] + eta - g
        gain_rhs = np.linalg.solve(_cov(g, g) + noise, innovation.T)
        new = theta + (_cov(theta, g) @ gain_rhs).T
    else:
        # A = Gamma^(-1/2) anomalies * sqrt(dt/(N-1)), stored as rows.
        a = cov.whiten(g - g.mean(axis=0)) * np.sqrt(dt / (n - 1))
        innovation = cov.whiten(y[None, :] - g) * np.sqrt(dt)
        if white_noise is not None:
            innovation += white_noise
        t = (theta - theta.mean(axis=0)) / np.sqrt(n - 1)
        if n <= g.shape[1]:
            weights = np.linalg.solve(np.eye(n) + a @ a.T, a @ innovation.T)
            new = theta + weights.T @ t
        else:
            # Large ensembles with few observations: never allocate N x N either.
            u, s, vt = np.linalg.svd(a, full_matrices=False)
            new = theta + ((innovation @ vt.T) * (s / (1 + s*s))) @ (u.T @ t)
    if not np.all(np.isfinite(new)):
        raise ValueError("non-finite EKI update (refusing the state)")
    new = _clip(problem, new)
    record = diagnostics(problem, theta, predictions)
    out = EKIState(theta=new, iteration=state.iteration + 1, history=state.history + [record])
    return out


def diagnostics(problem: EKIProblem, theta: np.ndarray, predictions: np.ndarray) -> dict:
    theta, g = _validated_members(problem, theta, predictions)
    resid = np.asarray(problem.obs, np.float64) - g.mean(axis=0)
    white = _observation_covariance(problem.obs_cov, resid.size).whiten(resid)
    misfit = float(white @ white) / resid.size
    if not np.isfinite(misfit):
        raise ValueError("non-finite EKI misfit")
    return {"theta_mean": theta.mean(axis=0).tolist(), "theta_std": theta.std(axis=0, ddof=1).tolist(),
            "normalised_misfit": misfit, "prediction_spread": float(g.std(axis=0, ddof=1).mean())}


def _validated_members(problem, theta, predictions):
    theta = np.asarray(theta, dtype=np.float64)
    g = np.asarray(predictions, dtype=np.float64)
    m, p = np.asarray(problem.obs).size, len(problem.names)
    if theta.ndim != 2 or theta.shape[1] != p or g.ndim != 2 or g.shape != (theta.shape[0], m):
        raise ValueError(f"predictions {g.shape} do not match ensemble {theta.shape} and obs ({m},)")
    if theta.shape[0] < 2:
        raise ValueError("EKI needs at least 2 members")
    if not all(np.all(np.isfinite(x)) for x in (theta, g, problem.obs, problem.prior_mean, problem.prior_cov)):
        raise ValueError("non-finite ensemble, observations or member predictions (refusing the update)")
    return theta, g


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
