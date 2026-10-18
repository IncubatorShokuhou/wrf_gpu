"""LETKF algebra (Hunt, Kostelich & Szunyogh 2007, Physica D 230, 112-126).

One local analysis with N members and p (localised) observations:

    A   = (N-1)/rho * I + Yb^T R_loc^-1 Yb          (N x N, symmetric positive definite)
    Pa~ = A^-1                                       (eigh: A = Q diag(lam) Q^T)
    w   = Pa~ Yb^T R_loc^-1 d                        (mean weight, d = yo - mean H(xb))
    W   = [(N-1) Pa~]^(1/2)  (symmetric square root)
    xa_i = xb_mean + sum_j Xb_j (w_j + W_ji)        -> transform T = W + w 1^T

``Yb`` holds the member deviations of H(xb) from their mean. ``R_loc^-1`` is the
R-localised inverse error variance rho_loc / sigma_o^2 (zero for padding, so
padded slots contribute nothing). Arrays are batched over a leading axis of
local analyses. The NumPy path is the fp64 reference; ``xp=jax.numpy`` runs
the same algebra in JAX (CPU tests, GPU later). Nothing here touches the
forecast program.
"""
from __future__ import annotations

import numpy as np

_SQRT_10_3 = float(np.sqrt(10.0 / 3.0))


def gaspari_cohn(r):
    """Gaspari & Cohn (1999, QJRMS 125, eq. 4.10) fifth-order taper.

    ``r`` = distance / c with half-width c; 1 at r = 0, 0 for r >= 2.
    """
    r = np.abs(np.asarray(r, dtype=np.float64))
    out = np.zeros_like(r)
    inner = r <= 1.0
    outer = (r > 1.0) & (r < 2.0)
    a = r[inner]
    b = r[outer]
    out[inner] = (((-0.25 * a + 0.5) * a + 0.625) * a - 5.0 / 3.0) * a * a + 1.0
    out[outer] = ((((b / 12.0 - 0.5) * b + 0.625) * b + 5.0 / 3.0) * b - 5.0) * b + 4.0 - 2.0 / (3.0 * b)
    return out


def gc_halfwidth(length):
    """Half-width c of the GC taper equivalent to a Gaussian length scale (c = sqrt(10/3) L)."""
    return np.asarray(length, dtype=np.float64) * _SQRT_10_3


def letkf_transform(yb_pert, d, rinv, *, rho=1.0, xp=np):
    """Batched LETKF weights.

    yb_pert: (n, p, N) member deviations of H(xb) at the local observations.
    d:       (n, p) innovations yo - mean H(xb).
    rinv:    (n, p) localised inverse error variances (0 = unused slot).
    rho:     multiplicative prior inflation (>= 1): a scalar, or one value per local analysis (n,).

    Returns (wmean (n, N), W (n, N, N), T (n, N, N)) with T[:, j, i] the weight
    of background deviation j in analysis member i (T = W + wmean[:, :, None]).
    """
    yb_pert = xp.asarray(yb_pert)
    n_members = yb_pert.shape[-1]
    c = yb_pert * xp.asarray(rinv)[..., None]                      # (n, p, N) = R^-1 Yb
    a = xp.einsum("npi,npj->nij", c, yb_pert)                       # Yb^T R^-1 Yb
    if np.ndim(rho) == 0:
        a = a + ((n_members - 1) / float(rho)) * xp.eye(n_members, dtype=a.dtype)
    else:
        rho = xp.asarray(rho, dtype=a.dtype)
        if rho.shape != a.shape[:1]:
            raise ValueError(f"per-analysis inflation must have shape {a.shape[:1]}, got {rho.shape}")
        a = a + ((n_members - 1) / rho)[:, None, None] * xp.eye(n_members, dtype=a.dtype)
    a = 0.5 * (a + xp.swapaxes(a, -1, -2))
    lam, q = xp.linalg.eigh(a)
    ct_d = xp.einsum("npi,np->ni", c, xp.asarray(d))                 # Yb^T R^-1 d
    qt_ctd = xp.einsum("nji,nj->ni", q, ct_d)
    wmean = xp.einsum("nij,nj->ni", q, qt_ctd / lam)
    w = xp.einsum("nij,nj,nkj->nik", q, xp.sqrt((n_members - 1) / lam), q)
    return wmean, w, w + wmean[:, :, None]


def apply_transform(xb, t):
    """xa_i = mean(xb) + sum_j (xb_j - mean) T_ji, pointwise.

    xb: (N, *shape); t: (*shape, N, N) (or broadcastable). Returns (N, *shape).
    """
    xb = np.asarray(xb, dtype=np.float64)
    mean = xb.mean(axis=0)
    pert = xb - mean
    return mean[None] + np.einsum("j...,...ji->i...", pert, t)


def rtps(xb, xa, alpha):
    """Relaxation to prior spread (Whitaker & Hamill 2012), pointwise.

    Scales analysis deviations by 1 + alpha (sigma_b - sigma_a) / sigma_a.
    Points with sigma_a == 0 keep their analysis (factor 1, counted by caller).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("RTPS alpha must lie in [0, 1]")
    xb = np.asarray(xb, dtype=np.float64)
    xa = np.asarray(xa, dtype=np.float64)
    mean = xa.mean(axis=0)
    pa = xa - mean
    sb = xb.std(axis=0, ddof=1)
    sa = pa.std(axis=0, ddof=1)
    safe = sa > 0.0
    factor = np.where(safe, 1.0 + alpha * (sb - sa) / np.where(safe, sa, 1.0), 1.0)
    return mean[None] + pa * factor[None]


def etkf_global(xb, yb, yo, sigma_o, *, rho=1.0):
    """Global ETKF on a small dense state (test reference via the same algebra).

    xb: (N, n_state), yb: (N, p) = H(xb), yo/sigma_o: (p,). Returns xa (N, n_state).
    """
    yb = np.asarray(yb, dtype=np.float64)
    ybar = yb.mean(axis=0)
    _, _, t = letkf_transform((yb - ybar).T[None], (np.asarray(yo) - ybar)[None],
                              (1.0 / np.asarray(sigma_o, dtype=np.float64) ** 2)[None], rho=rho)
    return apply_transform(np.asarray(xb, dtype=np.float64), t[0])


def kalman_reference(xb, h, yo, sigma_o):
    """Independent state-space Kalman update with the ensemble covariance.

    Returns (analysis mean, analysis covariance) from Pb = X X^T/(N-1),
    K = Pb H^T (H Pb H^T + R)^-1, xa = xb + K (yo - H xb), Pa = (I - K H) Pb.
    """
    xb = np.asarray(xb, dtype=np.float64)
    n_members = xb.shape[0]
    mean = xb.mean(axis=0)
    x = (xb - mean).T
    pb = x @ x.T / (n_members - 1)
    r = np.diag(np.asarray(sigma_o, dtype=np.float64) ** 2)
    k = pb @ h.T @ np.linalg.inv(h @ pb @ h.T + r)
    return mean + k @ (np.asarray(yo) - h @ mean), (np.eye(len(mean)) - k @ h) @ pb
