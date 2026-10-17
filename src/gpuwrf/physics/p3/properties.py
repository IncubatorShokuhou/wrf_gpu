"""WRF P3 rime consistency and three-moment spectral shape.

QZI wrapper conversion is deliberately withheld: XLA's subnormal treatment
does not yet reproduce original WRF (see proofs/s2_p3 and docs/P3_FOUNDATION.md).
"""

import jax.numpy as jnp

from .lookup import _real


def bulk_rime_density(qi, qir, bir):
    """calc_bulkRhoRime: return (adjusted rimed mass, volume, rime density).

    These are the original WRF consistency constraints, not NaN repair masks.
    The returned density is intentionally retained when the final qsmall test
    clears mass/volume, as in the original procedure.
    """
    qi, qir, bir = map(_real, (qi, qir, bir))
    active = bir >= _real(1.e-15)
    raw = qir / jnp.where(active, bir, _real(1))
    rho = jnp.where(active, jnp.clip(raw, _real(50), _real(900)), _real(0))
    limited = (raw < _real(50)) | (raw > _real(900))
    bir = jnp.where(active, jnp.where(limited, qir / jnp.where(active, rho, _real(1)), bir), _real(0))
    qir = jnp.where(active, qir, _real(0))
    cap = (qir > qi) & (rho > _real(0))
    qir = jnp.where(cap, qi, qir)
    bir = jnp.where(cap, qir / jnp.where(rho > _real(0), rho, _real(1)), bir)
    tiny = qir < _real(1.e-14)
    return jnp.where(tiny, _real(0), qir), jnp.where(tiny, _real(0), bir), rho


def compute_mu_3moment(mom0, mom3, mom6, mu_max=20.):
    """WRF's piecewise-polynomial approximation, without an invented mu cap.

    Caller precondition: mom3 > 1e-20 (WRF stops otherwise). Invalid moments
    return NaN, so a caller cannot silently admit an invalid column.
    """
    mom0, mom3, mom6, mu_max = map(_real, (mom0, mom3, mom6, mu_max))
    valid = (mom3 > _real(1.e-20)) & jnp.isfinite(mom0) & jnp.isfinite(mom3) & jnp.isfinite(mom6) & jnp.isfinite(mu_max)
    denom = jnp.where(valid, mom3, _real(1))
    g = (mom0 / denom) * (mom6 / denom)
    g2 = g ** 2
    mu = mu_max
    for lo, a, b, c in (
        (1.230, 3.0889e2, -9.0854e2, 6.8995e2),
        (1.405, 4.7552e1, -1.7958e2, 1.8126e2),
        (1.793, 4.3966, -2.6659e1, 4.5477e1),
        (2.946, 5.9070e-1, -5.7918, 1.6919e1),
        (4.200, 1.0730e-1, -1.7481, 8.4246),
        (7.123, 1.5900e-2, -4.8202e-1, 4.0108),
        (13.31, 3.3638e-3, -1.7152e-1, 2.0857),
    ):
        mu = jnp.where(g >= _real(lo), _real(a) * g2 + _real(b) * g + _real(c), mu)
    mu = jnp.where(g >= _real(20), _real(0), mu)
    return jnp.where(valid, mu, _real(jnp.nan))
