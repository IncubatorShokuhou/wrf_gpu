"""WRF radiation_driver cal_cldfra1 (icloud=1), WRF REAL arithmetic."""
from __future__ import annotations

import jax.numpy as jnp


def wrf_resolved_cloud_fraction(qv, qc, qi, qs, temperature, pressure, *, mp_physics=8,return_flags=False):
    """Resolved clouds for standard QC/QI/QS schemes and QC-only WRF options.

    Literal module_radiation_driver.F:3886–4111. Ferrier needs its separate
    F_ICE_PHY partition and is not a caller of this standard-species route.
    """
    f = jnp.float32
    qv, qc, qi, qs, t, p = (jnp.asarray(a, f) for a in (qv, qc, qi, qs, temperature, pressure))
    if int(mp_physics) == 0:
        zero=jnp.zeros_like(qv)
        return (zero,jnp.ones_like(qv,dtype=jnp.int32)) if return_flags else zero
    if int(mp_physics) in (5, 85, 95):
        raise NotImplementedError("Ferrier cal_cldfra1 requires F_ICE_PHY")
    tc = t - f(273.15)
    esw = f(1000.0) * f(0.61078) * jnp.exp(f(17.2693882) * tc / (t - f(35.86)))
    esi = f(1000.0) * f(0.61078) * jnp.exp(f(21.8745584) * tc / (t - f(7.66)))
    ep2 = f(287.0) / f(461.6)
    qvsw = ep2 * esw / (p - esw)
    qvsi = ep2 * esi / (p - esi)
    if int(mp_physics) in (1, 3):
        qcld = qc
        weight = jnp.where(t > f(273.15), f(0.0), f(1.0))
    else:
        ice = qi if int(mp_physics) in (50, 51) else qi + qs
        qcld = qi + qc if int(mp_physics) in (50, 51) else qi + qc + qs
        weight = ice / jnp.where(qcld < f(1.e-12), f(1.0), qcld)
    clear = qcld < f(1.e-12)
    weight = jnp.where(clear, f(0.0), weight)
    qvs = (f(1.0) - weight) * qvsw + weight * qvsi
    rh = qv / qvs
    subsat = jnp.maximum(f(1.e-10), f(1.0) * qvs - qv)
    denom = subsat ** f(0.49)
    arg = jnp.maximum(f(-6.9), -f(100.0) * qcld / denom)
    fraction = (jnp.maximum(f(1.e-10), rh) / f(1.0)) ** f(0.25) * (f(1.0) - jnp.exp(arg))
    fraction = jnp.where(fraction < f(0.01), f(0.0), fraction)
    fraction=jnp.where(clear, f(0.0), jnp.where(rh >= f(1.0), f(1.0), fraction))
    if return_flags:
        flag=jnp.where(clear,jnp.int32(1),jnp.where(rh>=f(1.),jnp.int32(2),jnp.int32(3)))
        return fraction,flag
    return fraction
