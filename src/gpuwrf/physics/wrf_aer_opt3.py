"""WRF V4.7.1 ``aer_opt=3``: Thompson water/ice-friendly aerosol (mp28 QNWFA/QNIFA) optics for RRTMG SW (REAL).

At every radiation call module_radiation_driver.F (:1652-1694) calls ``gt_aod`` (:5507-5744) on the domain's
p_hyd / dz8w / t_phy / QV / QNWFA / QNIFA (wif_input_opt=1: no black carbon) for the per-layer 550 nm AOD
``taod5503d = MAX(1e-3, AOD_wfa + AOD_ifa)`` and its column sum ``taod5502d``; then (:2374-2392)
``calc_aerosol_rrtmg_sw`` (module_ra_aerosol.F:397) with the driver PARAMETERs taer_type=1 (rural),
taer_aod550_opt=2, taer_angexp_opt=3, taer_ssa_opt=3, taer_asy_opt=3 turns it into band optics:
``calc_relative_humidity`` (Bolton, every level), ``calc_spectral_aod_rrtmg_sw`` 3-D branch
(tauaer3d = Lagrange-in-RH(raod_lut) x taod5503d), ``calc_spectral_{ssa,asy}_rrtmg_sw`` (Lagrange-in-RH).
RRTMG_SWRAD (aer_opt 3 -> iaer=10, module_ra_rrtmg_sw.F:11334-11350) passes them per layer and band with the
extra top layer at tau 0 / ssa 1 / asy 0 into the same spcvmc_sw clear-sky optics as aer_opt=1.
Every expression keeps WRF's REAL operation order.  Tables: physics/wrf_aer_opt3_tables.py (generated).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import wrf_aer_opt3_tables as tab
from gpuwrf.physics.thompson_saturation import saturation_mixing_ratio_liquid

F = jnp.float32
TAER_TYPE = 1  # module_radiation_driver.F:1015 (rural); the other driver PARAMETERs select the 3-D branches
_RHS = np.asarray(tab.RHS, np.float32)
_RAOD = np.asarray(tab.RAOD_LUT, np.float32)[TAER_TYPE - 1]   # (14 bands, 8 RH)
_SSA = np.asarray(tab.SSA_LUT, np.float32)[TAER_TYPE - 1]
_ASY = np.asarray(tab.ASY_LUT, np.float32)[TAER_TYPE - 1]
_GT_RH = np.asarray(tab.GT_AOD_RH, np.float32)
_GT_LOOK = np.asarray(tab.GT_AOD_LOOKUP, np.float32)          # (8 RH, 4 T, 3 kinds)
N_BANDS = 14


def _nint(x):
    """Fortran NINT (half away from zero) of a REAL, as int32."""

    return (jnp.sign(x) * jnp.floor(jnp.abs(x) + F(0.5))).astype(jnp.int32)


def wrf_gt_aod(p_pa, t_k, qv, nwfa, nifa, dz8w):
    """``gt_aod`` (wif_input_opt=1): per-layer 550 nm AOD from QNWFA/QNIFA; every input ``(..., nz)`` REAL."""

    p, t, qv, nwfa, nifa, dz = (jnp.asarray(a, F) for a in (p_pa, t_k, qv, nwfa, nifa, dz8w))
    rhoa = p / (F(287.0) * t)
    t_idx = jnp.clip(_nint(F(10.999) - F(0.0333) * t), 1, 4)
    qvsat = saturation_mixing_ratio_liquid(p, t)
    rh = jnp.minimum(F(99.1), jnp.maximum(F(10.1), qv / qvsat * F(100.0)))
    rh_arr = jnp.asarray(_GT_RH)

    def bracket(idx):
        rh_d = rh - rh_arr[jnp.clip(idx, 1, 8) - 1]
        low = jnp.where(rh_d < 0, idx - 1, idx)
        high = jnp.where(rh_d < 0, idx, idx + 1)
        over = high > 8
        return jnp.where(over, 7, low), jnp.where(over, 8, high)

    mid1, mid2 = bracket(_nint(F(0.1) * rh + F(-4.0)))
    top1, top2 = bracket(jnp.minimum(8, _nint(F(0.2) * rh + F(-12.0))))
    lo = rh < F(60.0)
    mid = (rh >= F(60.0)) & (rh < F(80.0))
    idx1 = jnp.where(lo, 1, jnp.where(mid, mid1, top1))
    idx2 = jnp.where(lo, 2, jnp.where(mid, mid2, top2))
    r1, r2 = rh_arr[idx1 - 1], rh_arr[idx2 - 1]
    hundred = F(100.0)
    rh_f = jnp.maximum(F(0.0), jnp.minimum(F(1.0), (rh / (hundred - rh) - r1 / (hundred - r1))
                                           / (r2 / (hundred - r2) - r1 / (hundred - r1))))
    look = jnp.asarray(_GT_LOOK)

    def bext(kind):
        a, b = look[idx1 - 1, t_idx - 1, kind], look[idx2 - 1, t_idx - 1, kind]
        return a + (b - a) * rh_f

    ntemp = jnp.maximum(F(1.0), jnp.minimum(F(99999.0e6), nwfa))
    aod_wfa = bext(0) * ntemp * dz * rhoa
    ntemp = jnp.maximum(F(0.01), jnp.minimum(F(9999.0e6), nifa))
    aod_ifa = bext(1) * ntemp * dz * rhoa
    return jnp.maximum(F(1.0e-3), aod_wfa + aod_ifa)


def wrf_relative_humidity(p_pa, t_k, qv):
    """``calc_relative_humidity`` (module_ra_aerosol.F:1437): Bolton RH in percent, clipped to [0, 99]."""

    p, t, qv = (jnp.asarray(a, F) for a in (p_pa, t_k, qv))
    tc = t - F(273.15)
    rv = jnp.maximum(F(0.0), qv)
    es = F(6.112) * jnp.exp((F(17.6) * tc) / (tc + F(243.5)))
    e = F(0.01) * rv * p / (rv + F(0.62197))
    return jnp.minimum(F(99.0), jnp.maximum(F(0.0), F(100.0) * e / es))


def _lagrange(rh, lut, scale=None):
    """WRF's windowed Lagrange interpolation in RH (calc_spectral_*_rrtmg_sw), literal loop order.

    ``rh`` ``(...)``, ``lut`` ``(14, 8)``; returns ``(..., 14)``.  ``ii`` = first node with ``rh <= rhs(ii)``,
    window ``max(1, ii-3) .. min(8, ii+2)``; ``scale`` multiplies each term after the table value (AOD branch).
    """

    rhs = [F(v) for v in _RHS]
    ii = 1 + sum((rh > r).astype(jnp.int32) for r in rhs)
    imin = jnp.maximum(1, ii - 3)
    imax = jnp.minimum(8, ii + 2)
    lut = jnp.asarray(lut, F)
    acc = jnp.zeros(rh.shape + (N_BANDS,), F)
    for jj in range(1, 9):
        in_j = (imin <= jj) & (jj <= imax)
        lj = jnp.ones_like(rh)
        for k in range(1, 9):
            if k == jj:
                continue
            in_k = (imin <= k) & (k <= imax)
            lj = jnp.where(in_k, lj * (rh - rhs[k - 1]) / (rhs[jj - 1] - rhs[k - 1]), lj)
        term = lj[..., None] * lut[:, jj - 1]
        if scale is not None:
            term = term * scale[..., None]
        acc = jnp.where(in_j[..., None], acc + term, acc)
    return acc


def wrf_aer_opt3_band_optics(p_pa, t_k, qv, nwfa, nifa, dz8w):
    """aer_opt=3 RRTMG SW band optics ``(tau, ssa, asy)`` each ``(..., nz, 14)`` plus ``taod5503d`` ``(..., nz)``."""

    taod = wrf_gt_aod(p_pa, t_k, qv, nwfa, nifa, dz8w)
    rh = wrf_relative_humidity(p_pa, t_k, qv)
    tau = _lagrange(rh, _RAOD, scale=taod)
    return tau, _lagrange(rh, _SSA), _lagrange(rh, _ASY), taod


__all__ = ["TAER_TYPE", "wrf_aer_opt3_band_optics", "wrf_gt_aod", "wrf_relative_humidity"]
