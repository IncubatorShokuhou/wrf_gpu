"""WRF MYNN-EDMF subgrid (BL) cloud diagnostics: ``mym_condensation`` CASE(2).

Faithful JAX transcription of the WRF v4 pristine
``phys/module_bl_mynnedmf.F`` pieces that produce the MYNN subgrid cloud state
(``qc_bl``/``qi_bl``/``cldfra_bl``) for the operational configuration:

  bl_mynn_cloudpdf = 2   (WRF Registry default; Chaboureau-Bechtold 2002 PDF,
                          sigma from the PROGNOSED total-water variance ``qsq``)
  bl_mynn_closure  = 2.6 (WRF Registry default; ``qsq`` prognostic, tsq/cov
                          level-2 diagnostic)
  bl_mynn_edmf     = 1   (DMP mass-flux shallow-cu overwrite of qc_bl/cldfra_bl)
  icloud_bl        = 1   (WRF Registry default; radiation consumes the BL cloud)

Scope notes (operational-config dead code intentionally omitted, with anchors):

* ``vt``/``vq`` buoyancy-flux functions: computed by WRF ``mym_condensation``
  but DEAD in this configuration -- ``mym_level2`` uses the ``thlv`` gradient
  form (module-level ``use_buoy=.false.``, module_bl_mynnedmf.F:327/1530) and
  ``mym_length`` CASE(1) uses ``vflx = fltv`` (line 224; the vt/vq form is
  commented out). The SGS-cloud buoyancy effect enters ONLY through the driver
  ``thlv1`` rebuild (lines 1005-1009) with ``max(qc_bl1, sqc1)`` /
  ``max(qi_bl1, sqi1+sqs1)``, which `mynn_pbl` now applies.
* ``sgm`` is returned to WRF ``DMP_mf`` but its only use there is commented out
  (``!sigq = SQRT(sigq**2 + sgm1(k)**2)``), so it is not exposed.
* SPP stochastic perturbations off (``spp_pbl=0``): ``qw_pert = qw``.

WRF file:line anchors are cited inline against
/home/user/src/wrf_pristine/WRF/phys/module_bl_mynnedmf.F.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.real_constfold import R4, real_constant
from gpuwrf.physics.mynn_edmf import _qsat_blend

_DMP_KTOP_BOUND = os.environ.get("GPUWRF_MYNN_DMP_KTOP_BOUND", "0") == "1"


def sgs_cloud_enabled() -> bool:
    """Single source of truth for the v0.15 MYNN SGS-cloud chain gate.

    Default ON (the WRF-faithful path). ``GPUWRF_MYNN_SGS_CLOUD=0`` rolls back
    BOTH the MYNN-side chain (condensation/qsq/thlv) and the icloud_bl
    radiation merge together, so a disabled chain never zeroes the radiation
    cloud fraction with an all-zero ``cldfra_bl``.
    """

    value = os.environ.get("GPUWRF_MYNN_SGS_CLOUD")
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}

# ---- WRF constants (share/module_model_constants.F + module_bl_mynnedmf_common.F)
R_D = 287.0
R_V = 461.6
CP = 7.0 * R_D / 2.0
CPV = 4.0 * R_V            # 1846.4
CLIQ = 4190.0
CICE = 2106.0
XLV = 2.5e6
XLS = 2.85e6
T0C = 273.15
TICE = 240.0               # module_bl_mynnedmf_common.F:68
TLIQ = 269.0               # module_bl_mynnedmf.F:310
XLVCP = XLV / CP

# mym_condensation CASE(2) parameters (module_bl_mynnedmf.F:3425-3434)
QLIM_SFC = 0.007
QLIM_PBL = 0.020
QLIM_TRP = 0.025
CLIM_SFC = 0.010
CLIM_PBL = 0.025
CLIM_TRP = 0.030
RHCRIT = 0.83
RHMAX = 1.10

# DMP_mf shallow-cu overwrite (module_bl_mynnedmf.F:5741)
CF_THRESH = 0.5


# WRF REAL folds (module_bl_mynnedmf.F90 xl_blend :7688/:7693, mym_condensation q1_rh :3669/:3677), w44-constfold.
# cpv = 4.*r_v and t0c = svpt0 come from module_model_constants / module_bl_mynnedmf_common (REAL PARAMETERs).
_CPV_M_CICE_REAL = R4(4.0) * R4(R_V) - R4(CICE)
_T0C_M_TICE_REAL = R4(T0C) - R4(TICE)
_ONE_M_RHCRIT_REAL = R4(1.0) - R4(RHCRIT)


def xl_blend(t):
    """WRF ``xl_blend(t)``: temperature-blended latent heat (liquid/ice)."""

    xlvt = XLV + (CPV - CLIQ) * (t - T0C)
    dtype = jnp.result_type(t)
    xlst = XLS + real_constant(_CPV_M_CICE_REAL, CPV - CICE, dtype) * (t - T0C)
    chi = (T0C - t) / real_constant(_T0C_M_TICE_REAL, T0C - TICE, dtype)
    blended = (1.0 - chi) * xlvt + chi * xlst
    return jnp.where(t >= T0C, xlvt, jnp.where(t <= TICE, xlst, blended))


def _k_tropo(th, p):
    """Tropopause level estimate (module_bl_mynnedmf.F:3458-3471, G. Thompson).

    Fortran scans ``k = kte-3, kts, -1`` and exits at the FIRST satisfying k
    (the HIGHEST satisfying level); ``k_tropo = MAX(kts+2, k+2)``.  0-based:
    ``k_tropo0 = max(2, k0 + 2)`` with ``k0 = -1`` when no level satisfies.
    Returns an integer array broadcast over the batch dims.
    """

    nz = th.shape[-1]
    ht = 44307.692 * (1.0 - (p / 101325.0) ** 0.190)
    theta1 = th[..., : nz - 2]
    theta2 = th[..., 2:]
    ht1 = ht[..., : nz - 2]
    ht2 = ht[..., 2:]
    lapse_ok = (theta2 - theta1) / (ht2 - ht1) < (10.0 / 1500.0)
    cond = lapse_ok & (ht1 < 19000.0) & (ht1 > 4000.0)
    # restrict to k0 <= nz-4 (Fortran kte-3)
    idx = jnp.arange(nz - 2)
    cond = cond & (idx <= nz - 4)
    k0 = jnp.max(jnp.where(cond, idx, -1), axis=-1)
    return jnp.maximum(2, k0 + 2)


def mym_condensation_cloudpdf2(
    *,
    theta,       # (..., nz) DRY potential temperature (WRF th1)
    p,           # (..., nz) pressure (Pa)
    exner,       # (..., nz)
    dz,          # (..., nz)
    qw,          # (..., nz) total water specific content (sqw)
    qc,          # (..., nz) resolved cloud water specific content (sqc)
    qi,          # (..., nz) resolved cloud ice specific content (sqi)
    qs,          # (..., nz) resolved snow specific content (sqs)
    qsq,         # (..., nz) prognosed total-water variance <q'w q'w>
    pblh,        # (...,)    GET_PBLH boundary-layer height (m)
    spp_pattern=None,
):
    """WRF ``mym_condensation`` CASE(2) subgrid cloud diagnosis.

    Returns ``(qc_bl, qi_bl, cldfra_bl)`` -- grid-mean SGS cloud water/ice and
    cloud fraction (module_bl_mynnedmf.F:3600-3760, the Chaboureau-Bechtold
    2002 statistical scheme with sigma from the prognosed ``qsq``).
    """

    nz = theta.shape[-1]
    zw = jnp.concatenate(
        (jnp.zeros_like(dz[..., :1]), jnp.cumsum(dz, axis=-1)[..., :-1]), axis=-1
    )
    zagl = zw + 0.5 * dz                       # mass-level height AGL
    pblh1 = pblh[..., None]
    pblh2 = jnp.maximum(10.0, pblh1)

    t = theta * exner
    qsat_tk = _qsat_blend(t, p)
    rh = jnp.maximum(jnp.minimum(RHMAX, qw / jnp.maximum(1.0e-10, qsat_tk)), 0.001)

    qmq = qw - qsat_tk                         # qw_pert = qw (spp off)
    if spp_pattern is not None:
        from gpuwrf.physics.stochastic_spp import total_water
        qmq = total_water(qw,spp_pattern)-qsat_tk

    # sigma from the prognosed variance, with dz inflation + z-dependent floors
    r3sq = jnp.maximum(qsq, 0.0)
    sgm = jnp.maximum(1.0e-13, jnp.sqrt(r3sq))
    sgm = jnp.minimum(sgm, qw / 3.0)
    wt = jnp.minimum(1.0, jnp.maximum(0.0, dz - 100.0) / 500.0)
    sgm = sgm + sgm * 0.2 * wt

    # cloud-fraction sigma floor (clim_*)
    wt = jnp.minimum(1.0, jnp.maximum(0.0, zagl - (pblh2 + 10.0)) / 300.0)
    clim = CLIM_PBL * (1.0 - wt) + CLIM_TRP * wt
    zsl = jnp.minimum(150.0, jnp.maximum(50.0, 0.1 * pblh2))
    wt = jnp.minimum(1.0, jnp.maximum(0.0, zagl - zsl) / 200.0)
    clim = CLIM_SFC * (1.0 - wt) + clim * wt
    sgmc = jnp.maximum(sgm, qw * clim)
    sgmc = jnp.maximum(1.0e-13, sgmc)
    sgmc = jnp.where(qmq >= 0.0, jnp.maximum(0.02 * qw, sgmc), sgmc)

    q1 = qmq / sgmc

    # falling/settling rh hacks (sequential, order preserved; lines 3680-3697)
    wt2_rh = jnp.minimum(1.0, jnp.maximum(0.0, zagl - pblh2) / 300.0)
    aloft = zagl > pblh2
    ice_tot = qi + qs
    has_ice = (ice_tot > 1.0e-9) & aloft
    rh_hack_i = jnp.minimum(
        RHMAX, RHCRIT + wt2_rh * 0.045 * (9.0 + jnp.log10(jnp.maximum(ice_tot, 1.0e-30)))
    )
    rh = jnp.where(has_ice, jnp.maximum(rh, rh_hack_i), rh)
    q1_rh_i = -3.0 + 3.0 * (rh - RHCRIT) / real_constant(_ONE_M_RHCRIT_REAL, 1.0 - RHCRIT, jnp.result_type(rh))
    q1 = jnp.where(has_ice, jnp.maximum(q1_rh_i, q1), q1)

    has_qc = (qc > 1.0e-6) & aloft
    rh_hack_c = jnp.minimum(
        RHMAX, RHCRIT + wt2_rh * 0.08 * (6.0 + jnp.log10(jnp.maximum(qc, 1.0e-30)))
    )
    rh = jnp.where(has_qc, jnp.maximum(rh, rh_hack_c), rh)
    q1_rh_c = -3.0 + 3.0 * (rh - RHCRIT) / real_constant(_ONE_M_RHCRIT_REAL, 1.0 - RHCRIT, jnp.result_type(rh))
    q1 = jnp.where(has_qc, jnp.maximum(q1_rh_c, q1), q1)

    q1k = q1

    # cloud fraction (lines 3699-3713); cldfra_rh intentionally zero in WRF
    wt2 = jnp.minimum(1.0, jnp.maximum(0.0, (zagl - (pblh1 - 100.0)) / 200.0))
    cldfra_qsq0 = jnp.clip(0.5 + 0.35 * jnp.arctan(4.1 * q1k), 0.0, 1.0)
    cldfra_qsq1 = jnp.clip(0.5 + 0.37 * jnp.arctan(2.1 * (q1k + 0.4)), 0.0, 1.0)
    cldfra_bl = cldfra_qsq0 * (1.0 - wt2) + cldfra_qsq1 * wt2

    # hydrometeor sigma floor (qlim_*) and grid-mean SGS condensate (CB02 Eq. 8)
    wt = jnp.minimum(1.0, jnp.maximum(0.0, zagl - (pblh2 + 10.0)) / 300.0)
    qlim = QLIM_PBL * (1.0 - wt) + QLIM_TRP * wt
    zsl = jnp.minimum(150.0, jnp.maximum(50.0, 0.1 * pblh2))
    wt = jnp.minimum(1.0, jnp.maximum(0.0, zagl - zsl) / 200.0)
    qlim = QLIM_SFC * (1.0 - wt) + qlim * wt
    sgmq = jnp.maximum(sgm, qw * qlim)

    ql_water = jnp.minimum(sgmq, 0.025 * qw) * cldfra_bl
    ql_ice = jnp.minimum(sgmq, 0.025 * qw) * cldfra_bl

    tiny_cf = cldfra_bl < 0.001
    ql_water = jnp.where(tiny_cf, 0.0, ql_water)
    ql_ice = jnp.where(tiny_cf, 0.0, ql_ice)
    cldfra_bl = jnp.where(tiny_cf, 0.0, cldfra_bl)

    liq_frac = jnp.clip((t - TICE) / (TLIQ - TICE), 0.0, 1.0)
    qc_bl = liq_frac * ql_water
    qi_bl = (1.0 - liq_frac) * ql_ice

    # above the tropopause: no CB subgrid clouds (lines 3754-3759)
    k_tropo = _k_tropo(theta, p)[..., None]
    idx = jnp.arange(nz)
    above = idx >= k_tropo
    cldfra_bl = jnp.where(above, 0.0, cldfra_bl)
    qc_bl = jnp.where(above, 0.0, qc_bl)
    qi_bl = jnp.where(above, 0.0, qi_bl)

    # top level (kte): zero (lines 3866-3868)
    top = idx == (nz - 1)
    cldfra_bl = jnp.where(top, 0.0, cldfra_bl)
    qc_bl = jnp.where(top, 0.0, qc_bl)
    qi_bl = jnp.where(top, 0.0, qi_bl)
    return qc_bl, qi_bl, cldfra_bl


def dmp_shallow_cu_overwrite(
    *,
    qc_bl,       # (..., nz) from mym_condensation
    cldfra_bl,   # (..., nz) from mym_condensation
    edmf_a,      # (..., nz) DMP plume area (Psig_w-tapered), interface idx K-1
    edmf_qc,     # (..., nz) DMP in-plume mean condensate, interface idx K-1
    edmf_qt,     # (..., nz) DMP in-plume mean total water, interface idx K-1
    theta,       # (..., nz) dry potential temperature
    thl,         # (..., nz) liquid potential temperature
    qw,          # (..., nz) env total water specific content (qt1)
    p,           # (..., nz)
    exner,       # (..., nz)
    dz,          # (..., nz)
    xland,       # (...,)    1=land, 2=water
):
    """WRF ``DMP_mf`` shallow-cumulus cldfra/qc_bl overwrite (lines 6585-6720).

    Where condensing plumes exist and the stratus cloud fraction from
    ``mym_condensation`` is below ``cf_thresh=0.5``, the convective cloud
    fraction/condensate from the mass-flux scheme replaces the CB values.
    ``edmf_*`` arrays use the GPU EDMF convention: 0-based index j = WRF level
    K = j+1 (interface above mass level j).  Returns updated
    ``(qc_bl, cldfra_bl)``; ``qi_bl`` is NOT touched by WRF here.
    """

    nz = qc_bl.shape[-1]
    # dzi(k) = 0.5*(dz(k)+dz(k+1)) -- spacing across the interface above level k
    dzi = 0.5 * (dz[..., :-1] + dz[..., 1:])               # (..., nz-1), idx m
    # mass levels m = 1..nz-3 use interface values at j=m (WRF k) and j=m-1 (k-1)
    a_hi = edmf_a[..., 1:-1]                               # j=m   for m=1..nz-2
    a_lo = edmf_a[..., :-2]                                # j=m-1
    qc_hi = edmf_qc[..., 1:-1]
    qc_lo = edmf_qc[..., :-2]
    qt_hi = edmf_qt[..., 1:-1]
    qt_lo = edmf_qt[..., :-2]
    dzi_hi = dzi[..., 1:]                                  # dzi(m)
    dzi_lo = dzi[..., :-1]                                 # dzi(m-1)
    wsum = dzi_lo + dzi_hi

    aup = (a_hi * dzi_lo + a_lo * dzi_hi) / wsum
    qtp = (qt_hi * dzi_lo + qt_lo * dzi_hi) / wsum
    qcp_interp = (qc_hi * dzi_lo + qc_lo * dzi_hi) / wsum
    both_pos = (qc_hi > 0.0) & (qc_lo > 0.0)
    qcp = jnp.where(both_pos, qcp_interp, jnp.maximum(qc_hi, qc_lo))

    # env values at mass level m (m = 1..nz-2 slice)
    sl = (slice(None),) * (qc_bl.ndim - 1) + (slice(1, nz - 1),)
    t_m = (theta * exner)[sl]
    p_m = p[sl]
    qt1_m = qw[sl]
    thl_m = thl[sl]
    ex_m = exner[sl]

    xl = xl_blend(t_m)
    qsat_tk = _qsat_blend(t_m, p_m)
    rsl = xl * qsat_tk / (R_V * t_m * t_m)
    cpm = CP + qt1_m * CPV
    a_cb = 1.0 / (1.0 + xl * rsl / cpm)

    sigq = 10.0 * aup * (qtp - qt1_m)
    sigq = jnp.maximum(sigq, qsat_tk * 0.02)
    sigq = jnp.minimum(sigq, qsat_tk * 0.25)
    qmq = a_cb * (qt1_m - qsat_tk)
    q1 = qmq / sigq

    mf_cf = jnp.minimum(jnp.maximum(0.5 + 0.36 * jnp.arctan(1.55 * q1), 0.01), 0.8)
    is_water = ((jnp.asarray(xland) - 1.5) >= 0.0)[..., None]
    mf_cf = jnp.maximum(mf_cf, jnp.where(is_water, 1.2, 1.8) * aup)

    qca = qcp * aup
    qc_bl_mf = jnp.where(qca > 5.0e-5, 1.86 * qca - 2.2e-5, 1.18 * qca)

    # overwrite condition (line 6603): condensing plume at the bracketing
    # interfaces AND non-stratus CB cloud fraction; mass levels 1..nz-3 only
    cond = (0.5 * (qc_hi + qc_lo) > 0.0) & (cldfra_bl[sl] < CF_THRESH)
    midx = jnp.arange(1, nz - 1)
    cond = cond & (midx <= nz - 3)
    if _DMP_KTOP_BOUND:
        # WRF module_bl_mynnedmf.F:6601-6602 stops the SGS overwrite at
        # KTOP, the highest live plume interface (one-based). The existing
        # plume area gives that extent; retain the CB cloud above it.
        ktop = jnp.max(
            jnp.where(edmf_a > 0.0, jnp.arange(nz) + 1, 0), axis=-1
        )
        cond = cond & (midx < ktop[..., None])

    qc_bl_new = jnp.where(cond, qc_bl_mf, qc_bl[sl])
    cf_new = jnp.where(cond, mf_cf, cldfra_bl[sl])

    qc_bl = jnp.concatenate((qc_bl[..., :1], qc_bl_new, qc_bl[..., nz - 1:]), axis=-1)
    cldfra_bl = jnp.concatenate(
        (cldfra_bl[..., :1], cf_new, cldfra_bl[..., nz - 1:]), axis=-1
    )
    return qc_bl, cldfra_bl


# ---------------------------------------------------------------------------
# bl_mynn_cloudpdf = 0 / 1 (and the -1/-2 mass-flux-isolation variants).
#
# Opt-in WRF namelist alternatives to the Registry default CASE(2) above
# (Registry.EM_COMMON:2477 "0:original, 1:Kuwano, 2:Chaboreau-Bechtold").
# Transcribed from the pristine WRF 4.7.1 module_bl_mynnedmf.F90
# mym_condensation: CASE(0) :3476-3542, CASE(1,-1) :3544-3599, negative-option
# zeroing :3807-3814, kte zeroing :3816-3821.  Both forms run k = kts..kte-1,
# have NO tropopause cut and NO cldfra<0.001 zeroing (those belong to CASE 2).
# Caller wiring (mynn_bl_driver :905-913): thl1/sqw1 (specific, no snow),
# tsq1/qsq1/cov1/sh1/el1 are the PREVIOUS-step MYNN state (driver copy-in
# module_bl_mynnedmf_driver.F90:420-428).  vt/vq/sgm outputs are dead for
# closure 2.6 + mixlength 1 (use_buoy=.false., see module docstring).
# ---------------------------------------------------------------------------

_F4 = np.float32
_R_D4 = _F4(287.0)                     # module_model_constants.F:19
_R_V4 = _F4(461.6)                     # :22
_CP4 = _F4(7.0) * _R_D4 / _F4(2.0)     # :20
_XLV4 = _F4(2.5e6)                     # :55
_EP_2_4 = _R_D4 / _R_V4                # :81
_EP_3_4 = _F4(1.0) - _EP_2_4           # module_bl_mynnedmf_common.F90:73
_XLVCP4 = _XLV4 / _CP4                 # common :79
_T0C4 = _F4(273.15)                    # SVPT0 :79
_TICE4 = _F4(240.0)                    # common :68
_B2_4 = _F4(15.0)                      # module_bl_mynnedmf.F90:283
_RR2_4 = _F4(0.7071068)                # :313
_RRP_4 = _F4(0.3989423)                # :313
_ESAT_J = tuple(map(_F4, (0.611583699e03, 0.444606896e02, 0.143177157e01, 0.264224321e-1,
                          0.299291081e-3, 0.203154182e-5, 0.702620698e-8, 0.379534310e-11,
                          -0.321582393e-13)))
_ESAT_K = tuple(map(_F4, (0.609868993e03, 0.499320233e02, 0.184672631e01, 0.402737184e-1,
                          0.565392987e-3, 0.521693933e-5, 0.307839583e-7, 0.105785160e-9,
                          0.161444444e-12)))

CLOUDPDF_OPTIONS = frozenset({0, 1, 2})


def esat_blend(t):
    """WRF ``esat_blend`` (module_bl_mynnedmf.F90, FUNCTION esat_blend), Pa."""

    xc = jnp.maximum(_F4(-80.0), t - _T0C4)

    def horner(coef):
        acc = coef[8]
        for c in coef[7::-1]:
            acc = c + xc * acc
        return acc

    esl = horner(_ESAT_J)
    esi = horner(_ESAT_K)
    # Preserve WRF's REAL division. A folded reciprocal changes blended ESAT
    # by one ulp and is amplified in near-saturated, minimum-variance clouds.
    chi = ((_T0C4 - _F4(6.0)) - t) / jax.lax.optimization_barrier((_T0C4 - _F4(6.0)) - _TICE4)
    blend = (_F4(1.0) - chi) * esl + chi * esi
    return jnp.where(t >= (_T0C4 - _F4(6.0)), esl, jnp.where(t <= _TICE4, esi, blend))


def _alp_bet(t, p, exner):
    """Shared CASE(0)/(1) saturation terms (:3489-3497 / :3547-3555)."""

    esat = esat_blend(t)
    qsl = _EP_2_4 * esat / jnp.maximum(_F4(1.0e-4), p - _EP_3_4 * esat)
    dqsl = qsl * _EP_2_4 * _XLV4 / (_R_D4 * (t * t))  # r_d*t**2
    alp = _F4(1.0) / (_F4(1.0) + dqsl * _XLVCP4)
    bet = dqsl * exner
    return qsl, alp, bet


def _case0_sigma(tsq, qsq, cov, bet):
    """CASE(0) Sommeria-Deardorff sigma; t3sq/r3sq/c3sq are WRF DOUBLE (:3410).

    ``bet(k)**2`` and ``2.0*bet(k)`` are REAL products promoted only where
    they meet the DOUBLE operands (Fortran mixed-mode rule); the DOUBLE
    ``SQRT(MAX(r3sq,1d-10))`` is rounded to REAL on assignment to ``sgm``.
    """

    out_dtype = jnp.result_type(qsq)
    bet2 = bet * bet
    bet_x2 = _F4(2.0) * bet
    with jax.enable_x64(True):
        f64 = jnp.float64
        t3sq = jnp.maximum(jnp.asarray(tsq).astype(f64), 0.0)
        r3sq = jnp.maximum(jnp.asarray(qsq).astype(f64), 0.0)
        c3sq = jnp.asarray(cov).astype(f64)
        c3sq = jnp.copysign(jnp.minimum(jnp.abs(c3sq), jnp.sqrt(t3sq * r3sq)), c3sq)
        r3sq = (r3sq + bet2.astype(f64) * t3sq) - bet_x2.astype(f64) * c3sq
        sgm = jnp.sqrt(jnp.maximum(r3sq, 1.0e-10)).astype(out_dtype)
    return sgm


def _case1_sigma(thl, qw, dz, el, sh, alp, bet):
    """CASE(1,-1) Nakanishi-Niino 2004 eq. B6 sigma (:3557-3566); REAL dth/dqw/dzk."""

    nz = thl.shape[-1]
    idx = jnp.arange(nz - 1)
    dzk = jnp.where(idx == 0, _F4(0.5) * dz[..., : nz - 1], dz[..., : nz - 1])
    thl_km1 = jnp.concatenate((thl[..., :1], thl[..., : nz - 2]), axis=-1)
    qw_km1 = jnp.concatenate((qw[..., :1], qw[..., : nz - 2]), axis=-1)
    thl_k, qw_k = thl[..., : nz - 1], qw[..., : nz - 1]
    dth = _F4(0.5) * (thl[..., 1:] + thl_k) - _F4(0.5) * (thl_k + thl_km1)
    dqw = _F4(0.5) * (qw[..., 1:] + qw_k) - _F4(0.5) * (qw_k + qw_km1)
    a = alp[..., : nz - 1]
    b = bet[..., : nz - 1]
    e = el[..., : nz - 1]
    s = sh[..., : nz - 1]
    grad = dqw / dzk - b * (dth / dzk)
    var = ((((a * a) * jnp.maximum(e * e, _F4(0.1))) * _B2_4) * jnp.maximum(s, _F4(0.03))) / _F4(4.0) * (grad * grad)
    return jnp.sqrt(jnp.maximum(var, _F4(1.0e-10)))


def _sommeria_deardorff_cloud(qmq, sgm, alp, t):
    """Shared CASE(0)/(1) cloud fraction + condensate (:3511-3524 / :3567-3581)."""

    q1 = qmq / sgm
    cld = _F4(0.5) * (_F4(1.0) + jax.lax.erf(q1 * _RR2_4))
    eq1 = _RRP_4 * jnp.exp(_F4(-0.5) * q1 * q1)
    qll = jnp.maximum(cld * q1 + eq1, _F4(0.0))
    ql = alp * sgm * qll
    liq_frac = jnp.minimum(_F4(1.0), jnp.maximum(_F4(0.0), (t - _F4(240.0)) / _F4(29.0)))
    return liq_frac * ql, (_F4(1.0) - liq_frac) * ql, cld


def _cloudpdf01_columns(cloudpdf, *, theta, p, exner, dz, thl, qw, tsq, qsq, cov, el, sh):
    """CASE(0)/(1) on levels kts..kte-1: ``(qc_bl, qi_bl, cldfra_bl, sgm)``."""

    nz = theta.shape[-1]
    sl = (Ellipsis, slice(0, nz - 1))
    t = theta * exner
    qsl, alp, bet = _alp_bet(t, p, exner)
    if cloudpdf == 0:
        sgm = _case0_sigma(tsq[sl], qsq[sl], cov[sl], bet[sl])
    else:
        sgm = _case1_sigma(thl, qw, dz, el, sh, alp, bet)
    qmq = qw[sl] - qsl[sl]
    qc_bl, qi_bl, cld = _sommeria_deardorff_cloud(qmq, sgm, alp[sl], t[sl])
    return qc_bl, qi_bl, cld, sgm


def mym_condensation_cloudpdf01(
    cloudpdf,
    *,
    theta,       # (..., nz) DRY potential temperature (WRF th1)
    p,           # (..., nz)
    exner,       # (..., nz)
    dz,          # (..., nz)
    thl,         # (..., nz) liquid potential temperature (WRF thl1)
    qw,          # (..., nz) total water specific content (sqw1 = sqv+sqc+sqi)
    tsq=None,    # (..., nz) previous-step <thl'2>   (CASE 0)
    qsq=None,    # (..., nz) previous-step <qw'2>    (CASE 0)
    cov=None,    # (..., nz) previous-step <thl'qw'> (CASE 0)
    el=None,     # (..., nz) previous-step el_pbl    (CASE 1)
    sh=None,     # (..., nz) previous-step sh3d      (CASE 1)
):
    """WRF ``mym_condensation`` for ``bl_mynn_cloudpdf`` in {0, 1, -1}.

    Returns ``(qc_bl, qi_bl, cldfra_bl)`` on all ``nz`` levels; level kte is
    zero (:3819-3821).  ``cloudpdf=-1`` evaluates CASE(1) then zeroes the SGS
    cloud (:3807-3814, "isolating on the mass-flux clouds").
    """

    if cloudpdf not in (0, 1, -1):
        raise ValueError(f"mym_condensation_cloudpdf01 handles cloudpdf 0/1/-1, got {cloudpdf!r}")
    qc_bl, qi_bl, cld, _sgm = _cloudpdf01_columns(
        cloudpdf, theta=theta, p=p, exner=exner, dz=dz, thl=thl, qw=qw,
        tsq=tsq, qsq=qsq, cov=cov, el=el, sh=sh)
    if cloudpdf < 0:
        qc_bl, qi_bl, cld = (jnp.zeros_like(qc_bl),) * 3
    top = jnp.zeros_like(qc_bl[..., :1])
    return tuple(jnp.concatenate((x, top), axis=-1) for x in (qc_bl, qi_bl, cld))
