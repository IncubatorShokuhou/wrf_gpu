"""JIT/vmap-traceable GFS PBL column kernel (``bl_pbl_physics=3``).

A faithful, ``jax.jit``/``jax.vmap``-traceable transcription of the single-column
EM-core path of pristine WRF ``phys/module_bl_gfs.F`` (``BL_GFS`` -> ``MONINP`` ->
``TRIDI2`` / ``TRIDIN`` / ``TRIDIT``): the GFS Hybrid-EDMF-ancestor nonlocal-K PBL
(Hong-Pan 1996 lineage, the operational NCEP-GFS first-order closure with the
Troen-Mahrt countergradient terms ``HGAMT``/``HGAMQ`` and the bulk-Richardson
``RBCR=0.25`` PBL-height diagnosis) with the leapfrog (``DT = 2*dt``) implicit
vertical-diffusion solve for heat, moisture, momentum and the cloud tracer.

Conventions: profile inputs are bottom-up WRF mass-level columns of length ``n``
(``u, v, t, qv, qc, p, pii, dz, z``); ``z`` is the full-level geopotential height
above sea level (the WRF ``z`` argument). UNLIKE the YSU/MRF kernels, the GFS
scheme already works bottom-up (``k=1`` = surface), so no internal flip is needed;
the batch axis is left free for ``jax.vmap`` over grid columns and ``n`` is a
static trace-time int.

Two constant sets are deliberately distinct, mirroring WRF exactly:

* the ``BL_GFS`` *wrapper* uses the WRF driver-passed constants (``G=9.81``,
  ``R=287.0``, ``CP=7*R/2``) for the ``Q1``/``HEAT``/``EVAP``/``PRSL`` setup and
  the ``RTHBLTEN=TAU/PI3D`` recovery;
* ``MONINP`` uses ``MODULE_GFS_PHYSCONS`` (``con_g=9.80665``, ``con_rd=287.05``,
  ``con_cp=1004.6``, ``con_hvap=2.5e6``) and ``FV=con_rv/con_rd-1``.

The GFS PBL internals run at ``kind_phys=selected_real_kind(13,60)`` (real*8)
regardless of the WRF build's default REAL, so the natural precision is fp64.

WRF-faithfulness is gated by the v0.17 operational oracle
(``proofs/v017/gfs_oracle.py``) against fp64 savepoints from a standalone Fortran
driver linked against the UNMODIFIED ``module_bl_gfs.F`` (+ ``module_gfs_machine.F``
/ ``module_gfs_physcons.F``) at ~1e-12 (NOT a JAX-vs-JAX self-compare).
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from typing import NamedTuple

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

configure_jax_x64()

# --- WRF driver-passed constants (BL_GFS wrapper) -------------------------- #
G = 9.81
R_D = 287.0
CP = 7.0 * R_D / 2.0
ROVCP = R_D / CP
ROVG = R_D / G
R_V = 461.6
EP1 = R_V / R_D - 1.0  # WRF driver virtual-temperature constant
KARMAN = 0.4
P1000 = 1.0e5

# --- MODULE_GFS_PHYSCONS constants used inside MONINP ----------------------- #
CON_G = 9.80665
CON_RD = 2.8705e2
CON_RV = 4.6150e2
CON_CP = 1.0046e3
CON_HVAP = 2.5000e6
CON_ROG = CON_RD / CON_G
FV = CON_RV / CON_RD - 1.0  # con_fvirt

# --- MONINP PARAMETER block (module_bl_gfs.F lines 663-672) ---------------- #
GOR = CON_G / CON_RD
GOCP = CON_G / CON_CP
CONT = 1000.0 * CON_CP / CON_G
CONQ = 1000.0 * CON_HVAP / CON_G
CONW = 1000.0 / CON_G
RLAM = 150.0
VK = 0.4
VK2 = VK * VK
PRMIN = 1.0
PRMAX = 4.0
DW2MIN = 0.0001
DKMIN = 1.0
DKMAX = 1000.0
RIMIN = -100.0
CFAC = 7.8
PFAC = 2.0
SFCFRAC = 0.1
QMIN = 1.0e-8
XKZO = 1.0
ZFMIN = 1.0e-8
APHI5 = 5.0
APHI16 = 16.0
GAMCRT = 3.0
GAMCRQ = 0.0  # WRF active PARAMETER (line 671): GAMCRQ=0.
RBCR_CONST = 0.25  # EM-core constant critical Richardson number
RCL = 1.0  # BL_GFS sets RCL=1 (no map-factor reduction)


class MoninpConstants(NamedTuple):
    """MONINP constants whose WRF value differs from the exact decimal (see WRF_MONINP)."""

    CON_G: float
    GOR: float
    GOCP: float
    FV: float
    VK: float
    DW2MIN: float
    CFAC: float
    SFCFRAC: float
    QMIN: float
    ZFMIN: float
    PRINV_LIT: float  # CFAC*VK*.1 (:976)
    DKU_LIT: float  # 1+1.746*SRI (:1054)
    DKT_LIT: float  # 1+1.286*SRI (:1055)
    PRNUM_LIT: float  # 1.0+2.1*RI (:1063)


# Legacy fp64 path (v0.17 oracle): exact decimal values.
LEGACY_MONINP = MoninpConstants(CON_G, GOR, GOCP, FV, VK, DW2MIN, CFAC, SFCFRAC, QMIN, ZFMIN,
                                0.1, 1.746, 1.286, 2.1)


def _kind_phys_literal(value):
    """A default-REAL literal stored into REAL(kind_phys): the REAL value, widened."""
    return float(np.float32(value))


# --- WRF typing of the BL_GFS call (t-cfix-sfc F13) -------------------------- #
# The BL_GFS wrapper constants are REAL dummy arguments (module_bl_gfs.F:75-82)
# passed from module_model_constants (module_pbl_driver.F:1413,1418), folded in REAL.
_REAL = np.float32
WRF_R_D = _REAL(287.0)
WRF_CP = _REAL(7.0) * WRF_R_D / _REAL(2.0)
WRF_G = _REAL(9.81)
WRF_RCP = WRF_R_D / WRF_CP
WRF_ROVG = WRF_R_D / WRF_G
WRF_EP_1 = _REAL(461.6) / WRF_R_D - _REAL(1.0)
WRF_KARMAN = _REAL(0.4)
# MODULE_GFS_PHYSCONS and the MONINP PARAMETERs/inline literals are REAL(kind_phys)
# initialised from default-REAL literals (no _kind suffix): the REAL value widened;
# derived parameters are DOUBLE arithmetic on those (module_gfs_physcons.F:13-36,
# module_bl_gfs.F:663-672).
WRF_CON_G = _kind_phys_literal(9.80665)
WRF_CON_RD = _kind_phys_literal(2.8705e2)
WRF_CON_RV = _kind_phys_literal(4.6150e2)
WRF_CON_CP = _kind_phys_literal(1.0046e3)
WRF_MONINP = MoninpConstants(
    CON_G=WRF_CON_G, GOR=WRF_CON_G / WRF_CON_RD, GOCP=WRF_CON_G / WRF_CON_CP,
    FV=WRF_CON_RV / WRF_CON_RD - 1.0, VK=_kind_phys_literal(0.4), DW2MIN=_kind_phys_literal(0.0001),
    CFAC=_kind_phys_literal(7.8), SFCFRAC=_kind_phys_literal(0.1), QMIN=_kind_phys_literal(1.0e-8),
    ZFMIN=_kind_phys_literal(1.0e-8), PRINV_LIT=_kind_phys_literal(0.1), DKU_LIT=_kind_phys_literal(1.746),
    DKT_LIT=_kind_phys_literal(1.286), PRNUM_LIT=_kind_phys_literal(2.1))


def _assemble_diag(au, al):
    """WRF tridiagonal diagonal assembly from super (``au``) / sub (``al``) arrays.

    Mirrors the WRF loop ``AD(k)-=AU(k); AD(k+1)=1-AL(k)`` (AD init = 1), giving
    ``AD(0)=1-AU(0)``, ``AD(k)=(1-AL(k-1))-AU(k)`` for ``1<=k<=n-2``, and
    ``AD(n-1)=1-AL(n-2)``. ``au``/``al`` have length ``n-1``.
    """
    n = au.shape[0] + 1
    diag0 = (1.0 - au[0])[None]
    diag_mid = (1.0 - al[: n - 2]) - au[1:]
    diag_last = (1.0 - al[n - 2])[None]
    return jnp.concatenate([diag0, diag_mid, diag_last])


def _assemble_rhs(r0, r_base, lower, upper):
    """WRF tridiagonal RHS assembly mirroring ``R(k)+=lower(k); R(k+1)=base(k+1)+upper(k)``.

    Surface ``R(0)`` is ``r0 + lower(0)``; interior ``R(k)=r_base(k)+upper(k-1)+lower(k)``
    for ``1<=k<=n-2``; top ``R(n-1)=r_base(n-1)+upper(n-2)``. ``lower``/``upper`` have
    length ``n-1``; ``r_base`` length ``n`` (the un-diffused base profile for k>=1).
    """
    n = r_base.shape[0]
    rhs0 = (r0 + lower[0])[None]
    rhs_mid = r_base[1 : n - 1] + upper[: n - 2] + lower[1:]
    rhs_last = (r_base[n - 1] + upper[n - 2])[None]
    return jnp.concatenate([rhs0, rhs_mid, rhs_last])


def _thomas2(cl, cm, cu, r1, r2):
    """Solve two tridiagonal systems sharing the LHS (WRF ``TRIDI2``).

    ``cl`` (sub, valid 1..n-1), ``cm`` (diag, 0..n-1), ``cu`` (super, 0..n-2).
    Bottom-up forward elimination then back-substitution, transcribed 1:1 from
    ``module_bl_gfs.F:TRIDI2`` (used for momentum). Sequential along ``k`` via
    ``lax.scan`` so it stays trace-stable for any static depth ``n``.
    """
    n = cm.shape[0]

    fk0 = 1.0 / cm[0]
    au0 = fk0 * cu[0]
    a1_0 = fk0 * r1[0]
    a2_0 = fk0 * r2[0]

    def fwd(carry, k):
        au_prev, a1_prev, a2_prev = carry
        fk = 1.0 / (cm[k] - cl[k] * au_prev)
        au_k = fk * cu[k]
        a1_k = fk * (r1[k] - cl[k] * a1_prev)
        a2_k = fk * (r2[k] - cl[k] * a2_prev)
        return (au_k, a1_k, a2_k), (au_k, a1_k, a2_k)

    # forward for k = 1 .. n-2 (cu defined up to n-2)
    ks = jnp.arange(1, n - 1)
    (_, _, _), (au_mid, a1_mid, a2_mid) = jax.lax.scan(fwd, (au0, a1_0, a2_0), ks)

    au = jnp.concatenate([au0[None], au_mid]) if n > 1 else au0[None]
    a1_fwd = jnp.concatenate([a1_0[None], a1_mid]) if n > 1 else a1_0[None]
    a2_fwd = jnp.concatenate([a2_0[None], a2_mid]) if n > 1 else a2_0[None]

    # last row k = n-1
    fkn = 1.0 / (cm[n - 1] - cl[n - 1] * au[n - 2])
    a1_n = fkn * (r1[n - 1] - cl[n - 1] * a1_fwd[n - 2])
    a2_n = fkn * (r2[n - 1] - cl[n - 1] * a2_fwd[n - 2])

    a1_full = jnp.concatenate([a1_fwd[: n - 1], a1_n[None]])
    a2_full = jnp.concatenate([a2_fwd[: n - 1], a2_n[None]])

    # back-substitution k = n-2 .. 0
    def back(carry, k):
        a1_next, a2_next = carry
        a1_k = a1_full[k] - au[k] * a1_next
        a2_k = a2_full[k] - au[k] * a2_next
        return (a1_k, a2_k), (a1_k, a2_k)

    ks_b = jnp.arange(n - 2, -1, -1)
    (_, _), (a1_b, a2_b) = jax.lax.scan(back, (a1_n, a2_n), ks_b)
    # a1_b/a2_b are in reverse order (k = n-2 down to 0)
    a1_lower = a1_b[::-1]
    a2_lower = a2_b[::-1]

    a1_out = jnp.concatenate([a1_lower, a1_n[None]])
    a2_out = jnp.concatenate([a2_lower, a2_n[None]])
    return a1_out, a2_out


def _thomas1(cl, cm, cu, r1):
    """Solve a single tridiagonal system sharing the WRF ``TRIDIN``/``TRIDIT`` LHS.

    Identical elimination to ``_thomas2`` for one RHS. Used for the heat/moisture
    pair (called twice with the shared ``DKT`` matrix: once for T+qv via TRIDIN,
    once for qc via TRIDIT) -- here factored as one-RHS solves with a common LHS.
    """
    n = cm.shape[0]
    fk0 = 1.0 / cm[0]
    au0 = fk0 * cu[0]
    a1_0 = fk0 * r1[0]

    def fwd(carry, k):
        au_prev, a1_prev = carry
        fk = 1.0 / (cm[k] - cl[k] * au_prev)
        au_k = fk * cu[k]
        a1_k = fk * (r1[k] - cl[k] * a1_prev)
        return (au_k, a1_k), (au_k, a1_k)

    ks = jnp.arange(1, n - 1)
    (_, _), (au_mid, a1_mid) = jax.lax.scan(fwd, (au0, a1_0), ks)
    au = jnp.concatenate([au0[None], au_mid]) if n > 1 else au0[None]
    a1_fwd = jnp.concatenate([a1_0[None], a1_mid]) if n > 1 else a1_0[None]

    fkn = 1.0 / (cm[n - 1] - cl[n - 1] * au[n - 2])
    a1_n = fkn * (r1[n - 1] - cl[n - 1] * a1_fwd[n - 2])
    a1_full = jnp.concatenate([a1_fwd[: n - 1], a1_n[None]])

    def back(carry, k):
        (a1_next,) = carry
        a1_k = a1_full[k] - au[k] * a1_next
        return (a1_k,), (a1_k,)

    ks_b = jnp.arange(n - 2, -1, -1)
    (_,), (a1_b,) = jax.lax.scan(back, (a1_n,), ks_b)
    a1_lower = a1_b[::-1]
    return jnp.concatenate([a1_lower, a1_n[None]])


def _bl_gfs_setup_legacy(u, v, t, qv, qc, p, dz, z, *, psfc, hfx, qfx, tsk, gz1oz0, psim, psih,
                         wspd_in, br, qi):
    """Legacy BL_GFS setup: every statement in the operand dtype (v0.17 fp64 path)."""
    n = u.shape[0]

    # ---- BL_GFS setup (driver constants G/R/CP) ----
    rrhox = (R_D * t[0] * (1.0 + EP1 * qv[0])) / psfc
    cpm = CP * (1.0 + 0.8 * qv[0])
    fmtmp = gz1oz0 - psim
    psk = (psfc * 1.0e-5) ** ROVCP
    fm = fmtmp
    fh = gz1oz0 - psih
    tsea = tsk
    heat = hfx / cpm * rrhox
    evap = qfx * rrhox

    stress = KARMAN * KARMAN * wspd_in * wspd_in / (fmtmp * fmtmp)
    spd1 = wspd_in
    rbsoil = br

    # Specific humidities (BL_GFS): Q1(:,1)=qv/(1+qv), Q1(:,2)=qc/(1+qc)
    q1 = qv / (1.0 + qv)
    q1c = qc / (1.0 + qc)
    u1 = u
    v1 = v
    t1 = t
    prsl = p * 1.0e-3  # Pa -> kPa

    prslk = (prsl * 0.01) ** ROVCP

    # DEL(k) = PRSL(k)/ROVG * dz8w(k)/T3D(k), bottom-up
    # PRSI built downward from surface: PRSI(1)=PSFC*.001; PRSI(k+1)=PRSI(k)-DEL(k)
    delp = prsl / ROVG * dz / t1  # DEL for k=0..n-1; DEL(n-1) overwritten below
    delp = delp.at[n - 1].set(delp[n - 2])

    psi0 = psfc * 1.0e-3
    # PRSI(k) = psi0 - sum_{j<k} DEL(j) ; length n+1
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(delp)])
    prsi = psi0 - cum  # prsi[0..n]

    # PHII(k) = (Z(k)-Z(0))*G ; PHIL(k) = 0.5*(Z(k+1)+Z(k)-2 Z(0))*G (k=0..n-2)
    z0 = z[0]
    phii = (z - z0) * G  # length n, plus top below
    phii_top = phii[n - 1] + dz[n - 1] * G
    phii_full = jnp.concatenate([phii, phii_top[None]])  # length n+1
    phil = 0.5 * (z[1:] + z[:-1] - 2.0 * z0) * G  # length n-1
    phil_top = phii_full[n - 1] - phil[n - 2] + phii_full[n - 1]
    phil_full = jnp.concatenate([phil, phil_top[None]])  # length n
    q1i = None if qi is None else qi / (1.0 + qi)
    return dict(u1=u1, v1=v1, t1=t1, q1=q1, q1c=q1c, q1i=q1i, prsl=prsl, prslk=prslk, delp=delp,
                prsi=prsi, phii=phii_full, phil=phil_full, psk=psk, fm=fm, fh=fh, tsea=tsea, heat=heat,
                evap=evap, stress=stress, spd1=spd1, rbsoil=rbsoil, qfx=qfx)


def _bl_gfs_setup_wrf(u, v, t, qv, qc, p, dz, z, *, psfc, hfx, qfx, tsk, gz1oz0, psim, psih,
                      wspd_in, br, qi):
    """WRF BL_GFS setup (module_bl_gfs.F:276-398) on REAL grid operands.

    Pure-REAL right-hand sides (REAL grid arrays, REAL dummy constants, default-REAL
    literals) are evaluated in REAL and only then stored into the REAL(kind_phys) work
    arrays; statements that read a kind_phys variable are DOUBLE. Returns the DOUBLE
    MONINP operands. Requires an x64-enabled scope.
    """
    n = u.shape[0]
    lit = jnp.float32  # default-REAL literal
    one = lit(1.0)
    dbl = lambda value: jnp.asarray(value, jnp.float64)  # noqa: E731  REAL -> kind_phys store
    # :278-302 surface statements
    rrhox = dbl((WRF_R_D * t[0] * (one + WRF_EP_1 * qv[0])) / psfc)
    cpm = dbl(WRF_CP * (one + lit(0.8) * qv[0]))
    fmtmp = dbl(gz1oz0 - psim)
    psk = dbl((psfc * lit(1.0e-5)) ** WRF_RCP)
    fh = dbl(gz1oz0 - psih)
    heat = dbl(hfx) / cpm * rrhox
    evap = dbl(qfx) * rrhox
    stress = dbl(WRF_KARMAN * WRF_KARMAN * wspd_in * wspd_in) / (fmtmp * fmtmp)
    # :349-366 profiles
    q1 = dbl(qv / (one + qv))
    q1c = dbl(qc / (one + qc))
    q1i = None if qi is None else dbl(qi / (one + qi))  # :392-398
    prsl = dbl(p * lit(0.001))
    prslk = (prsl * dbl(lit(0.01))) ** dbl(WRF_RCP)  # :368-372 DOUBLE
    # :375-390 DEL/PRSI/PHII/PHIL: DOUBLE where a kind_phys operand enters, REAL otherwise
    # Opaque divisor: XLA algsimp would rewrite PRSL/ROVG into PRSL*(1/ROVG) (E130).
    rovg = jax.lax.optimization_barrier(dbl(WRF_ROVG))
    delp = prsl[: n - 1] / rovg * dbl(dz[: n - 1]) / dbl(t[: n - 1])
    delp = jnp.concatenate([delp, delp[n - 2:n - 1]])
    prsi = [dbl(psfc * lit(0.001))]
    for k in range(1, n):
        prsi.append(prsi[k - 1] - delp[k - 1])
    prsi.append(prsi[n - 1] - delp[n - 2])
    phii = dbl((z - z[0]) * WRF_G)
    phii_full = jnp.concatenate([phii, (phii[n - 1] + dbl(dz[n - 1] * WRF_G))[None]])
    phil = dbl(lit(0.5) * (z[1:] + z[:-1] - lit(2.0) * z[0]) * WRF_G)
    phil_full = jnp.concatenate([phil, (phii_full[n - 1] - phil[n - 2] + phii_full[n - 1])[None]])
    return dict(u1=dbl(u), v1=dbl(v), t1=dbl(t), q1=q1, q1c=q1c, q1i=q1i, prsl=prsl, prslk=prslk,
                delp=delp, prsi=jnp.stack(prsi), phii=phii_full, phil=phil_full, psk=psk, fm=fmtmp,
                fh=fh, tsea=dbl(tsk), heat=heat, evap=evap, stress=stress, spd1=dbl(wspd_in),
                rbsoil=dbl(br), qfx=dbl(qfx))


def _moninp_and_store(w, pii, dt, c):
    """MONINP (module_bl_gfs.F:569-1298) + BL_GFS tendency/UST recovery (:432-466).

    ``w``: setup work arrays; ``c``: MoninpConstants. Returns
    (du, dv, rthblten, rqvblten, rqcblten, rqiblten, pbl, kpbl, ust).
    """
    u1, v1, t1, q1, q1c, q1i = w['u1'], w['v1'], w['t1'], w['q1'], w['q1c'], w['q1i']
    prsl, prslk, delp, prsi = w['prsl'], w['prslk'], w['delp'], w['prsi']
    phii_full, phil_full, psk, fm, fh = w['phii'], w['phil'], w['psk'], w['fm'], w['fh']
    tsea, heat, evap, stress, spd1 = w['tsea'], w['heat'], w['evap'], w['stress'], w['spd1']
    rbsoil, qfx = w['rbsoil'], w['qfx']
    CON_G, GOR, GOCP, FV, VK, DW2MIN, CFAC, SFCFRAC, QMIN, ZFMIN = c[:10]
    n = u1.shape[0]
    kmpbl = n // 2

    # ---- MONINP ----
    gravi = 1.0 / CON_G
    dt2 = 2.0 * dt
    rdt = 1.0 / dt2

    zi = phii_full * gravi  # length n+1
    zl = phil_full * gravi  # length n

    theta = t1 * psk / prslk  # used 1..kmpbl, fine for all

    # RDZT(k) = GOR*PRSI(k+1)/(PRSL(k)-PRSL(k+1)) for k=0..n-2
    rdzt = GOR * prsi[1:n] / (prsl[: n - 1] - prsl[1:n])

    rbcr = RBCR_CONST
    sfcflg = rbsoil <= 0.0  # SFCFLG true when RBSOIL<=0
    ustar = jnp.sqrt(stress)

    rdzt1 = GOR * prsl[0] / delp[0]
    beta = dt2 * rdzt1 / t1[0]

    thesv = tsea * (1.0 + FV * jnp.maximum(qfx * 0.0 + q1[0], QMIN))  # QSS=qv -> not used
    the1 = theta[0]
    the1v = the1 * (1.0 + FV * jnp.maximum(q1[0], QMIN))

    # ---- first-guess PBL height (no thermal) ----
    thekv = theta * (1.0 + FV * jnp.maximum(q1, QMIN))
    spdk2 = jnp.maximum(RCL * (u1 * u1 + v1 * v1), 1.0)
    rb_prof = (thekv - the1v) * (G * zl / the1v) / spdk2  # G here = driver? WRF uses G=grav

    # WRF MONINP uses G=grav (con_g) in the RBUP formula (PARAMETER g=grav).
    rb_prof = (thekv - the1v) * (CON_G * zl / the1v) / spdk2

    # scan upward k=1..kmpbl-1 to find first stable level (RBUP>RBCR)
    def pbl_scan(carry, k):
        kpbl_c, rbup_c, rbdn_c, stable_c = carry
        in_range = (k >= 1) & (k < kmpbl)
        rbdn_new = jnp.where((~stable_c) & in_range, rbup_c, rbdn_c)
        rbup_new = jnp.where((~stable_c) & in_range, rb_prof[k], rbup_c)
        kpbl_new = jnp.where((~stable_c) & in_range, k, kpbl_c)
        stable_new = jnp.where(in_range, stable_c | (rbup_new > rbcr), stable_c)
        return (kpbl_new, rbup_new, rbdn_new, stable_new), None

    init = (jnp.array(1), rbsoil, rbsoil, jnp.array(False))
    (kpbl1, rbup1, rbdn1, _), _ = jax.lax.scan(pbl_scan, init, jnp.arange(0, kmpbl))

    def interp_hpbl(kpbl, rbdn, rbup):
        rbint = jnp.where(
            rbdn >= rbcr, 0.0,
            jnp.where(rbup <= rbcr, 1.0, (rbcr - rbdn) / (rbup - rbdn)),
        )
        zk = zl[kpbl]
        zkm = zl[kpbl - 1]
        return zkm + rbint * (zk - zkm)

    hpbl1 = interp_hpbl(kpbl1, rbdn1, rbup1)
    kpbl1 = jnp.where(hpbl1 < zi[kpbl1], kpbl1 - 1, kpbl1)

    # ---- surface layer scales ----
    hol0 = jnp.maximum(rbsoil * fm * fm / fh, RIMIN)
    hol = jnp.where(sfcflg, jnp.minimum(hol0, -ZFMIN), jnp.maximum(hol0, ZFMIN))
    hol = hol * hpbl1 / zl[0] * SFCFRAC
    tem_u = 1.0 / (1.0 - APHI16 * hol)
    phih_u = jnp.sqrt(tem_u)
    phim_u = jnp.sqrt(phih_u)
    phim_s = 1.0 + APHI5 * hol
    phih_s = phim_s
    phim = jnp.where(sfcflg, phim_u, phim_s)
    phih = jnp.where(sfcflg, phih_u, phih_s)
    wscale = ustar / phim
    wscale = jnp.minimum(wscale, ustar * APHI16)
    wscale = jnp.maximum(wscale, ustar / APHI5)

    # ---- countergradient + thermal-enhanced PBL ----
    sflux = heat + evap * FV * the1
    pblflg = sfcflg & (sflux > 0.0)
    hgamt0 = jnp.minimum(CFAC * heat / wscale, GAMCRT)
    hgamq0 = jnp.minimum(CFAC * evap / wscale, GAMCRQ)
    vpert = hgamt0 + FV * the1 * hgamq0
    vpert = jnp.minimum(vpert, GAMCRT)
    thermal = jnp.where(pblflg, the1v + jnp.maximum(vpert, 0.0), the1v)
    hgamt = jnp.where(pblflg, jnp.maximum(hgamt0, 0.0), 0.0)
    hgamq = jnp.where(pblflg, jnp.maximum(hgamq0, 0.0), 0.0)

    rb_prof2 = (thekv - thermal) * (CON_G * zl / the1v) / spdk2

    def pbl_scan2(carry, k):
        kpbl_c, rbup_c, rbdn_c, stable_c = carry
        in_range = (k >= 1) & (k < kmpbl)
        active = (~stable_c) & in_range & pblflg
        rbdn_new = jnp.where(active, rbup_c, rbdn_c)
        rbup_new = jnp.where(active, rb_prof2[k], rbup_c)
        kpbl_new = jnp.where(active, k, kpbl_c)
        stable_new = jnp.where(in_range & pblflg, stable_c | (rbup_new > rbcr), stable_c)
        return (kpbl_new, rbup_new, rbdn_new, stable_new), None

    init2 = (jnp.array(1), rbsoil, rbsoil, jnp.array(False))
    (kpbl2, rbup2, rbdn2, _), _ = jax.lax.scan(pbl_scan2, init2, jnp.arange(0, kmpbl))

    hpbl2 = interp_hpbl(kpbl2, rbdn2, rbup2)
    kpbl2 = jnp.where(hpbl2 < zi[kpbl2], kpbl2 - 1, kpbl2)
    # module_bl_gfs.F:935 IF(KPBL(I).LE.1) PBLFLG=.FALSE. -- 1-based KPBL; kpbl2
    # here is 0-based (KPBL2D = kpbl+1 below), so the test is kpbl2 <= 0.
    pblflg2 = pblflg & (kpbl2 > 0)

    # Final HPBL/KPBL: thermal branch when pblflg else first-guess
    hpbl = jnp.where(pblflg, hpbl2, hpbl1)
    kpbl = jnp.where(pblflg, kpbl2, kpbl1)
    pblflg = jnp.where(pblflg, pblflg2, pblflg)

    # ---- diffusion coefficients below PBL (k=0..kmpbl-1) ----
    prinv = 1.0 / (phih / phim + CFAC * VK * c.PRINV_LIT)
    prinv = jnp.clip(prinv, PRMIN, PRMAX)
    kidx = jnp.arange(n)
    zfac = jnp.maximum(1.0 - (zi[1:] - zl[0]) / (hpbl - zl[0]), ZFMIN)  # zi[k+1], len n
    dku_below = XKZO + wscale * VK * zi[1:] * 1.0 * zfac ** PFAC  # ALPHA=1
    dkt_below = dku_below * prinv
    below_mask = (kidx < kpbl) & (kidx < kmpbl)
    dku_b = jnp.where(below_mask, jnp.clip(dku_below, DKMIN, DKMAX), 0.0)
    dkt_b = jnp.where(below_mask, jnp.clip(dkt_below, DKMIN, DKMAX), 0.0)

    # ---- diffusion coefficients in free atmosphere (k>=kpbl, k=0..n-2) ----
    ti = 2.0 / (t1[: n - 1] + t1[1:])
    rdz = rdzt * ti
    dw2 = RCL * ((u1[: n - 1] - u1[1:]) ** 2 + (v1[: n - 1] - v1[1:]) ** 2)
    shr2 = jnp.maximum(dw2, DW2MIN) * rdz ** 2
    tvd = t1[: n - 1] * (1.0 + FV * jnp.maximum(q1[: n - 1], QMIN))
    tvu = t1[1:] * (1.0 + FV * jnp.maximum(q1[1:], QMIN))
    bvf2 = CON_G * (GOCP + rdz * (tvu - tvd)) * ti
    ri = jnp.maximum(bvf2 / shr2, RIMIN)
    zk = VK * zi[1:n]
    rl2 = zk * RLAM / (RLAM + zk)
    dk = rl2 * rl2 * jnp.sqrt(shr2)
    sri = jnp.sqrt(jnp.maximum(-ri, 0.0))
    dku_unst = XKZO + dk * (1.0 + 8.0 * (-ri) / (1.0 + c.DKU_LIT * sri))
    dkt_unst = XKZO + dk * (1.0 + 8.0 * (-ri) / (1.0 + c.DKT_LIT * sri))
    tem_st = dk / (1.0 + 5.0 * ri) ** 2
    dkt_st = XKZO + tem_st
    prnum = jnp.minimum(1.0 + c.PRNUM_LIT * ri, PRMAX)
    dku_st = (dkt_st - XKZO) * prnum + XKZO
    dku_free = jnp.where(ri < 0.0, dku_unst, dku_st)
    dkt_free = jnp.where(ri < 0.0, dkt_unst, dkt_st)
    free_mask = jnp.arange(n - 1) >= kpbl
    dku_f = jnp.where(free_mask, jnp.clip(dku_free, DKMIN, DKMAX), 0.0)
    dkt_f = jnp.where(free_mask, jnp.clip(dkt_free, DKMIN, DKMAX), 0.0)

    # Combine: below-PBL coeffs (k<kpbl) and free-atmosphere (k>=kpbl).
    # dku/dkt are defined on k=0..n-2 (interfaces between k and k+1).
    dku = jnp.where(jnp.arange(n - 1) < kpbl, dku_b[: n - 1], dku_f)
    dkt = jnp.where(jnp.arange(n - 1) < kpbl, dkt_b[: n - 1], dkt_f)

    # ---- tridiagonal for heat (A1=T) and moisture (A2=qv) ----
    dt_arr = dt2
    rdz_m = rdzt * 2.0 / (t1[: n - 1] + t1[1:])  # k=0..n-2

    dtodsd = dt_arr / delp[: n - 1]
    dtodsu = dt_arr / delp[1:]
    dsig = prsl[: n - 1] - prsl[1:]
    tem1 = dsig * dkt * rdz_m

    below_pbl = (jnp.arange(n - 1) < kpbl) & pblflg
    inv_hpbl = 1.0 / hpbl
    dsdzt = jnp.where(below_pbl, tem1 * (GOCP - hgamt * inv_hpbl), tem1 * GOCP)
    dsdzq = jnp.where(below_pbl, tem1 * (-hgamq * inv_hpbl), 0.0)
    dsdz2 = tem1 * rdz_m

    au = -dtodsd * dsdz2  # super-diagonal CU(k), k=0..n-2
    al = -dtodsu * dsdz2  # sub-diagonal CL(k+1)

    # Build diagonal AD and RHS A1 (T), A2 (qv). WRF assembles AD via the loop
    #   AD(k)   = AD(k) - AU(k)      (k=0..n-2)
    #   AD(k+1) = 1 - AL(k)          (k+1=1..n-1, overwrites the AD=1 init)
    # so the net diagonal is AD(0)=1-AU(0); AD(k)=(1-AL(k-1))-AU(k) for 1..n-2;
    # AD(n-1)=1-AL(n-2). RHS likewise: A1(k)+=DTODSD*DSDZT then A1(k+1) is set.
    ad = _assemble_diag(au, al)
    a1 = _assemble_rhs(t1[0] + beta * heat, t1, dtodsd * dsdzt, -dtodsu * dsdzt)
    a2 = _assemble_rhs(q1[0] + beta * evap, q1, dtodsd * dsdzq, -dtodsu * dsdzq)

    cl = jnp.concatenate([jnp.zeros(1), al])  # CL(k) sub-diagonal, index 1..n-1
    cu = au  # CU(k) length n-1
    a1_sol, a2_sol = _thomas2(cl, ad, cu, a1, a2)

    ttend = (a1_sol - t1) * rdt
    qtend = (a2_sol - q1) * rdt
    tau = ttend
    rtg_qv = qtend

    # ---- tridiagonal for momentum (RHS = u1/v1; surface drag in AD(0)) ----
    dsdz2_m = dsig * dku * rdz_m * rdz_m
    au_m = -dtodsd * dsdz2_m
    al_m = -dtodsu * dsdz2_m
    ad_m = _assemble_diag(au_m, al_m)
    # WRF surface row: AD(0) = (1 + beta*stress/spd1) - AU(0).
    ad_m = ad_m.at[0].add(beta * stress / spd1)
    cl_m = jnp.concatenate([jnp.zeros(1), al_m])
    u_sol, v_sol = _thomas2(cl_m, ad_m, au_m, u1, v1)
    du = (u_sol - u1) * rdt
    dv = (v_sol - v1) * rdt

    # ---- tridiagonal for tracer (qc), shares DKT (no countergradient) ----
    # AD(0)=1 (plain diffusion), RHS = qc base profile.
    dsdz2_t = tem1 * rdz_m
    au_t = -dtodsd * dsdz2_t
    al_t = -dtodsu * dsdz2_t
    ad_t = _assemble_diag(au_t, al_t)
    cl_t = jnp.concatenate([jnp.zeros(1), al_t])
    qc_sol = _thomas1(cl_t, ad_t, au_t, q1c)
    qctend = (qc_sol - q1c) * rdt
    # Tracer 3 (QI) shares the TRIDIT LHS (module_bl_gfs.F:1250-1292); each RHS
    # column is eliminated independently, so a separate one-RHS solve is exact.
    qi = q1i  # tracer 3: None = NTRAC=2
    if qi is None:
        rqiblten = jnp.zeros_like(qctend)
    else:
        qi_sol = _thomas1(cl_t, ad_t, au_t, q1i)
        rqiblten = (qi_sol - q1i) * rdt / (1.0 - q1i) ** 2  # :455-461

    # ---- BL_GFS tendency recovery (driver constants) ----
    rthblten = tau / pii
    rqvblten = rtg_qv / (1.0 - q1) ** 2
    rqcblten = qctend / (1.0 - q1c) ** 2
    rublten = du
    rvblten = dv

    ust_out = jnp.sqrt(stress)
    pbl_out = hpbl
    kpbl_out = kpbl + 1  # WRF KPBL2D is 1-based

    return rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, pbl_out, kpbl_out, ust_out


def _gfs_column(u, v, t, qv, qc, p, pii, dz, z, *, psfc, ust_in, hfx, qfx,
                tsk, gz1oz0, psim, psih, wspd_in, br, dt, qi=None):
    """One-column GFS PBL: returns (du, dv, rthblten, rqvblten, rqcblten, rqiblten, pbl, kpbl, ust).

    ``qi`` is BL_GFS's QI3D tracer 3 (``NTRAC=3`` when ``P_QI>=P_FIRST_SCALAR``,
    module_bl_gfs.F:274,392-398); ``None`` = NTRAC=2 (no ice slot, RQIBLTEN 0).

    Direct transcription of ``BL_GFS`` (EM core, NMM/HWRF off) + ``MONINP``. All
    arrays are length-``n`` bottom-up WRF mass-level columns.
    """
    work = _bl_gfs_setup_legacy(u, v, t, qv, qc, p, dz, z, psfc=psfc, hfx=hfx, qfx=qfx, tsk=tsk,
                                gz1oz0=gz1oz0, psim=psim, psih=psih, wspd_in=wspd_in, br=br, qi=qi)
    return _moninp_and_store(work, pii, dt, LEGACY_MONINP)


def gfs_columns(u, v, t, qv, qc, p, pii, dz, z, *, psfc, ust, hfx, qfx,
                tsk, gz1oz0, psim, psih, wspd, br, dt=60.0, qi=None):
    """Batched GFS PBL kernel over the leading (column) axis.

    Profile args are ``(B, n)`` bottom-up WRF mass-level columns; scalar surface
    forcings are ``(B,)``. Returns a dict of WRF GFS tendencies + diagnostics.
    """
    fn = jax.vmap(
        lambda *a, **k: _gfs_column(*a, **k),
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0),
    )

    def one(u_, v_, t_, qv_, qc_, p_, pii_, dz_, z_,
            psfc_, ust_, hfx_, qfx_, tsk_, gz_, psim_, psih_, wspd_, br_, qi_):
        return _gfs_column(
            u_, v_, t_, qv_, qc_, p_, pii_, dz_, z_,
            psfc=psfc_, ust_in=ust_, hfx=hfx_, qfx=qfx_, tsk=tsk_,
            gz1oz0=gz_, psim=psim_, psih=psih_, wspd_in=wspd_, br=br_, dt=dt, qi=qi_,
        )

    batched = jax.vmap(one, in_axes=(0,) * 20)
    ru, rv, rth, rqv, rqc, rqi, pbl, kpbl, ust_o = batched(
        u, v, t, qv, qc, p, pii, dz, z,
        psfc, ust, hfx, qfx, tsk, gz1oz0, psim, psih, wspd, br, qi,
    )
    return {
        "rublten": ru, "rvblten": rv, "rthblten": rth,
        "rqvblten": rqv, "rqcblten": rqc, "rqiblten": rqi,
        "pblh": pbl, "kpbl": kpbl, "ust": ust_o,
    }


_WRF_SURFACE = ("psfc", "hfx", "qfx", "tsk", "gz1oz0", "psim", "psih", "wspd_in", "br")


def bl_gfs_wrf_stage(u, v, t, qv, qc, p, dz, z, *, psfc, hfx, qfx, tsk, gz1oz0, psim, psih, wspd, br,
                     qi=None):
    """Batched WRF BL_GFS setup: REAL ``(B, n)``/``(B,)`` operands -> DOUBLE MONINP work arrays.

    Must run in an x64-enabled scope (the kind_phys stores widen to float64).
    """
    def one(profiles, surface, qi_):
        return _bl_gfs_setup_wrf(*profiles, **dict(zip(_WRF_SURFACE, surface)), qi=qi_)
    return jax.vmap(one)((u, v, t, qv, qc, p, dz, z), (psfc, hfx, qfx, tsk, gz1oz0, psim, psih, wspd, br), qi)


def bl_gfs_wrf_moninp(work, pii, dt):
    """Batched DOUBLE MONINP + BL_GFS stores on ``bl_gfs_wrf_stage`` work arrays (x64 scope).

    ``pii`` is the REAL PI3D, ``dt`` the REAL ``dtbl`` (DELTIM=DT). Outputs are the
    kind_phys values before the REAL grid stores.
    """
    deltim = float(np.float32(dt))
    out = jax.vmap(lambda w, pi: _moninp_and_store(w, jnp.asarray(pi, jnp.float64), deltim, WRF_MONINP))(work, pii)
    return dict(zip(("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten", "pblh", "kpbl",
                     "ust"), out))


def bl_gfs_wrf_columns(u, v, t, qv, qc, p, pii, dz, z, *, psfc, ust, hfx, qfx, tsk, gz1oz0, psim, psih,
                       wspd, br, dt, qi=None):
    """WRF ``BL_GFS`` public call on REAL grid columns (t-cfix-sfc F13).

    REAL setup statements in REAL (module_bl_gfs.F:276-398), MONINP in DOUBLE
    (kind_phys, module_gfs_machine.F:8) with WRF's literal constants, REAL stores
    of the tendencies/UST/PBL (:432-466). ``ust`` (UST INOUT) is overwritten by
    BL_GFS and not read. Same column layout and output keys as ``gfs_columns``.
    """
    del ust
    real = lambda value: None if value is None else jnp.asarray(value, jnp.float32)  # noqa: E731
    with jax.enable_x64(True):
        work = bl_gfs_wrf_stage(*(real(a) for a in (u, v, t, qv, qc, p, dz, z)), psfc=real(psfc),
                                hfx=real(hfx), qfx=real(qfx), tsk=real(tsk), gz1oz0=real(gz1oz0),
                                psim=real(psim), psih=real(psih), wspd=real(wspd), br=real(br), qi=real(qi))
        out = bl_gfs_wrf_moninp(work, real(pii), dt)
        return {k: (v.astype(jnp.float32) if jnp.issubdtype(v.dtype, jnp.floating) else v)
                for k, v in out.items()}


__all__ = ["gfs_columns", "bl_gfs_wrf_columns", "bl_gfs_wrf_stage", "bl_gfs_wrf_moninp"]
