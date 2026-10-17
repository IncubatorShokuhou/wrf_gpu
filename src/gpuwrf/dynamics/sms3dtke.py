"""Standalone WRF SMS-3DTKE coefficients and implicit dissipation length.

Literal REAL transcription of ``module_diffusion_em.F:tke_km`` (km_opt=5)
and the km_opt=5 branch of ``tke_dissip``. The input boundary is immediately
after WRF common geometry/deformation/thermodynamic preprocessing; exact saved
operands isolate the SMS branches. Memory/indexing/length scaling reuse km2. These are component ports:
the operational km_opt=5 refusal remains until nonlocal fluxes, mixing lengths,
implicit vertical diffusion and the complete caller chain are qualified.

``hpbl`` [m] and ``dlk`` [m] are REQUIRED upstream WRF operands, not optional
approximations. TKE is WRF ``tke_2`` [m2 s-2], not MYNN QKE. No State, default
forecast path, or restart schema changes are made by this module.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.les3d_smagorinsky import _G, _PRANDTL, _Mem, _calc_l_scale, _k, _mass_ranges


@dataclasses.dataclass(frozen=True)
class SmsConfig:
    """Static SMS branch controls. km_opt=5 requires bl_pbl_physics=0."""

    boundary: str = 'specified'
    isfflx: int = 1
    c_s: float = .25
    c_k: float = .15
    tke_drag_coefficient: float = 0.
    tke_heat_flux: float = 0.

    def __post_init__(self):
        if self.boundary not in ('specified', 'periodic') or self.isfflx not in (0,1,2):
            raise ValueError('unsupported SMS boundary or surface-flux configuration')


class SmsInputs(NamedTuple):
    """WRF halo-memory savepoints (k,j,i), HALO=3; surface arrays (j,i).

    dthrdn is tke_km's common potential-temperature gradient [K m-1]; bn2
    is calculate_N2's moist stability [s-2]. rdzw [m-1], d11/d22/d12 [s-1],
    zx/zy [dimensionless metric], theta [K], TKE [m2 s-2], hpbl/dlk [m].
    dx/dy [m] are WRF REAL. Inputs include the unused upper k face and rings.
    No default or substitute is provided for missing upstream operands.
    """

    tke: object
    theta: object
    dthrdn: object
    bn2: object
    rdzw: object
    d11: object
    d22: object
    d12: object
    zx: object
    zy: object
    dlk: object
    msftx: object
    msfty: object
    hpbl: object
    dx: object
    dy: object



class SmsCoefficients(NamedTuple):
    """Halo-padded WRF outputs; K [m2 s-1], l_diss [m]."""

    xkmh: object
    xkmv: object
    xkhh: object
    xkhv: object
    xkmv_meso: object
    l_diss: object


def _partial(d, h, *, local: bool):
    # WRF REAL parameters, including the compile-time REAL subtraction 1.-a7.
    f = np.float32
    denom_h = jnp.where(h != 0., h, jnp.ones_like(h))
    doh = d / denom_h
    if local:
        num = f(1.) * doh ** f(2.) + f(.870) * doh ** f(.5) + f(-.913)
        den = f(1.) * doh ** f(2.) + f(.153) * doh ** f(.5) + f(.278)
        value = f(.280) * num / den + f(f(1.) - f(.280))
    else:
        num = f(1.) * doh ** f(2.) + f(.070) * doh ** f(.6666667)
        den = f(1.) * doh ** f(2.) + f(.142) * doh ** f(.6666667) + f(.071)
        value = num / den
    value = jnp.where(h != 0., value, f(1.))
    value = jnp.minimum(jnp.maximum(value, f(0.)), f(1.))
    return jnp.where(d <= f(100.), f(0.), value)


def _sms_km(m, cfg, tke, theta, dth, rdzw, msftx, msfty,
            hpbl, dlk, d11, d22, d12, zx, zy, dx, dy):
    periodic = cfg.boundary == 'periodic'
    jr, ir = _mass_ranges(m, periodic)
    kr = (m.kts, m.ktf)
    r = lambda a, kk=kr, **d: m.r(a, kk, jr, ir, **d)  # noqa: E731
    mtx, mty = m.r2(msftx, jr, ir), m.r2(msfty, jr, ir)
    # tke_km: SMS Smagorinsky part (no isotropic override and no explicit cap).
    def2 = .25 * ((r(d11) - r(d22)) * (r(d11) - r(d22)))
    tmp = .25 * (r(d12) + r(d12, dj=1) + r(d12, di=1) + r(d12, di=1, dj=1))
    def2 = def2 + tmp * tmp
    mlen_h = jnp.sqrt(dx / mtx * dy / mty)
    strain = jnp.sqrt(def2)
    cs, ck = np.float32(cfg.c_s), np.float32(cfg.c_k)
    kh_s = cs * cs * mlen_h * mlen_h * strain
    kh_s = jnp.minimum(kh_s, 10. * mlen_h)
    dxm, dym = dx / mtx, dy / mty
    tmpzx = (.25 * (jnp.abs(r(zx)) + jnp.abs(r(zx, di=1))
                    + jnp.abs(r(zx, dk=1)) + jnp.abs(r(zx, di=1, dk=1))) * r(rdzw) * dxm)
    tmpzy = (.25 * (jnp.abs(r(zy)) + jnp.abs(r(zy, dj=1))
                    + jnp.abs(r(zy, dk=1)) + jnp.abs(r(zy, dj=1, dk=1))) * r(rdzw) * dym)
    alpha = jnp.maximum(jnp.sqrt(tmpzx * tmpzx + tmpzy * tmpzy), 1.)
    limit = jnp.maximum(10. / mlen_h, 1.e-3)
    kh_s = jnp.where(strain > limit, kh_s / (alpha * alpha), kh_s / alpha)
    hh_s = kh_s / _k(_PRANDTL, kh_s)

    # WRF tke_seed uses epsilon=1e-10 here (not tke_shear's epsilon=1e-15).
    seed = np.float32(0.)
    if int(cfg.isfflx) == 0 and (cfg.tke_drag_coefficient < np.float32(1.e-10)
                               and cfg.tke_heat_flux < np.float32(1.e-10)):
        seed = np.float32(1.e-6)
    tmp = jnp.sqrt(jnp.maximum(r(tke), seed))
    deltas = 1. / r(rdzw)
    dd = r(dth)
    # Unselected WRF IF branch is not evaluated on neutral columns.
    mlen_s = .76 * tmp / jnp.sqrt(jnp.where(dd > 0., jnp.abs(_G / r(theta) * dd), 1.))
    mlen_v = jnp.where(dd > 0., jnp.minimum(deltas, mlen_s), deltas)
    delxy = jnp.sqrt(dx / mtx * dy / mty)
    pth, pu = _partial(delxy, m.r2(hpbl, jr, ir), local=True), _partial(delxy, m.r2(hpbl, jr, ir), local=False)
    kh_t = ck * tmp * mlen_h
    kv_meso = .4 * tmp * r(dlk)
    kv_meso = kv_meso + np.float32(0.)  # literal xkmvo
    kv_les = ck * tmp * mlen_v
    kv = (1. - pu) * kv_les + pu * kv_meso
    kv = jnp.minimum(kv, 1000.)
    pr_inv_h = np.float32(np.float32(1.) / _PRANDTL)
    pr_inv_v_les = 1. + 2. * mlen_v / deltas
    hh_t = kh_t * pr_inv_h
    hv = kv * (1. - pth) * pr_inv_v_les + pth * (1. * kv + np.float32(0.))
    hv = jnp.minimum(hv, 1000.)
    kh = pth * kh_s + (1. - pth) * kh_t
    hh = pth * hh_s + (1. - pth) * hh_t
    return tuple(m.s(m.zeros(), kr, jr, ir, x.astype(m.dtype)) for x in (kh, kv, hh, hv, kv_meso))


def _sms_l_diss(m, cfg, tke, bn2, rdzw, msftx, msfty, hpbl, dlk, dx, dy):
    periodic = cfg.boundary == 'periodic'
    jr, ir = _mass_ranges(m, periodic)
    kr = (m.kts, m.ktf)
    l_scale, _ = _calc_l_scale(m, periodic, tke, bn2, rdzw, msftx, msfty, dx, dy)
    deltas = jnp.power(dx / m.r2(msftx, jr, ir) * dy / m.r2(msfty, jr, ir)
                       / m.r(rdzw, kr, jr, ir), np.float32(.33333333))
    ce1 = np.float32(np.float32(np.float32(cfg.c_k) / np.float32(.10)) * np.float32(.19))
    ce2 = np.float32(max(np.float32(0.), np.float32(np.float32(.93) - ce1)))
    kidx = jnp.arange(m.kts, m.ktf + 1)[:, None, None]
    coef = jnp.where((kidx == m.kts) | (kidx == m.ktf), np.float32(3.9), ce1 + ce2 * l_scale / deltas)
    delxy = jnp.sqrt(dx / m.r2(msftx, jr, ir) * dy / m.r2(msfty, jr, ir))
    pu = _partial(delxy, m.r2(hpbl, jr, ir), local=False)
    length = (1. - pu) * l_scale / coef + pu * m.r(dlk, kr, jr, ir) / _k(.3, tke)
    return m.s(m.zeros(), kr, jr, ir, length.astype(m.dtype))


def common_gradient(m,cfg,theta,p8w,t8w,rdz,rdzw):
    """Shared WRF tke_km/meso potential-temperature derivative, literal association."""
    from gpuwrf.dynamics.les3d_smagorinsky import _P1000MB,_RCP
    jr,ir=_mass_ranges(m,cfg.boundary=='periodic');ki=(m.kts+1,m.ktf-1)
    r=lambda x,kk,**d:m.r(x,kk,jr,ir,**d)
    out=m.zeros();tmp=1./r(rdz,ki,dk=1)+1./r(rdz,ki)
    out=m.s(out,ki,jr,ir,(r(theta,ki,dk=1)-r(theta,ki,dk=-1))/tmp)
    k1,kt=(m.kts,m.kts),(m.ktf,m.ktf)
    tmp=1./r(rdzw,k1,dk=1)+1./r(rdzw,k1)
    ts=r(t8w,k1)/jnp.power(r(p8w,k1)/_k(_P1000MB,p8w),_RCP)
    out=m.s(out,k1,jr,ir,(r(theta,k1,dk=1)-ts)/tmp)
    tmp=1./r(rdz,kt)+.5/r(rdzw,kt)
    tt=r(t8w,(m.kde,m.kde))/jnp.power(r(p8w,(m.kde,m.kde))/_k(_P1000MB,p8w),_RCP)
    return m.s(out,kt,jr,ir,(tt-r(theta,kt,dk=-1))/tmp)


def sms_tke_coefficients(inp: SmsInputs, cfg: SmsConfig) -> SmsCoefficients:
    """Return only SMS-specific coefficients and implicit dissipation length.

    All common WRF preprocessing is supplied as explicit savepoints. Outputs
    precede phy_bc and retain WRF's zero outer coefficient ring. This function
    does not implement or admit an operational km_opt=5 forecast.
    """
    if any(x is None for x in inp):
        raise ValueError('SMS coefficients require all WRF upstream operands')
    shape = inp.tke.shape
    if len(shape) != 3 or shape[0] < 4 or min(shape[1:]) < 10:
        raise ValueError('SMS operands require halo-padded mass-grid shape')
    if any(getattr(x, 'shape', None) != shape for x in inp[:11]):
        raise ValueError('SMS 3-D operands must match the mass-grid shape')
    if any(getattr(x, 'shape', None) != shape[1:] for x in inp[11:14]):
        raise ValueError('SMS surface operands must match the mass-grid shape')
    # _Mem has an unused top face and HALO=3 plus the staggered far edge.
    m = _Mem(shape[2] - 7, shape[1] - 7, shape[0] - 1, jnp.asarray(inp.tke).dtype)
    x = SmsInputs(*(jnp.asarray(a, dtype=m.dtype) for a in inp))
    coeffs = _sms_km(m, cfg, x.tke, x.theta, x.dthrdn, x.rdzw, x.msftx, x.msfty, x.hpbl, x.dlk,
                     x.d11, x.d22, x.d12, x.zx, x.zy, x.dx, x.dy)
    length = _sms_l_diss(m, cfg, x.tke, x.bn2, x.rdzw, x.msftx, x.msfty, x.hpbl, x.dlk, x.dx, x.dy)
    return SmsCoefficients(*coeffs, length)
