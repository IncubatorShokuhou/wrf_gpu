"""Morrison two-moment (mp=10, mp=40) 10 cm reflectivity: WRF ``refl10cm_hm``.

Pristine WRF 4.7.1 ``module_mp_morr_two_moment.F:4502-4675`` (the mp=40 copy in
``module_mp_morr_two_moment_aero.F:5449-5623`` is arithmetic-identical; it only adds
``SAVE``). Constants follow ``MORR_TWO_MOMENT_INIT(morr_rimed_ice=0)`` (graupel,
IHAIL=0) and ``module_mp_radar.radar_init``. Inputs are the microphysics-exit columns
the caller passes (``qv1d, qr1d, nr1d, qs1d, ns1d, qg1d, ng1d, t1d, p1d``), trailing
bottom-to-top levels. Kinds are as declared in WRF: default-REAL float32 columns and
products, DOUBLE slopes/intercepts, melting-bin integral and log10. Integer powers use
gfortran's addition chains; constant divisors stay literal (no reciprocal rewrite).
The caller (``MP_MORR_TWO_MOMENT``) clamps with ``MAX(-35., dBZ)``.
"""
from __future__ import annotations

import cmath
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from gpuwrf.diagnostics.thompson_radar import _div, _mix_eps

F = np.float32
D = jnp.float64
R = jnp.float32

PI = F(3.1415926535897932384626434)           # module PARAMETER (REAL)
R_D = F(287.)                                   # module_model_constants r_d
NRBINS = 50


def _gammln(xx):
    """module_mp_radar GAMMLN: DOUBLE work, REAL result."""
    cof = (76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
           .1208650973866179e-2, -.5395239384953e-5)
    x = float(F(xx)); y = x
    tmp = x + 5.5
    tmp = (x + 0.5) * math.log(tmp) - tmp
    ser = 1.000000000190015
    for c in cof:
        y = y + 1.
        ser = ser + c / y
    return F(tmp + math.log(2.5066282746310005 * ser / x))


def _wgamma(y):
    return F(math.exp(float(_gammln(y))))


def _bins(upper):
    """radar_init diameter bins: geometric edges 100e-6..upper (DOUBLE), midpoints and widths."""
    edges = [100.e-6] + [math.exp((n - 1) / NRBINS * math.log(upper / 100.e-6) + math.log(100.e-6))
                         for n in range(2, NRBINS + 1)] + [upper]
    return (tuple(math.sqrt(edges[n] * edges[n + 1]) for n in range(NRBINS)),
            tuple(edges[n + 1] - edges[n] for n in range(NRBINS)))


def _permittivities():
    """radar_init m_w_0/m_i_0 at T=0 C (module_mp_radar.F:235-306), WRF literal kinds.

    REAL literals (273.16, 1e-2, 1.25664, ...) enter as rounded float32; the water value
    is SQRT of a default-kind CMPLX, i.e. a single-precision csqrt (taken correctly
    rounded); the ice value is a DOUBLE csqrt of CONJG with a REAL-rounded imaginary part.
    Returned as permittivities m**2 (DOUBLE) for the Maxwell-Garnett mixing.
    """
    f = lambda x: float(F(x))
    t = 0.0
    lam = f(.10)
    tk = t + f(273.16)
    epsinf = 5.27137 + 0.02164740 * t - 0.00131198 * t * t
    epss = 78.54 * (1.0 - 4.579e-3 * (t - 25.0) + 1.190e-5 * (t - 25.0) * (t - 25.0)
                    - 2.800e-8 * (t - 25.0) * (t - 25.0) * (t - 25.0))
    alpha = -16.8129 / tk + 0.0609265
    lambdas = 0.00033836 * math.exp(2513.98 / tk) * f(1e-2)
    pix = 3.1415926535897932384626434
    ratio = lambdas / lam
    nenner = 1. + 2. * ratio ** (1. - alpha) * math.sin(alpha * pix * 0.5) + ratio ** (2. - 2. * alpha)
    epsr = epsinf + ((epss - epsinf) * (ratio ** (1. - alpha) * math.sin(alpha * pix * 0.5) + 1.)) / nenner
    epsi = ((epss - epsinf) * (ratio ** (1. - alpha) * math.cos(alpha * pix * 0.5) + 0.)) / nenner \
        + lam * f(1.25664) / f(1.88496)
    root = cmath.sqrt(complex(f(epsr), -f(epsi)))
    m_w = complex(f(root.real), f(root.imag))
    c, freq = 2.99e8, 2.99e8 / lam * 1e-9
    b1, b2, b = f(0.0207), 1.16e-11, 335.0
    deltabeta = math.exp(f(-10.02) + f(0.0364) * (tk - f(273.16)))
    betam = (b1 / tk) * (math.exp(b / tk) / ((math.exp(b / tk) - 1) ** 2)) + b2 * freq * freq
    theta = 300. / tk - 1.
    alfa = (0.00504 + 0.0062 * theta) * math.exp(-22.1 * theta)
    del c
    ice = complex(f(3.1884) + f(9.1e-4) * (tk - f(273.16)), f(alfa / freq + (betam + deltabeta) * freq))
    m_i = cmath.sqrt(ice.conjugate())
    return m_w, m_i


def _constants():
    rhow, rhosn, rhog = F(997.), F(100.), F(400.)    # IHAIL=0: graupel density
    six = F(6.)
    am = {'r': F(F(PI * rhow) / six), 's': F(F(rhosn * PI) / six), 'g': F(F(rhog * PI) / six)}
    c = {}
    for key, a in am.items():
        bm, mu = F(3.), F(0.)
        ce = (F(1.) + bm, F(1.) + mu, F(F(1.) + bm) + mu, F(F(1.) + F(F(2.) * bm)) + mu)
        cg = tuple(_wgamma(x) for x in ce)
        c[key] = dict(am=a, bm=bm, mu=mu, ce=ce, cg=cg, og2=F(F(1.) / cg[1]), obm=F(F(1.) / bm))
    for key in ('s', 'g'):
        oam = F(F(1.) / c[key]['am'])
        c[key]['ocm'] = F(math.pow(float(oam), float(c[key]['obm'])))  # REAL xoams**xobms
    pi5 = F(F(F(F(F(3.14159) * F(3.14159)) * F(3.14159)) * F(3.14159)) * F(3.14159))
    lamda = float(F(.10))                                             # DOUBLE PARAMETER from REAL literal
    m_w, m_i = _permittivities()
    ew, ei = m_w * m_w, m_i * m_i
    kw = abs((ew - 1.) / (ew + 2.)) ** 2
    simpson = [0.] * (NRBINS + 1)
    for n in range(0, NRBINS - 1, 2):
        for offset, weight in enumerate((1. / 3., 4. / 3., 1. / 3.)):
            simpson[n + offset] += weight
    return dict(c=c, pi5=float(pi5), lamda4=lamda * lamda * lamda * lamda, kw=kw, m_w=m_w, m_i=m_i,
                ew=(ew.real, ew.imag), ei=(ei.real, ei.imag),
                simpson=tuple(simpson[:NRBINS]), snow_bins=_bins(.02), graupel_bins=_bins(.05))


CONSTANTS = _constants()


def _opaque(x, dtype):
    """Keep a constant divisor literal (XLA algsimp A/C -> A*(1/C), FINDINGS E123/E130)."""
    return lax.optimization_barrier(jnp.asarray(x, dtype))


def _pow3(x):
    return x * (x * x)                          # gfortran x**3 = x * x**2


def _pow6(x):
    x3 = _pow3(x)
    return x3 * x3                              # gfortran x**6 = x**3 * x**3


def _soak(x_g, a_geo, b_geo, fmelt, c):
    """``rayleigh_soak_wetgraupel`` (module_mp_radar.F:309-402), default maxwellgarnett mixing."""
    fm = jnp.maximum(jnp.minimum(fmelt, 1.), 0.)
    mra = .9 + (1. - .9) * fm
    x_w = x_g * fm
    d_g = a_geo * x_g ** b_geo
    pix = 3.1415926535897932384626434
    vg = pix / 6. * _pow3(d_g)
    rhog = jnp.maximum(jnp.minimum(x_g / vg, 900.), 10.)
    vg = x_g / rhog
    grenz = 1. - rhog / _opaque(1000., D)
    fmgrenz = (900. - rhog) / (mra * 900. - rhog + 900. * rhog / _opaque(1000., D))
    volg = jnp.where(mra <= grenz, vg * (1. - mra * fm),
                     jnp.where(fm <= fmgrenz, (1. - mra * fm) * vg,
                               (x_g - x_w) / _opaque(900., D) + x_w / _opaque(1000., D)))
    d_large = (6. / pix * volg) ** float(F(1.) / F(3.))
    volice = (x_g - x_w) / (volg * 900.)
    volwater = x_w / (1000. * volg)
    volair = 1. - volice - volwater
    iw = _mix_eps(tuple(jnp.asarray(v, D) for v in c['ew']), tuple(jnp.asarray(v, D) for v in c['ei']),
                  volice / jnp.maximum(volice + volwater, 1e-10))
    mixed = _mix_eps(iw, (jnp.asarray(1., D), jnp.asarray(0., D)), volair)
    q = _div((mixed[0] - 1., mixed[1]), (mixed[0] + 2., mixed[1]))
    back = (q[0] * q[0] + q[1] * q[1]) * c['pi5'] * _pow6(d_large) / _opaque(c['lamda4'], D)
    return jnp.where(d_g >= 1e-12, back, 0.)


def _species(q1d, n1d, rho, p, threshold):
    """Mass/number densities, DOUBLE slope inverse and intercept for one species (REAL RHS first)."""
    active = q1d > threshold
    rq = jnp.where(active, q1d * rho, F(1.e-12))
    nq = jnp.where(active, n1d * rho, F(1.e-12))
    lam = ((((p['am'] * p['cg'][2]) * p['og2']) * nq) / jnp.where(active, rq, F(1.))) ** p['obm']
    lam = lam.astype(D)
    ilam = 1. / lam
    n0 = (nq * p['og2']).astype(D) * lam ** float(p['ce'][1])
    return active, rq, ilam, n0


def morrison_reflectivity(qv1d, qr1d, nr1d, qs1d, ns1d, qg1d, ng1d, t1d, p1d):
    """WRF ``refl10cm_hm`` raw dBZ on ``(..., nz)`` microphysics-exit columns (REAL)."""
    c = CONSTANTS
    pr, ps, pg = c['c']['r'], c['c']['s'], c['c']['g']
    arrays = [jnp.asarray(x, R) for x in (qv1d, qr1d, nr1d, qs1d, ns1d, qg1d, ng1d, t1d, p1d)]
    qv1d, qr1d, nr1d, qs1d, ns1d, qg1d, ng1d, temp, pres = arrays
    nz = temp.shape[-1]
    qv = jnp.maximum(F(1.e-10), qv1d)
    rho = (F(.622) * pres) / ((R_D * temp) * (qv + F(.622)))
    thr = F(1.e-9)
    l_qr, rr, ilamr, n0r = _species(qr1d, nr1d, rho, pr, thr)
    l_qs, rs, ilams, n0s = _species(qs1d, ns1d, rho, ps, thr)
    l_qg, rg, ilamg, n0g = _species(qg1d, ng1d, rho, pg, thr)

    # Melting level: first k from the top (kte-1..kts) with T>273.15, rain and ice above.
    k = jnp.arange(nz)
    above = lambda a: jnp.concatenate((a[..., 1:], jnp.zeros_like(a[..., :1])), axis=-1)
    trigger = (k < nz - 1) & (temp > F(273.15)) & l_qr & (above(l_qs) | above(l_qg))
    k0 = jnp.max(jnp.where(trigger, k + 1, 0), axis=-1, keepdims=True)
    melti = jnp.any(trigger, axis=-1, keepdims=True)

    def ze_dry(active, n0, ilam, p, factor):
        # Left-to-right: [REAL prefactor *] N0 * xc?g(4) * ilam**xc?e(4)
        lead = n0 if factor is None else float(factor) * n0
        value = lead * float(p['cg'][3]) * ilam ** float(p['ce'][3])
        return jnp.where(active, value.astype(R), F(1.e-22))

    six_pi = F(F(6.) / PI)
    pre = F(F(F(.176) / F(.93)) * six_pi) * six_pi
    factor_s = F(F(pre * F(ps['am'] / F(900.))) * F(ps['am'] / F(900.)))
    factor_g = F(F(pre * F(pg['am'] / F(900.))) * F(pg['am'] / F(900.)))
    ze_rain = ze_dry(l_qr, n0r, ilamr, pr, None)
    ze_snow = ze_dry(l_qs, n0s, ilams, ps, factor_s)
    ze_graupel = ze_dry(l_qg, n0g, ilamg, pg, factor_g)

    take = lambda a: jnp.take_along_axis(a, k0, axis=-1)
    melt_levels = melti & (k0 >= 1) & (k < k0)

    def wet(active, rq, ilam, n0, p, bins):
        wet_at = melt_levels & active & take(active)
        fmelt = jnp.maximum(.005, jnp.minimum(1. - (rq / take(rq)).astype(D), .99))
        lam = 1. / ilam
        diam, width = (jnp.asarray(b, D) for b in bins)
        a_geo, b_geo = float(p['ocm']), float(p['obm'])
        simpson = jnp.asarray(c['simpson'], D)

        def body(n, eta):
            x = float(p['am']) * diam[n] ** float(p['bm'])
            back = _soak(x, a_geo, b_geo, fmelt, c)
            f_d = n0 * diam[n] ** float(p['mu']) * jnp.exp(-lam * diam[n])
            return eta + f_d * back * simpson[n] * width[n]
        eta = lax.fori_loop(0, NRBINS, body, jnp.zeros(rq.shape, D))
        value = (c['lamda4'] / (c['pi5'] * c['kw']) * eta).astype(R)
        return wet_at, value

    def melting(zs, zg):
        ws, vs = wet(l_qs, rs, ilams, n0s, ps, c['snow_bins'])
        wg, vg = wet(l_qg, rg, ilamg, n0g, pg, c['graupel_bins'])
        return jnp.where(ws, vs, zs), jnp.where(wg, vg, zg)

    ze_snow, ze_graupel = lax.cond(jnp.any(melt_levels), melting, lambda zs, zg: (zs, zg), ze_snow, ze_graupel)
    total = (ze_rain + ze_snow) + ze_graupel
    return (10. * jnp.log10(total.astype(D) * 1.e18)).astype(R)


def morrison_refl_10cm(*columns):
    """Caller value ``refl_10cm = MAX(-35., dBZ)`` (module_mp_morr_two_moment.F:913-916)."""
    return jnp.maximum(F(-35.), morrison_reflectivity(*columns))
