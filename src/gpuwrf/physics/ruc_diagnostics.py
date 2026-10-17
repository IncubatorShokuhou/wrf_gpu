"""Literal WRF module_sf_sfcdiags_ruclsm.F flux diagnostics (REAL arithmetic).

The temperature/humidity bounds below are WRF's diagnostic bounds, not model
state repairs. Exchange coefficients are inputs; surface_driver computes CQS
with the moisture availability AFTER LSMRUC.
"""

import jax.numpy as jnp


def _saturation(p, t, ice):
    water = (611.583699, 44.4606896, 1.43177157, .0264224321, .000299291081,
             .00000203154182, .00000000702620698, .00000000000379534310,
             -.0000000000000321582393)
    frozen = (609.868993, 49.9320233, 1.84672631, .0402737184, .000565392987,
              .00000521693933, .0000000307839583, .000000000105785160,
              .000000000000161444444)
    c = lambda v: jnp.asarray(v, t.dtype)
    x = jnp.maximum(c(-80.), t - c(273.16))

    def polynomial(coeff):
        e = c(coeff[-1])
        for value in reversed(coeff[:-1]):
            e = c(value) + x * e
        return c(.622) * e / (p - e)

    return jnp.where(ice, polynomial(frozen), polynomial(water))


def sfcdiags_ruclsm(*, hfx, qfx, tsk, qsfc, cqs, cqs2, chs2,
                   t1, qv1, rho1, p1, psfc, cp=1004.5, rovcp=287. / 1004.5):
    """Return ``T2, TH2, Q2`` as WRF SFCDIAGS_RUCLSM over mass columns.

    ``qsfc`` is specific humidity; ``qv1``/``q2`` are mixing ratios. ``p1`` is
    first-level hydrostatic pressure, ``psfc`` WRF PSFC (p8w at the surface).
    CHS and SNOW are unused by the pristine routine's selected flux method.
    """
    dtype = jnp.asarray(tsk).dtype
    hfx, qfx, tsk, qsfc, cqs, cqs2, chs2, t1, qv1, rho1, p1, psfc = (
        jnp.asarray(v, dtype) for v in
        (hfx, qfx, tsk, qsfc, cqs, cqs2, chs2, t1, qv1, rho1, p1, psfc))
    c = lambda v: jnp.asarray(v, dtype)
    fac = (c(1.e5) / psfc) ** c(rovcp)
    # jnp.where evaluates both branches: use a safe divisor only in the branch
    # WRF does not execute when an exchange coefficient is below its threshold.
    low_h = chs2 < c(1.e-5)
    th2 = jnp.where(low_h, t1 * fac,
                    tsk * fac - hfx / (rho1 * c(cp) * jnp.where(low_h, c(1.), chs2)))
    t2 = th2 * (c(1.e-5) * psfc) ** c(rovcp)
    t2 = jnp.minimum(jnp.maximum(tsk, t1), jnp.maximum(jnp.minimum(tsk, t1), t2))
    th2 = t2 * fac
    qsat = _saturation(p1, t1, t1 - c(273.15) <= c(0.))
    qlev1 = jnp.minimum(qsat, qv1)
    qsfcprox = qlev1 + qfx / (rho1 * cqs)
    qsfcmr = qsfc / (c(1.) - qsfc)
    low_q = cqs2 < c(1.e-5)
    q2 = jnp.where(low_q, qlev1,
                   qsfcprox - qfx / (rho1 * jnp.where(low_q, c(1.), cqs2)))
    q2 = jnp.minimum(jnp.maximum(qsfcmr, qlev1), jnp.maximum(jnp.minimum(qsfcmr, qlev1), q2))
    qsat = _saturation(psfc, t2, t2 - c(273.15) <= c(0.))
    return {"t2": t2, "th2": th2, "q2": jnp.minimum(qsat, q2)}
