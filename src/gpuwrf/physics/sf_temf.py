"""WRF TEMF surface layer, ``module_sf_temfsfclay.F:154–384``.

This is a column/batch kernel, not operational qualification of sfclay=10.
Inputs use the surface_driver caller convention: mass-level theta, Exner,
pressure in Pa, density in kg/m³, absolute mass height and terrain in m,
incoming HFX in W/m², and the preceding TEMF energy/thermal-height/velocity.
Every argument and result is WRF REAL (binary32). There are no DOUBLE islands.
The WRF availability divisions are retained; a dry zero-availability column
has WRF's undefined exchange coefficient and must not be silently repaired.
"""

from __future__ import annotations

import jax.numpy as jnp


def temf_surface(*, u, v, theta, qv, pressure, exner, rho, z, terrain,
                 psfc, roughness, mavail, tsk, coriolis, hfx, te_temf,
                 hd_temf, wm_temf):
    """Return WRF surface outputs, including the two updated TEMF carries.

    All operands broadcast over columns. ``te_temf`` is its lowest-level
    value; callers retain all other energy levels. ``wm_temf`` is initialized
    to zero on cold start and held across calls, as in temfsfclayinit.
    """
    r = jnp.float32
    u, v, theta, qv, pressure, exner, rho, z, terrain, psfc, roughness, mavail, tsk, coriolis, hfx, te_temf, hd_temf, wm_temf = (
        jnp.asarray(x, r) for x in (u, v, theta, qv, pressure, exner, rho, z, terrain,
                                    psfc, roughness, mavail, tsk, coriolis, hfx, te_temf,
                                    hd_temf, wm_temf))
    cp, g, xlv, karman = r(1004.5), r(9.81), r(2.5e6), r(0.4)
    ep2 = r(287.) / r(461.6)
    ftau0, fth0, ceps = r(.17), r(.145), r(.070)
    prt0 = ceps / ceps * ftau0**2 / r(2.) / fth0**2
    one = r(1.)
    e1 = r(.6112) * jnp.exp(r(17.67) * (tsk-r(273.15)) / (tsk-r(29.65)))
    qsfc = ep2 * e1 / (psfc/r(1000.)-e1)
    qsfc_air = qsfc * mavail
    thetav = tsk/exner * (one+r(.608)*qsfc_air)
    zm = z-terrain
    zt = (z-terrain-roughness)/r(2.)
    dthdz = (theta-tsk/exner) / (zt*jnp.log10(zm/roughness))
    dudz = u / (zt*jnp.log10(zm/roughness))
    dvdz = v / (zt*jnp.log10(zm/roughness))
    te = jnp.maximum(te_temf, r(1e-3))
    wstr = jnp.where(hfx > 0, g*hd_temf/thetav*(hfx/(rho*cp)), r(0.))**(one/r(3.))
    n2 = g/thetav*dthdz
    shear = jnp.sqrt(dudz**2+dvdz**2)
    ri = n2/jnp.where(shear < r(1e-15), one, shear**2)
    ri = jnp.where(shear < r(1e-15), jnp.where(n2 >= 0, r(.2), r(-1.)), ri)
    ri = jnp.minimum(ri, r(.2))
    beta = g/thetav
    stable = ri > 0
    ratio = jnp.where(stable,
                      ri/(ceps**2*ftau0**2/(r(2.)*ceps**2*fth0**2)+r(3.)*ri),
                      ri/(ceps**2*ftau0**2/(-r(2.)*ceps**2*fth0**2)+r(2.)*ri))
    ftau = jnp.where(stable, ftau0*((r(3.)/r(4.))/(one+r(4.)*ri)+one/r(4.)), ftau0)
    fth = jnp.where(stable, fth0/(one+r(4.)*ri), fth0)
    tke = te*(one-ratio)
    ustrtilde = jnp.sqrt(ftau*tke)
    linv = one/karman/zt+jnp.abs(coriolis)/(r(.185)*ustrtilde)
    linv = jnp.where(n2 > 0, linv+jnp.sqrt(jnp.maximum(n2, r(0.)))/(r(2.)*ustrtilde), linv)
    leps = jnp.maximum(one/linv, r(0.))
    ust = jnp.sqrt(ftau/ftau0)*jnp.sqrt(u**2+v**2+(r(.5)*wstr)**2)*leps/jnp.log(zm/roughness)/zt
    wm = (ust+wm_temf)/r(2.)
    flhc = rho*cp*fth/fth0*wm*leps/prt0/jnp.log(zm/roughness)/zt
    flqc = rho*fth/fth0*wm*leps/prt0/jnp.log(zm/roughness)/zt*mavail
    out_hfx = flhc*(tsk-theta*exner)
    qfx = jnp.maximum(flqc*(qsfc-qv), r(0.))
    t2 = (tsk/exner+(theta-tsk/exner)*jnp.log(r(2.)/roughness)/jnp.log(zm/roughness))*exner
    e1 = r(.6112)*jnp.exp(r(17.67)*(theta*exner-r(273.15))/(theta*exner-r(29.65)))
    return dict(te_temf=te, wm_temf=jnp.maximum(wm, r(1e-2)), ust=ust,
                flhc=flhc, flqc=flqc, exch_temf=flqc/mavail, chs=flqc/rho/mavail,
                hfx=out_hfx, qfx=qfx, lh=xlv*qfx, qsfc=qsfc,
                u10=u*jnp.log(r(10.)/roughness)/jnp.log(zm/roughness),
                v10=v*jnp.log(r(10.)/roughness)/jnp.log(zm/roughness),
                t2=t2, th2=t2/exner,
                q2=qsfc_air+(qv-qsfc_air)*jnp.log(r(2.)/roughness)/jnp.log(zm/roughness),
                chs2=fth/fth0*wm*leps/prt0/jnp.log(r(2.)/roughness)/zt,
                cqs2=fth/fth0*wm*leps/prt0/jnp.log(r(2.)/roughness)/zt,
                qgh=ep2*e1/(pressure/r(1000.)-e1), cpm=cp*(one+r(.8)*qv))
