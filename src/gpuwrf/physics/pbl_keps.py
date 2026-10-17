"""Experimental REAL32 KEPS from WRF 4.7.1 ``module_bl_keps.F``.

Full columns and defined helpers retain WRF operation order. Explicit zeros at
the upstream undefined ttke/radsum/top-gradient sites mirror the authorized
compiler-zero reference. Alternate initialization materially changes results.
Levels are bottom-up, n mass levels and n+1 interfaces; no DOUBLE islands exist.
The frozen CPU gate requires a non-contracting, non-reassociating target.
GPU and coupled-forecast qualification remain pending; see docs/EEPS_KEPS.md.
"""
from __future__ import annotations

from functools import wraps

import jax
import jax.numpy as jnp


def _real32(fn):
    @wraps(fn)
    def real(*args, **kwargs):
        with jax.enable_x64(False):
            def real_arg(x):
                a = jnp.asarray(x)
                return a if jnp.issubdtype(a.dtype, jnp.integer) or a.dtype == jnp.bool_ else a.astype(jnp.float32)
            return fn(*jax.tree.map(real_arg, args), **kwargs)
    return real


def _round(x):
    """Keep the stable-C cancellation associated as in WRF.

    This prevents HLO algebraic folding across assignments. It does not
    disable machine-level CPU FMA contraction; that limitation is separate.
    """
    return jax.lax.optimization_barrier(x)


@_real32
def shear(u, v, dz):
    """WRF SHEAR :676–696, squared wind gradient [s-2]."""
    u2 = (dz[2:] * u[1:-1] + dz[1:-1] * u[2:]) / (dz[1:-1] + dz[2:])
    u1 = (dz[1:-1] * u[:-2] + dz[:-2] * u[1:-1]) / (dz[:-2] + dz[1:-1])
    v2 = (dz[2:] * v[1:-1] + dz[1:-1] * v[2:]) / (dz[1:-1] + dz[2:])
    v1 = (dz[1:-1] * v[:-2] + dz[:-2] * v[1:-1]) / (dz[:-2] + dz[1:-1])
    return jnp.pad(((u2-u1)/dz[1:-1])**2. + ((v2-v1)/dz[1:-1])**2., (1, 1))


@_real32
def buoyancy(g, theta, theta0, dz, tke, tpe, prandtl):
    """WRF BUOY :629–648, including potential-energy contribution [s-2]."""
    phi = g/theta0[1:-1]*tpe[1:-1]/tke[1:-1]*prandtl[1:-1]*.09/.09
    lo = 2.*(theta[1:-1]-theta[:-2])/(dz[:-2]+dz[1:-1])
    hi = 2.*(theta[2:]-theta[1:-1])/(dz[2:]+dz[1:-1])
    gradient = .5*(lo+hi)
    return jnp.pad(-gradient*g/theta0[1:-1]+(g/theta0[1:-1])*phi, (1, 1))


@_real32
def exchange(tke, epsilon, dz, prandtl_face):
    """WRF CDTUR_KE :522–541, momentum/heat diffusivities [m2 s-1]."""
    km = (.09*tke[:-1]**2/epsilon[:-1]*dz[:-1]
          + .09*tke[1:]**2/epsilon[1:]*dz[1:])/(dz[1:]+dz[:-1])
    km = jnp.maximum(km, .09)
    kh = km/prandtl_face[1:-1]
    return (jnp.concatenate((jnp.zeros_like(km[:1]), km, km[-1:])),
            jnp.concatenate((jnp.zeros_like(kh[:1]), kh, kh[-1:])))


@_real32
def invert(lower, diagonal, upper, rhs):
    """WRF INVERT :701–723, reverse elimination then forward substitution."""
    n = rhs.shape[0]
    def eliminate(i, carry):
        diag, c = carry
        k = n-2-i
        c = c.at[k].set(c[k]-upper[k]*c[k+1]/diag[k+1])
        diag = diag.at[k].set(diag[k]-upper[k]*lower[k+1]/diag[k+1])
        return diag, c
    diag, c = jax.lax.fori_loop(0, n-1, eliminate, (diagonal, rhs))
    def substitute(k, c):
        return c.at[k].set(c[k]-lower[k]*c[k-1]/diag[k-1])
    c = jax.lax.fori_loop(1, n, substitute, c)
    return c/diag


@_real32
def diffusion(dt, concentration, rho, rho_face, diffusivity, implicit, explicit,
              surface, volume, dz, *, first=1, fixed_top=1):
    """WRF DIFF :547–624; ``first`` is WRF's 1-based iz1.

    ``fixed_top`` corresponds to izf (1 holds the top mass-level value).
    Return the new concentration and WRF's interface flux prefix [co m s-1].
    All interface operands have n+1 elements; mass operands have n elements.
    """
    n = concentration.shape[0]
    cddz = jnp.concatenate((
        (surface[0]*rho_face[0]*diffusivity[0]/dz[0])[None],
        2.*surface[1:-1]*rho_face[1:-1]*diffusivity[1:-1]/(dz[1:]+dz[:-1]),
        (surface[-1]*rho_face[-1]*diffusivity[-1]/dz[-1])[None]))
    dzv = volume*dz
    lower = -cddz[:-1]*dt/dzv/rho
    diag = 1+dt*(cddz[:-1]+cddz[1:])/dzv/rho-implicit*dt
    upper = -cddz[1:]*dt/dzv/rho
    rhs = concentration+explicit*dt
    active = (jnp.arange(n) >= first-1) & (jnp.arange(n) < n-fixed_top)
    solution = invert(jnp.where(active, lower, 0.), jnp.where(active, diag, 1.),
                      jnp.where(active, upper, 0.), jnp.where(active, rhs, concentration))
    flux = jnp.concatenate((jnp.zeros_like(solution[:1]),
                            -(cddz[1:n]*(solution[1:]-solution[:-1]))/rho[1:]))
    # Literal FC(kts:iz1)=0; flux above that includes the fixed top.
    flux = jnp.where(jnp.arange(n) < first, 0., flux)
    return solution, flux


@_real32
def tpe_terms(gradient, g, theta0, tke, epsilon, tpe, prandtl):
    """WRF TERMS_TPE :1131–1147; caller must supply EVERY gradient level."""
    awt = 1./tke/tpe/2.*(.09/prandtl*tke**2./epsilon*gradient)**2.
    coeff = jnp.maximum(.55, 3./2.*(1.+awt))
    implicit = -g/theta0*.09*tke/epsilon*gradient-coeff*epsilon/tke
    explicit = .09*tke**2./epsilon*gradient**2./prandtl
    return implicit, explicit, coeff


@_real32
def countergradient(dz, g, theta0, tke, epsilon, tpe, source):
    """WRF CALC_COUNTERGRADIENT :1151–1176, updated theta source [K s-1]."""
    phi = g/theta0*tke*tpe*.09/epsilon
    lo = 2.*(phi[1:-1]-phi[:-2])/(dz[:-2]+dz[1:-1])
    hi = 2.*(phi[2:]-phi[1:-1])/(dz[2:]+dz[1:-1])
    return source-jnp.pad(.5*(lo+hi), (1, 1))


@_real32
def analytic_tke(dt, dz, roughness, shear2, buoy, tke, epsilon, prandtl,
                 length, ustar, phi_epsilon, phi_m, richardson, nsquare,
                 epsilon_source):
    """WRF THETAPHI_ALPHABETA :883–944; all three C-sign branches.

    The first-level MOST prescription and the large TKE/epsilon-ratio rescue
    are part of WRF, not extra guards. No final driver clamps are applied here.
    """
    beta = jnp.asarray(-1., tke.dtype)/jnp.asarray(1.92, tke.dtype)
    f = jnp.asarray(1.92, tke.dtype)-1.
    d = jnp.where((jnp.abs(tke[1:])/jnp.abs(epsilon[1:]) > 1.e4)
                  & (epsilon[1:] <= 1.e-6),
                  1/1.4*tke[1:]**(3./2.)/length[1:], epsilon[1:])
    theta = jnp.maximum(1.e-4, tke[1:])/jnp.maximum(1.e-7, d)
    phi = jnp.maximum(1.e-7, d)**beta*jnp.maximum(1.e-4, tke[1:])
    c = .09*((jnp.asarray(1.44, tke.dtype)-1.)*shear2[1:]
             + ((jnp.asarray(1.44, tke.dtype)-1.)*buoy[1:])/prandtl[1:])
    a = .09*(shear2[1:]+buoy[1:]/prandtl[1:])
    b = .09*(1.44*shear2[1:]+1.44*buoy[1:]/prandtl[1:])
    # Select safe operands for inactive divisions; never alter the chosen branch.
    cn = jnp.where(c < 0., -c, 1.)
    cp = jnp.where(c > 0., c, 1.)
    numerator = _round(f**(1./2.)*_round(_round(_round(cn*theta**2.)-1.92)+1))
    tanh_arg = _round(_round(cn**(1./2.)*dt)*f**(1./2.))
    denominator = _round(_round(cn*theta)*_round(f**(1./2.)+_round(
        _round(cn**(1./2.)*theta)*jnp.tanh(tanh_arg))))
    negative = _round(_round(f/_round(cn*theta))+_round(numerator/denominator))
    positive = ((f/cp)**.5*(jnp.tanh(dt*(cp*f)**.5)+theta*(cp/f)**.5))/(
        theta*jnp.tanh(dt*(cp*f)**.5)*(cp/f)**.5+1.)
    theta_new = jnp.maximum(1., jnp.where(c < 0., negative,
                            jnp.where(c == 0., theta+f*dt, positive)))
    phi_new = phi*jnp.exp(theta_new*dt*(a+beta*b))
    te_new = (phi_new*theta_new**beta)**(1./(1.+beta))
    d_new = te_new/theta_new
    src = epsilon_source+jnp.where(richardson > 0., .27*jnp.minimum(
        1., jnp.sqrt(jnp.maximum(richardson, 0.)/.08))*jnp.sqrt(jnp.maximum(0., -nsquare)), 0.)
    bottom_d = (ustar**3./.4/(dz[0]/2.-roughness))*phi_epsilon*(1.+src[0]*epsilon[0]*dt)
    bottom_te = (ustar**2.)/jnp.sqrt(jnp.asarray(.09, tke.dtype))*jnp.sqrt(phi_epsilon/phi_m)
    return (jnp.concatenate((bottom_te[None], te_new)),
            jnp.concatenate((bottom_d[None], d_new)), src)


@_real32
def pbl_height(theta, qv, tke, dz, z):
    """GET_PBLH :775–879 with compiler-zero semantics of upstream ttke :822."""
    n = theta.shape[0]
    thv = theta*(1.+.61*qv)
    def find_min(c):
        k, value, index = c
        hit = value > thv[k]
        return k+1, jnp.where(hit, thv[k], value), jnp.where(hit, k, index)
    _, minimum, kthv = jax.lax.while_loop(
        lambda c: (c[0] < n) & (z[c[0]] <= 500.), find_min,
        (jnp.int32(1), jnp.float32(9.e9), jnp.int32(0)))
    # WRF GET_PBLH:822 reads ttke, not the assigned tke. Authorized compiler
    # zero initialization leaves maxtke=0 and ktke=1 (Fortran indexing).
    threshold = jnp.float32(.025/2.)
    def theta_height(c):
        k, height = c
        candidate = z[k]-dz[k-1]*jnp.minimum(
            (thv[k]-(minimum+1.5))/jnp.maximum(thv[k]-thv[k-1], 1.e-6), 1.)
        height = jnp.where(thv[k] >= minimum+1.5, candidate, height)
        return k+1, jnp.where(k+1 == n-2, z[1], height)
    _, zi = jax.lax.while_loop(lambda c: (c[1] == 0.) & (c[0] < n), theta_height,
                              (kthv+1, jnp.float32(0.)))
    def tke_height(c):
        k, height = c
        te, previous = jnp.maximum(tke[k]/2., 0.), jnp.maximum(tke[k-1]/2., 0.)
        candidate = z[k]-dz[k-1]*jnp.minimum((threshold-te)/jnp.maximum(previous-te, 1.e-6), 1.)
        height = jnp.where(te <= threshold, jnp.maximum(candidate, z[1]), height)
        return k+1, jnp.where(k+1 == n-2, z[1], height)
    _, htke = jax.lax.while_loop(lambda c: (c[1] == 0.) & (c[0] < n), tke_height,
                                (jnp.int32(1), jnp.float32(0.)))
    weight = .5*jnp.tanh((zi-200.)/400.)+.5
    height = htke*(1.-weight)+zi*weight
    kpbl = jnp.int32(1)+jnp.sum((z[:-1] <= height).astype(jnp.int32))
    return height, kpbl


@_real32
def stability(theta, qv, dz, z, rho, raten, exner, p8w, sh, nsquare,
              br, fm, fhh, pblh, kpbl, ustar, roughness, sflux):
    """SURFACE_BL_PRA_RI :1022–1126, zero radsum at upstream :1065–1068."""
    ri = jnp.minimum(-nsquare/jnp.where(sh != 0., sh, 1.), 100.)
    ri = jnp.where(sh < 1.e-15, jnp.where(-nsquare > 0., 10., -1.), ri)
    ri = ri.at[0].set(br).at[-1].set(ri[-2])
    def radiation(k, total):
        flux = raten[k]*exner[k]
        flux = flux*1004.5/9.81*(p8w[k]-p8w[k+1])
        return jnp.where(flux < 0., jnp.abs(flux)+total, total)
    # Compiler-defined initialization at WRF KEPS :1065–1068, not a body fix.
    rad = jax.lax.fori_loop(0, kpbl-1, radiation, jnp.float32(0.))
    rad = jnp.maximum(rad, 0.)/rho[kpbl-2]/1004.5
    unstable = br < 0.
    zl = dz[0]/2.
    zol = jnp.maximum(br*fm*fm/fhh, -100.)
    zol = (zl+roughness)/zl*jnp.where(unstable, jnp.minimum(zol, -1.e-8), jnp.maximum(zol, 1.e-8))
    hol = zol*pblh/zl*.1
    um, uh = 1.-16.*jnp.minimum(zol, 0.), 1.-16.*jnp.minimum(hol, 0.)
    phim = jnp.where(unstable, um**(-1./4.), 1.+5.*zol)
    phih = jnp.where(unstable, um**(-1./2.), 1.+5.*zol)
    phieps = jnp.where(unstable, 1.-zol, (1.+2.5*jnp.maximum(zol, 0.)**.6)**(3./2.))
    phim_sl = jnp.where(unstable, uh**(-1./4.), 1.+5.*hol)
    phih_sl = jnp.where(unstable, uh**(-1./2.), phim_sl)
    w3 = jnp.where(unstable, 9.81/theta[0]*jnp.maximum(sflux, 0.)*pblh, 0.)
    w3rad = jnp.where(unstable, 9.81/(theta[kpbl-2]*(1.+.606271777*qv[kpbl-2]))*rad*pblh, 0.)
    prfac2 = 15.9*(w3+w3rad)/ustar**3/(1.+4.*.4*(w3+w3rad)/ustar**3)
    pr0 = (phih_sl/phim_sl+6.8*.4*.1)/(1.+prfac2*.4*.1)
    pr0 = jnp.maximum(jnp.minimum(pr0, 4.), .25)
    pr_unstable = 1.+(pr0-1.)*jnp.exp(-3.*jnp.maximum(z[1:]-.1*pblh, 0.)**2./pblh**2.)
    pr_stable = jnp.maximum(1., jnp.minimum(4., 1.+3*ri))
    pr = jnp.where(kpbl > 1, jnp.where(unstable, pr_unstable, pr_stable), jnp.full_like(theta, 4.))
    interior = (pr[:-1]*dz[1:]+pr[1:]*dz[:-1])/(dz[1:]+dz[:-1])
    prface = jnp.concatenate((jnp.zeros_like(interior[:1]), interior, interior[-1:]))
    surface_gradient = -sflux/ustar/.4*phih/(dz[0]/2.-roughness)
    return ri, pr, prface, phim, phieps, surface_gradient


@_real32
def length_bougeault(theta, tke, dz, z):
    """WRF LENGTH_BOUGEAULT :949–1017, with its literal division association."""
    n = theta.shape[0]
    def level(k):
        beta = jnp.float32(9.81)/jnp.float32(300.)
        def upward(j, c):
            length, energy, previous, distance = c
            dzt = (dz[j+1]+dz[j])/2.
            energy = _round(energy-beta*theta[k]*dzt)
            energy = energy+beta*(theta[j+1]+theta[j])*dzt/2.
            distance = distance+dzt
            bbb = (theta[j+1]-theta[j])/dzt
            discriminant = jnp.maximum(0., (beta*(theta[j]-theta[k]))**2.+2.*bbb*beta*(tke[k]-previous))
            grad = (-beta*(theta[j]-theta[k])+jnp.sqrt(discriminant))/jnp.where(bbb != 0., bbb, 1.)/beta
            flat = jnp.where(theta[j] != theta[k], (tke[k]-previous)/jnp.where(theta[j] != theta[k], beta*(theta[j]-theta[k]), 1.), 0.)
            tl = jnp.where(bbb != 0., grad, flat)
            length = jnp.where((tke[k] < energy)&(tke[k] >= previous), distance-dzt+tl, length)
            return length, energy, energy, distance
        up, *_ = jax.lax.fori_loop(k, n-1, upward, (z[-1]-z[k]-dz[k]/2., jnp.float32(0.), jnp.float32(0.), jnp.float32(0.)))
        def downward(i, c):
            j = k-i
            length, energy, previous, distance = c
            dzt = (dz[j-1]+dz[j])/2.
            energy = _round(energy+beta*theta[k]*dzt)
            energy = energy-beta*(theta[j-1]+theta[j])*dzt/2.
            distance = distance+dzt
            bbb = (theta[j]-theta[j-1])/dzt
            discriminant = jnp.maximum(0., (beta*(theta[j]-theta[k]))**2.+2.*bbb*beta*(tke[k]-previous))
            grad = (beta*(theta[j]-theta[k])+jnp.sqrt(discriminant))/jnp.where(bbb != 0., bbb, 1.)/beta
            flat = jnp.where(theta[j] != theta[k], (tke[k]-previous)/jnp.where(theta[j] != theta[k], beta*(theta[j]-theta[k]), 1.), 0.)
            tl = jnp.where(bbb != 0., grad, flat)
            length = jnp.where((tke[k] < energy)&(tke[k] >= previous), distance-dzt+tl, length)
            return length, energy, energy, distance
        down, *_ = jax.lax.fori_loop(0, k, downward, (z[k]+dz[k]/2., jnp.float32(0.), jnp.float32(0.), jnp.float32(0.)))
        return jnp.sqrt(up*jnp.minimum(down, z[k]+dz[k]/2.))
    return jax.vmap(level)(jnp.arange(n))


@_real32
def keps_column(u, v, theta, qv, qc, rho, dz, exner, raten, p8w, tke, epsilon,
                tpe, prandtl, tke_adv, epsilon_adv, tpe_adv, dt, step, hfx, qfx,
                ustar, roughness, br, fm, fhh):
    """Full nonurban KEPS driver; compiler-zero upstream semantics, REAL32.

    WRF ignores its urban inputs in this module. bl_keps_adv=0 in pristine
    source: the three *_adv fields are initialized at steps 1/2 then retained.
    """
    n = u.shape[0]
    zero = jnp.zeros_like(u)
    def height_prefix(previous, width):
        value = previous+width
        return value, value
    z = jnp.concatenate((zero[:1], jax.lax.scan(height_prefix, jnp.float32(0.), dz)[1]))
    early = step <= 2
    te = jnp.clip(jnp.where(early, 1.e-4, tke), 1.e-4, 50.)
    ep = jnp.clip(jnp.where(early, 1.e-7, epsilon), 1.e-7, 50.)
    tp = jnp.clip(jnp.where(early, 1.e-7, tpe), 1.e-7, 5.)
    pr = jnp.where(early, jnp.ones_like(prandtl), prandtl)
    height, kpbl = pbl_height(theta, qv, te, dz, z)
    rho_face = jnp.concatenate((rho[:1], (rho[1:]*dz[:-1]+rho[:-1]*dz[1:])/(dz[:-1]+dz[1:]), rho[-1:]))
    # Original :245 rounds both constant operations in REAL, before use.
    ep1=jnp.float32(461.)/jnp.float32(287.)-jnp.float32(1.)
    sflux = hfx/rho[0]/1004.5+qfx/rho[0]*ep1*theta[0]
    def advance(_):
        sh = shear(u, v, dz)
        grad = jnp.pad(.5*(2.*(theta[1:-1]-theta[:-2])/(dz[:-2]+dz[1:-1])+
                           2.*(theta[2:]-theta[1:-1])/(dz[2:]+dz[1:-1])), (1, 1))
        grad = grad.at[0].set((-sflux/ustar)/.4/(dz[0]/2.-roughness))
        # Upstream c_nsquare leaves dtmdz(kte) unwritten; authorized compiler
        # initialization defines it as zero (WRF :660–668, :1139–1142).
        ns = (-grad*9.81/300.).at[-1].set(0.)
        ri, p, pf, pm, pe, ds = stability(theta, qv, dz, z, rho, raten, exner, p8w, sh, ns,
                                          br, fm, fhh, height, kpbl, ustar, roughness, sflux)
        grad = grad.at[0].set(ds)
        bu = buoyancy(9.81, theta, jnp.full_like(theta, 300.), dz, te, tp, p)
        km, kh = exchange(te, ep, dz, pf)
        length = length_bougeault(theta, te, dz, z)
        ten, epn, ad = analytic_tke(dt, dz, roughness, sh, bu, te, ep, p, length,
                                  ustar, pe, pm, ri, ns, zero)
        ones, faces = jnp.ones_like(u), jnp.ones_like(z)
        def diff(x, cd, aa=zero, bb=zero, first=1):
            return diffusion(dt, x, rho, rho_face, cd, aa, bb, faces, ones, dz, first=first)[0]
        ten = diff(ten, km, first=2)
        epn = diff(epn, km/1.3, ad, first=2)
        at, bt, _ = tpe_terms(grad, 9.81, jnp.full_like(theta, 300.), ten, epn, tp, p)
        theta_source = zero.at[0].set(hfx/dz[0]/rho[0]/1004.5)
        theta_source = countergradient(dz, 9.81, jnp.full_like(theta, 300.), ten, epn, tp, theta_source)
        tpn = diff(tp, km, at, bt)
        drag = zero.at[0].set(-ustar*ustar/dz[0]/jnp.sqrt(jnp.maximum(1.e-10, jnp.abs(u[0]))**2.+jnp.maximum(1.e-10, jnp.abs(v[0]))**2.))
        un, vn = diff(u, km, drag), diff(v, km, drag)
        thn = diff(theta, kh, bb=theta_source)
        qvn = diff(qv, kh, bb=zero.at[0].set(qfx/dz[0]/rho[0]))
        qcn = diff(qc, kh)
        return un, vn, thn, qvn, qcn, jnp.clip(ten, 1.e-4, 50.), jnp.clip(epn, 1.e-7, 50.), jnp.clip(tpn, 1.e-7, 5.), p, km[:-1]
    un, vn, thn, qvn, qcn, ten, epn, tpn, prn, km = jax.lax.cond(
        early, lambda _: (u, v, theta, qv, qc, te, ep, tp, pr, zero), advance, operand=None)
    ru, rv, rt = (un-u)/dt, (vn-v)/dt, (thn-theta)/dt
    rq, rqc = (qvn-qv)/dt, (qcn-qc)/dt
    # WRF :358 uses the U increment also in WV(kts); retain its literal behavior.
    def flux(incoming, increment):
        def body(previous, operands):
            r, delta, width = operands
            # Original driver :362–368 multiplies rho before dividing by dt.
            value = previous-r*delta/dt*width
            return value, value
        return jax.lax.scan(body, incoming, (rho, increment, dz))[1]
    wu = flux(-ustar*ustar, un-u)
    # WV's first element in WRF uses u, its remaining elements use v.
    wv = flux(-ustar*ustar, (vn-v).at[0].set(un[0]-u[0]))
    wt = flux(hfx/rho[0]/1004.5, thn-theta)
    return dict(rublten=ru, rvblten=rv, rthblten=rt, rqvblten=rq, rqcblten=rqc,
                rqiblten=zero, tke=ten, epsilon=epn, tpe=tpn, prandtl=prn,
                tke_adv=jnp.where(early, 1.e-4, tke_adv), epsilon_adv=jnp.where(early, 1.e-7, epsilon_adv),
                tpe_adv=jnp.where(early, 1.e-7, tpe_adv), exch_m=km, exch_h=km/prn,
                wu=wu, wv=wv, wt=wt, wq=zero, pblh=height)
