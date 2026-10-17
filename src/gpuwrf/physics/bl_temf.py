"""Column-local TEMF numerics from pristine WRF ``module_bl_temf.F``.

WRF REAL arithmetic; helper qualification does not qualify the full PBL.
The saturation adjustment intentionally returns the last *accepted* Tair,
not the converged trial Tt (WRF's GOTO precedes the assignment).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def rsat(pressure, temperature):
    """WRF ARM liquid-water saturation mixing ratio (Pa, K -> kg/kg)."""
    r = jnp.float32
    p, t = jnp.asarray(pressure, r), jnp.asarray(temperature, r)
    temp = t-r(273.15)
    c = (.6105851e3, .4440316e2, .1430341e1, .2641412e-1, .2995057e-3,
         .2031998e-5, .6936113e-8, .2564861e-11, -.3704404e-13)
    x = r(c[-1])
    for ci in c[-2::-1]:
        x = r(ci)+temp*x
    return (r(287.)/r(461.6))*x/(p-x)


def thlqt_to_thqvqc(thetal, qt, pressure, exner):
    """WRF's 20-iteration saturation adjustment, batched over mass levels.

    No cross-column dependency. The convergence mask freezes a level at the
    same early-exit point as WRF; it is the algorithm's branch, not repair.
    """
    r = jnp.float32
    thetal, qt, pressure, exner = (jnp.asarray(x, r) for x in (thetal, qt, pressure, exner))
    t1 = thetal*exner
    rt = qt/(r(1.)-qt)

    def iteration(_, carry):
        tair, rc, done = carry
        new_rc = jnp.maximum(rt-rsat(pressure, tair), r(0.))
        tt = r(.7)*tair+r(.3)*t1*(r(1.)+r(2.5e6)*new_rc/(r(1004.5)*jnp.maximum(tair, r(253.))))
        converged = jnp.abs(tt-tair) < r(.001)
        return (jnp.where(done | converged, tair, tt),
                jnp.where(done, rc, new_rc), done | converged)

    tair, rc, _ = jax.lax.fori_loop(0, 20, iteration, (t1, jnp.zeros_like(t1), jnp.zeros_like(t1, dtype=bool)))
    rv = rt-rc
    return tair/exner, rv/(r(1.)+rv), rc/(r(1.)+rc)


def trid(a, b, c, rhs):
    """Literal WRF Thomas recurrence, final axis = vertical column.

    Inputs may have arbitrary batch prefixes. The unused a[0]/c[-1] are
    never read, matching WRF; no LAPACK solver or alternate association.
    """
    a, b, c, rhs = (jnp.asarray(x, jnp.float32) for x in (a, b, c, rhs))
    n = b.shape[-1]
    if n < 1:
        raise ValueError('TEMF tridiagonal column is empty')
    bet = b[..., 0]
    first = rhs[..., 0]/bet

    def forward(carry, values):
        bet, previous = carry
        ak, bk, ckprev, rk = values
        gam = ckprev/bet
        bet = bk-ak*gam
        uk = (rk-ak*previous)/bet
        return (bet, uk), (gam, uk)

    _, (gam, u) = jax.lax.scan(forward, (bet, first),
                              tuple(jnp.moveaxis(x, -1, 0) for x in (a[..., 1:], b[..., 1:], c[..., :-1], rhs[..., 1:])))
    u = jnp.concatenate((first[None], u), axis=0)

    def backward(next_u, values):
        uk, gam_next = values
        current = uk-gam_next*next_u
        return current, current

    _, solved = jax.lax.scan(backward, u[-1], (u[:-1], gam), reverse=True)
    return jnp.moveaxis(jnp.concatenate((solved, u[-1:]), axis=0), 0, -1)


def solve_implicit(khalf, psi, surface_flux, dzm, dzt, dt):
    """WRF ``solve_implicit_temf`` including its literal lower/top rows.

    The argument views are the caller's views, not relabelled grids. Mean
    variables pass dzm[:-1],dzt[:-1]; TE passes dzt[1:],dzt[:-1]. This
    unusual pairing must be preserved when wiring the full TEMF column.
    """
    r = jnp.float32
    khalf, psi, surface_flux, dzm, dzt, dt = (
        jnp.asarray(x, r) for x in (khalf, psi, surface_flux, dzm, dzt, dt))
    if psi.shape[-1] < 2:
        raise ValueError('TEMF implicit solver needs at least two mass levels')
    au = khalf/(dzm*dzt)
    bu = jnp.empty_like(psi)
    cu = jnp.zeros_like(psi)
    yu = -psi/dt
    bu = bu.at[..., 0].set(-r(1.)/dt-khalf[..., 1]/(dzm[..., 1]*dzt[..., 1]))
    cu = cu.at[..., 0].set(khalf[..., 1]/(dzm[..., 0]*dzt[..., 1]))
    yu = yu.at[..., 0].add(-surface_flux/dzm[..., 0])
    bu = bu.at[..., 1:-1].set(-r(1.)/dt-(khalf[..., 1:-1]/dzt[..., 1:-1]+khalf[..., 2:]/dzt[..., 2:])/dzm[..., 1:-1])
    cu = cu.at[..., 1:-1].set(khalf[..., 2:]/(dzm[..., 1:-1]*dzt[..., 2:]))
    au = au.at[..., -1].set(r(0.))
    bu = bu.at[..., -1].set(-r(1.)/dt)
    return trid(au, bu, cu, yu)


def temf_column(*, u, v, temperature, theta, qv, qc, pressure, p8w, exner,
                rho, z, terrain, roughness, tskold, coriolis, hfx, qfx,
                dt, carry):
    """JAX reference column for ``temf2d``; full scheme qualification pending.

    Mass inputs have n levels. z includes n+1 entries, because WRF reads its
    upper memory level when constructing dzt before replacing the top entry.
    carry holds TEMF's Registry profiles and pblh/kpbl (retained by the stable
    WRF branch). The PBL caller supplies **tskold**, unlike the surface call.
    This reference is not wired into the timestep or labelled GPU-qualified.
    """
    r = jnp.float32
    u, v, temperature, theta, qv, qc, pressure, p8w, exner, rho, z = (
        jnp.asarray(x, r) for x in (u, v, temperature, theta, qv, qc, pressure, p8w, exner, rho, z))
    terrain, roughness, tskold, coriolis, hfx, qfx, dt = (
        jnp.asarray(x, r) for x in (terrain, roughness, tskold, coriolis, hfx, qfx, dt))
    n = u.shape[0]
    if n < 4 or z.shape != (n+1,):
        raise ValueError('TEMF needs >=4 mass levels and n+1 mass-height memory entries')
    one, zero = r(1.), r(0.)
    cp, cpv, rd, rv, g, lv = r(1004.5), r(4.)*r(461.6), r(287.), r(461.6), r(9.81), r(2.5e6)
    ep2, rcp, karman = rd/rv, rd/cp, r(.4)
    ftau0, fth0, ceps = r(.17), r(.145), r(.070)
    prt0 = ceps/ceps*ftau0**2/r(2.)/fth0**2
    te = jnp.asarray(carry['te_temf'], r)
    levels = jnp.arange(1, n+1, dtype=jnp.int32)
    prepend = lambda a, x: jnp.concatenate((jnp.atleast_1d(x), a))
    vapor = qv[:-1]/(one-qv[:-1])
    liquid = qc[:-1]/(one-qc[:-1])
    total = vapor+liquid
    chi = rcp*(one+vapor/ep2)/(one+vapor*cpv/cp)
    gam = total*rv/(cp+total*cpv)
    thl = prepend(theta[:-1]*((ep2+vapor)/(ep2+total))**chi*(vapor/total)**(-gam)
                  *jnp.exp(-lv*liquid/((cp+total*cpv)*temperature[1:])), tskold/exner[0])
    qt = prepend(qv[:-1]+qc[:-1], qv[0])
    thetav = thl*(one+r(.608)*qt-prepend(qc[:-1], qc[0]))
    ui, vi = prepend(u[:-1], zero), prepend(v[:-1], zero)
    zm = prepend(z[:-2]-terrain, roughness)
    zt = prepend((zm[1:]+z[1:n]-terrain)/r(2.), (z[0]-terrain-roughness)/r(2.))
    dzm = prepend(jnp.diff(zt), zt[1]-zt[0])
    dzt = prepend(z[2:]-z[1:-1], z[0]-terrain-roughness)
    dzt = dzt.at[-1].set(dzt[-2])

    def gradient(x):
        bottom = (x[1]-x[0])/(zt[0]*jnp.log10(zm[1]/roughness))
        interior = (x[2:]-x[1:-1])/dzt[1:-1]
        return jnp.concatenate((jnp.atleast_1d(bottom), interior, interior[-1:]))

    dth, dqt, du, dv = (gradient(x) for x in (thl, qt, ui, vi))
    sfc_thvf = hfx/(rho[0]*cp)*(one+r(.608)*(qv[0]+qc[0]))+r(.608)*thetav[0]*carry['qf_temf'][0]
    h0idx = jnp.min(jnp.where((levels > 1) & (levels < n) & (thetav > thetav[0]), levels, n))
    h0idx = jnp.where(h0idx == n, 1, h0idx)
    h0 = jnp.minimum(zm[h0idx-1], r(4000.))
    hmax_idx = jnp.minimum(n-1, jnp.min(jnp.where((levels > 1) & (levels < n) & (zm > r(4000.)), levels, n)))
    wstr = jnp.where(hfx > 0, g*h0/thetav[1]*hfx/(rho[0]*cp), zero)**(one/r(3.))
    convective = (wstr > 0) & (dth[0] < 0) & (dth[1] < 0)
    beta = r(2.)*g/(thetav[:-1]+thetav[1:])
    n2 = beta*dth[:-1]
    shear = jnp.sqrt(du[:-1]**2+dv[:-1]**2)
    ri = n2/jnp.where(shear < r(1e-15), one, shear**2)
    ri = jnp.where(shear < r(1e-15), jnp.where(n2 >= 0, r(10.), r(-1.)), ri)
    ratio = jnp.where(ri > 0,
                      ri/(ceps**2*ftau0**2/(r(2.)*ceps**2*fth0**2)+r(3.)*ri),
                      ri/(ceps**2*ftau0**2/(-r(2.)*ceps**2*fth0**2)+r(2.)*ri))
    ftau = jnp.where(ri > 0, ftau0*((r(3.)/r(4.))/(one+r(4.)*ri)+one/r(4.)), ftau0)
    fth = jnp.where(ri > 0, fth0/(one+r(4.)*ri), fth0)
    tke = te[:-1]*(one-ratio)
    ustrtilde = jnp.sqrt(ftau*tke)
    linv = one/karman/zt[:-1]+jnp.abs(coriolis)/(r(.185)*ustrtilde)
    linv += jnp.where(n2 > 0, jnp.sqrt(jnp.maximum(n2, zero))/(r(2.)*ustrtilde), zero)
    linv += one/r(200.)
    leps = jnp.maximum(one/linv, zero)
    km = tke**r(1.5)*ftau**2/(ceps*jnp.sqrt(tke*te[:-1])/leps)
    kh = r(2.)*leps*fth**2*tke/jnp.sqrt(te[:-1])/ceps
    lconv = jnp.where((levels[:-1] <= h0idx) & (h0-zt[:-1] > r(1e-15)),
                      one/(one/(karman*zt[:-1])+r(3.)/(karman*(h0-zt[:-1]))), zero)
    khconv = jnp.maximum(ftau0**2/ceps/prt0*jnp.sqrt(tke)*lconv, zero)
    kmconv = prt0*khconv
    below = convective & (zt[:-1] <= h0/r(2.))
    upper = convective & (zt[:-1] > h0/r(2.)) & (levels[:-1] <= h0idx)
    km = jnp.where(below, kmconv, jnp.where(upper, jnp.maximum(km, kmconv), km))
    kh = jnp.where(below, khconv, jnp.where(upper, jnp.maximum(kh, khconv), kh))
    km, kh = jnp.maximum(km, r(1.57e-4)), jnp.maximum(kh, r(1.57e-4)/r(.733))
    fz = jnp.concatenate((-kh*dth[:-1], jnp.zeros((1,), r)))
    km, kh = jnp.concatenate((km, km[-1:])), jnp.concatenate((kh, kh[-1:]))

    def mass_flux(_):
        eps = jnp.maximum(r(2.)/jnp.maximum(r(200.), h0), r(.002))
        # Internal parcel vectors; entries above the 4000 m limit are unused.
        parcel = (thl.at[0].set(thl[0]), qt, ui, vi, te.at[0].set(te[0]+g/thetav[0]*sfc_thvf),
                  prepend(qc[:-1], qc[0]), jnp.zeros_like(te).at[0].set(r(.5)*wstr),
                  jnp.zeros_like(te).at[0].set(r(.5)*wstr),
                  jnp.zeros_like(te).at[0].set(qt[0]/(one-qt[0])),
                  jnp.zeros_like(te).at[0].set(rsat(pressure[0], thl[0]*exner[0])))

        def rise(k, a):
            thup, qtup, uup, vup, teup, qlup, wdry, wmoist, rup, rst = a
            dz = dzm[k-1]
            th = thup[k-1]-eps*(thup[k-1]-thl[k-1])*dz
            q = qtup[k-1]-eps*(qtup[k-1]-qt[k-1])*dz
            uu = uup[k-1]-eps*(uup[k-1]-ui[k-1])*dz
            vv = vup[k-1]-eps*(vup[k-1]-vi[k-1])*dz
            tt = teup[k-1]-eps*(teup[k-1]-te[k-1])*dz
            tvdry = th*(one+r(.608)*q)
            buoy = g*(tvdry-thetav[k])/thetav[k]
            wd = jnp.where(wdry[k-1] < r(1e-15), zero,
                           wdry[k-1]+(-r(2.)*eps*wdry[k-1]+r(.33)*buoy/wdry[k-1])*dz)
            ru = q/(one-q)
            tup = th*exner[k]
            rs = rsat(pressure[k-1], tup)
            b1 = r(.622)*(lv/(rd*tup))*(lv/(cp*tup))
            rs = rs*(one+b1*ru)/(one+b1*rs)
            qs = rs/(one+rs)
            rl = ru-rs
            ql = jnp.where(ru > rs, rl/(one+rl), qc[k-1])
            tvmoist = jnp.where(ru > rs, (th+((lv/cp)*ql/exner[k]))*(one+r(.608)*qs-ql), tvdry)
            bm = g*(tvmoist-thetav[k])/thetav[k]
            wm = jnp.where(wmoist[k-1] < r(1e-15), zero,
                           wmoist[k-1]+(-r(2.)*eps*wmoist[k-1]+r(.33)*bm/wmoist[k-1])*dz)
            active = k+1 < hmax_idx
            vals = (th, q, uu, vv, tt, ql, wd, wm, ru, rs)
            fallback = (thl[k], qt[k], ui[k], vi[k], te[k], qc[k-1], zero, zero, zero, zero)
            return tuple(x.at[k].set(jnp.where(active, y, f)) for x, y, f in zip(a, vals, fallback))

        thup, qtup, uup, vup, teup, qlup, wdry, wup, rup, rst = jax.lax.fori_loop(1, n, rise, parcel)
        hdidx = jnp.min(jnp.where((levels > 1) & ((wdry <= 0) | (zm > r(4000.))), levels, n))
        hdidx = jnp.where(wdry[0] == 0, 1, hdidx)
        hd = zm[hdidx-1]
        lclidx = jnp.min(jnp.where((levels < hmax_idx) & (rup > rst), levels, n))
        lcl = zm[lclidx-1]
        hctidx = jnp.min(jnp.where((levels > 1) & ((wup <= 0) | (zm > r(4000.))), levels, n))
        hctidx = jnp.where(wup[0] == 0, 1, hctidx)
        hctidx = jnp.where((hd > lcl) & (hctidx > hdidx+1), hctidx, hdidx)
        hct = zm[hctidx-1]
        height = jnp.maximum(hd, hct)
        # WRF declares tval INTEGER: the linear taper is truncated deliberately.
        taper = jnp.where(zm < r(.9)*height, one,
                          jnp.where(zm <= height, one-(zm-r(.9)*height)/(height-r(.9)*height), zero)).astype(jnp.int32)
        thup = taper*thup+(1-taper)*thl
        qtup = taper*qtup+(1-taper)*qt
        uup, vup, teup = (taper*x+(1-taper)*env for x, env in ((uup, ui), (vup, vi), (teup, te)))
        qlup = qlup.at[1:].set(taper[1:]*qlup[1:]+(1-taper[1:])*qc[:-1])
        wup = jnp.where(zm > height, zero, wup)

        def detraining(k, delt):
            cloud = r(.9)*eps+r(.006)*(jnp.arctan((zm[k]-(lcl+(hct-lcl)/r(1.5)))/((hct-lcl)/r(8.)))+r(3.1415926)/r(2.))/r(3.1415926)
            dry = jnp.where(k+1 < hdidx, eps+r(.05)*one/(hd-zm[k]), delt[k-1])
            return delt.at[k].set(jnp.where(hctidx > hdidx+1, cloud, dry))

        delt = jax.lax.fori_loop(1, n-1, detraining, jnp.zeros_like(te).at[0].set(eps))

        def mass(k, mf):
            return mf.at[k].set(mf[k-1]+(eps-delt[k])*mf[k-1]*dzt[k])

        mf = jax.lax.fori_loop(1, n-1, mass, jnp.asarray(carry['mf_temf'], r).at[0].set(r(.03)*wstr))
        second = mf[1]*(thup[1]-thl[1]+thup[2]-thl[2])/r(2.)
        mf = jnp.where(second > fz[1], mf*(fz[1]/second), mf)

        def flux(parcel, env):
            bottom = mf[0]*(parcel[0]-env[0]+(parcel[1]-env[1]-(parcel[0]-env[0]))*jnp.log(zt[0]/roughness)/jnp.log(zm[1]/roughness))
            interior = mf[1:-1]*(parcel[1:-1]-env[1:-1]+parcel[2:]-env[2:])/r(2.)
            return jnp.concatenate((jnp.atleast_1d(bottom), interior, jnp.zeros((1,), r)))

        mfth, mfq, mfu, mfv = (flux(x, env) for x, env in ((thup, thl), (qtup, qt), (uup, ui), (vup, vi)))
        mfql = flux(qlup, prepend(qc[:-1], qc[0]))
        # WRF's surface MFCql second operand is qcx(2), not shifted qcx(1).
        mfql = mfql.at[0].set(mf[0]*(qlup[0]-qc[0]+(qlup[1]-qc[1]-(qlup[0]-qc[0]))*jnp.log(zt[0]/roughness)/jnp.log(zm[1]/roughness)))
        mfte = flux(teup, te).at[1:-1].set(mf[1:-1]*(teup[1:-1]-te[1:-1]))
        area = jnp.where((wup[:-1] >= r(1e-15)) & (wup[1:] >= r(1e-15)),
                         ((mf[:-1]+mf[1:])/r(2.))/((wup[:-1]+wup[1:])/r(2.)), zero)
        variance = area*(qtup[1:]-qt[1:])
        sigma = jnp.sqrt(jnp.maximum(variance, zero))
        rs = rsat(pressure[:-1], theta[:-1]*exner[:-1])
        deficit = qt[1:]-rs/(one+rs)
        cf = jnp.where(deficit <= 0,
                       jnp.where(sigma > r(1e-15), jnp.maximum(r(.5)+r(.36)*jnp.arctan(r(1.55)*(deficit/sigma)), zero), zero), one)
        cf = prepend(jnp.where(zm[1:] < lcl, zero, cf), zero)
        cfm = jnp.max(jnp.where(zt <= r(4000.), cf, zero))
        return (mfth, mfq, mfu, mfv, mfql, mfte, thup, qtup, qlup.at[-1].set(qc[-1]), wup, mf,
                cf.at[-1].set(zero), cfm, hd, lcl, hct, hd, hdidx)

    def stable(_):
        zeros = jnp.zeros_like(te)
        cf = prepend(jnp.where(qc[:-1] > r(1e-15), one, zero), zero)
        cfm = jnp.max(jnp.where(zt <= r(4000.), cf, zero))
        return (zeros, zeros, zeros, zeros, zeros, zeros,
                jnp.asarray(carry['thup_temf'], r), jnp.asarray(carry['qtup_temf'], r),
                jnp.concatenate((qc[:1], qc[:-2], qc[-1:])),
                jnp.asarray(carry['wupd_temf'], r), jnp.asarray(carry['mf_temf'], r),
                cf.at[-1].set(zero), cfm, zm[0], zm[-2], zm[0],
                jnp.asarray(carry['pblh'], r), jnp.asarray(carry['kpbl'], jnp.int32))

    mfth, mfq, mfu, mfv, mfql, mfte, thup, qtup, qlup, wup, mf, cf, cfm, hd, lcl, hct, pblh, kpbl = jax.lax.cond(convective, mass_flux, stable, None)
    shf = fz+mfth
    qfk = -kh*dqt
    qf = qfk+mfq
    uw = -km*du+mfu
    vw = -km*dv+mfv
    ust = jnp.sqrt(ftau[0]/ftau0)*jnp.sqrt(ui[1]**2+vi[1]**2+(r(.5)*wstr)**2)*leps[0]/jnp.log(zm[1]/roughness)/zt[0]
    ang = jnp.arctan2(vi[1], ui[1])
    uw, vw = uw.at[0].set(-jnp.cos(ang)*ust**2), vw.at[0].set(-jnp.sin(ang)*ust**2)
    shf, qf = shf.at[0].set(hfx/(rho[0]*cp)), qf.at[0].set(qfx/rho[0])
    fz, qfk = fz.at[0].set(shf[0]-mfth[0]), qfk.at[0].set(qf[0]-mfq[0])
    interp_theta = thl[0]+(thl[1]-thl[0])*(zt[1]-roughness)/(zm[1]-roughness)
    alpha = prepend(r(.61)*(thl[1:-1]+thl[2:])/r(2.), r(.61)*interp_theta)
    alpha = jnp.concatenate((alpha, jnp.atleast_1d(r(.61)*thl[-1])))
    beta2 = (r(100000.)/p8w)**r(.286)*r(2.45e-6)/r(1004.67)
    beta2 -= r(1.61)*jnp.concatenate((jnp.atleast_1d(interp_theta), (thl[1:-1]+thl[2:])/r(2.), thl[-1:]))
    mfthv = (one+r(.61)*(qtup[:-1]+qtup[1:]))/r(2.)*mfth[:-1]+alpha[:-1]*mfq[:-1]+beta2[:-1]*mfql[:-1]
    mfthv = jnp.concatenate((jnp.where(convective, mfthv, zero), jnp.zeros((1,), r)))
    thvf = (one+r(.61)*qt)*fz+alpha*qfk+mfthv

    def update(variable, diffusivity, flux, mf_flux):
        new = solve_implicit(diffusivity[:-1], variable[1:], flux, dzm[:-1], dzt[:-1], dt)
        new = new.at[:-1].add(dt*(-(mf_flux[1:-1]-mf_flux[:-2]))/dzm[1:-1])
        return prepend(new, variable[0])

    unew, vnew = update(ui, km, uw[0], mfu), update(vi, km, vw[0], mfv)
    thlnew, qtnew = update(thl, kh, fz[0], mfth), update(qt, kh, qfk[0], mfq)
    surface_te = ust**2/ftau[0]*(one+ratio[0])
    convection_src = jnp.where(convective, r(2.)*beta[0]*shf[0], zero)
    surface_te = jnp.minimum(jnp.maximum(surface_te, (leps[0]/ceps*(ust**2*shear[0]+convection_src))**(r(2.)/r(3.))), r(20.))
    te = te.at[0].set(surface_te)
    sfc_te_flux = -(km[0]+km[1])/r(2.)*(te[1]-te[0])/dzm[1]
    buoy = jnp.where(thvf >= 0, r(2.)*g/thetav*thvf, zero)
    linv = jnp.concatenate((linv, linv[-1:]))
    src = -uw*du-vw*dv-ceps*te**r(1.5)*linv+buoy
    tenew = solve_implicit((km[:-1]+km[1:])/r(2.), te[1:], sfc_te_flux, dzt[1:], dzt[:-1], dt)
    tenew = tenew.at[:-1].add(dt*src[1:-1])
    tenew = tenew.at[:-1].add(dt*(-(mfte[1:-1]-mfte[:-2]))/dzt[1:-1])
    tenew = jnp.concatenate((jnp.clip(tenew[:-1], r(1e-3), r(20.)), jnp.zeros((1,), r)))
    te = prepend(tenew, surface_te)
    thnew, qvnew, qcnew = thlqt_to_thqvqc(thlnew[1:], qtnew[1:], pressure[:-1], exner[:-1])
    tendency = lambda new, old: jnp.concatenate(((new-old[:-1])/dt, jnp.zeros((1,), r)))
    return dict(te_temf=te, kh_temf=kh, km_temf=km, shf_temf=shf, qf_temf=qf,
                uw_temf=uw, vw_temf=vw, wupd_temf=wup, mf_temf=mf,
                thup_temf=thup, qtup_temf=qtup, qlup_temf=qlup, cf3d_temf=cf,
                cfm_temf=cfm, hd_temf=hd, lcl_temf=lcl, hct_temf=hct,
                pblh=pblh, kpbl=kpbl, ust=ust,
                rublten=tendency(unew[1:], u), rvblten=tendency(vnew[1:], v),
                rthblten=tendency(thnew, theta), rqvblten=tendency(qvnew, qv), rqcblten=tendency(qcnew, qc),
                u10=unew[1]*jnp.log(r(10.)/roughness)/jnp.log(zm[1]/roughness),
                v10=vnew[1]*jnp.log(r(10.)/roughness)/jnp.log(zm[1]/roughness),
                t2=(tskold/exner[0]+(thnew[0]-tskold/exner[0])*jnp.log(r(2.)/roughness)/jnp.log(zm[1]/roughness))*exner[0])
