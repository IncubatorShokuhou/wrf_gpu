"""Standalone JAX column transcription of WRF ARW ``module_bl_qnsepbl.F``.

Selected by the mandatory operational bl=4/sf=4 pair. Inputs are
bottom-up mass columns (n >= 3), LOWLYR=1, after the paired QNSESFC call. WRF
REAL precision follows the profile dtype; fp64 is available for oracle tests.
MFSHCONV is the preceding separate qnse_edmf caller. In WRF the
optional WTHV_MF/LM_BL89 arguments do not enter active QNSEPBL expressions.

The column recurrences are statically unrolled with local scalar lists: no
host materialization or domain-wide level scan. GPU compile/launch/temp cost
remains unmeasured. The adapter keeps each driver in one column kernel.
"""
from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=2)
def _constants(dtype):
    """Host scalar PARAMETER evaluation, rounded at each WRF REAL operation."""
    f = np.dtype(dtype).type
    g, rd = f(9.81), f(287.)
    cp = f(7.) * rd / f(2.)
    a1, a2, b1, b2, c1 = map(f, (.659888514560862645, .6574209922667784586,
                                11.87799326209552761, 7.226971804046074028,
                                .000830955950095854396))
    btg = (f(1.) / f(273.)) * g
    adnh = f(9.)*a1*a2*a2*(f(12.)*a1+f(3.)*b2)*btg*btg
    adnm = f(18.)*a1*a1*a2*(b2-f(3.)*a2)*btg
    bdnh = f(3.)*a2*(f(7.)*a1+b2)*btg
    bdnm = f(6.)*a1*a1
    aeqh = f(9.)*a1*a2*a2*b1*btg*btg + adnh
    aeqm = (f(3.)*a1*a2*b1*(f(3.)*a2+f(3.)*b2*c1+f(18.)*a1*c1-b2)*btg
            + f(18.)*a1*a1*a2*(b2-f(3.)*a2)*btg)
    requ = -aeqh/aeqm
    ubryl = (f(18.)*requ*a1*a1*a2*b2*c1*btg + f(9.)*a1*a2*a2*b2*btg*btg)/(requ*adnm+adnh)
    ubry3 = f(3.)*((f(1.)+f(1.e-7))*ubryl)
    cubr = f(1.)-ubry3
    return dict(G=g, R_D=rd, CP=cp, CAPA=rd/cp, P608=f(461.6)/rd-f(1.),
                ELOCP=f(2.72e6)/cp, EPSGM=requ*f(1.e-9), REQU=requ,
                ADNH=adnh, ADNM=adnm, BDNH=bdnh, BDNM=bdnm,
                AUBH=f(27.)*a1*a2*a2*b2*btg*btg-adnh*ubry3,
                AUBM=f(54.)*a1*a1*a2*b2*c1*btg-adnm*ubry3,
                BUBH=(f(9.)*a1*a2+f(3.)*a2*b2)*btg-bdnh*ubry3,
                BUBM=f(18.)*a1*a1*c1-bdnm*ubry3, CUBR=cubr, RCUBR=f(1.)/cubr,
                CEPS=f(.55)**3, RC02=f(2.)/(f(.55)*f(.55)), EPSQ_FH=f(.01)*f(1.01),
                PQ0SEA=f(379.90516)*f(.98), RLIVWV=f(2.85e6)/f(2.5e6))


def _level_pick(values, index):
    """Column-local selection also supports a vector of independent columns."""
    value = values[0]
    for k in range(1, len(values)):
        value = jnp.where(index == k, values[k], value)
    return value


def _mixlen(u, v, t, the, q, cwm, q2, z, ustar, corf, ct, c):
    n = len(q2)
    # Here k is Fortran K-1; z[k] is top interface of mass level k.
    lpbl = jnp.asarray(0, jnp.int32)
    for k in range(n-1):
        lpbl = jnp.where(q2[k] <= c['EPSQ_FH'], k, lpbl)
    pblh = _level_pick(z, lpbl) - z[n]
    dth = [the[k]-the[k+1] for k in range(n-1)]
    found = jnp.asarray(False)
    for k in range(n-3, -1, -1):
        use = (dth[k] > 0.) & (dth[k+1] <= 0.) & (found == False)
        dth[k] = jnp.where(use, dth[k]+ct, dth[k])
        found = found | use
    s2, gh, ri, en2, elm = [], [], [], [], []
    for k in range(n-1):
        rdz = 2./(z[k]-z[k+2])
        shear = jnp.maximum(((u[k]-u[k+1])**2+(v[k]-v[k+1])**2)*rdz*rdz, c['EPSGM'])
        tem = (t[k]+t[k+1])*.5
        thm = (the[k]+the[k+1])*.5
        a = thm*c['P608']
        b = (c['ELOCP']/tem-1.-c['P608'])*thm
        grad = (dth[k]*((q[k]+q[k+1]+cwm[k]+cwm[k+1])*(.5*c['P608'])+1.)
                +(q[k]-q[k+1]+cwm[k]-cwm[k+1])*a+(cwm[k]-cwm[k+1])*b)*rdz
        grad = jnp.where(jnp.abs(grad) <= 1.e-9, jnp.asarray(1.e-9, grad.dtype), grad)
        nsq = grad*c['G']/thm
        s2.append(shear); gh.append(grad); en2.append(nsq); ri.append(nsq/shear)
        aubr = (c['AUBM']*shear+c['AUBH']*grad)*grad
        bubr = c['BUBM']*shear+c['BUBH']*grad
        qol2st = (-.5*bubr+jnp.sqrt(bubr*bubr*.25-aubr*c['CUBR']))*c['RCUBR']
        stable = jnp.maximum(jnp.sqrt((1./qol2st)*q2[k]), .32)
        aden = (c['ADNM']*shear+c['ADNH']*grad)*grad
        bden = c['BDNM']*shear+c['BDNH']*grad
        qol2un = -.5*bden+jnp.sqrt(bden*bden*.25-aden)
        unstable = jnp.maximum(jnp.sqrt((1./(qol2un+1.e-7))*q2[k]), .32)
        elm.append(jnp.where(grad >= 1.e-9, jnp.where(
            shear/grad <= c['REQU'], jnp.asarray(.32,q2[k].dtype), stable), unstable))
    q1 = [jnp.where(k >= lpbl, jnp.sqrt(q2[k]), jnp.zeros_like(q2[k])) for k in range(n)]
    szq, sq = q2[0]*0., q2[0]*0.
    for k in range(n-1):
        qdzl = (q1[k]+q1[k+1])*(z[k+1]-z[k+2])
        szq = (z[k+1]+z[k+2]-z[n]-z[n])*qdzl+szq
        sq = qdzl+sq
    el0 = jnp.maximum(jnp.minimum(.30*szq*.5/sq, 1000.), 1.)
    el, rel = [], []
    for k in range(n-1):
        above = jnp.minimum((z[k]-z[k+2])*(.23*.5), elm[k])
        vkrmz = (z[k+1]-z[n])*.4
        inside = jnp.minimum(vkrmz/(vkrmz/el0+1.), elm[k])
        ell = jnp.where(k >= lpbl, inside, above)
        el.append(ell); rel.append(ell/elm[k])
    for k in range(1, n-2):
        srel = jnp.minimum(((rel[k-1]+rel[k+1])*.5+rel[k])*.5, rel[k])
        el[k] = jnp.where(k > lpbl, jnp.maximum(srel*elm[k], .32), el[k])
    rlambda = jnp.maximum(corf, 1.e-12)/(.0063*ustar)
    for k in range(n-1):
        rlb = rlambda+1./((z[k+1]-z[n])*.4)
        rln = jnp.sqrt(2.*en2[k]/q2[k])/.75
        el[k] = jnp.where(en2[k] >= 0., 1./(rlb+rln), el[k])
    return s2, ri, el, pblh, n-lpbl


def _difcof(el, ri, q2, z):
    akm, akh = [], []
    for k in range(len(el)):
        ril = ri[k]; ril2 = ril*ril
        aril = jnp.minimum(jnp.abs(ril), 2.*.127)
        aril2 = aril*aril; aril4 = aril2*aril2
        am = jnp.where(ril <= 0., 1.+2.88*aril+16.*aril2,
                       (1.+8.*ril2)/(1.+2.3*ril+35.*ril2))
        ah = jnp.where(ril <= 0., 1.4+3.6*aril+16.*aril2+720.*aril4,
                       (1.4+(-.01)*ril+1.29*ril2)/(1.+2.44*ril+19.8*ril2))
        ak0 = .55*el[k]*jnp.sqrt(.5*q2[k])
        rdz = 2./(z[k]-z[k+2])
        ak0dz = ak0*rdz; akmin = .001*rdz
        akm.append(jnp.maximum(ak0dz*am, akmin)); akh.append(jnp.maximum(ak0dz*ah, akmin))
    return akm, akh


def _prodq2(q2, s2, ri, el, z, akm, akh, ustar, dt, c):
    out = []
    for k in range(len(el)):
        deltaz = .5*(z[k]-z[k+2])
        pr = (akm[k]*deltaz)*s2[k]
        bpr = (akh[k]*deltaz)*(ri[k]*s2[k])
        dis = c['CEPS']*(.5*q2[k])**1.5/el[k]
        out.append(jnp.maximum(q2[k]+2.*(pr-bpr-dis)*dt, .01))
    out.append(jnp.maximum(c['RC02']*ustar*ustar, .01))
    return out


def _vdifq(q2, el, z, dt):
    n = len(q2)
    dtoz, akq, cr = [], [], []
    for k in range(n-2):
        d = (dt+dt)/(z[k]-z[k+2])
        a = jnp.sqrt((q2[k]+q2[k+1])*.5)*(el[k]+el[k+1])*(.5*5.)/(z[k+1]-z[k+2])
        dtoz.append(d); akq.append(a); cr.append(-d*a)
    cm = [dtoz[0]*akq[0]+1.]; rs = [q2[0]]
    for k in range(1, n-2):
        cf = -dtoz[k]*akq[k-1]/cm[k-1]
        cm.append(-cr[k-1]*cf+(akq[k-1]+akq[k])*dtoz[k]+1.)
        rs.append(-rs[k-1]*cf+q2[k])
    d = (dt+dt)/(z[n-2]-z[n])
    a = jnp.sqrt((q2[n-2]+q2[n-1])*.5)*(el[n-2]+0.)*(.5*5.)/(z[n-1]-z[n])
    cf = -d*akq[n-3]/cm[n-3]
    out = list(q2)
    out[n-2] = (d*a*q2[n-1]-rs[n-3]*cf+q2[n-2])/((akq[n-3]+a)*d-cr[n-3]*cf+1.)
    for k in range(n-3, -1, -1):
        out[k] = (-cr[k]*out[k+1]+rs[k])/cm[k]
    return out


def _vdifh(the, q, cwm, rkh, z, rho, dt, rkhs, chklowq, thz0, qz0):
    # MIXLEN has reset CT=0 before this solve, as in WRF.
    n = len(the)
    dtoz = [dt/(z[k]-z[k+1]) for k in range(n-1)]
    cr = [-dtoz[k]*rkh[k] for k in range(n-1)]
    cm = [dtoz[0]*rkh[0]+rho[0]]
    rhs = [[a[0]*rho[0]] for a in (the, q, cwm)]
    for k in range(1, n-1):
        cf = -dtoz[k]*rkh[k-1]/cm[k-1]
        cm.append(-cr[k-1]*cf+(rkh[k-1]+rkh[k])*dtoz[k]+rho[k])
        for r, a in zip(rhs, (the, q, cwm)):
            r.append(-r[k-1]*cf+a[k]*rho[k])
    d = dt/(z[n-1]-z[n]); rkhh = rkh[n-2]
    cf = -d*rkhh/cm[n-2]; cmb = cr[n-2]*cf
    result = []
    for a, r, rks, s0 in zip((the,q,cwm), rhs, (rkhs,rkhs*chklowq,0.), (thz0,qz0,0.)):
        den = -cmb+(rkhh+rks)*d+rho[n-1]
        rsb = -r[n-2]*cf+a[n-1]*rho[n-1]
        out = list(a); out[n-1] = (d*rks*s0+rsb)/den
        for k in range(n-2, -1, -1):
            out[k] = (-cr[k]*out[k+1]+r[k])*(1./cm[k])
        result.append(out)
    return result


def _vdifv(u, v, rkm, z, rho, dt, rkms, uz0, vz0):
    n = len(u)
    dtoz = [dt/(z[k]-z[k+1]) for k in range(n-1)]
    cr = [-dtoz[k]*rkm[k] for k in range(n-1)]
    cm = [dtoz[0]*rkm[0]+rho[0]]
    rhs = [[a[0]*rho[0]] for a in (u,v)]
    for k in range(1, n-1):
        cf = -dtoz[k]*rkm[k-1]/cm[k-1]
        cm.append(-cr[k-1]*cf+(rkm[k-1]+rkm[k])*dtoz[k]+rho[k])
        for r, a in zip(rhs, (u,v)):
            r.append(-r[k-1]*cf+a[k]*rho[k])
    d = dt/(z[n-1]-z[n]); rkmh = rkm[n-2]
    cf = -d*rkmh/cm[n-2]
    reciprocal = 1./((rkmh+rkms)*d-cr[n-2]*cf+rho[n-1])
    dtozak = d*rkms
    result = []
    for a, r, s0 in zip((u,v), rhs, (uz0,vz0)):
        out = list(a); out[n-1] = (dtozak*s0-r[n-2]*cf+a[n-1]*rho[n-1])*reciprocal
        for k in range(n-2, -1, -1):
            out[k] = (-cr[k]*out[k+1]+r[k])*(1./cm[k])
        result.append(out)
    return result


def qnsepbl_column(*, u, v, temperature, theta, qv, qc, p_mid, p_int,
                   exner, dz, tke, xland, ustar, akhs, akms, hfx, elflx,
                   thz0, qz0, uz0, vz0, qsfc, corf, chklowq=1.,
                   snow=0., sice=0., ct=0., dt=60., stepbl=1, ht=0.,
                   RUBLTEN=None, RVBLTEN=None, RTHBLTEN=None,
                   RQVBLTEN=None, RQCBLTEN=None, EXCH_H=None, EXCH_M=None,
                   _column_kernel=False):
    """Apply QNSEPBL to one column, adding to the caller's five tendencies.

    EXCH_H/M have n mass-indexed slots here: index0 is preserved; interfaces
    1..n-1 receive diffusivities, matching WRF's EXCH_*(K+1) write. No surface
    layer substitute or shallow-convection approximation is selected here.
    """
    u = u if _column_kernel else jnp.asarray(u)
    dtype = u.dtype
    if dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
        raise ValueError('QNSE requires float32 or float64 profile inputs')
    if u.ndim != (2 if _column_kernel else 1) or u.shape[0] < 3:
        raise ValueError('QNSE requires a one-dimensional column with n >= 3')
    n = u.shape[0]
    def profile(a):
        a = a if _column_kernel else jnp.asarray(a, dtype=dtype)
        if a.shape != u.shape:
            raise ValueError(f'QNSE profile shape must be {(n,)}, got {a.shape}')
        return a
    v, temperature, theta, qv, qc, p_mid, exner, dz, tke = map(
        profile, (v, temperature, theta, qv, qc, p_mid, exner, dz, tke))
    p_int = p_int if _column_kernel else jnp.asarray(p_int, dtype=dtype)
    if p_int.shape != (n+1, *u.shape[1:]):
        raise ValueError('QNSE p_int must have n+1 interfaces')
    # Cast every surface operand to REAL, including weak Python scalars.
    xland, ustar, akhs, akms, hfx, elflx, thz0, qz0, uz0, vz0, qsfc, corf, chklowq, snow, sice, ct, dt, ht = (
        jnp.asarray(a, dtype=dtype) for a in
        (xland,ustar,akhs,akms,hfx,elflx,thz0,qz0,uz0,vz0,qsfc,corf,chklowq,snow,sice,ct,dt,ht))
    c = _constants(str(dtype))
    dtturbl = dt*jnp.asarray(stepbl, dtype=dtype); rdt = 1./dtturbl
    z = [ht]
    for k in range(n):
        z.append(z[-1]+dz[k])
    z = z[::-1]
    u_top, v_top, t_top, p_top = (list(a[::-1]) for a in (u,v,temperature,p_mid))
    q = [qv[k]/(1.+qv[k]) for k in range(n-1,-1,-1)]; cwm = list(qc[::-1])
    the = [(qc[k]*(-c['ELOCP']/temperature[k])+1.)*theta[k] for k in range(n-1,-1,-1)]
    q2 = [2.*tke[k] for k in range(n-1,-1,-1)]
    s2, ri, el, pblh, kpbl = _mixlen(u_top,v_top,t_top,the,q,cwm,q2,z,ustar,corf,ct,c)
    # WRF computes coefficients from OLD TKE, before its production solve.
    akm, akh = _difcof(el,ri,q2,z)
    exch_h = [jnp.zeros_like(u[0])]*n if EXCH_H is None else profile(EXCH_H)
    exch_m = [jnp.zeros_like(u[0])]*n if EXCH_M is None else profile(EXCH_M)
    eh = [exch_h[0]]; em = [exch_m[0]]
    for k in range(n-2, -1, -1):
        deltaz = .5*(z[k]-z[k+2])
        eh.append(akh[k]*deltaz); em.append(akm[k]*deltaz)
    q2 = _prodq2(q2,s2,ri,el,z,akm,akh,ustar,dtturbl,c)
    q2 = _vdifq(q2,el,z,dtturbl)
    rho = [p_top[k]/(c['R_D']*t_top[k]*(1.+c['P608']*q[k]-cwm[k])) for k in range(n)]
    akh_dens = [akh[k]*.5*(rho[k]+rho[k+1]) for k in range(n-1)]
    seamask = xland-1.
    # This is the HFX boundary in QNSE, deliberately different from MYJ TSK.
    rlow = p_int[0]/(c['R_D']*temperature[0])
    thsk = hfx/(akhs*rlow*c['CP'])+theta[0]
    thz0 = (1.-seamask)*thsk+seamask*thz0
    akhs_dens = akhs*rho[n-1]
    qfc1 = 2.5e6*chklowq*akhs_dens
    qfc1 = jnp.where((snow > 0.) | (sice > .5), qfc1*c['RLIVWV'], qfc1)
    land_qsfc = jnp.where(qfc1 > 0., q[n-1]+elflx/qfc1, qsfc)
    exnsfc = (1.e5/p_int[0])**c['CAPA']
    sea_qsfc = c['PQ0SEA']/p_int[0]*jnp.exp(
        17.2693882*(thsk-273.16*exnsfc)/(thsk-35.86*exnsfc))
    qsfc = jnp.where(seamask < .5, land_qsfc, sea_qsfc)
    qz0 = (1.-seamask)*qsfc+seamask*qz0
    the_new,q_new,cwm_new = _vdifh(the,q,cwm,akh_dens,z,rho,dtturbl,akhs_dens,chklowq,thz0,qz0)
    akm_dens = [akm[k]*(rho[k]+rho[k+1])*.5 for k in range(n-1)]
    u_new,v_new = _vdifv(u_top,v_top,akm_dens,z,rho,dtturbl,akms*rho[n-1],uz0,vz0)
    bottom = lambda a: a[::-1]
    pack = (lambda a:a) if _column_kernel else jnp.stack
    q_out = bottom(q_new); qc_out = bottom(cwm_new)
    # APE = 1/EXNER is formed before the WRF THNEW product.
    thnew = [bottom(the_new)[k]+qc_out[k]*c['ELOCP']*(1./exner[k]) for k in range(n)]
    tend = {'RUBLTEN': [(bottom(u_new)[k]-u[k])*rdt for k in range(n)],
            'RVBLTEN': [(bottom(v_new)[k]-v[k])*rdt for k in range(n)],
            'RTHBLTEN': [(thnew[k]-theta[k])*rdt for k in range(n)],
            'RQVBLTEN': [((q_out[k]-qv[k]/(1.+qv[k]))*rdt)/(1.-q_out[k])**2 for k in range(n)],
            'RQCBLTEN': [(qc_out[k]-qc[k])*rdt for k in range(n)]}
    for key, seed in zip(tend, (RUBLTEN,RVBLTEN,RTHBLTEN,RQVBLTEN,RQCBLTEN)):
        if seed is not None:
            seed=profile(seed)
            tend[key] = [value+seed[k] for k,value in enumerate(tend[key])]
    tend = {key:pack(value) for key,value in tend.items()}
    return {**tend, 'tke': pack([.5*jnp.maximum(value,.01) for value in bottom(q2)]),
            'EXCH_H': pack(eh), 'EXCH_M': pack(em),
            'EL_MYJ': pack([jnp.zeros_like(u[0]), *el[::-1]]),
            'PBLH': pblh, 'KPBL': kpbl, 'thz0': thz0, 'qz0': qz0, 'qsfc': qsfc,
            'ct': jnp.zeros_like(ustar)}
