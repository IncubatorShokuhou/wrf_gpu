"""WRF 4.7.1 CU_OSAS/OSASCNV (cu_physics=95), recorded-draw kernel.

Distinct old stochastic deep scheme, not a SAS94 alias. GFS kind_phys
internals are WRF DOUBLE; driver inputs/outputs use WRF REAL. Operational admission
requires the ARW shallow-convection path as well as this deep-convection routine.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.cumulus_scalesas import _consts, _fpvs, _ar, _scan

def _osascnv_column(t1, q1, ql1, ql2, u1, v1, delp, prslp, psp, phil, dot, islimsk,
                    delt, xkt2, *, km: int, mode: str):
    c = _consts(mode)
    L = c.lit
    f64 = jnp.float64
    g = c.grav
    cp, hvap, rd = c.cp, c.hvap, c.rd
    eps, epsm1 = c.eps, c.epsm1
    elocp = hvap / cp
    el2orc = hvap * hvap / (c.rv * cp)
    c0, c1 = L(0.002), L(0.002)
    delta = c.fv
    fact1 = (c.cvap - c.cliq) / c.rv
    fact2 = hvap / c.rv - fact1 * c.t0c
    cthk, dthk = 150.0, 25.0
    cinpcrmx, cinpcrmn = 180.0, 120.0
    tf, tcr = L(233.16), L(263.16)
    tcrf = 1.0 / (tcr - tf)
    ncloud = 1
    fpvs = lambda t: _fpvs(t, mode)  # noqa: E731

    K = _ar(0, km + 1)
    km1 = km - 1
    valid = (K >= 1) & (K <= km)
    zero = jnp.zeros(km + 1, f64)

    ps, prsl, dele = psp, prslp, delp

    cnvflg = jnp.asarray(True)
    mbdt = 10.0

    dt2 = delt
    dtmin = jnp.maximum(dt2, 1200.0)
    dtmax = jnp.maximum(dt2, 3600.0)
    edtmaxl = edtmaxs = L(0.3)
    clam, aafac = L(0.1), L(0.1)
    betal = betas = L(0.05)
    evfact = evfactl = L(0.5)
    cxlamu = L(1.0e-4)
    xlamde, xlamdd = L(1.0e-4), L(1.0e-4)
    w1l, w2l, w3l, w4l = L(-8.0e-3), L(-4.0e-2), L(-5.0e-3), L(-5.0e-4)
    w1s, w2s, w3s, w4s = L(-2.0e-4), L(-2.0e-3), L(-1.0e-3), L(-2.0e-5)

    # kbmax / kbm / kmax: last level satisfying, +1 (default km), clipped
    tx1 = 1.0 / ps
    rp = prsl * tx1

    def _lastp1(cond):
        last = jnp.max(jnp.where(cond & valid, K, 0))
        return jnp.where(last > 0, last + 1, km)

    kmax = _lastp1(rp > L(0.04))
    kbmax = _lastp1(rp > L(0.45))
    kbm = _lastp1(rp > L(0.70))
    kmax = jnp.minimum(km, kmax)
    kbmax = jnp.minimum(kbmax, kmax)
    kbm = jnp.minimum(kbm, kmax)

    zo = phil / g
    zi = jnp.where((K >= 1) & (K <= km1), 0.5 * (zo + jnp.roll(zo, -1)), 0.0)  # zi(km) := 0
    xlamue = jnp.where((K >= 1) & (K <= km1), clam / jnp.where(zi != 0.0, zi, 1.0), 0.0)

    inK = valid & (K <= kmax)
    pfld = jnp.where(inK, prsl * 10.0, 0.0)
    to = jnp.where(inK, t1, 0.0)
    qo = jnp.where(inK, q1, 0.0)
    uo = jnp.where(inK, u1, 0.0)
    vo = jnp.where(inK, v1, 0.0)
    eta = jnp.where(inK, 1.0, 0.0)
    etad = jnp.where(inK, 1.0, 0.0)

    # qeso / qo floor (k <= kmax)
    qeso = L(0.01) * fpvs(to)
    qeso = eps * qeso / (pfld + epsm1 * qeso)
    qeso = jnp.where(inK, jnp.maximum(qeso, L(1.0e-8)), 0.0)
    qo = jnp.where(inK, jnp.maximum(qo, L(1.0e-10)), 0.0)
    tem = phil + cp * to
    heo = jnp.where(inK, tem + hvap * qo, 0.0)
    heso = jnp.where(inK, tem + hvap * qeso, 0.0)

    # kb: level of max moist static energy in 1..kbm (first occurrence of strict max)
    kb = jnp.argmax(jnp.where(valid & (K <= kbm), heo, -jnp.inf)).astype(jnp.int32)

    # interface (layer-top) environment, k <= kmax-1 (uses the not-yet-updated k+1 values)
    def _interface(to, qo, qeso):
        to1 = jnp.roll(to, -1)
        qo1 = jnp.roll(qo, -1)
        qeso1 = jnp.roll(qeso, -1)
        pf1 = jnp.roll(pfld, -1)
        zo1 = jnp.roll(zo, -1)
        m = valid & (K <= km1) & (K <= kmax - 1)
        dz = 0.5 * (zo1 - zo)
        dp = 0.5 * (pf1 - pfld)
        es = L(0.01) * fpvs(jnp.where(m, to1, 300.0))
        pprime = pf1 + epsm1 * es
        qs = eps * es / pprime
        dqsdp = -qs / pprime
        desdt = es * (fact1 / to1 + fact2 / (to1 * to1))
        dqsdt = qs * pf1 * desdt / (es * pprime)
        gamma = el2orc * qeso1 / (to1 * to1)
        dt = (g * dz + hvap * dqsdp * dp) / (cp * (1.0 + gamma))
        dq = dqsdt * dt + dqsdp * dp
        to_n = jnp.where(m, to1 + dt, to)
        qo_n = jnp.where(m, qo1 + dq, qo)
        po = jnp.where(m, 0.5 * (pfld + pf1), 0.0)
        return to_n, qo_n, po, m

    to, qo, po, mI = _interface(to, qo, qeso)
    zo1 = jnp.roll(zo, -1)
    qeso_i = L(0.01) * fpvs(jnp.where(mI, to, 300.0))
    qeso_i = eps * qeso_i / (po + epsm1 * qeso_i)
    qeso = jnp.where(mI, jnp.maximum(qeso_i, L(1.0e-8)), qeso)
    qo = jnp.where(mI, jnp.maximum(qo, L(1.0e-10)), qo)
    frh = jnp.where(mI, 1.0 - jnp.minimum(qo / jnp.where(mI, qeso, 1.0), 1.0), 0.0)
    heo = jnp.where(mI, 0.5 * g * (zo + zo1) + cp * to + hvap * qo, heo)
    heso = jnp.where(mI, 0.5 * g * (zo + zo1) + cp * to + hvap * qeso, heso)
    uo = jnp.where(mI, 0.5 * (uo + jnp.roll(uo, -1)), uo)
    vo = jnp.where(mI, 0.5 * (vo + jnp.roll(vo, -1)), vo)

    def sum_ordered(values, mask, reverse=False, initial=0.):
        def body(old,k):
            return jnp.where(mask[k],old+values[k],old),None
        return _scan(body,jnp.asarray(initial,f64),_ar(1,km+1),reverse=reverse)[0]

    base = valid & (K <= kbmax) & (K > kb) & (heo[kb] > heso)
    kbcon = jnp.where(jnp.any(base),jnp.argmax(base),kmax).astype(jnp.int32)
    cnvflg = (kbcon != kmax) & ((-pfld[kbcon]+pfld[kb]) <= 150.)
    pdot = 10.*dot[kbcon]
    land = islimsk == 1
    alpha = jnp.where(land,L(.5),L(.75))
    dz = .5*(zo[kbcon]+zo[kbcon-1])-jnp.where(kb==1,zo[1],.5*(zo[kb]+zo[kb-1]))
    xlamb = jnp.where(kbcon!=kb,-jnp.log(alpha)/dz,0.)
    def eta_sub(old,k):
        dz=jnp.where(k==1,.5*(zo[2]-zo[1]),.5*(zo[k+1]-zo[k-1]))
        val=jnp.where((k<=kbmax)&(k<kbcon)&(k>=kb),old*jnp.exp(-xlamb*dz),eta[k])
        return val,val
    _,values=_scan(eta_sub,eta[km],_ar(1,km),reverse=True)
    eta=eta.at[1:km].set(values)
    etau=eta
    def initial_plume(environment):
        def body(old,k):
            factor=eta[k-1]/eta[k]
            value=jnp.where(k<=kbcon,factor*old+(1.-factor)*.5*(environment[k]+environment[k+1]),old)
            value=jnp.where((k>kb)&(k<kmax),value,jnp.where(k==kb,environment[kb],0.))
            return value,value
        _,v=_scan(body,jnp.where(kb==1,environment[1],0.),_ar(2,km))
        return zero.at[kb].set(environment[kb]).at[2:km].set(v)
    hcko,ucko,vcko = [initial_plume(a) for a in (heo,uo,vo)]
    dbyo=jnp.where((K>kb)&(K<kmax),hcko-heso,0.)
    top=valid&(K<=kmax)&(K>kbcon)&(dbyo<0.)
    ktcon=jnp.where(jnp.any(top),jnp.argmax(top),1).astype(jnp.int32)
    cnvflg &= (pfld[kbcon]-pfld[ktcon]>=150.)
    hmask=valid&(K>=kbcon)&(K<=kbmax)
    minimum=jnp.argmin(jnp.where(hmask,heo,jnp.inf)).astype(jnp.int32)
    lmin=jnp.where(heo[minimum]<heo[kbcon],minimum+1,kbmax)
    jmin=jnp.maximum(jnp.minimum(lmin,ktcon-1),kbcon+1)
    def sumenv(old,k):
        sz,sh=old
        mask=(k>jmin)&(k<=ktcon)&(k<kmax)
        dz=.5*(zo[k+1]-zo[k-1])
        sz=jnp.where(mask,sz+dz,sz);sh=jnp.where(mask,sh+dz*heo[k],sh)
        return (sz,sh),(sz,sh)
    _,(sz,sh)=_scan(sumenv,(jnp.asarray(0.,f64),jnp.asarray(0.,f64)),_ar(2,km))
    sumz=zero.at[2:km].set(sz);sumh=zero.at[2:km].set(sh)
    # Fortran NINT uses ties away from zero (jnp.rint uses ties-to-even).
    draw=xkt2*(ktcon-jmin).astype(f64)-.5
    kt2=(jnp.sign(draw)*jnp.floor(jnp.abs(draw)+.5)).astype(jnp.int32)+jmin+1
    tem1=hcko[jmin]-heso[kt2]
    tem2=sumz[kt2]*heso[kt2]-sumh[kt2]
    cnvflg &= jnp.abs(tem2)>L(.000001)
    xlamb=jnp.minimum(jnp.maximum(tem1/tem2,0.),L(2.3)/sumz[kt2])
    dwnflg=cnvflg&(kt2<ktcon)&(xlamb>L(1.e-30))&(tem1>L(1.e-30))
    dwnflg &= ~jnp.any(valid&(K>=jmin)&(K<=kt2)&(heo>heso[kt2]))
    dwnflg2=cnvflg  # ARW PDPDWN=0, so the depth test is already satisfied.
    xlambu=L(1.e-4)
    def mass_up(old,k):
        e,eu=old;dz=.5*(zo[k+1]-zo[k-1])
        mask=(k>jmin)&(k<kmax)
        e=jnp.where(mask&dwnflg&(k<=kt2),e*(1.+xlamb*dz),eta[k])
        eu=jnp.where(mask&dwnflg&(k<=kt2),eu*(1.+(xlamb+xlambu)*dz),etau[k])
        eu=jnp.where(mask&~dwnflg&(k<=ktcon),old[1]*(1.+xlambu*dz),eu)
        return (e,eu),(e,eu)
    _,(e,eu)=_scan(mass_up,(eta[1],etau[1]),_ar(2,km))
    eta=eta.at[2:km].set(e);etau=etau.at[2:km].set(eu)
    ktcon=jnp.where(dwnflg,kt2,ktcon)
    def entrained_plume(environment,old,mass):
        def body(prev,k):
            factor=mass[k-1]/mass[k]
            value=factor*prev+(1.-factor)*.5*(environment[k]+environment[k+1])
            value=jnp.where((k>kbcon)&(k<=ktcon)&(k<kmax),value,old[k])
            return value,value
        _,a=_scan(body,old[1],_ar(2,km))
        return old.at[2:km].set(a)
    hcko=entrained_plume(heo,hcko,eta)
    ucko=entrained_plume(uo,ucko,etau);vcko=entrained_plume(vo,vcko,etau)
    dbyo=jnp.where((K>kbcon)&(K<=ktcon)&(K<kmax),hcko-heso,dbyo)
    cnvflg &= ~(dwnflg2&(jmin<=kbcon))
    dwnflg &= cnvflg;dwnflg2 &= cnvflg

    def cloud_work(to,qo,qeso,heo,heso,hcko,*,feedback=False):
        """Original sequential condensate and buoyancy work; fixed mass flux profile."""
        def qbody(old,k):
            qc,aa,pwav=old
            active=(k>kb)&(k<ktcon)&(k<kmax)
            dz=.5*(zo[k+1]-zo[k-1]);dz1=zo[k]-zo[k-1]
            gamma=el2orc*qeso[k]/(to[k]**2)
            dby=hcko[k]-heso[k]
            if feedback:dby=jnp.maximum(dby,0.)
            qrch=qeso[k]+gamma*dby/(hvap*(1.+gamma))
            factor=eta[k-1]/eta[k]
            qk=factor*qc+(1.-factor)*.5*(qo[k]+qo[k+1])
            dq=eta[k]*qk-eta[k]*qrch
            etah=.5*(eta[k]+eta[k-1])
            qlk=dq/(eta[k]+etah*c0*dz)
            precip=etah*c0*dz*qlk
            aa=jnp.where(active&(dq>0.),aa-dz1*g*qlk,aa)
            pwav=jnp.where(active&(dq>0.),pwav+precip,pwav)
            if feedback:
                gamma0=el2orc*qeso[k-1]/to[k-1]**2
                value=aa+dz1*(g/(cp*to[k-1]))*(hcko[k-1]-heso[k-1])/(1.+gamma0)*(1.+delta*cp*gamma0*to[k-1]/hvap)
                value=value+dz1*g*delta*jnp.maximum(0.,qeso[k-1]-qo[k-1])
                aa=jnp.where((k>kbcon)&(k<=ktcon)&(k<kmax),value,aa)
            qc=jnp.where(active,jnp.where(dq>0.,qlk+qrch,qk),jnp.where(k==kb,qo[kb],0.))
            return (qc,aa,pwav),(qc,jnp.where(active&(dq>0.),precip,0.))
        (_,aa,pwav),(qc,pwo)=_scan(qbody,(jnp.where(kb==1,qo[1],0.),jnp.asarray(0.,f64),jnp.asarray(0.,f64)),_ar(2,km))
        qcko=zero.at[kb].set(qo[kb]).at[2:km].set(qc)
        pwo=zero.at[2:km].set(pwo)
        def buoyancy(old,k):
            mask=(k>kbcon)&(k<=ktcon)&(k<=kmax)
            dz1=zo[k]-zo[k-1];gamma=el2orc*qeso[k-1]/(to[k-1]**2)
            rfact=1.+delta*cp*gamma*to[k-1]/hvap
            value=old+dz1*(g/(cp*to[k-1]))*(hcko[k-1]-heso[k-1])/(1.+gamma)*rfact
            value=value+dz1*g*delta*jnp.maximum(0.,qeso[k-1]-qo[k-1])
            return jnp.where(mask,value,old),None
        if not feedback:
            aa=_scan(buoyancy,aa,_ar(1,km+1))[0]
        return aa,pwav,qcko,pwo
    aa1,pwavo,qcko,pwo=cloud_work(to,qo,qeso,heo,heso,hcko)
    gamma=el2orc*qeso[ktcon]/(to[ktcon]**2)
    qrch=qeso[ktcon]+gamma*dbyo[ktcon]/(hvap*(1.+gamma))
    qlko_ktcon=jnp.maximum(qcko[ktcon-1]-qrch,0.)
    qcko=qcko.at[ktcon-1].set(jnp.where(qlko_ktcon>0.,qrch,qcko[ktcon-1]))
    cnvflg &= aa1>0.;dwnflg2 &= cnvflg
    shear=jnp.sqrt((jnp.roll(uo,-1)-uo)**2+(jnp.roll(vo,-1)-vo)**2)
    vshear=sum_ordered(shear,valid&(K>=kb)&(K<=ktcon)&(K<=kmax))
    vshear=1000.*vshear/(zo[ktcon]-zo[kb])
    edt=jnp.clip(1.-(L(1.591)-L(.639)*vshear+L(.0953)*(vshear**2)-L(.00496)*(vshear**3)),0.,L(.9))
    kbdtr=jnp.maximum(kbcon,1)
    dz=.5*zo[kbdtr]+.5*zo[kbdtr-1]-zo[1]
    xlamd=jnp.where(kbdtr>1,jnp.log(L(.05))/dz,0.)
    def mass_down(old,k):
        dz=jnp.where(k==1,.5*(zo[2]-zo[1]),.5*(zo[k+1]-zo[k-1]))
        val=jnp.where((k<kbdtr)&(k<=kbmax),old*jnp.exp(xlamd*dz),etad[k])
        return val,val
    _,a=_scan(mass_down,etad[km],_ar(1,km),reverse=True)
    etad=etad.at[1:km].set(a)
    hcdo,ucdo,vcdo=heo[jmin],uo[jmin],vo[jmin]
    def downdraft(to,qo,qeso,heo,heso,feedback=False):
        hcd=heo[jmin]
        def body(old,k):
            qc,pwev=old
            mask=(k<jmin)&(k<kmax)
            gamma=el2orc*qeso[k]/to[k]**2
            qrc=qeso[k]+(1./hvap)*(gamma/(1.+gamma))*(hcd-heso[k])
            detad=etad[k+1]-etad[k]
            pw=etad[k+1]*qc-etad[k]*qrc
            pw=pw-detad*.5*(qrc+qc)
            pwev=jnp.where(mask,pwev+pw,pwev)
            # Original first downdraft starts QC=QO; feedback starts QESO.
            qc=jnp.where(mask,qrc,jnp.where(k==jmin,jnp.where(feedback,qeso[jmin],qo[jmin]),qc))
            return (qc,pwev),(jnp.where(mask,qrc,jnp.where(k==jmin,qeso[jmin],0.)),jnp.where(mask,pw,0.))
        (qc,pwev),(qrc,pw)=_scan(body,(qeso[jmin] if feedback else qo[jmin],jnp.asarray(0.,f64)),_ar(1,km),reverse=True)
        return qc,pwev,zero.at[1:km].set(qrc),zero.at[1:km].set(pw)
    qcdo,pwevo,qrcdo,pwdo=downdraft(to,qo,qeso,heo,heso)
    edto=jnp.where(dwnflg2&(pwevo<0.),jnp.minimum(-edt*pwavo/pwevo,L(.3)),0.)
    def dd_work(aa,to,qo,qeso,heso,hcd,edt0):
        def body(old,k):
            dz=-1.*(zo[k+1]-zo[k]);t=to[k+1]
            gamma=el2orc*qeso[k+1]/t**2
            val=old+edt0*dz*(g/(cp*t))*((hcd-heso[k+1])/(1.+gamma))*(1.+delta*cp*gamma*t/hvap)
            val=val+edt0*dz*g*delta*jnp.maximum(0.,qeso[k+1]-qo[k+1])
            return jnp.where(dwnflg2&(k<jmin)&(k<kmax),val,old),None
        return _scan(body,aa,_ar(1,km),reverse=True)[0]
    aa1=dd_work(aa1,to,qo,qeso,heso,hcdo,edto)
    cnvflg &= aa1>0.;dwnflg2 &= cnvflg
    # A-grid budget for the original mass flux; no PGCON in old OSAS.
    aup=jnp.where(K<=kb,0.,1.);adw=jnp.where(K>jmin,0.,1.)
    eta0=jnp.roll(eta,1);etad0=jnp.roll(etad,1)
    deta=eta-eta0;detad=etad-etad0
    def budget(env,plume,down,moisture=False):
        dv2=.5*(env+jnp.roll(env,-1))
        dd=.5*(qrcdo+jnp.roll(qrcdo,1)) if moisture else down
        value=((aup*eta-adw*edto*etad)*env-(aup*eta0-adw*edto*etad0)*jnp.roll(env,1)
               -aup*deta*dv2+adw*edto*detad*dd)*g/(1000.*dele)
        value=jnp.where((K>=2)&(K<ktcon)&(K<kmax),value,0.)
        value=value.at[1].set(edto*etad[1]*(down-env[1])*g/(1000.*dele[1]))
        value=value.at[ktcon].set(eta[ktcon-1]*(plume[ktcon-1]-env[ktcon-1])*g/(1000.*dele[ktcon]))
        return value
    dellah=budget(heo,hcko,hcdo);dellaq=budget(qo,qcko,qcdo,True)
    dellau=budget(uo,ucko,ucdo);dellav=budget(vo,vcko,vcdo)
    dellal=eta[ktcon-1]*qlko_ktcon*g/(1000.*dele[ktcon])
    # Recompute work after the source's 10 s trial response.
    mask=valid&(K<=ktcon)&(K<=kmax)
    tq=jnp.where(mask,(dellah-hvap*dellaq)/cp*mbdt+t1,t1)
    qq=jnp.where(mask,jnp.maximum(dellaq*mbdt+q1,L(1.e-10)),q1)
    qe=L(.01)*fpvs(tq);qe=jnp.maximum(eps*qe/(pfld+epsm1*qe),L(1.e-8))
    tfb,qfb,pfb,mi=_interface(tq,qq,qe)
    eq=L(.01)*fpvs(tfb);eq=jnp.where(mi,jnp.maximum(eps*eq/(pfb+epsm1*eq),L(1.e-8)),qe)
    qfb=jnp.where(mi,jnp.maximum(qfb,L(1.e-10)),qfb)
    hfb=jnp.where(mi,.5*g*(zo+zo1)+cp*tfb+hvap*qfb,g*zo+cp*tfb+hvap*qfb)
    hsfb=jnp.where(mi,.5*g*(zo+zo1)+cp*tfb+hvap*eq,g*zo+cp*tfb+hvap*eq)
    def feedback_h(old,k):
        factor=eta[k-1]/eta[k]
        val=factor*old+(1.-factor)*.5*(hfb[k]+hfb[k+1])
        val=jnp.where((k>kb)&(k<=ktcon)&(k<kmax),val,jnp.where(k==kb,hfb[kb],0.))
        return val,val
    _,a=_scan(feedback_h,jnp.where(kb==1,hfb[1],0.),_ar(2,km))
    htrial=zero.at[kb].set(hfb[kb]).at[2:km].set(a)
    xaa0,xpwav,_,_=cloud_work(tfb,qfb,eq,hfb,hsfb,htrial,feedback=True)
    _,xpwev,_,_=downdraft(tfb,qfb,eq,hfb,hsfb,feedback=True)
    edtx=jnp.where(dwnflg2&(xpwev<0.),jnp.minimum(-edt*xpwav/xpwev,L(.3)),0.)
    xaa0=dd_work(xaa0,tfb,qfb,eq,hsfb,hfb[jmin],edtx)
    pcrit=jnp.asarray([850.,800.,750.,700.,650.,600.,550.,500.,450.,400.,350.,300.,250.,200.,150.],f64)
    acrit=jnp.asarray([.0633,.0445,.0553,.0664,.075,.1082,.1521,.2216,.3151,.3677,.41,.5255,.7663,1.1686,1.6851],
                       dtype=jnp.float32 if mode=='r4' else f64).astype(f64)*(975.-pcrit)
    # ACRITT DATA values are kind_phys, but unsuffixed literals round as REAL.
    ix=jnp.clip(((850.-pfld[ktcon])/50.).astype(jnp.int32)+1,1,14)
    acrt=acrit[ix]+(acrit[ix-1]-acrit[ix])*(pfld[ktcon]-pcrit[ix])/(pcrit[ix-1]-pcrit[ix])
    acrt=jnp.where(pfld[ktcon]<pcrit[14],acrit[14]*(975.-pfld[ktcon])/(975.-pcrit[14]),
                   jnp.where(pfld[ktcon]>pcrit[0],acrit[0],acrt))
    w1,w2,w3,w4=[jnp.where(land,a,b) for a,b in ((w1l,w1s),(w2l,w2s),(w3l,w3s),(w4l,w4s))]
    acrtfct=jnp.where(pdot<=w4,(pdot-w4)/(w3-w4),jnp.where(pdot>=-w4,-(pdot+w4)/(w4-w3),0.))
    acrtfct=1.-jnp.clip(acrtfct,-1.,1.)
    dtconv=jnp.clip(dt2+jnp.maximum(1800.-dt2,0.)*(pdot-w2)/(w1-w2),dtmin,dtmax)
    fld=(aa1-acrt*acrtfct)/dtconv
    xk=(xaa0-aa1)/mbdt
    cnvflg &= (fld>0.)&(xk<0.)
    xmb=jnp.minimum(-fld/xk,L(.1))
    mask=cnvflg&valid&(K<=ktcon)&(K<=kmax)
    t1n=jnp.where(mask,t1+(dellah-hvap*dellaq)/cp*xmb*dt2,t1)
    q1n=jnp.where(mask,q1+dellaq*xmb*dt2,q1)
    # WRF only detrains old OSAS cloud condensate at KTCON.
    tem=dellal*xmb*dt2
    frac=jnp.clip((tcr-t1n)*tcrf,0.,1.)
    qm=mask&(K==ktcon);keep=ql2>L(-999.)
    total=ql1+tem
    ql1n=jnp.where(qm,jnp.where(keep,ql1+tem*frac,total*frac),ql1)
    ql2n=jnp.where(qm,jnp.where(keep,ql2+tem*(1.-frac),total-total*frac),ql2)
    qeso=L(.01)*fpvs(t1n);qeso=jnp.maximum(eps*qeso/(pfld+epsm1*qeso),L(1.e-8))
    rain=aup*pwo+adw*edto*pwdo
    rntot=sum_ordered(rain*xmb*L(.001)*dt2,mask,reverse=True)
    def evaporation(old,k):
        rn,flg,delqev,delq2,tn,qn=old
        rn=jnp.where(mask[k],rn+rain[k]*xmb*L(.001)*dt2,rn)
        active=flg&(k<=ktcon)&(k<=kmax)
        qcond=edt*evfact*(qn[k]-qeso[k])/(1.+el2orc*qeso[k]/tn[k]**2)
        dp=1000.*dele[k]
        evap=jnp.where(active&(rn>0.)&(qcond<0.),
                       jnp.minimum(-qcond*(1.-jnp.exp(-L(.32)*jnp.sqrt(dt2*rn))),rn*1000.*g/dp),0.)
        delq2=jnp.where(active&(rn>0.)&(qcond<0.),delqev+L(.001)*evap*dp/g,delq2)
        exhaust=active&(rn>0.)&(qcond<0.)&(delq2>rntot)
        evap=jnp.where(exhaust,1000.*g*(rntot-delqev)/dp,evap)
        flg &= ~exhaust
        change=active&(rn>0.)&(evap>0.)
        qn=qn.at[k].set(jnp.where(change,qn[k]+evap,qn[k]))
        tn=tn.at[k].set(jnp.where(change,tn[k]-elocp*evap,tn[k]))
        rn=jnp.where(change,rn-L(.001)*evap*dp/g,rn)
        delqev=jnp.where(change,delqev+L(.001)*dp*evap/g,delqev)
        return (rn,flg,delqev,delq2,tn,qn),None
    (rn,_,_,_,t1n,q1n),_=_scan(evaporation,(jnp.asarray(0.,f64),cnvflg,jnp.asarray(0.,f64),
        jnp.asarray(0.,f64),t1n,q1n),_ar(1,km+1),reverse=True)
    rn=jnp.maximum(rn,0.)
    active=cnvflg&(rn>0.)
    t1n=jnp.where(cnvflg&(rn<=0.)&inK,t1,t1n)
    q1n=jnp.where(cnvflg&(rn<=0.)&inK,q1,q1n)
    return t1n,q1n,ql1n,ql2n,rn,jnp.where(active,kbcon,km+1),jnp.where(active,ktcon,0),jnp.where(active,1,0)



def cu_osas95_columns(t3d, qv3d, qc3d, qi3d, pcps, pi3d, rho3d, dz8w,
                     u3d, v3d, w, psfc, xland, dt, stepcu, *, draws, shallow=True, literals="r4"):
    """CU_OSAS driver arrays with recorded XKT2, bottom-up (columns, levels), SI units.

    ``shallow=False`` isolates OSASCNV for component qualification. ARW defaults
    to SHALCV enabled; the operational adapter uses that default.
    """
    c = _consts(literals)
    f64 = jnp.float64
    dtype = jnp.float32 if literals == "r4" else f64
    R = lambda x: jnp.asarray(x, dtype)
    T, QV, QC, QI, P, PI, RHO, DZ, U, V, W, PS, XL = map(
        R, (t3d, qv3d, qc3d, qi3d, pcps, pi3d, rho3d, dz8w, u3d, v3d, w, psfc, xland))
    ncol, kx = T.shape
    dt_ = c.lit(dt)
    stepcu = int(stepcu)

    def zi_body(carry, dz):
        nxt = carry + dz.astype(f64)
        return nxt, nxt

    _, zi = jax.lax.scan(zi_body, jnp.zeros(ncol, f64), DZ[:, :-1].T)
    zi = jnp.concatenate([jnp.zeros((ncol, 1), f64), zi.T], axis=1)
    zlm = (zi[:, 1:] + zi[:, :-1]) * 0.5
    zl = jnp.concatenate([zlm, (2.0 * zi[:, -1] - zlm[:, -1])[:, None]], axis=1)
    # PCPS*.001 and W(k)+W(k+1) are default REAL expressions in the driver;
    # introducing a kind_phys constant later promotes the already-rounded sum.
    prsl = R(P * c.lit(.001)).astype(f64)
    ps = R(PS * c.lit(.001)).astype(f64)
    phil = zl * c.grav
    dot = (-jnp.asarray(c.lit(5.0e-4),f64)*c.grav * RHO.astype(f64)
           * (W[:, :-1] + W[:, 1:]).astype(f64))
    dele = prsl * c.grav / c.rd * DZ / T
    q1 = R(QV / R(1.0 + QV))
    ql1 = R(QI / R(1.0 + QI))
    ql2 = R(QC / R(1.0 + QC))
    delt = (R(dt_) * stepcu).astype(f64)
    pad = lambda a: jnp.pad(a.astype(f64), ((0, 0), (1, 0)))
    col = jax.vmap(lambda *a: _osascnv_column(*a, km=kx, mode=literals),
                    in_axes=(0,) * 12 + (None, 0))
    t1, q1, ql1, ql2, rn, kbot, ktop, kuo = col(
        pad(T), pad(q1), pad(ql1), pad(ql2), pad(U), pad(V), pad(dele), pad(prsl),
        ps, pad(phil), pad(dot), jnp.abs(XL - 2.0).astype(f64), delt,
        jnp.asarray(draws,f64))
    if shallow:
        from gpuwrf.physics.gfs_sas_shallow import old_arw_shalcv
        # PRSI(k+1)=PRSI(k)-DEL(k), sequential like CU_SAS.
        def pressure_step(prev, thickness):
            nxt = prev-thickness
            return nxt, nxt
        _, prsi_up = jax.lax.scan(pressure_step, ps, dele.T)
        prsi = jnp.concatenate([jnp.zeros((ncol,1), f64), ps[:,None], prsi_up.T], axis=1)
        prslk = (prsl*c.lit(.01))**(c.rd/c.cp)
        t1, q1 = jax.vmap(lambda *a: old_arw_shalcv(*a, dt=delt, mode=literals))(
            t1, q1, pad(dele), prsi, pad(prsl), pad(prslk), kuo)
    t1, q1, ql1, ql2 = (a[:, 1:] for a in (t1, q1, ql1, ql2))
    rdelt = 1.0 / delt
    return {key: R(value) for key, value in {
        "RTHCUTEN": R((t1 - T) / PI * rdelt),
        "RQVCUTEN": R((q1 / (1.0 - q1) - QV) * rdelt),
        "RQCCUTEN": R((ql2 / (1.0 - ql2) - QC) * rdelt),
        "RQICUTEN": R((ql1 / (1.0 - ql1) - QI) * rdelt),
        "RAINCV": R(rn * 1000.0 / stepcu),
        "PRATEC": R(rn * 1000.0 / (stepcu * R(dt_)).astype(f64)),
        "HBOT": kbot, "HTOP": ktop,
    }.items()}
