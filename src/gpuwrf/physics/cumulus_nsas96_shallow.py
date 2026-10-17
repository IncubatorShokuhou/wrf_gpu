"""Native-REAL NSAS nscv2d shallow convection, separate unadmitted candidate."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from gpuwrf.physics.cumulus_scalesas import _ar,_scan
from gpuwrf.physics.cumulus_ksas14 import _constants,_fpvs


def nsas96_shallow_column(t1,q1,ql1,ql2,u1,v1,dele,prsl,ps,prslk,zo,dot,islimsk,
                          hpbl,hfx,qfx,delt,kbot_in,ktop_in,icps,*,km:int,mode:str,
                          ncloud:int,pgcon:float=.55):
    c = _constants(mode)
    L = c.lit
    real_t = jnp.float32 if mode=="r4" else jnp.float64
    g,cp,hvap,rd = c.grav,c.cp,c.hvap,c.rd
    eps,epsm1,delta = c.eps,c.epsm1,c.fv
    el2orc = hvap*hvap/(c.rv*cp)
    fact1 = (c.cvap-c.cliq)/c.rv
    fact2 = hvap/c.rv-fact1*c.t0c
    c0,c1,clam,aafac = L(.002),L(5.e-4),L(.3),L(.1)
    cinpcrmx,cinpcrmn,dthk = L(180.),L(120.),L(25.)
    tcr,tf = L(263.16),L(233.16)
    tcrf = L(1.)/(tcr-tf)
    fpvs = lambda t:_fpvs(t,mode)
    pgcon = jnp.asarray(pgcon,real_t)
    K = _ar(0,km+1)
    km1 = km-1
    valid = (K>=1)&(K<=km)
    zero = jnp.zeros(km+1,real_t)
    dt2 = delt
    rhox = prsl[1]*L(1.e3)/(rd*t1[1]*(1.+delta*q1[1]))
    thx1 = t1[1]/prslk[1]
    sflx = hfx/rhox/cp+qfx/rhox*delta*thx1
    initial_active = (icps!=1)&(sflx>0.)
    cnvflg = initial_active
    initial_kbot = jnp.where(initial_active,km+1,kbot_in)
    initial_ktop = jnp.where(initial_active,0,ktop_in)
    def lastp1(condition):
        last = jnp.max(jnp.where(valid&condition,K,0))
        return jnp.where(last>0,last+1,km)
    kmax = lastp1(prsl>ps*L(.60))
    kbm = jnp.minimum(lastp1(prsl>ps*L(.70)),kmax)
    zi = jnp.where((K>=1)&(K<=km1),.5*(zo+jnp.roll(zo,-1)),0.)
    xlamue = jnp.where((K>=1)&(K<=km1),clam/jnp.where(zi!=0.,zi,1.),0.)
    xlamue = xlamue.at[km].set(xlamue[km1])
    def pbl_search(carry,k):
        searching,kpbl = carry
        hit = searching&(zo[k]<=hpbl)
        return (hit,jnp.where(hit,k,kpbl)),None
    (_,kpbl),_ = _scan(pbl_search,(cnvflg,jnp.asarray(1,jnp.int32)),_ar(2,km))
    kpbl = jnp.minimum(kpbl,kbm)
    c0t = jnp.full_like(t1,c0)
    inK = valid & (K <= kmax)
    pfld = jnp.where(inK, prsl * 10.0, 0.0)
    to = jnp.where(inK, t1, 0.0)
    qo = jnp.where(inK, q1, 0.0)
    uo = jnp.where(inK, u1, 0.0)
    vo = jnp.where(inK, v1, 0.0)
    eta = jnp.where(inK, L(1.), L(0.))

    # qeso / qo floor (k <= kmax)
    qeso = L(0.01) * fpvs(to)
    qeso = eps * qeso / (pfld + epsm1 * qeso)
    qeso = jnp.where(inK, jnp.maximum(qeso, L(1.0e-8)), 0.0)
    qo = jnp.where(inK, jnp.maximum(qo, L(1.0e-10)), 0.0)
    tem = g * zo + cp * to
    heo = jnp.where(inK, tem + hvap * qo, 0.0)
    heso = jnp.where(inK, tem + hvap * qeso, 0.0)

    kb = jnp.argmax(jnp.where(valid&(K<=kpbl),heo,-jnp.inf)).astype(jnp.int32)

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

    c_kbcon = (K>=2)&(K<=km1)&(K<kbm)&(K>kb)&(heo[kb]>heso)
    kbcon = jnp.where(jnp.any(c_kbcon),jnp.argmax(c_kbcon),kmax).astype(jnp.int32)
    cnvflg = cnvflg&(kbcon!=kmax)
    pdot = 10.*dot[kbcon]
    land = islimsk==1
    w3,w4 = jnp.where(land,L(-5.e-3),L(-1.e-3)),jnp.where(land,L(-5.e-4),L(-2.e-5))
    ptem = jnp.where(pdot<=w4,(pdot-w4)/(w3-w4),jnp.where(pdot>=-w4,-(pdot+w4)/(w4-w3),0.))
    ptem = 1.-jnp.minimum(jnp.maximum(ptem,-1.),1.)
    cincr = cinpcrmx-ptem*.5*(cinpcrmx-cinpcrmn)
    cnvflg = cnvflg&~((pfld[kb]-pfld[kbcon])>cincr)
    xlamx = xlamue[kbcon]
    # updraft mass flux below cloud base (downward recurrence from eta(kbcon)=1)
    def eta_sub(carry, k):
        eta_next = carry  # eta(k+1)
        m = (k < kbcon) & (k >= kb)
        dz = zi[k + 1] - zi[k]
        tem = xlamx
        ptem = 0.5 * (xlamue[k] + xlamue[k + 1]) - tem
        val = jnp.where(m, eta_next / (1.0 + ptem * dz), eta[k])
        return val, val

    _, eta_lo = _scan(eta_sub, eta[km], _ar(1, km), reverse=True)
    eta = eta.at[1:km].set(eta_lo)

    def eta_up(prev,k):
        dz = zi[k]-zi[k-1]
        ptem = .5*(xlamue[k]+xlamue[k-1])-xlamx
        val = jnp.where((k>kbcon)&(k<kmax),prev*(1.+ptem*dz),eta[k])
        return val,val
    _, hi = _scan(eta_up,eta[1],_ar(2,km))
    eta = eta.at[2:km].set(hi)
    # updraft moist static energy (hcko), buoyancy excess dbyo
    hcko0 = jnp.where(K == kb, heo[kb], 0.0)

    def hcko_up(carry, k):
        h_prev = carry
        m = (k > kb) & (k < kmax)
        dz = zi[k] - zi[k - 1]
        tem = 0.5 * (xlamue[k] + xlamue[k - 1]) * dz
        tem1 = 0.5*xlamx * dz
        factor = 1.0 + tem - tem1
        h = ((1.0 - tem1) * h_prev + tem * 0.5 * (heo[k] + heo[k - 1])) / factor
        h = jnp.where(m, h, hcko0[k])
        db = jnp.where(m, h - heso[k], 0.0)
        return h, (h, db)

    _, (hcko_hi, dbyo_hi) = _scan(hcko_up, hcko0[1], _ar(2, km))
    hcko = hcko0.at[2:km].set(hcko_hi)
    dbyo = zero.at[2:km].set(dbyo_hi)

    # Shallow updraft momentum is computed before the cloud-top search.
    def momentum_up(prev,k):
        up,vp = prev
        dz = zi[k]-zi[k-1]
        tem = .5*(xlamue[k]+xlamue[k-1])*dz
        tem1 = .5*xlamx*dz
        factor = 1.+tem-tem1
        ptem,ptem1 = .5*tem+pgcon,.5*tem-pgcon
        u = ((1.-tem1)*up+ptem*uo[k]+ptem1*uo[k-1])/factor
        v = ((1.-tem1)*vp+ptem*vo[k]+ptem1*vo[k-1])/factor
        m = (k>kb)&(k<kmax)
        u,v = jnp.where(m,u,jnp.where(k==kb,uo[kb],0.)),jnp.where(m,v,jnp.where(k==kb,vo[kb],0.))
        return (u,v),(u,v)
    ucko = jnp.where(K==kb,uo[kb],0.)
    vcko = jnp.where(K==kb,vo[kb],0.)
    _,(uu,vv) = _scan(momentum_up,(ucko[1],vcko[1]),_ar(2,km))
    ucko,vcko = ucko.at[2:km].set(uu),vcko.at[2:km].set(vv)

    candidates = (K>=2)&(K<=km1)&(K<kbm)&(K>=kbcon)&(dbyo>0.)
    kbcon1 = jnp.where(jnp.any(candidates),jnp.argmax(candidates),kmax).astype(jnp.int32)
    cnvflg = cnvflg&(kbcon1!=kmax)&~((pfld[kbcon]-pfld[kbcon1])>dthk)
    candidates = (K>=2)&(K<=km1)&(K<kbm)&(K>kbcon1)&(dbyo<0.)
    ktcon = jnp.where(jnp.any(candidates),jnp.argmax(candidates),kbm).astype(jnp.int32)
    xmbmax = 1000.*dele[kbcon]/(g*dt2)
    # updraft moisture, condensate, precipitation, buoyancy (k < ktcon)
    qcko0 = jnp.where(K == kb, qo[kb], 0.0)

    def qcko_main(carry, k):
        q_prev = carry
        m = (k > kb) & (k < ktcon)
        dz = zi[k] - zi[k - 1]
        gamma = el2orc * qeso[k] / (to[k] * to[k])
        qrch = qeso[k] + gamma * dbyo[k] / (hvap * (1.0 + gamma))
        tem = 0.5 * (xlamue[k] + xlamue[k - 1]) * dz
        tem1 = 0.5*xlamx * dz
        factor = 1.0 + tem - tem1
        qc = ((1.0 - tem1) * q_prev + tem * 0.5 * (qo[k] + qo[k - 1])) / factor
        qrc = qc
        dq = eta[k]*(qc-qrch)
        cond = (k >= kbcon) & (dq > 0.0)
        etah = 0.5 * (eta[k] + eta[k - 1])
        dp = 1000.0 * dele[k]
        upper = (ncloud > 0)
        ptem = c0t[k] + c1
        qlk = jnp.where(upper, dq / (eta[k] + etah * ptem * dz), dq / (eta[k] + etah * c0t[k] * dz))
        dlal = jnp.where(cond & upper, etah * c1 * dz * qlk * g / dp, 0.0)
        buo = jnp.where(cond, -dz * g * qlk, 0.0)
        qc = jnp.where(cond, qlk + qrch, qc)
        pw = jnp.where(cond, etah * c0t[k] * dz * qlk, 0.0)
        drag = jnp.asarray(0.,real_t)
        qc = jnp.where(m, qc, qcko0[k])
        out = (qc, jnp.where(m, qrc, jnp.where(k == kb, qo[kb], 0.0)), jnp.where(m, dlal, 0.0),
               jnp.where(m, buo, 0.0), jnp.where(m, pw, 0.0), jnp.where(m, drag, 0.0))
        return qc, out

    _, (qc_hi, qrc_hi, dlal_hi, buo_hi, pw_hi, drag_hi) = _scan(qcko_main, qcko0[1], _ar(2, km))
    qcko = qcko0.at[2:km].set(qc_hi)
    qrcko = jnp.where(K == kb, qo[kb], 0.0).at[2:km].set(qrc_hi)
    dellal = zero.at[2:km].set(dlal_hi)
    buo = zero.at[2:km].set(buo_hi)
    pwo = zero.at[2:km].set(pw_hi)
    drag = zero.at[2:km].set(drag_hi)

    def seq_sum(vals, mask, init, ks, reverse=False):
        def body(acc, k):
            return jnp.where(mask[k], acc + vals[k], acc), None
        out, _ = _scan(body, init, ks, reverse=reverse)
        return out

    pwavo = seq_sum(pwo, (K > kb) & (K < ktcon), jnp.asarray(0.0, real_t), _ar(2, km))
    # SASCNVN first sums condensate loading with dz, then environmental work with dz1.
    dzi = zi - jnp.roll(zi, 1)
    aa1 = seq_sum(buo, (K > kb) & (K < ktcon), jnp.asarray(0.0, real_t), _ar(2, km))
    def work_up(acc, k):
        m = (k >= kbcon) & (k < ktcon)
        dz1 = zo[k + 1] - zo[k]
        gamma = el2orc * qeso[k] / (to[k] * to[k])
        rfact = 1.0 + delta * cp * gamma * to[k] / hvap
        a = acc + dz1 * (g / (cp * to[k])) * dbyo[k] / (1.0 + gamma) * rfact
        a = a + dz1 * g * delta * jnp.maximum(0.0, qeso[k] - qo[k])
        return jnp.where(m, a, acc), None
    aa1, _ = _scan(work_up, aa1, _ar(2, km))
    cnvflg = cnvflg & ~(aa1 <= 0.0)

    # overshooting: level where aafac*aa1 + work above ktcon becomes negative
    def aa2_acc(carry, k):
        aa2, flg, ktc1 = carry
        m = flg & (k >= ktcon) & (k < kbm)
        dz1 = zo[k + 1] - zo[k]
        gamma = el2orc * qeso[k] / (to[k] * to[k])
        rfact = 1.0 + delta * cp * gamma * to[k] / hvap
        a = aa2 + dz1 * (g / (cp * to[k])) * dbyo[k] / (1.0 + gamma) * rfact
        aa2 = jnp.where(m, a, aa2)
        stop = m & (a < 0.0)
        return (aa2, flg & ~stop, jnp.where(stop, k, ktc1)), None

    (_, _, ktcon1), _ = _scan(aa2_acc, (aafac * aa1, cnvflg, kbm), _ar(2, km))

    # condensate in overshooting layers ktcon <= k < ktcon1 (continues the qcko recurrence)
    def qcko_over(carry, k):
        q_prev = carry
        m = (k >= ktcon) & (k < ktcon1)
        dz = zi[k] - zi[k - 1]
        gamma = el2orc * qeso[k] / (to[k] * to[k])
        qrch = qeso[k] + gamma * dbyo[k] / (hvap * (1.0 + gamma))
        tem = 0.5 * (xlamue[k] + xlamue[k - 1]) * dz
        tem1 = 0.5*xlamx * dz
        factor = 1.0 + tem - tem1
        qc = ((1.0 - tem1) * q_prev + tem * 0.5 * (qo[k] + qo[k - 1])) / factor
        qrc = qc
        dq = eta[k]*(qc-qrch)
        cond = dq > 0.0
        etah = 0.5 * (eta[k] + eta[k - 1])
        dp = 1000.0 * dele[k]
        ptem = c0t[k] + c1
        qlk = jnp.where(ncloud>0,dq/(eta[k]+etah*ptem*dz),dq/(eta[k]+etah*c0t[k]*dz))
        dlal = jnp.where(ncloud>0,etah*c1*dz*qlk*g/dp,0.)
        qc = jnp.where(cond, qlk + qrch, qc)
        pw = jnp.where(cond, etah * c0t[k] * dz * qlk, 0.0)
        qc = jnp.where(m, qc, qcko[k])
        out = (qc, jnp.where(m, qrc, qrcko[k]), jnp.where(m & cond, dlal, dellal[k]),
               jnp.where(m, pw, pwo[k]))
        return qc, out

    _, (qc_hi, qrc_hi, dlal_hi, pw_hi) = _scan(qcko_over, qcko[1], _ar(2, km))
    qcko = qcko.at[2:km].set(qc_hi)
    qrcko = qrcko.at[2:km].set(qrc_hi)
    dellal = dellal.at[2:km].set(dlal_hi)
    pwo = pwo.at[2:km].set(pw_hi)
    pwavo = seq_sum(pwo, (K >= ktcon) & (K < ktcon1) & (pwo != 0.0), pwavo, _ar(2, km))

    ktcon_f = jnp.where(cnvflg,ktcon1,ktcon)
    ktcon1 = jnp.where(cnvflg,ktcon,ktcon1)
    ktcon = ktcon_f
    # liquid/vapour separation at cloud top
    kt1 = ktcon - 1
    gamma = el2orc * qeso[kt1] / (to[kt1] * to[kt1])
    qrch = qeso[kt1] + gamma * dbyo[kt1] / (hvap * (1.0 + gamma))
    dq = qcko[kt1] - qrch
    sep = cnvflg & (dq > 0.0) & (ncloud>0)
    qlko_ktcon = jnp.where(sep,dq,0.)
    qcko = jnp.where(sep & (K==kt1),qrch,qcko)

    # precipitation efficiency from wind shear
    uo_m = jnp.roll(uo, 1)
    vo_m = jnp.roll(vo, 1)
    shear = jnp.sqrt((uo - uo_m) ** 2 + (vo - vo_m) ** 2)
    vshear = seq_sum(shear, (K > kb) & (K <= ktcon), jnp.asarray(0.0, real_t), _ar(2, km + 1))
    vshear = 1.0e3 * vshear / (zi[ktcon] - zi[kb])
    e1 = L(1.591) - L(0.639) * vshear + L(0.0953) * (vshear * vshear) - L(0.00496) * (vshear * vshear * vshear)
    edt = 1.0 - e1
    edt = jnp.minimum(edt, L(0.9))
    edt = jnp.maximum(edt, 0.0)
    edto = edt
    edtx = edt

    def unit_change(env,cloud,wind=False):
        def body(_,k):
            dv1,dv2,dv3 = env[k],.5*(env[k]+env[k-1]),env[k-1]
            dp,dz = 1000.*dele[k],zi[k]-zi[k-1]
            tem,tem1 = .5*(xlamue[k]+xlamue[k-1]),xlamx
            value = (eta[k]*dv1-eta[k-1]*dv3-tem*eta[k-1]*dv2*dz
                     +tem1*eta[k-1]*.5*(cloud[k]+cloud[k-1])*dz)
            if wind:value = value-pgcon*eta[k-1]*(dv1-dv3)
            value = (0.+value*g/dp)
            return None,jnp.where((k>kb)&(k<ktcon),value,0.)
        _,out = _scan(body,None,_ar(2,km))
        result = zero.at[2:km].set(out)
        top = eta[ktcon-1]*(cloud[ktcon-1]-env[ktcon-1])*g/(1000.*dele[ktcon])
        return result.at[ktcon].set(top)
    dellah = unit_change(heo,hcko)
    dellaq = unit_change(qo,qcko)
    dellau = unit_change(uo,ucko,True)
    dellav = unit_change(vo,vcko,True)
    dellal = dellal.at[ktcon].set(eta[ktcon-1]*qlko_ktcon*g/(1000.*dele[ktcon]))
    ptem = g*sflx*hpbl/t1[1]
    wstar = ptem**L(.33333333)
    density = po[kbcon]*100./(rd*t1[kbcon])
    xmb = jnp.minimum(L(.03)*density*wstar,xmbmax)
    feedback = cnvflg&valid&(K>kb)&(K<=ktcon)
    dellat = (dellah-hvap*dellaq)/cp
    t1n = jnp.where(feedback,t1+dellat*xmb*dt2,t1)
    q1n = jnp.where(feedback,q1+dellaq*xmb*dt2,q1)
    u1n = jnp.where(feedback,u1+dellau*xmb*dt2,u1)
    v1n = jnp.where(feedback,v1+dellav*xmb*dt2,v1)
    qeso_f = L(.01)*fpvs(jnp.where(inK,t1n,300.))
    qeso_f = eps*qeso_f/(pfld+epsm1*qeso_f)
    qeso_f = jnp.where(inK,jnp.maximum(qeso_f,L(1.e-8)),0.)
    rain_mask = valid&(K>kb)&(K<ktcon)
    rntot = seq_sum(pwo*xmb*L(.001)*dt2,rain_mask,jnp.asarray(0.,real_t),_ar(1,km+1),reverse=True)
    evef = edt*L(.3)
    def evaporation(carry,k):
        rain,flag,delqev,delq2,tc,qc = carry
        ink = k<=kmax
        add_rain = cnvflg&ink&(k>kb)&(k<ktcon)
        rain = jnp.where(add_rain,rain+pwo[k]*xmb*L(.001)*dt2,rain)
        enabled = flag&ink&(k<ktcon)
        qcond = evef*(qc[k]-qeso_f[k])/(1.+el2orc*qeso_f[k]/(tc[k]*tc[k]))
        dp = 1000.*dele[k]
        ev = enabled&(rain>0.)&(qcond<0.)
        positive_rain = jnp.where(ev,rain,0.)
        amount = -qcond*(1.-jnp.exp(-L(.32)*jnp.sqrt(dt2*positive_rain)))
        amount = jnp.where(ev,jnp.minimum(amount,rain*1000.*g/dp),0.)
        delq2 = jnp.where(ev,delqev+L(.001)*amount*dp/g,delq2)
        cap = ev&(delq2>rntot)
        amount = jnp.where(cap,1000.*g*(rntot-delqev)/dp,amount)
        flag = flag&~cap
        apply = enabled&(rain>0.)&(amount>0.)
        tem = L(.001)*dp/g
        tem1 = amount*tem
        exhaust = apply&(tem1>rain)
        amount = jnp.where(exhaust,rain/tem,amount)
        rain = jnp.where(apply,jnp.where(exhaust,0.,rain-tem1),rain)
        qc = qc.at[k].set(jnp.where(apply,qc[k]+amount,qc[k]))
        tc = tc.at[k].set(jnp.where(apply,tc[k]-(hvap/cp)*amount,tc[k]))
        delqev = jnp.where(apply,delqev+L(.001)*dp*amount/g,delqev)
        return (rain,flag,delqev,delq2,tc,qc),None
    zero_s = jnp.asarray(0.,real_t)
    (rn,flag,_,_,t1n,q1n),_ = _scan(evaporation,(zero_s,cnvflg,zero_s,zero_s,t1n,q1n),
                                   _ar(1,km+1),reverse=True)
    rn = jnp.where(cnvflg&((rn<0.)|~flag),0.,rn)
    mq = cnvflg&(K>=kbcon)&(K<=ktcon)&(K>=1)&(K<=km1)
    condensate = dellal*xmb*dt2
    icefrac = jnp.maximum(0.,jnp.minimum(1.,(tcr-t1n)*tcrf))
    ql1n = jnp.where(mq&(ncloud>=2),ql1+condensate*icefrac,ql1)
    ql2n = jnp.where(mq&(ncloud>0),jnp.where(ncloud>=2,ql2+condensate*(1.-icefrac),ql2+condensate),ql2)
    kbot = jnp.where(cnvflg,kbcon,initial_kbot)
    ktop = jnp.where(cnvflg,ktcon,initial_ktop)
    return t1n,q1n,ql1n,ql2n,u1n,v1n,rn,kbot,ktop,jnp.where(cnvflg,0,icps)
