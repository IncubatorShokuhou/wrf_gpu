"""WRF NSAS (cu96) native-REAL deep core, unadmitted candidate.

Source module_cu_nsas.F:nsas2d. Distinct ACRIT closure, no KSAS velocity/PBL
closure. The CU_NSAS wrapper also requires nscv2d shallow convection.
kbm/kbmax/kmax are shared WRF tile search limits and must be supplied by the
caller, not independently recomputed in a vmap column.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from gpuwrf.physics.cumulus_scalesas import _ar,_scan
from gpuwrf.physics.cumulus_ksas14 import _constants,_fpvs

def nsas96_deep_column(t1,q1,ql1,ql2,u1,v1,dele,prsl,zo,dot,islimsk,
                       delt,delx,kbm,kbmax,kmax,*,km:int,mode:str,ncloud:int,pgcon:float=.55,dx_factor_nsas:int=0,debug:bool=False):
    c = _constants(mode)
    L = c.lit
    f64 = jnp.float32 if mode == "r4" else jnp.float64
    g = c.grav
    cp, hvap, rd = c.cp, c.hvap, c.rd
    eps, epsm1 = c.eps, c.epsm1
    delta = c.fv
    elocp = hvap / cp
    el2orc = hvap * hvap / (c.rv * cp)
    fact1 = (c.cvap - c.cliq) / c.rv
    fact2 = hvap / c.rv - fact1 * c.t0c
    c0, c1 = L(.002), L(.002)
    cthk, dthk, cinacrmx = L(150.), L(25.), L(-120.)
    tf, tcr = L(233.16), L(263.16)
    tcrf = L(1.) / (tcr-tf)
    fpvs = lambda t: _fpvs(t, mode)

    K = _ar(0, km + 1)
    km1 = km - 1
    valid = (K >= 1) & (K <= km)
    zero = jnp.zeros(km + 1, f64)

    cnvflg = jnp.asarray(True)
    dt2 = delt
    mbdt = L(10.)
    dtmin, dtmax = jnp.maximum(dt2,1200.), jnp.maximum(dt2,3600.)
    edtmaxl = edtmaxs = L(.3)
    clam, aafac = L(.1), L(.1)
    betal = betas = L(.05)
    evfact = evfactl = L(.3)
    cxlamu, xlamde, xlamdd = L(1.e-4), L(1.e-4), L(1.e-4)
    pgcon = jnp.asarray(pgcon,f64)
    cinpcrmx,cinpcrmn = L(180.),L(120.)
    w1l,w2l,w3l,w4l = L(-8.e-3),L(-4.e-2),L(-5.e-3),L(-5.e-4)
    w1s,w2s,w3s,w4s = L(-2.e-4),L(-2.e-3),L(-1.e-3),L(-2.e-5)
    if dx_factor_nsas==1:
        dxf = L(250.)/delx
        w1l,w2l,w3l,w4l = dxf*L(.1)*L(-1.),dxf*L(-1.),dxf*L(-1.),dxf*L(.1)*L(-1.)
        w1s,w2s,w3s,w4s = w1l,w2l,w3l,w4l
    # zi[k] is source zi(k+1): midpoint of adjacent mass levels.
    zi = jnp.where((K>=1)&(K<=km1),.5*(zo+jnp.roll(zo,-1)),0.)
    xlamue = jnp.where((K>=1)&(K<=km1),clam/jnp.where(zi!=0.,zi,1.),0.)
    c0t = jnp.full_like(t1,c0)

    inK = valid & (K <= kmax)
    pfld = jnp.where(inK, prsl * 10.0, 0.0)
    to = jnp.where(inK, t1, 0.0)
    qo = jnp.where(inK, q1, 0.0)
    uo = jnp.where(inK, u1, 0.0)
    vo = jnp.where(inK, v1, 0.0)
    eta = jnp.where(inK, L(1.), L(0.))
    etad = jnp.where(inK, L(1.), L(0.))

    # qeso / qo floor (k <= kmax)
    qeso = L(0.01) * fpvs(to)
    qeso = eps * qeso / (pfld + epsm1 * qeso)
    qeso = jnp.where(inK, jnp.maximum(qeso, L(1.0e-30)), 0.0)
    qo = jnp.where(inK, jnp.maximum(qo, L(1.0e-10)), 0.0)
    tem = g * zo + cp * to
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
    qeso = jnp.where(mI, jnp.maximum(qeso_i, L(1.0e-30)), qeso)
    qo = jnp.where(mI, jnp.maximum(qo, L(1.0e-10)), qo)
    frh = jnp.where(mI, 1.0 - jnp.minimum(qo / jnp.where(mI, qeso, 1.0), 1.0), 0.0)
    heo = jnp.where(mI, 0.5 * g * (zo + zo1) + cp * to + hvap * qo, heo)
    heso = jnp.where(mI, 0.5 * g * (zo + zo1) + cp * to + hvap * qeso, heso)
    uo = jnp.where(mI, 0.5 * (uo + jnp.roll(uo, -1)), uo)
    vo = jnp.where(mI, 0.5 * (vo + jnp.roll(vo, -1)), vo)
    if debug:
        phases = {"ENV_TO":to,"ENV_QO":qo,"ENV_QESO":qeso,"ENV_HEO":heo,"ENV_HESO":heso}

    # kbcon: level of free convection
    heokb = heo[kb]
    c_kbcon = valid & (K <= km1) & (K <= kbmax) & (K > kb) & (heokb > heso)
    kbcon = jnp.where(jnp.any(c_kbcon), jnp.argmax(c_kbcon), kmax).astype(jnp.int32)
    cnvflg = cnvflg & (kbcon != kmax)

    def seq_sum(vals,mask,init,ks,reverse=False):
        def body(acc,k):
            return jnp.where(mask[k],acc+vals[k],acc),None
        return _scan(body,init,ks,reverse=reverse)[0]
    pdot = 10.*dot[kbcon]
    land = islimsk==1
    w3,w4 = jnp.where(land,w3l,w3s),jnp.where(land,w4l,w4s)
    tem = jnp.where(pdot<=w4,(pdot-w4)/(w3-w4),jnp.where(pdot>=-w4,-(pdot+w4)/(w4-w3),0.))
    tem = jnp.minimum(jnp.maximum(tem,-1.),1.)
    cincr = cinpcrmx-(1.-tem)*.5*(cinpcrmx-cinpcrmn)
    cnvflg = cnvflg & ~((-pfld[kbcon]+pfld[kb])>cincr)

    # entrainment / detrainment
    xlamx = xlamue[kbcon]
    above = (K > kbcon) & (K < kmax) & (K >= 2) & (K <= km1)
    xlamud = jnp.where((K >= 1) & (K <= km1) & (K < kmax), xlamx, 0.0)
    qesokbcon = qeso[kbcon]
    temr = qeso / qesokbcon
    fent1 = jnp.where(above, temr * temr, 1.0)
    fent2 = jnp.where(above, temr * temr * temr, 1.0)
    xlamue = jnp.where(above, xlamx, xlamue)
    xlamue = jnp.where((K>=2)&(K<kmax)&(K>=kbcon), xlamue*fent1+cxlamu*frh*fent2,xlamue)

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
    dz = zi[2]-zi[1]
    ptem = .5*(xlamue[1]+xlamue[2])-xlamx
    eta = eta.at[1].set(eta[2]/(1.+ptem*dz))

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

    # Updraft momentum: source runs after the final cloud-top exchange.
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

    # kbcon1: first level >= kbcon with positive buoyancy
    c1m = (K >= 2) & (K <= km1) & (K < kmax) & (K >= kbcon) & (dbyo > 0.0)
    kbcon1 = jnp.where(jnp.any(c1m), jnp.argmax(c1m), kmax).astype(jnp.int32)
    cnvflg = cnvflg & (kbcon1 != kmax)
    cnvflg = cnvflg & ~((pfld[kbcon] - pfld[kbcon1]) > dthk)

    # first-guess cloud top: first level above kbcon1 with negative buoyancy
    ct = (K >= 2) & (K <= km1) & (K < kmax) & (K > kbcon1) & (dbyo < 0.0)
    ktcon = jnp.where(jnp.any(ct), jnp.argmax(ct), 1).astype(jnp.int32)
    cnvflg = cnvflg & ~((pfld[kbcon] - pfld[ktcon]) < cthk)

    # downdraft originating level
    hmin0 = heo[kbcon1]
    cm_ = (K >= 2) & (K <= km1) & (K <= kbmax) & (K > kbcon1)
    hvals = jnp.where(cm_, heo, jnp.inf)
    kmin = jnp.argmin(hvals)
    lmin = jnp.where(hvals[kmin] < hmin0, kmin + 1, kbmax)
    jmin = jnp.minimum(lmin, ktcon - 1)
    jmin = jnp.maximum(jmin, kbcon1 + 1).astype(jnp.int32)
    cnvflg = cnvflg & ~(jmin >= ktcon)

    xmbmax = (1000.0 * dele[kbcon]) / (g * dt2)

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
        dq = eta[k]*qc-eta[k]*qrch
        cond = (k >= kbcon) & (dq > 0.0)
        etah = 0.5 * (eta[k] + eta[k - 1])
        dp = 1000.0 * dele[k]
        upper = (ncloud > 0) & (k > jmin)
        ptem = c0t[k] + c1
        qlk = jnp.where(upper, dq / (eta[k] + etah * ptem * dz), dq / (eta[k] + etah * c0t[k] * dz))
        dlal = jnp.where(cond & upper, etah * c1 * dz * qlk * g / dp, 0.0)
        buo = jnp.where(cond, -dz * g * qlk, 0.0)
        qc = jnp.where(cond, qlk + qrch, qc)
        pw = jnp.where(cond, etah * c0t[k] * dz * qlk, 0.0)
        drag = jnp.asarray(0.,f64)
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

    pwavo = seq_sum(pwo, (K > kb) & (K < ktcon), jnp.asarray(0.0, f64), _ar(2, km))
    # SASCNVN first sums condensate loading with dz, then environmental work with dz1.
    dzi = zi - jnp.roll(zi, 1)
    aa1 = seq_sum(buo, (K > kb) & (K < ktcon), jnp.asarray(0.0, f64), _ar(2, km))
    def work_up(acc, k):
        m = (k >= kbcon) & (k < ktcon)
        dz1 = zo[k + 1] - zo[k]
        gamma = el2orc * qeso[k] / (to[k] * to[k])
        rfact = 1.0 + delta * cp * gamma * to[k] / hvap
        a = acc + dz1 * (g / (cp * to[k])) * dbyo[k] / (1.0 + gamma) * rfact
        a = a + dz1 * g * delta * jnp.maximum(0.0, qeso[k] - qo[k])
        return jnp.where(m, a, acc), None
    aa1, _ = _scan(work_up, aa1, _ar(2, km))
    if debug:
        phases.update(AA1_UP=aa1,ETA=eta,XLAMB=xlamue,HCKO=hcko,QCKO=qcko,PWO=pwo)
    cnvflg = cnvflg & ~(aa1 <= 0.0)

    # overshooting: level where aafac*aa1 + work above ktcon becomes negative
    def aa2_acc(carry, k):
        aa2, flg, ktc1 = carry
        m = flg & (k >= ktcon) & (k < kmax)
        dz1 = zo[k + 1] - zo[k]
        gamma = el2orc * qeso[k] / (to[k] * to[k])
        rfact = 1.0 + delta * cp * gamma * to[k] / hvap
        a = aa2 + dz1 * (g / (cp * to[k])) * dbyo[k] / (1.0 + gamma) * rfact
        aa2 = jnp.where(m, a, aa2)
        stop = m & (a < 0.0)
        return (aa2, flg & ~stop, jnp.where(stop, k, ktc1)), None

    (_, _, ktcon1), _ = _scan(aa2_acc, (aafac * aa1, cnvflg, kmax-1), _ar(2, km))

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
        dq = eta[k]*qc-eta[k]*qrch
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
    vshear = seq_sum(shear, (K > kb) & (K <= ktcon), jnp.asarray(0.0, f64), _ar(2, km + 1))
    vshear = 1.0e3 * vshear / (zi[ktcon] - zi[kb])
    e1 = L(1.591) - L(0.639) * vshear + L(0.0953) * (vshear * vshear) - L(0.00496) * (vshear * vshear * vshear)
    edt = 1.0 - e1
    edt = jnp.minimum(edt, L(0.9))
    edt = jnp.maximum(edt, 0.0)
    edto = edt
    edtx = edt

    # detrainment rate between 1 and kbcon
    sumx = seq_sum(jnp.roll(zi, -1) - zi, (K >= 1) & (K < kbcon), jnp.asarray(0.0, f64), _ar(1, km))
    beta = jnp.where(islimsk == 1, betal, betas)
    fk = kbcon.astype(f64)
    dzk = (sumx + zi[1]) / fk
    temk = 1.0 / fk
    if mode == "r4":
        temk = temk.astype(jnp.float32).astype(f64)  # 1./float(kbcon): single precision
    xlamd = (1.0 - beta ** temk) / dzk

    # downdraft mass flux (downward from etad(jmin)=1)
    def etad_dn(carry, k):
        e_next = carry
        m = k <= kmax - 1
        dz = zi[k + 1] - zi[k]
        b1 = (k < jmin) & (k >= kbcon)
        b2 = k < kbcon
        e1_ = e_next * (1.0 - (xlamdd - xlamde) * dz)
        e2_ = e_next * (1.0 - (xlamd + xlamdd - xlamde) * dz)
        val = jnp.where(m & b1, e1_, jnp.where(m & ~b1 & b2, e2_, etad[k]))
        return val, val

    _, etad_lo = _scan(etad_dn, etad[km], _ar(1, km), reverse=True)
    etad = etad.at[1:km].set(etad_lo)

    # downdraft moist static energy
    hcdo0 = jnp.where(K == jmin, heo[jmin], 0.0)

    def _dd_tems(k):
        dz = zi[k + 1] - zi[k]
        tem = xlamde * dz
        tem1 = jnp.where(k >= kbcon, 0.5 * xlamdd * dz, 0.5 * (xlamd + xlamdd) * dz)
        return dz, tem, tem1

    def hcdo_dn(carry, k, heo=heo, heso=heso, hcdo0=hcdo0):
        h_next = carry
        m = k < jmin
        dz, tem, tem1 = _dd_tems(k)
        factor = 1.0 + tem - tem1
        h = ((1.0 - tem1) * h_next + tem * 0.5 * (heo[k] + heo[k + 1])) / factor
        h = jnp.where(m, h, hcdo0[k])
        return h, (h, h - heso[k])

    _, (hcdo_lo, dbd_lo) = _scan(hcdo_dn, hcdo0[km], _ar(1, km), reverse=True)
    hcdo = hcdo0.at[1:km].set(hcdo_lo)
    dbyo = dbyo.at[1:km].set(jnp.where(K[1:km] < jmin, dbd_lo, dbyo[1:km]))

    # downdraft moisture: qrcdo elementwise, qcdo from qrcdo(k+1)
    gam = el2orc * qeso / (to * to + (to == 0.0))
    qrcdo = jnp.where((K < jmin) & valid, qeso + (1.0 / hvap) * (gam / (1.0 + gam)) * dbyo, 0.0)
    qrcdo = jnp.where(K==jmin,qeso[jmin],qrcdo)
    dzd = jnp.roll(zi, -1) - zi
    temd = xlamde * dzd
    tem1d = jnp.where(K >= kbcon, 0.5 * xlamdd * dzd, 0.5 * (xlamd + xlamdd) * dzd)
    factord = 1.0 + temd - tem1d
    q_next = jnp.where(K+1==jmin,qo[jmin],jnp.roll(qrcdo,-1))
    qcdo_v = ((1.-tem1d)*q_next+temd*.5*(qo+jnp.roll(qo,-1)))/factord
    mdd = (K >= 1) & (K < jmin)
    qcdo = jnp.where(mdd, qcdo_v, jnp.where(K == jmin, qo[jmin], 0.0))
    pwdo = jnp.where(mdd,jnp.roll(etad,-1)*qcdo-jnp.roll(etad,-1)*qrcdo,0.)
    qcdo = jnp.where(mdd,qrcdo,qcdo)
    pwevo = seq_sum(pwdo, mdd, jnp.asarray(0.0, f64), _ar(1, km), reverse=True)

    edtmax = jnp.where(islimsk == 0, edtmaxs, edtmaxl)
    edto = jnp.where(pwevo < 0.0, jnp.minimum(-edto * pwavo / jnp.where(pwevo < 0.0, pwevo, -1.0), edtmax), 0.0)

    # downdraft cloud work function (continues aa1, descending k)
    def _dd_work(edt_, hcdo_, heso_, qeso_, to_, qo_):
        def body(acc, k):
            m = k < jmin
            gamma = el2orc * qeso_[k] / (to_[k] * to_[k])
            dhh, dt, dg, dh = hcdo_[k], to_[k], gamma, heso_[k]
            dz = -1.0 * (zo[k + 1] - zo[k])
            a = acc + edt_ * dz * (g / (cp * dt)) * ((dhh - dh) / (1.0 + dg)) * (1.0 + delta * cp * dg * dt / hvap)
            a = a + edt_ * dz * g * delta * jnp.maximum(0.0, qeso_[k] - qo_[k])
            return jnp.where(m, a, acc), None
        return body

    aa1, _ = _scan(_dd_work(edto, hcdo, heso, qeso, to, qo), aa1, _ar(1, km), reverse=True)
    if debug:
        phases.update(AA1_DD=aa1,EDTO=edto,PWEVO=pwevo)
    cnvflg = cnvflg & ~(aa1 <= 0.0)

    # Downdraft momentum; same descending source recurrence as hcdo.
    def momentum_dn(prev,k):
        un,vn = prev
        _,tem,tem1 = _dd_tems(k)
        factor = 1.+tem-tem1
        ptem,ptem1 = .5*tem-pgcon,.5*tem+pgcon
        u = ((1.-tem1)*un+ptem*uo[k+1]+ptem1*uo[k])/factor
        v = ((1.-tem1)*vn+ptem*vo[k+1]+ptem1*vo[k])/factor
        u,v = jnp.where(k<jmin,u,jnp.where(k==jmin,uo[jmin],0.)),jnp.where(k<jmin,v,jnp.where(k==jmin,vo[jmin],0.))
        return (u,v),(u,v)
    ucdo,vcdo = zero,zero
    ucdo,vcdo = ucdo.at[jmin].set(uo[jmin]),vcdo.at[jmin].set(vo[jmin])
    _,(uu,vv) = _scan(momentum_dn,(ucdo[km],vcdo[km]),_ar(1,km),reverse=True)
    ucdo,vcdo = ucdo.at[1:km].set(uu),vcdo.at[1:km].set(vv)
    def unit_change(env,up,down,wind=False):
        def body(_,k):
            aup,adw = jnp.where(k<=kb,L(0.),L(1.)),jnp.where(k>jmin,L(0.),L(1.))
            dv1,dv2,dv3 = env[k],.5*(env[k]+env[k-1]),env[k-1]
            dp,dz = 1000.*dele[k],zi[k]-zi[k-1]
            tem,tem1 = .5*(xlamue[k]+xlamue[k-1]),xlamx
            ptem,ptem1 = xlamde,jnp.where(k<=kbcon,xlamd+xlamdd,xlamdd)
            # Keep the literal source operand order, including adw*edto.
            v = ((aup*eta[k]-adw*edto*etad[k])*dv1
                 -(aup*eta[k-1]-adw*edto*etad[k-1])*dv3
                 -(aup*tem*eta[k-1]+adw*edto*ptem*etad[k])*dv2*dz
                 +aup*tem1*eta[k-1]*.5*(up[k]+up[k-1])*dz
                 +adw*edto*ptem1*etad[k]*.5*(down[k]+down[k-1])*dz)
            if wind:
                v = v-pgcon*(aup*eta[k-1]-adw*edto*etad[k])*(dv1-dv3)
            v = (0.+v*g/dp)
            return None,jnp.where(k<ktcon,v,0.)
        _,hi = _scan(body,None,_ar(2,km))
        low = edto*etad[1]*(down[1]-env[1])*g/(1000.*dele[1])
        arr = zero.at[1].set(low).at[2:km].set(hi)
        top = eta[ktcon-1]*(up[ktcon-1]-env[ktcon-1])*g/(1000.*dele[ktcon])
        arr = arr.at[ktcon].set(top)
        return jnp.where(inK,arr,0.)
    dellah = unit_change(heo,hcko,hcdo)
    dellaq = unit_change(qo,qcko,qrcdo)
    dellau = unit_change(uo,ucko,ucdo,True)
    dellav = unit_change(vo,vcko,vcdo,True)
    dellal = dellal.at[ktcon].set(eta[ktcon-1]*qlko_ktcon*g/(1000.*dele[ktcon]))

    # ---- quasi-equilibrium closure (grid size >= dxcrtas) -------------------------------------
    inK = valid & (K <= kmax)
    qo_a = jnp.where(K > ktcon, q1, jnp.maximum(dellaq * mbdt + q1, L(1.0e-10)))
    dellat = (dellah - hvap * dellaq) / cp
    to_a = jnp.where(K > ktcon, t1, dellat * mbdt + t1)
    qo_a = jnp.where(inK, qo_a, 0.0)
    to_a = jnp.where(inK, to_a, 0.0)
    qeso_a = L(0.01) * fpvs(jnp.where(inK, to_a, 300.0))
    qeso_a = eps * qeso_a / (pfld + epsm1 * qeso_a)
    qeso_a = jnp.where(inK, jnp.maximum(qeso_a, L(1.0e-30)), 0.0)
    to_a, qo_a, _po_a, mIa = _interface(to_a, qo_a, qeso_a)
    qeso_ai = L(0.01) * fpvs(jnp.where(mIa, to_a, 300.0))
    qeso_ai = eps * qeso_ai / (po + epsm1 * qeso_ai)
    qeso_a = jnp.where(mIa, jnp.maximum(qeso_ai, L(1.0e-30)), qeso_a)
    qo_a = jnp.where(mIa, jnp.maximum(qo_a, L(1.0e-10)), qo_a)
    heo_a = jnp.where(mIa, 0.5 * g * (zo + zo1) + cp * to_a + hvap * qo_a, heo)
    heso_a = jnp.where(mIa, 0.5 * g * (zo + zo1) + cp * to_a + hvap * qeso_a, heso)
    kx_ = kmax
    heo_a = heo_a.at[kx_].set(g * zo[kx_] + cp * to_a[kx_] + hvap * qo_a[kx_])
    heso_a = heso_a.at[kx_].set(g * zo[kx_] + cp * to_a[kx_] + hvap * qeso_a[kx_])

    hcko_a0 = jnp.where(K == kb, heo_a[kb], hcko)

    def hcko_a_up(carry, k):
        h_prev = carry
        m = (k > kb) & (k <= ktcon)
        dz = zi[k] - zi[k - 1]
        tem = 0.5 * (xlamue[k] + xlamue[k - 1]) * dz
        tem1 = 0.5*xlamx * dz
        factor = 1.0 + tem - tem1
        h = ((1.0 - tem1) * h_prev + tem * 0.5 * (heo_a[k] + heo_a[k - 1])) / factor
        h = jnp.where(m, h, hcko_a0[k])
        return h, h

    _, hck_hi = _scan(hcko_a_up, hcko_a0[1], _ar(2, km))
    hcko_a = hcko_a0.at[2:km].set(hck_hi)

    qcko_a0 = jnp.where(K == kb, qo_a[kb], qcko)

    def xaa0_up(carry, k):
        q_prev, xaa0, xpwav = carry
        m = (k > kb) & (k < ktcon)
        dz = zi[k] - zi[k - 1]
        gamma = el2orc * qeso_a[k] / (to_a[k] * to_a[k])
        xdby = hcko_a[k] - heso_a[k]
        xqrch = qeso_a[k] + gamma * xdby / (hvap * (1.0 + gamma))
        tem = 0.5 * (xlamue[k] + xlamue[k - 1]) * dz
        tem1 = 0.5*xlamx * dz
        factor = 1.0 + tem - tem1
        qc = ((1.0 - tem1) * q_prev + tem * 0.5 * (qo_a[k] + qo_a[k - 1])) / factor
        dq = eta[k]*qc-eta[k]*xqrch
        cond = m & (k >= kbcon) & (dq > 0.0)
        etah = 0.5 * (eta[k] + eta[k - 1])
        upper = (ncloud > 0) & (k > jmin)
        ct = c0
        ptem = ct+c1
        qlk = jnp.where(upper,dq/(eta[k]+etah*ptem*dz),dq/(eta[k]+etah*ct*dz))
        xaa0 = jnp.where(cond & (k < ktcon1), xaa0 - dz * g * qlk, xaa0)
        qc = jnp.where(cond, qlk + xqrch, qc)
        xpwav = jnp.where(cond, xpwav + etah * ct * dz * qlk, xpwav)
        qc = jnp.where(m, qc, qcko_a0[k])
        m2 = (k >= kbcon) & (k < ktcon1)
        dz1 = zo[k + 1] - zo[k]
        rfact = 1.0 + delta * cp * gamma * to_a[k] / hvap
        a = xaa0 + dz1 * (g / (cp * to_a[k])) * xdby / (1.0 + gamma) * rfact
        a = a + dz1 * g * delta * jnp.maximum(0.0, qeso_a[k] - qo_a[k])
        xaa0 = jnp.where(m2, a, xaa0)
        return (qc, xaa0, xpwav), None

    (_, xaa0, xpwav), _ = _scan(xaa0_up, (qcko_a0[1], jnp.asarray(0.0, f64), jnp.asarray(0.0, f64)),
                                _ar(2, km))

    hcdo_a0 = jnp.where(K == jmin, heo_a[jmin], hcdo)
    _, (hcdo_alo, _unused) = _scan(
        lambda c_, k: hcdo_dn(c_, k, heo=heo_a, heso=heso_a, hcdo0=hcdo_a0),
        hcdo_a0[km], _ar(1, km), reverse=True)
    hcdo_a = hcdo_a0.at[1:km].set(hcdo_alo)
    gam_a = el2orc * qeso_a / (to_a * to_a + (to_a == 0.0))
    dh_a = hcdo_a - heso_a
    qrcd = jnp.where((K < jmin) & valid, qeso_a + (1.0 / hvap) * (gam_a / (1.0 + gam_a)) * dh_a, 0.0)
    qrcd = jnp.where(K==jmin,qeso_a[jmin],qrcd)
    qa_next = jnp.where(K+1==jmin,qo_a[jmin],jnp.roll(qrcd,-1))
    qcdo_a = ((1.-tem1d)*qa_next+temd*.5*(qo_a+jnp.roll(qo_a,-1)))/factord
    xpwd = jnp.where(mdd,jnp.roll(etad,-1)*(qcdo_a-qrcd),0.)
    xpwev = seq_sum(xpwd, mdd, jnp.asarray(0.0, f64), _ar(1, km), reverse=True)
    edtx = jnp.where(xpwev >= 0.0, 0.0,
                     jnp.minimum(-edtx * xpwav / jnp.where(xpwev >= 0.0, -1.0, xpwev), edtmax))
    xaa0, _ = _scan(_dd_work(edtx, hcdo_a, heso_a, qeso_a, to_a, qo_a), xaa0, _ar(1, km), reverse=True)

    pcrit = jnp.asarray([850.,800.,750.,700.,650.,600.,550.,500.,450.,400.,350.,300.,250.,200.,150.], f64)
    acritt = jnp.asarray([L(x) for x in (.0633,.0445,.0553,.0664,.075,.1082,.1521,.2216,.3151,.3677,.41,.5255,.7663,1.1686,1.6851)], f64)
    acrit = acritt * (975.0 - pcrit)
    pc = pfld[ktcon]
    k = jnp.clip(((850.0 - pc) / 50.0).astype(jnp.int32) + 2, 2, 15) - 1
    acrt = jnp.where(pc < pcrit[14], acrit[14] * (975.0 - pc) / (975.0 - pcrit[14]),
                    jnp.where(pc > pcrit[0], acrit[0],
                              acrit[k] + (acrit[k-1] - acrit[k]) * (pc - pcrit[k]) / (pcrit[k-1] - pcrit[k])))
    acrtfct = jnp.where(pdot <= w4, (pdot-w4)/(w3-w4),
                       jnp.where(pdot >= -w4, -(pdot+w4)/(w4-w3), 0.0))
    acrtfct = 1.0 - jnp.minimum(jnp.maximum(acrtfct, -1.0), 1.0)
    w1, w2 = jnp.where(land, w1l, w1s), jnp.where(land, w2l, w2s)
    dtconv = dt2 + jnp.maximum(1800.0 - dt2, 0.0) * (pdot-w2)/(w1-w2)
    dtconv = jnp.minimum(jnp.maximum(dtconv, dtmin), dtmax)
    fld = (aa1 - acrt * acrtfct) / dtconv
    cnvflg = cnvflg & ~(fld <= 0.0)
    xk = (xaa0 - aa1) / mbdt
    cnvflg = cnvflg & ~(xk >= 0.0)
    xmb = jnp.minimum(-fld / xk, xmbmax)
    if debug:
        phases.update(XAA0=xaa0,ACRT=acrt,ACRTFCT=acrtfct,DTCONV=dtconv,F=fld,XK=xk,XMB=xmb)

    # feedback
    inK = valid & (K <= kmax)
    fb = cnvflg & inK & (K <= ktcon)
    dellat = (dellah - hvap * dellaq) / cp
    t1n = jnp.where(fb, t1 + dellat * xmb * dt2, t1)
    q1n = jnp.where(fb,q1+dellaq*xmb*dt2,q1)
    u1n = jnp.where(fb,u1+dellau*xmb*dt2,u1)
    v1n = jnp.where(fb,v1+dellav*xmb*dt2,v1)
    qeso_f = L(0.01) * fpvs(jnp.where(inK, t1n, 300.0))
    qeso_f = eps * qeso_f / (pfld + epsm1 * qeso_f)
    qeso_f = jnp.where(inK, jnp.maximum(qeso_f, L(1.0e-30)), 0.0)

    aupv = jnp.where(K <= kb, L(0.), L(1.))
    adwv = jnp.where(K >= jmin, L(0.), L(1.))
    rain_v = aupv * pwo + adwv * edto * pwdo
    mr = inK & (K < ktcon)
    rntot = seq_sum(rain_v * xmb * L(0.001) * dt2, mr, jnp.asarray(0.0, f64), _ar(1, km + 1), reverse=True)

    evef = jnp.where(islimsk == 1, edt * evfactl, edt * evfact)

    def evap(carry, k):
        rn, flg, delqev, delq2, t1c, q1c = carry
        ink = k <= kmax
        mrain = ink & cnvflg & (k < ktcon)
        rain = aupv[k] * pwo[k] + adwv[k] * edto * pwdo[k]
        rn = jnp.where(mrain, rn + rain * xmb * L(0.001) * dt2, rn)
        me = ink & flg & (k < ktcon)
        qcond = evef * (q1c[k] - qeso_f[k]) / (1.0 + el2orc * qeso_f[k] / (t1c[k] * t1c[k]))
        dp = 1000.0 * dele[k]
        c_a = me & (rn > 0.0) & (qcond < 0.0)
        qevap = -qcond * (1.0 - jnp.exp(-L(0.32) * jnp.sqrt(dt2 * rn)))
        qevap = jnp.minimum(qevap, rn * 1000.0 * g / dp)
        qevap = jnp.where(c_a, qevap, 0.0)
        delq2 = jnp.where(c_a, delqev + L(0.001) * qevap * dp / g, delq2)
        c_b = c_a & (delq2 > rntot)
        qevap = jnp.where(c_b, 1000.0 * g * (rntot - delqev) / dp, qevap)
        flg = flg & ~c_b
        c_c = me & (rn > 0.0) & (qevap > 0.0)
        q1c = q1c.at[k].set(jnp.where(c_c, q1c[k] + qevap, q1c[k]))
        t1c = t1c.at[k].set(jnp.where(c_c, t1c[k] - elocp * qevap, t1c[k]))
        rn = jnp.where(c_c, rn - L(0.001) * qevap * dp / g, rn)
        delqev = jnp.where(c_c, delqev + L(0.001) * dp * qevap / g, delqev)
        return (rn, flg, delqev, delq2, t1c, q1c), None

    zero_s = jnp.asarray(0.0, f64)
    (rn, flg, _, _, t1n, q1n), _ = _scan(
        evap, (zero_s, cnvflg, zero_s, zero_s, t1n, q1n), _ar(1, km + 1), reverse=True)

    rn = jnp.where(cnvflg & (rn < 0.0) & ~flg, 0.0, rn)
    rn = jnp.where(cnvflg & (rn <= 0.0), 0.0, rn)
    active = cnvflg & (rn > 0.0)
    ktop = jnp.where(active, ktcon, 0)
    kbot = jnp.where(active, kbcon, km + 1)

    # detrained cloud condensate -> ql (ice fraction by temperature)
    mq = active & (K >= kbcon) & (K <= ktcon) & valid
    temq = dellal * xmb * dt2
    tem1q = jnp.maximum(0.0, jnp.minimum(1.0, (tcr - t1n) * tcrf))
    ql1n = jnp.where(mq & (ncloud>=2),ql1+temq*tem1q,ql1)
    ql2n = jnp.where(mq & (ncloud>0),jnp.where(ncloud>=2,ql2+temq*(1.-tem1q),ql2+temq),ql2)

    # no rain: restore the environment (to/qo were restored to t1/q1)
    restore = cnvflg & (rn <= 0.0) & inK
    t1n = jnp.where(restore, t1, t1n)
    q1n = jnp.where(restore, q1, q1n)
    u1n,v1n = jnp.where(restore,u1,u1n),jnp.where(restore,v1,v1n)
    result = t1n,q1n,ql1n,ql2n,u1n,v1n,rn,kbot,ktop,jnp.where(active,1,0)
    return (result,phases) if debug else result



def cu_nsas96_columns(t3d,qv3d,qc3d,qi3d,p3d,pi3d,rho3d,dz8w,u3d,v3d,w,p8w,
                      xland,hpbl,hfx,qfx,*,dt,stepcu,dx,mp_physics=8,pgcon=.55,
                      dx_factor_nsas=0,row_width=None,literals="r4"):
    """Original CU_NSAS deep + nscv2d shallow; unadmitted until full oracle gate.

    Each row_width group represents one original CU_NSAS i-row. nsas2d shares
    pressure search limits over that row; nscv2d computes limits per column.
    Arrays (ncol,nlev), P8W/W (ncol,nlev+1); other fields (ncol,).
    """
    dtype = jnp.float32 if literals=="r4" else jnp.float64
    c = _constants(literals)
    T,QV,QC,QI,P,PI,RHO,DZ,U,V,W = [jnp.asarray(a,dtype) for a in
        (t3d,qv3d,qc3d,qi3d,p3d,pi3d,rho3d,dz8w,u3d,v3d,w)]
    ncol,kx = T.shape
    dt,delx = jnp.asarray(dt,dtype),jnp.asarray(dx,dtype)
    delt = dt*int(stepcu)
    rdelt = 1./delt
    prsl = P*jnp.asarray(.001,dtype)
    dele = prsl*c.grav/c.rd*DZ/T
    dot = jnp.asarray(-5.e-4,dtype)*c.grav*RHO*(W[:,:-1]+W[:,1:])
    def zi_body(old,dz):
        new = old+dz
        return new,new
    _,zi = jax.lax.scan(zi_body,jnp.zeros(ncol,dtype),DZ.T)
    zi = jnp.concatenate((jnp.zeros((ncol,1),dtype),zi.T),axis=1)
    zl = .5*(zi[:,:-1]+zi[:,1:])
    pad = lambda a:jnp.concatenate((jnp.zeros((ncol,1),dtype),a),axis=1)
    from gpuwrf.physics.cumulus_nsas96_shallow import nsas96_shallow_column
    ncloud = 0 if mp_physics==0 else (1 if mp_physics in (1,3) else 2)
    row_width = ncol if row_width is None else int(row_width)
    if row_width<1 or ncol%row_width:
        raise ValueError("NSAS row_width must divide the column count")
    PS = jnp.asarray(p8w,dtype)[:,0]*jnp.asarray(.001,dtype)
    indices = jnp.arange(1,kx+1,dtype=jnp.int32)
    def shared_limit(fraction):
        mask = (prsl>PS[:,None]*jnp.asarray(fraction,dtype)).reshape(-1,row_width,kx)
        last = jnp.max(jnp.where(jnp.any(mask,axis=1),indices,0),axis=1)
        limit = jnp.where(last>0,last+1,kx)
        return jnp.repeat(limit,row_width)
    kbm = jnp.minimum(shared_limit(.70),kx)
    kbmax = shared_limit(.45)
    kmax = jnp.minimum(shared_limit(.04),kx)
    col = jax.vmap(lambda *a:nsas96_deep_column(*a[:11],delt,delx,*a[11:],
        km=kx,mode=literals,ncloud=ncloud,pgcon=pgcon,dx_factor_nsas=dx_factor_nsas))
    base = [pad(a) for a in (T,QV,QI,QC,U,V,dele,prsl,zl,dot)]
    XL = jnp.asarray(xland,dtype)
    tn,qn,qin,qcn,un,vn,rn,kbot,ktop,icps = col(*base,XL,kbm,kbmax,kmax)
    shallow = jax.vmap(lambda *a:nsas96_shallow_column(*a[:16],delt,*a[16:],
        km=kx,mode=literals,ncloud=ncloud,pgcon=pgcon))
    tn,qn,qin,qcn,un,vn,rn2,kbot,ktop,_ = shallow(tn,qn,qin,qcn,un,vn,pad(dele),pad(prsl),PS,
        pad(PI),pad(zl),pad(dot),XL,jnp.asarray(hpbl,dtype),jnp.asarray(hfx,dtype),
        jnp.asarray(qfx,dtype),kbot,ktop,icps)
    tn,qn,qin,qcn,un,vn = [a[:,1:] for a in (tn,qn,qin,qcn,un,vn)]
    return {"RTHCUTEN":(tn-T)/PI*rdelt,"RQVCUTEN":(qn-QV)*rdelt,
            "RQCCUTEN":(qcn-QC)*rdelt,"RQICUTEN":(qin-QI)*rdelt,
            "RUCUTEN":(un-U)*rdelt,"RVCUTEN":(vn-V)*rdelt,
            "RAINCV":rn*1000./int(stepcu)+rn2*1000./int(stepcu),
            "PRATEC":rn*1000./delt+rn2*1000./delt,
            "HBOT":kbot.astype(dtype),"HTOP":ktop.astype(dtype)}
