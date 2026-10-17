"""Source-specific old Kain-Fritsch KFPARA components, WRF cu=99.

This is a distinct native-REAL port, not an alias of the KF-eta endpoint.
Components are staged against pristine checkpoints before operational admission.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from gpuwrf.physics.kf99_thermo import constants,tpmix,dtfrznew,envirtht,condload,tpdd


class Environment(NamedTuple):
    t:object;q:object;p:object;dz:object;rho:object;w:object;u:object;v:object
    dp:object;tv:object;z:object;dza:object;qes:object;ml:object;l5:object;l4:object;llfc:object


def environment(t,qv,p,dz,rho,w,u,v,*,mode='r4'):
    c=constants(mode);L=c.L
    dtype=jnp.float32 if mode=='r4' else jnp.float64
    pad=lambda a:jnp.pad(jnp.asarray(a,dtype),(1,0))
    t,qv,p,dz,rho,w,u,v=map(pad,(t,qv,p,dz,rho,w,u,v))
    kx=t.shape[0]-1;k=jnp.arange(kx+1,dtype=jnp.int32);valid=k>=1
    es=c.aliq*jnp.exp((c.bliq*t-c.cliq)/(t-c.dliq))
    qs=c.ep2*es/(p-es)
    q=jnp.maximum(L(.000001),jnp.minimum(qs,qv))
    tv=t*(L(1.)+L(.608)*q)
    dp=rho*c.g*dz
    def heights(old,k):
        z=jnp.where(k==1,L(.5)*dz[1],old+L(.5)*(dz[k]+dz[k-1]))
        return z,z
    _,z=jax.lax.scan(heights,jnp.asarray(0.,dtype),jnp.arange(1,kx+1,dtype=jnp.int32))
    z=jnp.concatenate((jnp.zeros(1,dtype),z))
    dza=jnp.concatenate((jnp.zeros(1,dtype),z[2:]-z[1:-1],jnp.zeros(1,dtype)))
    last=lambda mask:jnp.max(jnp.where(mask&valid,k,0))
    return Environment(t,q,p,dz,rho,w,u,v,dp,tv,z,dza,qs,
                       last(t>L(273.16)),last(p>=L(50000.)),last(p>=L(40000.)),last(p>=p[1]-L(30000.)))


def ordered_sum(a,start,stop):
    def body(old,k):return jnp.where((k>=start)&(k<=stop),old+a[k],old),None
    return jax.lax.scan(body,jnp.zeros((),a.dtype),jnp.arange(1,len(a),dtype=jnp.int32))[0]


def prof5(eq,*,mode='r4'):
    """Original PROF5; source is identical to KF-eta, native literal folding retained."""
    L=constants(mode).L
    sqrt2p,a1,a2,a3,p,sigma,fe=map(L,(2.506628,.4361836,-.1201676,.9372980,.33267,.166666667,.202765151))
    y=L(6.)*eq-L(3.);ey=jnp.exp(y*y/L(-2.));e45=jnp.exp(L(-4.5))
    t2=L(1.)/(L(1.)+p*jnp.abs(y));t1=L(.500498)
    c1=a1*t1+a2*t1*t1+a3*t1*t1*t1
    c2=a1*t2+a2*t2*t2+a3*t2*t2*t2
    ee=jnp.where(y>=L(0.),sigma*(L(.5)*(sqrt2p-e45*c1-ey*c2)+sigma*(e45-ey))-e45*eq*eq/L(2.),
        sigma*(L(.5)*(ey*c2-e45*c1)+sigma*(e45-ey))-e45*eq*eq/L(2.))
    ud=jnp.where(y>=L(0.),sigma*(L(.5)*(ey*c2-e45*c1)+sigma*(e45-ey))-e45*(L(.5)+eq*eq/L(2.)-eq),
        sigma*(L(.5)*(sqrt2p-e45*c1-ey*c2)+sigma*(e45-ey))-e45*(L(.5)+eq*eq/L(2.)-eq))
    return ee/fe,ud/fe


def trigger(env,lc,dx,*,mode='r4'):
    """KFPARA25-45: 60-hPa pressure-weighted dry-theta source mixture and LCL."""
    c=constants(mode);L=c.L
    kx=len(env.t)-1;k=jnp.arange(kx+1,dtype=jnp.int32)
    def depth(old,k):
        dp,n,found,kpbl=old
        use=(k>=lc)&~found
        dp=jnp.where(use,dp+env.dp[k],dp);n=jnp.where(use,n+1,n)
        found1=found|(use&(dp>L(6000.)))
        kpbl=jnp.where(~found&found1,k,kpbl)
        return (dp,n,found1,kpbl),None
    (depth,_,found,kpbl),_=jax.lax.scan(depth,(jnp.zeros((),env.t.dtype),jnp.int32(0),jnp.asarray(False),jnp.int32(kx)),jnp.arange(1,kx+1,dtype=jnp.int32))
    rocpq=L(.2854)*(L(1.)-L(.28)*env.q)
    thmix=ordered_sum(env.dp*env.t*(L(1.e5)/env.p)**rocpq,lc,kpbl)/depth
    qmix=ordered_sum(env.dp*env.q,lc,kpbl)/depth
    zmix=ordered_sum(env.dp*env.z,lc,kpbl)/depth
    pmix=ordered_sum(env.dp*env.p,lc,kpbl)/depth
    rocpq=L(.2854)*(L(1.)-L(.28)*qmix)
    tmix=thmix*(pmix/L(1.e5))**rocpq
    emix=qmix*pmix/(c.ep2+qmix)
    tlog=jnp.log(emix/c.aliq)
    tdpt=(c.cliq-c.dliq*tlog)/(c.bliq-tlog)
    tlcl=tdpt-(L(.212)+L(1.571e-3)*(tdpt-L(273.16))-L(4.36e-4)*(tmix-L(273.16)))*(tmix-tdpt)
    tlcl=jnp.minimum(tlcl,tmix)
    tvlcl=tlcl*(L(1.)+L(.608)*qmix)
    plcl=L(1.e5)*(tlcl/thmix)**(L(1.)/rocpq)
    candidates=(k>=lc)&(k>=1)&(plcl>=env.p)
    klcl=jnp.where(jnp.any(candidates),jnp.argmax(candidates),kx).astype(jnp.int32)
    km=klcl-1
    dlp=jnp.log(plcl/env.p[km])/jnp.log(env.p[klcl]/env.p[km])
    tenv=env.t[km]+(env.t[klcl]-env.t[km])*dlp
    qenv=env.q[km]+(env.q[klcl]-env.q[km])*dlp
    tven=tenv*(L(1.)+L(.608)*qenv)
    zlcl=env.z[km]+(env.z[klcl]-env.z[km])*dlp
    wklcl=L(.02)*zlcl/L(2500.)
    wkl=(env.w[km]+(env.w[klcl]-env.w[km])*dlp)*dx/L(25000.)-wklcl
    wabs=jnp.abs(wkl)+L(1.e-10);sign=wkl/wabs
    dtlcl=L(4.64)*sign*wabs**L(.33)
    gdt=c.g*dtlcl*(zlcl-env.z[lc])/(env.tv[lc]+tven)
    wlcl=L(1.)+L(.5)*sign*jnp.sqrt(jnp.abs(gdt)+L(1.e-10))
    buoyant=tlcl+dtlcl>tenv
    within_top=jnp.any(candidates)
    active=found&within_top&buoyant&(wlcl>=L(0.))
    stop=~found|~within_top|(~buoyant&(kpbl>=env.llfc))
    thte=tmix*(L(1.e5)/pmix)**(L(.2854)*(L(1.)-L(.28)*qmix))*jnp.exp((L(3374.6525)/tlcl-L(2.5403))*qmix*(L(1.)+L(.81)*qmix))
    tvavg=L(.5)*(env.tv[klcl]+tenv*(L(1.)+L(.608)*qenv))
    plcl=env.p[klcl]*jnp.exp(c.g/(L(287.)*tvavg)*(env.z[klcl]-zlcl))
    es=c.aliq*jnp.exp((tenv*c.bliq-c.cliq)/(tenv-c.dliq))
    qese=c.ep2*es/(plcl-es)
    thtes=tenv*(L(1.e5)/plcl)**(L(.2854)*(L(1.)-L(.28)*qese))*jnp.exp((L(3374.6525)/tenv-L(2.5403))*qese*(L(1.)+L(.81)*qese))
    rho=plcl/(L(287.)*tvlcl)
    au0=L(3.141592654)*L(1500.)*L(1500.)
    vmflcl=rho*au0
    return dict(active=active,stop=stop,lc=lc,kpbl=kpbl,k=km,klcl=klcl,zlcl=zlcl,tlcl=tlcl,tvlcl=tvlcl,
        tven=tven,tmix=tmix,qmix=qmix,pmix=pmix,zmix=zmix,depth=depth,wlcl=wlcl,
        thte=thte,thtes=thtes,vmflcl=vmflcl,au0=au0)


_PLUME_PROFILES=('TU','TVU','QU','WU','UMF','UER','UDR','QLIQ','QICE','QLQOUT','QICOUT',
                 'DETLQ','DETIC','PPTLIQ','PPTICE','RATIO2','THETEU','THETEE','THTES','QDT','EQFRC')


def updraft(env,tr,*,mode='r4'):
    """KFPARA680-1024: gradual freezing, Gaussian mixing, overshoot detrainment."""
    c=constants(mode);L=c.L;dtype=env.t.dtype;kx=len(env.t)-1
    k=tr['k'];kpbl=tr['kpbl'];vmf=tr['vmflcl']
    zero=lambda:jnp.zeros_like(env.t)
    s={name:zero() for name in _PLUME_PROFILES}
    for name,value in (('WU',tr['wlcl']),('UMF',vmf),('TU',tr['tlcl']),('TVU',tr['tvlcl']),
                       ('QU',tr['qmix']),('EQFRC',L(1.)),('THETEU',tr['thte']),('THTES',tr['thtes'])):
        s[name]=s[name].at[k].set(value)
    s.update(nk=k,LET=tr['klcl'],KFRZ=tr['lc'],WTW=tr['wlcl']*tr['wlcl'],
             UPOLD=vmf,UPNEW=vmf,ABE=jnp.asarray(0.,dtype),TRPPT=jnp.asarray(0.,dtype),
             # Original RL is unset before the first liquid-only TPMIX, which
             # does not read it. SS43 zero/sNaN initialization has identical outputs.
             RL=jnp.asarray(0.,dtype),TTEMP=jnp.asarray(L(268.16),dtype),IFLAG=jnp.int32(0),
             EE1=jnp.asarray(1.,dtype),UD1=jnp.asarray(0.,dtype),done=jnp.asarray(False))
    ttfrz,tbfrz=L(268.16),L(248.16)
    def body(s):
        nk=s['nk'];nk1=nk+1
        t,q,ql,qi,qnl,qni=tpmix(env.p[nk1],s['THETEU'][nk],env.t[nk1],s['QU'][nk],
            s['QLIQ'][nk],s['QICE'][nk],s['RATIO2'][nk],s['RL'],mode=mode)
        ratio=s['RATIO2'][nk]
        tvu=t*(L(1.)+L(.608)*q)
        freezes=(t<=ttfrz)&(s['IFLAG']<1)
        prior=jnp.minimum(s['TTEMP'],ttfrz)
        middle=t>tbfrz
        frc=jnp.where(freezes,jnp.where(middle,(prior-t)/(ttfrz-tbfrz),(prior-tbfrz)/(ttfrz-tbfrz)),L(0.))
        r1=jnp.where(middle,(prior-t)/(prior-tbfrz),L(1.))
        flag=jnp.where(freezes&~middle,1,s['IFLAG'])
        qnw=jnp.where(freezes,qnl,L(0.))
        qni=jnp.where(freezes,qni+qnl*r1*L(.5),qni)
        qnl=jnp.where(freezes,qnl-qnl*r1*L(.5),qnl)
        eff=(ttfrz-tbfrz)/(prior-tbfrz)
        ttemp=jnp.where(freezes,t,s['TTEMP'])
        first=nk==k
        be=jnp.where(first,(tr['tvlcl']+tvu)/(tr['tven']+env.tv[nk1])-L(1.),
                     (s['TVU'][nk]+tvu)/(env.tv[nk]+env.tv[nk1])-L(1.))
        dzz=jnp.where(first,env.z[nk1]-tr['zlcl'],env.dza[nk])
        boterm=L(2.)*dzz*c.g*be/L(1.5)
        enterm=jnp.where(first,L(0.),L(2.)*s['UER'][nk]*s['WTW']/s['UPOLD'])
        ql,qi,wtw,qlout,qiout=condload(ql,qi,s['WTW'],dzz,boterm,enterm,L(.01),qnl,qni,mode=mode)
        rising=wtw>L(0.)
        wu=wtw/jnp.sqrt(jnp.abs(wtw))
        thtes=env.t[nk1]*(L(1.e5)/env.p[nk1])**(L(.2854)*(L(1.)-L(.28)*env.qes[nk1]))
        thtes=thtes*jnp.exp((L(3374.6525)/env.t[nk1]-L(2.5403))*env.qes[nk1]*(L(1.)+L(.81)*env.qes[nk1]))
        udlbe=(L(2.)*tr['thte']/(s['THTES'][nk]+thtes)-L(1.))*dzz
        abe=jnp.where(rising&(udlbe>L(0.)),s['ABE']+udlbe*c.g,s['ABE'])
        thte=s['THETEU'][nk];rl=s['RL']
        def freeze(_):
            return dtfrznew(t,env.p[nk1],thte,q,ql,qi,ratio,ttfrz,tbfrz,qnw,rl,frc,eff,flag,mode=mode)
        def unchanged(_):
            return t,thte,q,ql,qi,ratio,jnp.asarray(0.,dtype),rl,frc,flag.astype(dtype)
        tfr,thfr,qfr,qlfr,qifr,rfr,_,rlfr,_,iffr=jax.lax.cond(rising&(frc>L(1.e-6)),freeze,unchanged,None)
        t,thte,q,ql,qi,ratio,rl,flag=tfr,thfr,qfr,qlfr,qifr,rfr,rlfr,iffr.astype(jnp.int32)
        thtee=envirtht(env.p[nk1],env.t[nk1],env.q[nk1],ratio,rl,mode=mode)
        rei=vmf*env.dp[nk1]*L(.03)/L(1500.)
        tvqu=t*(L(1.)+L(.608)*q-ql-qi)
        cold=tvqu<=env.tv[nk1]
        def warm_rates(_):
            f1=L(.95);f2=L(1.)-f1
            thtmp=f1*thtee+f2*thte;qtmp=f1*env.q[nk1]+f2*q
            tl,qq,qll,qii,_,_=tpmix(env.p[nk1],thtmp,tvqu,qtmp,f2*ql,f2*qi,ratio,rl,mode=mode)
            tv95=tl*(L(1.)+L(.608)*qq-qll-qii)
            def rate10(_):
                f1=L(.10);f2=L(1.)-f1
                thtmp=f1*thtee+f2*thte;qtmp=f1*env.q[nk1]+f2*q
                # Original TPMIX updated TTMP in the 95% trial. It is the
                # initial temperature of the 10% trial; do not reset it.
                tt,qq,qll,qii,_,_=tpmix(env.p[nk1],thtmp,tl,qtmp,f2*ql,f2*qi,ratio,rl,mode=mode)
                tv10=tt*(L(1.)+L(.608)*qq-qll-qii)
                eq=jnp.where(tv10==tvqu,L(1.),(env.tv[nk1]-tvqu)*f1/(tv10-tvqu))
                eq=jnp.clip(eq,L(0.),L(1.))
                ee,ud=prof5(eq,mode=mode)
                ee=jnp.where(eq==L(1.),L(1.),jnp.where(eq==L(0.),L(0.),ee))
                ud=jnp.where(eq==L(1.),L(0.),jnp.where(eq==L(0.),L(1.),ud))
                return eq,ee,ud
            one=lambda _:(jnp.asarray(1.,dtype),jnp.asarray(1.,dtype),jnp.asarray(0.,dtype))
            return jax.lax.cond(tv95>env.tv[nk1],one,rate10,None)
        zero_rates=lambda _:(jnp.asarray(0.,dtype),jnp.asarray(0.,dtype),jnp.asarray(1.,dtype))
        eq,ee2,ud2=jax.lax.cond(cold,zero_rates,warm_rates,None)
        ee1=jnp.where(first,L(1.),s['EE1']);ud1=jnp.where(first,L(0.),s['UD1'])
        uer=jnp.where(cold,L(0.),L(.5)*rei*(ee1+ee2))
        udr=jnp.where(cold,rei,L(.5)*rei*(ud1+ud2))
        too_small=s['UMF'][nk]-udr<L(10.)
        continuing=rising&~too_small
        abe=jnp.where(rising&too_small&(udlbe>L(0.)),abe-udlbe*c.g,abe)
        let=jnp.where(rising&~cold,nk1,s['LET'])
        let=jnp.where(rising&too_small,nk,let)
        upold=s['UMF'][nk]-udr;upnew=upold+uer
        detlq=ql*udr;detic=qi*udr;qdt=q
        qnew=(upold*q+uer*env.q[nk1])/upnew
        thnew=(thte*upold+thtee*uer)/upnew
        qlnew=ql*upold/upnew;qinew=qi*upold/upnew
        ppl=qlout*(s['UMF'][nk]-udr);ppi=qiout*(s['UMF'][nk]-udr)
        trppt=s['TRPPT']+ppl+ppi
        kfrz=jnp.where(jnp.abs(ratio-L(1.))>L(1.e-6),nk1,s['KFRZ'])
        uer_final=jnp.where(nk1<=kpbl,uer+vmf*env.dp[nk1]/tr['depth'],uer)
        updates={'TU':t,'TVU':tvu,'QU':jnp.where(continuing,qnew,q),
            'QLIQ':jnp.where(continuing,qlnew,ql),'QICE':jnp.where(continuing,qinew,qi),
            'QLQOUT':qlout,'QICOUT':qiout,'RATIO2':ratio,
            'THETEU':jnp.where(continuing,thnew,thte),
            'WU':jnp.where(rising,wu,s['WU'][nk1]),'THTES':jnp.where(rising,thtes,s['THTES'][nk1]),
            'THETEE':jnp.where(rising,thtee,s['THETEE'][nk1]),
            'EQFRC':jnp.where(rising,eq,s['EQFRC'][nk1]),
            'UER':jnp.where(rising,jnp.where(continuing,uer_final,uer),s['UER'][nk1]),
            'UDR':jnp.where(rising,udr,s['UDR'][nk1]),
            'UMF':jnp.where(continuing,upnew,s['UMF'][nk1]),
            'DETLQ':jnp.where(continuing,detlq,s['DETLQ'][nk1]),
            'DETIC':jnp.where(continuing,detic,s['DETIC'][nk1]),
            'QDT':jnp.where(continuing,qdt,s['QDT'][nk1]),
            'PPTLIQ':jnp.where(continuing,ppl,s['PPTLIQ'][nk1]),
            'PPTICE':jnp.where(continuing,ppi,s['PPTICE'][nk1])}
        result={name:s[name].at[nk1].set(value) for name,value in updates.items()}
        result.update(nk=jnp.where(continuing,nk1,nk),LET=let,
            KFRZ=jnp.where(continuing,kfrz,s['KFRZ']),WTW=wtw,
            UPOLD=jnp.where(continuing,upold,s['UPOLD']),UPNEW=jnp.where(continuing,upnew,s['UPNEW']),
            ABE=abe,TRPPT=jnp.where(continuing,trppt,s['TRPPT']),RL=jnp.where(rising,rl,s['RL']),
            TTEMP=jnp.where(rising,ttemp,s['TTEMP']),IFLAG=jnp.where(rising,flag,s['IFLAG']),
            EE1=jnp.where(continuing,ee2,s['EE1']),UD1=jnp.where(continuing,ud2,s['UD1']),
            done=~continuing)
        return result
    s=jax.lax.while_loop(lambda s:(s['nk']<kx)&~s['done'],body,s)
    ltop=s['nk'];let=s['LET']
    height=env.z[ltop]-tr['zlcl']
    usable=(height>=L(3000.))&(s['ABE']>=L(1.))
    def at_top(s):
        udr=s['UMF'][ltop]+s['UDR'][ltop]-s['UER'][ltop]
        detlq=s['QLIQ'][ltop]*udr*s['UPNEW']/s['UPOLD']
        detic=s['QICE'][ltop]*udr*s['UPNEW']/s['UPOLD']
        return dict(s,UDR=s['UDR'].at[ltop].set(udr),DETLQ=s['DETLQ'].at[ltop].set(detlq),
            DETIC=s['DETIC'].at[ltop].set(detic),UER=s['UER'].at[ltop].set(L(0.)),
            UMF=s['UMF'].at[ltop].set(L(0.)),TRPPT=s['TRPPT']-(s['PPTLIQ'][ltop]+s['PPTICE'][ltop]))
    def overshoot(s):
        dptt=ordered_sum(env.dp,let+1,ltop)
        dumfdp=s['UMF'][let]/dptt
        def body(s,nk):
            udr=env.dp[nk]*dumfdp;umf=s['UMF'][nk-1]-udr
            detlq=s['QLIQ'][nk]*udr;detic=s['QICE'][nk]*udr
            trppt=s['TRPPT']-s['PPTLIQ'][nk]-s['PPTICE'][nk]
            ppl=(s['UMF'][nk-1]-udr)*s['QLQOUT'][nk]
            ppi=(s['UMF'][nk-1]-udr)*s['QICOUT'][nk]
            vals=dict(UDR=udr,UMF=umf,DETLQ=detlq,DETIC=detic,PPTLIQ=ppl,PPTICE=ppi)
            use=(nk>let)&(nk<=ltop)
            result=dict(s)
            for name,value in vals.items():result[name]=s[name].at[nk].set(jnp.where(use,value,s[name][nk]))
            result['TRPPT']=jnp.where(use,trppt+ppl+ppi,s['TRPPT'])
            return result,None
        return jax.lax.scan(body,s,jnp.arange(1,kx+1,dtype=jnp.int32))[0]
    # Source rejects short/nonbuoyant plumes before the overshoot calculation.
    s=jax.lax.cond(usable,lambda s:jax.lax.cond(let==ltop,at_top,overshoot,s),lambda s:s,s)
    s.update(LTOP=ltop,usable=usable,CLDHGT=height)
    return s


def prepare_downdraft(env,tr,plume,dt,dx,*,mode='r4'):
    """KFPARA1029-1191: below-cloud fluxes, thermodynamics, lifetime/efficiency."""
    c=constants(mode);L=c.L;kx=len(env.t)-1;k=jnp.arange(kx+1,dtype=jnp.int32)
    dt=jnp.asarray(dt,env.t.dtype);dx=jnp.asarray(dx,env.t.dtype)
    s=dict(plume);ltop=s['LTOP'];lc=tr['lc'];kpbl=tr['kpbl'];kb=tr['k'];vmf=tr['vmflcl']
    def lower(s,nk):
        uer=vmf*env.dp[nk]/tr['depth']
        umf=jnp.where(nk==lc,uer,jnp.where(nk<=kpbl,s['UMF'][nk-1]+uer,vmf))
        uer=jnp.where((nk==lc)|(nk<=kpbl),uer,L(0.))
        within=nk>=lc
        values={'UMF':jnp.where(within,umf,L(0.)),'UER':jnp.where(within,uer,L(0.)),
                'TU':jnp.where(within,tr['tmix']+(env.z[nk]-tr['zmix'])*(-c.g/L(1004.5)),L(0.)),
                'QU':jnp.where(within,tr['qmix'],L(0.)),'WU':jnp.where(within,tr['wlcl'],L(0.))}
        for name in ('UDR','QDT','QLIQ','QICE','QLQOUT','QICOUT','PPTLIQ','PPTICE','DETLQ','DETIC','RATIO2'):
            values[name]=jnp.asarray(0.,env.t.dtype)
        values['THETEE']=envirtht(env.p[nk],env.t[nk],env.q[nk],L(0.),s['RL'],mode=mode)
        thta=env.t[nk]*(L(1.e5)/env.p[nk])**(L(.2854)*(L(1.)-L(.28)*env.q[nk]))
        values['THTES']=thta*jnp.exp((L(3374.6525)/env.t[nk]-L(2.5403))*env.qes[nk]*(L(1.)+L(.81)*env.qes[nk]))
        values['EQFRC']=jnp.asarray(1.,env.t.dtype)
        result=dict(s)
        for name,value in values.items():result[name]=s[name].at[nk].set(jnp.where(nk<=kb,value,s[name][nk]))
        return result,None
    s=jax.lax.scan(lower,s,jnp.arange(1,kx+1,dtype=jnp.int32))[0]
    above=k>ltop
    for name in ('UMF','UDR','UER','QDT','QLIQ','QICE','QLQOUT','QICOUT','DETLQ','DETIC','PPTLIQ','PPTICE'):
        s[name]=jnp.where(above,L(0.),s[name])
    for name in ('TU','QU','WU'):s[name]=jnp.where(k>ltop+1,L(0.),s[name])
    s['EMS']=env.dp*(dx*dx)/c.g;s['EMSD']=L(1.)/s['EMS']
    exn=(L(1.e5)/env.p)**(L(.2854)*(L(1.)-L(.28)*s['QDT']))
    s['THTAU']=jnp.where((k>=1)&~above,s['TU']*exn,L(0.))
    exn=(L(1.e5)/env.p)**(L(.2854)*(L(1.)-L(.28)*env.q))
    s['THTA0']=jnp.where((k>=1)&~above,env.t*exn,L(0.))
    s['THTAD']=jnp.zeros_like(env.t)
    s['TG']=env.t;s['QG']=env.q
    for name in ('QLG','QIG','QRG','QSG','DMF','DER','DDR','WD','TZ','QD','THETED'):
        s[name]=jnp.zeros_like(env.t)
    s['OMG']=jnp.zeros(kx+2,env.t.dtype)
    p150=env.p[tr['klcl']]-L(15000.)
    lvf=jnp.minimum(jnp.max(jnp.where((k>=1)&(k<=ltop)&(env.p>p150),k,0)),s['LET'])
    usr=s['UMF'][lvf+1]*(s['QU'][lvf+1]+s['QLIQ'][lvf+1]+s['QICE'][lvf+1])
    usr=jnp.minimum(usr,s['TRPPT']);usr=jnp.where(usr<L(1.e-8),s['TRPPT'],usr)
    wind=jnp.sqrt(env.u*env.u+env.v*env.v)
    vconv=L(.5)*(wind[tr['klcl']]+wind[env.l5])
    time_adv=jnp.where(vconv>L(0.),dx/vconv,L(3600.))
    timec=jnp.minimum(jnp.maximum(L(1800.),time_adv),L(3600.))
    nic=jnp.floor(timec/dt+L(.5)).astype(jnp.int32)
    timec=nic.astype(env.t.dtype)*dt
    sign=jnp.where(wind[ltop]>wind[tr['klcl']],L(1.),L(-1.))
    vws=(env.u[ltop]-env.u[tr['klcl']])*(env.u[ltop]-env.u[tr['klcl']])
    vws=vws+(env.v[ltop]-env.v[tr['klcl']])*(env.v[ltop]-env.v[tr['klcl']])
    vws=L(1000.)*sign*jnp.sqrt(vws)/(env.z[ltop]-env.z[tr['klcl']])
    pef=L(1.591)+vws*(L(-.639)+vws*(L(.0953)-vws*L(.00496)))
    pef=jnp.minimum(jnp.maximum(pef,L(.2)),L(.9))
    cbh=(tr['zlcl']-env.z[1])*L(.003281)
    rcbh=L(.96729352)+cbh*(L(-.70034167)+cbh*(L(.162179896)+cbh*(L(-.012569798)+cbh*(L(.00042772)-cbh*L(.00000544)))))
    rcbh=jnp.where(cbh<L(3.),L(.02),rcbh);rcbh=jnp.where(cbh>L(25.),L(2.4),rcbh)
    pefcbh=jnp.minimum(L(1.)/(L(1.)+rcbh),L(.9))
    peff=L(.5)*(pef+pefcbh)
    s.update(LVF=lvf,USR=usr,TADVEC=time_adv,TIMEC=timec,NIC=nic,PEFF=peff,PEFF2=peff)
    return s


def downdraft(env,tr,s,*,mode='r4'):
    """KFPARA1206-1463: old analytic downdraft and precipitation evaporation."""
    c=constants(mode);L=c.L;dtype=env.t.dtype;kx=len(env.t)-1;k=jnp.arange(kx+1,dtype=jnp.int32)
    s=dict(s);ltop=s['LTOP'];klcl=tr['klcl']
    start=jnp.maximum(tr['kpbl'],klcl)
    mask=(k>=start+1)&(k<=jnp.maximum(start+1,ltop-1))
    minimum=jnp.min(jnp.where(mask,s['THTES'],jnp.inf))
    lfs=jnp.max(jnp.where(mask&(s['THTES']==minimum),k,0))
    liquid=envirtht(env.p[lfs],env.t[lfs],env.q[lfs],L(0.),s['RL'],mode=mode)
    thee=jnp.where(s['RATIO2'][lfs]>L(0.),liquid,s['THETEE'][lfs])
    eq=(s['THTES'][lfs]-s['THETEU'][lfs])/(thee-s['THETEU'][lfs])
    eq=jnp.minimum(jnp.maximum(eq,L(0.)),L(1.))
    dtm=jnp.where(env.ml>0,L(.5)*(s['QU'][klcl]-s['QU'][ltop])*L(3.339e5)/L(1004.5),L(0.))
    tz=env.t[lfs]-dtm
    es=c.aliq*jnp.exp((tz*c.bliq-c.cliq)/(tz-c.dliq));qs=c.ep2*es/(env.p[lfs]-es)
    qd=eq*env.q[lfs]+(L(1.)-eq)*s['QU'][lfs]
    thad=tz*(L(1.e5)/env.p[lfs])**(L(.2854)*(L(1.)-L(.28)*qd))
    thed=jnp.where(qd>=qs,thad*jnp.exp((L(3374.6525)/tz-L(2.5403))*qs*(L(1.)+L(.81)*qs)),
                  envirtht(env.p[lfs],tz,qd,L(0.),s['RL'],mode=mode))
    s['THETEE']=s['THETEE'].at[lfs].set(thee);s['EQFRC']=s['EQFRC'].at[lfs].set(eq)
    for name,value in (('TZ',tz),('QD',qd),('THTAD',thad),('THETED',thed)):
        s[name]=s[name].at[lfs].set(value)
    crossing=(k<lfs)&(k>=1)&((thed>s['THTES'])|(k==1))
    ldb=jnp.max(jnp.where(crossing,k,0));ldt=ldb
    allowed=(ldb!=lfs-1)&((env.p[ldb]-env.p[lfs])>=L(5000.))
    tvd=env.t[lfs]*(L(1.)+L(.608)*env.qes[lfs])
    density=env.p[lfs]/(L(287.)*tvd)
    dmf=-(L(1.)-s['PEFF'])*tr['au0']*density
    s['DMF']=s['DMF'].at[lfs].set(dmf)
    s['DER']=s['DER'].at[lfs].set(eq*dmf)
    def descent(s,nk):
        uses=(nk<lfs)&(nk>=ldb)
        detrain=nk<=ldt
        der=jnp.where(detrain,L(0.),dmf*L(.03)*env.dp[nk]/L(1500.))
        ddr=jnp.where(detrain,-s['DMF'][ldt+1]*env.dp[nk]/env.dp[ldb],L(0.))
        mass=s['DMF'][nk+1]+jnp.where(detrain,ddr,der)
        liquid=envirtht(env.p[nk],env.t[nk],env.q[nk],L(0.),s['RL'],mode=mode)
        thee=jnp.where(s['RATIO2'][nk]>L(0.),liquid,s['THETEE'][nk])
        h=jnp.where(detrain,s['THETED'][nk+1],(s['THETED'][nk+1]*s['DMF'][nk+1]+thee*der)/mass)
        q=jnp.where(detrain,s['QD'][nk+1],(s['QD'][nk+1]*s['DMF'][nk+1]+env.q[nk]*der)/mass)
        out=dict(s)
        for name,value in (('DER',der),('DDR',ddr),('DMF',mass),('THETED',h),('QD',q)):
            out[name]=s[name].at[nk].set(jnp.where(uses,value,s[name][nk]))
        out['THETEE']=s['THETEE'].at[nk].set(jnp.where(uses&~detrain,thee,s['THETEE'][nk]))
        return out,None
    # This pristine version fixes LDT=LDB, and the physical descent only
    # visits LFS-1..LDB. Do not run 40 masked thermodynamic solves.
    s=jax.lax.cond(allowed,
        lambda s:jax.lax.fori_loop(jnp.int32(0),lfs-ldb,lambda i,s:descent(s,lfs-1-i)[0],s),
        lambda s:s,s)
    def evaporate(old,nk):
        s,tder=old
        temp,_=tpdd(env.p[nk],s['THETED'][ldt],env.t[nk],s['QD'][nk],L(1.),mode=mode)
        es=c.aliq*jnp.exp((temp*c.bliq-c.cliq)/(temp-c.dliq));qs=c.ep2*es/(env.p[nk]-es)
        dss=(c.cliq-c.bliq*c.dliq)/((temp-c.dliq)*(temp-c.dliq))
        rl=c.xlv0-c.xlv1*temp
        delta=rl*qs*(L(1.)-L(.90))/(L(1004.5)+rl*L(.90)*qs*dss)
        t1rh=temp+delta
        es=L(.90)*c.aliq*jnp.exp((c.bliq*t1rh-c.cliq)/(t1rh-c.dliq))
        qsrh=c.ep2*es/(env.p[nk]-es)
        t1rh=jnp.where(qsrh<s['QD'][nk],temp,t1rh)
        qsrh=jnp.where(qsrh<s['QD'][nk],s['QD'][nk],qsrh)
        th=t1rh*(L(1.e5)/env.p[nk])**(L(.2854)*(L(1.)-L(.28)*qsrh))
        use=(nk>=ldb)&(nk<=ldt)
        tder=jnp.where(use,tder+(qsrh-s['QD'][nk])*s['DDR'][nk],tder)
        out=dict(s)
        for name,value in (('TZ',t1rh),('QD',qsrh),('THTAD',th)):
            out[name]=s[name].at[nk].set(jnp.where(use,value,s[name][nk]))
        out['RL']=jnp.where(use,rl,s['RL'])
        return (out,tder),None
    # LDT=LDB makes the original 135 loop a single TPDD call. SS66/69
    # observed exactly one call per active sounding (two total).
    s,tder=jax.lax.cond(allowed,lambda s:evaporate((s,jnp.asarray(0.,dtype)),ldb)[0],
                       lambda s:(s,jnp.asarray(0.,dtype)),s)
    def precip_sum(old,nk):return jnp.where((nk>=klcl)&(nk<=lfs),old+s['PPTLIQ'][nk]+s['PPTICE'][nk],old),None
    ppr=jax.lax.scan(precip_sum,jnp.asarray(0.,dtype),jnp.arange(1,kx+1,dtype=jnp.int32))[0]
    dev=tder/s['DMF'][lfs]
    ppt=s['PEFF']*s['USR'];rced=s['TRPPT']-ppt
    dpptdf=jnp.where(lfs>=klcl,(L(1.)-s['PEFF'])*ppr*(L(1.)-eq)/s['UMF'][lfs],L(0.))
    cndtnf=(s['QLIQ'][lfs]+s['QICE'][lfs])*(L(1.)-eq)
    ddmass=rced/(dev+dpptdf+cndtnf)
    no_dd=(tder<L(1.))|(ddmass>L(0.))
    upd=jnp.where(lfs>=klcl,(s['UMF'][lfs]-(L(1.)-eq)*ddmass)/s['UMF'][lfs],L(1.))
    cap=upd>L(1.5);upd=jnp.where(cap,L(1.5),upd)
    ddmass2=s['UMF'][lfs]*(upd-L(1.))/(eq-L(1.))
    rced2=ddmass2*(dev+dpptdf+cndtnf)
    ppt=jnp.where(cap,ppt+(rced-rced2),ppt)
    peff2=jnp.where(cap,ppt/s['USR'],s['PEFF2'])
    ddmass=jnp.where(cap,ddmass2,ddmass)
    ddinc=ddmass/s['DMF'][lfs]
    cpr=s['TRPPT']+ppr*(upd-L(1.))
    ppt=ppt+s['PEFF']*ppr*(upd-L(1.))
    tder=tder*ddinc
    for name in ('DMF','DER','DDR'):
        val=jnp.where((k>=ldb)&(k<=lfs),s[name]*ddinc,L(0.))
        s[name]=jnp.where(no_dd,L(0.),val)
    for name in ('THTAD','WD','TZ','QD'):
        val=jnp.where((k>=ldb)&(k<=lfs),s[name],L(0.))
        if name in ('TZ','QD'):val=jnp.where((k>ldt)&(k<lfs),L(0.),val)
        s[name]=jnp.where(no_dd&(k<=ltop),L(0.),val)
    for name in ('UMF','UDR','UER','PPTLIQ','PPTICE','DETLQ','DETIC'):
        s[name]=jnp.where(~no_dd&(k>=tr['lc'])&(k<=lfs),s[name]*upd,s[name])
    s.update(LFS=lfs,LDB=jnp.where(no_dd,lfs,ldb),LDT=ldt,
             TDER=jnp.where(no_dd,L(0.),tder),PPTFLX=jnp.where(no_dd,s['TRPPT'],ppt),
             CPR=jnp.where(no_dd,s['TRPPT'],cpr),UPDINC=jnp.where(no_dd,L(1.),upd),
             CNDTNF=jnp.where(no_dd,L(0.),cndtnf),
             PEFF=jnp.where(no_dd,s['PEFF'],peff2),PEFF2=jnp.where(no_dd,s['PEFF2'],peff2))
    return s


def feedback(env,tr,s,dx,*,mode='r4'):
    """KFPARA185-245: conservative environmental response and remaining CAPE."""
    c=constants(mode);L=c.L;dtype=env.t.dtype;kx=len(env.t)-1
    dx=jnp.asarray(dx,dtype);dsq=dx*dx
    k=jnp.arange(kx+1,dtype=jnp.int32);valid=(k>=1)&(k<=s['LTOP'])
    dom=-(s['UER']-s['DER']-s['UDR']-s['DDR'])*s['EMSD']
    def omega(old,nk):
        prev,dtt=old
        value=prev-env.dp[nk-1]*dom[nk-1]
        dtt1=L(.75)*env.dp[nk-1]/(jnp.abs(value)+L(1.e-10))
        use=nk<=s['LTOP']
        return (jnp.where(use,value,prev),jnp.where(use,jnp.minimum(dtt,dtt1),dtt)),jnp.where(use,value,L(0.))
    (_,dtt),values=jax.lax.scan(omega,(jnp.asarray(0.,dtype),s['TIMEC']),jnp.arange(2,kx+1,dtype=jnp.int32))
    omg=jnp.zeros(kx+2,dtype).at[2:kx+1].set(values)
    nstep=jnp.floor(s['TIMEC']/dtt+L(1.)+L(.5)).astype(jnp.int32)
    dtime=s['TIMEC']/nstep.astype(dtype)
    fxm=omg[:kx+1]*dsq/c.g
    def advection(_i,old):
        th,q=old
        previous_th=jnp.roll(th,1);previous_q=jnp.roll(q,1)
        thbot=jnp.where(valid&(k>=2),-fxm*jnp.where(omg[:kx+1]<=L(0.),previous_th,th),L(0.))
        qbot=jnp.where(valid&(k>=2),-fxm*jnp.where(omg[:kx+1]<=L(0.),previous_q,q),L(0.))
        thtop=-jnp.roll(thbot,-1);qtop=-jnp.roll(qbot,-1)
        thn=th+(thbot+s['UDR']*s['THTAU']+s['DDR']*s['THTAD']+thtop-(s['UER']-s['DER'])*s['THTA0'])*dtime*s['EMSD']
        qn=q+(qbot+s['UDR']*s['QDT']+s['DDR']*s['QD']+qtop-(s['UER']-s['DER'])*env.q)*dtime*s['EMSD']
        return jnp.where(valid,thn,th),jnp.where(valid,qn,q)
    thg,qg=jax.lax.fori_loop(jnp.int32(0),nstep,advection,(s['THTA0'],env.q))
    def borrow(old,nk):
        qg,events,fatal=old
        negative=(nk<=s['LTOP'])&(qg[nk]<L(0.))
        fatal |= negative&(nk==1)
        use=negative&(nk>1)
        nk1=jnp.where(nk==s['LTOP'],tr['klcl'],nk+1)
        tma=qg[nk1]*s['EMS'][nk1];tmb=qg[nk-1]*s['EMS'][nk-1]
        tmm=(qg[nk]-L(1.e-9))*s['EMS'][nk]
        bcoeff=-tmm/((tma*tma)/tmb+tmb);acoeff=bcoeff*tma/tmb
        tmb=tmb*(L(1.)-bcoeff);tma=tma*(L(1.)-acoeff)
        qg=qg.at[nk].set(jnp.where(use,L(1.e-9),qg[nk]))
        qg=qg.at[nk1].set(jnp.where(use,tma*s['EMSD'][nk1],qg[nk1]))
        qg=qg.at[nk-1].set(jnp.where(use,tmb*s['EMSD'][nk-1],qg[nk-1]))
        return (qg,events+use.astype(jnp.int32),fatal),None
    qg,borrow_events,fatal=jax.lax.fori_loop(jnp.int32(1),s['LTOP']+1,
        lambda nk,old:borrow(old,nk)[0],(qg,jnp.int32(0),jnp.asarray(False)))
    topomg=(s['UDR'][s['LTOP']]-s['UER'][s['LTOP']])*env.dp[s['LTOP']]*s['EMSD'][s['LTOP']]
    mass_fatal=jnp.abs(topomg-omg[s['LTOP']])>L(.001)
    exn=(L(1.e5)/env.p)**(L(.2854)*(L(1.)-L(.28)*qg))
    tg=jnp.where(valid,thg/exn,env.t)
    tvg=tg*(L(1.)+L(.608)*qg)
    rocpq=L(.2854)*(L(1.)-L(.28)*qg)
    thmix=ordered_sum(env.dp*tg*(L(1.e5)/env.p)**rocpq,tr['lc'],tr['kpbl'])/tr['depth']
    qmix=ordered_sum(env.dp*qg,tr['lc'],tr['kpbl'])/tr['depth']
    pmix=ordered_sum(env.dp*env.p,tr['lc'],tr['kpbl'])/tr['depth']
    rocpq=L(.2854)*(L(1.)-L(.28)*qmix)
    tmix=thmix*(pmix/L(1.e5))**rocpq
    es=c.aliq*jnp.exp((tmix*c.bliq-c.cliq)/(tmix-c.dliq));qs=c.ep2*es/(pmix-es)
    supersaturated=qmix>qs
    rl=c.xlv0-c.xlv1*tmix;cpm=L(1004.5)*(L(1.)+L(.887)*qmix)
    dss=qs*(c.cliq-c.bliq*c.dliq)/((tmix-c.dliq)*(tmix-c.dliq))
    dq=(qmix-qs)/(L(1.)+rl*dss/cpm)
    tmoist=tmix+rl/L(1004.5)*dq;qmoist=qmix-dq
    rocmoist=L(.2854)*(L(1.)-L(.28)*qmoist)
    thmoist=tmoist*(L(1.e5)/pmix)**rocmoist
    qdry=jnp.maximum(qmix,L(0.))
    emix=qdry*pmix/(c.ep2+qdry);tlog=jnp.log(emix/c.aliq)
    tdpt=(c.cliq-c.dliq*tlog)/(c.bliq-tlog)
    tlcl=tdpt-(L(.212)+L(1.571e-3)*(tdpt-L(273.16))-L(4.36e-4)*(tmix-L(273.16)))*(tmix-tdpt)
    tlcl=jnp.minimum(tlcl,tmix)
    plcl=L(1.e5)*(tlcl/thmix)**(L(1.)/rocpq)
    tmix=jnp.where(supersaturated,tmoist,tmix);qmix=jnp.where(supersaturated,qmoist,qdry)
    thmix=jnp.where(supersaturated,thmoist,thmix)
    tlcl=jnp.where(supersaturated,tmix,tlcl);plcl=jnp.where(supersaturated,pmix,plcl)
    candidates=(k>=tr['lc'])&(k>=1)&(plcl>=env.p)
    klcl=jnp.where(jnp.any(candidates),jnp.argmax(candidates),kx).astype(jnp.int32)
    km=klcl-1
    dlp=jnp.log(plcl/env.p[km])/jnp.log(env.p[klcl]/env.p[km])
    tenv=tg[km]+(tg[klcl]-tg[km])*dlp
    qenv=qg[km]+(qg[klcl]-qg[km])*dlp
    tven=tenv*(L(1.)+L(.608)*qenv)
    zlcl=env.z[km]+(env.z[klcl]-env.z[km])*dlp
    tvavg=L(.5)*(tven+tg[klcl]*(L(1.)+L(.608)*qg[klcl]))
    plcl=env.p[klcl]*jnp.exp(c.g/(L(287.)*tvavg)*(env.z[klcl]-zlcl))
    thte=tmix*(L(1.e5)/pmix)**(L(.2854)*(L(1.)-L(.28)*qmix))*jnp.exp((L(3374.6525)/tlcl-L(2.5403))*qmix*(L(1.)+L(.81)*qmix))
    es=c.aliq*jnp.exp((tenv*c.bliq-c.cliq)/(tenv-c.dliq));qese=c.ep2*es/(plcl-es)
    at_lcl=tenv*(L(1.e5)/plcl)**(L(.2854)*(L(1.)-L(.28)*qese))*jnp.exp((L(3374.6525)/tenv-L(2.5403))*qese*(L(1.)+L(.81)*qese))
    es=c.aliq*jnp.exp((tg*c.bliq-c.cliq)/(tg-c.dliq));qese=c.ep2*es/(env.p-es)
    thes=tg*(L(1.e5)/env.p)**(L(.2854)*(L(1.)-L(.28)*qese))*jnp.exp((L(3374.6525)/tg-L(2.5403))*qese*(L(1.)+L(.81)*qese))
    thes=thes.at[km].set(at_lcl)
    def remaining(old,nk):
        dzz=jnp.where(nk==km,env.z[klcl]-zlcl,env.dza[nk])
        be=(L(2.)*thte/(thes[nk+1]+thes[nk])-L(1.))*dzz
        # Old KFPARA245 includes overshoot through LTOP-1, rather than LET.
        return jnp.where((nk>=km)&(nk<s['LTOP'])&(be>L(0.)),old+be*c.g,old),None
    abeg=jax.lax.scan(remaining,jnp.asarray(0.,dtype),jnp.arange(1,kx,dtype=jnp.int32))[0]
    return dict(s,TG=tg,QG=qg,OMG=omg,FXM=fxm,NSTEP=nstep,DTIME=dtime,ABEG=abeg,
                BORROW_EVENTS=borrow_events,FATAL_SURFACE=fatal,FATAL_MASS=mass_fatal)


def closure(env,tr,s,dx,*,mode='r4'):
    """Old-KF CAPE-removal iteration (source165-265), with original exit rules."""
    c=constants(mode);L=c.L;dtype=env.t.dtype;kx=len(env.t)-1
    maximum_level=jnp.maximum(tr['klcl'],s['LFS'])
    def limit(old,nk):
        prior,maximum=old
        use=(nk>=tr['lc'])&(nk<=maximum_level)
        positive=(s['UER'][nk]-s['DER'][nk])>L(0.)
        value=s['EMS'][nk]/((s['UER'][nk]-s['DER'][nk])*s['TIMEC'])
        prior=jnp.where(use&positive,value,prior)
        maximum=jnp.where(use,jnp.minimum(maximum,prior),maximum)
        return (prior,maximum),None
    # AINCM1 is unassigned until its first positive (UER-DER). The source's
    # below-cloud UER at LC provides that assignment on qualified soundings.
    (_,aincmx),_=jax.lax.scan(limit,(jnp.asarray(0.,dtype),jnp.asarray(1000.,dtype)),
                            jnp.arange(1,kx+1,dtype=jnp.int32))
    ainc=jnp.minimum(L(1.),aincmx)
    base=dict(s)
    fluxes=('UMF','DMF','DETLQ','DETIC','UDR','UER','DER','DDR')
    k=jnp.arange(kx+1,dtype=jnp.int32);valid=(k>=1)&(k<=s['LTOP'])
    def rescale(s,amplitude):
        out=dict(s)
        for name in fluxes:out[name]=jnp.where(valid,base[name]*amplitude,s[name])
        out['TDER']=base['TDER']*amplitude
        out['PPTFLX']=base['PPTFLX']*amplitude
        return out
    s=dict(s,FXM=jnp.zeros_like(env.t),NSTEP=jnp.int32(0),DTIME=jnp.asarray(0.,dtype),
           ABEG=jnp.asarray(0.,dtype),BORROW_EVENTS=jnp.int32(0),BORROW_TOTAL=jnp.int32(0),FATAL_SURFACE=jnp.asarray(False),
           FATAL_MASS=jnp.asarray(False))
    s.update(AINC=ainc,AINCMX=aincmx,AINCOLD=jnp.asarray(0.,dtype),FABEOLD=jnp.asarray(0.,dtype),
             NCOUNT=jnp.int32(0),NOITR=jnp.asarray(False),DONE=jnp.asarray(False),ABORT=ainc<L(.05))
    bound=ainc/aincmx>L(.999)
    s=jax.lax.cond(bound,lambda s:rescale(s,s['AINC']),lambda s:s,s)
    stab=L(.95);upper=L(1.05)-stab;lower=L(.95)-stab
    def body(s):
        s=dict(s,NCOUNT=s['NCOUNT']+1)
        result=feedback(env,tr,s,dx,mode=mode)
        result['BORROW_TOTAL']=s['BORROW_TOTAL']+result['BORROW_EVENTS']
        abeg=result['ABEG'];fabe=abeg/(result['ABE']+L(1.e-8))
        dabe=jnp.maximum(result['ABE']-abeg,L(.1)*result['ABE'])
        original_done=result['NOITR']|result['FATAL_MASS']|result['FATAL_SURFACE']
        abort=(fabe>L(1.))&~original_done
        dfda=(fabe-result['FABEOLD'])/(result['AINC']-result['AINCOLD'])
        restore=(result['NCOUNT']!=1)&(dfda>L(0.))&~original_done&~abort
        done=original_done|((result['AINC']/result['AINCMX']>L(.999))&(fabe>upper))
        done |= (fabe<=upper)&(fabe>=lower)
        done |= result['NCOUNT']>10
        # A positive derivative takes source goto255 before the ordinary
        # convergence/iteration exits, and forces one final NOITR response.
        done=jnp.where(restore,False,done)
        normal_update=~original_done&~abort&~restore
        old_ainc=result['AINC'];old_fabe=fabe
        result['AINCOLD']=jnp.where(normal_update,old_ainc,result['AINCOLD'])
        result['FABEOLD']=jnp.where(normal_update,old_fabe,result['FABEOLD'])
        next_ainc=jnp.where(fabe==L(0.),result['AINC']*L(.5),
                           result['AINC']*stab*result['ABE']/(dabe+L(1.e-8)))
        next_ainc=jnp.where(restore,s['AINCOLD'],next_ainc)
        next_ainc=jnp.minimum(result['AINCMX'],next_ainc)
        scale=(~done|restore)&~abort
        result['AINC']=jnp.where(scale,next_ainc,result['AINC'])
        result['NOITR']=result['NOITR']|restore
        result['DONE']=done
        result['ABORT']=abort|(scale&(next_ainc<L(.05)))
        return jax.lax.cond(scale,lambda s:rescale(s,s['AINC']),lambda s:s,result)
    return jax.lax.while_loop(lambda s:~s['DONE']&~s['ABORT'],body,s)


def finish(env,tr,s,dt,dx,*,warm_rain=False,qi_flag=True,qs_flag=True,mode='r4'):
    """KFPARA270-320: condensate transport, phase mediation and REAL rates."""
    c=constants(mode);L=c.L;dtype=env.t.dtype;kx=len(env.t)-1
    dt=jnp.asarray(dt,dtype);dx=jnp.asarray(dx,dtype)
    k=jnp.arange(kx+1,dtype=jnp.int32);valid=(k>=1)&(k<=s['LTOP'])
    frc2=s['PPTFLX']/(s['CPR']*s['AINC'])
    rainfb=s['PPTLIQ']*s['AINC']*L(0.)*frc2
    snowfb=s['PPTICE']*s['AINC']*L(0.)*frc2
    def transport(_i,old):
        ql,qi,qr,qs=old
        def flux(q):
            inflow=jnp.where(valid&(k>=2)&(s['OMG'][:kx+1]<=L(0.)),-s['FXM']*jnp.roll(q,1),L(0.))
            outflow=jnp.where(valid&(k>=2)&(s['OMG'][:kx+1]>L(0.)),s['FXM']*q,L(0.))
            outflow=outflow+jnp.roll(inflow,-1)
            inflow=inflow+jnp.roll(jnp.where(valid&(k>=2)&(s['OMG'][:kx+1]>L(0.)),s['FXM']*q,L(0.)),-1)
            return inflow,outflow
        qlin,qlout=flux(ql);qiin,qiout=flux(qi);qrin,qrout=flux(qr);qsin,qsout=flux(qs)
        qln=ql+(qlin+s['DETLQ']-qlout)*s['DTIME']*s['EMSD']
        qin=qi+(qiin+s['DETIC']-qiout)*s['DTIME']*s['EMSD']
        qrn=qr+(qrin+s['QLQOUT']*s['UDR']-qrout+rainfb)*s['DTIME']*s['EMSD']
        qsn=qs+(qsin+s['QICOUT']*s['UDR']-qsout+snowfb)*s['DTIME']*s['EMSD']
        return tuple(jnp.where(valid,new,old) for new,old in zip((qln,qin,qrn,qsn),(ql,qi,qr,qs)))
    zeros=jnp.zeros_like(env.t)
    qlg,qig,qrg,qsg=jax.lax.fori_loop(jnp.int32(0),s['NSTEP'],transport,(zeros,zeros,zeros,zeros))
    def water(old,nk):
        initial,final=old
        initial=initial+env.q[nk]*s['EMS'][nk]
        final=final+s['QG'][nk]*s['EMS'][nk]
        final=final+(qlg[nk]+qig[nk]+qrg[nk]+qsg[nk])*s['EMS'][nk]
        return initial,final
    initial,final=jax.lax.fori_loop(jnp.int32(1),s['LTOP']+1,
        lambda nk,old:water(old,nk),(jnp.asarray(0.,dtype),jnp.asarray(0.,dtype)))
    final=final+s['PPTFLX']*s['TIMEC']*(L(1.)-L(0.))
    err=(final-initial)*L(100.)/initial
    fatal=s['FATAL_MASS']|s['FATAL_SURFACE']|(jnp.abs(err)>L(.05))
    nic=jnp.where(s['TADVEC']<s['TIMEC'],jnp.floor(s['TADVEC']/dt+L(.5)).astype(jnp.int32),s['NIC'])
    nca=nic.astype(dtype)*dt
    tg=s['TG']
    if not qi_flag:
        cpm=L(1004.5)*(L(1.)+L(.887)*s['QG'])
        if warm_rain:
            tg=tg-(qig+qsg)*L(3.339e5)/cpm
        else:
            tg=jnp.where(k<=env.ml,tg-(qig+qsg)*L(3.339e5)/cpm,tg+(qlg+qrg)*L(3.339e5)/cpm)
        rqc=(qlg+qig)/s['TIMEC'];rqi=zeros
        rqr=(qrg+qsg)/s['TIMEC'];rqs=zeros
    else:
        rqc=qlg/s['TIMEC'];rqi=qig/s['TIMEC'];rqr=qrg/s['TIMEC']
        rqs=qsg/s['TIMEC'] if qs_flag else zeros
        if not qs_flag:rqi=rqi+qsg/s['TIMEC']
    dtdt=(tg-env.t)/s['TIMEC'];rqv=(s['QG']-env.q)/s['TIMEC']
    prate=s['PPTFLX']*(L(1.)-L(0.))/(dx*dx)
    # Original KFPARA stops on these budget failures. Keep an explicit flag;
    # the driver maps it to the existing nonfinite failure path, never a repair.
    return dict(DTDT=dtdt[1:],RQVCUTEN=rqv[1:],RQCCUTEN=rqc[1:],RQICUTEN=rqi[1:],
        RQRCUTEN=rqr[1:],RQSCUTEN=rqs[1:],PRATEC=prate,RAINCV=dt*prate,NCA=nca,
        TIMEC=s['TIMEC'],LTOP=s['LTOP'],LCL=tr['klcl'],BORROW_EVENTS=s['BORROW_TOTAL'],
        SOURCE_FATAL=fatal,WATER_ERROR_PERCENT=err)


def find_plume(env,dx,*,mode='r4'):
    """Original sequential source-layer search; retain the accepted profiles."""
    dtype=env.t.dtype
    zero=lambda:jnp.zeros((),dtype)
    tr={name:zero() for name in ('zlcl','tlcl','tvlcl','tven','tmix','qmix','pmix','zmix','depth',
                                 'wlcl','thte','thtes','vmflcl','au0')}
    tr.update(active=jnp.asarray(False),stop=jnp.asarray(False),lc=jnp.int32(0),kpbl=jnp.int32(0),k=jnp.int32(0),klcl=jnp.int32(0))
    pl={name:jnp.zeros_like(env.t) for name in _PLUME_PROFILES}
    pl.update({name:zero() for name in ('WTW','UPOLD','UPNEW','ABE','TRPPT','RL','TTEMP','EE1','UD1','CLDHGT')})
    pl.update({name:jnp.int32(0) for name in ('nk','LET','KFRZ','IFLAG','LTOP')})
    pl.update(done=jnp.asarray(False),usable=jnp.asarray(False))
    def search(old):
        lc,done,accepted,old_tr,old_pl=old
        candidate=trigger(env,lc,dx,mode=mode)
        def try_plume(_):
            plume=updraft(env,candidate,mode=mode)
            return candidate,plume,plume['usable']
        def skip(_):return old_tr,old_pl,jnp.asarray(False)
        tr,pl,accepted=jax.lax.cond(candidate['active'],try_plume,skip,None)
        return lc+1,candidate['stop']|accepted,accepted,tr,pl
    _,_,accepted,tr,pl=jax.lax.while_loop(lambda s:~s[1]&(s[0]<=env.llfc),search,
        (jnp.int32(1),jnp.asarray(False),jnp.asarray(False),tr,pl))
    return accepted,tr,pl


def kf99_column(t,qv,p,dz,rho,w0avg,u,v,dt,dx,*,pi_phy=None,nca=-100.,
                 warm_rain=False,qi_flag=True,qs_flag=True,qr_flag=True,mode='r4'):
    """KFPARA + CPS output conversion, one column; W0AVG already updated by CPS.

    dt seconds, dx metres, pressure Pa, humidity kg/kg. W0AVG and all profiles
    are mass-level bottom-up arrays. Core output is WRF REAL; mode=r8 is solely
    the promoted-pristine diagnostic. The driver preserves an expired NCA on
    failed triggering and retains held rates separately while NCA is positive.
    """
    dtype=jnp.float32 if mode=='r4' else jnp.float64
    dt=jnp.asarray(dt,dtype);dx=jnp.asarray(dx,dtype);nca=jnp.asarray(nca,dtype)
    t=jnp.asarray(t,dtype);kx=len(t)
    def empty(_):
        z=jnp.zeros(kx,dtype);zs=jnp.zeros((),dtype)
        return dict(DTDT=z,RQVCUTEN=z,RQCCUTEN=z,RQICUTEN=z,RQRCUTEN=z,RQSCUTEN=z,
            PRATEC=zs,RAINCV=zs,NCA=nca,TIMEC=zs,LTOP=jnp.int32(0),LCL=jnp.int32(0),
            BORROW_EVENTS=jnp.int32(0),SOURCE_FATAL=jnp.asarray(False),WATER_ERROR_PERCENT=zs)
    def run(_):
        env=environment(t,qv,p,dz,rho,w0avg,u,v,mode=mode)
        accepted,tr,pl=find_plume(env,dx,mode=mode)
        def cloud(_):
            s=prepare_downdraft(env,tr,pl,dt,dx,mode=mode)
            s=downdraft(env,tr,s,mode=mode)
            s=closure(env,tr,s,dx,mode=mode)
            return jax.lax.cond(s['ABORT'],empty,
                lambda _:finish(env,tr,s,dt,dx,warm_rain=warm_rain,qi_flag=qi_flag,qs_flag=qs_flag,mode=mode),None)
        return jax.lax.cond(accepted,cloud,empty,None)
    # KFCPS uses >0.5*DT to skip, so equality must run (unlike the old eta alias).
    out=jax.lax.cond(nca<=jnp.asarray(.5,dtype)*dt,run,empty,None)
    if not qr_flag:
        out['RQCCUTEN']=out['RQCCUTEN']+out['RQRCUTEN']
        out['RQRCUTEN']=jnp.zeros_like(out['RQRCUTEN'])
    pi=(jnp.asarray(p,dtype)/jnp.asarray(1.e5,dtype))**jnp.asarray(2./7.,dtype) if pi_phy is None else jnp.asarray(pi_phy,dtype)
    out['RTHCUTEN']=out['DTDT']/pi
    # Pristine stops on its two mass/water budget failures. Signal the retained
    # operational nonfinite guard instead of returning apparently valid rates.
    for name in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RAINCV','PRATEC'):
        out[name]=jnp.where(out['SOURCE_FATAL'],jnp.asarray(jnp.nan,dtype),out[name])
    return out


def update_w0avg99(w0avg,w,dt,*,stepcu=1,cudt=0.,adapt_step_flag=False,mode='r4'):
    """Literal KFCPS running mean over full W faces, before the NCA trigger gate."""
    dtype=jnp.float32 if mode=='r4' else jnp.float64
    old=jnp.asarray(w0avg,dtype);faces=jnp.asarray(w,dtype)
    if faces.shape[-1]!=old.shape[-1]+1:
        raise ValueError('old KF requires nlev+1 W faces for the CPS running mean')
    dt=jnp.asarray(dt,dtype);cudt=jnp.asarray(cudt,dtype)
    tst=jnp.asarray(2*int(stepcu),dtype)
    if adapt_step_flag:
        den=jnp.asarray(2.,dtype)*jnp.maximum(cudt*jnp.asarray(60.,dtype),dt)
        average_factor=den-dt;w_factor=dt
    else:
        average_factor=tst-jnp.asarray(1.,dtype);w_factor=jnp.asarray(1.,dtype);den=tst
    mean=jnp.asarray(.5,dtype)*(faces[...,:-1]+faces[...,1:])
    return (old*average_factor+mean*w_factor)/den


def step_kf99_column(T0,QV0,P0,DZQ,RHOE,w0avg,U0,V0,dt,dx,*,w=None,nca=-100.,
                     stepcu=1,cudt=0.,adapt_step_flag=False,warm_rain=False,
                     f_qi=True,f_qs=True,f_qr=True,pi_phy=None):
    """Retained PhysicsStepResult API for the distinct native old-KF kernel.

    ``w=None`` replays recorded post-CPS W0AVG; operational callers pass W faces
    and retain both the returned carry and the held rates between calls.
    """
    from gpuwrf.contracts.physics_interfaces import PhysicsCarry,PhysicsDiagnostics,PhysicsStepResult,PhysicsTendency
    mean=jnp.asarray(w0avg,jnp.float32) if w is None else update_w0avg99(w0avg,w,dt,
        stepcu=stepcu,cudt=cudt,adapt_step_flag=adapt_step_flag)
    out=kf99_column(T0,QV0,P0,DZQ,RHOE,mean,U0,V0,dt,dx,pi_phy=pi_phy,nca=nca,
                   warm_rain=warm_rain,qi_flag=f_qi,qs_flag=f_qs,qr_flag=f_qr)
    zero=jnp.zeros_like(out['RTHCUTEN'])
    st=dict(u=zero,v=zero,theta=out['RTHCUTEN'],qv=out['RQVCUTEN'],qc=out['RQCCUTEN'],
            qr=out['RQRCUTEN'],qi=out['RQICUTEN'],qs=out['RQSCUTEN'])
    diagnostics={name.lower():out[name] for name in
                 ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RAINCV','PRATEC','TIMEC')}
    diagnostics.update(source_fatal=out['SOURCE_FATAL'],borrow_events=out['BORROW_EVENTS'],
                       water_error_percent=out['WATER_ERROR_PERCENT'])
    return PhysicsStepResult(tendency=PhysicsTendency(state_tendencies=st,
            accumulator_increments={'rainc_acc':out['RAINCV']}),
        carry=PhysicsCarry(cumulus={'w0avg':mean,'nca':out['NCA']}),
        diagnostics=PhysicsDiagnostics(cumulus=diagnostics))
