"""CAM Zhang–McFarlane column, original module_cu_camzm.F DOUBLE arithmetic.

Work in progress: parcel and dilute-buoyancy equations. This module is not
an admitted cu=7 endpoint. Indices follow WRF's CAM top-down, one-based grid;
the REAL caller's conversions belong outside this explicit DOUBLE island.
Every iterative parcel inversion is column local and reports its original
failure condition; no convergence floor or finite-value replacement is used.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as plt
from gpuwrf.kernels.phys_kf_column import _get,_set,_where,_roll,_gather_p

# Original module_cam_physconst / module_cam_shr_const_mod, not WRF REAL CP.
_CP=1004.64
_G=9.80616
_RGAS=6.02214e26*1.38065e-23
_R=_RGAS/28.966
_RV=_RGAS/18.016
_EPS=18.016/28.966
_LV=2.501e6
_LF=3.337e5
_CPLIQ=4188.0
_CPWV=1810.0
_TFREEZ=273.15


def _fori(start,stop,body,carry):
    # Original CAM loop indices are INTEGER4 despite REAL DOUBLE fields.
    return jax.lax.fori_loop(jnp.asarray(start,jnp.int32),jnp.asarray(stop,jnp.int32),body,carry)



def _cond(predicate,yes,no,operand):
    # JAX 0.10's Triton cond emitter cannot yield literal branch leaves.
    # Identity selects materialize them without arithmetic or changed bits;
    # native optimization folds these selects (CAM-MG uses the same enabler).
    def materialize(branch):
        def run(value):
            return jax.tree.map(lambda x:jax.lax.select(
                jnp.broadcast_to(predicate,jnp.shape(x)),x,x),branch(value))
        return run
    return jax.lax.cond(predicate,materialize(yes),materialize(no),operand)


def _saturation(t,p):
    es=6.112*jnp.exp(17.67*(t-_TFREEZ)/(243.5+t-_TFREEZ))
    return _EPS*es/(p-es)


def _entropy(t,p,qt):
    """Pristine:3692–3718; pressure hPa, total-water mixing ratio kg/kg."""
    latent=_LV-(_CPLIQ-_CPWV)*(t-_TFREEZ)
    qs=_saturation(t,p)
    qv=jnp.minimum(qt,qs)
    e=qv*p/(_EPS+qv)
    return ((_CP+qt*_CPLIQ)*jnp.log(t/_TFREEZ)-_R*jnp.log((p-e)/1000.)
            +latent*qv/t-qv*_RV*jnp.log(qv/qs))


def _ientropy(s,p,qt,guess):
    """Original Newton iterations 0..99; failed=original ENDRUN condition.

    WRF aborts at iteration 99 when still unconverged. The device caller
    must report/guard ``failed`` before accepting the physical update.
    """
    def cond(st):
        i,t,done=st
        return (i<100)&~done
    def step(st):
        i,t,done=st
        fs1=_entropy(t,p,qt)-s
        latent=_LV-(_CPLIQ-_CPWV)*(t-1.-_TFREEZ)
        # Source denominator is ((c3+Ts)-1)-tfreez; entropy(Ts-1)
        # would change this evaluation order before Newton's cancellation.
        es=6.112*jnp.exp(17.67*(t-1.-_TFREEZ)/(243.5+t-1.-_TFREEZ))
        qs=_EPS*es/(p-es)
        qv=jnp.minimum(qt,qs)
        e=qv*p/(_EPS+qv)
        fs2=((_CP+qt*_CPLIQ)*jnp.log((t-1.)/_TFREEZ)-_R*jnp.log((p-e)/1000.)
             +latent*qv/(t-1.)-qv*_RV*jnp.log(qv/qs)-s)
        delta=fs1/(fs2-fs1)
        return i+1,t+delta,jnp.abs(delta)<.001
    i,t,done=jax.lax.while_loop(cond,step,(jnp.int32(0),guess,jnp.array(False)))
    return t,_saturation(t,p),~done


def _parcel_dilute(p,t,q,tpert,launch,msg,kx):
    """Source:3376–3689, including two ordered condensate/freezing passes."""
    zero=jnp.zeros_like(t)
    # Compiler-defined zero at undefined scratch sites follows wave policy.
    state=dict(tmix=zero,qtmix=zero,qsmix=zero,smix=zero,
               sp=jnp.float64(0.),qp=jnp.float64(0.),mp=jnp.float64(0.),
               sp0=jnp.float64(0.),qp0=jnp.float64(0.),mp0=jnp.float64(0.),
               pl=_get(p,launch),tl=_get(t,launch),lcl=launch,failed=jnp.array(False))
    def first_step(i,st):
        k=kx-i
        def at_launch(st):
            st=dict(st)
            sp0=_entropy(_get(t,k),_get(p,k),_get(q,k))
            temp,qs,failed=_ientropy(sp0,_get(p,k),_get(q,k),_get(t,k))
            st.update(sp0=sp0,qp0=_get(q,k),mp0=jnp.float64(1.),
                      tmix=_set(st['tmix'],k,temp),qsmix=_set(st['qsmix'],k,qs),
                      smix=_set(st['smix'],k,sp0),qtmix=_set(st['qtmix'],k,_get(q,k)),
                      failed=st['failed']|failed)
            return st
        def above(st):
            st=dict(st)
            dp=_get(p,k)-_get(p,k+1)
            qe=.5*(_get(q,k)+_get(q,k+1))
            te=.5*(_get(t,k)+_get(t,k+1))
            pe=.5*(_get(p,k)+_get(p,k+1))
            se=_entropy(te,pe,qe)
            dpdz=-(pe*_G)/(_R*te)
            dzdp=1./dpdz
            dmpdp=-1.e-3*dzdp
            sp=st['sp']-dmpdp*dp*se
            qp=st['qp']-dmpdp*dp*qe
            mp=st['mp']-dmpdp*dp
            sm=(st['sp0']+sp)/(st['mp0']+mp)
            qm=(st['qp0']+qp)/(st['mp0']+mp)
            temp,qs,failed=_ientropy(sm,_get(p,k),qm,_get(st['tmix'],k+1))
            tmix=_set(st['tmix'],k,temp);qsmix=_set(st['qsmix'],k,qs)
            smix=_set(st['smix'],k,sm);qtmix=_set(st['qtmix'],k,qm)
            st.update(sp=sp,qp=qp,mp=mp,tmix=tmix,qsmix=qsmix,smix=smix,qtmix=qtmix,
                      failed=st['failed']|failed)
            crosses=(qs<=qm)&(_get(qsmix,k+1)>_get(qtmix,k+1))
            def cloud_base(st):
                st=dict(st)
                excess=qm-qs
                excess_next=_get(qtmix,k+1)-_get(qsmix,k+1)
                dqxsdp=(excess-excess_next)/dp
                pl=_get(p,k+1)-excess_next/dqxsdp
                dsdp=(sm-_get(smix,k+1))/dp
                dqtdp=(qm-_get(qtmix,k+1))/dp
                slcl=_get(smix,k+1)+dsdp*(pl-_get(p,k+1))
                qtlcl=_get(qtmix,k+1)+dqtdp*(pl-_get(p,k+1))
                tl,_qs,failed=_ientropy(slcl,pl,qtlcl,temp)
                st.update(lcl=k,pl=pl,tl=tl,failed=st['failed']|failed)
                return st
            return _cond(crosses,cloud_base,lambda st:st,st)
        return _cond(k==launch,at_launch,
                            lambda st:_cond(k<launch,above,lambda st:st,st),st)
    state=_fori(0,kx-msg,first_step,state)
    state.update(tp=t,qstp=q,tpv=t*(1.+1.608*q)/(1.+q),xsh2o=zero,dsx=zero,dsf=zero)
    def second_step(i,st):
        k=kx-i
        def at_launch(st):
            st=dict(st)
            temp=_get(st['tmix'],k);qv=_get(q,k)
            st.update(tp=_set(st['tp'],k,temp),qstp=_set(st['qstp'],k,qv),
                      tpv=_set(st['tpv'],k,(temp+tpert)*(1.+1.608*qv)/(1.+qv)))
            return st
        def above(st):
            def latent_pass(ii,st):
                st=dict(st)
                temp=_get(st['tmix'],k);qs=_get(st['qsmix'],k)
                excess=jnp.maximum(0.,_get(st['qtmix'],k)-qs-1.e-3)
                dsx=_get(st['dsx'],k+1)-_CPLIQ*jnp.log(temp/_TFREEZ)*jnp.maximum(0.,excess-_get(st['xsh2o'],k+1))
                previous=_get(st['dsf'],k+1)
                dsf=_get(st['dsf'],k)
                dsf=_where((temp<=_TFREEZ)&(previous==0.),
                    (_LF/temp)*jnp.maximum(0.,_get(st['qtmix'],k)-qs-excess),dsf)
                dsf=_where((temp<=_TFREEZ)&(previous!=0.),
                    previous+(_LF/temp)*jnp.maximum(0.,_get(st['qsmix'],k+1)-qs),dsf)
                new_s=_get(st['smix'],k)+dsx+dsf
                new_q=_get(st['qtmix'],k)-excess
                temp,qs,failed=_ientropy(new_s,_get(p,k),new_q,temp)
                st.update(xsh2o=_set(st['xsh2o'],k,excess),dsx=_set(st['dsx'],k,dsx),
                          dsf=_set(st['dsf'],k,dsf),tmix=_set(st['tmix'],k,temp),
                          qsmix=_set(st['qsmix'],k,qs),failed=st['failed']|failed)
                return st
            st=_fori(0,2,latent_pass,st)
            st=dict(st)
            temp=_get(st['tmix'],k)
            new_q=_get(st['qtmix'],k)-_get(st['xsh2o'],k)
            qv=jnp.minimum(new_q,_get(st['qsmix'],k))
            st.update(tp=_set(st['tp'],k,temp),qstp=_set(st['qstp'],k,qv),
                      tpv=_set(st['tpv'],k,(temp+tpert)*(1.+1.608*qv)/(1.+new_q)))
            return st
        return _cond(k==launch,at_launch,
                            lambda st:_cond(k<launch,above,lambda st:st,st),st)
    state=_fori(0,kx-msg,second_step,state)
    return state


def _buoyan_dilute(q,t,p,z,pf,pblt,tpert,msg,kx):
    """Original dilute plume launch and ordered five-equilibrium CAPE search."""
    def launch_step(i,st):
        k=kx-i;mx,hmax=st
        hmn=_CP*_get(t,k)+_G*_get(z,k)+_LV*_get(q,k)
        yes=(k>=jnp.floor(pblt+.5).astype(jnp.int32))&(hmn>hmax)
        return _where(yes,k,mx),_where(yes,hmn,hmax)
    launch,_=_fori(0,kx-msg,launch_step,(jnp.int32(kx),jnp.float64(0.)))
    parcel=_parcel_dilute(p,t,q,tpert,launch,msg,kx)
    idx=jnp.arange(t.shape[0],dtype=jnp.int32)
    tv=t*(1.+1.608*q)/(1.+q)
    eligible=(idx<=launch)&(parcel['pl']>=600.)&(idx>msg)&(idx<=kx)
    buoy=_where(eligible,parcel['tpv']-tv+.5,0.)
    tp=_where(eligible,parcel['tp'],t)
    qs=_where(eligible,parcel['qstp'],q)
    def equilibrium(k,st):
        count,tops=st
        yes=(k<parcel['lcl'])&(parcel['pl']>=600.)&(_get(buoy,k+1)>0.)&(_get(buoy,k)<=0.)
        count=_where(yes,jnp.minimum(5,count+1),count)
        return count,_where(yes,_set(tops,count,k),tops)
    _,tops=_fori(msg+2,kx+1,equilibrium,
                            (jnp.int32(0),jnp.full((8,),kx,jnp.int32)))
    def cape_for(n,st):
        cape,lel=st;top=_get(tops,n)
        def add(k,total):
            yes=(parcel['pl']>=600.)&(k<=launch)&(k>top)
            return total+_where(yes,_R*_get(buoy,k)*jnp.log(_get(pf,k+1)/_get(pf,k)),0.)
        capen=_fori(msg+1,kx+1,add,jnp.float64(0.))
        return _where(capen>cape,capen,cape),_where(capen>cape,top,lel)
    cape,lel=_fori(1,6,cape_for,(jnp.float64(0.),jnp.int32(kx)))
    return dict(tp=tp,qstp=qs,tl=parcel['tl'],cape=jnp.maximum(cape,0.),
                lcl=parcel['lcl'],lel=lel,lon=jnp.int32(kx),mx=launch,failed=parcel['failed'])


def _cldprp(q,t,p,z,s,zf,shat,*,jb,lel,mx,limcnv,landfrac,msg,kx):
    """Original:2114–2779, normalized up/down plume properties and rain."""
    idx=jnp.arange(t.shape[0],dtype=jnp.int32)
    lev=(idx>=1)&(idx<=kx)
    zero=jnp.zeros_like(t)
    dz=zf-_roll(zf,-1)
    c0=.045*(1.-landfrac)+.0059*landfrac
    est=6.112*jnp.exp((17.67*(t-_TFREEZ))/((t-_TFREEZ)+243.5))
    qst=_where(p-est>0.,_EPS*est/(p-est),1.)
    gamma=qst*(1.+qst/_EPS)*_EPS*_LV/(_R*t**2)*_LV/_CP
    hmn=_CP*t+_G*z+_LV*q
    hsat=_CP*t+_G*z+_LV*qst
    dq=_roll(qst,1)-qst
    qsthat=_where(jnp.abs(dq)>1.e-6,jnp.log(_roll(qst,1)/qst)*_roll(qst,1)*qst/dq,qst)
    dg=_roll(gamma,1)-gamma
    gamhat=_where(jnp.abs(dg)>1.e-6,jnp.log(_roll(gamma,1)/gamma)*_roll(gamma,1)*gamma/dg,gamma)
    hsthat=_CP*shat+_LV*qsthat
    qsthat=_set(qsthat,msg+1,_get(qst,msg+1))
    hsthat=_set(hsthat,msg+1,_get(hsat,msg+1))
    gamhat=_set(gamhat,msg+1,_get(gamma,msg+1))
    jt=jnp.minimum(jnp.maximum(lel,limcnv+1),kx)
    def minimum(k,carry):
        hmin,j0=carry
        yes=(_get(hsat,k)<=hmin)&(k>=jt)&(k<=jb)
        return _where(yes,_get(hsat,k),hmin),_where(yes,k,j0)
    _,j0=_fori(msg+1,kx+1,minimum,(jnp.float64(1.e6),jnp.int32(0)))
    j0=jnp.minimum(jnp.maximum(jnp.minimum(j0,jb-2),jt+2),kx)
    hu=_where(lev&(idx>=jt)&(idx<=jb),_get(hmn,mx)+_CP*.5,hmn)
    su=_where(lev&(idx>=jt)&(idx<=jb),_get(s,mx)+.5,s)
    def moments(i,values):
        k=kx-1-i;i1,i2,i3,i4=values
        yes=(k<jb)&(k>=jt)
        a=_get(i1,k+1)+(_get(hmn,mx)-_get(hmn,k))*_get(dz,k)
        ihat=.5*(_get(i1,k+1)+a)
        b=_get(i2,k+1)+ihat*_get(dz,k)
        idag=.5*(_get(i2,k+1)+b)
        c=_get(i3,k+1)+idag*_get(dz,k)
        iprm=.5*(_get(i3,k+1)+c)
        d=_get(i4,k+1)+iprm*_get(dz,k)
        return tuple(_set(arr,k,_where(yes,val,_get(arr,k))) for arr,val in zip(values,(a,b,c,d),strict=True))
    i1,i2,i3,i4=_fori(0,kx-msg-1,moments,(zero,zero,zero,zero))
    def difference(k,carry):
        hmin,expdif=carry
        yes=(k>=j0)&(k<=jb)&(_get(hmn,k)<=hmin)
        hmin=_where(yes,_get(hmn,k),hmin)
        return hmin,_where(yes,_get(hmn,mx)-hmin,expdif)
    _,expdif=_fori(msg+1,kx+1,difference,(jnp.float64(1.e6),jnp.float64(0.)))
    inside=lev&(idx>=jt)&(idx<jb)&(idx>=msg+2)
    i1=_where(inside,i1,0.)
    expnum=_get(hmn,mx)-( _roll(hsat,1)*(zf-z)+hsat*(_roll(z,1)-zf))/(_roll(z,1)-z)
    expnum=_where(inside,expnum,0.)
    yes=(expdif>100.)&(expnum>0.)&(i1>expnum*dz)&(idx>=msg+2)&lev
    ft=expnum/i1
    f=ft+i2/i1*ft**2+(2.*i2**2-i1*i3)/i1**2*ft**3+(-5.*i1*i2*i3+5.*i2**3+i1**2*i4)/i1**3*ft**4
    f=_where(yes,jnp.minimum(jnp.maximum(f,0.),.0002),0.)
    j0=_where((j0<jb)&(_get(f,j0)<1.e-6)&(_get(f,j0+1)>_get(f,j0)),j0+1,j0)
    def monotone(k,f):
        return _set(f,k,_where((k>=jt)&(k<=j0),jnp.maximum(_get(f,k),_get(f,k-1)),_get(f,k)))
    f=_fori(msg+2,kx+1,monotone,f)
    eps0=_get(f,j0)
    eps=_where(lev&(idx>=jt)&(idx<=jb),_where(idx>=j0,eps0,f),0.)
    enabled=eps0>0.
    mu=_set(zero,jb,_where(enabled,1.,0.))
    eu=_set(zero,jb,_where(enabled,_get(mu,jb)/_get(dz,jb),0.))
    zuef=zf-_get(zf,jb)
    rmue=(1./eps0)*(jnp.exp(_roll(eps,-1)*zuef)-1.)/zuef
    raw_mu=(1./eps0)*(jnp.exp(eps*zuef)-1.)/zuef
    # Source's ascending-in-height mass flux is needed by the following EU.
    def upflux(i,carry):
        k=kx-i;mu,eu,du=carry
        yes=(k>=jt)&(k<jb)&enabled
        value=_get(raw_mu,k)
        euv=(_get(rmue,k)-_get(mu,k+1))/_get(dz,k)
        duv=(_get(rmue,k)-value)/_get(dz,k)
        return (_set(mu,k,_where(yes,value,_get(mu,k))),
                _set(eu,k,_where(yes,euv,_get(eu,k))),_set(du,k,_where(yes,duv,_get(du,k))))
    mu,eu,du=_fori(0,kx-msg,upflux,(mu,eu,zero))
    def upenergy(i,carry):
        k=jb-1-i;hu,mu,eu,du=carry
        yes=(k>=lel)&enabled
        tiny=_get(mu,k)<.01
        hvalue=_get(mu,k+1)/_get(mu,k)*_get(hu,k+1)+_get(dz,k)/_get(mu,k)*(_get(eu,k)*_get(hmn,k)-_get(du,k)*_get(hsat,k))
        hu=_set(hu,k,_where(yes,_where(tiny,_get(hu,jb),hvalue),_get(hu,k)))
        du=_set(du,k,_where(yes&tiny,_get(mu,k+1)/_get(dz,k),_get(du,k)))
        mu=_set(mu,k,_where(yes&tiny,0.,_get(mu,k)))
        eu=_set(eu,k,_where(yes&tiny,0.,_get(eu,k)))
        return hu,mu,eu,du
    hu,mu,eu,du=_fori(0,jb-lel,upenergy,(hu,mu,eu,du))
    def top_step(i,carry):
        k=jb-2-i;jt,doit=carry
        cross=(_get(hu,k)<=_get(hsthat,k))&(_get(hu,k+1)>_get(hsthat,k+1))&(_get(mu,k)>=.02)
        low=_get(hu,k)-_get(hsthat,k)<-2000.
        other=(_get(hu,k)>_get(hu,jb))|(_get(mu,k)<.01)
        active=doit&(k>=lel-1)
        jt=_where(active,_where(cross,_where(low,k+1,k),_where(other,k+1,jt)),jt)
        doit=doit&~(active&(_where(cross,low|(eps0<=0.),other)))
        return jt,doit
    jt,_=_fori(0,jb-lel,top_step,(jt,jnp.array(True)))
    clear=lev&(idx>=lel)&(idx<=jt)&enabled
    hu=_where(clear,_get(hu,jb),hu)
    mu=_where(clear,0.,mu);eu=_where(clear,0.,eu);du=_where(clear,0.,du)
    du=_set(du,jt,_where(enabled,_get(mu,jt+1)/_get(dz,jt),_get(du,jt)))
    jt=jnp.minimum(jt,jb-1)
    jd=jnp.minimum(jnp.maximum(j0,jt+1),jb)
    hd=_set(hmn,jd,_get(hmn,jd-1))
    md=_set(zero,jd,_where((jd<jb)&enabled,-.1*eps0/eps0,0.))
    zdef=_get(zf,jd)-zf
    raw_md=-.1/(2.*eps0)*(jnp.exp(2.*eps0*zdef)-1.)/zdef
    md=_where(lev&(idx>jd)&(idx<=jb)&enabled,raw_md,md)
    ratio=jnp.minimum(jnp.abs(_get(mu,jb)/_get(md,jb)),1.)
    md=_where(lev&(idx>=jt)&(idx<=jb)&enabled&(jd<jb),md*ratio,md)
    def downenergy(k,carry):
        ed,hd=carry;yes=(k>=jt)&enabled
        entrain=(_get(md,k-1)-_get(md,k))/_get(dz,k-1)
        enthalpy=(_get(md,k-1)*_get(hd,k-1)-_get(dz,k-1)*entrain*_get(hmn,k-1))/jnp.minimum(_get(md,k),-1.e-20)
        return (_set(ed,k-1,_where(yes,entrain,_get(ed,k-1))),
                _set(hd,k,_where(yes,enthalpy,_get(hd,k))))
    ed,hd=_fori(msg+1,kx+1,downenergy,(zero,hd))
    qds=_where(lev&(idx>=msg+2)&(idx>=jd)&(idx<=jb)&enabled&(jd<jb),
                qsthat+gamhat*(hd-hsthat)/(_LV*(1.+gamhat)),q)
    def unsat_step(i,carry):
        k=kx-i;su,qu,jlcl,done=carry
        yes=~done&(k>jt)&(k<jb)&enabled
        sun=_get(mu,k+1)/_get(mu,k)*_get(su,k+1)+_get(dz,k)/_get(mu,k)*(_get(eu,k)-_get(du,k))*_get(s,k)
        qun=_get(mu,k+1)/_get(mu,k)*_get(qu,k+1)+_get(dz,k)/_get(mu,k)*(_get(eu,k)*_get(q,k)-_get(du,k)*_get(qst,k))
        temp=sun-_G/_CP*_get(zf,k)
        es=6.112*jnp.exp((17.67*(temp-_TFREEZ))/((temp-_TFREEZ)+243.5))
        qs=_EPS*es/((_get(p,k)+_get(p,k-1))/2.-es)
        stop=yes&(qun>=qs)
        return (_set(su,k,_where(yes,sun,_get(su,k))),_set(qu,k,_where(yes,qun,_get(qu,k))),
                _where(stop,k,jlcl),done|stop)
    su,qu,jlcl,_=_fori(0,kx-msg-1,unsat_step,(su,q,lel,jnp.array(False)))
    qu=_set(qu,jb,_where((jb>=msg+2)&enabled,_get(q,mx),_get(qu,jb)))
    su=_set(su,jb,_where((jb>=msg+2)&enabled,(_get(hu,jb)-_LV*_get(qu,jb))/_CP,_get(su,jb)))
    sat=lev&(idx>=msg+2)&(idx>jt)&(idx<=jlcl)&enabled
    su=_where(sat,shat+(hu-hsthat)/(_CP*(1.+gamhat)),su)
    qu=_where(sat,qsthat+gamhat*(hu-hsthat)/(_LV*(1.+gamhat)),qu)
    cu=((mu*su-_roll(mu,-1)*_roll(su,-1))/dz-(eu-du)*s)/(_LV/_CP)
    cu=_where(lev&(idx>=msg+2)&(idx>jt)&(idx<jb)&enabled,jnp.maximum(0.,cu),0.)
    def rain_step(i,carry):
        k=kx-i;ql,rprd,tot=carry
        yes=(k>=jt)&(k<jb)&enabled&(_get(mu,k)>=0.)
        ql1=1./_get(mu,k)*(_get(mu,k+1)*_get(ql,k+1)-_get(dz,k)*_get(du,k)*_get(ql,k+1)+_get(dz,k)*_get(cu,k))
        qln=_where(_get(mu,k)>0.,ql1/(1.+_get(dz,k)*c0),0.)
        tot=tot+_where(yes,_get(dz,k)*(_get(cu,k)-_get(du,k)*_get(ql,k+1)),0.)
        return (_set(ql,k,_where(yes,qln,_get(ql,k))),
                _set(rprd,k,_where(yes,c0*_get(mu,k)*qln,0.)),tot)
    ql,rprd,totpcp=_fori(0,kx-msg-1,rain_step,(zero,zero,jnp.float64(0.)))
    qd=_set(q,jd,_get(qds,jd));sd=_set(s,jd,(_get(hd,jd)-_LV*_get(qd,jd))/_CP)
    def evaporate(k,carry):
        qd,sd,evp,tot=carry;yes=(k>=jd)&(k<jb)&enabled
        qdn=_get(qds,k+1)
        rate=-_get(ed,k)*_get(q,k)+(_get(md,k)*_get(qd,k)-_get(md,k+1)*qdn)/_get(dz,k)
        rate=jnp.maximum(rate,0.)
        sdn=((_LV/_CP*rate-_get(ed,k)*_get(s,k))*_get(dz,k)+_get(md,k)*_get(sd,k))/jnp.minimum(_get(md,k+1),-1.e-20)
        return (_set(qd,k+1,_where(yes,qdn,_get(qd,k+1))),_set(sd,k+1,_where(yes,sdn,_get(sd,k+1))),
                _set(evp,k,_where(yes,rate,_get(evp,k))),tot-_where(yes,_get(dz,k)*_get(ed,k)*_get(q,k),0.))
    qd,sd,evp,totevp=_fori(msg+2,kx+1,evaporate,(qd,sd,zero,jnp.float64(0.)))
    totevp=totevp+_get(md,jd)*_get(qd,jd)-_get(md,jb)*_get(qd,jb)
    totpcp,totevp=jnp.maximum(totpcp,0.),jnp.maximum(totevp,0.)
    scale=_where((totevp>0.)&(totpcp>0.),jnp.minimum(1.,totpcp/(totevp+totpcp)),0.)
    lower=lev&(idx>=msg+2)
    md=_where(lower,md*scale,md);ed=_where(lower,ed*scale,ed);evp=_where(lower,evp*scale,evp)
    cmeg=_where(lower,cu-evp,0.);rprd=_where(lower,rprd-evp,rprd)
    pflx=_fori(2,kx+2,lambda k,a:_set(a,k,_get(a,k-1)+_get(rprd,k-1)*_get(dz,k-1)),zero)
    return dict(mu=mu,eu=eu,du=du,md=md,ed=ed,sd=sd,qd=qd,mc=_where(lev&(idx>msg),mu+md,0.),
                qu=qu,su=su,qst=qst,hmn=hmn,hsat=hsat,ql=ql,cmeg=cmeg,pflx=pflx,
                evp=evp,cu=cu,rprd=rprd,jt=jt,jlcl=jlcl,j0=j0,jd=jd)


def _closure(q,t,p,z,s,parcel,props,qhat,shat,dp,zf,*,dsubcld,tau,msg,kx):
    """Source:2781–2996, pressure-coordinate mass-flux closure."""
    tp,qstp,tl=parcel['tp'],parcel['qstp'],parcel['tl']
    lcl,lel,mx=parcel['lcl'],parcel['lel'],parcel['mx']
    mu,md,mc,du,qu,su,qd,sd,ql=(props[k] for k in ('mu','md','mc','du','qu','su','qd','sd','ql'))
    jt=props['jt']
    idx=jnp.arange(t.shape[0],dtype=jnp.int32)
    lev=(idx>msg)&(idx<=kx)
    eb=_get(p,mx)*_get(q,mx)/(_EPS+_get(q,mx))
    dtb=(1./dsubcld)*(_get(mu,mx)*(_get(shat,mx)-_get(su,mx))+_get(md,mx)*(_get(shat,mx)-_get(sd,mx)))
    dqb=(1./dsubcld)*(_get(mu,mx)*(_get(qhat,mx)-_get(qu,mx))+_get(md,mx)*(_get(qhat,mx)-_get(qd,mx)))
    deb=_EPS*_get(p,mx)/(_EPS+_get(q,mx))**2*dqb
    dtl=-2840.*(3.5/_get(t,mx)*dtb-deb/eb)/(3.5*jnp.log(_get(t,mx))-jnp.log(eb)-4.805)**2
    top=lev&(idx==jt)&(idx<kx)
    dtm=_where(top,(1./dp)*(_roll(mu,-1)*(_roll(su,-1)-_roll(shat,-1)-_LV/_CP*_roll(ql,-1))
                          +_roll(md,-1)*(_roll(sd,-1)-_roll(shat,-1))),0.)
    dqm=_where(top,(1./dp)*(_roll(mu,-1)*(_roll(qu,-1)-_roll(qhat,-1)+_roll(ql,-1))
                          +_roll(md,-1)*(_roll(qd,-1)-_roll(qhat,-1))),0.)
    inside=lev&(idx>jt)&(idx<mx)&(idx<kx)
    beta=jnp.float64(0.)
    qld=beta*ql+(1.-beta)*_roll(ql,-1)
    dtm=_where(inside,(mc*(shat-s)+_roll(mc,-1)*(s-_roll(shat,-1)))/dp-_LV/_CP*du*qld,dtm)
    dqm=_where(inside,(_roll(mu,-1)*(_roll(qu,-1)-_roll(qhat,-1)+_CP/_LV*(_roll(su,-1)-s))
                      -mu*(qu-qhat+_CP/_LV*(su-s))
                      +_roll(md,-1)*(_roll(qd,-1)-_roll(qhat,-1)+_CP/_LV*(_roll(sd,-1)-s))
                      -md*(qd-qhat+_CP/_LV*(sd-s)))/dp+du*qld,dqm)
    exner=(1000./p)**(_R/_CP)
    thetap=tp*exner*(1.+1.608*qstp-_get(q,mx))
    thetam=t*exner*(1.+.608*q)
    dqs=qstp*(1.+qstp/_EPS)*_EPS*_LV/(_R*tp**2)
    dtp=tp/(1.+_LV/_CP*(dqs-qstp/tp))*(dtb/_get(t,mx)+_LV/_CP*(dqb/tl-_get(q,mx)/tl**2*dtl))
    saturated=((dtp/tp+1./(1.+1.608*qstp-_get(q,mx))*(1.608*dqs*dtp-dqb))
               -(dtm/t+.608/(1.+.608*q)*dqm))*_G*thetap/thetam
    # Original:2857 has unassigned DBOYDT outside both physical branches;
    # initialize those precise sites to zero under the standing policy.
    dboy=_where(lev&(idx>=lel)&(idx<=lcl),saturated,0.)
    thetap=tp*exner*(1.+.608*_get(q,mx))
    thetam=t*exner*(1.+.608*q)
    unsat=(dtb/_get(t,mx)+.608/(1.+.608*_get(q,mx))*dqb-dtm/t-.608/(1.+.608*q)*dqm)*_G*thetap/thetam
    dboy=_where(lev&(idx>lcl)&(idx<mx),unsat,dboy)
    dadt=_fori(lel,mx,lambda k,a:a+_get(dboy,k)*(_get(zf,k)-_get(zf,k+1)),jnp.float64(0.))
    dltaa=-1.*(parcel['cape']-70.)
    return _where(dadt!=0.,jnp.maximum(dltaa/tau/dadt,0.),0.)


def _q1q2(q,qhat,shat,dp,props,*,dsubcld,mx,msg,kx):
    """Source:2998–3124; cloud-source thermal/moisture rates and detrainment."""
    mu,md,su,sd,qu,qd,ql,du,cu,evp=(props[k] for k in ('mu','md','su','sd','qu','qd','ql','du','cu','evp'))
    idx=jnp.arange(q.shape[0],dtype=jnp.int32)
    active=(idx>=props['jt'])&(idx<kx)&(idx>msg)
    emc=-cu+evp
    dsdt=_where(active,-_LV/_CP*emc+(_roll(mu,-1)*(_roll(su,-1)-_roll(shat,-1))
                                   -mu*(su-shat)+_roll(md,-1)*(_roll(sd,-1)-_roll(shat,-1))-md*(sd-shat))/dp,0.)
    dqdt=_where(active,emc+(_roll(mu,-1)*(_roll(qu,-1)-_roll(qhat,-1))
                          -mu*(qu-qhat)+_roll(md,-1)*(_roll(qd,-1)-_roll(qhat,-1))-md*(qd-qhat))/dp,0.)
    dl=_where(active,du*_roll(ql,-1),0.)
    dsdt=_set(dsdt,mx,(1./dsubcld)*(-_get(mu,mx)*(_get(su,mx)-_get(shat,mx))
                                  -_get(md,mx)*(_get(sd,mx)-_get(shat,mx))))
    dqdt=_set(dqdt,mx,(1./dsubcld)*(-_get(mu,mx)*(_get(qu,mx)-_get(qhat,mx))
                                  -_get(md,mx)*(_get(qd,mx)-_get(qhat,mx))))
    def below(k,rates):
        a,b=rates
        return _set(a,k,_get(a,k-1)),_set(b,k,_get(b,k-1))
    dsdt,dqdt=_fori(mx+1,kx+1,below,(dsdt,dqdt))
    return dqdt,dsdt,dl


def _convr(t,q,pap,paph,dpp,zm,zi,*,geos,pblh,tpert,landfrac,dt,dx,kx,limcnv=2,no_deep_pbl=True):
    """Original:272–894, one-column gathering is the identity permutation."""
    msg=limcnv-1
    idx=jnp.arange(t.shape[0],dtype=jnp.int32)
    lev=(idx>msg)&(idx<=kx)
    zero=jnp.zeros_like(t)
    zs=geos*(1./_G)
    p=pap*.01
    pf=paph*.01
    z=zm+zs
    zf=zi+zs
    def pbl_index(i,pblt):
        k=kx-1-i
        yes=jnp.abs(_get(z,k)-zs-pblh)<(_get(zf,k)-_get(zf,k+1))*.5
        return _where(yes,k,pblt)
    pblt=_fori(0,kx-msg-1,pbl_index,jnp.int32(kx))
    s=t+(_G/_CP)*z
    parcel=_buoyan_dilute(q,t,p,z,pf,pblt,tpert,msg,kx)
    float_fields=('qtnd','heat','mcon','dlf','pflx','cme','rprd','zdu','ql','mu','md','du','eu','ed','dp')
    out={key:zero for key in float_fields}
    out.update(cape=parcel['cape'],prec=jnp.float64(0.),rliq=jnp.float64(0.),
               jctop=jnp.float64(kx),jcbot=jnp.float64(1.),dsubcld=jnp.float64(0.),
               jt=jnp.int32(0),maxg=jnp.int32(1),lengath=jnp.int32(0),failed=parcel['failed'])
    def active(out):
        dp=.01*dpp
        mx=parcel['mx']
        dsub=_fori(msg+1,kx+1,lambda k,a:a+_where(k>=mx,_get(dp,k),0.),jnp.float64(0.))
        def logmean(a):
            previous=_roll(a,1)
            relative=_where((a>0.)|(previous>0.),jnp.abs((a-previous)/jnp.maximum(previous,a)),0.)
            result=_where(relative>1.e-6,jnp.log(previous/a)*previous*a/(previous-a),.5*(a+previous))
            return _where((idx>=msg+2)&(idx<=kx),result,a)
        shat,qhat=logmean(s),logmean(q)
        props=_cldprp(q,t,p,z,s,zf,shat,jb=mx,lel=parcel['lel'],mx=mx,
                      limcnv=limcnv,landfrac=landfrac,msg=msg,kx=kx)
        props=dict(props)
        for key in ('du','eu','ed','cu','cmeg','rprd','evp'):
            props[key]=_where(lev,props[key]*(zf-_roll(zf,-1))/dp,props[key])
        tau=jnp.maximum(600.,3600.*jnp.minimum(1.,dx/275000.))
        mb=_closure(q,t,p,z,s,parcel,props,qhat,shat,dp,zf,dsubcld=dsub,tau=tau,msg=msg,kx=kx)
        mumax=jnp.max(_where((idx>=msg+2)&(idx<=kx),props['mu']/dp,0.))
        mb=_where(mumax>0.,jnp.minimum(mb,.5/(dt*mumax)),0.)
        if no_deep_pbl:
            mb=_where(_get(zm,props['jt'])<pblh,0.,mb)
        for key in ('mu','md','mc','du','eu','ed','cmeg','rprd','cu','evp'):
            props[key]=_where(lev,props[key]*mb,props[key])
        props['pflx']=_where((idx>=msg+2)&(idx<=kx+1),props['pflx']*mb*100./_G,props['pflx'])
        dqdt,dsdt,dl=_q1q2(q,qhat,shat,dp,props,dsubcld=dsub,mx=mx,msg=msg,kx=kx)
        # Original:850 stores q before subtracting it for the water budget;
        # keep both roundings rather than replacing the difference by a rate.
        new_q=_where(lev,q+2.*dt*dqdt,q)
        def precipitation(i,acc):
            k=kx-i
            return acc-_get(dpp,k)*(_get(new_q,k)-_get(q,k))-_get(dpp,k)*_get(dl,k)*2.*dt
        prec=_fori(0,kx-msg,precipitation,jnp.float64(0.))
        prec=(1./_G)*jnp.maximum(prec,0.)/(2.*dt)/1000.
        rliq=_fori(1,kx+1,lambda k,a:a+_get(dl,k)*_get(dpp,k)/_G,jnp.float64(0.))/1000.
        result=dict(out)
        result.update(qtnd=dqdt,heat=dsdt*_CP,mcon=props['mc'],dlf=dl,pflx=props['pflx'],
                      cme=props['cmeg'],rprd=props['rprd'],zdu=props['du'],ql=props['ql'],
                      prec=prec,rliq=rliq,jctop=props['jt'].astype(jnp.float64),
                      jcbot=mx.astype(jnp.float64),dsubcld=dsub,jt=props['jt'],maxg=mx,lengath=jnp.int32(1),
                      mu=props['mu'],md=props['md'],du=props['du'],eu=props['eu'],ed=props['ed'],dp=dp)
        return result
    return _cond(parcel['cape']>70.,active,lambda out:out,out)


def _aqsat(t,p,estbl):
    """Original CAM aqsat/estblf; shared original Goff–Gratch table input."""
    bounded=jnp.maximum(jnp.minimum(t,375.16),173.16)
    ai=jnp.floor(bounded-173.16)
    index=ai.astype(jnp.int32)
    es=(173.16+ai-bounded+1.)*estbl[index]-(173.16+ai-bounded)*estbl[index+1]
    qs=_EPS*es/(p-(1.-_EPS)*es)
    qs=jnp.minimum(1.,qs)
    return _where(qs<0.,1.,qs)


def _conv_evap(t,p,dp,q,prdprec,cldfra,prec,dt,estbl,kx):
    """Source:897–1079, precipitation evaporation, snow and latent heating."""
    qs=_aqsat(t,p,estbl)
    fsnow=_where(t>_TFREEZ,0.,_where(t<_TFREEZ-5.,1.,(_TFREEZ-t)/5.))
    zero=jnp.zeros_like(t)
    state=dict(flxprec=zero,flxsnow=zero,ntprprd=zero,ntsnprd=zero,
               snow_heat=zero,melt_heat=zero,tend_s=zero,tend_q=zero,evpvint=jnp.float64(0.))
    total=prec*1000.
    # Original zmconv_readnl:138 uses unsuffixed REAL ke=1.0E-6, promoted.
    ke=jnp.float32(1.e-6).astype(jnp.float64)
    def layer(k,st):
        st=dict(st)
        rain=_get(st['flxprec'],k);snow=_get(st['flxsnow'],k)
        warm=_get(t,k)>_TFREEZ
        flxsntm=_where(warm,0.,snow)
        melt=_where(warm,snow*_G/_get(dp,k),0.)
        limit=jnp.maximum(1.-_get(q,k)/_get(qs,k),0.)
        evap=ke*(1.-_get(cldfra,k))*limit*jnp.sqrt(rain)
        limit=jnp.maximum(0.,(_get(qs,k)-_get(q,k))/dt)
        limit=jnp.minimum(limit,rain*_G/_get(dp,k))
        limit=jnp.minimum(limit,(total-st['evpvint'])*_G/_get(dp,k))
        evap=jnp.minimum(limit,evap)
        snow_evap=_where(rain>0.,evap*jnp.minimum(jnp.maximum(0.,flxsntm/rain),1.),0.)
        st['evpvint']=st['evpvint']+evap*_get(dp,k)/_G
        net=_get(prdprec,k)-evap
        snow_fraction=_where(rain>0.,jnp.minimum(jnp.maximum(0.,snow/rain),1.),0.)
        work2=jnp.maximum(_get(fsnow,k),snow_fraction)
        work2=_where(melt>0.,0.,work2)
        snet=_get(prdprec,k)*work2-snow_evap-melt
        st['flxprec']=_set(st['flxprec'],k+1,jnp.maximum(rain+net*_get(dp,k)/_G,0.))
        st['flxsnow']=_set(st['flxsnow'],k+1,jnp.maximum(snow+snet*_get(dp,k)/_G,0.))
        for key,value in (('ntprprd',net),('ntsnprd',snet),
                          ('snow_heat',_get(prdprec,k)*work2*_LF),('melt_heat',-(snow_evap+melt)*_LF),
                          ('tend_s',-evap*_LV+snet*_LF),('tend_q',evap)):
            st[key]=_set(st[key],k,value)
        return st
    state=_fori(1,kx+1,layer,state)
    state.update(prec=_get(state['flxprec'],kx+1)/1000.,snow=_get(state['flxsnow'],kx+1)/1000.)
    return state


def _momtran(u,v,props,dt,kx):
    """Source:1382–1795, two wind tracers and kinetic-energy conversion."""
    mu,md,du,eu,ed,dp=(props[key] for key in ('mu','md','du','eu','ed','dp'))
    jt,mx=props['jt'],props['maxg']
    idx=jnp.arange(u.shape[0],dtype=jnp.int32)
    lev=(idx>=1)&(idx<=kx)
    zero=jnp.zeros_like(u)
    fields=[]
    for wind in (u,v):
        previous=_where(idx==1,wind,_roll(wind,1))
        chat=.5*(wind+previous)
        gradient_u=mu*(wind-previous)/_roll(dp,1)+_roll(mu,-1)*(_roll(wind,-1)-wind)/dp
        gradient_d=md*(wind-previous)/_roll(dp,1)+_roll(md,-1)*(_roll(wind,-1)-wind)/dp
        pgu=_where(lev&(idx>=2)&(idx<kx),-.4*.5*gradient_u,0.)
        pgd=_where(lev&(idx>=2)&(idx<kx),-.4*.5*gradient_d,0.)
        last_u=_get(mu,kx)*(_get(wind,kx)-_get(wind,kx-1))/_get(dp,kx-1)
        last_d=_get(md,kx)*(_get(wind,kx)-_get(wind,kx-1))/_get(dp,kx-1)
        pgu=_set(pgu,kx,-.4*last_u)
        pgd=_set(pgd,kx,-.4*last_d)
        base=_get(mu,kx)+_get(du,kx)*_get(dp,kx)
        conu=_set(chat,kx,_where(base>1.e-15,
            (_get(eu,kx)*_get(wind,kx)*_get(dp,kx)+_get(pgu,kx)*_get(dp,kx))/base,_get(chat,kx)))
        # Original:1633 divides only the pressure-gradient term by MD(2).
        # Retain its literal parentheses rather than rewriting the equation.
        cond=_set(chat,2,_where(_get(md,2)<-1.e-15,
            (-_get(ed,1)*_get(wind,1)*_get(dp,1))-_get(pgd,1)*_get(dp,1)/_get(md,2),_get(chat,2)))
        def up(i,a):
            k=kx-1-i
            mass=_get(mu,k)+_get(du,k)*_get(dp,k)
            value=(_get(mu,k+1)*_get(a,k+1)+_get(eu,k)*_get(wind,k)*_get(dp,k)+_get(pgu,k)*_get(dp,k))/mass
            return _set(a,k,_where(mass>1.e-15,value,_get(a,k)))
        conu=_fori(0,kx-1,up,conu)
        def down(k,a):
            value=(_get(md,k-1)*_get(a,k-1)-_get(ed,k-1)*_get(wind,k-1)*_get(dp,k-1)-_get(pgd,k-1)*_get(dp,k-1))/_get(md,k)
            return _set(a,k,_where(_get(md,k)<-1.e-15,value,_get(a,k)))
        cond=_fori(3,kx+1,down,cond)
        kp=jnp.minimum(idx+1,kx)
        rate=(_get_vector(mu,kp)*(_get_vector(conu,kp)-_get_vector(chat,kp))-mu*(conu-chat)
              +_get_vector(md,kp)*(_get_vector(cond,kp)-_get_vector(chat,kp))-md*(cond-chat))/dp
        rate=_where(lev&(idx>=jt),rate,0.)
        rate=_set(rate,mx,(1./_get(dp,mx))*(-_get(mu,mx)*(_get(conu,mx)-_get(chat,mx))-_get(md,mx)*(_get(cond,mx)-_get(chat,mx))))
        mflux=_where(lev&(idx>=jt),-mu*(conu-chat)-md*(cond-chat),0.)
        final=_where(lev&(idx>=jt),wind-(_roll(mflux,-1)-mflux)*dt/dp,0.)
        fields.append(dict(rate=rate,pgu=-pgu,pgd=-pgd,conu=conu,cond=cond,mflux=mflux,final=final))
    x,y=fields
    uprev=_where(idx==1,u,_roll(u,1));vprev=_where(idx==1,v,_roll(v,1))
    unext=_get_vector(u,jnp.minimum(idx+1,kx));vnext=_get_vector(v,jnp.minimum(idx+1,kx))
    utop=(u+uprev)/2.;vtop=(v+vprev)/2.
    ubot=(unext+u)/2.;vbot=(vnext+v)/2.
    fket=utop*x['mflux']+vtop*y['mflux']
    fkeb=ubot*_roll(x['mflux'],-1)+vbot*_roll(y['mflux'],-1)
    ketend_cons=(fket-fkeb)/dp
    ketend=((x['final']**2+y['final']**2)-(u**2+v**2))*.5/dt
    return dict(u=x,v=y,seten=_where(lev&(idx>=jt),ketend_cons-ketend,0.))


def _get_vector(a,index):
    # Register gather in one column; vector indices are physically bounded.
    return _gather_p.bind(a,index)


def _convtran_moist(const,props,kx):
    """Original:1083–1378 for the two non-chemistry moist cloud tracers.

    WRF CAM_INIT registers CLDLIQ/CLDICE as wet mixing ratios; the caller
    supplies FRACIS=1. Other constituent types/chemistry are not this boundary.
    """
    mu,md,du,eu,ed,dp=(props[key] for key in ('mu','md','du','eu','ed','dp'))
    jt,mx=props['jt'],props['maxg']
    idx=jnp.arange(const.shape[0],dtype=jnp.int32)
    lev=(idx>=1)&(idx<=kx)
    previous=_where(idx==1,const,_roll(const,1))
    minimum=jnp.minimum(previous,const);maximum=jnp.maximum(previous,const)
    difference=_where(minimum<0.,0.,jnp.abs(const-previous)/jnp.maximum(maximum,1.e-36))
    cabv=jnp.maximum(previous,maximum*1.e-12)
    cbel=jnp.maximum(const,maximum*1.e-12)
    chat=_where(difference>1.e-6,jnp.log(cabv/cbel)/(cabv-cbel)*cabv*cbel,.5*(const+previous))
    mass=_get(mu,kx)+_get(du,kx)*_get(dp,kx)
    conu=_set(chat,kx,_where(mass>1.e-15,(_get(eu,kx)*1.*_get(const,kx)*_get(dp,kx))/mass,_get(chat,kx)))
    cond=_set(chat,2,_where(_get(md,2)<-1.e-15,(-_get(ed,1)*1.*_get(const,1)*_get(dp,1))/_get(md,2),_get(chat,2)))
    def up(i,a):
        k=kx-1-i
        mass=_get(mu,k)+_get(du,k)*_get(dp,k)
        value=(_get(mu,k+1)*_get(a,k+1)+_get(eu,k)*1.*_get(const,k)*_get(dp,k))/mass
        return _set(a,k,_where(mass>1.e-15,value,_get(a,k)))
    conu=_fori(0,kx-1,up,conu)
    def down(k,a):
        value=(_get(md,k-1)*_get(a,k-1)-_get(ed,k-1)*1.*_get(const,k-1)*_get(dp,k-1))/_get(md,k)
        return _set(a,k,_where(_get(md,k)<-1.e-15,value,_get(a,k)))
    cond=_fori(3,kx+1,down,cond)
    kp=jnp.minimum(idx+1,kx)
    below=_get_vector(const,kp)
    mu_next=_get_vector(mu,kp);md_next=_get_vector(md,kp);chat_next=_get_vector(chat,kp)
    fluxin=mu_next*_get_vector(conu,kp)+mu*jnp.minimum(chat,previous)-(md*cond+md_next*jnp.minimum(chat_next,below))
    fluxout=mu*conu+mu_next*jnp.minimum(chat_next,const)-(md_next*_get_vector(cond,kp)+md*jnp.minimum(chat,const))
    net=fluxin-fluxout
    net=_where(jnp.abs(net)<jnp.maximum(fluxin,fluxout)*1.e-12,0.,net)
    rate=_where(lev&(idx>=jt),net/dp,0.)
    fin=_get(mu,mx)*jnp.minimum(_get(chat,mx),_get(const,jnp.maximum(1,mx-1)))-_get(md,mx)*_get(cond,mx)
    fout=_get(mu,mx)*_get(conu,mx)-_get(md,mx)*jnp.minimum(_get(chat,mx),_get(const,mx))
    net=fin-fout
    net=_where(jnp.abs(net)<jnp.maximum(fin,fout)*1.e-12,0.,net)
    rate=_set(rate,mx,net/_get(dp,mx))
    return _where(lev&(idx<=mx),rate,0.)


def _tpert_myj(th,t,qv,dz,tke,ctrl,kx):
    """Original get_tpert:939–1014; the entire calculation is WRF REAL."""
    with jax.enable_x64(False):
        psfc,pblh,kpbl,hfx,qfx,ust,mavail,tsk,dx,u,v=ctrl
        kpbl=kpbl.astype(jnp.int32)
        cp,rd,g=jnp.float32(1004.5),jnp.float32(287.),jnp.float32(9.81)
        ep1=jnp.float32(jnp.float32(461.6)/rd-jnp.float32(1.))
        ep2=rd/jnp.float32(461.6)
        e1=jnp.float32(.6112)*jnp.exp(jnp.float32(17.67)*(tsk-jnp.float32(273.15))/(tsk-jnp.float32(29.65)))
        ps=psfc/jnp.float32(1000.)
        qs=ep2*e1/(ps-e1)
        thgb=tsk*(jnp.float32(100.)/ps)**(rd/cp)
        tskv=thgb*(jnp.float32(1.)+ep1*qs*mavail)
        tvcon=jnp.float32(1.)+ep1*_get(qv,1)
        thv=_get(th,1)*tvcon
        govrth=g/_get(th,1)
        rho=psfc/(rd*_get(t,1)*tvcon)
        flux=jnp.maximum(hfx/rho/cp+ep1*tskv*qfx/rho,jnp.float32(0.))
        vconv=(g/tsk*pblh*flux)**jnp.float32(.33)
        vsgd=jnp.float32(.32)*jnp.maximum(dx/jnp.float32(5000.)-jnp.float32(1.),jnp.float32(0.))**jnp.float32(.33)
        wspd=jnp.sqrt(u*u+v*v)
        wspd=jnp.sqrt(wspd*wspd+vconv*vconv+vsgd*vsgd)
        wspd=jnp.maximum(wspd,jnp.float32(.1))
        br2=govrth*(jnp.float32(.5)*_get(dz,1))*(thv-tskv)/(wspd*wspd)
        def unstable(_):
            ebrk=_fori(1,jnp.minimum(kpbl,kx+1)+1,lambda k,a:a+_get(tke,k),jnp.float32(0.))
            ebrk=ebrk/kpbl.astype(jnp.float32)
            wpert=jnp.maximum(jnp.float32(1.)*jnp.sqrt(ebrk),jnp.float32(1.e-6))
            return jnp.maximum(jnp.abs(hfx/rho/cp)*jnp.float32(1.)/wpert,jnp.float32(0.))
        value=_cond(br2<=0.,unstable,
            lambda _:jnp.maximum(hfx/rho/cp*jnp.float32(8.5)/ust,jnp.float32(0.)),None)
    return value.astype(jnp.float64)


_CAM_PROFILE=('RUCUTEN','RVCUTEN','RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN','RQCNCUTEN','RQINCUTEN',
              'ICWMRDP','DLF_OUT','MU_OUT','MD_OUT','ZMDT','ZMDQ','ZMDICE','ZMDLIQ','CMFMC','ZMFLXPRC','ZMFLXSNW')
_CAM_SCALAR=('RAINCV','PRATEC','CAPE_OUT','CUTOP','CUBOT','PCONVB','PCONVT','RLIQ_OUT',
             'ENTROPY_FAILED','DLF_CLIPPED','VAPOR_FLOORED')


def _camzm_cps(profiles,ctrl,estbl,*,kx,pbl):
    """Original CAMZM_DRIVER:REAL caller assignments around DOUBLE routines.

    Non-chemistry/non-CAMMGMP boundary. Caller cadence and held fields are
    separate. The surface and PBL pairing is validated before entering here.
    """
    t,qv,p,p8w,pi,u,v,z,zw,tke,th,qc,qi,cldfra,dz=profiles
    # Read only the live controls; the native caller appends an unused lane.
    # Explicit scalar indexing avoids an unsupported in-kernel array slice.
    dt,dx,cudt,step,psfc,ht,pblh,kpbl,hfx,qfx,ust,mavail,xland,tsk,tp_cam=(ctrl[i] for i in range(15))
    idx=jnp.arange(t.shape[0],dtype=jnp.int32)
    lev=(idx>=1)&(idx<=kx)
    r4=lambda a:jnp.asarray(a,jnp.float32)
    r8=lambda a:jnp.asarray(a,jnp.float64)
    def flip(a):
        return _where(lev,_get_vector(a,jnp.clip(kx-idx+1,0,a.shape[0]-1)),r4(0.)).astype(jnp.float64)
    face=(idx>=1)&(idx<=kx+1)
    def flip_face(a):
        return _where(face,_get_vector(a,jnp.clip(kx-idx+2,0,a.shape[0]-1)),r4(0.)).astype(jnp.float64)
    # Original:334–338 CUDTS is DOUBLE, even though dt/cudt are REAL inputs.
    cudts=_where(cudt==r4(0.),r8(dt),r8(cudt)*60.)
    half=cudts*.5
    with jax.enable_x64(False):
        denominator=r4(1.)+qv
        raw_qh=qv/denominator
        floored=jnp.sum((raw_qh<r4(1.e-30))&lev,dtype=jnp.int32)
        qh32=jnp.maximum(raw_qh,r4(1.e-30))
        liquid32=qc/denominator;ice32=qi/denominator
        pdel32=p8w-_roll(p8w,-1)
        zrel=z-ht;zwrel=zw-ht
        cloud32=_where(step==r4(1.),r4(0.),cldfra)
    t8,qh,liquid,ice=flip(t),flip(qh32),flip(liquid32),flip(ice32)
    pmid,pint,pdel=flip(p),flip_face(p8w),flip(pdel32)
    zm,zi=flip(zrel),flip_face(zwrel)
    if pbl==9:
        tpert=r8(tp_cam)
    else:
        tpctrl=jnp.asarray([psfc,pblh,kpbl,hfx,qfx,ust,mavail,tsk,dx,_get(u,1),_get(v,1)],jnp.float32)
        tpert=_tpert_myj(th,t,qv,dz,tke,tpctrl,kx)
    # Original:460 assigns LAND=1 for XLAND==2, despite WRF's XLAND names.
    # Preserve the literal upstream mapping and disclose it in UPSTREAM_WRF_BUGS.
    landfrac=_where(xland==r4(2.),1.,0.)
    core=_convr(t8,qh,pmid,pint,pdel,zm,zi,geos=r8(ht)*_G,pblh=r8(pblh),tpert=tpert,
                landfrac=landfrac,dt=half,dx=r8(dx),kx=kx,limcnv=2,no_deep_pbl=True)
    dlf=jnp.maximum(jnp.minimum(1.e-6,core['dlf']),0.)
    clipped=jnp.sum(((core['dlf']<0.)|(core['dlf']>1.e-6))&lev,dtype=jnp.int32)
    # ZMDT/ZMDQ are REAL output arrays and are read back by the driver.
    zmdt=flip(core['heat']/_CP).astype(jnp.float32)
    zmdq=flip(core['qtnd']).astype(jnp.float32)
    with jax.enable_x64(False):
        rth=zmdt/pi
    rqv=(r8(zmdq)*(1.+r8(qv))**2).astype(jnp.float32)
    t8=t8+flip(zmdt)*cudts
    qh=qh+flip(zmdq)*cudts
    evap=_conv_evap(t8,pmid,pdel,qh,core['rprd'],flip(cloud32),core['prec'],cudts,estbl,kx)
    evapt=flip(evap['tend_s']/_CP).astype(jnp.float32)
    evapq=flip(evap['tend_q']).astype(jnp.float32)
    with jax.enable_x64(False):
        rth=rth+evapt/pi
        rqv=rqv+evapq*(r4(1.)+qv)**2
    t8=t8+flip(evapt)*cudts
    qh=qh+flip(evapq)*cudts
    def transport(_):
        mom=_momtran(flip(u),flip(v),core,half,kx)
        ctliq=_convtran_moist(liquid,core,kx)
        ctice=_convtran_moist(ice,core,kx)
        return mom['u']['rate'],mom['v']['rate'],mom['seten'],ctliq,ctice
    zero=jnp.zeros_like(t8)
    du,dv,energy,ctliq,ctice=_cond(core['lengath']>0,transport,lambda _:(zero,zero,zero,zero,zero),None)
    rth=(r8(rth)+flip(energy)/_CP/r8(pi)).astype(jnp.float32)
    t8=t8+energy/_CP*cudts
    frozen=_where(t8>268.15,0.,_where(t8<238.15,1.,(268.15-t8)/30.))
    detliq=dlf*(1.-frozen);detice=dlf*frozen
    detnc=3.*(dlf*(1.-frozen))/(4.*3.14159*(8.e-6**3)*997.)
    detni=3.*(dlf*frozen)/(4.*3.14159*(25.e-6**3)*500.)
    rth=(r8(rth)+flip(dlf*frozen*_LF)/_CP/r8(pi)).astype(jnp.float32)
    with jax.enable_x64(False):
        wet_factor=r4(1.)+qv
    rate= r4(evap['prec']*1.e3)
    rain=rate*dt
    pressure_active=(core['lengath']>0)&(core['maxg']>core['jt'])
    pconvt=_where(pressure_active,_get(pmid,core['jt']),r8(_get(p8w,1))).astype(jnp.float32)
    pconvb=_where(pressure_active,_get(pmid,core['maxg']),r8(_get(p8w,1))).astype(jnp.float32)
    return dict(RUCUTEN=r4(flip(du)),RVCUTEN=r4(flip(dv)),RTHCUTEN=rth,RQVCUTEN=rqv,
        RQCCUTEN=r4(flip(ctliq+detliq)*r8(wet_factor)),RQICUTEN=r4(flip(ctice+detice)*r8(wet_factor)),
        RQCNCUTEN=r4(flip(detnc)*r8(wet_factor)),RQINCUTEN=r4(flip(detni)*r8(wet_factor)),
        ICWMRDP=r4(flip(core['ql'])),DLF_OUT=r4(flip(dlf)),MU_OUT=r4(flip(core['mu']*100./_G)),MD_OUT=r4(flip(core['md']*100./_G)),
        ZMDT=zmdt,ZMDQ=zmdq,ZMDICE=r4(flip(ctice+detice)),ZMDLIQ=r4(flip(ctliq+detliq)),
        CMFMC=r4(flip_face(core['mcon']*100./_G)),ZMFLXPRC=r4(flip(evap['flxprec'])),ZMFLXSNW=r4(flip(evap['flxsnow'])),
        RAINCV=rain,PRATEC=rate,CAPE_OUT=r4(core['cape']),CUTOP=r4(core['jctop']),CUBOT=r4(core['jcbot']),
        PCONVB=pconvb,PCONVT=pconvt,RLIQ_OUT=r4(core['rliq']),
        ENTROPY_FAILED=core['failed'].astype(jnp.int32),DLF_CLIPPED=clipped,VAPOR_FLOORED=floored)


def camzm_buoyancy(q,t,p,z,pf,*,pblt,tpert,msg,interpret=False):
    """Qualification boundary for the original dilute-buoyancy call."""
    kx=t.shape[0]
    size=1<<(kx+2).bit_length()
    profiles=('tp','qstp')
    scalars=('tl','cape','lcl','lel','lon','mx','failed')
    with jax.enable_x64(True):
        inputs=tuple(jnp.pad(jnp.asarray(a,jnp.float64),(1,size-a.shape[0]-1)) for a in (q,t,p,z,pf))
        control=jnp.asarray([pblt,tpert],jnp.float64)
        def kernel(*refs):
            with jax.enable_x64(True):
                out=_buoyan_dilute(*(ref[...] for ref in refs[:5]),refs[5][0],refs[5][1],msg,kx)
            for key,ref in zip(profiles+scalars,refs[6:],strict=True):
                ref[...]=out[key].astype(jnp.int32) if key=='failed' else out[key]
        shapes=tuple(jax.ShapeDtypeStruct((size,),jnp.float64) for _ in profiles)
        shapes+=tuple(jax.ShapeDtypeStruct((),jnp.float64 if key in ('tl','cape') else jnp.int32) for key in scalars)
        output=pl.pallas_call(kernel,out_shape=shapes,grid=(),interpret=interpret,
            compiler_params=plt.CompilerParams(num_warps=4),name='camzm_dilute_wrf_double')(*inputs,control)
    out={key:value[1:kx+1] for key,value in zip(profiles,output[:2],strict=True)}
    out.update(zip(scalars,output[2:],strict=True))
    return out


def camzm_convr(t,q,pap,paph,dpp,zm,zi,*,geos,pblh,tpert,landfrac,dt,dx,
                limcnv=2,no_deep_pbl=True,interpret=False):
    """Native qualification boundary for complete original ZM_CONVR."""
    kx=t.shape[0]
    size=1<<(kx+2).bit_length()
    profiles=('qtnd','heat','mcon','dlf','pflx','cme','rprd','zdu','ql','mu','md','du','eu','ed','dp')
    scalars=('cape','prec','rliq','jctop','jcbot','dsubcld','jt','maxg','lengath','failed')
    with jax.enable_x64(True):
        inputs=tuple(jnp.pad(jnp.asarray(a,jnp.float64),(1,size-a.shape[0]-1)) for a in (t,q,pap,paph,dpp,zm,zi))
        control=jnp.asarray([geos,pblh,tpert,landfrac,dt,dx],jnp.float64)
        def kernel(*refs):
            with jax.enable_x64(True):
                ctl=refs[7][...]
                out=_convr(*(ref[...] for ref in refs[:7]),geos=ctl[0],pblh=ctl[1],tpert=ctl[2],
                            landfrac=ctl[3],dt=ctl[4],dx=ctl[5],kx=kx,limcnv=limcnv,no_deep_pbl=no_deep_pbl)
            for key,ref in zip(profiles+scalars,refs[8:],strict=True):
                ref[...]=out[key].astype(jnp.int32) if key=='failed' else out[key]
        shapes=tuple(jax.ShapeDtypeStruct((size,),jnp.float64) for _ in profiles)
        shapes+=tuple(jax.ShapeDtypeStruct((),jnp.int32 if key in ('jt','maxg','lengath','failed') else jnp.float64) for key in scalars)
        output=pl.pallas_call(kernel,out_shape=shapes,grid=(),interpret=interpret,
            compiler_params=plt.CompilerParams(num_warps=4),name='camzm_convr_wrf_double')(*inputs,control)
    out={key:value[1:kx+2 if key in ('mcon','pflx') else kx+1] for key,value in zip(profiles,output[:len(profiles)],strict=True)}
    out.update(zip(scalars,output[len(profiles):],strict=True))
    return out


def camzm_fresh_column(t,qv,p,p8w,pi,u,v,z,zw,tke,th,qc,qi,cldfra,dz,*,
                       dt,dx,cudt,step,psfc,terrain,pblh,kpbl,hfx,qfx,ust,mavail,xland,tsk,
                       tpert_camuw,pbl,sfclay,interpret=False):
    """Complete non-chemistry/non-CAMMGMP original CAMZM_DRIVER column.

    Bottom-up REAL inputs/outputs, with the original top-down cloud-index
    diagnostics retained. Original source events are returned alongside rates.
    Operational cadence, persistent rate fields and failure guards are separate.
    """
    if not (pbl==9 or (pbl==2 and sfclay==2)):
        raise ValueError('WRF CAM-ZM requires CAM-UW PBL or MYJ PBL with MYJ surface layer')
    from gpuwrf.physics.bl_camuw import ESTBL
    kx=t.shape[0]
    size=1<<(kx+2).bit_length()
    arrays=(t,qv,p,p8w,pi,u,v,z,zw,tke,th,qc,qi,cldfra,dz)
    inputs=tuple(jnp.pad(jnp.asarray(a,jnp.float32),(1,size-a.shape[0]-1)) for a in arrays)
    # The final lane is unused: native Triton requires power-of-two loads.
    control=jnp.asarray([dt,dx,cudt,step,psfc,terrain,pblh,kpbl,hfx,qfx,ust,mavail,xland,tsk,tpert_camuw,0.],jnp.float32)
    with jax.enable_x64(True):
        estbl=jnp.asarray(ESTBL+(-99999.,)*6,jnp.float64)
        def kernel(*refs):
            with jax.enable_x64(True):
                ctrl=tuple(plt.load(refs[15].at[i]) for i in range(15))
                out=_camzm_cps(tuple(ref[...] for ref in refs[:15]),ctrl,refs[16],kx=kx,pbl=pbl)
            for key,ref in zip(_CAM_PROFILE+_CAM_SCALAR,refs[17:],strict=True):ref[...]=out[key]
        shapes=tuple(jax.ShapeDtypeStruct((size,),jnp.float32) for _ in _CAM_PROFILE)
        shapes+=tuple(jax.ShapeDtypeStruct((),jnp.int32 if key in ('ENTROPY_FAILED','DLF_CLIPPED','VAPOR_FLOORED') else jnp.float32) for key in _CAM_SCALAR)
        output=pl.pallas_call(kernel,out_shape=shapes,grid=(),interpret=interpret,
            compiler_params=plt.CompilerParams(num_warps=4),name='camzm_cps_wrf_real_double')(*inputs,control,estbl)
    result={key:value[1:kx+2 if key=='CMFMC' else kx+1] for key,value in zip(_CAM_PROFILE,output[:len(_CAM_PROFILE)],strict=True)}
    result.update(zip(_CAM_SCALAR,output[len(_CAM_PROFILE):],strict=True))
    return result
