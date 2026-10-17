"""Literal old-KF thermodynamics from pristine module_cu_kf.F.

These analytic liquid/ice/secant routines are distinct from KF-eta's lookup
tables. REAL input dtype governs arithmetic. Component qualification precedes
any KFCPS admission; this module does not expose an operational scheme.
"""
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp


@lru_cache(None)
def constants(mode='r4'):
    L=np.float32 if mode=='r4' else np.float64
    return SimpleNamespace(L=L,ep2=L(287.)/L(461.6),aliq=L(.6112)*L(1000.),
        bliq=L(17.67),cliq=L(17.67)*L(273.15),dliq=L(29.65),
        aice=L(613.2),bice=L(22.452),cice=L(6133.),dice=L(.61),
        xlv0=L(3.15e6),xlv1=L(2370.),xls0=L(2.905e6),xls1=L(259.532),g=L(9.81))


def envirtht(p,t,q,ratio,rl,*,mode='r4'):
    c=constants(mode);L=c.L
    ee=q*p/(c.ep2+q)
    logliq=jnp.log(ee/c.aliq);logice=jnp.log(ee/c.aice)
    tdpt=(c.cliq-c.dliq*logliq)/(c.bliq-logliq)
    tfpt=(c.cice-c.dice*logice)/(c.bice-logice)
    tsl=tdpt-(L(.212)+L(1.571e-3)*(tdpt-L(273.16))-L(4.36e-4)*(t-L(273.16)))*(t-tdpt)
    tsi=tfpt-(L(.182)+L(1.13e-3)*(tfpt-L(273.16))-L(3.58e-4)*(t-L(273.16)))*(t-tfpt)
    tht=t*(L(1.e5)/p)**(L(.2854)*(L(1.)-L(.28)*q))
    common=q*(L(1.)+L(.81)*q)
    liquid=tht*jnp.exp((L(3374.6525)/tsl-L(2.5403))*common)
    ice=tht*jnp.exp((L(3114.834)/tsi-L(.278296))*common)
    tsat=ratio*tsi+(L(1.)-ratio)*tsl
    mixed=tht*jnp.exp(rl*q*L(1.0723e-3)/tsat*(L(1.)+L(.81)*q))
    return jnp.where(ratio<L(1.e-6),liquid,jnp.where(jnp.abs(ratio-L(1.))<L(1.e-6),ice,mixed))


def saturated(p,t,ratio,rl,*,mode='r4'):
    c=constants(mode);L=c.L
    el=c.aliq*jnp.exp((c.bliq*t-c.cliq)/(t-c.dliq))
    ei=c.aice*jnp.exp((c.bice*t-c.cice)/(t-c.dice))
    liquid=ratio<L(1.e-6);ice=jnp.abs(ratio-L(1.))<L(1.e-6)
    es=jnp.where(liquid,el,jnp.where(ice,ei,(L(1.)-ratio)*el+ratio*ei))
    qs=c.ep2*es/(p-es)
    pi=(L(1.e5)/p)**(L(.2854)*(L(1.)-L(.28)*qs))
    exponent=jnp.where(liquid,(L(3374.6525)/t-L(2.5403))*qs*(L(1.)+L(.81)*qs),
        jnp.where(ice,(L(3114.834)/t-L(.278296))*qs*(L(1.)+L(.81)*qs),
                  rl*qs*L(1.0723e-3)/t*(L(1.)+L(.81)*qs)))
    return t*pi*jnp.exp(exponent),qs


def secant(p,thte,t,ratio,rl,tolerance,*,mode='r4'):
    c=constants(mode);L=c.L
    initial,_=saturated(p,t,ratio,rl,mode=mode)
    f0=initial-thte
    t1=t-L(.5)*f0
    # The original evaluates the final iterate before its >10-iteration exit.
    def body(old,_):
        t0,f0,t1,qs,count,done=old
        th,qs1=saturated(p,t1,ratio,rl,mode=mode)
        f1=th-thte
        converged=jnp.abs(f1)<L(tolerance)
        count1=jnp.where(~done&~converged,count+1,count)
        stop=done|converged|(count1>10)
        delta=f1*(t1-t0)/(f1-f0)
        return (jnp.where(stop,t0,t1),jnp.where(stop,f0,f1),
                jnp.where(stop,t1,t1-delta),jnp.where(done,qs,qs1),count1,stop),None
    (_,_,tn,qs,count,_),_=jax.lax.scan(body,
        (t,f0,t1,jnp.zeros_like(t),jnp.asarray(0,jnp.int32),jnp.asarray(False)),None,length=12)
    return tn,qs,count


def tpmix(p,thte,t,q,ql,qi,ratio,rl,*,mode='r4'):
    c=constants(mode);L=c.L
    tn,qs,_=secant(p,thte,t,ratio,rl,L(.01),mode=mode)
    dq=qs-q;total=ql+qi
    supersat=qs<=q
    enough=total>=dq
    qla=ql-(L(1.)-ratio)*dq
    dqice=jnp.where(qla<L(0.),-qla,L(0.));qla=jnp.maximum(qla,L(0.))
    qia=qi-ratio*dq+dqice
    dqliq=jnp.where(qia<L(0.),-qia,L(0.));qia=jnp.maximum(qia,L(0.));qla=qla+dqliq
    rll=jnp.where(ratio<L(1.e-6),c.xlv0-c.xlv1*tn,
                  jnp.where(jnp.abs(ratio-L(1.))<L(1.e-6),c.xls0-c.xls1*tn,rl))
    ccp=L(1005.7)*(L(1.)+L(.89)*q)
    small=total<L(1.e-10)
    tnew=tn+jnp.where(small,rll*(dq/(L(1.)+dq))/ccp,
                      rll*((dq-total)/(L(1.)+dq-total))/ccp)
    qnew=jnp.where(supersat,q-qs,L(0.))
    return (jnp.where(supersat|enough,tn,tnew),
            jnp.where(supersat|enough,qs,jnp.where(small,q,q+total)),
            jnp.where(supersat,ql,jnp.where(enough,qla,jnp.where(small,ql,L(0.)))),
            jnp.where(supersat,qi,jnp.where(enough,qia,jnp.where(small,qi,L(0.)))),
            (L(1.)-ratio)*qnew,ratio*qnew)


def tpdd(p,thte,t,rd,rh,*,mode='r4'):
    c=constants(mode);L=c.L
    tn,rs,_=secant(p,thte,t,L(0.),L(0.),L(.05),mode=mode)
    rl=c.xlv0-c.xlv1*tn
    dssdt=(c.cliq-c.bliq*c.dliq)/((tn-c.dliq)*(tn-c.dliq))
    delta=rl*rs*(L(1.)-rh)/(L(1005.7)+rl*rh*rs*dssdt)
    trh=tn+delta
    es=rh*c.aliq*jnp.exp((c.bliq*trh-c.cliq)/(trh-c.dliq))
    qrh=c.ep2*es/(p-es)
    trh=jnp.where(qrh<rd,tn+(rs-rd)*rl/L(1005.7),trh)
    qrh=jnp.where(qrh<rd,rd,qrh)
    return jnp.where(rh==L(1.),tn,trh),jnp.where(rh==L(1.),rs,qrh)


def condload(ql,qi,wtw,dz,boterm,enterm,rate,qnewlq,qnewic,*,mode='r4'):
    c=constants(mode);L=c.L
    total=ql+qi;new=qnewlq+qnewic
    qest=L(.5)*(total+new)
    g1=wtw+boterm-enterm-L(2.)*c.g*dz*qest/L(1.5)
    g1=jnp.where(g1<L(0.),L(0.),g1)
    wavg=(jnp.sqrt(wtw)+jnp.sqrt(g1))/L(2.)
    conv=rate*dz/wavg
    ratio3=qnewlq/(new+L(1.e-10))
    total=total+L(.6)*new;old=total
    ratio4=(L(.6)*qnewlq+ql)/(total+L(1.e-10))
    total=total*jnp.exp(-conv)
    dq=old-total
    qlout=ratio4*dq;qiout=(L(1.)-ratio4)*dq
    drag=L(.5)*(old+total-L(.2)*new)
    wnew=wtw+boterm-enterm-L(2.)*c.g*dz*drag/L(1.5)
    return (ratio4*total+ratio3*L(.4)*new,
            (L(1.)-ratio4)*total+(L(1.)-ratio3)*L(.4)*new,wnew,qlout,qiout)


def dtfrznew(t,p,thte,q,ql,qi,ratio,ttfrz,tbfrz,qnw,rl,frc,eff,iflag,*,mode='r4'):
    """Old-KF gradual glaciation; distinct from KF-eta's abrupt freezing helper."""
    c=constants(mode);L=c.L
    qlfrz=ql*eff
    new=qnw*eff*L(.5)
    esl=c.aliq*jnp.exp((c.bliq*t-c.cliq)/(t-c.dliq))
    esi=c.aice*jnp.exp((c.bice*t-c.cice)/(t-c.dice))
    rlc=L(2.5e6)-L(2369.276)*(t-L(273.16))
    rls=L(2833922.)-L(259.532)*(t-L(273.16))
    rlf=rls-rlc
    ccp=L(1005.7)*(L(1.)+L(.89)*q)
    a=(c.cice-c.bice*c.dice)/((t-c.dice)*(t-c.dice))
    b=rls*c.ep2/p
    cc=a*b*esi/ccp
    dq=b*(esl-esi)/(rls+rls*cc)-rlf*(qlfrz+new)/(rls+rls/cc)
    dt=(rlf*(qlfrz+new)+b*(esl-esi))/(ccp+a*b*esi)
    tn=t+frc*dt;qn=q-frc*dq
    es=qn*p/(c.ep2+qn)
    esl=c.aliq*jnp.exp((c.bliq*tn-c.cliq)/(tn-c.dliq))
    esi=c.aice*jnp.exp((c.bice*tn-c.cice)/(tn-c.dice))
    rn=(esl-es)/(esl-esi)
    finish=(iflag>0)&(rn<L(1.))
    overshoot=~finish&(rn>L(1.))
    fn=jnp.where(finish,frc+(L(1.)-rn),
                  jnp.where(overshoot,jnp.maximum(L(0.),frc-(rn-L(1.))),frc))
    adjust=finish|overshoot
    tn=jnp.where(adjust,t+fn*dt,tn);qn=jnp.where(adjust,q-fn*dq,qn)
    rn=jnp.where(adjust,L(1.),rn)
    flag=jnp.where(adjust,1,jnp.asarray(iflag,jnp.int32))
    rlc=c.xlv0-c.xlv1*tn;rls=c.xls0-c.xls1*tn
    rln=rn*rls+(L(1.)-rn)*rlc
    pi=(L(1.e5)/p)**(L(.2854)*(L(1.)-L(.28)*qn))
    thn=tn*pi*jnp.exp(rln*qn*L(1.0723e-3)/tn*(L(1.)+L(.81)*qn))
    qin=jnp.where(flag==1,qi+fn*dq+ql,qi+fn*(dq+qlfrz))
    qln=jnp.where(flag==1,L(0.),ql-fn*qlfrz)
    return tn,thn,qn,qln,qin,rn,jnp.zeros_like(qnw),rln,fn,flag.astype(tn.dtype)
