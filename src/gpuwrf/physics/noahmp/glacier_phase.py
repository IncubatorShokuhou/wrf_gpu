"""Literal glacier snow/ice phase change, pristine WRF:1609-1996."""
from __future__ import annotations
import jax.numpy as jnp
from .glacier_energy import TFRZ, HFUS


def _any(values):
    result = values[0]
    for value in values[1:]:
        result = result | value
    return result


def phase_change(v, dz, fact, opt_gla, xp=jnp):
    t=v['stc']; n=v['isnow']; dt=v['dt']; zero=xp.zeros_like(v['tg'])
    ice=list(v['snice'])+[zero]*4; liquid=list(v['snliq'])+[zero]*4
    if opt_gla==1:
        for k in range(3,7):
            liquid[k]=v['sh2o'][k-3]*dz[k]*1000.
            ice[k]=(v['smc'][k-3]-v['sh2o'][k-3])*dz[k]*1000.
    oldice=list(ice); mass=[a+b for a,b in zip(ice,liquid)]
    imelt=[xp.zeros_like(n)]*7; hm=[zero]*7; xm=[zero]*7
    qmelt=zero; pond=zero

    def identify(k, active, shallow=False):
        melt=(ice[k]>0.)&(t[k]>=TFRZ)
        if shallow:
            melt=melt | ((n==0)&(v['sneqv']>0.)&(t[k]>=TFRZ))
        imelt[k]=xp.where(active & melt,1,imelt[k])
        imelt[k]=xp.where(active & (liquid[k]>0.) & (t[k]<TFRZ),2,imelt[k])
        hm[k]=xp.where(active & (imelt[k]>0),(t[k]-TFRZ)/xp.where(active,fact[k],1.),0.)
        t[k]=xp.where(active & (imelt[k]>0),TFRZ,t[k])
        invalid=((imelt[k]==1)&(hm[k]<0.))|((imelt[k]==2)&(hm[k]>0.))
        imelt[k]=xp.where(invalid,0,imelt[k]);hm[k]=xp.where(invalid,0.,hm[k])
        xm[k]=hm[k]*dt/HFUS

    def resolve(k, active, snow):
        do=active & (imelt[k]>0) & (xp.abs(hm[k])>0.)
        ni=xp.where(xm[k]>0.,xp.maximum(0.,oldice[k]-xm[k]),
                    xp.where(xm[k]<0.,xp.minimum(mass[k],oldice[k]-xm[k]),ice[k]))
        heat=xp.where(xp.abs(xm[k])>0.,hm[k]-HFUS*(oldice[k]-ni)/dt,0.)
        ice[k]=xp.where(do,ni,ice[k]);liquid[k]=xp.where(do,xp.maximum(0.,mass[k]-ni),liquid[k])
        nt=t[k]+fact[k]*heat
        if snow:
            nt=xp.where(liquid[k]*ice[k]>0.,TFRZ,nt)
        t[k]=xp.where(do & (xp.abs(heat)>0.),nt,t[k])
        return xp.where(do,xp.maximum(0.,oldice[k]-ice[k])/dt,0.)

    for k in range(3):
        identify(k,k>=n+3)
    if opt_gla==2:
        do=(n==0)&(v['sneqv']>0.)&(t[3]>=TFRZ)
        h=(t[3]-TFRZ)/fact[3]; melt=h*dt/HFUS
        old=v['sneqv']; new=xp.maximum(0.,old-melt)
        heat=h-HFUS*(old-new)/dt
        v['sneqv']=xp.where(do,new,old)
        v['snowh']=xp.where(do,xp.maximum(0.,new/xp.where(old>0.,old,1.)*v['snowh']),v['snowh'])
        t[3]=xp.where(do,TFRZ+xp.where(heat>0.,fact[3]*heat,0.),t[3])
        qmelt=xp.where(do,xp.maximum(0.,old-new)/dt,qmelt);pond=xp.where(do,old-new,pond)
    for k in range(3):
        qmelt=qmelt+resolve(k,k>=n+3,True)
    if opt_gla==1:
        for k in range(3,7):
            identify(k,True,shallow=k==3)
        do=(n==0)&(v['sneqv']>0.)&(xm[3]>0.)
        old=v['sneqv'];new=xp.maximum(0.,old-xm[3]);heat=hm[3]-HFUS*(old-new)/dt
        v['sneqv']=xp.where(do,new,old)
        v['snowh']=xp.where(do,xp.maximum(0.,new/xp.where(old>0.,old,1.)*v['snowh']),v['snowh'])
        xm[3]=xp.where(do,xp.where(heat>0.,heat*dt/HFUS,0.),xm[3])
        hm[3]=xp.where(do,xp.where(heat>0.,heat,0.),hm[3]);imelt[3]=xp.where(do,xp.where(heat>0.,1,0),imelt[3])
        qmelt=xp.where(do,xp.maximum(0.,old-new)/dt,qmelt);pond=xp.where(do,old-new,pond)
        for k in range(3,7):
            resolve(k,True,False)
        # WRF redistributes heat/cold between ice layers in this exact order.
        # These fixed 4x4 operations stay inside the column kernel.
        for cold in (False,True):
            global_do=_any([x>TFRZ for x in t[3:]]) & _any([x<TFRZ for x in t[3:]])
            for j in range(3,7):
                do=global_do & (t[j]<TFRZ if cold else t[j]>TFRZ)
                h=(t[j]-TFRZ)/fact[j]
                for k in range(3,7):
                    if k==j:
                        continue
                    hit=do & (t[k]>TFRZ if cold else t[k]<TFRZ) & (h<-.1 if cold else h>.1)
                    hk=(t[k]-TFRZ)/fact[k]
                    absorbs=hk>xp.abs(h) if cold else xp.abs(hk)>h
                    t[k]=xp.where(hit,xp.where(absorbs,TFRZ+(hk+h)*fact[k],TFRZ),t[k])
                    h=xp.where(hit,xp.where(absorbs,0.,h+hk),h)
                t[j]=xp.where(do,TFRZ+h*fact[j],t[j])
        for freeze in (False,True):
            global_do=_any([x<TFRZ if freeze else x>TFRZ for x in t[3:]])
            global_do=global_do & _any([x>0. for x in (liquid[3:] if freeze else ice[3:])])
            for j in range(3,7):
                do=global_do & (t[j]<TFRZ if freeze else t[j]>TFRZ)
                x=(t[j]-TFRZ)/fact[j]*dt/HFUS
                for k in range(3,7):
                    if k==j:
                        continue
                    available=liquid[k] if freeze else ice[k]
                    hit=do & (available>0.) & (x<-.1 if freeze else x>.1)
                    absorbs=available>xp.abs(x) if freeze else available>x
                    ni=xp.where(absorbs,ice[k]-x,mass[k] if freeze else 0.)
                    nx=xp.where(absorbs,0.,x+available if freeze else x-available)
                    ice[k]=xp.where(hit,ni,ice[k]);liquid[k]=xp.where(hit,xp.maximum(0.,mass[k]-ice[k]),liquid[k])
                    t[k]=xp.where(hit,TFRZ,t[k]);x=xp.where(hit,nx,x)
                t[j]=xp.where(do,TFRZ+(x*HFUS/dt)*fact[j],t[j])
    v['snice']=ice[:3];v['snliq']=liquid[:3];v['stc']=t
    v['sh2o']=[xp.maximum(0.,xp.minimum(1.,liquid[k+3]/(1000.*dz[k+3]))) if opt_gla==1 else zero for k in range(4)]
    v['smc']=[xp.ones_like(zero)]*4
    v.update(qmelt=qmelt,ponding=pond)
    return imelt
