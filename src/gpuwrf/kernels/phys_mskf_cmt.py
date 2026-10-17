"""MSKF momentum transport: original module_cu_mskf.F:7555–8038 DOUBLE island.

One column program owns the ten harmonic tridiagonal solves and the cloud
wind recurrence. Harmonic work is reused between solves rather than storing
all ten systems. The REAL32 caller's conversions precede this DOUBLE body.
The experimental CPU-oracle-qualified cu=11 caller includes this body by default.
GPU and full-forecast qualification remain pending.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from gpuwrf.backends import pallas as pl
from jax.experimental.pallas import triton as plt

from gpuwrf.kernels.phys_kf_column import _get, _set, _roll, _where

_ALFA1=(4.33675e-5,7.88429e-5,1.13065e-4,1.45836e-4,1.76823e-4,
        2.05660e-4,2.32101e-4,2.55829e-4,2.76672e-4,2.94347e-4)
_ALFA2=(-8.64731e-5,-1.56092e-4,-2.21315e-4,-2.80995e-4,-3.33799e-4,
        -3.78455e-4,-4.14020e-4,-4.39600e-4,-4.54652e-4,-4.58761e-4)
_BSJ0=(-.402759,.300116,-.249705,.218359,-.196465,.180062,-.167183,.156722,-.148011,.140605)
_BSJ1=(.114193,.205841,.291209,.368619,.436201,.492233,.535486,.564886,.579879,.580195)
_TAU=(3.83170,7.01560,10.1735,13.3237,16.4705,19.6120,22.7560,25.8980,29.0480,32.1926)
_INPUTS=('DP','ED','EU','MC','MD','MU','P','QD','QU','QHAT','SD','SU','SHAT','T','U','V','Z','ZF')


def _cmt(profiles,control,kx,harmonics):
    DP,ED,EU,MC,MD,MU,P,QD,QU,QHAT,SD,SU,SHAT,T,U,V,Z,ZF=profiles
    dt,grav,cpres,rgas,dsubcld,jb,jd,jt=control
    idx=jnp.arange(DP.shape[0],dtype=jnp.int32)
    lev=(idx>=1)&(idx<=kx)
    jb,jd,jt=jb.astype(jnp.int32),jd.astype(jnp.int32),jt.astype(jnp.int32)
    rc,rmax,chi,shape,factor=3000.,50000.,3.8317/3000.,.8,.179503
    zero=jnp.zeros_like(DP)
    alpha=_where(lev,.5,0.)
    ec=EU+ED
    # Original:7686 only assigns RHO for T>0; zero-init policy defines the
    # other locations. In physical positive-temperature columns it is unused.
    rho=_where(lev&(T>0),100.*P/(rgas*T),0.)
    def buoyancy(s,q):
        dry=SHAT-grav/cpres*ZF
        return _where(lev&(idx>1),(s-SHAT+.608*(q*(s-grav/cpres*ZF)-QHAT*dry))
                      /(dry*(1.+.608*QHAT))*grav,0.)
    bu,bd=buoyancy(SU,QU),buoyancy(SD,QD)
    def mean(j,uv):
        u,v=uv
        active=(j>=jt)&(j<jb)
        dz=_get(ZF,j)-_get(ZF,j+1)
        return (u+_where(active,_get(U,j)*_get(P,j)/_get(T,j)*dz,0.),
                v+_where(active,_get(V,j)*_get(P,j)/_get(T,j)*dz,0.))
    umn,vmn=jax.lax.fori_loop(1,kx+1,mean,(jnp.float64(0.),jnp.float64(0.)))
    windmag=jnp.sqrt(umn**2+vmn**2)
    cosa,sina=_where(windmag!=0,umn/windmag,0.),_where(windmag!=0,vmn/windmag,0.)
    def hat(a):return _where(idx>1,alpha*_roll(a,1)+(1.-alpha)*a,a)
    rhohat,uhat,vhat=hat(rho),hat(U),hat(V)
    dzf=_where(idx>1,_roll(Z,1)-Z,ZF-Z)
    dz=_where(idx<kx,ZF-_roll(ZF,-1),DP*100./(rho*grav))
    au=_where(lev&(idx>=jt),_get(MU,jb)*100./grav/_get(rhohat,jb),0.)
    ad=_where(lev&(idx>=jd),-_get(MD,jd)*100./grav/_get(rhohat,jd),0.)
    ac=au+ad
    w0=_where(lev&(ac>0),MC*100./grav/(rhohat*ac),0.)
    def layer(a):return _where(idx<kx,alpha*a+(1.-alpha)*_roll(a,-1),alpha*a)
    acfl,aufl,adfl,w0fl=layer(ac),layer(au),layer(ad),layer(w0)
    cloud=(idx<jb)&(idx>=jt)
    d0=_where(cloud&(acfl>0),(_roll(MC,-1)-MC)*100./grav/(rho*dz*acfl),
              _where((idx==jb)&(acfl>0),-MC*100./grav/(rho*dz*acfl),0.))
    e=_where(cloud&(acfl>0)&(adfl>0)&(aufl>0),
             1./chi**2*adfl/(acfl*rho)*(1./(aufl*dz)*(MU-_roll(MU,-1))*100./grav
             -1./(adfl*dz)*(MD-_roll(MD,-1))*100./grav)*shape/factor,0.)
    d0hat=_where(idx>1,alpha*_roll(d0,1)+(1.-alpha)*d0,(1.-alpha)*d0)
    ehat=_where(idx>1,alpha*_roll(e,1)+(1.-alpha)*e,(1.-alpha)*e)
    dual=(idx>=jd)&(idx<=jb)&(au>0)&(ad>0)&(ac>0)
    w1=_where(dual,shape/factor*(MU/au-MD/ad)*100./grav*ad/(ac*rhohat),0.)
    b1=_where(dual,shape/factor*(bu-bd)*ad/ac,0.)
    w1fl=layer(w1)
    # Original:7847 evaluates MC(ILEV+1) inside a false compound guard.
    # The explicit padding is zero under the standing undefined-read policy.
    active=lev&cloud&(MC>0)&(_roll(MC,-1)>0)
    def solve(n,total):
        a1,a2,bj0,bj1,tau=(_get(a,n) for a in harmonics)
        def forcing(hatwind,angle):
            f=2.*rho/bj0**2*(d0*e*chi**2*angle*a1
                +2.*w0fl*rc/rmax**2*bj1*((ehat-_roll(ehat,-1))/dz*chi*angle*(-.402759)
                                       +(hatwind-_roll(hatwind,-1))/dz)
                -(d0hat-_roll(d0hat,-1))/dz*w1fl*angle*(a2-a1)
                -2.*((w0-_roll(w0,-1))*(w1-_roll(w1,-1))/dz**2
                      -w0fl*w1fl*(jnp.log(_roll(rho,1))/(dz*dzf)
                        -(1./dzf+1./_roll(dzf,-1))/dz*jnp.log(rho)
                        +jnp.log(_roll(rho,-1))/(dz*_roll(dzf,-1))))*angle*a1
                +(rhohat*b1-_roll(rhohat,-1)*_roll(b1,-1))/(dz*rho)*angle*a1)
            return _where(active,f,0.)
        a=1./(dz*dzf)
        b=-(1./dz*(1./dzf+1./_roll(dzf,-1))+(tau/rmax)**2)
        c=1./(dz*_roll(dzf,-1))
        x,y=forcing(uhat,cosa),forcing(vhat,sina)
        a=_set(_set(a,1,0.),kx,1./_get(dzf,kx))
        b=_set(_set(b,1,1./_get(dzf,2)),kx,-1./_get(dzf,kx))
        c=_set(_set(c,1,-1./_get(dzf,2)),kx,0.)
        x=_set(_set(x,1,_get(b1,2)*a1*2.*cosa/bj0**2),kx,_get(b1,kx)*a1*2.*cosa/bj0**2)
        y=_set(_set(y,1,_get(b1,2)*a1*2.*sina/bj0**2),kx,_get(b1,kx)*a1*2.*sina/bj0**2)
        c,x,y=(_set(q,1,_get(q,1)/_get(b,1)) for q in (c,x,y))
        def forward(j,s):
            c,x,y=s
            denominator=_get(b,j)-_get(a,j)*_get(c,j-1)
            return (_set(c,j,_get(c,j)/denominator),
                    _set(x,j,(_get(x,j)-_get(a,j)*_get(x,j-1))/denominator),
                    _set(y,j,(_get(y,j)-_get(a,j)*_get(y,j-1))/denominator))
        c,x,y=jax.lax.fori_loop(2,kx+1,forward,(c,x,y))
        def backward(i,xy):
            j=kx-1-i;x,y=xy
            return (_set(x,j,_get(x,j)-_get(c,j)*_get(x,j+1)),
                    _set(y,j,_get(y,j)-_get(c,j)*_get(y,j+1)))
        x,y=jax.lax.fori_loop(0,kx-1,backward,(x,y))
        return total[0]+x*bj1,total[1]+y*bj1
    px,py=jax.lax.fori_loop(0,10,solve,(zero,zero))
    px,py=acfl*(px/rc)/rho,acfl*(py/rc)/rho
    def cloud_wind(i,uv):
        j=kx-1-i;uc,vc=uv
        yes=(_get(MC,j)>0)&(_get(MC,j+1)>0)&(j>jt)&(j<jb)
        scale=_get(rho,j)*_get(dz,j)/((_get(MC,j)+_get(MC,j+1))*.5*100./grav)
        un=_get(uc,j+1)+scale*(_get(ec,j)*(_get(U,j)-_get(uc,j+1))-_get(px,j))
        vn=_get(vc,j+1)+scale*(_get(ec,j)*(_get(V,j)-_get(vc,j+1))-_get(py,j))
        return _set(uc,j,_where(yes,un,_get(uc,j))),_set(vc,j,_where(yes,vn,_get(vc,j)))
    uc,vc=jax.lax.fori_loop(0,kx-1,cloud_wind,(uhat,vhat))
    # Original:7981 references DUDT(MSG) while RNU=0. The caller initializes
    # all physical DUDT/DVDT to zero; padding defines the out-of-bounds site.
    rnu=jnp.float64(0.)
    def tendencies(j,uv):
        du,dv=uv
        uh=_get(uhat,j)+rnu*_get(alpha,j)*dt*_get(du,j-1)
        vh=_get(vhat,j)+rnu*_get(alpha,j)*dt*_get(dv,j-1)
        def rate(c,h,now):
            denominator=1.+rnu*_get(alpha,j+1)*_get(MC,j+1)*dt/_get(DP,j)
            upper=_get(MC,j+1)*(_get(c,j+1)-_get(h,j+1))
            interior=(_get(MC,j+1)*(_get(c,j+1)-_get(h,j+1))
                      -_get(MC,j)*(_get(c,j)-now))
            value=_where(j==jt,1./denominator*upper/_get(DP,j),
                         _where(j<jb,1./denominator*interior/_get(DP,j),
                                -1./dsubcld*_get(MC,j)*(_get(c,j)-now)))
            return _where((j>=jt)&(j<=jb),value,0.)
        return _set(du,j,rate(uc,uhat,uh)),_set(dv,j,rate(vc,vhat,vh))
    du,dv=jax.lax.fori_loop(1,kx+1,tendencies,(zero,zero))
    return du,dv,_where(lev,px,0.),_where(lev,py,0.)


def mskf_cmt(profiles,*,dt,gravity,cp,gas_constant,dsubcld,jb,jd,jt,interpret=False):
    """Original top-down, one-based cloud indices; inputs keyed by _INPUTS."""
    kx=profiles['DP'].shape[0]
    size=1<<(kx+2).bit_length()
    with jax.enable_x64(True):
        inputs=tuple(jnp.pad(jnp.asarray(profiles[key],jnp.float64),(1,size-kx-1)) for key in _INPUTS)
        control=jnp.asarray([dt,gravity,cp,gas_constant,dsubcld,jb,jd,jt],jnp.float64)
        harmonics=tuple(jnp.asarray(a+(0.,)*6,jnp.float64) for a in (_ALFA1,_ALFA2,_BSJ0,_BSJ1,_TAU))
        def kernel(*refs):
            with jax.enable_x64(True):
                output=_cmt(tuple(ref[...] for ref in refs[:18]),refs[18][...],kx,
                            tuple(ref[...] for ref in refs[19:24]))
            for ref,value in zip(refs[24:],output,strict=True):ref[...]=value
        out=pl.pallas_call(kernel,out_shape=tuple(jax.ShapeDtypeStruct((size,),jnp.float64) for _ in range(4)),
                           grid=(),interpret=interpret,compiler_params=plt.CompilerParams(num_warps=4),
                           name='mskf_cmt_wrf_double')(*inputs,control,*harmonics)
    return dict(zip(('DUDT','DVDT','DPDX','DPDY'),(a[1:kx+1] for a in out),strict=True))
