"""WRF ARW MFSHCONVPBL caller (KRR=2, KRRL=1, KRRI=0).

Literal column path: momentum mixing is disabled and the mass-flux transport
is explicit (PIMPL=0), as in module_pbl_driver.F. REAL follows input dtype.
"""
from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.pbl_qnse import _level_pick


@lru_cache(maxsize=2)
def _constants(dtype):
    f=np.dtype(dtype).type
    rd,rv=f(287.),f(461.6)
    cpd=f(7.)*rd/f(2.); cpv=f(4.)*rv
    cl,ci,tt,lv,ls=map(f,(4218.,2106.,273.16,2.5008e6,2.8345e6))
    gamw=(cl-cpv)/rv; betaw=lv/rv+gamw*tt
    alpw=f(6.415326)+betaw/tt+gamw*f(5.610058)
    return SimpleNamespace(G=f(9.80665),RD=rd,RV=rv,CPD=cpd,CPV=cpv,
                           CL=cl,CI=ci,TT=tt,LV=lv,LS=ls,GAMW=gamw,
                           BETAW=betaw,ALPW=alpw,EPS= f(18.0153e-3)/f(28.9644e-3))


def _thermo_liquid(p,thl,rt,c):
    """TH_R_FROM_THL_RT_{1D,2D}: its ice path is disabled in pristine ARW."""
    exn=(p/1.e5)**(c.RD/c.CPD)
    th=thl; rv=rt; rc=jnp.zeros_like(rt)
    coef=jnp.asarray(.8,rt.dtype)
    def iteration(_, state):
        th,rv,rc=state
        temp=th*exn
        foes=jnp.exp(c.ALPW-c.BETAW/temp-c.GAMW*jnp.log(temp))
        qsat=(c.RD/c.RV)*foes/p/(1.+(c.RD/c.RV-1.)*foes/p)
        rvsat=(1.-coef)*qsat*(1.+rt)+coef*rv
        rlt=jnp.maximum(0.,rv-rvsat)
        rv_new=rv-rlt
        rc_new=(1.-0.)*(rt-rv_new)
        cph=c.CPD+c.CPV*rv_new+c.CL*rc_new+c.CI*0.
        lv=(c.LV+(c.CPV-c.CL)*(temp-c.TT))/(cph*exn)
        ls=(c.LS+(c.CPV-c.CI)*(temp-c.TT))/(cph*exn)
        th_new=thl+lv*rc_new+ls*0.
        warm=temp>273.15
        rv=jnp.where(warm,rv_new,rt)
        rc=jnp.where(warm,rc_new,jnp.zeros_like(rc_new))
        th=jnp.where(warm,th_new,thl)
        return th,rv,rc
    if getattr(c,'column_kernel',False):
        # This loop is lowered INSIDE the Pallas column kernel. It never
        # launches a domain-wide kernel per saturation iteration.
        th,rv,rc=jax.lax.fori_loop(0,20,iteration,(th,rv,rc))
    else:
        for k in range(20):th,rv,rc=iteration(k,(th,rv,rc))
    return th,rv,rc


def _virtual(p,thl,rt,c):
    th,rv,rc=_thermo_liquid(p,thl,rt,c)
    return th*(1.+(c.RV/c.RD)*rv)/(1.+rt),rc


def _bl89_one(dzz,tke,gref,vpt,k,up):
    n=len(vpt)
    delta=[jnp.zeros_like(vpt[0])]+[vpt[j]-vpt[j-1] for j in range(1,n)]
    delta=[jnp.where(jnp.abs(a)<1.e-10,jnp.asarray(1.e-10,a.dtype),a) for a in delta]
    half=[vpt[0]]+[.5*(vpt[j]+vpt[j-1]) for j in range(1,n)]
    energy=tke[k]; work=jnp.zeros_like(energy); active=jnp.asarray(True)
    for j in (range(k+1,n-1) if up else range(k,-1,-1)):
        test0=jnp.where(jnp.signbit(energy),jnp.zeros_like(energy),jnp.ones_like(energy))
        pote=(1. if up else -1.)*test0*(gref[k]*(half[j]-vpt[k]))*dzz[j]
        test=jnp.where(jnp.signbit(energy-pote),jnp.zeros_like(energy),jnp.ones_like(energy))
        off=gref[k]*((vpt[j-1] if up else vpt[j])-vpt[k])
        part=((-off if up else off)+jnp.sqrt(jnp.abs(off**2+2.*energy*gref[k]*delta[j]/dzz[j])))/(gref[k]*delta[j]/dzz[j])
        increment=test0*(test*dzz[j]+(1.-test)*part)
        work=jnp.where(active,work+increment,work)
        energy=jnp.where(active,energy-pote,energy)
        active=active & (test0>0.)
    return work


def _bl89(z,dzz,theta_l,rv,rt,tke,thvm,c):
    n=len(theta_l)
    # Preserve WRF's caller/helper double inversion; PLM is a diagnostic.
    gref=[c.G/(c.G/a) for a in thvm]
    vpt=[theta_l[k]*(1.+(c.RV/c.RD)*rv[k])/(1.+rt[k]) for k in range(n)]
    result=[]
    for k in range(n-1):
        dn=jnp.minimum(_bl89_one(dzz,tke,gref,vpt,k,False),.5*(z[k]+z[k+1])-z[0])
        up=_bl89_one(dzz,tke,gref,vpt,k,True)
        result.append(jnp.maximum((.5*(jnp.maximum(dn,1.e-10)**(-2./3.)+
                                          jnp.maximum(up,1.e-10)**(-2./3.)))**(-1.5),.32))
    result[n-2]=result[n-3]
    result.append(result[n-3])
    return result


def _entr_detr(test,lcl,k,p,z,dzz,thvm,thl,rt,w2,thl_up,rt_up,lup,c):
    """COMPUTE_ENTR_DETR, with dead uninitialized LCL2 predicate made explicit."""
    half=(z[k+1]-z[k])/2.
    pm=p[k] if k==0 else ((p[k]-p[k-1])/dzz[k])*.5*(z[k]-z[k-1])+p[k-1]
    pp=((p[k+1]-p[k])/dzz[k+1])*.5*(z[k+1]-z[k])+p[k]
    mix_th=.1*thl[k]+(1.-.1)*thl_up
    mix_rt=.1*rt[k]+(1.-.1)*rt_up
    mixv1,_=_virtual(p[k],mix_th,mix_rt,c)
    mixv2,_=_virtual(pp,mix_th,mix_rt,c)
    upv1,_=_virtual(pm,thl_up,rt_up,c)
    # WRF aliases ZRCMIX across ALL four calls. These updraft calls are the
    # final writers before the condensation tests (lines824,830,845).
    upv,rcmix=_virtual(p[k],thl_up,rt_up,c)
    upth2,uprv2,rcmix2=_thermo_liquid(pp,thl_up,rt_up,c)
    upv2=upth2*(1.+(c.RV/c.RD)*uprv2)/(1.+rt_up)
    local1=(rcmix>0.) & ~lcl
    coeff=(rcmix2-rcmix)/half
    rc_ord=rcmix-coeff*half
    stop=jnp.where(local1,-rc_ord/coeff,half)
    partdry=jnp.where(local1,stop/(z[k+1]-z[k]),jnp.ones_like(stop))
    minus=jnp.zeros_like(thvm[k]) if k==0 else (thvm[k]-thvm[k-1])/dzz[k]
    plus=(thvm[k+1]-thvm[k])/dzz[k+1]
    upminus=(upv-upv1)/half; upplus=(upv2-upv)/half
    vm=thvm[k] if k==0 else minus*.5*(z[k]-z[k-1])+thvm[k-1]
    vp=plus*.5*(z[k+1]-z[k])+vm  # WRF line874 uses vm here.
    gref=c.G/thvm[k]
    buo=gref*half*(.5*(upminus-minus)*half-vm+upv1)
    b1=gref*stop*(.5*(-minus)*stop-vm+upv1)
    positive=b1>=0.
    entr1=.5/(1.-1.*.55)*jnp.log(1.+(2.*(1.-1.*.55)/w2)*b1)
    detr1=.5/1.*jnp.log(1.+(2.*1./w2)*jnp.maximum(0.,-b1))
    dry=test & ~lcl
    entr=jnp.where(dry & positive,entr1,jnp.zeros_like(entr1))
    detr=jnp.where(dry & ~positive,detr1,jnp.zeros_like(detr1))
    w_half=jnp.where(positive,w2+2.*(1.-1.*.55)*b1,w2+2.*1.*b1)
    stop=half
    buo=buo+gref*half*(.5*(upplus-plus)*half-thvm[k]+upv)
    local2=(rcmix2>0.) & (rcmix<=0.) & ~lcl & ~local1
    temp=upth2*((p[k+1]/1.e5)**(c.RD/c.CPD))
    qsat=jnp.exp(c.ALPW-c.BETAW/temp-c.GAMW*jnp.log(temp))
    qsat=(c.RD/c.RV)*qsat/p[k+1]/(1.+(c.RD/c.RV-1.)*qsat/p[k+1])
    # k+2 is accessed only for a second-half transition. The plume loop ends
    # at n-3, so this is within its native WRF extent.
    coeff2=(rt_up-qsat-rcmix)/(.5*(z[k+2]-z[k+1]))
    rc_ord2=rcmix2-coeff2*half
    stop=jnp.where(local2,-rc_ord2/coeff2,stop)
    partdry=jnp.where(local2,.5+stop/(z[k+1]-z[k]),partdry)
    b2=gref*stop*(.5*(-plus)*stop-thvm[k]+upv)
    use2=dry & ~local1
    alive=w_half>0.
    ent2=.5/(1.-1.*.55)*jnp.log(1.+(2.*(1.-1.*.55)/w_half)*b2)
    det2=.5/1.*jnp.log(1.+(2.*1./w_half)*jnp.maximum(-b2,0.))
    entr=jnp.where(use2 & alive & (b2>=0.),entr+ent2,entr)
    detr=jnp.where(use2 & alive & (b2<0.),detr+det2,detr)
    test=test & ~(use2 & ~alive)
    entr=.55*entr/(z[k+1]-z[k])
    detr=10.*detr/(z[k+1]-z[k])
    detr=jnp.maximum(partdry*1./(lup-.5*(z[k]+z[k+1])),detr)
    kic=jnp.maximum(0.,upv-thvm[k])*.1/(upv-mixv1+1.e-10)
    kic=jnp.maximum(0.,jnp.minimum(1.,kic))
    eps=kic**2.;delta=(1.-kic)**2.
    cloud_factor=(.035*c.G)/50.
    cloud=test & lcl
    entr=jnp.where(cloud,cloud_factor*jnp.minimum(delta,eps),entr)
    detr=jnp.where(cloud,cloud_factor*delta,detr)
    kic2=jnp.maximum(0.,upv2-vp)*.1/(upv2-mixv2+1.e-10)
    kic2=jnp.maximum(0.,jnp.minimum(1.,kic2))
    eps2=kic2**2.;delta2=(1.-kic2)**2.
    transition=test & ~lcl & (local1 | local2)
    entr=jnp.where(transition,entr+(1.-partdry)*cloud_factor*jnp.minimum(delta2,eps2),entr)
    detr=jnp.where(transition,detr+(1.-partdry)*cloud_factor*delta2,detr)
    return test,entr,detr,buo


def _updraft(z,dzz,p,rho,tke,theta,rv,thl,rt,u,v,sfth,sfrv,c):
    n=len(theta)
    flux=lambda a:[a[0]]+[.5*(a[k-1]+a[k]) for k in range(1,n)]
    th_f,thl_f,rt_f,tke_f,p_f,rho_f,rv_f,u_f,v_f=map(
        flux,(theta,thl,rt,tke,p,rho,rv,u,v))
    thvm_f=[th_f[k]*((1.+(c.RV/c.RD)*rv_f[k])/(1.+rt_f[k])) for k in range(n)]
    thvm=[theta[k]*((1.+(c.RV/c.RD)*rv[k])/(1.+rt[k])) for k in range(n)]
    zero=jnp.zeros_like(theta[0])
    up_thl=list(thl_f); up_rt=list(rt_f); up_thv=list(thvm_f)
    up_rv=[zero]*n; up_rc=[zero]*n
    w2=[zero]*n; frac=[zero]*n; emf=[zero]*n; entr=[zero]*n; detr=[zero]*n
    up_thl[0]=thl_f[0]+jnp.maximum(0.,jnp.minimum(2.,sfth/jnp.sqrt(tke_f[0])*.3))
    up_rt[0]=rt_f[0]+jnp.maximum(0.,jnp.minimum(1.e-3,sfrv/jnp.sqrt(tke_f[0])*.3))
    w2[0]=jnp.maximum(.0001,(2./3.)*tke_f[0])
    th,up_rv[0],up_rc[0]=_thermo_liquid(p_f[0],up_thl[0],up_rt[0],c)
    up_thv[0]=th*((1.+(c.RV/c.RD)*up_rv[0])/(1.+up_rt[0]))
    gref=[c.G/a for a in thvm_f]
    tke_f[0]=zero
    lup=jnp.maximum(_bl89_one(dzz,tke_f,gref,thvm_f,0,True),1.e-10)
    wthv=(thvm_f[0]/th_f[0])*sfth+(.61*th_f[0])*sfrv
    active=wthv>0.
    initial=.065*rho_f[0]*((gref[0])*wthv*lup)**(1./3.)
    emf[0]=jnp.where(active,initial,zero)
    frac[0]=jnp.where(active,jnp.minimum(emf[0]/(jnp.sqrt(w2[0])*rho_f[0]),.33),zero)
    w2[0]=jnp.where(active,(emf[0]/(frac[0]*rho_f[0]))**2,w2[0])
    lcl=jnp.asarray(False)
    klcl=jnp.asarray(n-2,jnp.int32); ketl=klcl; kctl=klcl
    entered=[]
    for k in range(n-2):
        before=active
        entered.append(before)
        new_lcl=active & (up_rc[k]>0.) & (lcl == False)
        klcl=jnp.where(new_lcl,k,klcl)
        lcl=lcl | new_lcl
        live,e,d,buo=_entr_detr(active,lcl,k,p,z,dzz,thvm,thl,rt,w2[k],up_thl[k],up_rt[k],lup,c)
        active=jnp.where(before,live,active)
        entr[k]=jnp.where(before,e,entr[k])
        detr[k]=jnp.where(before,d,detr[k])
        if k==0:
            detr[k]=zero
        mix1=.5*(z[k+1]-z[k])*(entr[k]-detr[k])
        emf[k+1]=jnp.where(active,emf[k]*jnp.exp(2.*mix1),emf[k+1])
        dead=active & (emf[k+1]<=0.)
        emf[k+1]=jnp.where(dead,zero,emf[k+1])
        kctl=jnp.where(dead,k+1,kctl)
        active=active & ~dead
        mix2=(z[k+1]-z[k])*entr[k]
        mix3=(z[k+1]-z[k])*detr[k]
        up_thl[k+1]=jnp.where(active,(up_thl[k]*(1.-.5*mix2)+thl[k]*mix2)/(1.+.5*mix2),up_thl[k+1])
        up_rt[k+1]=jnp.where(active,(up_rt[k]*(1.-.5*mix2)+rt[k]*mix2)/(1.+.5*mix2),up_rt[k+1])
        # WRF calls this for the whole column section, including inactive lanes.
        th,new_rv,new_rc=_thermo_liquid(p_f[k+1],up_thl[k+1],up_rt[k+1],c)
        up_rv[k+1]=jnp.where(before,new_rv,up_rv[k+1])
        up_rc[k+1]=jnp.where(before,new_rc,up_rc[k+1])
        up_thv[k+1]=jnp.where(active,th*((1.+(c.RV/c.RD)*up_rv[k+1])/(1.+up_rt[k+1])),up_thv[k+1])
        dryw=jnp.where(buo>0.,w2[k]+2.*(1.-1.*.55)*buo,w2[k]+2.*1.*buo)
        cloudw=w2[k]*(1.-(0.*mix3+1.*mix2))/(1.+(0.*mix3+1.*mix2))+2.*1.*buo/(1.+(0.*mix3+1.*mix2))
        w2[k+1]=jnp.where(active,jnp.where(lcl,cloudw,dryw),w2[k+1])
        etl=active & (buo<=0.)
        ketl=jnp.where(etl,k+1,ketl)
        dead=active & ((w2[k+1]<=0.) | (emf[k+1]<=0.))
        w2[k+1]=jnp.where(dead,zero,w2[k+1]); emf[k+1]=jnp.where(dead,zero,emf[k+1])
        up_thl[k+1]=jnp.where(dead,thl_f[k+1],up_thl[k+1])
        up_rt[k+1]=jnp.where(dead,rt_f[k+1],up_rt[k+1])
        up_rc[k+1]=jnp.where(dead,zero,up_rc[k+1]); up_rv[k+1]=jnp.where(dead,zero,up_rv[k+1])
        up_thv[k+1]=jnp.where(dead,thvm_f[k+1],up_thv[k+1])
        frac[k+1]=jnp.where(dead,zero,frac[k+1])
        kctl=jnp.where(dead,k+1,kctl); active=active & ~dead
        f=emf[k+1]/(jnp.sqrt(w2[k+1])*rho_f[k+1])
        frac[k+1]=jnp.where(active,jnp.minimum(.33,f),frac[k+1])
        frac[k+1]=jnp.where(active & etl & lcl,jnp.minimum(frac[k+1],frac[k]),frac[k+1])
        emf[k+1]=jnp.where(before,frac[k+1]*jnp.sqrt(w2[k+1])*rho_f[k+1],emf[k+1])
    emf[0]=zero; emf[n-1]=zero; frac[n-1]=zero
    depth=_level_pick(z,kctl)-_level_pick(z,klcl)
    coef=jnp.minimum(jnp.maximum(1.-(depth-3000.)/1000.,0.),1.)
    for k in range(n):
        emf[k]=jnp.where(lcl & (depth>3000.),emf[k]*coef,emf[k])
        frac[k]=jnp.where(lcl & (depth>3000.),frac[k]*coef,frac[k])
    return dict(thl=up_thl,rt=up_rt,rv=up_rv,rc=up_rc,thv=up_thv,u=u_f,v=v_f,
                emf=emf,frac=frac,entr=entr,detr=detr,klcl=klcl,entered=entered)


def _massflux_explicit(var,flux,dzz,rhodj,dt):
    """TRIDIAG_MASSFLUX for caller PIMPL=0 (the implicit path is unreachable)."""
    n=len(var)
    mzm=[rhodj[1]]+[.5*(rhodj[k]+rhodj[k-1]) for k in range(1,n)]
    out=[]
    for k in range(n-1):
        rhs=rhodj[k]*var[k]/dt-mzm[k+1]*flux[k+1]/dzz[k+1]+mzm[k]*flux[k]/dzz[k]
        out.append(rhs*dt/rhodj[k])
    out.append(out[-1])
    return out


def mfshconvpbl_column(*,theta,qv,p_mid,exner,dz,rho,tke,u,v,hfx,qfx,
                       rc_mf=None,dt=60.,stepbl=1,_column_kernel=False,_diagnostics=True):
    """Pristine ARW mass-flux tendency producer preceding QNSEPBL.

    RQCBLTEN is zero in WRF's final assignment; its MF cloud content is a
    separate prognostic coupling ``rc_mf``. Returned updraft fields use WRF's
    mass-indexed arrays with the upper dummy level included (n total slots).
    """
    theta=theta if _column_kernel else jnp.asarray(theta)
    dtype=theta.dtype; n=theta.shape[0]
    if theta.ndim!=(2 if _column_kernel else 1) or n<3 or dtype not in (jnp.dtype('float32'),jnp.dtype('float64')):
        raise ValueError('MFSHCONVPBL requires REAL column n >= 3')
    def col(a):
        a=a if _column_kernel else jnp.asarray(a,dtype)
        if a.shape!=theta.shape:raise ValueError('Mass-flux profile shapes must match')
        return list(a)
    profile=theta
    qv,p,pii,dz,rho,tke,u,v=map(col,(qv,p_mid,exner,dz,rho,tke,u,v))
    rc=([jnp.zeros_like(profile[0])]*n if _column_kernel else col(jnp.zeros_like(profile))) if rc_mf is None else col(rc_mf)
    theta=list(theta)
    c=SimpleNamespace(**vars(_constants(str(dtype))),column_kernel=_column_kernel)
    dt=jnp.asarray(dt,dtype)*jnp.asarray(stepbl,dtype)
    hfx=jnp.asarray(hfx,dtype);qfx=jnp.asarray(qfx,dtype)
    zero=jnp.zeros_like(theta[0])
    z=[zero]
    for k in range(n-1):z.append(z[-1]+dz[k])
    zm=[.5*(z[k+1]+z[k]) for k in range(n-1)]+[z[n-1]+.5*dz[n-2]]
    dzz=[2.*zm[0]]+[zm[k]-zm[k-1] for k in range(1,n)]
    rv=[qv[k]-rc[k] for k in range(n)]
    rhoref=[rho[k]/(1.+qv[k]) for k in range(n)]
    rhodj=[rhoref[k]*dz[k] for k in range(n)]
    for a in (theta,tke,rv,rc,u,v,rhoref,pii,p,rhodj):a[-1]=a[-2]
    sfth=hfx/(rhoref[0]*c.CPD);sfrv=qfx/rhoref[0]
    thl=[];rt=[];thvm=[]
    for k in range(n):
        temp=theta[k]*pii[k]
        cp=c.CPD+c.CPV*rv[k]+c.CL*rc[k]
        lv=(c.LV+(c.CPV-c.CL)*(temp-c.TT))/cp
        lv=lv/pii[k]
        thl.append(theta[k]-lv*rc[k]);rt.append(rv[k]+rc[k])
        thvm.append(theta[k]*((1.+(c.RV/c.RD)*rv[k])/(1.+rt[k])))
    plm=_bl89(z,dzz,thl,rv,rt,tke,thvm,c) if _diagnostics else None
    up=_updraft(z,dzz,p,rhoref,tke,theta,rv,thl,rt,u,v,sfth,sfrv,c)
    mzm=lambda a:[a[1]]+[.5*(a[k]+a[k-1]) for k in range(1,n)]
    emf=[up['emf'][k]/rhoref[k] for k in range(n)]
    thlf=mzm(thl);rtf=mzm(rt);thvf=mzm(thvm)
    flth=[emf[k]*(up['thl'][k]-thlf[k]) for k in range(n)]
    flrt=[emf[k]*(up['rt'][k]-rtf[k]) for k in range(n)]
    wthv=[9.81/thvm[k]*emf[k]*(up['thv'][k]-thvf[k]) for k in range(n)]
    thnew=_massflux_explicit(thl,flth,dzz,rhodj,dt)
    rtnew=_massflux_explicit(rt,flrt,dzz,rhodj,dt)
    tht=[(thnew[k]-thl[k])/dt for k in range(n)]
    rtt=[(rtnew[k]-rt[k])/dt for k in range(n)]
    cloud=[]
    for k in range(n-1):
        value=.5*2.75*(up['frac'][k]*up['rc'][k]+up['frac'][k+1]*up['rc'][k+1])
        cloud.append(jnp.where(k>=up['klcl'],value,zero))
    cloud.append(zero)
    stack=(lambda a:a) if _column_kernel else jnp.stack
    zeros=stack([jnp.zeros_like(profile[0])]*n)
    return dict(RTHBLTEN=stack(tht),RQVBLTEN=stack(rtt),RQCBLTEN=zeros,
                RUBLTEN=zeros,RVBLTEN=zeros,
                massflux_EDKF=stack(up['emf']),entr_EDKF=stack(up['entr']),detr_EDKF=stack(up['detr']),
                thl_up=stack(up['thl']),thv_up=stack(up['thv']),rt_up=stack(up['rt']),
                rv_up=stack(up['rv']),rc_up=stack(up['rc']),u_up=stack(up['u']),v_up=stack(up['v']),
                frac_up=stack(up['frac']),rc_mf=stack(cloud),WTHV=stack(wthv),
                PLM_BL89=stack(plm) if _diagnostics else None)
