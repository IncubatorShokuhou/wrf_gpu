"""Literal WRF SMS vertical_diffusion_implicit on the shared halo-memory layout.

Distinct U/V/W/T/TKE equations and WRF's precipitation-number exclusion are
retained. Linear solves are column-local Pallas kernels in WRF Thomas order.
"""
import jax.numpy as jnp
import numpy as np
from gpuwrf.dynamics.les3d_smagorinsky import _G,_CP,_RVOVRD,_EPSILON,_k,_mass_ranges,_ranges_spec
from gpuwrf.kernels.phys_sms_tridiag import solve


def vertical_sms(m,cfg,inp,*,u,v,w,thp,theta,moist,tke,mut,c1h,c2h,c1f,c2f,
                 xkmh,xkmv,xkhv,xkmv_meso,l_diss,nlflux,gamu,gamv,
                 rho,rdz,rdzw,fnm,fnp,dn,dnw,ust,hfx,qfx,t_base,qv_base):
    periodic=cfg.boundary=='periodic';kr=(m.kts,m.ktf);ka=(m.kts+1,m.ktf)
    dt=_k(inp.dt,u)
    ru,rv,rw,rt,tket=m.zeros(),m.zeros(),m.zeros(),m.zeros(),m.zeros()
    mt=[m.zeros() for _ in moist]
    # Momentum U/V: density*K averages on interfaces; fixed top, Robin drag bottom.
    for which in ('u','v'):
        if which=='u':
            jr,ir=_ranges_spec(m,periodic,i_end_off=1,j_end_off=2,i_px_end_off=0,j_base_end=min(m.jte,m.jde-1))
            shift={'di':-1};var=u;gamma=gamu
            cross=.25*(m.r(v,(1,1),jr,ir)+m.r(v,(1,1),jr,ir,dj=1)+m.r(v,(1,1),jr,ir,di=-1)+m.r(v,(1,1),jr,ir,di=-1,dj=1))
        else:
            jr,ir=_ranges_spec(m,periodic,i_end_off=2,j_end_off=1,i_px_end_off=1,i_base_end=min(m.ite,m.ide-1))
            shift={'dj':-1};var=v;gamma=gamv
            cross=.25*(m.r(u,(1,1),jr,ir)+m.r(u,(1,1),jr,ir,dj=-1)+m.r(u,(1,1),jr,ir,di=1)+m.r(u,(1,1),jr,ir,di=1,dj=-1))
        r=lambda a,kk=kr,**s:m.r(a,kk,jr,ir,**s)
        rr=.5*(m.v1(fnm,ka)*(r(rho,ka)+r(rho,ka,**shift))+m.v1(fnp,ka)*(r(rho,ka,dk=-1)+r(rho,ka,dk=-1,**shift)))
        avg=lambda kv:.5*rr*(m.v1(fnm,ka)*(r(kv,ka)+r(kv,ka,**shift))+m.v1(fnp,ka)*(r(kv,ka,dk=-1)+r(kv,ka,dk=-1,**shift)))
        km,ks=m.zeros(),m.zeros()
        km=m.s(km,ka,jr,ir,avg(xkmv_meso));ks=m.s(ks,ka,jr,ir,avg(xkmv))
        rz=2./(1./r(rdz)+1./r(rdz,**shift))
        mu=.5*((m.v1(c1h,kr)*m.r2(mut,jr,ir)+m.v1(c2h,kr))+(m.v1(c1h,kr)*m.r2(mut,jr,ir,**shift)+m.v1(c2h,kr)))
        us=.5*(m.r2(ust,jr,ir)+m.r2(ust,jr,ir,**shift))
        v0=jnp.sqrt(r(var,(1,1))**2+cross**2)+_EPSILON
        tao=us*us*(r(rho,(1,1))+r(rho,(1,1),**shift))/(2.*v0)
        factor=-_G/m.v1(dnw,kr)/mu
        rznext=2./(1./r(rdz,dk=1)+1./r(rdz,dk=1,**shift))
        a=-factor*rz*r(ks)*dt
        b=1.+factor*(rznext*r(ks,dk=1)+rz*r(ks))*dt
        c=-factor*rznext*r(ks,dk=1)*dt
        gg=.5*(m.r2(gamma,jr,ir)+m.r2(gamma,jr,ir,**shift))
        d=-factor*(r(km,dk=1)-r(km))*dt*gg+r(var)
        a=a.at[0].set(0.);b=b.at[0].set(1.+factor[0]*(rznext[0]*r(ks,dk=1)[0]+tao[0])*dt);d=d.at[0].set(r(var)[0])
        a=a.at[-1].set(0.);b=b.at[-1].set(1.);c=c.at[-1].set(0.);d=d.at[-1].set(r(var)[-1])
        sol=solve(a,b,c,d);tend=mu*(sol-r(var))/dt
        if which=='u':ru=m.s(ru,kr,jr,ir,tend)
        else:rv=m.s(rv,kr,jr,ir,tend)
    # W has fixed surface/lid and only the interior faces are unknowns.
    jr,ir=_mass_ranges(m,periodic)
    r=lambda a,kk=kr,**s:m.r(a,kk,jr,ir,**s)
    mu=lambda kk:m.v1(c1h,kk)*m.r2(mut,jr,ir)+m.v1(c2h,kk)
    ff=-_G/m.v1(dn,ka)/(m.v1(c1f,ka)*m.r2(mut,jr,ir)+m.v1(c2f,ka))
    kw=xkmh*rho
    a=-ff*r(rdzw,ka,dk=-1)*r(kw,ka,dk=-1)*dt
    b=1.+ff*(r(rdzw,ka)*r(kw,ka)+r(rdzw,ka,dk=-1)*r(kw,ka,dk=-1))*dt
    c=-ff*r(rdzw,ka)*r(kw,ka)*dt
    sol=solve(a,b,c,r(w,ka))
    rw=m.s(rw,ka,jr,ir,(m.v1(c1f,ka)*m.r2(mut,jr,ir)+m.v1(c2f,ka))*(sol-r(w,ka))/dt)
    rhof=m.v1(fnm,ka)*r(rho,ka)+m.v1(fnp,ka)*r(rho,ka,dk=-1)
    def interfaces(kv,tke_double=False):
        out=m.zeros()
        vals=(m.v1(fnm,ka)*2.*r(kv,ka)+m.v1(fnp,ka)*2.*r(kv,ka,dk=-1)) if tke_double else (m.v1(fnm,ka)*r(kv,ka)+m.v1(fnp,ka)*r(kv,ka,dk=-1))
        return m.s(out,ka,jr,ir,vals*rhof)
    fac=-_G/m.v1(dnw,kr)/mu(kr)
    kheat=interfaces(xkhv)
    def scalar_matrix(var,kavg):
        a=-fac*r(rdz)*r(kavg)*dt
        b=1.+fac*(r(rdz,dk=1)*r(kavg,dk=1)+r(rdz)*r(kavg))*dt
        c=-fac*r(rdz,dk=1)*r(kavg,dk=1)*dt
        a=a.at[0].set(0.);b=b.at[0].set(1.+fac[0]*r(rdz,dk=1)[0]*r(kavg,dk=1)[0]*dt)
        a=a.at[-1].set(0.);b=b.at[-1].set(1.);c=c.at[-1].set(0.)
        return a,b,c,r(var)
    var=thp if cfg.mix_full_fields else thp-m.v1(t_base,(m.kts,m.kde))
    a,b,c,d=scalar_matrix(var,kheat)
    # The interior qfac deliberately retains lowest-level moisture (WRF caller).
    qfac=1.+_RVOVRD*r(moist[cfg.iqv],(1,1)) if cfg.use_theta_m==1 else jnp.ones_like(r(theta,(1,1)))
    nlr=m.s(m.zeros(),ka,jr,ir,r(nlflux,ka)*rhof)
    d=-fac*(r(nlr,dk=1)-r(nlr))*qfac*dt+d
    cpm=_CP*(1.+.8*r(moist[cfg.iqv],(1,1)))
    h=m.r2(hfx,jr,ir)[0]
    if cfg.isfflx in (0,2):h=np.float32(cfg.tke_heat_flux)*cpm[0]*r(rho,(1,1))[0]
    flux=qfac*h[None]/cpm+(np.float32(1.61)*r(theta,(1,1))*m.r2(qfx,jr,ir) if cfg.use_theta_m==1 else 0.)
    d=d.at[0].set(r(var)[0]-dt*fac[0]*r(nlr,dk=1)[0]*qfac[0]+dt*fac[0]*flux[0])
    d=d.at[-1].set(r(var)[-1]);sol=solve(a,b,c,d)
    rt=m.s(rt,kr,jr,ir,mu(kr)*(sol-r(var))/dt)
    # TKE has doubled interface K AND explicit factor2 in its matrix, literal WRF.
    kt=interfaces(xkmv,True);ldiss=r(l_diss);tk=r(tke)
    beta=jnp.where(ldiss!=0,1.5*jnp.sqrt(tk)/jnp.where(ldiss!=0,ldiss,1.),0.)
    a=-2.*r(kt)*dt*fac*r(rdz)
    b=1.+2.*dt*fac*(r(rdz)*r(kt)+r(rdz,dk=1)*r(kt,dk=1))+dt*beta
    c=-2.*r(kt,dk=1)*dt*r(rdz,dk=1)*fac
    d=tk+jnp.where(ldiss!=0,.5*dt*jnp.power(tk,np.float32(1.5))/jnp.where(ldiss!=0,ldiss,1.),0.)
    a=a.at[-1].set(0.);b=b.at[-1].set(1.);c=c.at[-1].set(0.);d=d.at[-1].set(0.)
    sol=solve(a,b,c,d);tket=m.s(tket,kr,jr,ir,mu(kr)*(sol-tk)/dt)
    for n,q in enumerate(moist):
        if n in cfg.sms_precip_scalar_indices:continue
        var=q if cfg.mix_full_fields or n!=cfg.iqv else q-m.v1(qv_base,(m.kts,m.kde))
        a,b,c,d=scalar_matrix(var,kheat)
        if n==cfg.iqv:d=d.at[0].set(d[0]+dt*fac[0]*m.r2(qfx,jr,ir)[0])
        sol=solve(a,b,c,d);mt[n]=m.s(mt[n],kr,jr,ir,mu(kr)*(sol-r(var))/dt)
    return ru,rv,rw,rt,mt,tket
