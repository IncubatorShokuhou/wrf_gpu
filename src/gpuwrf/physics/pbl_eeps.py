"""Experimental REAL32 EEPS from unchanged WRF 4.7.1 module_bl_eepsilon.F.

The active top-down path explicitly mirrors compiler-zero initialization of
upstream radsum(:):748. Full-column CPU parity is qualified at frozen bounds;
alternate initialization materially affects outputs. GPU and coupled-forecast
qualification remain pending. The driver reverses bottom-up inputs for EEPS.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.physics.pbl_keps import _real32


@_real32
def tridiag(a, b, c, rhs):
    """Literal EEPS TRIDIAG :1053–1090, reverse then forward sweep.

    WRF's a is the upper diagonal and c the lower diagonal. WRF overwrites
    c(1)=0 before the solve. Return (solution, modified c), preserving that
    side effect for caller/oracle checks. Leading batch axes use jax.vmap.
    """
    n = rhs.shape[0]
    c = c.at[0].set(0.)
    q = jnp.zeros_like(rhs).at[-1].set(-c[-1]/b[-1])
    d = rhs.at[-1].set(rhs[-1]/b[-1])
    def reverse(i, carry):
        q, d = carry
        k = n-2-i
        p = 1./(b[k]+a[k]*q[k+1])
        q = q.at[k].set(-c[k]*p)
        d = d.at[k].set((d[k]-a[k]*d[k+1])*p)
        return q, d
    q, d = jax.lax.fori_loop(0, n-1, reverse, (q, d))
    def forward(k, d):
        return d.at[k].set(d[k]+q[k]*d[k-1])
    return jax.lax.fori_loop(1, n, forward, d), c


@_real32
def _eeps2d_column(u, v, temperature, qv, qc, qi, qr, qs, qg, pressure, z, zw,
                   rho, raten, tke, epsilon, dt, ustar, rmol, wspd, tsk, qsfc,
                   psfc, hfx, qfx):
    """Literal top-down EEPS2D; q* inputs here are specific humidities."""
    n = u.shape[0]
    g, cp = jnp.float32(9.81), jnp.float32(1004.5)
    rd, rv = jnp.float32(287.), jnp.float32(461.6)
    ep1, ep2, rcp = rv/rd-1., rd/rv, rd/cp
    mdt = jnp.int32(jnp.sqrt(dt/9.)+.00001)+1
    dtt = dt/mdt
    dzz = zw[:-1]-zw[1:]
    doz = z[:-1]-z[1:]
    alt, amt = dt/dzz, dtt/doz
    up2 = g/jnp.maximum(.1, u*u+v*v)
    psp = (pressure/100000.)**rcp
    pt = temperature/psp
    cpm = cp*(1.+.8*qv)
    szv = pt*(1.+ep1*qv)
    pci = qc+qi
    fac = qc+qi+qr+qs+qg
    cx = 1.+qv+fac
    alh = 3.1484e6-2.37e3*temperature
    water = jnp.minimum(.5*pressure, 611.2*jnp.exp(17.67*(temperature-273.15)/(temperature-29.65)))
    ice = jnp.minimum(.5*pressure, 611.*jnp.exp(22.514*(temperature-273.16)/temperature))
    qsw, qsi = .622*water/(pressure-water), .622*ice/(pressure-ice)
    fraction = jnp.where(temperature >= 273.15, 0., jnp.where(temperature <= 233.15, 1.,
                jnp.where(qi <= 1.e-20, 0., jnp.where(qc <= 1.e-20, 1., qi/jnp.maximum(qc+qi, 1.e-30)))))
    qsat = (1.-fraction)*qsw+fraction*qsi
    latent = (1.-fraction)*alh+fraction*2850000.
    rdz = 1./doz
    du, dv = u[:-1]-u[1:], v[:-1]-v[1:]
    sh = jnp.concatenate(((du*du+dv*dv)*(rdz*rdz), jnp.zeros_like(u[:1])))
    dry_st = g*jnp.log(szv[:-1]/szv[1:])*rdz
    tk, qvk, alk = .5*(temperature[1:]+temperature[:-1]), .5*(qv[1:]+qv[:-1]), .5*(latent[1:]+latent[:-1])
    af, ag = alk*qvk/(rd*tk), alk/(cp*tk)
    aco = (1.+af)/(1.+ep2*af*ag)
    wet_st = (aco*(jnp.log(pt[:-1]/pt[1:])+ag*(qv[:-1]-qv[1:]))-(cx[:-1]-cx[1:]))*g*rdz
    st = jnp.concatenate((jnp.where((qv[:-1]<qsat[:-1])|(qv[1:]<qsat[1:]), dry_st, wet_st), jnp.zeros_like(u[:1])))
    rich = st/jnp.maximum(1.e-7, sh)
    def diagnose(reference, old_height, old_level):
        bulk = z*(szv/reference-1.)*up2
        def search(i, found):
            k = n-2-i
            return jnp.where((found==n-1)&(bulk[k]>=.25), k, found)
        level = jax.lax.fori_loop(0, n-5, search, jnp.int32(n-1))
        # The no-crossing path keeps the previous estimate during the second diagnosis.
        k = jnp.minimum(level, n-2)
        dum = (z[k]-z[k+1])/(bulk[k]-bulk[k+1])
        height = jnp.minimum(jnp.maximum(z[k+1],z[k]-(bulk[k]-.25)*dum),z[k])
        return jnp.where(level<n-1,height,old_height),jnp.where(level<n-1,level,old_level)
    height, kpbl = diagnose(szv[-1], z[-1], jnp.int32(n-1))
    density = psfc/(rd*temperature[-1]*(1.+ep1*qv[-1]))
    hs, hq = hfx/(density*cpm[-1]), qfx/density
    duj = ustar*ustar
    eps = epsilon.at[-1].set(jnp.clip(duj*ustar/(.4*z[-1]),1.e-7,25.))
    sflux = hs+hq*ep1*pt[-1]
    wstar3 = g/pt[-1]*jnp.maximum(sflux,0.)*height
    third = jnp.float32(1.)/jnp.float32(3.)
    wstar2 = (wstar3**2.)**third
    sfa = tsk*(100000./psfc)**rcp
    rich = rich.at[-1].set(g*z[-1]*jnp.log(szv[-1]/(sfa*(1.+ep1*qsfc)))/(wspd**2.))
    convective = (rmol<0.)&(sflux>0.)
    bottom = jnp.where(convective,3.75*duj+.2*wstar2+duj*((-z[-1]*rmol)**2.)**third,3.75*duj)
    te = tke.at[-1].set(jnp.clip(bottom,1.e-4,160.))
    reference = szv[-1]+jnp.minimum(5.,5.*sflux/jnp.maximum(.1,jnp.sqrt(wstar2)))
    second_h, second_k = diagnose(reference,height,kpbl)
    height, kpbl = jnp.where(convective,second_h,height),jnp.where(convective,second_k,kpbl)
    denominator = height*jnp.maximum(.1,jnp.sqrt(wstar2))
    gamat = jnp.where(convective,jnp.maximum(5.*hs/denominator,0.),0.)
    gamaq = jnp.where(convective,jnp.maximum(5.*hq/denominator,0.),0.)
    production, cloud_active = _topdown(temperature,qv,qc,qi,pressure,z,rho,raten,szv,psp,height,kpbl,ep2)
    return _integrate(u,v,pt,qv,qc,qi,temperature,psp,cpm,z,dzz,doz,alt,amt,
                      te,eps,sh,st,rich,szv,height,gamat,gamaq,production,dt,dtt,mdt,
                      ustar,wspd,hs,hq,kpbl,cloud_active)


def _topdown(t,qv,qc,qi,p,z,rho,rad,thv,exner,height,kpbl,ep2):
    """EEPS :640–779; explicit zero at upstream uninitialized radsum :748."""
    n=t.shape[0]
    def pressure_level(limit):
        def body(i,level):
            k=n-1-i
            return jnp.where((level==n-1)&(p[-1]-p[k]>limit),k,level)
        return jax.lax.fori_loop(0,n-1,body,jnp.int32(n-1))
    k700,k950=pressure_level(30000.),pressure_level(5000.)
    def enthalpy(k):
        return 1004.5*t[k]*(1.+5.87*(qv[k]+(qc[k]+qi[k])))-2500000.*qc[k]-2850000.*qi[k]+9.81*z[k]
    eis=jnp.maximum(enthalpy(k700)-enthalpy(k950),enthalpy(k950)-enthalpy(n-1))/1004.5
    allowed=eis>=5.
    def cloud_search(k,level):
        return jnp.where((level==n-1)&allowed&(k>=k700)&(qc[k]+qi[k]>=1.e-6),k,level)
    kcloud=jax.lax.fori_loop(k700,n,cloud_search,jnp.int32(n-1))
    def cooling(k,c):
        minimum,level=c
        hit=allowed&(k>=kcloud)&(qc[k]+qi[k]>=1.e-6)&(rad[k]<minimum)
        return jnp.where(hit,rad[k],minimum),jnp.where(hit,k,level)
    minrad,kmin=jax.lax.fori_loop(k700,n,cooling,(jnp.float32(100.),jnp.int32(n-1)))
    active=allowed&(minrad<0.)&(kcloud>=kpbl)&(kmin<n-2)
    def enabled(_):
        templ=t[kmin]
        saturation=100.*6.112*jnp.exp(17.67*(templ-273.16)/(templ-29.65))*(ep2/p[kmin])
        temps=templ+(qv[kmin]+(qc[kmin]+qi[kmin])-saturation)/(1004.5/2500000.+ep2*2500000.*saturation/(287.*templ**2.))
        saturation=100.*6.112*jnp.exp(17.67*(temps-273.15)/(temps-29.65))*(ep2/p[kmin])
        rcldb=jnp.maximum(qv[kmin]+(qc[kmin]+qi[kmin])-saturation,0.)
        dthv=jnp.maximum(thv[kmin-2]-thv[kmin],.1)
        efficiency=.2+.2*8.*(2500000./1004.5*rcldb/(exner[kmin]*dthv))
        def sum_cooling(i,total):
            k=n-1-i
            flux=rad[k]*exner[k]
            flux=flux*1004.5/9.81*(p[k]-p[k-1])
            return jnp.where((k<=kmin+1)&(k>=kcloud)&(flux<0.),jnp.abs(flux)+total,total)
        # WRF EEPS :748 reads radsum before assignment; manager-defined zero.
        radsum=jax.lax.fori_loop(0,n-k700,sum_cooling,jnp.float32(0.))
        buoyancy=jnp.maximum(jnp.minimum(radsum,120.)/rho[kmin]/1004.5,0.)
        wm3=9.81/thv[kmin]*buoyancy*jnp.minimum(height,1500.)
        zfac=jnp.clip(1.-(z-z[-1])/(z[kmin]-z[-1]),.01,1.)
        zent=10.*jnp.maximum((z[kmin]-z)/z[kmin],0.)*(1.-zfac)**3
        production=jnp.maximum(2.*efficiency*wm3/jnp.maximum(height,100.)*zent,0.)
        return jnp.where(jnp.arange(n)>=kcloud,production,0.)
    return jax.lax.cond(active,enabled,lambda _:jnp.zeros_like(t),None),active


def _integrate(u,v,pt,qv,qc,qi,t,psp,cpm,z,dzz,doz,alt,amt,te,eps,sh,st,rich,
               thv,height,gamat,gamaq,production,dt,dtt,mdt,ustar,wspd,hs,hq,kpbl,cloud_active):
    """EEPS :789–1048: prognostic solve, mean diffusion, dissipation/melting."""
    richf=jnp.where(rich<.145,.6588*(rich+.1776-jnp.sqrt(jnp.maximum(
        rich*rich-.3221*rich+.03156,0.))),.1620092)
    richf=jnp.clip(richf,-1000.,.1620092)
    factor=jnp.where(richf<.1620092,1.318*(.2231-richf)/(.2341-richf),1.116893)
    def substep(_,carry):
        te,eps=carry
        km=.09*te*te/eps
        kh=factor*km
        km=km.at[:-1].set(jnp.minimum(1600.,.1+km[:-1]))
        kh=kh.at[:-1].set(jnp.minimum(1600.,.01+kh[:-1]))
        # WRF :806 uses REAL constants .1/C3; do not fold them in Python fp64.
        drain_ratio=jnp.float32(.1)/jnp.float32(1.44)
        edrain=jnp.where(st[:-1]>0.,drain_ratio*jnp.minimum(1.,jnp.sqrt(jnp.maximum(rich[:-1]/.145,0.)))*jnp.sqrt(jnp.maximum(st[:-1],0.))*te[:-1],0.)
        correction=jnp.where(z[:-1]>height,st[:-1],st[:-1]-2.*9.81*gamat/(thv[1:]+thv[:-1]))
        src=km[:-1]*sh[:-1]-kh[:-1]*correction+production[:-1]
        # Preserve WRF's association for the second source (edrain before TD).
        src1=km[:-1]*sh[:-1]-kh[:-1]*correction+edrain+production[:-1]
        am=jnp.concatenate(((.5*km[0]/dzz[0])[None],.5*(km[:-1]+km[1:])/dzz[1:]))
        dum=(1.44*src1-1.92*eps[:-1])*dtt/te[:-1]
        a=-.77*amt*am[1:];c=-.77*amt*am[:-1]
        b=1.+.77*amt*(am[:-1]+am[1:])-jnp.where(dum<=.5,dum,0.)
        rhs=jnp.where(dum<=.5,eps[:-1],eps[:-1]+dum*eps[:-1])
        a=a.at[-1].set(0.);c=c.at[0].set(0.)
        rhs=rhs.at[-1].add(.77*amt[-1]*am[-1]*eps[-1])
        next_eps=tridiag(a,b,c,rhs)[0]
        eps=eps.at[:-1].set(jnp.clip(next_eps,1.e-7,25.))
        a=-1.35*amt*am[1:];c=-1.35*amt*am[:-1]
        b=1.+1.35*amt*(am[:-1]+am[1:])
        rhs=te[:-1]+(src-eps[:-1])*dtt
        a=a.at[-1].set(0.);c=c.at[0].set(0.)
        rhs=rhs.at[-1].add(1.35*amt[-1]*am[-1]*te[-1])
        te=te.at[:-1].set(jnp.clip(tridiag(a,b,c,rhs)[0],1.e-4,160.))
        return te,eps
    te,eps=jax.lax.fori_loop(0,mdt,substep,(te,eps))
    km=jnp.minimum(1600.,.1+.09*te[:-1]*te[:-1]/eps[:-1])
    kh=jnp.minimum(1600.,.01+factor[1:]*(.09*te[:-1]*te[:-1]/eps[:-1]))
    pnt=jnp.where(z[1:]>height,0.,kh)
    km=jnp.pad(km/doz,(1,1));kh=jnp.pad(kh/doz,(1,1));pnt=jnp.pad(pnt,(1,1))
    a=-alt*km[1:];c=-alt*km[:-1];b=1.+alt*(km[:-1]+km[1:])
    b=b.at[-1].add(alt[-1]*ustar*ustar/wspd)
    un=tridiag(a,b,c,u)[0];vn=tridiag(a,b,c,v)[0]
    a=-alt*kh[1:];c=-alt*kh[:-1];b=1.+alt*(kh[:-1]+kh[1:])
    previous=jnp.concatenate((jnp.zeros_like(eps[:1]),eps[:-1]))
    previous=jnp.where(previous<=1.e-6,0.,previous)
    current=jnp.where(eps<=1.e-6,0.,eps)
    dissipative_heat=.5*(previous+current)/cpm
    rhs=pt+dissipative_heat*dt/psp+alt*gamat*(pnt[1:]-pnt[:-1])
    rhs=rhs.at[-1].add(alt[-1]*hs)
    thn=tridiag(a,b,c,rhs)[0]
    rhs=qv+alt*gamaq*(pnt[1:]-pnt[:-1])
    rhs=rhs.at[-1].add(alt[-1]*hq)
    qvn=tridiag(a,b,c,rhs)[0]
    qcn=tridiag(a,b,c,qc)[0];qin=tridiag(a,b,c,qi)[0]
    tn=thn*psp
    qvn,qcn,qin=jnp.maximum(1.e-20,qvn),jnp.maximum(1.e-20,qcn),jnp.maximum(1.e-20,qin)
    melt=(tn>=273.15)&(qin>1.e-20)
    new_cloud=jnp.where(qcn<=1.e-20,qin,qcn+qin)
    latent=3.1484e6-2.37e3*tn
    tn=jnp.where(melt,tn-(2850000.-latent)*qin/cpm,tn)
    qcn=jnp.where(melt,new_cloud,qcn);qin=jnp.where(melt,1.e-20,qin)
    exch_m=jnp.pad(km[1:-1]*doz,(1,1));exch_h=jnp.pad(kh[1:-1]*doz,(1,1))
    return dict(u=un,v=vn,temperature=tn,qv=qvn,qc=qcn,qi=qin,tke=te,epsilon=eps,
                exch_m=exch_m,exch_h=exch_h,pblh=height,kpbl=kpbl,cloud_active=cloud_active)


@_real32
def eeps_column(u,v,theta,temperature,pressure,qv,qc,qi,qr,qs,qg,rho,dz,exner,
                raten,tke,epsilon,tke_adv,epsilon_adv,dt,step,hfx,qfx,ustar,
                rmol,wspd,tsk,qsfc,psfc,dx):
    """Full EEPS driver, bottom-up mixing ratios; private source switches=1."""
    del dx  # Passed by WRF, unused by the original module.
    def prefix(previous,width):
        value=previous+width
        return value,value
    zf=jnp.concatenate((jnp.zeros_like(dz[:1]),jax.lax.scan(prefix,jnp.float32(0.),dz)[1]))
    z=.5*(zf[:-1]+zf[1:])
    specific=lambda q:(q/(1.+q))[::-1]
    te=jnp.clip(jnp.where(step==1,tke,tke_adv),1.e-4,160.)
    ep=jnp.clip(jnp.where(step==1,epsilon,epsilon_adv),1.e-7,25.)
    out=_eeps2d_column(u[::-1],v[::-1],temperature[::-1],specific(qv),specific(qc),specific(qi),
        specific(qr),specific(qs),specific(qg),pressure[::-1],z[::-1],zf[::-1],rho[::-1],raten[::-1],
        te[::-1],ep[::-1],dt,ustar,rmol,wspd,tsk,qsfc,psfc,hfx,qfx)
    un,vn,tn=out['u'][::-1],out['v'][::-1],out['temperature'][::-1]
    mixing=lambda key:out[key][::-1]/(1.-out[key][::-1])
    qvn,qcn,qin=mixing('qv'),mixing('qc'),mixing('qi')
    zero=jnp.zeros_like(u)
    # Original driver :224/:320–325 rounds 1/dt once, then multiplies.
    rdelt=jnp.float32(1.)/dt
    return dict(rublten=(un-u)*rdelt,rvblten=(vn-v)*rdelt,rthblten=(tn-temperature)/exner*rdelt,
                rqvblten=(qvn-qv)*rdelt,rqcblten=(qcn-qc)*rdelt,rqiblten=(qin-qi)*rdelt,
                tke=out['tke'][::-1],epsilon=out['epsilon'][::-1],
                tke_adv=out['tke'][::-1],epsilon_adv=out['epsilon'][::-1],
                exch_m=out['exch_m'][::-1][:-1],exch_h=out['exch_h'][::-1][:-1],
                wu=zero,wv=zero,wt=zero,wq=zero,pblh=out['pblh'],kpbl=u.shape[0]-out['kpbl'],
                cloud_active=out['cloud_active'])
