"""WRF lake thermal diffusion, phase changes and convective mixing (r8)."""
from __future__ import annotations

from jax import numpy as jnp
from .lake_fluxes import TFRZ, CPICE, CPLIQ, HFUS, TKWAT, TKICE, TKAIR, GRAV, VKC, real
from .lake_thermodynamics import lake_tridiagonal


def soil_thermal(s):
    dz, t, wi, wl = s.dz3d,s.t_soisno3d,s.h2osoi_ice3d,s.h2osoi_liq3d
    fl = wl[...,5:]/(wi[...,5:]+wl[...,5:])
    frozen = s.tkmg3d*.249**(fl*s.watsat3d)*2.29**s.watsat3d
    soil = jnp.where(t[...,5:] >= TFRZ,s.tksatu3d,frozen)
    active = jnp.arange(-4,1) > s.snl2d[...,None]
    bw = (wi[...,:5]+wl[...,:5])/jnp.where(active,dz[...,:5],1.)
    snow = TKAIR+(7.75e-5*bw+1.105e-6*bw*bw)*(TKICE-TKAIR)
    thk = jnp.concatenate((snow,soil),axis=-1)
    z,zi = s.z3d,s.zi3d
    interface = thk[...,:-1]*thk[...,1:]*(z[...,1:]-z[...,:-1])/(thk[...,:-1]*(z[...,1:]-zi[...,1:-1])+thk[...,1:]*(zi[...,1:-1]-z[...,:-1]))
    tk = jnp.concatenate((interface,jnp.zeros_like(soil[...,:1])),axis=-1)
    tk = tk.at[...,4].set(snow[...,4])
    cvsoil = s.csol3d*(1.-s.watsat3d)*dz[...,5:]+(wi[...,5:]*CPICE+wl[...,5:]*CPLIQ)
    cvs = CPLIQ*wl[...,:5]+CPICE*wi[...,:5]
    cv = jnp.concatenate((jnp.where(active,cvs,0.),cvsoil),axis=-1)
    return tk,cv,soil[...,0]


def phase_change(s,cv,cvl):
    tl,ice,ts,wi,wl = s.t_lake3d,s.lake_icefrac3d,s.t_soisno3d,s.h2osoi_ice3d,s.h2osoi_liq3d
    thin = (s.snl2d == 0)&(s.h2osno2d > 0.)&(tl[...,0] > TFRZ)
    avail = (tl[...,0]-TFRZ)*cvl[...,0]
    melt = jnp.where(thin,jnp.minimum(s.h2osno2d,avail/HFUS),0.)
    rem = jnp.maximum(avail-melt*HFUS,0.)
    tl = tl.at[...,0].set(jnp.where(thin,TFRZ+rem/cvl[...,0],tl[...,0]))
    sd = jnp.where(thin,s.snowdp2d*(1.-melt/jnp.where(thin,s.h2osno2d,1.)),s.snowdp2d)
    swe = s.h2osno2d-melt
    swe = jnp.where(thin&(swe < 1e-7),0.,swe)
    sd = jnp.where(thin&(sd < 1e-7),0.,sd)
    snomelt = melt
    melting,freezing = (tl > TFRZ)&(ice > 0.),(tl < TFRZ)&(ice < 1.)
    change = melting|freezing
    avail = (tl-TFRZ)*cvl
    amount = jnp.where(melting,jnp.minimum(ice*1000.*s.dz_lake3d,avail/HFUS),jnp.maximum(-(1.-ice)*1000.*s.dz_lake3d,avail/HFUS))
    rem = jnp.where(melting,jnp.maximum(avail-amount*HFUS,0.),jnp.minimum(avail-amount*HFUS,0.))
    updated_ice = ice-amount/(1000.*s.dz_lake3d)
    updated_ice = jnp.where(updated_ice > 1.-1e-7,1.,updated_ice)
    updated_ice = jnp.where(updated_ice < 1e-7,0.,updated_ice)
    ice = jnp.where(change,updated_ice,ice)
    cvl = jnp.where(change,cvl+amount*(CPLIQ-CPICE),cvl)
    tl = jnp.where(change,TFRZ+rem/cvl,tl)
    active = jnp.arange(-4,11) > s.snl2d[...,None]
    melting,freezing = active&(ts > TFRZ)&(wi > 0.),active&(ts < TFRZ)&(wl > 0.)
    change = melting|freezing
    avail = (ts-TFRZ)*cv
    amount = jnp.where(melting,jnp.minimum(wi,avail/HFUS),jnp.maximum(-wl,avail/HFUS))
    rem = jnp.where(melting,jnp.maximum(avail-amount*HFUS,0.),jnp.minimum(avail-amount*HFUS,0.))
    wi = jnp.where(change,wi-amount,wi)
    wl = jnp.where(change,wl+amount,wl)
    cv = jnp.where(change,cv+amount*(CPLIQ-CPICE),cv)
    ts = jnp.where(change,TFRZ+rem/jnp.where(change,cv,1.),ts)
    wi = jnp.where(change&(wi < 1e-7),0.,wi)
    wl = jnp.where(change&(wl < 1e-7),0.,wl)
    imelt = jnp.where(melting,1,jnp.where(freezing,2,0)).astype(jnp.int32)
    for j in range(5):
        snomelt = snomelt+jnp.where(change[...,j],amount[...,j],0.)
    return s._replace(t_lake3d=tl,lake_icefrac3d=ice,t_soisno3d=ts,h2osoi_ice3d=wi,h2osoi_liq3d=wl,h2osno2d=swe,snowdp2d=sd),snomelt,imelt


def density(t,ice):
    return (1.-ice)*1000.*(1.-1.9549e-5*jnp.abs(t-277.)**1.68)+ice*917.


def energy(s,cv,cvl):
    result = jnp.zeros_like(s.snl2d)
    for j in range(10):
        result = result+cvl[...,j]*(s.t_lake3d[...,j]-TFRZ)+HFUS*1000.*s.dz_lake3d[...,j]*(1.-s.lake_icefrac3d[...,j])
    for j in range(15):
        active = j > s.snl2d+4
        result = jnp.where(active,result+cv[...,j]*(s.t_soisno3d[...,j]-TFRZ)+HFUS*s.h2osoi_liq3d[...,j],result)
        if j == 5:
            result = jnp.where((s.snl2d == 0)&(s.h2osno2d > 0.),result-s.h2osno2d*HFUS,result)
    return result


def thermal_step(s,f,flux):
    s = s._replace(t_grnd2d=flux.ground_temperature)
    tl,ice = s.t_lake3d,s.lake_icefrac3d
    cwat,cice = CPLIQ*1000.,CPICE*1000.
    tkice_eff,km = TKICE*917./1000.,TKWAT/cwat
    rho = density(tl,ice)
    zl,dl = s.z_lake3d,s.dz_lake3d
    drhodz = (rho[...,1:]-rho[...,:-1])/(zl[...,1:]-zl[...,:-1])
    n2 = GRAV/rho[...,:-1]*drhodz
    num = 40.*n2*(VKC*zl[...,:-1])**2
    den = jnp.maximum(flux.ws[...,None]**2*jnp.exp(-2.*flux.ks[...,None]*zl[...,:-1]),1e-10)
    ri = (-1.+jnp.sqrt(jnp.maximum(1.+num/den,0.)))/20.
    deep = s.lakedepth2d > 15.
    multiplier = jnp.where(tl[...,0] > 277.15,jnp.where(deep,1e2,1.),jnp.where(deep,jnp.where(s.lakedepth2d > 150.,1e5,1e4),1.))
    ke = multiplier[...,None]*VKC*flux.ws[...,None]*zl[...,:-1]*jnp.exp(-flux.ks[...,None]*zl[...,:-1])/(1.+37.*ri*ri)
    opened = (s.t_grnd2d > TFRZ)&(tl[...,0] > TFRZ)&(s.snl2d == 0)
    kme = jnp.where(opened[...,None],km+ke,km)
    kme = jnp.concatenate((kme,kme[...,-1:]),axis=-1)
    molecular = TKWAT*tkice_eff/((1.-ice)*tkice_eff+TKWAT*ice)
    tkl = jnp.where(opened[...,None],kme*cwat,molecular)
    s = s._replace(savedtke12d=kme[...,0]*cwat)
    eta = 1.1925*s.lakedepth2d**real(-.424)
    zin,zout = zl-.5*dl,zl+.5*dl
    rsfin = jnp.exp(-eta[...,None]*jnp.maximum(zin-.6,0.))
    rsfout = jnp.exp(-eta[...,None]*jnp.maximum(zout-.6,0.))
    phi = jnp.where(opened[...,None],(rsfin-rsfout)*f.absorbed_solar[...,None]*(1.-.4),0.)
    phi = phi.at[...,0].set(jnp.where((~opened)&(s.snl2d == 0),f.absorbed_solar*(1.-.4),phi[...,0]))
    phisoil = jnp.where(opened,rsfout[...,-1]*f.absorbed_solar*(1.-.4),0.)
    cvl = dl*(cwat*(1.-ice)+cice*ice)
    tk,cv,topsoil = soil_thermal(s)
    old_energy = energy(s,cv,cvl)
    fracold = jnp.where(jnp.arange(-4,11) > s.snl2d[...,None],s.h2osoi_ice3d/(s.h2osoi_liq3d+s.h2osoi_ice3d),0.)
    zx = jnp.concatenate((s.z3d[...,:5],zl,zl[...,-1:]+dl[...,-1:]/2.+s.z3d[...,5:]),axis=-1)
    cvx = jnp.concatenate((cv[...,:5],cvl,cv[...,5:]),axis=-1)
    phix = jnp.concatenate((jnp.zeros_like(phi[...,:5]),phi,phisoil[...,None],jnp.zeros_like(phi[...,:9])),axis=-1)
    tx = jnp.concatenate((s.t_soisno3d[...,:5],tl,s.t_soisno3d[...,5:]),axis=-1)
    bottom_snow = tkl[...,0]*tk[...,4]*(zx[...,5]-zx[...,4])/(tk[...,4]*zl[...,0]+tkl[...,0]*(-s.z3d[...,4]))
    lake_int = tkl[...,:-1]*tkl[...,1:]*(dl[...,1:]+dl[...,:-1])/(tkl[...,:-1]*dl[...,1:]+tkl[...,1:]*dl[...,:-1])
    bottom_lake = topsoil*tkl[...,-1]*(zx[...,15]-zx[...,14])/(topsoil*dl[...,-1]/2.+tkl[...,-1]*s.z3d[...,5])
    tkx = jnp.concatenate((tk[...,:4],bottom_snow[...,None],lake_int,bottom_lake[...,None],tk[...,5:]),axis=-1)
    top = s.snl2d.astype(jnp.int32)+5
    active = jnp.arange(25) >= top[...,None]
    fact = f.dt[...,None]/jnp.where(active,cvx,1.)
    fn = tkx[...,:-1]*(tx[...,1:]-tx[...,:-1])/(zx[...,1:]-zx[...,:-1])
    fn = jnp.concatenate((fn,jnp.zeros_like(fn[...,:1])),axis=-1)
    a,b,c,r = (jnp.zeros_like(tx) for _ in range(4))
    for j in range(25):
        fj = fact[...,j]
        dzm = zx[...,j]-zx[...,max(j-1,0)] if j else jnp.ones_like(fj)
        dzp = zx[...,min(j+1,24)]-zx[...,j] if j < 24 else jnp.ones_like(fj)
        is_top = j == top
        aj = -.5*fj*tkx[...,max(j-1,0)]/dzm
        cj = -.5*fj*tkx[...,j]/dzp if j < 24 else jnp.zeros_like(fj)
        bj = 1.+.5*fj*(tkx[...,j]/dzp+tkx[...,max(j-1,0)]/dzm)
        rj = tx[...,j]+.5*fj*(fn[...,j]-fn[...,max(j-1,0)])+fj*phix[...,j]
        if j == 24:
            bj = 1.+.5*fj*tkx[...,j-1]/dzm
            rj = tx[...,j]-.5*fj*fn[...,j-1]
        a = a.at[...,j].set(jnp.where(is_top,0.,aj))
        b = b.at[...,j].set(jnp.where(is_top,1.+.5*fj*tkx[...,j]/dzp,bj))
        c = c.at[...,j].set(cj)
        r = r.at[...,j].set(jnp.where(is_top,tx[...,j]+fj*(flux.gnet+phix[...,j]+.5*fn[...,j]),rj))
    # Inactive slots never enter WRF's solve. Keep them algebraically benign.
    a,c = jnp.where(active,a,0.),jnp.where(active,c,0.)
    b,r = jnp.where(active,b,1.),jnp.where(active,r,tx)
    tx = lake_tridiagonal(a,b,c,r,jtop=top,previous=tx)
    s = s._replace(t_soisno3d=jnp.concatenate((tx[...,:5],tx[...,15:]),axis=-1),t_lake3d=tx[...,5:15])
    s,snomelt,imelt = phase_change(s,cv,cvl)
    tl,ice = s.t_lake3d,s.lake_icefrac3d
    rho = density(tl,ice)
    for j in range(9):
        mix = (rho[...,j] > rho[...,j+1])|((ice[...,j] < 1.)&(ice[...,j+1] > 0.))
        qav,iceav,nav = (jnp.zeros_like(s.snl2d) for _ in range(3))
        for i in range(j+2):
            qav = qav+dl[...,i]*(tl[...,i]-TFRZ)*((1.-ice[...,i])*cwat+ice[...,i]*cice)
            iceav = iceav+ice[...,i]*dl[...,i]
            nav = nav+dl[...,i]
        qav,iceav = qav/nav,iceav/nav
        tf = jnp.where(qav < 0.,qav/jnp.where(iceav > 0,iceav*cice,1.),0.)
        tu = jnp.where(qav > 0.,qav/jnp.where(iceav < 1,(1.-iceav)*cwat,1.),0.)
        zsum = jnp.zeros_like(nav)
        for i in range(j+2):
            frozen = (zsum+dl[...,i])/nav <= iceav
            partial = zsum/nav < iceav
            newice = jnp.where(frozen,1.,jnp.where(partial,(iceav*nav-zsum)/dl[...,i],0.))
            newt = jnp.where(frozen,tf+TFRZ,jnp.where(partial,(newice*tf*cice+(1.-newice)*tu*cwat)/(newice*cice+(1.-newice)*cwat)+TFRZ,tu+TFRZ))
            ice = ice.at[...,i].set(jnp.where(mix,newice,ice[...,i]))
            tl = tl.at[...,i].set(jnp.where(mix,newt,tl[...,i]))
            rho = rho.at[...,i].set(jnp.where(mix,density(tl[...,i],ice[...,i]),rho[...,i]))
            zsum = zsum+dl[...,i]
    s = s._replace(t_lake3d=tl,lake_icefrac3d=ice)
    _,cv,_ = soil_thermal(s)
    cvl = dl*(cwat*(1.-ice)+cice*ice)
    fin = flux.gnet
    for j in range(10):
        fin = fin+phi[...,j]
    fin = fin+phisoil
    residual = (energy(s,cv,cvl)-old_energy)/f.dt-fin
    correction = jnp.where(jnp.abs(residual) < 10.,residual,0.)
    flux = flux._replace(sensible=flux.sensible-correction,ground=flux.ground+correction,gnet=flux.gnet+correction)
    return s,flux,fracold,snomelt,imelt,residual
