"""WRF ShalLakeHydrology for Lake's do_capsnow=.false. caller (r8)."""
from jax import numpy as jnp
from .lake_fluxes import TFRZ, CPICE, CPLIQ, HFUS, at, set_at
from .lake_snow import snow_water,compact,combine,divide


def hydrology_step(s,f,flux,fracold,snomelt,imelt):
    snowfall = jnp.where((f.precipitation > 0.)&(f.temperature <= TFRZ+2.5),f.precipitation,0.)
    rain = jnp.where((f.precipitation > 0.)&(f.temperature > TFRZ+2.5),f.precipitation,0.)
    dt = f.dt
    bifall = jnp.where(f.temperature > TFRZ+2.,50.+1.7*17.**1.5,
                        jnp.where(f.temperature > TFRZ-15.,50.+1.7*jnp.maximum(f.temperature-TFRZ+15.,0.)**1.5,50.))
    dzsnow = snowfall/bifall
    sd = s.snowdp2d+dzsnow*dt
    swe = s.h2osno2d+snowfall*dt
    snl = s.snl2d.astype(jnp.int32)
    newnode = (snl == 0)&(snowfall > 0.)&(sd >= .01)
    snl = jnp.where(newnode,-1,snl)
    dz,z,zi,t,wi,wl = s.dz3d,s.z3d,s.zi3d,s.t_soisno3d,s.h2osoi_ice3d,s.h2osoi_liq3d
    dz = dz.at[...,4].set(jnp.where(newnode,sd,dz[...,4]))
    z = z.at[...,4].set(jnp.where(newnode,-.5*sd,z[...,4]))
    zi = zi.at[...,4].set(jnp.where(newnode,-sd,zi[...,4]))
    t = t.at[...,4].set(jnp.where(newnode,jnp.minimum(TFRZ,f.temperature),t[...,4]))
    wi = wi.at[...,4].set(jnp.where(newnode,swe,wi[...,4]))
    wl = wl.at[...,4].set(jnp.where(newnode,0.,wl[...,4]))
    fracold = fracold.at[...,4].set(jnp.where(newnode,1.,fracold[...,4]))
    top = snl+5
    existing = (snl < 0)&~newnode
    wi = set_at(wi,top,jnp.where(existing,at(wi,top)+dt*snowfall,at(wi,top)))
    dz = set_at(dz,top,jnp.where(existing,at(dz,top)+dzsnow*dt,at(dz,top)))
    snow = snl < 0
    positive = flux.evaporation >= 0.
    water = at(wi,top)+at(wl,top)
    limited = jnp.minimum(flux.evaporation,water/dt)
    evap = jnp.where(positive,jnp.where(snow,jnp.where(water > 0.,jnp.maximum(limited*at(wl,top)/jnp.where(water > 0.,water,1.),0.),0.),flux.evaporation-jnp.minimum(flux.evaporation,swe/dt)),0.)
    sub = jnp.where(positive,jnp.where(snow,limited-evap,jnp.minimum(flux.evaporation,swe/dt)),0.)
    dew_frozen = flux.ground_temperature < jnp.where(snow,TFRZ,TFRZ-.1)
    dew_snow = jnp.where((~positive)&dew_frozen,jnp.abs(flux.evaporation),0.)
    dew = jnp.where((~positive)&~dew_frozen,jnp.abs(flux.evaporation),0.)
    new_swe = swe+(-sub+dew_snow)*dt
    new_sd = jnp.where(swe > 0.,sd*new_swe/jnp.where(swe > 0.,swe,1.),new_swe/250.)
    swe = jnp.where(snow,swe,jnp.maximum(new_swe,0.))
    sd = jnp.where(snow,sd,new_sd)
    s = s._replace(snl2d=snl.astype(jnp.float64),snowdp2d=sd,h2osno2d=swe,dz3d=dz,z3d=z,zi3d=zi,t_soisno3d=t,h2osoi_ice3d=wi,h2osoi_liq3d=wl)
    s = snow_water(s,dt,rain,sub,evap,dew_snow,dew)
    wi,wl = s.h2osoi_ice3d,s.h2osoi_liq3d
    soil_liq = jnp.where(s.h2osoi_vol3d[...,5:] < s.watsat3d,(s.watsat3d*s.dz3d[...,5:]-wi[...,5:]/917.)*1000.,
                         jnp.where(wl[...,5:] > s.watsat3d*1000.*s.dz3d[...,5:],s.watsat3d*1000.*s.dz3d[...,5:],wl[...,5:]))
    s = s._replace(h2osoi_liq3d=wl.at[...,5:].set(soil_liq))
    s = compact(s,dt,imelt,fracold)
    s = combine(s,snow)
    s = divide(s,snow)
    unfrozen = (s.t_lake3d[...,0] > TFRZ)&(s.lake_icefrac3d[...,0] == 0.)&(s.snl2d < 0)
    sum_ice,heat = jnp.zeros_like(dt),jnp.zeros_like(dt)
    for j in range(5):
        active = j > s.snl2d+4
        sum_ice = jnp.where(active,sum_ice+s.h2osoi_ice3d[...,j],sum_ice)
        heat = jnp.where(active,heat+s.h2osoi_ice3d[...,j]*CPICE*(TFRZ-s.t_soisno3d[...,j])+s.h2osoi_liq3d[...,j]*CPLIQ*(TFRZ-s.t_soisno3d[...,j]),heat)
    heat = heat+sum_ice*HFUS
    rem = (s.t_lake3d[...,0]-TFRZ)*CPLIQ*1000.*s.dz_lake3d[...,0]-heat
    remove = unfrozen&(rem+1000.*s.dz_lake3d[...,0]*HFUS > 0.)
    temp = jnp.where(rem > 0.,s.t_lake3d[...,0]-rem/(CPLIQ*1000.*s.dz_lake3d[...,0]),TFRZ)
    ice = jnp.where(rem > 0.,s.lake_icefrac3d[...,0],-rem/(1000.*s.dz_lake3d[...,0]*HFUS))
    s = s._replace(h2osno2d=jnp.where(remove,0.,s.h2osno2d),snl2d=jnp.where(remove,0.,s.snl2d),
                   t_lake3d=s.t_lake3d.at[...,0].set(jnp.where(remove,temp,s.t_lake3d[...,0])),
                   lake_icefrac3d=s.lake_icefrac3d.at[...,0].set(jnp.where(remove,ice,s.lake_icefrac3d[...,0])))
    clean = snow[...,None]&(jnp.arange(-4,1) <= s.snl2d[...,None])&(s.snl2d[...,None] > -5)
    update = {}
    for name in ("h2osoi_ice3d","h2osoi_liq3d","t_soisno3d","dz3d","z3d","zi3d"):
        value = getattr(s,name)
        value = value.at[...,:5].set(jnp.where(clean,0.,value[...,:5]))
        update[name] = value
    s = s._replace(**update)
    volume = s.h2osoi_liq3d[...,5:]/(s.dz3d[...,5:]*1000.)+s.h2osoi_ice3d[...,5:]/(s.dz3d[...,5:]*917.)
    return s._replace(h2osoi_vol3d=s.h2osoi_vol3d.at[...,5:].set(volume))
