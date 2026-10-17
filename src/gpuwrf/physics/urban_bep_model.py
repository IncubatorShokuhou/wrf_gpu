"""Complete WRF BEP column step on a fixed table-defined urban geometry.

Arrays retain WRF's full 18 urban levels and 10 material layers. Only compact
urban columns should be supplied by the operational adapter. Geometry and
radiation pivot plans are host statics; all evolving arithmetic stays in JAX.
"""
from typing import NamedTuple

import jax.numpy as jnp

from gpuwrf.physics import urban_bep_components as c


class BepState(NamedTuple):
    roof: object             # (...,2,18,10)
    wall: object             # (...,4,18,10), opposing sides interleaved
    ground: object           # (...,2,10)
    wall_heat: object        # (...,4,18)
    roof_heat: object        # (...,2,18)
    ground_heat: object      # (...,2)


class BepForcing(NamedTuple):
    z: object               # mesoscale interfaces, (...,nz+1)
    u: object
    v: object
    theta: object
    rho: object
    pressure: object
    sw: object
    lw: object
    coszen: object
    declination: object
    hour_angle: object
    dt: object


class BepResult(NamedTuple):
    state: BepState
    sources: c.MesoSources
    ground_length: object
    building_length: object
    upward_longwave: object
    absorbed_shortwave: object
    emissivity: object
    ground_flux: object


def apply_gauss_plan(rhs, plan):
    """WRF gaussj RHS operations; matrix-only work was done at initialization."""
    b=[rhs[...,j] for j in range(rhs.shape[-1])]
    inv=jnp.asarray(plan.inverse,rhs.dtype);coeff=jnp.asarray(plan.elimination,rhs.dtype)
    for step,(i,j) in enumerate(plan.swaps):
        if i!=j:b[i],b[j]=b[j],b[i]
        pivot=plan.pivots[step];b[pivot]=b[pivot]*inv[step]
        for row in range(len(b)):
            if row!=pivot:b[row]=b[row]-b[pivot]*coeff[step,row]
    return jnp.stack(b,axis=-1)


def _shadow(forcing,geo,dtype,latitude=None):
    z=jnp.asarray(geo.z,dtype);ss=jnp.asarray(geo.ss,dtype);pb=jnp.asarray(geo.pb,dtype)
    zr=jnp.arccos(forcing.coszen);sinz=jnp.sin(zr)
    term=jnp.cos(forcing.declination)*jnp.sin(forcing.hour_angle)
    ratio=term/jnp.where(jnp.abs(sinz)>1.e-10,sinz,1.)
    pi=jnp.asarray(3.141592653,dtype)
    bbb=jnp.where(jnp.abs(sinz)>1.e-10,jnp.arcsin(jnp.clip(ratio,-1.,1.)),jnp.where(term>=0.,pi/2.,-pi/2.))
    if latitude is not None:
        reverse=(jnp.abs(sinz)>1.e-10)&(ratio>-1.)&(ratio<1.)&(
            jnp.sin(forcing.declination)<jnp.cos(zr)*jnp.sin(latitude*pi/180))
        bbb=jnp.where(reverse,pi-bbb,bbb)
    night=(forcing.sw==0.) | (sinz==1.)
    ground=[];walls=[]
    for d in range(geo.nd):
        angle=bbb-jnp.asarray(geo.angle[d],dtype);width=jnp.asarray(geo.width[d],dtype)
        sines=(jnp.sin(angle),jnp.sin(angle+pi))
        side=[]
        for sine in sines:
            levels=[];denom=jnp.where(jnp.abs(sine)>1.e-10,sine,1.)
            for i in range(geo.n):
                total=jnp.zeros_like(forcing.sw)
                for j in range(geo.n):
                    x1=jnp.minimum((z[j+1]-z[i])*jnp.tan(zr),jnp.maximum(0.,width/denom))
                    x2=jnp.maximum((z[j+1]-z[i+1])*jnp.tan(zr),0.)
                    rd=jnp.maximum(0.,sine*jnp.maximum(0.,x1-x2)/(z[i+1]-z[i]))
                    contribution=forcing.sw*rd*ss[j+1]/jnp.where(pb[i+1]>0.,pb[i+1],1.)
                    total=total+jnp.where((pb[i+1]>0.) & (jnp.abs(sine)>1.e-10),contribution,0.)
                levels.append(jnp.where(night,0.,total))
            side.append(jnp.stack(levels,axis=-1))
        walls.extend(side)
        sine=sines[0];wsd=jnp.abs(width/jnp.where(jnp.abs(sine)>1.e-10,sine,1.))
        total=jnp.zeros_like(forcing.sw)
        for j in range(geo.n):
            rd=jnp.maximum(0.,wsd-z[j+1]*jnp.tan(zr))
            total=total+forcing.sw*rd*ss[j+1]/wsd
        ground.append(jnp.where(night,0.,jnp.where(jnp.abs(sine)>1.e-10,total,forcing.sw)))
    return jnp.stack(walls,axis=-2),jnp.stack(ground,axis=-1)


def canyon_radiation(state,forcing,geo):
    dtype=state.roof.dtype;n=geo.n
    sw_wall,sw_ground=_shadow(forcing,geo,dtype)
    pb=jnp.asarray(geo.pb,dtype);eg,ew,er=jnp.asarray(geo.emissivity,dtype)
    sigma=jnp.asarray(5.67e-8,dtype)
    long_wall=[];long_ground=[];short_wall=[];short_ground=[]
    for d in range(geo.nd):
        fww,fwg,fgw,fsw,fws,fsg=[jnp.asarray(a,dtype) for a in geo.views[d]]
        rows=[]
        for side in range(2):
            opposing=2*d+1-side
            for i in range(n):
                value=fsw[i]*forcing.lw+eg*fgw[i]*sigma*state.ground[...,d,-1]**4
                for j in range(n):
                    value=value+pb[j+1]*ew*sigma*fww[j,i]*state.wall[...,opposing,j,-1]**4+fww[j,i]*forcing.lw*(1.-pb[j+1])
                rows.append(value)
        value=fsg*forcing.lw
        for i in range(n):
            value=value+ew*sigma*fwg[i]*pb[i+1]*(state.wall[...,2*d,i,-1]**4+state.wall[...,2*d+1,i,-1]**4)+2.*fwg[i]*(1.-pb[i+1])*forcing.lw
        rows.append(value)
        solution=apply_gauss_plan(jnp.stack(rows,axis=-1),geo.long_plans[d])
        long_wall.extend((solution[...,:n],solution[...,n:2*n]));long_ground.append(solution[...,-1])
        rhs=jnp.concatenate((sw_wall[...,2*d,:],sw_wall[...,2*d+1,:],sw_ground[...,d,None]),axis=-1)
        solution=apply_gauss_plan(rhs,geo.short_plans[d])
        short_wall.extend((solution[...,:n],solution[...,n:2*n]));short_ground.append(solution[...,-1])
    return (jnp.stack(short_wall,axis=-2),jnp.stack(short_ground,axis=-1),
            jnp.stack(long_wall,axis=-2),jnp.stack(long_ground,axis=-1))


def upward_radiation(state,forcing,geo,sw_wall,sw_ground,lw_wall,lw_ground):
    dtype=state.roof.dtype
    eg,ew,er=jnp.asarray(geo.emissivity,dtype);ag,aw,ar=jnp.asarray(geo.albedo,dtype)
    ss=jnp.asarray(geo.ss,dtype);pb=jnp.asarray(geo.pb,dtype);sigma=jnp.asarray(5.67e-8,dtype)
    emit=jnp.zeros_like(forcing.sw);inc=jnp.zeros_like(emit);absorbed=jnp.zeros_like(emit);flux=jnp.zeros_like(emit)
    for d in range(geo.nd):
        ws=jnp.asarray(geo.width[d],dtype);bs=jnp.asarray(geo.building[d],dtype)
        t=state.ground[...,d,-1]
        emit=emit-(eg*sigma*t**4+(1-eg)*lw_ground[...,d])*ws/(ws+bs)/geo.nd
        inc=inc+lw_ground[...,d]*ws/(ws+bs)/geo.nd
        absorbed=absorbed+(1.-ag)*sw_ground[...,d]*ws/(ws+bs)/geo.nd
        gfl=(1.-ag)*sw_ground[...,d]+eg*lw_ground[...,d]-eg*sigma*t**4+state.ground_heat[...,d]
        flux=flux-gfl*ws/(ws+bs)/geo.nd
        for i in range(1,geo.n):
            t=state.roof[...,d,i,-1]
            emit=emit-(er*sigma*t**4+(1-er)*forcing.lw)*ss[i]*bs/(ws+bs)/geo.nd
            inc=inc+forcing.lw*ss[i]*bs/(ws+bs)/geo.nd
            absorbed=absorbed+(1.-ar)*forcing.sw*ss[i]*bs/(ws+bs)/geo.nd
            gfl=(1.-ar)*forcing.sw+er*forcing.lw-er*sigma*t**4+state.roof_heat[...,d,i]
            flux=flux-gfl*ss[i]*bs/(ws+bs)/geo.nd
        for i in range(geo.n):
            t4=state.wall[...,2*d,i,-1]**4+state.wall[...,2*d+1,i,-1]**4
            lw=lw_wall[...,2*d,i]+lw_wall[...,2*d+1,i]
            sw=sw_wall[...,2*d,i]+sw_wall[...,2*d+1,i]
            sh=state.wall_heat[...,2*d,i]+state.wall_heat[...,2*d+1,i]
            emit=emit-(ew*sigma*t4+(1-ew)*lw)*5.*pb[i+1]/(ws+bs)/geo.nd
            inc=inc+lw*5.*pb[i+1]/(ws+bs)/geo.nd
            absorbed=absorbed+(1.-aw)*sw*5.*pb[i+1]/(ws+bs)/geo.nd
            gfl=(1.-aw)*sw+ew*lw-ew*sigma*t4+sh
            flux=flux-gfl*5.*pb[i+1]/(ws+bs)/geo.nd
    return (inc+emit)-forcing.lw,absorbed,(eg+ew+er)/3.,flux


def bep_column_step(state:BepState,forcing:BepForcing,geo):
    dtype=state.roof.dtype;n=geo.n
    z=jnp.asarray(geo.z,dtype);ss=jnp.asarray(geo.ss,dtype);pb=jnp.asarray(geo.pb,dtype)
    urb=[c.bep_interpolate(forcing.z,z,field) for field in
         (forcing.u,forcing.v,forcing.theta,jnp.full_like(forcing.theta,300.),forcing.pressure,forcing.rho)]
    u,v,theta,theta0,pressure,rho=urb
    sw_wall,sw_ground,lw_wall,lw_ground=canyon_radiation(state,forcing,geo)
    upward,absorbed,emissivity,ground_flux=upward_radiation(state,forcing,geo,sw_wall,sw_ground,lw_wall,lw_ground)
    ag,aw,ar=jnp.asarray(geo.albedo,dtype);eg,ew,er=jnp.asarray(geo.emissivity,dtype)
    cg,cw,cr=jnp.asarray(geo.capacity,dtype);kg,kw,kr=jnp.asarray(geo.diffusivity,dtype)
    road_depth=jnp.asarray([.2,.12,.08,.05,.03,.02,.02,.01,.005,.0025],dtype)
    wall_depth=jnp.asarray([.02,.02,.02,.02,.02,.02,.02,.01,.005,.0025],dtype)
    roof=state.roof;wall=state.wall;ground=state.ground
    rough=jnp.asarray(geo.roughness,dtype)
    flats=[];walls=[]
    for d in range(geo.nd):
        temp,ptg,_,_=c.bep_soil_temp(road_depth,ground[...,d,:],jnp.full_like(road_depth,kg),
            jnp.full_like(road_depth,cg),sw_ground[...,d],lw_ground[...,d],pressure[...,0],forcing.dt,eg,ag,state.ground_heat[...,d])
        ground=ground.at[...,d,:].set(temp)
        pts=[]
        for i in range(n):
            if i>0 and geo.ss[i]>0.:
                temp,ptr,_,_=c.bep_soil_temp(wall_depth,roof[...,d,i,:],jnp.full_like(wall_depth,kr),
                    jnp.full_like(wall_depth,cr),forcing.sw,forcing.lw,pressure[...,i],forcing.dt,er,ar,state.roof_heat[...,d,i])
                roof=roof.at[...,d,i,:].set(temp)
            else:ptr=jnp.zeros_like(ptg)
            pts.append(ptg if i==0 else ptr)
        ptroof=jnp.stack(pts,axis=-1)
        flat=c.bep_flux_flat(5.,rough[d],u,v,theta,theta0,ptroof)
        enabled=jnp.asarray([i==0 or geo.ss[i]>0. for i in range(n)])
        flat=c.FlatFlux(*(jnp.where(enabled,field,0.) for field in flat));flats.append(flat)
        for side in range(2):
            s=2*d+side
            temp,ptw,_,_=c.bep_soil_temp(wall_depth,wall[...,s,:n,:],jnp.full_like(wall_depth,kw),
                jnp.full_like(wall_depth,cw),sw_wall[...,s,:],lw_wall[...,s,:],pressure,forcing.dt[...,None] if hasattr(forcing.dt,'ndim') and forcing.dt.ndim else forcing.dt,
                ew,aw,state.wall_heat[...,s,:n])
            wall=wall.at[...,s,:n,:].set(temp)
            walls.append(c.bep_flux_wall(u,v,theta,rho,ptw,jnp.asarray(geo.angle[d],dtype),
                         forcing.dt[...,None] if hasattr(forcing.dt,'ndim') and forcing.dt.ndim else forcing.dt))
    flat=c.FlatFlux(*(jnp.stack([f[j] for f in flats],axis=-2) for j in range(4)))
    wallflux=c.WallFlux(*(jnp.stack([f[j] for f in walls],axis=-2) for j in range(7)))
    ground_heat=state.ground_heat.at[...,:geo.nd].set(-rho[...,0,None]*1004.*flat.heat[...,0])
    roof_heat=state.roof_heat.at[...,:geo.nd,1:n].set(-rho[...,None,1:]*1004.*flat.heat[...,1:])
    wall_heat=state.wall_heat.at[...,:2*geo.nd,:n].set(-rho[...,None,:]*1004.*(wallflux.b_t+wallflux.a_t*theta[...,None,:]))
    newstate=BepState(roof,wall,ground,wall_heat,roof_heat,ground_heat)
    sources=c.bep_urban_meso(forcing.z,forcing.z[...,1:]-forcing.z[...,:-1],z,pb,ss,
                          jnp.asarray(geo.building,dtype),jnp.asarray(geo.width,dtype),wallflux,flat,nd=geo.nd)
    dlg,dl=c.bep_interp_length(forcing.z,z,ss,jnp.asarray(geo.width,dtype),jnp.asarray(geo.building,dtype),nd=geo.nd)
    return BepResult(newstate,sources,dlg,dl,upward,absorbed,emissivity,ground_flux)
