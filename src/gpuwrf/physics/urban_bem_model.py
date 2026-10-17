"""WRF BEP+BEM canyon/building step (ordinary solid roof, no PV/green roof)."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.physics import urban_bep_components as c
from gpuwrf.physics.urban_bep_model import BepResult,_shadow,apply_gauss_plan
from gpuwrf.physics.urban_bem_building import BuildingState,building_step


class BemState(NamedTuple):
    roof: object             # (...,2,18,10)
    wall: object             # (...,15,4,18,10)
    ground: object           # (...,2,10)
    wall_heat: object        # (...,15,4,18)
    roof_heat: object        # (...,2,18)
    ground_heat: object      # (...,2)
    temperature: object     # (...,15,18)
    humidity: object
    window: object          # (...,15,4,18)
    window_heat: object
    floor: object           # (...,15,2,17,10)
    under_ground: object    # (...,15,2,10)
    roof_water: object      # (...,2,18)
    ground_water: object    # (...,2)
    roof_latent: object     # (...,2,18)
    ground_latent: object   # (...,2)
    pv_temperature: object  # (...,2,18), held when PV=0
    extra: object           # unchanged/disabled green-roof Registry arrays


class BemDiagnostics(NamedTuple):
    sensible_ac: object
    latent_ac: object
    consumption: object
    sensible_vent: object
    latent_vent: object


class BemResult(NamedTuple):
    state: BemState
    sources: object
    ground_length: object
    building_length: object
    upward_longwave: object
    absorbed_shortwave: object
    emissivity: object
    ground_flux: object
    moisture_source: object
    diagnostics: BemDiagnostics


def _averages(state,bem):
    geo=bem.bep;n=geo.n;dtype=state.roof.dtype
    shape=state.wall.shape[:-4]+(4,n)
    wall=jnp.zeros(shape,dtype);window=jnp.zeros(shape,dtype);heat=jnp.zeros(shape,dtype);window_heat=jnp.zeros(shape,dtype)
    for level in range(n):
        if geo.pb[level+1]>0.:
            for building,floors in enumerate(bem.levels):
                if level<floors:
                    weight=jnp.asarray(bem.probabilities[building]/geo.pb[level+1],dtype)
                    wall=wall.at[...,level].add(weight*state.wall[...,building,:,level,-1]**4)
                    window=window.at[...,level].add(weight*state.window[...,building,:,level]**4)
                    heat=heat.at[...,level].add(weight*state.wall_heat[...,building,:,level])
                    window_heat=window_heat.at[...,level].add(weight*state.window_heat[...,building,:,level])
            wall=wall.at[...,level].set(wall[...,level]**.25)
            window=window.at[...,level].set(window[...,level]**.25)
    return wall,window,heat,window_heat


def _radiation(state,forcing,bem,latitude,sw_direct,sw_diffuse,averages):
    geo=bem.bep;n=geo.n;dtype=state.roof.dtype
    wall,window,_,_=averages
    direct=forcing._replace(sw=sw_direct)
    raw_wall,raw_ground=_shadow(direct,geo,dtype,latitude=latitude)
    pb=jnp.asarray(geo.pb,dtype);eg,ew,er=jnp.asarray(geo.emissivity,dtype)
    pw=jnp.asarray(bem.pwin,dtype);win=jnp.asarray(.9,dtype);sigma=jnp.asarray(5.67e-8,dtype)
    sw_wall=[];sw_ground=[];lw_wall=[];lw_ground=[]
    for d in range(geo.nd):
        fww,fwg,fgw,fsw,fws,fsg=[jnp.asarray(a,dtype) for a in geo.views[d]]
        rows=[]
        for side in range(2):
            other=2*d+1-side
            for i in range(n):
                value=fsw[i]*forcing.lw+eg*fgw[i]*sigma*state.ground[...,d,-1]**4
                for j in range(n):
                    value=value+pb[j+1]*sigma*fww[j,i]*(ew*(1.-pw)*wall[...,other,j]**4+win*pw*window[...,other,j]**4)+fww[j,i]*forcing.lw*(1.-pb[j+1])
                rows.append(value)
        value=fsg*forcing.lw
        for i in range(n):
            value=value+sigma*fwg[i]*pb[i+1]*(ew*(1.-pw)*(wall[...,2*d,i]**4+wall[...,2*d+1,i]**4)+win*pw*(window[...,2*d,i]**4+window[...,2*d+1,i]**4))+2.*fwg[i]*(1.-pb[i+1])*forcing.lw
        rows.append(value)
        solution=apply_gauss_plan(jnp.stack(rows,axis=-1),bem.long_plans[d])
        lw_wall.extend((solution[...,:n],solution[...,n:2*n]));lw_ground.append(solution[...,-1])
        rhs=jnp.concatenate((raw_wall[...,2*d,:]+fsw*sw_diffuse[...,None],
            raw_wall[...,2*d+1,:]+fsw*sw_diffuse[...,None],(raw_ground[...,d]+fsg*sw_diffuse)[...,None]),axis=-1)
        solution=apply_gauss_plan(rhs,bem.short_plans[d])
        sw_wall.extend((solution[...,:n],solution[...,n:2*n]));sw_ground.append(solution[...,-1])
    return jnp.stack(sw_wall,axis=-2),jnp.stack(sw_ground,axis=-1),jnp.stack(lw_wall,axis=-2),jnp.stack(lw_ground,axis=-1)


def _upward(state,forcing,bem,averages,sw_wall,sw_ground,lw_wall,lw_ground):
    geo=bem.bep;dtype=state.roof.dtype
    wall,window,wall_heat,window_heat=averages
    roof=(state.roof[...,-1]**4)**.25
    eg,ew,er=jnp.asarray(geo.emissivity,dtype);ag,aw,ar=jnp.asarray(geo.albedo,dtype)
    pw=jnp.asarray(bem.pwin,dtype);win=jnp.asarray(.9,dtype);albw=jnp.asarray(bem.window_albedo,dtype)
    sigma=jnp.asarray(5.67e-8,dtype);ss=jnp.asarray(geo.ss,dtype);pb=jnp.asarray(geo.pb,dtype)
    emit=jnp.zeros_like(forcing.sw);inc=jnp.zeros_like(emit);absorbed=jnp.zeros_like(emit);ground=jnp.zeros_like(emit)
    for d in range(geo.nd):
        ws=jnp.asarray(geo.width[d],dtype);bs=jnp.asarray(geo.building[d],dtype)
        emit=emit-(eg*sigma*state.ground[...,d,-1]**4+(1-eg)*lw_ground[...,d])*ws/(ws+bs)/geo.nd
        inc=inc+lw_ground[...,d]*ws/(ws+bs)/geo.nd
        absorbed=absorbed+(1.-ag)*sw_ground[...,d]*ws/(ws+bs)/geo.nd
        gfl=(1.-ag)*sw_ground[...,d]+eg*lw_ground[...,d]-eg*sigma*state.ground[...,d,-1]**4+state.ground_heat[...,d]+state.ground_latent[...,d]
        ground=ground-gfl*ws/(ws+bs)/geo.nd
        for i in range(1,geo.n):
            emit=emit-(er*sigma*roof[...,d,i]**4+(1-er)*forcing.lw)*ss[i]*bs/(ws+bs)/geo.nd
            inc=inc+forcing.lw*ss[i]*bs/(ws+bs)/geo.nd
            absorbed=absorbed+(1.-ar)*forcing.sw*ss[i]*bs/(ws+bs)/geo.nd
            gfl=(1.-ar)*forcing.sw+er*forcing.lw-er*sigma*roof[...,d,i]**4+state.roof_heat[...,d,i]+state.roof_latent[...,d,i]
            ground=ground-gfl*ss[i]*bs/(ws+bs)/geo.nd
        for i in range(geo.n):
            t4=wall[...,2*d,i]**4+wall[...,2*d+1,i]**4;v4=window[...,2*d,i]**4+window[...,2*d+1,i]**4
            lw=lw_wall[...,2*d,i]+lw_wall[...,2*d+1,i];sw=sw_wall[...,2*d,i]+sw_wall[...,2*d+1,i]
            emit=emit-(ew*(1.-pw)*sigma*t4+win*pw*sigma*v4+((1.-ew)*(1.-pw)+pw*(1.-win))*lw)*5.*pb[i+1]/(ws+bs)/geo.nd
            inc=inc+lw*5.*pb[i+1]/(ws+bs)/geo.nd
            absorbed=absorbed+((1.-aw)*(1.-pw)+(1.-albw)*pw)*sw*5.*pb[i+1]/(ws+bs)/geo.nd
            gfl=(1.-aw)*sw+ew*lw-ew*sigma*t4+wall_heat[...,2*d,i]+wall_heat[...,2*d+1,i]
            gwin=(1.-albw)*sw+win*lw-win*sigma*v4+window_heat[...,2*d,i]+window_heat[...,2*d+1,i]
            ground=ground-(gfl*(1.-pw)+pw*gwin)*5.*pb[i+1]/(ws+bs)/geo.nd
    return (inc+emit)-forcing.lw,absorbed,(eg+ew+er)/3.,ground


def _map_scalar(z,urban_z,pb,ss,bs,ws,volume,horizontal,wall,nd):
    dz=z[...,1:]-z[...,:-1];n=len(urban_z)-1;total=jnp.zeros_like(volume)
    for d in range(nd):
        fact=1./volume[...,0]/dz[...,0]*ws[d]/(ws[d]+bs[d])/nd
        total=total.at[...,0].add(horizontal[...,d,0]*fact)
        roof=jnp.zeros_like(total)
        for k in range(1,n):
            hit=(z[...,:-1]<=urban_z[k])&(z[...,1:]>urban_z[k])
            roof=roof+jnp.where(hit,ss[k]*horizontal[...,d,k,None],0.)
        total=total+roof*(bs[d]/(ws[d]+bs[d])/volume/dz/nd)
    for d in range(nd):
        value=jnp.zeros_like(total)
        for k in range(n):
            overlap=jnp.maximum(jnp.minimum(urban_z[k+1],z[...,1:])-jnp.maximum(urban_z[k],z[...,:-1]),0.)
            fact=overlap/(ws[d]+bs[d])
            value=value+pb[k+1]*(wall[...,2*d,k,None]+wall[...,2*d+1,k,None])*fact
        total=total+value*(1./volume/dz/nd)
    return total


def bem_column_step(state:BemState,forcing,bem,*,latitude,sw_direct,sw_diffuse,rain,qv,fraction):
    """module_sf_bep_bem.F:BEP1D, including building/AC/ventilation and wet surfaces."""
    geo=bem.bep;dtype=state.roof.dtype;n=geo.n;C=lambda x:jnp.asarray(x,dtype)
    z=C(geo.z);ss=C(geo.ss);pb=C(geo.pb);bs=C(geo.building);ws=C(geo.width)
    u,v,theta,theta0,pressure,rho,humidity=[c.bep_interpolate(forcing.z,z,x) for x in
        (forcing.u,forcing.v,forcing.theta,jnp.full_like(forcing.theta,300.),forcing.pressure,forcing.rho,qv)]
    average=_averages(state,bem)
    sw_wall,sw_ground,lw_wall,lw_ground=_radiation(state,forcing,bem,latitude,sw_direct,sw_diffuse,average)
    upward,absorbed,emissivity,ground_flux=_upward(state,forcing,bem,average,sw_wall,sw_ground,lw_wall,lw_ground)
    dt=forcing.dt;latent=C(2.45e6)
    gw=jnp.clip(jnp.where(state.ground_water<=1.,state.ground_water+rain[...,None]+state.ground_latent*dt[...,None]/latent,state.ground_water),0.,1.)
    rw=state.roof_water.at[...,1:].set(jnp.clip(jnp.where(state.roof_water[...,1:]<=1.,state.roof_water[...,1:]+rain[...,None,None]+state.roof_latent[...,1:]*dt[...,None,None]/latent,state.roof_water[...,1:]),0.,1.))
    ground=state.ground;ptg=[]
    depth=C([.2,.12,.08,.05,.03,.02,.02,.01,.005,.0025]);cg,cw,cr=C(geo.capacity);kg,kw,kr=C(geo.diffusivity)
    eg,ew,er=C(geo.emissivity);ag,aw,ar=C(geo.albedo)
    for d in range(geo.nd):
        temp,potential,_,_=c.bep_soil_temp(depth,ground[...,d,:],jnp.full_like(depth,kg),jnp.full_like(depth,cg),
            sw_ground[...,d],lw_ground[...,d],pressure[...,0],dt,eg,ag,state.ground_heat[...,d]+state.ground_latent[...,d])
        ground=ground.at[...,d,:].set(temp);ptg.append(potential)
    p0=jax.lax.optimization_barrier(C(1.e5));rcp=C(287./1004.)
    outdoor=theta*(pressure/p0)**rcp
    hour=forcing.hour_angle/C(3.141592653)*180./15.+12.
    hour=jnp.where(hour>=24.,hour-24.,hour);hour=jnp.where(hour<0.,hour+24.,hour)
    roof=state.roof;wall=state.wall;window=state.window;floor=state.floor;under=state.under_ground
    temp=state.temperature;hum=state.humidity;pv=state.pv_temperature
    buildings=[]
    for index,levels in enumerate(bem.levels):
        building=BuildingState(temp[...,index,:],hum[...,index,:],wall[...,index,:,:,:],window[...,index,:,:],
            floor[...,index,1,:,:],under[...,index,1,:],roof[...,1,levels,:])
        result=building_step(building,levels=levels,geo=bem.room,params=bem.params,dt=dt,hour=hour,
            outdoor_temperature=outdoor,outdoor_humidity=humidity,outdoor_density=rho,pressure=pressure,
            sw_wall=sw_wall,lw_wall=lw_wall,sw=forcing.sw,lw=forcing.lw,wall_heat=state.wall_heat[...,index,:,:n],
            window_heat=state.window_heat[...,index,:,:n],roof_heat=state.roof_heat[...,0,levels],
            roof_latent=state.roof_latent[...,0,levels],roof_average=(state.roof[...,0,levels,-1]**4)**.25)
        buildings.append(result);b=result.state
        temp=temp.at[...,index,:].set(b.temperature);hum=hum.at[...,index,:].set(b.humidity)
        wall=wall.at[...,index,:,:,:].set(b.wall);window=window.at[...,index,:,:].set(b.window)
        for d in range(geo.nd):
            roof=roof.at[...,d,levels,:].set(b.roof);floor=floor.at[...,index,d,:,:].set(b.floor)
            under=under.at[...,index,d,:].set(b.ground)
        pv=pv.at[...,1,levels].set(pv[...,0,levels])
    roof_heat=state.roof_heat;roof_latent=state.roof_latent
    ground_heat=state.ground_heat;ground_latent=state.ground_latent
    wall_heat=state.wall_heat;window_heat=state.window_heat
    horizontal=[];qhorizontal=[]
    for d in range(geo.nd):
        heat=[];moist=[];uu=[];vv=[];ee=[]
        for k in range(n):
            if k==0 or geo.ss[k]>0.:
                potential=ptg[d] if k==0 else roof[...,d,k,-1]*(pressure[...,k]/p0)**(-rcp)
                values=c.bem_flux_flat(5.,C(geo.roughness[d,k]),u[...,k],v[...,k],theta[...,k],theta0[...,k],potential,humidity[...,k],rho[...,k],pressure[...,k])
                uh,vh,sf,lf,eh=values
                water=gw[...,d] if k==0 else rw[...,d,k]
                lf=jnp.where(water>0.,-rho[...,k]*latent*(-(water*lf)/(rho[...,k]*latent)),0.)
                if k==0:
                    ground_heat=ground_heat.at[...,d].set(sf);ground_latent=ground_latent.at[...,d].set(lf)
                else:
                    roof_heat=roof_heat.at[...,d,k].set(sf);roof_latent=roof_latent.at[...,d,k].set(lf)
                heat.append(-sf/(rho[...,k]*1004.));moist.append(-lf/(rho[...,k]*latent));uu.append(uh);vv.append(vh);ee.append(eh)
            else:
                zero=jnp.zeros_like(forcing.sw);heat.append(zero);moist.append(zero);uu.append(zero);vv.append(zero);ee.append(zero)
        horizontal.append(c.FlatFlux(jnp.stack(uu,axis=-1),jnp.stack(vv,axis=-1),jnp.stack(heat,axis=-1),jnp.stack(ee,axis=-1)))
        qhorizontal.append(jnp.stack(moist,axis=-1))
    zeros=jnp.zeros(state.wall.shape[:-4]+(4,n),dtype)
    wf=[zeros for _ in range(7)];qwall=zeros
    for d in range(geo.nd):
        for index,levels in enumerate(bem.levels):
            result=buildings[index]
            for k in range(levels):
                plan_area=jnp.asarray(0.,dtype)
                for direction in range(geo.nd):plan_area=plan_area+bs[direction]/(ws[direction]+bs[direction])*pb[k]
                plan_area=plan_area/geo.nd
                drag=jnp.where(plan_area<.29,3.32*plan_area**.47,1.85)
                for side in range(2):
                    s=2*d+side
                    ptw=wall[...,index,s,k,-1]*(pressure[...,k]/p0)**(-rcp)
                    ptwin=window[...,index,s,k]*(pressure[...,k]/p0)**(-rcp)
                    au,av,bu,bv,sf,swin,be=c.bem_flux_wall(u[...,k],v[...,k],theta[...,k],rho[...,k],ptw,ptwin,C(geo.angle[d]),dt,drag)
                    wall_heat=wall_heat.at[...,index,s,k].set(sf);window_heat=window_heat.at[...,index,s,k].set(swin)
                    weight=C(bem.probabilities[index])/pb[k+1]
                    for j,value in [(0,au),(1,av),(2,bu),(3,bv),(6,be)]:wf[j]=wf[j].at[...,s,k].add(weight*value)
                    thermal=-weight*(sf*(1.-C(bem.pwin))+swin*C(bem.pwin))/rho[...,k]/1004.-.25*weight*(result.sensible_ventilation[...,k]-result.sensible_exhaust[...,k])/(5.*bs[d])/rho[...,k]/1004.
                    wf[5]=wf[5].at[...,s,k].add(thermal)
                    moisture=-.25*weight*(result.latent_ventilation[...,k]-result.latent_exhaust[...,k])/(5.*bs[d])/rho[...,k]/latent
                    qwall=qwall.at[...,s,k].add(moisture)
    flat=c.FlatFlux(*(jnp.stack([f[j] for f in horizontal],axis=-2) for j in range(4)))
    wall_flux=c.WallFlux(*wf)
    dz=forcing.z[...,1:]-forcing.z[...,:-1]
    sources=c.bep_urban_meso(forcing.z,dz,z,pb,ss,bs,ws,wall_flux,flat,nd=geo.nd)
    moisture=_map_scalar(forcing.z,z,pb,ss,bs,ws,sources.volume,jnp.stack(qhorizontal,axis=-2),qwall,geo.nd)
    dlg,dl=c.bep_interp_length(forcing.z,z,ss,ws,bs,nd=geo.nd)
    factor=.25*fraction/((bs[0]+ws[0])*bs[1])+.25*fraction/((bs[1]+ws[1])*bs[0])
    factor=factor*C(bem.params['building_ac_fraction'])*C(bem.params['cooled_fraction'])
    diagnostics=[jnp.zeros_like(forcing.sw) for _ in range(5)]
    for index,result in enumerate(buildings):
        for k in range(bem.levels[index]):
            for j,value in enumerate(result[1:6]):diagnostics[j]=diagnostics[j]+factor*C(bem.probabilities[index])*value[...,k]
    new=BemState(roof,wall,ground,wall_heat,roof_heat,ground_heat,temp,hum,window,window_heat,floor,under,rw,gw,roof_latent,ground_latent,pv,state.extra)
    return BemResult(new,sources,dlg,dl,upward,absorbed,emissivity,ground_flux,moisture,BemDiagnostics(*diagnostics))
