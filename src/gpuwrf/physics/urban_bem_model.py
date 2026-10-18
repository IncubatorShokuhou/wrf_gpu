"""WRF BEP+BEM canyon/building step (solid roof, optional rooftop PV and green roofs from URBPARM.TBL)."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import urban_bep_components as c
from gpuwrf.physics.urban_bep_model import BepResult,_shadow,apply_gauss_plan
from gpuwrf.physics.urban_bem_building import BuildingState,building_step,real_literal


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
    pv_temperature: object  # (...,2,18) T_PV_URB3D, held when PV=0
    extra: object           # PV/green-roof state (PV_EXTRA/GR_EXTRA keys) when selected; else {} or unchanged Registry arrays


# Rooftop-PV diagnostic held in BemState.extra (WRF EP_PV_URB3D, Registry 'rh').
PV_EXTRA=('ep_pv',)
# Green-roof (GR_FLAG=1) state in BemState.extra: TRV_URB4D, QR_URB4D (levels x 10 soil layers,
# deepest first), SFRV/LFRV/DRAIN_URB3D/4D per roof level, the urban_var_init QR seed for the
# never-evolved levels nzu+1..nz_um-1, and the per-call history sums QGR/TGR/DRAINGR.
GR_EXTRA=('gr_temperature','gr_moisture','gr_moisture0','gr_sensible','gr_latent','gr_drain','qgr','tgr','draingr')
NZ_UM=18  # module_sf_bep_bem.F nz_um: history sums run over iz=2..nz_um-1


def pv_fraction(bem):
    return float(bem.params.get('pv_frac_roof',0.))


def green_config(bem):
    from gpuwrf.physics.urban_green_roof import config_from_params
    return config_from_params(bem.params)


def extensions(bem):
    """Active BEM extensions, sorted: the minimal PV/green-roof selection a checkpoint binds (E78)."""
    return tuple(name for name,active in (('green_roof',green_config(bem) is not None),('pv',pv_fraction(bem)>0.)) if active)


def extra_shapes(selection,count,levels):
    """Exact BemState.extra key set and shapes of an extension selection (initial_extra layout)."""
    shapes={}
    if 'green_roof' in selection:
        layer=(count,2,levels,10);level=(count,2,levels)
        shapes.update(gr_temperature=layer,gr_moisture=layer,gr_moisture0=(count,),gr_sensible=level,
            gr_latent=level,gr_drain=level,qgr=(count,),tgr=(count,),draingr=(count,))
    if 'pv' in selection:
        shapes['ep_pv']=(count,)
    return shapes


def initial_extra(bem,count,dtype=jnp.float32,*,smois_top=None,levels=None):
    """urban_var_init (module_sf_urban.F:2930-2955) values of the PV/green-roof branches.

    EP_PV=0; green roofs: TRV=TBLEND, QR=SMOIS(top), SFRV=LFRV=DRAIN=0, QGR=SMOIS(top), TGR=DRAINGR=0.
    """
    extra={}
    if pv_fraction(bem)>0.:
        extra['ep_pv']=jnp.zeros((count,),dtype)
    if green_config(bem) is not None:
        if smois_top is None:
            raise ValueError('BEP+BEM green roofs need the Noah-MP SMOIS top layer (QR_URB4D seed)')
        n=bem.bep.n if levels is None else levels
        q=jnp.asarray(smois_top,dtype).reshape(count)
        extra.update(gr_temperature=jnp.full((count,2,n,10),jnp.asarray(bem.params['tblend'],dtype)),
            gr_moisture=jnp.broadcast_to(q[:,None,None,None],(count,2,n,10)),gr_moisture0=q,
            gr_sensible=jnp.zeros((count,2,n),dtype),gr_latent=jnp.zeros((count,2,n),dtype),
            gr_drain=jnp.zeros((count,2,n),dtype),qgr=q,tgr=jnp.zeros((count,),dtype),draingr=jnp.zeros((count,),dtype))
    return extra


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
    geo=bem.bep;dtype=state.roof.dtype;pv=pv_fraction(bem);green=green_config(bem)
    wall,window,wall_heat,window_heat=averages
    if green is None:
        roof=(state.roof[...,-1]**4)**.25
    else:
        from gpuwrf.physics.urban_green_roof import roof_average
        roof=roof_average(state.roof[...,-1],state.extra['gr_temperature'][...,-1],green.fraction)
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
            if pv>0. or green is not None:
                # upward_rad PV/green-roof terms (module_sf_bep_bem.F:5315-5321).
                # tr_av**4. is libm powf (real exponent); tpvlev**4 an integer power.
                P=jnp.asarray(pv,dtype);tpv4=state.pv_temperature[...,d,i]**4;tr4=roof[...,d,i]**real_literal(4.,dtype)
                emit=emit-(er*sigma*(1.-P)*tr4+.79*sigma*P*tpv4+(1-er)*forcing.lw*(1.-P)
                           +(1-jnp.asarray(.79,dtype))*P*forcing.lw)*ss[i]*bs/(ws+bs)/geo.nd
                inc=inc+forcing.lw*ss[i]*bs/(ws+bs)/geo.nd
                absorbed=absorbed+((1.-ar)*forcing.sw*(1.-P)+(1.-jnp.asarray(.11,dtype))*forcing.sw*P)*ss[i]*bs/(ws+bs)/geo.nd
                gfl=(1.-ar)*forcing.sw*(1-P)+er*forcing.lw*(1-P)+P*er*sigma*tpv4-er*sigma*tr4
                if green is None:
                    gfl=gfl+state.roof_heat[...,d,i]+state.roof_latent[...,d,i]
                else:
                    G=real_literal(green.fraction,dtype)
                    gfl=(gfl+(1-G)*state.roof_heat[...,d,i]+(state.extra['gr_sensible'][...,d,i]+state.extra['gr_latent'][...,d,i])*G
                         +(1.-G)*state.roof_latent[...,d,i])
            else:
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


def _green_history(ex,gstep,gr_sensible,gr_latent,bem,fraction,dtype):
    """New green-roof state and bep_bem's per-call QGR/TGR/DRAINGR sums (module_sf_bep_bem.F:1097-1106).

    Sums run over id, iz=2..nz_um-1, ig in WRF order; levels above nzu keep their urban_var_init
    values (TRV=TBLEND, QR=SMOIS top), drain there is 0. DRAINGR uses WRF's d_urb(iz-1), the
    building-type weight indexed by level, times roof_frac=FRC_URB*bs(1)/(bs(1)+ws(1)).
    """
    R=lambda x:real_literal(x,dtype);C=lambda x:jnp.asarray(x,dtype)
    trv=ex['gr_temperature'].at[...,:,1:gstep.temperature.shape[-2]+1,:].set(gstep.temperature)
    qr=ex['gr_moisture'].at[...,:,1:gstep.moisture.shape[-2]+1,:].set(gstep.moisture)
    drain=ex['gr_drain'].at[...,:,1:gstep.drain.shape[-1]+1].set(gstep.drain)
    geo=bem.bep;bs=C(geo.building);ws=C(geo.width)
    roof_frac=fraction*bs[0]/R(np.asarray(geo.building[0],np.dtype(dtype))+np.asarray(geo.width[0],np.dtype(dtype)))
    weights=list(bem.probabilities)+[0.]*NZ_UM
    tblend=C(bem.params['tblend']);q0=ex['gr_moisture0'];stored=trv.shape[-2];active=gstep.drain.shape[-1]+1
    draingr=jnp.zeros_like(q0);qgr=jnp.zeros_like(q0);tgr=jnp.zeros_like(q0)
    for d in range(2):
        for level in range(1,NZ_UM-1):
            if level<active and weights[level-1]>0.:
                draingr=draingr+C(weights[level-1])*roof_frac*drain[...,d,level]*C(1000.)
            for ig in range(10):
                qv=qr[...,d,level,ig] if level<stored else q0
                tv=trv[...,d,level,ig] if level<stored else tblend
                qgr=qgr+qv/R(2.)/R(16.)/R(10.)
                tgr=tgr+tv/R(2.)/R(16.)/R(10.)
    return dict(gr_temperature=trv,gr_moisture=qr,gr_sensible=gr_sensible,gr_latent=gr_latent,gr_drain=drain,
                qgr=qgr,tgr=tgr,draingr=draingr)


def bem_column_step(state:BemState,forcing,bem,*,latitude,sw_direct,sw_diffuse,rain,qv,fraction):
    """module_sf_bep_bem.F:BEP1D, including building/AC/ventilation and wet surfaces."""
    geo=bem.bep;dtype=state.roof.dtype;n=geo.n;C=lambda x:jnp.asarray(x,dtype)
    # E78: refuse a carry whose PV/green-roof state does not match the URBPARM selection before any use.
    pv_on=pv_fraction(bem)>0.
    if pv_on!=all(key in state.extra for key in PV_EXTRA):
        raise ValueError('BEP+BEM PV_FRAC_ROOF state (EP_PV_URB3D) differs from the URBPARM selection (E78)')
    green=green_config(bem);ex=state.extra
    if (green is not None)!=all(key in ex for key in GR_EXTRA):
        raise ValueError('BEP+BEM green-roof state (TRV/QR_URB4D) differs from the URBPARM GR_FLAG selection (E78)')
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
    if green is not None:
        from gpuwrf.physics import urban_green_roof as G
        col=lambda x:x[...,None,None]
        # roof_temp_veg runs on every urban level iz=2..nzu before BEM (:1588-1595).
        gstep=G.roof_temp_veg(ex['gr_temperature'][...,:,1:n,:],ex['gr_moisture'][...,:,1:n,:],
            ex['gr_sensible'][...,:,1:n],ex['gr_latent'][...,:,1:n],state.roof[...,:,1:n,4],col(forcing.sw),col(forcing.lw),
            pressure[...,None,1:n],col(dt),col(rain),col(G.irrigation(green,hour,dtype)),config=green,
            roof_conductivity5=bem.params['roof_conductivity'][4],roof_capacity5=bem.params['roof_capacity'][4],
            roof_depth5=bem.params['roof_depth'][4],pv=pv_fraction(bem),tpv=state.pv_temperature[...,:,1:n])
        gfrac=real_literal(green.fraction,dtype)
        tr_av=G.roof_average(state.roof[...,0,:,-1],ex['gr_temperature'][...,0,:,-1],green.fraction)
    buildings=[];pv_sensible={}
    for index,levels in enumerate(bem.levels):
        building=BuildingState(temp[...,index,:],hum[...,index,:],wall[...,index,:,:,:],window[...,index,:,:],
            floor[...,index,1,:,:],under[...,index,1,:],roof[...,1,levels,:])
        result=building_step(building,levels=levels,geo=bem.room,params=bem.params,dt=dt,hour=hour,
            outdoor_temperature=outdoor,outdoor_humidity=humidity,outdoor_density=rho,pressure=pressure,
            sw_wall=sw_wall,lw_wall=lw_wall,sw=forcing.sw,lw=forcing.lw,wall_heat=state.wall_heat[...,index,:,:n],
            window_heat=state.window_heat[...,index,:,:n],
            # BEM hsrof=sfr_avb=(1-gr)*sfr+gr*sfrv, tr_av=tr_avb (module_sf_bep_bem.F:1528,1655).
            roof_heat=(state.roof_heat[...,0,levels] if green is None else
                       (1-gfrac)*state.roof_heat[...,0,levels]+gfrac*ex['gr_sensible'][...,0,levels]),
            roof_latent=state.roof_latent[...,0,levels],
            roof_average=(state.roof[...,0,levels,-1]**4)**.25 if green is None else tr_av[...,levels],
            # BEM tpv=tpvlevb(1,ibui); hsfluxpv uses tout/uout/vout(nlev+1) (module_sf_bem.F:635).
            pv_panel=(pv[...,0,levels],outdoor[...,levels],u[...,levels],v[...,levels]) if pv_on else None,
            green_roof=None if green is None else (green.fraction,gstep.roof_flux[...,0,levels-1]))
        buildings.append(result);b=result.state
        temp=temp.at[...,index,:].set(b.temperature);hum=hum.at[...,index,:].set(b.humidity)
        wall=wall.at[...,index,:,:,:].set(b.wall);window=window.at[...,index,:,:].set(b.window)
        for d in range(geo.nd):
            roof=roof.at[...,d,levels,:].set(b.roof);floor=floor.at[...,index,d,:,:].set(b.floor)
            under=under.at[...,index,d,:].set(b.ground)
        if pv_on:
            # hsfluxpv updates tpvlevb(1,ibui); then tpvlevb(2)=tpvlevb(1) and
            # sfrpv(id,iz)=-sfrbpv(id,ibui) (module_sf_bep_bem.F:1736,1798-1801).
            pv=pv.at[...,0,levels].set(result.pv.temperature).at[...,1,levels].set(result.pv.temperature)
            pv_sensible[levels]=-result.pv.sensible
        else:
            pv=pv.at[...,1,levels].set(pv[...,0,levels])
    roof_heat=state.roof_heat;roof_latent=state.roof_latent
    ground_heat=state.ground_heat;ground_latent=state.ground_latent
    wall_heat=state.wall_heat;window_heat=state.window_heat
    horizontal=[];qhorizontal=[]
    if green is not None:
        gr_sensible=ex['gr_sensible'];gr_latent=ex['gr_latent']
    for d in range(geo.nd):
        heat=[];moist=[];uu=[];vv=[];ee=[]
        for k in range(n):
            if k==0 or geo.ss[k]>0.:
                potential=ptg[d] if k==0 else roof[...,d,k,-1]*(pressure[...,k]/p0)**(-rcp)
                values=c.bem_flux_flat(5.,C(geo.roughness[d,k]),u[...,k],v[...,k],theta[...,k],theta0[...,k],potential,humidity[...,k],rho[...,k],pressure[...,k])
                uh,vh,sf,lf,eh=values
                water=gw[...,d] if k==0 else rw[...,d,k]
                lf=jnp.where(water>0.,-rho[...,k]*latent*(-(water*lf)/(rho[...,k]*latent)),0.)
                heated=sf
                if k in pv_sensible:
                    # buildings(): sfr=sfr+pv*sfrpv; thb=-sfr/(da*cp); sfr=sfr-pv*sfrpv (:2810-2817).
                    share=real_literal(pv_fraction(bem),dtype)*pv_sensible[k]
                    heated=sf+share;sf=heated-share
                if green is not None and k>0:
                    # flux_flat_roof on the new ptrv/qr and the gr-weighted blend (:2796-2809).
                    gu,gv,gs,gl,ge=G.flux_flat_roof(5.,u[...,k],v[...,k],theta[...,k],theta0[...,k],gstep.potential[...,d,k-1],
                        rho[...,k],humidity[...,k],pressure[...,k],forcing.sw,gstep.moisture[...,d,k-1,:],gr_type=green.gr_type,pv=pv_fraction(bem))
                    gr_sensible=gr_sensible.at[...,d,k].set(gs);gr_latent=gr_latent.at[...,d,k].set(gl)
                    flux=-((1.-gfrac)*heated+gfrac*gs)/(rho[...,k]*1004.)
                    vh=(1.-gfrac)*vh+gfrac*gv;uh=(1.-gfrac)*uh+gfrac*gu;eh=(1.-gfrac)*eh+gfrac*ge
                    wet=-(gfrac*gl+(1.-gfrac)*lf)/(rho[...,k]*latent)
                else:
                    flux=None;wet=None
                if k==0:
                    ground_heat=ground_heat.at[...,d].set(sf);ground_latent=ground_latent.at[...,d].set(lf)
                else:
                    roof_heat=roof_heat.at[...,d,k].set(sf);roof_latent=roof_latent.at[...,d,k].set(lf)
                heat.append(-heated/(rho[...,k]*1004.) if flux is None else flux)
                moist.append(-lf/(rho[...,k]*latent) if wet is None else wet);uu.append(uh);vv.append(vh);ee.append(eh)
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
    ep_pv=jnp.zeros_like(forcing.sw) if pv_on else None
    for index,result in enumerate(buildings):
        if pv_on:  # ep_pv_urb3d=ep_pv_urb3d+meso_urb_ac*d_urb(ibui)*eppvlev1D(ibui) (:1082)
            ep_pv=ep_pv+factor*C(bem.probabilities[index])*result.pv.power
        for k in range(bem.levels[index]):
            for j,value in enumerate(result[1:6]):diagnostics[j]=diagnostics[j]+factor*C(bem.probabilities[index])*value[...,k]
    extra=dict(state.extra,ep_pv=ep_pv) if pv_on else state.extra
    if green is not None:
        extra=dict(extra,**_green_history(ex,gstep,gr_sensible,gr_latent,bem,fraction,dtype))
    new=BemState(roof,wall,ground,wall_heat,roof_heat,ground_heat,temp,hum,window,window_heat,floor,under,rw,gw,roof_latent,ground_latent,pv,extra)
    return BemResult(new,sources,dlg,dl,upward,absorbed,emissivity,ground_flux,moisture,BemDiagnostics(*diagnostics))
