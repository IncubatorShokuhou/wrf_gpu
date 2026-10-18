"""WRF BEM indoor building energy (module_sf_bem.F), solid roof with optional rooftop PV.

Six-surface room radiation plans are static; the changing radiation, material
temperatures, ventilation, occupants, humidity and AC terms remain on device.
Rooftop PV panels (PV_FRAC_ROOF>0) follow radfluxspv/hsfluxpv; green roofs are
handled by the caller (urban_bem_model).
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.urban_bep_components import bem_regtemp,bem_reghum,bem_air_cond
from gpuwrf.physics.urban_bep_model import apply_gauss_plan
from gpuwrf.physics.urban_bep_static import gauss_plan
from gpuwrf.physics.urban_bep_pbl import _invert


def _parallel(x,y,T):
    return (T(2)/(T(3.141592653)*x*y))*(
        np.log(np.sqrt((T(1)+x*x)*(T(1)+y*y)/(T(1)+x*x+y*y)))
        +y*np.sqrt(T(1)+x*x)*np.arctan(y/np.sqrt(T(1)+x*x))
        +x*np.sqrt(T(1)+y*y)*np.arctan(x/np.sqrt(T(1)+y*y))-y*np.arctan(y)-x*np.arctan(x))


def _normal(x,y,z,T):
    return (T(1)/(T(3.141592653)*y))*(y*np.arctan(T(1)/y)+x*np.arctan(T(1)/x)
        -np.sqrt(z)*np.arctan(T(1)/np.sqrt(z))+T(.25)*(
        np.log((T(1)+x*x)*(T(1)+y*y)/(T(1)+z))
        +y*y*np.log(y*y*(T(1)+z)/(z*(T(1)+y*y)))
        +x*x*np.log(x*x*(T(1)+z)/(z*(T(1)+x*x)))))


@dataclass(frozen=True,eq=False)
class RoomGeometry:
    width: object
    length: object
    height: object
    area: np.ndarray
    short_plan: object
    long_plan: object
    long_view_groups: tuple
    window_fraction: object
    window_albedo: object
    window_emissivity: object


def room_geometry(width,length,window_fraction,window_albedo,dtype):
    T=np.dtype(dtype).type;aw=T(width);bw=T(length);zw=T(5);pw=T(window_fraction)
    albwal=T(.3);albwin=T(window_albedo);albm=pw*albwin+(T(1)-pw)*albwal
    a=np.zeros((6,6),dtype)
    a[0,0]=T(-1);a[0,1]=albm*_parallel(aw/bw,zw/bw,T)
    a[0,2]=albm*(bw/aw)*_normal(aw/zw,bw/zw,(aw*aw+bw*bw)/(zw*zw),T);a[0,3]=a[0,2]
    a[0,4]=albwal*(bw/zw)*_normal(zw/aw,bw/aw,(bw*bw+zw*zw)/(aw*aw),T);a[0,5]=a[0,4]
    a[1]=a[0];a[1,0]=a[0,1];a[1,1]=T(-1)
    a[2,0]=albm*(aw/bw)*_normal(bw/zw,aw/zw,(bw*bw+aw*aw)/(zw*zw),T);a[2,1]=a[2,0];a[2,2]=T(-1)
    a[2,3]=albm*_parallel(zw/aw,bw/aw,T)
    a[2,4]=albwal*(aw/zw)*_normal(zw/bw,aw/bw,(aw*aw+zw*zw)/(bw*bw),T);a[2,5]=a[2,4]
    a[3]=a[2];a[3,2]=a[2,3];a[3,3]=T(-1)
    a[4,0]=albm*(zw/bw)*_normal(bw/aw,zw/aw,(bw*bw+zw*zw)/(aw*aw),T);a[4,1]=a[4,0]
    a[4,2]=albm*(zw/aw)*_normal(aw/bw,zw/bw,(aw*aw+zw*zw)/(bw*bw),T);a[4,3]=a[4,2]
    a[4,4]=T(-1);a[4,5]=albwal*_parallel(aw/zw,bw/zw,T)
    a[5]=a[4];a[5,4]=a[4,5];a[5,5]=T(-1)
    # WRF algebra_long rows are ceil,floor,wall3,wall4,wall1,wall2.
    emwal=T(.9);emwin=T(.9);emav=(T(1)-pw)*emwal+pw*emwin
    reflect=np.asarray([T(1)-emav]*4+[T(1)-emwal]*2,dtype)
    views=np.empty((6,6),dtype)
    for i in range(6):
        for j in range(6):
            views[i,j]=T(0) if i==j else a[i,j]/(albm if j<4 else albwal)
    long=(np.eye(6,dtype=dtype)-views*reflect[None,:])[[5,4,2,3,0,1]]
    # RHS groups retain the source operation order, rather than a reordered GEMV.
    groups=(
      (_parallel(aw/zw,bw/zw,T),zw/aw,_normal(aw/bw,zw/bw,(aw*aw+zw*zw)/(bw*bw),T),zw/bw,_normal(bw/aw,zw/aw,(bw*bw+zw*zw)/(aw*aw),T)),
      (_parallel(zw/aw,bw/aw,T),aw/bw,_normal(bw/zw,aw/zw,(bw*bw+aw*aw)/(zw*zw),T),aw/zw,_normal(zw/bw,aw/bw,(aw*aw+zw*zw)/(bw*bw),T)),
      (_parallel(aw/bw,zw/bw,T),bw/aw,_normal(aw/zw,bw/zw,(aw*aw+bw*bw)/(zw*zw),T),bw/zw,_normal(zw/aw,bw/aw,(bw*bw+zw*zw)/(aw*aw),T)))
    area=np.asarray([zw*aw,zw*aw,zw*bw,zw*bw,aw*bw,aw*bw],dtype)
    return RoomGeometry(aw,bw,zw,area,gauss_plan(a),gauss_plan(long),groups,pw,albwin,emwin)


def internal_shortwave(incoming,geo):
    area=jnp.asarray(geo.area,incoming.dtype);pw=jnp.asarray(geo.window_fraction,incoming.dtype)
    albwin=jnp.asarray(geo.window_albedo,incoming.dtype)
    transmitted=jnp.zeros_like(incoming[...,0]);total=jnp.asarray(0.,incoming.dtype)
    for i in range(4):transmitted=transmitted+(area[i]*pw)*((1.-albwin)*incoming[...,i])
    for i in range(6):total=total+area[i]
    transmitted=transmitted/total
    rhs=jnp.stack([-transmitted]*6,axis=-1)
    return apply_gauss_plan(rhs,geo.short_plan)


def internal_longwave(wall,window,geo):
    dtype=wall.dtype;pw=jnp.asarray(geo.window_fraction,dtype);ew=jnp.asarray(.9,dtype)
    wm=(1-pw)*ew;win=pw*jnp.asarray(geo.window_emissivity,dtype);sigma=jnp.asarray(5.67e-8,dtype)
    w=wall**4;v=window**4;rows=[]
    for i in range(6):
        fp,sx,fx,sy,fy=[jnp.asarray(x,dtype) for x in geo.long_view_groups[0 if i<2 else (1 if i<4 else 2)]]
        if i<2:
            bw=ew*sigma*w[...,4+i]*fp+sigma*(wm*w[...,2]+wm*w[...,3])*sx*fx+sigma*(wm*w[...,0]+wm*w[...,1])*sy*fy
            bv=sigma*(win*v[...,2]+win*v[...,3])*sx*fx+sigma*(win*v[...,0]+win*v[...,1])*sy*fy
        else:
            opposite=(3,2,1,0)[i-2]
            pair=(1,0) if i<4 else (2,3)
            bw=wm*sigma*w[...,opposite]*fp+sigma*(wm*w[...,pair[0]]+wm*w[...,pair[1]])*sx*fx+sigma*(ew*w[...,4]+ew*w[...,5])*sy*fy
            bv=win*sigma*v[...,opposite]*fp+sigma*(win*v[...,pair[0]]+win*v[...,pair[1]])*sx*fx
        rows.append(bw+bv)
    return apply_gauss_plan(jnp.stack(rows,axis=-1),geo.long_plan)


def wall_temperature(temp,depth,conductivity,capacity,flux_inner,flux_outer,dt):
    """module_sf_bem.F:wall/wall_coeff/tridia, two flux boundary conditions."""
    n=temp.shape[-1];dtype=temp.dtype
    dz=jnp.asarray(depth,dtype);k=jnp.asarray(conductivity,dtype);cs=jnp.asarray(capacity,dtype)
    kc=dt[...,None]/(dz*cs) if hasattr(dt,'ndim') and dt.ndim else dt/(dz*cs)
    kf=jnp.concatenate(((2*k[0]/dz[0])[None],2*k[1:n]/(dz[1:]+dz[:-1])))
    k1=kc*kf;k2=kc[...,:-1]*kf[1:]*cs[:-1]/cs[1:]
    lower=jnp.concatenate((jnp.zeros_like(k1[...,:1]),-k1[...,1:]),axis=-1)
    upper=jnp.concatenate((-k2,jnp.zeros_like(k1[...,:1])),axis=-1)
    diag=jnp.concatenate(((1+k2[...,0])[...,None],1+k1[...,1:-1]+k2[...,1:],(1+k1[...,-1])[...,None]),axis=-1)
    rhs=temp.at[...,0].set(temp[...,0]+flux_inner*kc[...,0])
    rhs=rhs.at[...,-1].set(temp[...,-1]+flux_outer*kc[...,-1])
    return _invert(lower,diag,upper,rhs)


def real_literal(value,dtype):
    """A REAL constant hidden from XLA algebraic rewrites (x/c -> x*(1/c), x**.5 -> sqrt).

    gfortran -O2 keeps IEEE division by a non-power-of-two constant and calls libm
    powf for real exponents; XLA:CPU pow equals glibc powf when it sees an operand.
    """
    return jax.lax.optimization_barrier(jnp.asarray(value,dtype))


def _pv_hrad(emissivity,T):
    """sigma/((1-empv)/empv+1/F12+(1-em)/em), empv=.95, F12=1 (radfluxspv/hsfluxpv)."""
    em=T(emissivity)
    return T(5.67e-8)/((T(1)-T(.95))/T(.95)+T(1)/T(1)+(T(1)-em)/em)


def roof_radiation_pv(rs,rl,roof_average,pv_temperature,*,albedo,emissivity,pv):
    """module_sf_bem.F:radfluxspv (:2543-2585), literal REAL operation order."""
    dtype=roof_average.dtype;T=np.dtype(dtype).type
    alb=T(albedo);em=T(emissivity);pv=T(pv)
    return ((T(1)-alb)*(T(1)-pv)*rs+em*(T(1)-pv)*rl
            +pv*_pv_hrad(em,T)*(pv_temperature**4-roof_average**4)-em*T(5.67e-8)*(T(1)-pv)*roof_average**4)


class PvPanel(NamedTuple):
    temperature: object   # updated tpv [K]
    sensible: object      # hspv [W m-2]
    power: object         # eppv [W]


def pv_panel_step(tpv,roof_average,tair,u,v,rs,rl,dt,*,emissivity,pv,length,width):
    """module_sf_bem.F:hsfluxpv (:823-905): PV panel energy balance and production.

    ``tair``/``u``/``v`` are the urban-grid values at the roof level (tair(n+1)).
    """
    dtype=tpv.dtype;T=np.dtype(dtype).type;R=lambda x:real_literal(x,dtype)
    cm=T(2330.)*T(677.)*T(.0003)+T(1200.)*T(1250.)*T(.0005)+T(3000.)*T(500.)*T(.003)
    hrad=_pv_hrad(emissivity,T)
    uroof=(u**2+v**2)**R(.5)
    deltat=tpv-tair
    hf=T(2.5)*(T(40.)/T(100.)*uroof)**R(.5)
    third=jnp.abs(deltat)**R(T(1.)/T(3.))
    hc=T(9.482)*third/R(T(7.238)-T(1))
    hup=jnp.sqrt(hc**2+hf**2)
    hc=T(1.810)*third/R(T(1.382)+T(1))
    hdown=jnp.sqrt(hc**2+hf**2)
    # module parameter effpv=0.19 (:30); T_amb=25 -> T_amb+273.15 folded in REAL.
    enerpv=T(.19)*rs*jnp.minimum(T(1),T(1)-T(.005)*(tpv-(T(25)+T(273.15))))
    sw_d=(T(1)-T(.11))*rs
    lw_d=T(.79)*rl
    tpv4=tpv**R(4.)
    lwpv_out=T(.79)*T(5.67e-8)*tpv4
    lwupr=hrad*(roof_average**4-tpv4)
    hspv=(hup+hdown)*(tpv-tair)
    new=tpv+(sw_d+lw_d-lwpv_out+lwupr-hspv-enerpv)/R(cm)*dt
    eppv=enerpv*R(pv)*R(length)*R(width)
    return PvPanel(new,hspv,eppv)


class BuildingState(NamedTuple):
    temperature: object      # (...,18)
    humidity: object
    wall: object             # (...,4,18,10)
    window: object           # (...,4,18)
    floor: object            # (...,17,10), low face first
    ground: object           # (...,10), deep face first
    roof: object             # (...,10), indoor face first


class BuildingResult(NamedTuple):
    state: BuildingState
    sensible_exhaust: object
    latent_exhaust: object
    consumption: object
    sensible_ventilation: object
    latent_ventilation: object
    wall_ground_flux: object
    roof_ground_flux: object
    indoor_roof_heat: object
    pv: object = None        # PvPanel when PV_FRAC_ROOF>0


def building_step(state,*,levels,geo,params,dt,hour,outdoor_temperature,outdoor_humidity,outdoor_density,
                  pressure,sw_wall,lw_wall,sw,lw,wall_heat,window_heat,roof_heat,roof_latent,roof_average,
                  pv_panel=None,green_roof=None):
    """module_sf_bem.F:BEM solid-roof path, all active floors.

    ``pv_panel=(tpv, tair, u, v)`` selects WRF's PV_FRAC_ROOF>0 branches: radfluxspv
    for the roof (:423-428, old tpv) and hsfluxpv after the indoor fluxes (:634-638).
    """
    dtype=state.temperature.dtype;n=levels
    C=lambda x:jnp.asarray(x,dtype)
    pw=C(geo.window_fraction);ew=C(.9);sigma=C(5.67e-8);area=C(geo.area)
    albwin=C(geo.window_albedo);emwin=C(geo.window_emissivity)
    sw=sw[...,None] if sw.ndim==state.temperature.ndim-1 else sw
    rsint=internal_shortwave(jnp.moveaxis(sw_wall[...,:,:n],-2,-1),geo)
    # Floors exchange heat through their old material state. Radiation and
    # conduction for different rooms are independent within this WRF step.
    # Batch that axis; keep the six-surface and material elimination order.
    bottom=jnp.concatenate((state.ground[...,-1:],state.floor[...,:n-1,-1]),axis=-1)
    top=jnp.concatenate((state.floor[...,:n-1,0],state.roof[...,:1]),axis=-1)
    walls=jnp.concatenate((jnp.moveaxis(state.wall[...,:,:n,0],-2,-1),bottom[...,None],top[...,None]),axis=-1)
    windows=jnp.moveaxis(state.window[...,:,:n],-2,-1)
    irradiation=internal_longwave(walls,windows,geo)
    interior=(1.-.3)*rsint+ew*irradiation-ew*sigma*walls**4
    inner_heat=C(5.678)*C(1.09)*(state.temperature[...,:n,None]-walls)
    inner_window=C(5.678)*C(.99)*(state.temperature[...,:n,None]-windows)
    winrad=emwin*irradiation[...,:4]-emwin*sigma*windows**4
    outside_wall=jnp.moveaxis(state.wall[...,:,:n,-1],-2,-1)
    outside_sw=jnp.moveaxis(sw_wall[...,:,:n],-2,-1);outside_lw=jnp.moveaxis(lw_wall[...,:,:n],-2,-1)
    outer=(1.-C(params['wall_albedo']))*outside_sw+C(params['wall_emissivity'])*outside_lw-C(params['wall_emissivity'])*sigma*outside_wall**4
    wall=state.wall;window=state.window;floor=state.floor;ground=state.ground;roof=state.roof
    outer_flux=jnp.moveaxis(wall_heat[...,:,:n],-2,-1)+outer
    updated=wall_temperature(jnp.moveaxis(wall[...,:,:n,:],-3,-2),params['wall_depth'],params['wall_conductivity'],params['wall_capacity'],
        inner_heat[...,:4]+interior[...,:4],outer_flux,dt[...,None,None] if hasattr(dt,'ndim') and dt.ndim else dt)
    wall=wall.at[...,:,:n,:].set(jnp.moveaxis(updated,-2,-3))
    outside_window=emwin*outside_lw-emwin*sigma*windows**4
    updated_window=windows+(dt[...,None,None]/(C(2.268e6)*C(.006)))*(
        inner_window+winrad+jnp.moveaxis(window_heat[...,:,:n],-2,-1)+outside_window)
    window=window.at[...,:,:n].set(jnp.moveaxis(updated_window,-2,-1))
    if n>1:
        updated=wall_temperature(floor[...,:n-1,:],params['floor_depth'],params['floor_conductivity'],params['floor_capacity'],
            interior[...,:n-1,5]+inner_heat[...,:n-1,5],interior[...,1:n,4]+inner_heat[...,1:n,4],dt[...,None] if hasattr(dt,'ndim') and dt.ndim else dt)
        floor=floor.at[...,:n-1,:].set(updated)
    ground=wall_temperature(ground,params['ground_depth'],params['ground_conductivity'],params['ground_capacity'],
        jnp.zeros_like(hour),inner_heat[...,0,4]+interior[...,0,4],dt)
    if pv_panel is None:
        roof_rad=(1.-C(params['roof_albedo']))*sw[...,0]+C(params['roof_emissivity'])*lw-C(params['roof_emissivity'])*sigma*roof_average**4
        panel=None
    else:
        tpv,tair,u_roof,v_roof=pv_panel
        roof_rad=roof_radiation_pv(sw[...,0],lw,roof_average,tpv,albedo=params['roof_albedo'],
            emissivity=params['roof_emissivity'],pv=params['pv_frac_roof'])
        # BEM(...,bw=bs_u(1),bl=bs_u(2),...): eppv=enerpv*pv_frac_roof*bl*bw.
        panel=pv_panel_step(tpv,roof_average,tair,u_roof,v_roof,sw[...,0],lw,dt,emissivity=params['roof_emissivity'],
            pv=params['pv_frac_roof'],length=geo.length,width=geo.width)
    roof_ground=roof_heat+roof_rad+roof_latent
    if green_roof is None:
        roof=wall_temperature(roof,params['roof_depth'],params['roof_conductivity'],params['roof_capacity'],
            inner_heat[...,n-1,5]+interior[...,n-1,5],roof_ground,dt)
    else:
        # gr_flag=1: wall_gr couples roof layer 5 to the green-roof soil (module_sf_bem.F:735-737).
        from gpuwrf.physics.urban_green_roof import roof_wall_gr
        fraction,hfgr=green_roof
        roof=roof_wall_gr(roof,params['roof_depth'],params['roof_conductivity'],params['roof_capacity'],
            inner_heat[...,n-1,5]+interior[...,n-1,5],roof_ground,dt,fraction=fraction,hfgr=hfgr)
    temperature=state.temperature;humidity=state.humidity
    sensible=[];latent=[];power=[];svent=[];lvent=[]
    schedule=C(params['equipment_profile']);slot=hour.astype(jnp.int32)
    equipment=schedule[slot]*C(params['equipment_scale'])
    area_floor=C(geo.width)*C(geo.length);volume=area_floor*C(5.)
    hseq=area_floor*C(params['occupants'])*160.+area_floor*equipment
    hleq=area_floor*C(params['occupants'])*C(1.96e6/86400.)
    for level in range(n):
        t=temperature[...,level];q=humidity[...,level];dry=q/(1.-q)
        density=pressure[...,level]/(287.*(1.+.61*dry)*t);cpint=1004.*(1.+.84*dry)
        ventilation=(1.-C(params['beta']))*cpint*density*(volume/3600.)*(outdoor_temperature[...,level]-t)
        moisture=(1.-C(params['beta']))*C(2.45e6)*density*(volume/3600.)*(outdoor_humidity[...,level]-q)
        conduction=jnp.zeros_like(hour)
        for side in range(4):
            conduction=conduction+area[side]*(1-pw)*inner_heat[...,level,side]+area[side]*pw*inner_window[...,level,side]
        for side in range(4,6):conduction=conduction+area[side]*inner_heat[...,level,side]
        conduction=(-1)*conduction
        sensible_room=hseq+ventilation+conduction;latent_room=hleq+moisture
        capacity=density*cpint*volume
        need_s=bem_regtemp(params['ac_switch'],hour,dt,capacity,sensible_room,t,params['time_on'],params['time_off'],params['target_temperature'],params['temperature_gap'])
        need_l=bem_reghum(params['ac_switch'],hour,dt,volume,density,C(2.45e6),latent_room,q,params['time_on'],params['time_off'],params['target_humidity'],params['humidity_gap'])
        out_s,out_l,consumption,need_l=bem_air_cond(need_s,need_l,C(params['cop']))
        temperature=temperature.at[...,level].set(t+(dt/capacity)*(sensible_room-need_s))
        humidity=humidity.at[...,level].set(q+(dt/(volume*density*C(2.45e6)))*(latent_room-need_l))
        sensible.append(out_s);latent.append(out_l);power.append(consumption);svent.append(ventilation);lvent.append(moisture)
    result=BuildingState(temperature,humidity,wall,window,floor,ground,roof)
    return BuildingResult(result,jnp.stack(sensible,axis=-1),jnp.stack(latent,axis=-1),jnp.stack(power,axis=-1),
        jnp.stack(svent,axis=-1),jnp.stack(lvent,axis=-1),jnp.moveaxis(outer_flux,-2,-1),roof_ground,inner_heat[...,n-1,5],panel)
