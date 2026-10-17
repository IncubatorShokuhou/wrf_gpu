"""WRF Noah-MP BEP/BEM coupling, stored only on table-defined urban columns.

Default ISURBAN maps to UTYPE=2 in urban_var_init. LCZ/NUDAPT and PV/green
roofs are refused at load time. Static indices avoid carrying Registry-sized
building arrays across rural columns or padding inactive building types.
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.urban_bep_model import BepState,BepForcing,bep_column_step
from gpuwrf.physics.urban_bep_blend import RuralSurface,UrbanPblSources,noahmp_bep_blend
from gpuwrf.coupling.urban_slucm_hook import mptable_natural,urban_noahmp_static


@dataclass(frozen=True,eq=False)
class UrbanMultilayerBundle:
    option: int
    indices: tuple
    shape: tuple
    geometry: object
    bem: object
    fraction_default: object
    material_temperature: object
    isurban: int
    natural: int


class UrbanMultilayerCarry(NamedTuple):
    state: object
    fraction: object
    sources: UrbanPblSources
    surface_layer: object
    # BEP+BEM only: WRF QGR_URB3D (history QGR_URB4D, 'rh'), full (ny, nx) grid.
    # urban_var_init seeds every column with SMOIS(top) (module_sf_urban.F:2947);
    # bep_bem resets urban columns to 0 every call and adds green-roof moisture
    # only under GR_FLAG=1 (module_sf_bep_bem.F:1059,1097-1106), refused at load.
    green_roof: object = None


class UrbanMultilayerDiagnostics(NamedTuple):
    indices: object
    theta: object
    qv: object
    u: object
    v: object


def load_multilayer_bundle(table_dir,*,option,ivgtyp,isurban,dataset='MODIFIED_IGBP_MODIS_NOAH'):
    from pathlib import Path
    from gpuwrf.physics.urban_bep_static import read_bep_classes
    from gpuwrf.physics.urban_bem_static import bem_class
    ivg=np.asarray(ivgtyp)
    if np.any((ivg>=51)&(ivg<=61)):
        raise NotImplementedError('BEP/BEM WUDAPT LCZ requires URBPARM_LCZ.TBL; only default UTYPE=2 is qualified')
    indices=tuple(np.flatnonzero(ivg.reshape(-1)==int(isurban)).tolist())
    _,classes=read_bep_classes(Path(table_dir)/'URBPARM.TBL',np.float32)
    geo=classes[1];bem=bem_class(geo,1,np.float32) if option==3 else None
    return UrbanMultilayerBundle(int(option),indices,ivg.shape,geo,bem,geo.raw['FRC_URB'][1],
        geo.raw['TBLEND'][1],int(isurban),mptable_natural(table_dir,dataset))


def initial_multilayer_carry(bundle,*,tslb,frc_urb2d,nz,smois=None):
    """Literal urban_var_init active entries; inactive buildings/levels omitted.

    ``smois`` is the Noah-MP-initialized SMOIS(nsoil, ny, nx) that WRF passes as
    SMOIS_URB (module_physics_init.F:3474); BEP+BEM keeps its top layer as QGR_URB3D.
    """
    if bundle.option==3 and smois is None:
        raise ValueError('BEP+BEM urban_var_init needs Noah-MP SMOIS for QGR_URB3D')
    idx=jnp.asarray(bundle.indices,jnp.int32);n=bundle.geometry.n;nb=len(bundle.bem.levels) if bundle.bem else 1
    floors=max(bundle.bem.levels) if bundle.bem else n
    t0=jnp.asarray(tslb,jnp.float32)[0].reshape(-1)[idx]
    def fill(value,*shape):return jnp.broadcast_to(value.reshape((-1,)+(1,)*len(shape)),(len(idx),)+shape)
    zero=jnp.zeros_like(t0);tm=jnp.full_like(t0,bundle.material_temperature)
    state=BepState(fill(tm,2,n,10),fill(tm,4,n,10),fill(t0,2,10),fill(zero,4,n),fill(zero,2,n),fill(zero,2))
    if bundle.option==3:
        from gpuwrf.physics.urban_bem_model import BemState
        state=BemState(state.roof,fill(tm,nb,4,n,10),state.ground,fill(zero,nb,4,n),state.roof_heat,state.ground_heat,
            fill(t0,nb,floors),fill(zero+.01,nb,floors),fill(t0,nb,4,n),fill(zero,nb,4,n),
            fill(t0,nb,2,floors-1,10),fill(t0,nb,2,10),fill(zero,2,n),fill(zero,2),
            fill(zero,2,n),fill(zero,2),fill(t0,2,n),{})
    fraction=jnp.asarray(frc_urb2d,jnp.float32).reshape(-1)[idx]
    fraction=jnp.where((fraction>0)&(fraction<=1),fraction,jnp.float32(bundle.fraction_default))
    zz=fill(zero,nz);ones=jnp.ones_like(zz)
    sources=UrbanPblSources(*([zz]*12),fill(zero+1,nz+1),ones,fraction)
    from gpuwrf.coupling.urban_surface_layer import UrbanSurfaceLayer
    surface_layer=UrbanSurfaceLayer(*(jnp.zeros(bundle.shape,jnp.float32) for _ in range(6)))
    green_roof=None if bundle.option!=3 else jnp.asarray(smois,jnp.float32)[0].reshape(bundle.shape)
    return UrbanMultilayerCarry(state,fraction,sources,surface_layer,green_roof)


def assemble_multilayer_forcing(state,bundle,rad,dt,grid=None):
    """WRF phy_prep mass winds, full pressure/density and cumulative DZ8W."""
    from gpuwrf.coupling.physics_couplers import (_u_mass,_v_mass,_rho_from_state,
        _wrf_phy_rv_over_rd,_wrf_phy_prep_rho_from_state,_wrf_hydrostatic_pressure_from_state)
    idx=jnp.asarray(bundle.indices,jnp.int32);dtype=jnp.float32
    def columns(value):return jnp.moveaxis(jnp.asarray(value,dtype),0,-1).reshape(-1,value.shape[0])[idx]
    def surface(value):return jnp.broadcast_to(jnp.asarray(value,dtype),bundle.shape).reshape(-1)[idx]
    zw=jnp.asarray(state.ph,dtype)/jnp.float32(9.81);dz=columns(zw[1:]-zw[:-1])
    heights=[jnp.zeros((len(idx),),dtype)]
    for k in range(dz.shape[-1]):heights.append(heights[-1]+dz[:,k])
    if rad.urban_radiation is None:raise ValueError('BEP/BEM requires held direct/diffuse radiation and solar angles')
    direct,diffuse,declination,hour_angle=rad.urban_radiation
    theta=jnp.asarray(state.theta,dtype)/(1+_wrf_phy_rv_over_rd(dtype)*jnp.asarray(state.qv,dtype))
    pressure=state.p;rho=_rho_from_state(state)
    if grid is not None and grid.metrics is not None:
        pressure,_=_wrf_hydrostatic_pressure_from_state(state,grid.metrics)
        rho=_wrf_phy_prep_rho_from_state(state,grid.metrics)
    forcing=BepForcing(jnp.stack(heights,axis=-1),columns(_u_mass(state)),columns(_v_mass(state)),
        columns(theta),columns(rho),columns(pressure),surface(rad.swnorm),surface(rad.glw),
        surface(rad.coszen),surface(declination),surface(hour_angle),jnp.full((len(idx),),dt,dtype))
    return forcing,columns(state.qv),surface(direct),surface(diffuse)


def apply_urban_multilayer(bundle,carry,*,atmosphere,forcing,nm,ust,dt):
    """After rural Noah-MP, run urban BEP/BEM and blend literal noahmp_urban."""
    profile,qv,direct,diffuse,latitude=atmosphere
    idx=jnp.asarray(bundle.indices,jnp.int32);shape=bundle.shape
    take=lambda value:jnp.asarray(value,jnp.float32).reshape(-1)[idx]
    rural=RuralSurface(take(nm.tsk),take(nm.hfx),take(nm.qfx),take(nm.grdflx),take(nm.albedo),take(nm.emiss),take(ust))
    if bundle.option==2:result=bep_column_step(carry.state,profile,bundle.geometry);raw_moisture=None
    else:
        from gpuwrf.physics.urban_bem_model import bem_column_step
        rain=jnp.maximum((take(forcing.prcpconv)+take(forcing.prcpnonc))*jnp.float32(dt),0.)
        result=bem_column_step(carry.state,profile,bundle.bem,latitude=latitude,sw_direct=direct,sw_diffuse=diffuse,
            rain=rain,qv=qv,fraction=carry.fraction)
        raw_moisture=result.moisture_source
    surface,sources=noahmp_bep_blend(result,profile,rural,carry.fraction,raw_moisture=raw_moisture)
    def put(old,new):return jnp.asarray(old).reshape(-1).at[idx].set(jnp.asarray(new,jnp.asarray(old).dtype)).reshape(shape)
    nmout=nm._replace(tsk=put(nm.tsk,surface.skin),hfx=put(nm.hfx,surface.heat),qfx=put(nm.qfx,surface.moisture),
        lh=put(nm.lh,surface.latent),grdflx=put(nm.grdflx,surface.ground),albedo=put(nm.albedo,surface.albedo),
        emiss=put(nm.emiss,surface.emissivity))
    if nm.history is not None and 'GRDFLX' in nm.history:
        history=dict(nm.history);history['GRDFLX']=put(history['GRDFLX'],surface.ground);nmout=nmout._replace(history=history)
    step=UrbanMultilayerDiagnostics(idx,profile.theta[:,0],qv[:,0],profile.u[:,0],profile.v[:,0])
    green_roof=carry.green_roof
    if green_roof is not None:  # bep_bem: qgr_urb3d(ix,iy)=0. on every urban column (GR_FLAG=0)
        green_roof=put(green_roof,jnp.zeros((len(bundle.indices),),jnp.float32))
    return nmout,put(ust,surface.ust),carry._replace(state=result.state,sources=sources,green_roof=green_roof),step


def multilayer_history_overwrite(t2,q2,u10,v10,step,psfc):
    """surface_driver:3510–3523: lowest-model-level urban diagnostics."""
    idx=step.indices;shape=t2.shape
    ps=jnp.asarray(psfc,jnp.float32).reshape(-1)[idx]
    t=step.theta/((jnp.float32(1.e5)/ps)**jnp.float32(287./1004.5))
    def put(old,new):return old.reshape(-1).at[idx].set(jnp.asarray(new,old.dtype)).reshape(shape)
    return put(t2,t),put(q2,step.qv),put(u10,step.u),put(v10,step.v)
