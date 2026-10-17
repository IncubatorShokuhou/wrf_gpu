"""WRF surface_driver CASE SSIBSCHEME array/caller seam.

The paired revised-MM5 surface layer runs before this hook. Land diagnostics
then overwrite its handles (WRF restores held SSiB values on subsequent calls).
Three-layer soil + all 48 snow scalars remain resident in OperationalCarry.
"""
from __future__ import annotations
from typing import Any, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.physics.lsm_ssib import ssib_columns

class SsibStatic(NamedTuple):
    ivgtyp: Any
    xland: Any
    xice: Any
    xlat: Any
    mminlu: int
    emiss: Any
    has_ice: bool = False
    ice_threshold: float = 0.02
    initial_mavail: Any = None
    initial_znt: Any = None

class SsibRadiationSurface(NamedTuple):
    albedo: Any
    emiss: Any

def ssib_radiation_surface(land,static):
    return SsibRadiationSurface(land.bedo,static.emiss)


def ssib_day(time_utc,lead_seconds,clock_base=None,yearlen=365.):
    """surface_driver:4311 DAY=FLOAT(INT(current JULIAN_IN+.01))+1.

    JULIAN_IN is zero-based and includes UTC fraction; the retained CLWRF
    calendar supplies the current year across leap/year boundaries when dated.
    """
    from gpuwrf.coupling.physics_couplers import _resolve_clock_parts,_wrf_solar_julian_real
    ghg_clock=getattr(clock_base,'ghg_clock',None)
    if ghg_clock is not None:
        from gpuwrf.physics.wrf_clwrf_ghg import wrf_julian_yearlen_at_lead
        julian,_=wrf_julian_yearlen_at_lead(ghg_clock,lead_seconds)
    else:
        day,minute=_resolve_clock_parts(time_utc,clock_base)
        julian=jnp.mod(_wrf_solar_julian_real(day,minute,lead_seconds),jnp.float32(yearlen))
    return jnp.float32(jnp.int32(jnp.float32(julian)+jnp.float32(.01)))+jnp.float32(1)


def ssib_surface_step(state,land,static,dt,grid,*,radiation,precipitation,day,
                      sw_physics,first_timestep=False,cloud_fraction=None,water_history=None,physics=None,entry_state=None):
    from gpuwrf.coupling.physics_couplers import _surface_column_view
    from gpuwrf.physics.surface_constants import CP_D,EP1
    physics=ssib_physics_inputs(state,grid) if physics is None else physics
    from types import SimpleNamespace
    view=SimpleNamespace(**{name:physics[name] for name in ("u","v","theta","qv","p","dz","t_air","psfc","rho")})
    shape=state.t_skin.shape; r=jnp.float32
    if land is None or static is None:
        raise ValueError('SSiB requires explicit ssib_static and seeded ssib_land')
    if precipitation is None:
        raise ValueError('SSiB requires the WRF precipitation carry')
    sw,lw,coszen=radiation
    # REAL radiation driver components are absolute W/m2, not fractions.
    # The caller uses them with ra_sw=3/4. Independent radiation adapters must
    # provide the same four components, rather than estimating their split.
    if hasattr(sw,'swvisdir'):
        rad=sw; soldn=rad.swdown
        components=[rad.swvisdir,rad.swvisdif,rad.swnirdir,rad.swnirdif]
    elif int(sw_physics) in (3,4):
        raise ValueError('SSiB ra_sw=3/4 requires four held spectral/direct radiation components')
    else:
        soldn=sw; components=[jnp.zeros(shape,r)]*4
    zero=jnp.zeros(shape,r)
    if cloud_fraction is None: cloud=zero
    else: cloud=jnp.maximum(r(0),jnp.max(jnp.minimum(jnp.asarray(cloud_fraction,r),r(1)),axis=0))  # surface_driver:4328-4331
    forcing=dict(ix=jnp.ones(shape,jnp.int32),jx=jnp.ones(shape,jnp.int32),ddtt=r(dt),
        itime=jnp.where(first_timestep,1,2),zlat=static.xlat,sunangle=coszen,
        ppl=precipitation.prcpnonc*r(dt),ppc=precipitation.prcpconv*r(dt),rlwdown=lw,
        zwind2=view.dz[0],umm=view.u[0],vmm=view.v[0],qm=view.qv[0],
        tm=view.t_air[0],pm=view.p[0],psur=view.psfc,ivgtyp=static.ivgtyp,
        swdown1=soldn,day=day,cloud=cloud,sw_physics=int(sw_physics),mminlu=static.mminlu,
        **dict(zip(('radfrac11','radfrac12','radfrac21','radfrac22'),components)))
    # The sea-ice branch is included only for a bundle that contains sea ice.
    if static.has_ice:
        forcing.update(yice=static.xice,ice_threshold=static.ice_threshold)
    active=static.xland<1.5; ice=static.xice>=static.ice_threshold
    new=ssib_columns(land=land,forcing=forcing,is_land=active,is_ice=ice)
    # TA actual argument is WRF T2, overwritten by TSK immediately after SSIB.
    new=new._replace(ta=new.tgeff)
    rho=jnp.asarray(view.rho[0],r); q=jnp.asarray(view.qv[0],r)
    theta_flux=new.xhsflx/(rho*r(CP_D)*(r(1)+r(.84)*q))
    qflux=new.xhlflx/rho
    fltv=(r(1)+r(EP1)*q)*theta_flux+r(EP1)*view.theta[0]*qflux
    u,v=view.u[0],view.v[0]; speed=jnp.sqrt(u*u+v*v)
    # WRF surface_driver:4392 overwrites T2 with TSK after SSIB.
    updates={'t_skin':new.tgeff,'roughness_m':new.xxz0,'soil_moisture':new.www1,
             'ustar':new.ustar,'hfx':new.xhsflx,'qfx':new.xhlflx,
             'theta_flux':theta_flux,'qv_flux':qflux,'fltv':fltv,
             'tau_u':-new.ustar**2*u/speed,'tau_v':-new.ustar**2*v/speed}
    values={name:jnp.where(active,jnp.asarray(value,getattr(state,name).dtype),getattr(state,name))
            for name,value in updates.items() if getattr(state,name,None) is not None}
    if entry_state is not None and getattr(state,'sfc_wspd',None) is not None:
        # surface_driver restores the incoming land WSPD on steps after ITIME=1.
        values['sfc_wspd']=jnp.where(active & ~jnp.asarray(first_timestep),entry_state.sfc_wspd,state.sfc_wspd)
    out=state.replace(**values)
    if water_history is None:
        raise ValueError('SSiB requires diagnostics from its paired surface-layer call')
    history={name:jnp.where(active,value,water_history[name]) for name,value in
        dict(t2=new.tgeff,q2=new.q2m,u10=new.uv10*u/speed,v10=new.uv10*v/speed,lh=new.elaten,hfx=new.xhsflx).items()}
    history['psfc']=view.psfc
    return out,new,history


from gpuwrf.io.ssib_history import SSIB_EXTRA_HISTORY,SSIB_STANDARD_HISTORY
SSIB_HISTORY={**SSIB_STANDARD_HISTORY,**SSIB_EXTRA_HISTORY}

def ssib_history_fields(land):
    out={name:getattr(land,attr) for name,attr in SSIB_HISTORY.items()}
    out['TSLB']=jnp.stack((land.tc,land.tgs,land.td))
    out['SMOIS']=jnp.stack((land.www1,land.www2,land.www3))
    out['ZS']=np.asarray([0.,.05,1.05],np.float32)
    out['DZS']=np.asarray([.05,1.,0.],np.float32)
    return jax.device_get(out)



def ssib_physics_inputs(state,grid):
    """One REAL phy_prep view shared by revised-MM5, SSiB and YSU.

    first_rk_step_part1.F:637/1132 passes p_hyd and p_hyd_w; temperature
    and pi_phy come from nonhydrostatic p+pb (big_step_utilities:4853-4856).
    No second surface solve or host transfer is used to construct PBL forcing.
    """
    from gpuwrf.coupling.physics_couplers import (_wrf_hydrostatic_pressure_profiles_from_state,
        _wrf_phy_prep_rho_from_state,_u_mass,_v_mass)
    if grid.metrics is None: raise ValueError('SSiB requires WRF phy_prep hybrid metrics')
    f=jnp.float32
    with jax.enable_x64(False):
        q=f(state.qv);theta=f(state.theta)/(f(1)+(f(461.6)/f(287.))*q)
        exner=(f(state.p)/f(100000.))**(f(287.)/f(1004.5))
        temp=theta*exner
        p,faces,psfc=_wrf_hydrostatic_pressure_profiles_from_state(state,grid.metrics,output_dtype=f)
        rho=_wrf_phy_prep_rho_from_state(state,grid.metrics,output_dtype=f)
        z=f(state.ph)/f(9.81);dz=jnp.maximum(z[1:]-z[:-1],f(1))
        return dict(theta=theta,t_air=temp,exner=exner,p=p,p8w=faces,psfc=psfc,rho=rho,dz=dz,
                    qv=q,u=_u_mass(state,dtype=f),v=_v_mass(state,dtype=f))


def ssib_surface_inputs(physics):
    return {name:physics[key][0] for name,key in
        (('u','u'),('v','v'),('temperature','t_air'),('qv','qv'),('pressure','p'),('dz','dz'))} | {'psfc':physics['psfc']}


def ssib_ysu_step(state,land,static,grid,dt,*,physics,surface_diagnostics,history,rthraten=None,ysu_topdown_pblmix=1,
                  _with_sources=False):
    """Original YSU caller operands; consume actual SSiB HFX/QFX/UST/BR."""
    from gpuwrf.physics.pbl_ysu import ysu_columns
    from gpuwrf.coupling.scan_adapters import _apply_pbl_increment,_apply_ysu_cloud_rates
    from gpuwrf.coupling.physics_couplers import _theta_m_tendency_from_dry
    nz,ny,nx=state.theta.shape;ncol=ny*nx
    cols=lambda x:jnp.moveaxis(jnp.asarray(x,jnp.float32),0,-1).reshape(ncol,-1)
    flat=lambda x:jnp.asarray(x,jnp.float32).reshape(ncol)
    active=static.xland<1.5
    with jax.enable_x64(False):
        result=ysu_columns(cols(physics['u']),cols(physics['v']),cols(physics['t_air']),cols(physics['qv']),
            cols(physics['p']),cols(physics['p8w']),cols(physics['exner']),cols(physics['dz']),
            psfc=flat(physics['psfc']),znt=flat(state.roughness_m),ust=flat(state.ustar),
            hfx=flat(state.hfx),qfx=flat(state.qfx),wspd=flat(state.sfc_wspd),
            br=flat(jnp.where(active,land.rib,surface_diagnostics['br'])),
            psim=flat(surface_diagnostics['fm']),psih=flat(surface_diagnostics['fh']),dt=float(dt),
            xland=flat(static.xland),u10=flat(history['u10']),v10=flat(history['v10']),
            qc=cols(state.qc),qi=cols(state.qi),rthraten=None if rthraten is None else cols(rthraten),
            ysu_topdown_pblmix=int(ysu_topdown_pblmix))
        raw=dict(result)  # original dry/A-grid YSU rates for the once-RK source seam
        result['theta']=_theta_m_tendency_from_dry(result['theta'],result['qv'],cols(state.theta),cols(state.qv),jnp.float32)
    updated=_apply_pbl_increment(state,dt,result,ny=ny,nx=nx,nz=nz)
    updated=_apply_ysu_cloud_rates(updated,state,dt,result,ny=ny,nx=nx,nz=nz)
    if getattr(state,'pblh',None) is not None:
        updated=updated.replace(pblh=result['pblh'].reshape(ny,nx).astype(state.pblh.dtype))
    if state.cam_kvh3d is not None:
        from gpuwrf.physics.cammg.pbl import retain_columns
        updated=retain_columns(updated,exch_h=result['exch_h'])
    return (updated,raw) if _with_sources else updated


def ssib_dudhia_columns(state,land,static,namelist,lead_seconds,clock_base,physics=None):
    from gpuwrf.coupling.physics_couplers import (_dudhia_sw_column_inputs,_resolve_clock_parts,
        _wrf_solar_geometry_real,_wrf_solcon_real,_grid_lat_lon)
    from gpuwrf.physics.ra_sw_dudhia import solve_dudhia_sw_column
    physics=ssib_physics_inputs(state,namelist.grid) if physics is None else physics
    nz,ny,nx=state.theta.shape;cols=lambda x:jnp.moveaxis(x,0,-1).reshape(ny*nx,nz)
    with jax.enable_x64(False):
        column,geometry=_dudhia_sw_column_inputs(state,namelist.grid,time_utc=namelist.time_utc,
            lead_seconds=lead_seconds,clock_base=clock_base,radiation_static=namelist.radiation_static,
            land_state=ssib_radiation_surface(land,static))
        # radiation_driver:1207-1209 offsets XTIME by RADT/2 for COSZEN;
        # radconst/DECLIN/SOLCON retain the current Julian (:1189-1191).
        rad_static=namelist.radiation_static
        lat,lon=(_grid_lat_lon(state.t_skin.shape,namelist.grid,jnp.float32)
                 if rad_static is None else (rad_static.xlat_deg,rad_static.xlong_deg))
        julian,minute=_resolve_clock_parts(namelist.time_utc,clock_base)
        interval=float(namelist.radiation_interval_s)
        if interval==0.0 and int(namelist.radiation_cadence_steps)>1:
            interval=float(namelist.dt_s)*int(namelist.radiation_cadence_steps)
        solar_minutes=(jnp.float32(lead_seconds)/jnp.float32(60)
                       +jnp.float32(interval/60)*jnp.float32(.5))
        geometry=_wrf_solar_geometry_real(lat,lon,julian,minute,lead_seconds,
                                         solar_xtime_minutes=solar_minutes)
        solcon=jnp.broadcast_to(_wrf_solcon_real(julian,minute,lead_seconds),state.t_skin.shape).reshape(ny*nx)
        # SWPARA uses its own RO=P/(R*T); its supplied RHO01D is unused (:360).
        column=column._replace(T=cols(physics['t_air']),p=cols(physics['p']),dz=cols(physics['dz']),
                               albedo=jnp.zeros_like(column.albedo),coszen=geometry.coszen.reshape(ny*nx),solcon=solcon)
        raw=solve_dudhia_sw_column(column)
    return raw,geometry,physics


def ssib_radiation(state,land,static,namelist,lead_seconds,clock_base,ozone=None):
    """One Dudhia SW + RRTMG LW solve; never substitute RRTMG SW heating."""
    from gpuwrf.coupling.physics_couplers import _rrtmg_column_inputs,RRTMGRadiationDiagnostics,_from_columns
    from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column
    from gpuwrf.physics.p3.radiation import mp_re_active
    sw,geometry,physics=ssib_dudhia_columns(state,land,static,namelist,lead_seconds,clock_base)
    sw_input,lw_input,albedo,emiss,_geometry,_topo=_rrtmg_column_inputs(
        state,namelist.grid,time_utc=namelist.time_utc,lead_seconds=lead_seconds,clock_base=clock_base,
        radiation_static=namelist.radiation_static,land_state=ssib_radiation_surface(land,static),
        use_mp_re=int(mp_re_active(namelist)),mp_physics=int(namelist.mp_physics),
        ozone_vmr_override=ozone,use_sgs_cloud=False,force_real=True)
    with jax.enable_x64(False):
        lw=solve_rrtmg_lw_column(lw_input,debug=False,with_clear_sky=True)
        nz,ny,nx=state.theta.shape
        rate=((_from_columns(sw.heating_rate.reshape(ny,nx,nz))+_from_columns(lw.heating_rate))/physics['exner']).astype(state.theta.dtype)
        shape=state.t_skin.shape;zero=jnp.zeros(shape,jnp.float32);one=jnp.ones(shape,jnp.float32)
        down=sw.gsw.reshape(shape);up=down*albedo
        diag=RRTMGRadiationDiagnostics(albedo,emiss,geometry.coszen,down,zero,up,up,
            lw.flux_down[...,0],lw.flux_up[...,0],one,jnp.zeros(shape,jnp.int32),
            zero,zero,lw.flux_down[...,-1],lw.flux_up[...,-1],
            zero,zero,zero,zero,lw.clear_flux_down[...,-1],lw.clear_flux_up[...,-1],
            lw.clear_flux_down[...,0],lw.clear_flux_up[...,0],
            _from_columns(sw_input.cloud_fraction))
    return rate.astype(state.theta.dtype),diag
