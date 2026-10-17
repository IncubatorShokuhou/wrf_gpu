"""WRF Goddard radiation_driver -> goddardrad pairing (ra_lw/sw=5, mp8).

REAL wrapper, original explicit DOUBLE solvers, one paired cadence refresh.
Held fluxes feed land/history; no radiation recomputation at output. The new
diagnostic counter records upstream's pre-zero negative SW heating events.
"""
from collections import namedtuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.physics.ra_goddard_common import sounding_interp, ozone_interp, opt_cloud
from gpuwrf.physics.ra_goddard_lw import solve_lw_column
from gpuwrf.physics.ra_goddard_sw import solve_sw_column
from gpuwrf.physics.column_tiling import tiled_column_apply


GoddardRadiationDiagnostics = namedtuple('GoddardRadiationDiagnostics',
    (*pc.RRTMGRadiationDiagnostics._fields,'sw_heating_negative_count',
     'cod2d_out','ctop2d_out','swddir','swddni','swddif','radius_inputs'), defaults=(None,))
RADIUS_HISTORY=('RE_ICE_GSFC','RE_CLOUD_GSFC','RE_RAIN_GSFC',
                'RE_SNOW_GSFC','RE_GRAUPEL_GSFC','RE_HAIL_GSFC')
HISTORY_FIELDS=('COD2D_OUT','CTOP2D_OUT','SWDDIR','SWDDNI','SWDDIF')


def history_fields(diagnostics,variable_subset=None):
    fields = {name:getattr(diagnostics,name.lower()) for name in HISTORY_FIELDS
            if variable_subset is None or name in variable_subset}
    if diagnostics.radius_inputs is None:
        raise ValueError('Goddard history lacks held radiation-call radius inputs (E78)')
    ny,nx=diagnostics.glw.shape
    # Output boundary only: transfer the retained tensor once, then split on
    # the host. Six device slices/transposes would add avoidable output launches.
    radii=None
    for i,name in enumerate(RADIUS_HISTORY):
        if variable_subset is None or name in variable_subset:
            if radii is None:
                radii=np.asarray(jax.device_get(diagnostics.radius_inputs))
            fields[name]=np.moveaxis(radii[...,i].reshape(ny,nx,-1),-1,0)
    return fields


def initial_diagnostics(state):
    fields={name:None if '_clear_' in name else jnp.zeros_like(state.t_skin,dtype=jnp.float32)
            for name in pc.RRTMGRadiationDiagnostics._fields}
    fields['shadow_mask']=jnp.zeros(state.t_skin.shape,jnp.int32)
    fields['cloud_fraction']=jnp.zeros_like(state.qv,dtype=jnp.float32)
    fields['sw_heating_negative_count']=jnp.uint32(0)
    nz,ny,nx=state.qv.shape
    fields['radius_inputs']=jnp.zeros((ny*nx,nz,6),jnp.float32)
    fields.update({name:jnp.zeros_like(state.t_skin,dtype=jnp.float32)
                   for name in ('cod2d_out','ctop2d_out','swddir','swddni','swddif')})
    return GoddardRadiationDiagnostics(**fields)


def goddard_column(T,p,p8w,t8w,qv,condensate,radius,rho,dz,pi,cloud,latitude,julday,cosz,solcon,albedo,emiss,tsk,*,run_shortwave=True):
    """Literal REAL4 goddardrad wrapper; model-order inputs, six species.

    radius is the radiation_driver's supplied ice/cloud/rain/snow/graupel/hail
    radius in micrometres. The MP8 caller supplies zero rain/graupel/hail radii,
    as the original radiation_driver (1890-1892), rather than inventing values.
    """
    T,p,p8w,t8w,qv,condensate,radius,rho,dz,pi,cloud,latitude,cosz,solcon,albedo,emiss,tsk=[jnp.asarray(x,jnp.float32) for x in (T,p,p8w,t8w,qv,condensate,radius,rho,dz,pi,cloud,latitude,cosz,solcon,albedo,emiss,tsk)]
    n=T.shape[0]
    pressure=(p8w[::-1]*jnp.float32(.01)).at[0].set(jnp.float32(0))
    temp=T[::-1];water=jnp.maximum(qv[::-1],jnp.float32(0))
    water=jnp.maximum(jnp.float32(0),water/(jnp.float32(1)+water))
    pmid=p[::-1]*jnp.float32(.01)
    cf=cloud[::-1]
    sounding=sounding_interp(latitude,julday)
    ozone=ozone_interp(sounding,pmid)
    concentration=jnp.maximum(condensate[::-1,:],jnp.float32(0))*rho[::-1,None]*jnp.float32(1e3)
    swtau,swssa,swasy=opt_cloud(dz[::-1],concentration,radius[::-1,:],shortwave=True)
    lwtau,_lwssa,_lwasy=opt_cloud(dz[::-1],concentration,radius[::-1,:],shortwave=False)
    icb=jnp.argmin(jnp.abs(pmid-jnp.float32(700))).astype(jnp.int32)+1
    # WRF REAL at every wrapper assignment, DOUBLE only inside the two cores.
    with jax.enable_x64(True):
        lwzero=jnp.zeros_like(lwtau)
        lw_net,lw_down,lw_up=solve_lw_column(pressure,temp,water,ozone,t8w[0],tsk,jnp.full((10,),emiss,jnp.float32),cf,lwtau,lwzero,lwzero,lwzero)
        def day(_):
            zero=jnp.zeros_like(swtau)
            # Upstream WRF4.7.1 :5315 reads uninitialized taux. Manager policy:
            # mirror -finit-real=zero at exactly that read, never substitute taucl.
            reduction_tau=jnp.zeros((n,),jnp.float64)
            # vmap predicates mixed day/night columns. Give inactive columns
            # a finite unused solar angle before the original night selection.
            safe_cosz=jnp.where(cosz>jnp.float32(.0001),cosz,jnp.float32(.5))
            return solve_sw_column(pressure,temp,water,ozone,cf,safe_cosz,swtau,swssa,swasy,zero,zero,zero,albedo,icb,reduction_tau)
        def night(_):
            z=jnp.zeros((n+1,),jnp.float32)
            return z,z,z,jnp.zeros((4,),jnp.float32)
        sw_net,sw_down,sw_up,sw_surface=(lax.cond(cosz>jnp.float32(.0001),day,night,None)
            if run_shortwave else night(None))
    daytime=cosz>jnp.float32(.0001)
    sw_net=jnp.where(daytime,sw_net*solcon*cosz,jnp.float32(0))
    fac=jnp.float32(.01)*jnp.float32(9.81)/jnp.float32(1004.5)
    lw_heat=-fac*(lw_net[:-1]-lw_net[1:])/(pressure[:-1]-pressure[1:])
    sw_heat=-fac*(sw_net[:-1]-sw_net[1:])/(pressure[:-1]-pressure[1:])
    negative=jnp.sum(sw_heat<0,dtype=jnp.uint32)
    # Literal original SW correction :2137-2141; count before zeroing.
    sw_heat=jnp.maximum(sw_heat,jnp.float32(0))
    sw_heat=jnp.where(daytime,sw_heat,jnp.float32(0))
    lw_rate=lw_heat[::-1]/pi;sw_rate=sw_heat[::-1]/pi
    down=jnp.where(daytime,sw_down*solcon*cosz,jnp.float32(0))
    up=jnp.where(daytime,sw_up*solcon*cosz,jnp.float32(0))
    cod=lax.fori_loop(0,n,lambda k,a:a+swtau[k,7],jnp.float32(0))
    top=jnp.min(jnp.where(swtau[:,7]>=jnp.float32(.01),jnp.arange(n),n))
    ctop=jnp.where(top<n,pmid[jnp.minimum(top,n-1)],jnp.float32(-999))
    dni=jnp.where(daytime,(sw_surface[0]+sw_surface[2])*solcon,jnp.float32(0))
    direct=jnp.where(daytime,dni*cosz,jnp.float32(0))
    diffuse=jnp.where(daytime,(sw_surface[1]+sw_surface[3])*solcon*cosz,jnp.float32(0))
    return (lw_rate,sw_rate,down[-1],-up[-1],lw_down[-1],-lw_up[-1],
            down[0],-up[0],lw_down[0],-lw_up[0],negative,cod,ctop,direct,dni,diffuse)


@jax.jit
def solve_goddard_columns(columns,julday,solcon):
    def solve(c):
        def active(c):
            return jax.vmap(lambda **kw:goddard_column(**kw,julday=julday,solcon=solcon,run_shortwave=True))(**c)
        def night(c):
            return jax.vmap(lambda **kw:goddard_column(**kw,julday=julday,solcon=solcon,run_shortwave=False))(**c)
        # Original goddardrad skips swrad for a wholly nocturnal vector chunk.
        return lax.cond(jnp.any(c['cosz']>jnp.float32(.0001)),active,night,c)
    ncol=columns['T'].shape[0]
    # Level-pair/weighted recurrences launch many small kernels. A 128-column
    # tile repeats that launch sequence 14 times on the 42x42 Swiss grid.
    # 2048 retains a bounded working set while covering that grid in one call.
    return solve(columns) if ncol<=2048 else tiled_column_apply(solve,columns,ncol=ncol,tile_cols=2048)


def paired_radiation(state,grid,*,time_utc=None,lead_seconds=0.,solar_lead_seconds=None,clock_base=None,radiation_static=None,land_state=None,julday=None,_kernel_call=None):
    """Build the common WRF caller operands once, then solve both Goddard halves."""
    _sw,lw,albedo,emiss,geometry,_topo=pc._rrtmg_column_inputs(
        state,grid,time_utc=time_utc,lead_seconds=lead_seconds,solar_lead_seconds=solar_lead_seconds,
        clock_base=clock_base,radiation_static=radiation_static,land_state=land_state,use_mp_re=0)
    nz,ny,nx=state.theta.shape;ncol=ny*nx
    flat=lambda x:jnp.asarray(x,jnp.float32).reshape(ncol,-1)
    surf=lambda x:jnp.asarray(x,jnp.float32).reshape(ncol)
    p8w=lw.pressure_interfaces if lw.pressure_interfaces is not None else pc._interface_pressure_from_state(state)
    t8w=lw.temperature_interfaces if lw.temperature_interfaces is not None else pc._interface_temperature_from_state(state,pc._from_columns(lw.T))
    pfull=pc._to_columns(state.p).astype(jnp.float32)
    pi=jnp.power(pfull/jnp.float32(1e5),jnp.float32(np.float32(287)/np.float32(1004.5)))
    static=pc._radiation_static_for_grid(state.t_skin.shape,grid,radiation_static,state.t_skin.dtype)
    lat=static.xlat_deg if static is not None else pc._grid_lat_lon(state.t_skin.shape,grid,state.t_skin.dtype)[0]
    if julday is None:
        julday,_=pc._resolve_clock_parts(time_utc,clock_base)
    solcon=jnp.asarray(pc._solcon_for_time(time_utc,lead_seconds,clock_base=clock_base),jnp.float32)
    raw_radii=[getattr(state,key) for key in ('re_ice','re_cloud','re_snow')]
    if any(x is None for x in raw_radii):
        raise ValueError('Goddard radiation requires held MP8 effective radii (use_mp_re=1)')
    ri,rc,rs=[flat(pc._to_columns(x))*jnp.float32(1e6) for x in raw_radii]
    zero=jnp.zeros_like(ri)
    radius=jnp.stack((ri,rc,zero,rs,zero,zero),axis=-1)
    condensate=jnp.stack((flat(lw.qi),flat(lw.qc),flat(pc._to_columns(state.qr)),flat(lw.qs),flat(lw.qg),zero),axis=-1)
    columns=dict(T=flat(lw.T),p=flat(lw.p),p8w=flat(p8w),t8w=flat(t8w),qv=flat(lw.qv),
        condensate=condensate,radius=radius,rho=flat(lw.rho),dz=flat(lw.dz),pi=flat(pi),cloud=flat(lw.cloud_fraction),
        latitude=surf(lat),cosz=surf(geometry.coszen),albedo=surf(albedo),emiss=surf(emiss),tsk=surf(lw.surface_temperature))
    kernel_call=_kernel_call or (lambda fn,*args:fn(*args))
    values=kernel_call(solve_goddard_columns,columns,julday,solcon)
    lw_rate,sw_rate,swdown,swup,glw,glw_up,swdnt,swupt,lwdnt,lwupt,negative,cod,ctop,direct,dni,diffuse=values
    field=lambda x:jnp.reshape(x,(ny,nx)).astype(jnp.float32)
    diagnostics=initial_diagnostics(state)._replace(surface_albedo=field(albedo),surface_emissivity=field(emiss),
        coszen=field(geometry.coszen),swdown=field(swdown),swnorm=field(swdown),swup=field(swup),swup_topographic=field(swup),
        glw=field(glw),glw_up=field(glw_up),topographic_correction_factor=jnp.ones((ny,nx),jnp.float32),
        sw_toa_down=field(swdnt),sw_toa_up=field(swupt),lw_toa_down=field(lwdnt),lw_toa_up=field(lwupt),
        cloud_fraction=pc._from_columns(lw.cloud_fraction).astype(jnp.float32),sw_heating_negative_count=jnp.sum(negative,dtype=jnp.uint32),
        cod2d_out=field(cod),ctop2d_out=field(ctop),swddir=field(direct),swddni=field(dni),swddif=field(diffuse),
        radius_inputs=radius)
    return pc._from_columns((lw_rate+sw_rate).reshape(ny,nx,nz)).astype(state.theta.dtype),diagnostics


def refresh_driver(carry,namelist,lead_seconds,run_radiation,clock_base):
    from gpuwrf.diagnostics.census import count_work
    from gpuwrf.kernels.dyn_carry_fp32 import like
    from gpuwrf.runtime.operational_mode import _rad_clock_base,_lake_radiation_state,_explicit_ruc
    interval=float(namelist.radiation_interval_s) or float(namelist.dt_s)*int(namelist.radiation_cadence_steps)
    def refresh(_):
        land_for_rad=_lake_radiation_state(carry,namelist)
        if _explicit_ruc(namelist):
            from gpuwrf.coupling.ruc_surface_hook import ruc_radiation_surface
            land_for_rad=ruc_radiation_surface(carry.ruc_land)
        rate,diag=paired_radiation(carry.state,namelist.grid,time_utc=namelist.time_utc,lead_seconds=lead_seconds,
            solar_lead_seconds=lead_seconds+.5*interval,clock_base=_rad_clock_base(clock_base),
            radiation_static=namelist.radiation_static,land_state=land_for_rad)
        diag=diag._replace(sw_heating_negative_count=carry.radiation_diagnostics.sw_heating_negative_count+diag.sw_heating_negative_count)
        rate,diag=like((rate,diag),(carry.rthraten,carry.radiation_diagnostics))
        return rate,diag,count_work(carry.census,'radiation_tendency_calls')
    held=(carry.rthraten,carry.radiation_diagnostics,carry.census)
    result=refresh(None) if run_radiation is True else held if run_radiation is False else lax.cond(run_radiation,refresh,lambda _:held,None)
    return carry.replace(rthraten=result[0],radiation_diagnostics=result[1],census=result[2])
