"""Paired FLG/GFDL radiation with one cadence solve and held native outputs.

CPU-qualified column reference route. GPU execution stays fail-closed until a
column kernel is qualified; the vertical recurrences are never domain-vmapped
into a GPU timestep graph. CPU lax.map bounds live work to a single column.
"""
from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.physics import ra_flg as flg
from gpuwrf.physics.gfdleta_column import prepare_solver,native_column


@dataclass(frozen=True)
class GfdlGrid:
    full: bytes
    half: bytes
    top: bytes

    def solver(self):
        return prepare_solver(np.frombuffer(self.full,np.float32),
            np.frombuffer(self.half,np.float32),np.frombuffer(self.top,np.float32)[0])

    def baked_tables(self):
        return self.solver().gpuwrf_baked_tables


def prepare_grid(grid):
    if grid is None:raise ValueError('GFDL radiation needs the actual WRF sigma grid')
    full=np.asarray(grid.vertical.eta_levels,np.float32)
    # WRF ARW ZNU is the exact REAL midpoint of its ZNW interfaces.
    half=np.float32(.5)*(full[:-1]+full[1:])
    bundle=GfdlGrid(full.tobytes(),half.tobytes(),np.asarray(grid.vertical.top_pressure_pa,np.float32).tobytes())
    bundle.solver()
    return bundle


class GfdlCarry(NamedTuple):
    top: object
    bottom: object
    cuppt: object
    acfrcv: object
    ncfrcv: object
    acfrst: object
    ncfrst: object
    lw_rate: object
    sw_rate: object
    htopr: object
    hbotr: object
    cfracl: object
    cfracm: object
    cfrach: object


LegacyRadiationDiagnostics=namedtuple('LegacyRadiationDiagnostics',
    (*pc.RRTMGRadiationDiagnostics._fields,'rswtoa','rlwtoa','czmean','rswinc','swddir','swddif','swddni','gsw'))


def initial_gfdl(state):
    z=jnp.zeros(state.t_skin.shape,jnp.float32)
    zi=jnp.zeros(state.t_skin.shape,jnp.int32)
    rate=jnp.zeros(state.theta.shape,jnp.float32)
    # HTOP/HBOT/CUPPT start at WRF's allocation zero (Registry.EM_COMMON:
    # 1245-1251); cu=0 has no producer. Only ETARA:372 sets HBOT=nz+1 AFTER
    # the first solve. Do not substitute that reset state at initialization.
    return GfdlCarry(z,z,z,z,zi,z,zi,rate,rate,z,z,z,z,z)


def initial_diagnostics(state):
    z=jnp.zeros(state.t_skin.shape,jnp.float32)
    fields=[z for _ in range(15)];fields[9]=jnp.ones_like(z);fields[10]=jnp.zeros_like(z,dtype=jnp.int32)
    base=pc.RRTMGRadiationDiagnostics(*fields,*(None for _ in range(8)),jnp.zeros_like(state.qv,dtype=jnp.float32))
    return LegacyRadiationDiagnostics(*base,*(z for _ in range(8)))


def previous_columns(gfdl,ncol):
    """Flatten native saved state without converting its INTEGER counters."""
    return {key:jnp.asarray(getattr(gfdl,key)).reshape(ncol)
            for key in ('top','bottom','cuppt','acfrcv','ncfrcv','acfrst','ncfrst')}


def history_fields(diag,variable_subset=None):
    """Only native legacy writes; internal TOA fluxes are not aliases of OLR.

    WRF Registry keeps RSWTOA/RLWTOA separate from its RRTMG flux/history
    members. Those members stay at their allocation zero with these schemes.
    """
    zero=jnp.zeros_like(diag.glw)
    fields={name:zero for name in ('SWDNB','SWUPB','LWDNB','LWUPB','SWDNT','SWUPT','LWDNT','LWUPT','OLR',
        'SWDNBC','SWUPBC','LWDNBC','LWUPBC','SWDNTC','SWUPTC','LWDNTC','LWUPTC')}
    fields.update(SWDOWN=diag.swdown,GLW=diag.glw,GSW=diag.gsw,COSZEN=diag.coszen,
        SWDDIR=diag.swddir,SWDDIF=diag.swddif,SWDDNI=diag.swddni,CLDFRA=diag.cloud_fraction)
    return {key:value for key,value in fields.items() if variable_subset is None or key in variable_subset}


def _clock(namelist,lead_seconds,clock_base):
    when=pc._coerce_datetime_utc(namelist.time_utc)
    if when is None:raise ValueError('FLG/GFDL radiation needs the actual initial UTC date')
    date=jnp.asarray((when.month,when.day,when.year),jnp.int32)
    gmt=jnp.float32(when.hour+when.minute/60.+when.second/3600.)
    start_hour=jnp.floor(gmt+.5).astype(jnp.int32)
    minutes=jnp.asarray(lead_seconds,jnp.float32)/jnp.float32(60.)
    initial_day,initial_minute=pc._resolve_clock_parts(namelist.time_utc,clock_base)
    julian=pc._wrf_solar_julian_real(initial_day,initial_minute,lead_seconds)
    return date,start_hour,minutes,jnp.float32(julian),gmt,jnp.int32(when.timetuple().tm_yday)


def paired_radiation(state,namelist,lead_seconds,clock_base=None,*,land_state=None,gfdl=None):
    """Exact WRF caller fields; paired output is the held prognostic theta rate."""
    from gpuwrf.runtime.operational_mode import _rrtmg_pbl_cloud_kwargs
    scheme=int(namelist.ra_lw_physics)
    interval=float(namelist.radiation_interval_s) or float(namelist.dt_s)*int(namelist.radiation_cadence_steps)
    _sw,lw,albedo,emiss,geometry,_=pc._rrtmg_column_inputs(
        state,namelist.grid,time_utc=namelist.time_utc,lead_seconds=lead_seconds,
        solar_lead_seconds=jnp.asarray(lead_seconds,jnp.float32)+jnp.float32(.5*interval),
        clock_base=clock_base,radiation_static=namelist.radiation_static,
        land_state=land_state,use_mp_re=0,force_real=True,
        **_rrtmg_pbl_cloud_kwargs(namelist))
    nz,ny,nx=state.theta.shape;ncol=ny*nx
    flat=lambda a:jnp.asarray(a,jnp.float32).reshape(ncol,-1)
    surf=lambda a:jnp.asarray(a,jnp.float32).reshape(ncol)
    full_pressure=pc._to_columns(state.p).astype(jnp.float32)
    pi=jnp.power(full_pressure/jnp.float32(1.e5),jnp.float32(np.float32(287.)/np.float32(1004.5)))
    p8w=lw.pressure_interfaces
    t8w=lw.temperature_interfaces
    if p8w is None or t8w is None:raise ValueError('FLG/GFDL require actual WRF phy_prep interface fields')
    static=pc._radiation_static_for_grid(state.t_skin.shape,namelist.grid,namelist.radiation_static,jnp.float32)
    if static is None:raise ValueError('FLG/GFDL require supplied WRF XLAT/XLONG')
    snow=jnp.zeros_like(state.t_skin,dtype=jnp.float32)
    if land_state is not None and hasattr(land_state,'sneqv'):
        snow=jnp.where(state.xland<1.5,jnp.asarray(land_state.sneqv,jnp.float32),snow)
    # The Noah-MP loader binds VEGFRA/100 to its supplied SHDFAC field.
    vegetation=getattr(namelist.noahmp_static,'shdfac',None)
    if scheme==99 and vegetation is None:raise ValueError('GFDL caller needs supplied WRF VEGFRA (Noah-MP SHDFAC)')
    vegetation=jnp.zeros_like(state.t_skin) if vegetation is None else jnp.float32(100.)*jnp.asarray(vegetation,jnp.float32)
    rho=pc._to_columns(pc._wrf_phy_prep_rho_from_state(state,namelist.grid.metrics,output_dtype=jnp.float32)) if scheme==7 else lw.rho
    columns=dict(t=flat(lw.T),p=flat(lw.p),p8w=flat(p8w),t8w=flat(t8w),pi=flat(pi),
        # radiation_driver.F:1401-1430 merges QC_BL/QI_BL before both
        # RAD_FLG and ETARA; reuse that view, never the raw State moisture.
        qv=flat(pc._to_columns(state.qv)),qc=flat(lw.qc),
        qi=flat(lw.qi),qs=flat(lw.qs),
        qr=flat(pc._to_columns(state.qr)),qg=flat(lw.qg),rho=flat(rho),
        latitude=surf(static.xlat_deg),longitude=surf(static.xlong_deg),
        albedo=surf(albedo),emissivity=surf(emiss),tsk=surf(lw.surface_temperature),
        xland=surf(state.xland),snow=surf(snow),vegetation=surf(vegetation))
    date,hour,minutes,julian,gmt,julday=_clock(namelist,lead_seconds,clock_base)
    def back(a):return jnp.moveaxis(jnp.reshape(a,(ny,nx,nz)),-1,0)
    surface=lambda a:a.reshape(ny,nx)
    if scheme==99:
        if gfdl is None or namelist.gfdl_grid is None:raise ValueError('GFDL initialized provider/carry missing')
        solver=namelist.gfdl_grid.solver()
        previous=previous_columns(gfdl,ncol)
        degrad=jnp.float32(np.float32(3.141592653589793)/np.float32(180.))
        def one(args):
            c,old=args;c=dict(c,latitude=c['latitude']*degrad,longitude=c['longitude']*degrad)
            raw=native_column(solver,c,(date,hour,minutes,julian),old,dt=namelist.dt_s,
                              stepra=namelist.radiation_cadence_steps)
            lw_rate=raw['tendl'][0,::-1,0]/c['pi'];sw_rate=raw['tends'][0,::-1,0]/c['pi']
            return dict(lw_rate=lw_rate,sw_rate=sw_rate,total_rate=(jnp.zeros_like(lw_rate)+lw_rate)+sw_rate,
                **{key:raw[key].reshape(()) for key in ('glw','gsw','rswin','rswinc','rswtoa','rlwtoa','czmean',
                    'acfrcv','ncfrcv','acfrst','ncfrst','cfracl','cfracm','cfrach')},
                cloud=raw['cldfra'][0,:nz,0])
        with jax.enable_x64(True):out=lax.map(one,(columns,previous))
        gfdl=gfdl._replace(top=jnp.zeros_like(gfdl.top),bottom=jnp.full_like(gfdl.bottom,nz+1),
            cuppt=jnp.zeros_like(gfdl.cuppt),htopr=gfdl.top,hbotr=gfdl.bottom,
            lw_rate=back(out['lw_rate']),sw_rate=back(out['sw_rate']),
            **{key:surface(out[key]) for key in ('acfrcv','ncfrcv','acfrst','ncfrst','cfracl','cfracm','cfrach')})
        swdown,gsw,glw,swtop,lwtop,coszen=(out[k] for k in ('rswin','gsw','glw','rswtoa','rlwtoa','czmean'))
        cloud=back(out['cloud'])
        clear_down=out['rswinc'];mean_cos=out['czmean']
        direct=diffuse=dni=jnp.zeros_like(swdown)
    elif scheme==7:
        solar_constant=jnp.float32(pc._solcon_for_time(namelist.time_utc,lead_seconds,clock_base=clock_base))
        def one(c):
            # The upper mass padding stays at WRF's allocation value zero:
            # frame/module_domain.F:773 -> tools/gen_allocs.c:487/491 ->
            # frame/wrf_num_bytes_between.c:GET_INITIAL_DATA_VALUE. Physical
            # moisture updates/input operate only through kde-1. RAD_FLG:8365
            # reads the padding for its top interface, then floors at1e-20.
            pad=lambda a:jnp.concatenate((a,jnp.zeros((1,),a.dtype)))
            pre=flg.preprocess_column(c['p8w'],c['t8w'],pad(c['t']),pad(c['rho']),
                pad(c['qv']),pad(c['qc']),pad(c['qr']),pad(c['qi']),pad(c['qs']),pad(c['qg']),
                c['albedo'],c['emissivity'],c['tsk'],c['latitude'],c['longitude'],
                julday,gmt,minutes,jnp.float32(interval/60.),jnp.float32(geometry.declination_rad),solar_constant)
            return flg._column_from_preprocessed(pre,c['pi'])
        out=lax.map(one,columns)
        swdown,gsw,glw=(out[k] for k in ('swdn','gsw','glw'))
        # Upstream RAD_FLG assigns USWTOP to the whole tile array each column
        # (:8996), so the final column owns the native scalar tile broadcast.
        swtop=jnp.full_like(swdown,out['sw_up_top'][-1]);lwtop=out['lw_up_top']
        coszen=surf(geometry.coszen)
        cloud=pc._from_columns(lw.cloud_fraction).astype(jnp.float32)
        clear_down=mean_cos=jnp.zeros_like(swdown)
        direct,diffuse,dni=(out[k] for k in ('direct','diffuse','dni'))
    else:
        raise ValueError('Expected paired FLG7/7 or GFDL99/99')
    diagnostics=initial_diagnostics(state)._replace(surface_albedo=surface(columns['albedo']),
        surface_emissivity=jnp.ones((ny,nx),jnp.float32) if scheme==99 else surface(columns['emissivity']),coszen=geometry.coszen.astype(jnp.float32),
        swdown=surface(swdown),swnorm=surface(swdown),swup=surface(swdown-gsw),
        swup_topographic=surface(swdown-gsw),glw=surface(glw),lw_toa_up=surface(lwtop),
        sw_toa_up=surface(swtop),cloud_fraction=cloud,rswtoa=surface(swtop),rlwtoa=surface(lwtop),
        czmean=surface(mean_cos),rswinc=surface(clear_down),swddir=surface(direct),swddif=surface(diffuse),swddni=surface(dni),
        gsw=surface(gsw))
    return back(out['total_rate']).astype(state.theta.dtype),diagnostics,gfdl


def refresh_driver(carry,namelist,lead_seconds,run_radiation,clock_base):
    from gpuwrf.runtime.operational_mode import _rad_clock_base
    from gpuwrf.diagnostics.census import count_work
    def refresh(_):
        rate,diag,gfdl=paired_radiation(carry.state,namelist,lead_seconds,_rad_clock_base(clock_base),
                                      land_state=carry.noahmp_land,gfdl=carry.gfdl_radiation)
        return rate,diag,gfdl,count_work(carry.census,'radiation_tendency_calls')
    held=(carry.rthraten,carry.radiation_diagnostics,carry.gfdl_radiation,carry.census)
    result=refresh(None) if run_radiation is True else held if run_radiation is False else lax.cond(run_radiation,refresh,lambda _:held,None)
    return carry.replace(rthraten=result[0],radiation_diagnostics=result[1],gfdl_radiation=result[2],census=result[3])
