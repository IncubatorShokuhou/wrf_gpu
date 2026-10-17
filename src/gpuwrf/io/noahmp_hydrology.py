"""Actual WRF runoff/tile tables and maps, initialized once before stepping."""
from dataclasses import replace
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.physics.noahmp.tables import NoahMPParameters,_parse_mptable_block
from gpuwrf.physics.noahmp.runoff_options import HydrologyParameters,RunoffParameters
from gpuwrf.physics.noahmp.tile_drain import TileParameters,equivalent_depth


SelectedHydrologyParameters=NamedTuple('SelectedHydrologyParameters',[(key,object) for key in NoahMPParameters._fields]+
    [('kdt',object),('frzx',object),('soiltype',object),('hydrology',object)])


def bind_hydrology(namelist,run_dir,domain,*,state):
    opts=namelist.noahmp_options
    if opts is None or (opts.opt_run not in (6,7,8) and opts.opt_tdrn==0):return namelist
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.noahmp_land_init import build_noahmp_params
    static=namelist.noahmp_static;old=static.parameters;shape=tuple(state.t_skin.shape)
    soil=np.asarray(getattr(old,'soiltype',np.broadcast_to(np.asarray(static.isltyp,np.int32),(4,*shape))),np.int32)
    # Driver's non-XICE water-category fallback applies all layers together.
    with Dataset(Gen2Run(run_dir).wrfinput_file(domain)) as ds:
        xice=np.asarray(ds['XICE'][0]) if 'XICE' in ds.variables else np.zeros(shape,np.float32)
        soil=np.where((np.any(soil==14,axis=0)&(xice==0.))[None,...],7,soil)
        if opts.opt_tdrn:
            key=next((k for k in ('TD_FRACTION','td_fraction') if k in ds.variables),None)
            if key is None:raise ValueError('Noah-MP tile drainage requires original TD_FRACTION map')
            var=ds[key];raw=np.ma.asarray(var[0] if var.dimensions[0]=='Time' else var[:])
            if raw.shape!=shape or np.ma.getmaskarray(raw).any() or not np.isfinite(raw).all() or np.any((raw<0.)|(raw>1.)):
                raise ValueError('Noah-MP TD_FRACTION requires finite surface fractions within [0,1]')
            fraction=jnp.asarray(np.asarray(raw,np.float32),jnp.float32)
        else:fraction=None
    table_dir=wrf_run_dir();runoff=None;tile=None
    if opts.opt_run in (6,7,8):
        lines=(table_dir/'SOILPARM.TBL').read_text().splitlines();count=int(lines[2].split(',')[0])
        values=np.asarray([[float(v) for v in line.split(',')[11:18]] for line in lines[3:3+count]],np.float32)
        if np.any((soil[0]<1)|(soil[0]>count)):raise ValueError('Noah-MP runoff surface soil category is invalid')
        runoff=RunoffParameters(*(jnp.asarray(values[soil[0]-1,k],jnp.float32) for k in range(7)))
        if any(not np.isfinite(np.asarray(v)).all() for v in runoff):raise ValueError('Noah-MP runoff parameters must be finite')
        if opts.opt_run in (6,8) and np.any(np.asarray(runoff.bvic if opts.opt_run==6 else runoff.bdvic)<=0.):
            raise ValueError('Noah-MP VIC requires a positive infiltration shape parameter on every modeled column')
    if opts.opt_tdrn:
        table=_parse_mptable_block((table_dir/'MPTABLE.TBL').read_text(),'noahmp_tiledrain_parameters')
        layer=int(table['DRAIN_LAYER_OPT'][0])
        if layer not in range(6):raise ValueError('WRF DRAIN_LAYER_OPT must be 0..5')
        opts=replace(opts,drain_layer_opt=layer)
        keys=('TDSMC_FAC','TD_DC','TD_DEPTH','TD_DCOEF','KLAT_FAC','TD_ADEPTH','TD_DDRAIN','TD_SPAC','TD_RADI')
        fields={k:jnp.asarray(np.asarray(table[k],np.float32)[soil[0]-1],jnp.float32) for k in keys}
        for k in keys:
            if not np.isfinite(np.asarray(fields[k])).all():raise ValueError(f'Noah-MP tile parameter {k} is nonfinite')
        if any(np.any(np.asarray(fields[k])<=0.) for k in ('TD_SPAC','TD_RADI')):
            raise ValueError('Noah-MP drain spacing/radius must be positive')
        if np.any(np.asarray(fields['TD_ADEPTH'])<np.asarray(fields['TD_DDRAIN'])):
            raise ValueError('Noah-MP impermeable layer must be below tile drain depth')
        depth=jax.jit(equivalent_depth)(fields['TD_ADEPTH']-fields['TD_DDRAIN'],fields['TD_SPAC'],fields['TD_RADI'])
        tile=TileParameters(*(fields[k] for k in keys),depth)
    hydro=HydrologyParameters(runoff,tile,fraction,jnp.full(shape,static.dx_m,jnp.float32) if opts.opt_tdrn else None)
    from gpuwrf.physics.noahmp.water_hydro import _soil_param
    template=namelist.noahmp_land.smois
    dk=_soil_param(old,'dksat',static,template,1.e-6)[0]
    smx=_soil_param(old,'smcmax',static,template,.45)[0];ref=_soil_param(old,'smcref',static,template,.35)[0]
    kdt=getattr(old,'kdt',jnp.asarray(old.refkdt,jnp.float32)*dk/jnp.asarray(old.refdk,jnp.float32))
    frzx=getattr(old,'frzx',jnp.asarray(old.frzk,jnp.float32)*(smx/ref)*(jnp.float32(.412)/jnp.float32(.468)))
    parameters=SelectedHydrologyParameters(*(getattr(old,k) for k in NoahMPParameters._fields),kdt,frzx,jnp.asarray(soil,jnp.int32),hydro)
    static=static.replace(parameters=parameters);ep,rp,nroot=build_noahmp_params(static)
    # Rebuilding ep must retain Liu's already selected crop photosynthesis rows.
    if opts.opt_crop:
        ep=ep._replace(**{key:getattr(namelist.noahmp_energy_params,key) for key in ('c3psn','kc25','akc','ko25','ako','avcmx','vcmx25','bp','mp','folnmx','qe25')})
    return replace(namelist,noahmp_options=opts,noahmp_static=static,noahmp_energy_params=ep,noahmp_rad_params=rp,noahmp_nroot=nroot)


def validate_tile_restart(group,namelist,version,shape):
    from collections.abc import Mapping
    opts=namelist.get('noahmp_options') if isinstance(namelist,Mapping) else getattr(namelist,'noahmp_options',None)
    active=opts is not None and opts.opt_tdrn!=0
    if active!=(group is not None):raise ValueError('Noah-MP tile drainage restart state is missing or disagrees with selection (E78)')
    if not active:return
    value=np.asarray(group)
    if version!=1 or value.shape!=tuple(shape) or value.dtype!=np.dtype(np.float32) or not np.isfinite(value).all():
        raise ValueError('Noah-MP tile drainage restart requires exact finite REAL32 QTDRAIN and schema1 (E78)')
