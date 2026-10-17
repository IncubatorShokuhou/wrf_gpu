"""Opt-in layered/category/composition soil tables, bound once outside stepping."""
from dataclasses import replace
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.physics.noahmp.tables import NoahMPParameters,_parse_mptable_block
from gpuwrf.physics.noahmp.soil_options import pedotransfer_sr2006,OUTPUTS
from gpuwrf.io.netcdf_lock import Dataset


# The released table type and pytree stay unchanged. Only selected soil modes
# carry expanded layer arrays and TRANSFER's pre-pedotransfer KDT/FRZX.
SelectedSoilParameters=NamedTuple('SelectedSoilParameters',[(key,object) for key in NoahMPParameters._fields]+[('kdt',object),('frzx',object),('soiltype',object)])


def bind_soil(namelist,run_dir,domain,*,state):
    if namelist.noahmp_options is None or namelist.noahmp_options.opt_soil==1:return namelist
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.noahmp_land_init import build_noahmp_params
    static=namelist.noahmp_static;shape=tuple(state.t_skin.shape);old=static.parameters
    soil=np.broadcast_to(np.asarray(static.isltyp,np.int32),(4,*shape)).copy()
    with Dataset(Gen2Run(run_dir).wrfinput_file(domain)) as ds:
        def read(name,layers=None):
            key=next((k for k in (name,name.lower(),name.upper()) if k in ds.variables),None)
            if key is None:raise ValueError(f'Noah-MP opt_soil={namelist.noahmp_options.opt_soil} requires {name}')
            var=ds[key];raw=np.ma.asarray(var[0] if var.dimensions[0]=='Time' else var[:])
            expected=shape if layers is None else (layers,*shape)
            if raw.shape!=expected or np.ma.getmaskarray(raw).any() or not np.isfinite(raw).all():
                raise ValueError(f'Noah-MP {name} requires finite shape {expected}')
            return np.asarray(raw,np.float32)
        if namelist.noahmp_options.opt_soil==2:
            raw=np.stack([read('SOILCL'+str(k)) for k in range(1,5)])
            # Fortran NINT rounds positive halves up (not ties to even).
            soil=np.floor(raw+.5).astype(np.int32)
            if np.any((soil<1)|(soil>=len(old.bexp))):raise ValueError('Layered Noah-MP soil categories are outside SOILPARM.TBL')
        xice=read('XICE') if 'XICE' in ds.variables else np.zeros(shape,np.float32)
        soil=np.where((np.any(soil==14,axis=0)&(xice==0.))[None,...],7,soil)
        comp=read('SOILCOMP',8) if namelist.noahmp_options.opt_soil==3 else None
    fields=old._asdict()
    fields.update({key:jnp.asarray(np.asarray(getattr(old,key),np.float32)[soil],jnp.float32) for key in OUTPUTS})
    urban=(np.asarray(static.ivgtyp)==13)|((np.asarray(static.ivgtyp)>=51)&(np.asarray(static.ivgtyp)<=61))
    # TRANSFER:1693-1700 precedes KDT/FRZX and the composition override.
    for key,value in (('smcmax',.45),('smcref',.42),('smcwlt',.40),('smcdry',.40)):
        fields[key]=jnp.where(jnp.asarray(urban)[None,...],jnp.float32(value),fields[key])
    kdt=jnp.asarray(old.refkdt,jnp.float32)*fields['dksat'][0]/jnp.asarray(old.refdk,jnp.float32)
    frzx=jnp.asarray(old.frzk,jnp.float32)*(fields['smcmax'][0]/fields['smcref'][0])*(jnp.float32(.412)/jnp.float32(.468))
    # TRANSFER:1704 skips FRZX for soil14; mirror compiler-init zero there.
    frzx=jnp.where(jnp.asarray(soil[0])!=14,frzx,jnp.float32(0.))
    if comp is not None:
        if np.any((comp<0.)|(comp>100.)):raise ValueError('Noah-MP SOILCOMP percentages must lie within [0,100]')
        table=_parse_mptable_block((wrf_run_dir()/'MPTABLE.TBL').read_text(),'noahmp_optional_parameters')
        coefficients={k.lower():np.float32(v[0]) for k,v in table.items() if k.lower().startswith('sr2006_')}
        values=jax.jit(pedotransfer_sr2006)(jnp.float32(.01)*jnp.asarray(comp[:4]),jnp.float32(.01)*jnp.asarray(comp[4:]),jnp.zeros((4,*shape),jnp.float32),coefficients)
        for key,value in values.items():fields[key]=jnp.where(jnp.asarray(urban)[None,...],fields[key],value)
    parameters=SelectedSoilParameters(*(fields[key] for key in NoahMPParameters._fields),kdt,frzx,jnp.asarray(soil,jnp.int32))
    static=static.replace(parameters=parameters)
    ep,rp,nroot=build_noahmp_params(static)
    # NOAHMP_INIT:2082-2084 intentionally used surface ISLTYP BEFORE these
    # driver-time overrides; do not reconstruct initial SH2O with the new maps.
    return replace(namelist,noahmp_static=static,noahmp_energy_params=ep,noahmp_rad_params=rp,noahmp_nroot=nroot)
