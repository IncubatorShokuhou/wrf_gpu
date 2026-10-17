"""Dynamic Noah-MP carbon initialization and exact restart admission."""
from dataclasses import replace
import numpy as np
import jax.numpy as jnp


def bind_carbon(namelist,run_dir,domain,*,state):
    from gpuwrf.config.paths import wrf_run_dir
    from gpuwrf.io.land_history import load_land_history_inputs
    from gpuwrf.physics.noahmp.tables import _parse_mptable_block
    from gpuwrf.physics.noahmp.carbon import CarbonState,CarbonParameters,CarbonStatic
    from pathlib import Path
    if namelist.noahmp_options is not None and namelist.noahmp_options.opt_crop==1:
        from gpuwrf.io.noahmp_crop import bind_crop
        return bind_crop(namelist,run_dir,domain,state=state)
    if namelist.noahmp_options is None or namelist.noahmp_options.dveg not in (2,5,6):return namelist
    static=namelist.noahmp_static
    veg=np.asarray(static.ivgtyp,np.int32)
    if np.any((veg<1)|(veg>20)):raise ValueError("Dynamic Noah-MP vegetation currently requires the MODIS 20-category table")
    table_dir=wrf_run_dir()
    table=_parse_mptable_block((table_dir/"MPTABLE.TBL").read_text(),"noahmp_modis_parameters")
    params=CarbonParameters(*(jnp.asarray(np.asarray(table[key.upper()],np.float32)[veg-1],jnp.float32) for key in CarbonParameters._fields))
    inputs=load_land_history_inputs(Path(run_dir)/("wrfinput_"+domain),table_dir=table_dir,parameters=static.parameters)
    pools=CarbonState(*(jnp.asarray(inputs[key.upper()],jnp.float32) for key in CarbonState._fields))
    land=namelist.noahmp_land.replace(lai=jnp.asarray(inputs["LAI_INIT"],jnp.float32),sai=jnp.asarray(inputs["XSAI_INIT"],jnp.float32))
    validate_restart(pools,namelist,1,state.t_skin.shape)
    return replace(namelist,noahmp_carbon_static=CarbonStatic(params,pools),noahmp_land=land)


def validate_restart(group,namelist,version,shape):
    from gpuwrf.physics.noahmp.carbon import CarbonState
    from collections.abc import Mapping
    opts=namelist.get("noahmp_options") if isinstance(namelist,Mapping) else getattr(namelist,"noahmp_options",None)
    crop=opts is not None and opts.opt_crop==1
    active=opts is not None and (opts.dveg in (2,5,6) or crop)
    if active!=(group is not None):raise ValueError("Noah-MP carbon restart is missing or disagrees with selection (E78)")
    if not active:return
    if crop:
        from gpuwrf.physics.noahmp.crop import CropCarbonState
        expected=CropCarbonState
    else:expected=CarbonState
    if version!=1 or not isinstance(group,expected):raise ValueError("Noah-MP carbon restart schema is missing/incompatible (E78)")
    for name,value in group._asdict().items():
        array=np.asarray(value)
        dtype=np.int32 if name=='pgs' else np.float32
        if array.shape!=tuple(shape) or array.dtype!=np.dtype(dtype) or not np.isfinite(array).all():
            raise ValueError("Noah-MP carbon restart needs finite REAL32 surface pools and INTEGER PGS (E78)")
        if name=='pgs' and np.any((array<0)|(array>8)):raise ValueError('Noah-MP carbon restart PGS must be in 0..8 (E78)')
