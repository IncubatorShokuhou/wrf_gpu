"""WRF lakeini ingestion shared by both operational CLI pipelines."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset


def bind_lake_model(namelist,run_dir,domain,*,state):
    from gpuwrf.physics.lake_initialization import LakeInitConfig,LakeInitInputs,lake_initial_state
    from gpuwrf.physics.lake_step import LakeSurface
    from gpuwrf.coupling.lake_surface_hook import LakeBundle,LakeCarry
    import jax
    from jax import numpy as jnp
    if namelist.sf_surface_physics not in (None,4) or int(namelist.mp_physics) != 8 or int(namelist.bl_pbl_physics) != 5 or int(namelist.sf_sfclay_physics) != 5:
        raise ValueError('sf_lake_physics=1 currently requires Noah-MP, Thompson and MYNN/MYNN-SL')
    if not bool(namelist.use_noahmp):
        # The compatibility daily driver otherwise has State-only land replay.
        # An active lake opts into the same complete Noah-MP carry as native.
        from gpuwrf.io.noahmp_land_init import build_noahmp_land_state,build_noahmp_params
        land,static,_ = build_noahmp_land_state(run_dir,domain)
        ep,rp,nroot = build_noahmp_params(static)
        namelist = replace(namelist,use_noahmp=True,sf_surface_physics=4,
            noahmp_land=land,noahmp_static=static,noahmp_energy_params=ep,noahmp_rad_params=rp,noahmp_nroot=nroot)
    with Dataset(Path(run_dir)/f'wrfinput_{domain}','r') as ds:
        def read(name,default=None):
            if name not in ds.variables:
                if default is None:
                    raise ValueError(f'WRF lake init requires wrfinput {name}')
                return np.broadcast_to(np.float32(default),state.t_skin.shape).copy()
            var=ds.variables[name]
            out=np.asarray(var[0] if var.dimensions[0]=='Time' else var[:])
            if out.shape != state.t_skin.shape or not np.isfinite(out).all():
                raise ValueError(f'Invalid lake input {name}: shape/nonfinite')
            return out
        fields={name:read(name) for name in ('IVGTYP','ISLTYP','HGT','SNOW','TSK','XLAND','XLAT')}
        fields['LAKEMASK']=read('LAKEMASK',0.)
        fields['SEAICE']=read('SEAICE',0.)
        depth_present='LAKE_DEPTH' in ds.variables or 'lake_depth' in ds.variables
        fields['LAKE_DEPTH']=read('LAKE_DEPTH' if 'LAKE_DEPTH' in ds.variables else 'lake_depth',-1.)
        emiss=read('EMISS',.98)
        cfg=LakeInitConfig(iswater=int(getattr(ds,'ISWATER',17)),lakeflag=int(getattr(ds,'LAKEFLAG',1 if 'LAKEMASK' in ds.variables else 0)),
                           lake_depth_flag=int(getattr(ds,'LAKE_DEPTH_FLAG',int(depth_present))),use_lakedepth=int(namelist.use_lakedepth),
                           lakedepth_default=float(namelist.lakedepth_default),lake_min_elev=float(namelist.lake_min_elev),
                           xice_threshold=float(getattr(ds,'XICE_THRESHOLD',.5)))
        expected_lake=(fields['LAKEMASK']==1.) if cfg.lakeflag else ((fields['HGT']>=cfg.lake_min_elev)&((fields['IVGTYP']==cfg.iswater)|(fields['SEAICE']>cfg.xice_threshold)))
        if np.any(expected_lake&((fields['ISLTYP']<1)|(fields['ISLTYP']>19))):
            raise ValueError('Lake soil categories must be 1..19')
        inp=LakeInitInputs(*(jnp.asarray(fields[k],jnp.int32 if k in ('IVGTYP','ISLTYP') else jnp.float32)
                             for k in ('IVGTYP','ISLTYP','HGT','SNOW','TSK','LAKEMASK','SEAICE','XLAND','LAKE_DEPTH')))
        initialized=lake_initial_state(inp,config=cfg)
        mask=np.asarray(initialized.lake)
        indices=tuple(int(i) for i in np.flatnonzero(mask))
        nonlake=tuple(int(i) for i in np.flatnonzero(~mask))
        if np.any(mask&(np.asarray(initialized.xland)<1.5)):
            raise ValueError('Lake inputs must mark lake points XLAND=2, as WRF real.exe; prepare lake land-use inputs')
        if np.any(mask&((fields['ISLTYP']<1)|(fields['ISLTYP']>19))):
            raise ValueError('Lake soil categories must be 1..19')
        def compact(x):
            return jnp.asarray(x).reshape((mask.size,)+x.shape[2:])[jnp.asarray(indices,jnp.int32)]
        columns=jax.tree.map(compact,initialized.columns)
        surface=LakeSurface(*(compact(read(name,0. if name not in ('TSK','ALBEDO') else (290. if name=='TSK' else .08))) for name in
                              ('HFX','LH','GRDFLX','TSK','QFX','T2','TH2','Q2','ALBEDO')))
        initial=LakeCarry(columns,surface,jnp.zeros((len(indices),),jnp.float32),
                          jnp.asarray(np.stack((read('GRDFLX',0.),read('ALBEDO',.08))),jnp.float32))
        bundle=LakeBundle(tuple(mask.shape),indices,nonlake,initial,jax.device_get(initialized.columns),
                           compact(fields['XLAT']),jnp.asarray(emiss,jnp.float32),jnp.asarray(initialized.lakemask,jnp.float32),jnp.asarray(initialized.xland,jnp.float32))
    return replace(namelist,lake_bundle=bundle)
