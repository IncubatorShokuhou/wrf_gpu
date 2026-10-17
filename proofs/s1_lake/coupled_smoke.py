"""CPU-only real Swiss case with two explicitly synthetic lake input tiles.

Uses the real CLI loader and full RK/physics steps, with the accepted CPU
Thomas shim. JIT is disabled for execution to avoid compiling the entire
legacy CPU model; this is a coupled finite/bypass/IO smoke, not a speed claim.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/v025'))
import cpu_guard  # noqa: E402,F401
sys.path.insert(0,str(REPO/'proofs/o1_nlbind'))
import cpu_safe  # noqa: E402
cpu_safe.install()
import jax
from jax import numpy as jnp
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om
from gpuwrf.physics import noahmp_coupler
from gpuwrf.coupling.lake_surface_hook import overlay_lake_history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',type=Path,default=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu'))
    p.add_argument('--steps',type=int,default=3)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    assert jax.devices()[0].platform=='cpu'
    state_contract._gpu_device=lambda:jax.devices('cpu')[0]
    case=args.out/'case'; case.mkdir(exist_ok=False)
    for f in args.case.iterdir():
        if f.name not in ('namelist.input','wrfinput_d01'):
            (case/f.name).symlink_to(f)
    shutil.copyfile(args.case/'wrfinput_d01',case/'wrfinput_d01')
    text=(args.case/'namelist.input').read_text()
    for key,value in {'sf_lake_physics':1,'use_lakedepth':1,'lakedepth_default':10.,'lake_min_elev':5.}.items():
        line=f' {key} = {value},'
        pattern=rf'^\s*{key}\s*=.*$'
        text=re.sub(pattern,line,text,flags=re.M|re.I) if re.search(pattern,text,re.M|re.I) else re.sub(r'(&physics\s*\n)',lambda m:m.group(1)+line+'\n',text,count=1,flags=re.I)
    (case/'namelist.input').write_text(text)
    with Dataset(case/'wrfinput_d01','a') as ds:
        shape=ds.variables['TSK'].shape[-2:]
        points=((shape[0]//2,shape[1]//2),(shape[0]//2+1,shape[1]//2))
        mask=np.zeros(shape,np.float32)
        for y,x in points:mask[y,x]=1.
        if 'LAKEMASK' not in ds.variables:ds.createVariable('LAKEMASK','f4',('Time','south_north','west_east'))
        ds.variables['LAKEMASK'][0]=mask
        if 'LAKE_DEPTH' not in ds.variables:ds.createVariable('LAKE_DEPTH','f4',('Time','south_north','west_east'))
        depth=np.full(shape,-1.,np.float32)
        depth[points[0]]=10.;depth[points[1]]=2.
        ds.variables['LAKE_DEPTH'][0]=depth
        for name,value in (('XLAND',2.),('LANDMASK',0.),('IVGTYP',int(getattr(ds,'ISWATER',17))),('LU_INDEX',int(getattr(ds,'ISWATER',17))),('ISLTYP',8)):
            if name in ds.variables:
                a=np.asarray(ds.variables[name][0]); a[mask==1.]=value;ds.variables[name][0]=a
        ds.LAKEFLAG=1;ds.LAKE_DEPTH_FLAG=1
    report={'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'backend':'cpu','scope':'Real Swiss meteorology and geometry; two synthetic lake masks/depth/category tiles; full CPU RK/physics smoke, no original-WRF coupled forecast or GPU claim',
            'modified_fields':['LAKEMASK','LAKE_DEPTH','XLAND','LANDMASK','IVGTYP','LU_INDEX','ISLTYP','LAKEFLAG','LAKE_DEPTH_FLAG'],
            'original_wrfinput_sha256':hashlib.sha256((args.case/'wrfinput_d01').read_bytes()).hexdigest(),
            'synthetic_wrfinput_sha256':hashlib.sha256((case/'wrfinput_d01').read_bytes()).hexdigest(),
            'points':points,'steps':[],'noah_calls':[]}
    def save():
        (args.out/'receipt.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    save();print('Loading real Swiss CLI case with two synthetic lake points',flush=True)
    start=time.perf_counter()
    cfg=pipeline.NestedPipelineConfig(case,args.out/'output',args.out/'proof',hours=1,max_dom=1)
    _,bundles,_,_,_,carries=pipeline._load_domains(cfg,('d01',))
    n=bundles['d01'].namelist;c=carries['d01']
    report['load_s']=time.perf_counter()-start
    assert len(n.lake_bundle.indices)==2 and c.lake is not None
    original_land=jax.device_get(c.noahmp_land)
    clock=om.build_clock_base(n)
    # Retain the loaded, fixed-input state for changed-code smoke retries.
    # This is a lane-owned restart, never a modification to the CPU reference.
    from gpuwrf.io.restart import write_restart
    write_restart(c,n,n.grid,0,args.out/'initial.pkl')
    real_kernel=noahmp_coupler.noah_mp_step
    def record_kernel(land,*a,**k):
        report['noah_calls'].append({'surface_shape':list(land.t_skin.shape),'soil_shape':list(land.tslb.shape)})
        return real_kernel(land,*a,**k)
    noahmp_coupler.noah_mp_step=record_kernel
    save();print('Loaded; executing full production RK/physics on CPU',flush=True)
    with jax.disable_jit():
        for step in range(1,args.steps+1):
            before=c.lake
            start=time.perf_counter()
            c=om._advance_chunk_fori(c,n,step,clock,n_steps=1,cadence=int(n.radiation_cadence_steps))
            jax.block_until_ready(c)
            bad=sum(int((~np.isfinite(np.asarray(v))).sum()) for v in jax.tree.leaves(c))
            row={'step':step,'wall_s':time.perf_counter()-start,'nonfinite':bad,
                 'lake_max_temperature_change':float(np.max(np.abs(np.asarray(c.lake.columns.t_lake3d)-np.asarray(before.columns.t_lake3d)))),
                 'lake_energy_residual_W_m2':np.asarray(c.lake.energy_residual).tolist(),
                 'lake_payload_dtypes':sorted({str(np.asarray(x).dtype) for x in jax.tree.leaves(c.lake)})}
            report['steps'].append(row);save();print(row,flush=True)
            assert bad==0
    assert all(call['surface_shape']==[1,int(np.prod(n.lake_bundle.shape))-2] for call in report['noah_calls'])
    report['noah_lake_payload_unchanged']=True
    for name in type(original_land).__slots__:
        a,b=np.asarray(getattr(original_land,name)),np.asarray(getattr(c.noahmp_land,name))
        if a.ndim>=2 and a.shape[-2:]==n.lake_bundle.shape:
            for y,x in points:
                assert np.array_equal(a[...,y,x],b[...,y,x]),name
    # The actual coupled carry must round-trip both native restart formats.
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    path=args.out/'coupled.pkl';write_restart(c,n,n.grid,args.steps,path)
    cp,_,_,_=read_restart(path)
    stamp=n.time_utc or datetime(2026,7,25,tzinfo=timezone.utc)
    pathnc=args.out/'coupled.nc'
    write_wrfrst_carry(c,n.grid,n,pathnc,run_start=stamp,valid_time=stamp+timedelta(seconds=args.steps*n.dt_s),step_index=args.steps)
    cn,_=read_wrfrst_carry(pathnc,namelist=n)
    for other in (cp,cn):
        for a,b in zip(jax.tree.leaves(c.lake),jax.tree.leaves(other.lake),strict=True):np.testing.assert_array_equal(a,b)
    report['restart_formats_exact']=['pickle','netcdf']
    assert report['steps'][-1]['lake_max_temperature_change']>0.
    report['status']='PASS';save();print('PASS full coupled CPU smoke',flush=True)


if __name__=='__main__':
    main()
