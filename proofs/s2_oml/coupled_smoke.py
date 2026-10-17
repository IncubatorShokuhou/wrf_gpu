"""Real Swiss CPU RK/physics, two explicitly synthetic open-ocean tiles.

This is coupled finite/carry/IO smoke, not a CPU-WRF coupled forecast or speed
claim. Real geometry/meteorology with synthetic water category/TSK at two cells.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/v025'),str(REPO/'proofs/o1_nlbind')]
import cpu_guard
import cpu_safe
cpu_safe.install()
import jax
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',type=Path,default=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu'))
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--control',action='store_true',help='same modified inputs, original release OML0/isftcflx0')
    p.add_argument('--start-step',type=int,default=1,help='absolute step; fresh initialized states must begin at 1')
    p.add_argument('--resume-initial',type=Path,help='reuse an exact initialized checkpoint for the matched control')
    args=p.parse_args()
    if args.resume_initial is None and args.start_step!=1:
        raise ValueError('Fresh states require step1 WRF surface bootstrap; later steps require a matching checkpoint')
    assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
    assert jax.devices()[0].platform=='cpu'
    state_contract._gpu_device=lambda:jax.devices('cpu')[0]
    args.out.mkdir(parents=True,exist_ok=True)
    case=args.out/'case';case.mkdir(exist_ok=False)
    for f in args.case.iterdir():
        if f.name not in ('namelist.input','wrfinput_d01'):
            (case/f.name).symlink_to(f)
    shutil.copyfile(args.case/'wrfinput_d01',case/'wrfinput_d01')
    text=(args.case/'namelist.input').read_text()
    for key,value in {'sf_ocean_physics':0 if args.control else 1,'oml_hml0':20.,'oml_gamma':.14,'oml_relaxation_time':3600.,'isftcflx':0 if args.control else 2,'sst_update':0}.items():
        line=f' {key} = {value},'
        pattern=rf'^\s*{key}\s*=.*$'
        text=re.sub(pattern,line,text,flags=re.M|re.I) if re.search(pattern,text,re.M|re.I) else re.sub(r'(&physics\s*\n)',lambda m:m.group(1)+line+'\n',text,count=1,flags=re.I)
    (case/'namelist.input').write_text(text)
    with Dataset(case/'wrfinput_d01','a') as ds:
        shape=ds['TSK'].shape[-2:]
        points=((shape[0]//2,shape[1]//2),(shape[0]//2+1,shape[1]//2))
        for name,value in (('XLAND',2.),('LANDMASK',0.),('IVGTYP',int(getattr(ds,'ISWATER',17))),
                           ('LU_INDEX',int(getattr(ds,'ISWATER',17))),('ISLTYP',8),('TSK',302.),('SST',302.)):
            if name in ds.variables:
                a=np.asarray(ds[name][0]);
                for y,x in points:a[y,x]=value
                ds[name][0]=a
    report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            'backend':'cpu','scope':'real Swiss geometry/meteorology, two synthetic ocean-category/TSK tiles; full RK/physics smoke; no CPU-WRF forecast or GPU speed claim',
            'source_wrfinput_sha256':hashlib.sha256((args.case/'wrfinput_d01').read_bytes()).hexdigest(),
            'input_sha256':hashlib.sha256((case/'wrfinput_d01').read_bytes()).hexdigest(),'points':points,'steps':[],
            'control':args.control,'start_step':args.start_step,'fast_defaults':os.environ.get('GPUWRF_FAST_DEFAULTS','1'),
            'guard_observations':[],'f2_observations':[],'physics_phases':[]}
    save=lambda:(args.out/'receipt.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    save();start=time.perf_counter();print('Loading Swiss OML/hurricane CLI case',flush=True)
    cfg=pipeline.NestedPipelineConfig(case,args.out/'output',args.out/'proof',hours=1,max_dom=1)
    if args.resume_initial is not None:
        from gpuwrf.io.restart import read_restart
        c,n,_,checkpoint_step=read_restart(args.resume_initial)
        if checkpoint_step!=args.start_step-1:
            raise ValueError('Checkpoint step does not precede requested start-step')
        report['initial_checkpoint_sha256']=hashlib.sha256(args.resume_initial.read_bytes()).hexdigest()
        if args.control:
            n=replace(n,sf_ocean_physics=0,isftcflx=0,ocean_initial=None)
            c=c.replace(ocean=None,hurricane_ustm=None)
        om._resolve_operational_suite(n)
    else:
        _,bundles,_,_,_,carries=pipeline._load_domains(cfg,('d01',))
        n=bundles['d01'].namelist;c=carries['d01']
    assert n.sf_ocean_physics==(0 if args.control else 1) and n.isftcflx==(0 if args.control else 2) and n.oml_hml0==20. and n.oml_relaxation_time==3600.
    assert (c.ocean is not None)==(not args.control) and (c.hurricane_ustm is not None)==(not args.control)
    report['load_s']=time.perf_counter()-start
    report['carry_bytes']=sum(int(x.size*x.dtype.itemsize) for x in jax.tree.leaves((c.ocean,c.hurricane_ustm)))
    report['strict_no_restore']=bool(n.disable_guards)
    clock=om.build_clock_base(n)
    save();print('Loaded; full production CPU RK/physics',flush=True)
    from gpuwrf.io.restart import write_restart,read_restart
    if args.resume_initial is None:
        write_restart(c,n,n.grid,0,args.out/'initial.pkl')
    # Eager CPU-only observer: preserve census=None and every model return.
    # Reads happen at the original pre-mask call sites; no model carry, JIT
    # reduction or sanitizer is added. This is diagnostic smoke, not timing.
    import gpuwrf.diagnostics.census as census_module
    def events(state,fields,phase,bounds=None):
        rows=[]
        for field in fields:
            raw=getattr(state,field,None)
            if raw is None:continue
            value=np.asarray(raw)
            finite=np.isfinite(value)
            outside=np.zeros_like(finite)
            if bounds is not None and field in bounds:
                lower,upper=bounds[field]
                if lower is not None:outside|=value<lower
                if upper is not None:outside|=value>upper
            mask=(~finite)|(outside&finite)
            indices=np.argwhere(mask)
            ring0=0
            if value.ndim>=2 and len(indices):
                ring0=int(np.sum((indices[:,-1]==0)|(indices[:,-1]==value.shape[-1]-1)|(indices[:,-2]==0)|(indices[:,-2]==value.shape[-2]-1)))
            rows.append({'phase':phase,'field':field,'nonfinite':int((~finite).sum()),
                         'out_of_range':int((outside&finite).sum()),'ring0':ring0,'interior':int(mask.sum())-ring0})
        return rows
    original_dyn=census_module.count_dynamics_guards
    def dynamics_observer(counter,state,*,theta_min,theta_max):
        bounds={field:(0.,.05) for field in census_module.MOISTURE_FIELDS}
        bounds.update(theta=(theta_min,theta_max),mu_total=(1.,None))
        report['guard_observations'].extend(events(state,('theta','mu_total',*census_module.MOISTURE_FIELDS),'dynamics',bounds))
        save()
        return original_dyn(counter,state,theta_min=theta_min,theta_max=theta_max)
    original_bdy=census_module.count_boundary_guards
    def boundary_observer(counter,state):
        report['guard_observations'].extend(events(state,(*census_module.BOUNDARY_FINITE_FIELDS,'qv','mu_total','mu_perturbation'),
            'boundary',{'qv':(0.,.05),'mu_total':(1.,None)}))
        save()
        return original_bdy(counter,state)
    original_f2=census_module.observe_f2_state
    def f2_observer(counter,phase,state,step_index,*a,**kw):
        rows=events(state,census_module.F2_FIELDS,phase,{field:(0.,None) for field in census_module.F2_FIELDS})
        for row in rows:row['step']=int(step_index)
        report['f2_observations'].extend(rows);save()
        return original_f2(counter,phase,state,step_index,*a,**kw)
    census_module.count_dynamics_guards=dynamics_observer
    census_module.count_boundary_guards=boundary_observer
    census_module.observe_f2_state=f2_observer
    original_forcing=om._physics_step_forcing
    def forcing_observer(*a,**kw):
        start=time.perf_counter();print('Physics forcing begin',flush=True)
        out=original_forcing(*a,**kw)
        report['physics_phases'].append({'phase':'forcing','wall_s':time.perf_counter()-start})
        save();print('Physics forcing end',report['physics_phases'][-1],flush=True)
        return out
    om._physics_step_forcing=forcing_observer
    def timed_phase(module,name):
        if not hasattr(module,name):return
        original=getattr(module,name)
        def call(*a,**kw):
            start=time.perf_counter();print(name,'begin',flush=True)
            value=original(*a,**kw)
            jax.block_until_ready(value)
            row={'phase':name,'wall_s':time.perf_counter()-start}
            report['physics_phases'].append(row);save();print(name,'end',row,flush=True)
            return value
        setattr(module,name,call)
    for name in ('_refresh_rrtmg_driver','noahmp_surface_step','mynn_adapter','mynn_adapter_with_source_leaves'):
        timed_phase(om,name)
    import gpuwrf.coupling.hurricane_surface_hook as hurricane
    timed_phase(hurricane,'surface_step')
    with jax.disable_jit():
        for step in range(args.start_step,args.start_step+args.steps):
            old=c.ocean;start=time.perf_counter()
            c=om._advance_chunk_fori(c,n,step,clock,n_steps=1,cadence=int(n.radiation_cadence_steps))
            jax.block_until_ready(c)
            bad=sum(int((~np.isfinite(np.asarray(v))).sum()) for v in jax.tree.leaves(c))
            water=np.asarray(c.state.xland)>1.5
            if not args.control:np.testing.assert_array_equal(np.asarray(c.state.t_skin)[water],np.asarray(c.ocean.tml)[water])
            row={'step':step,'wall_s':time.perf_counter()-start,'nonfinite':bad,
                 'ocean_TML_max_change_K':None if args.control else float(np.max(np.abs(np.asarray(c.ocean.tml)-np.asarray(old.tml)))),
                 'ocean_HML_max_change_m':None if args.control else float(np.max(np.abs(np.asarray(c.ocean.hml)-np.asarray(old.hml)))),
                 'ustm_max':None if args.control else float(np.max(np.asarray(c.hurricane_ustm))),
                 'dtypes':sorted({str(x.dtype) for x in jax.tree.leaves((c.ocean,c.hurricane_ustm))})}
            report['steps'].append(row);save();print(row,flush=True)
            assert bad==0 and row['dtypes']==([] if args.control else ['float32'])
    path=args.out/'restart.pkl';write_restart(c,n,n.grid,args.start_step+args.steps-1,path)
    restored,_,_,_=read_restart(path)
    for a,b in zip(jax.tree.leaves((c.ocean,c.hurricane_ustm)),jax.tree.leaves((restored.ocean,restored.hurricane_ustm)),strict=True):np.testing.assert_array_equal(a,b)
    report['restart_byte_exact']=True;save()


if __name__=='__main__':main()
