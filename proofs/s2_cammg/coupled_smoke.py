"""Real Swiss full CPU RK/physics continuation, CAM-MG or matched control.

Original WRF columns establish fidelity. This is a finite/ABI/restart smoke,
not a coupled CPU-WRF trajectory comparison or a GPU performance claim.
"""
import argparse
import ctypes
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'proofs/o1_nlbind')]
import cpu_safe
cpu_safe.install()
import jax
import numpy as np
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
from gpuwrf.runtime import operational_mode as om
from gpuwrf.io.restart import write_restart,read_restart


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--scratch',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mp',type=int,choices=(8,11),default=11);p.add_argument('--steps',type=int,default=2)
    a=p.parse_args();a.scratch.mkdir(parents=True,exist_ok=True)
    assert jax.devices()[0].platform=='cpu'
    case=a.scratch/'case';case.mkdir(exist_ok=True)
    for file in a.case.iterdir():
        if file.name!='namelist.input' and not (case/file.name).exists():
            (case/file.name).symlink_to(file.resolve())
    text,count=re.subn(r'(?im)^\s*mp_physics\s*=.*$',f' mp_physics = {a.mp},',
                       (a.case/'namelist.input').read_text());assert count==1
    (case/'namelist.input').write_text(text)
    report=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='Real Swiss full CPU RK/physics finite smoke; no original-WRF coupled/GPU claim',
        case=str(a.case),mp=a.mp,steps=[],GPU_execution=False,cache_dir='NONE',
        scratch=str(a.scratch),namelist_sha256=hashlib.sha256(text.encode()).hexdigest(),
        input_sha256=hashlib.sha256((a.case/'wrfinput_d01').read_bytes()).hexdigest(),
        fast_defaults=os.environ.get('GPUWRF_FAST_DEFAULTS','<default>'))
    def save():a.out.write_text(json.dumps(report,indent=2)+'\n')
    save();start=time.monotonic()
    # Proof-only CPU memory accounting. Clearing compiled primitive caches does
    # not change any operand, numerical function, or cadence. It prevents the
    # disable_jit interpreter retaining all earlier phase executables at once.
    def phase_event(name, edge):
        report.setdefault('phase_memory',[]).append(dict(name=name,edge=edge,
            elapsed_s=time.monotonic()-start,
            rss_KiB=int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])))
        save();print('PHASE',report['phase_memory'][-1],flush=True)
    def release_cpu_caches():
        jax.clear_caches();gc.collect();ctypes.CDLL(None).malloc_trim(0)
    pbl_current={}
    def instrument(module,name):
        original=getattr(module,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            release_cpu_caches();phase_event(name,'enter')
            result=original(*args,**kwargs);jax.block_until_ready(result)
            if name=='_physics_step_forcing' and a.mp==11:
                pbl_current.update({key:np.asarray(getattr(result.state,key)).copy()
                    for key in ('cam_kvh3d','cam_tke_pbl')})
            phase_event(name,'exit');release_cpu_caches()
            return result
        setattr(module,name,measured)
    for name in ('_refresh_rrtmg_driver','_refresh_noahmp_rad','noahmp_surface_step',
                 '_physics_step_forcing','_rk_scan_step','_apply_post_rk_microphysics'):
        instrument(om,name)
    from gpuwrf.coupling import physics_couplers
    for name in ('solve_rrtmg_sw_column','solve_rrtmg_lw_column'):
        instrument(physics_couplers,name)
    report['rrtmg_column_tile_cols']=os.environ.get('GPUWRF_RRTMG_COLUMN_TILE_COLS','default')
    config=NestedPipelineConfig(case,a.scratch/'output',a.scratch/'proof',hours=1,max_dom=1)
    _,bundles,_,_,_,carries=_load_domains(config,('d01',))
    n,c=bundles['d01'].namelist,carries['d01'];report['load_s']=time.monotonic()-start
    assert n.mp_physics==a.mp;om._resolve_operational_suite(n)
    report['work']={name:getattr(n,name) for name in ('mp_physics','dt_s','ra_sw_physics','ra_lw_physics',
        'bl_pbl_physics','sf_sfclay_physics','sf_surface_physics','cu_physics','shcu_physics',
        'accum_mode','aitken_mode','coarse_mode','radiation_cadence_steps','cumulus_cadence_steps','force_fp64')}
    report['microphysics_itimestep']=[]
    if a.mp==11:
        from gpuwrf.physics.cammg.registry import STATE_FIELDS
        report['selected_carries']={name:dict(shape=list(getattr(c.state,name).shape),
            dtype=str(getattr(c.state,name).dtype)) for name in STATE_FIELDS}
        from gpuwrf.physics.cammg import adapter
        original_adapter=adapter.cammg_adapter
        def observed(*args,**kw):
            it=int(np.asarray(kw['timestep']));report['microphysics_itimestep'].append(it)
            row=dict(step=it)
            for key,expected in pbl_current.items():
                got=np.asarray(getattr(args[0],key));assert got.tobytes()==expected.tobytes(),key
                row[key]=dict(same_current_PBL_call_BYTE=True,nonzero=int(np.count_nonzero(got)),max=float(got.max()))
            assert row['cam_kvh3d']['nonzero']>0
            report.setdefault('same_step_PBL_to_MP_handoff',[]).append(row)
            return original_adapter(*args,**kw)
        adapter.cammg_adapter=observed
        # CPU harness only: compile the same interpreted Pallas body while the
        # surrounding full model is in disable_jit mode. Otherwise every inner
        # vertical/adaptive loop is Python-interpreted separately per column.
        from gpuwrf.kernels import phys_cammg
        original_column=phys_cammg.full_column
        compiled=jax.jit(original_column,static_argnames=('cu','shcu','interpret','output_names'))
        def portable(*args,**kw):
            with jax.disable_jit(False):return compiled(*args,**kw)
        phys_cammg.full_column=portable
    release_cpu_caches();phase_event('load','released')
    clock=om.build_clock_base(n);save();print('LOADED',report['work'],flush=True)
    with jax.disable_jit():
        for step in range(1,a.steps+1):
            start=time.monotonic();before=c.state
            c=om._advance_chunk_fori(c,n,step,clock,n_steps=1,cadence=int(n.radiation_cadence_steps))
            jax.block_until_ready(c)
            bad=sum(int((~np.isfinite(np.asarray(v))).sum()) for v in jax.tree.leaves(c)
                    if np.issubdtype(np.asarray(v).dtype,np.floating))
            guards=getattr(getattr(c,'census',None),'guards',None)
            row=dict(step=step,wall_s=time.monotonic()-start,nonfinite=bad,
                fatal=0 if a.mp==8 else int(np.count_nonzero(np.asarray(c.state.cam_fatal))),
                guards=None if guards is None else np.asarray(guards).tolist(),
                qv_delta_max=float(np.max(np.abs(np.asarray(c.state.qv)-np.asarray(before.qv)))))
            report['steps'].append(row);save();print('STEP',row,flush=True)
            assert bad==0 and row['fatal']==0
    if a.mp==11:assert report['microphysics_itimestep']==list(range(1,a.steps+1))
    path=a.scratch/'restart.pkl';write_restart(c,n,n.grid,a.steps,path)
    resumed,*_=read_restart(path)
    assert all(np.asarray(x).tobytes()==np.asarray(y).tobytes()
               for x,y in zip(jax.tree.leaves(c),jax.tree.leaves(resumed),strict=True))
    path.unlink();report['restart_all_leaves_BYTE']=True
    report['maxrss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['status']='PASS';save()


if __name__=='__main__':main()
