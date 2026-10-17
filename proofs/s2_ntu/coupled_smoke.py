"""Real Swiss CLI input, selected NTU, full CPU RK/physics continuation.

CPU smoke only; original WRF column gates establish numerical fidelity.
No meteorological input is modified and no reference values are overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import ctypes
import functools
import gc
import sys
import time
from cpu_executor import CPUExecutor
from f2_trace import F2Trace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'proofs/o1_nlbind')]
import cpu_safe
cpu_safe.install()
import jax
import numpy as np
from gpuwrf.contracts.state import conditional_state_leaves_for_mp
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
from gpuwrf.runtime import operational_mode as om
from gpuwrf.io.restart import write_restart,read_restart
SCHEMES=(8,56)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mp',type=int,choices=SCHEMES,required=True)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--f2-operands',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert jax.devices()[0].platform=='cpu'
    case=a.out/'case';case.mkdir(exist_ok=True)
    for file in a.case.iterdir():
        if file.name!='namelist.input' and not (case/file.name).exists():
            (case/file.name).symlink_to(file.resolve())
    text=(a.case/'namelist.input').read_text()
    import re
    text,count=re.subn(r'(?im)^\s*mp_physics\s*=.*$',f' mp_physics = {a.mp},',text)
    assert count==1
    (case/'namelist.input').write_text(text)
    report={'source':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'Real Swiss full CPU RK/physics smoke; no CPU-WRF coupled trajectory or GPU claim',
        'case':str(a.case),'mp':a.mp,'steps':[],'GPU_execution':False,
        'namelist_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'source_input_sha256':hashlib.sha256((a.case/'wrfinput_d01').read_bytes()).hexdigest(),
        'fast_defaults':os.environ.get('GPUWRF_FAST_DEFAULTS','<default>')}
    def save():
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Release CPU interpreter executables between phases, in both mp8 and mp56.
    # No numerical expression, meteorological input, or cadence changes.
    def release_cpu_caches():
        jax.clear_caches();gc.collect();ctypes.CDLL(None).malloc_trim(0)
    def instrument(module,name):
        original=getattr(module,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            release_cpu_caches();print('PHASE',name,'enter',flush=True)
            result=original(*args,**kwargs);jax.block_until_ready(result)
            report.setdefault('phase_memory',[]).append(dict(name=name,
                rss_KiB=int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])))
            save();print('PHASE',name,'exit',report['phase_memory'][-1],flush=True)
            release_cpu_caches()
            return result
        setattr(module,name,measured)
    for name in ('_refresh_rrtmg_driver','_refresh_noahmp_rad','noahmp_surface_step',
                 '_physics_step_forcing','_rk_scan_step','_apply_post_rk_microphysics'):
        instrument(om,name)
    from gpuwrf.coupling import physics_couplers
    for name in ('solve_rrtmg_sw_column','solve_rrtmg_lw_column'):
        instrument(physics_couplers,name)
    report['rrtmg_column_tile_cols']=os.environ.get('GPUWRF_RRTMG_COLUMN_TILE_COLS','default')
    # The M9 caller explicitly supplies 512, overriding the solver environment.
    # Use its existing column tiling control for this CPU-only memory cap.
    om._M9_RRTMG_COLUMN_TILE_COLS=int(os.environ.get('GPUWRF_RRTMG_COLUMN_TILE_COLS','128'))
    report['cpu_m9_rrtmg_column_tile_cols']=om._M9_RRTMG_COLUMN_TILE_COLS
    report['NTU_execution']='NumPy shared column equations; other CPU physics uses JAX interpreter'
    cfg=NestedPipelineConfig(case,a.out/'output',a.out/'proof',hours=1,max_dom=1)
    t=time.monotonic();_,bundles,_,_,_,carries=_load_domains(cfg,('d01',))
    n,c=bundles['d01'].namelist,carries['d01'];report['load_s']=time.monotonic()-t
    assert n.mp_physics==a.mp
    om._resolve_operational_suite(n)
    names=conditional_state_leaves_for_mp(a.mp)
    # New NTU-owned storage is REAL4 even in the FAST0 CPU reference path.
    # Existing shared qh/Nh/hail follow that path's legacy wider State policy.
    assert all(getattr(c.state,name).dtype==np.float32 for name in names if name.startswith('ntu_'))
    report['selected_carries']={name:dict(shape=list(getattr(c.state,name).shape),
        dtype=str(getattr(c.state,name).dtype)) for name in names}
    report['work']={name:getattr(n,name) for name in ('mp_physics','ccnty','dt_s','ra_sw_physics','ra_lw_physics',
        'bl_pbl_physics','sf_sfclay_physics','sf_surface_physics','cu_physics','use_mp_re',
        'radiation_cadence_steps','cumulus_cadence_steps','force_fp64')}
    clock=om.build_clock_base(n);save();print('LOADED',report['load_s'],report['work'],flush=True)
    release_cpu_caches()
    report['cache_dir']=os.environ.get('JAX_COMPILATION_CACHE_DIR')
    from contextlib import nullcontext
    trace=F2Trace(om,a.out/'f2_operands') if a.f2_operands else None
    with CPUExecutor() as ntu_cpu, jax.disable_jit(), (trace or nullcontext()):
        for it in range(1,a.steps+1):
            if trace is not None:trace.step=it
            before=c.state;t=time.monotonic()
            c=om._advance_chunk_fori(c,n,it,clock,n_steps=1,cadence=int(n.radiation_cadence_steps))
            jax.block_until_ready(c)
            bad=sum(int((~np.isfinite(np.asarray(x))).sum()) for x in jax.tree.leaves(c)
                    if np.issubdtype(np.asarray(x).dtype,np.floating))
            guards=getattr(getattr(c,'census',None),'guards',None)
            row={'step':it,'wall_s':time.monotonic()-t,'nonfinite':bad,
                'guards':None if guards is None else np.asarray(guards).tolist(),
                'particle_active':{name:int(np.count_nonzero(np.asarray(getattr(c.state,name)))) for name in names},
                'qv_delta_max':float(np.max(np.abs(np.asarray(c.state.qv)-np.asarray(before.qv))))}
            census=getattr(c,'census',None)
            row['f2']={name:np.asarray(value).tolist() for name in
                ('f2_counts','f2_first_step','f2_first_index','f2_first_value',
                 'f2_min_negative','f2_min_step','f2_min_index',
                 'f2_negative_location','f2_field_observations')
                if (value:=getattr(census,name,None)) is not None}
            report['steps'].append(row);save();print(row,flush=True)
            assert bad==0
    report['NTU_CPU_reference_execution']=ntu_cpu.receipts
    path=a.out/'continuation.pkl';write_restart(c,n,n.grid,a.steps,path)
    resumed,*_=read_restart(path)
    assert all(np.asarray(x).tobytes()==np.asarray(y).tobytes()
               for x,y in zip(jax.tree.leaves(c),jax.tree.leaves(resumed),strict=True))
    report['restart_all_leaves_BYTE']=True;path.unlink()
    report['maxrss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['status']='PASS';save()


if __name__=='__main__':main()
