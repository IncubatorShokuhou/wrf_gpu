"""Real Swiss CLI input, retained P3/Jensen diagnostics, full CPU continuation.

CPU smoke only; original WRF column gates establish numerical fidelity.
No meteorological input is modified and no reference values are overwritten.
"""
import argparse
import ctypes
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

# Manager 23:2xZ: numerical + actual Swiss two-step checks now in <=6G slots;
# only the five standard v013 singleton nodes wait for the 12G window.
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
SCHEMES=(50,51,52,53,55)
from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_leaves


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--mp',type=int,choices=SCHEMES,required=True)
    p.add_argument('--steps',type=int,default=2)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert jax.devices()[0].platform=='cpu'
    case=a.out/'case';case.mkdir(exist_ok=False)  # Fresh private path: never follow a prior namelist symlink.
    for file in a.case.iterdir():
        if file.name!='namelist.input' and not (case/file.name).exists():
            (case/file.name).symlink_to(file.resolve())
    text=(a.case/'namelist.input').read_text()
    import re
    text,count=re.subn(r'(?im)^\s*mp_physics\s*=.*$',f' mp_physics = {a.mp},',text)
    assert count==1
    assert not (case/'namelist.input').is_symlink()
    (case/'namelist.input').write_text(text)
    report={'source':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'Real Swiss full CPU RK/physics smoke; no CPU-WRF coupled trajectory or GPU claim',
        'case':str(a.case),'mp':a.mp,'steps':[],'GPU_execution':False,
        'namelist_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'source_input_sha256':hashlib.sha256((a.case/'wrfinput_d01').read_bytes()).hexdigest(),
        'fast_defaults':os.environ.get('GPUWRF_FAST_DEFAULTS','<default>')}
    def save():
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Same proven proof-only cache release as s2-cammg/coupled_smoke.py.
    # No operands, model code, cadence or arithmetic are changed.
    start=time.monotonic()
    def release_cpu_caches():
        jax.clear_caches();gc.collect();ctypes.CDLL(None).malloc_trim(0)
    def instrument(module,name):
        original=getattr(module,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            release_cpu_caches()
            result=original(*args,**kwargs);jax.block_until_ready(result)
            report.setdefault('phase_memory',[]).append(dict(name=name,
                elapsed_s=time.monotonic()-start,
                rss_KiB=int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0])))
            save();print('PHASE',report['phase_memory'][-1],flush=True)
            release_cpu_caches()
            return result
        setattr(module,name,measured)
    for name in ('_refresh_rrtmg_driver','_refresh_noahmp_rad','noahmp_surface_step',
                 '_physics_step_forcing','_rk_scan_step','_apply_post_rk_microphysics'):
        instrument(om,name)
    from gpuwrf.coupling import physics_couplers
    for name in ('solve_rrtmg_sw_column','solve_rrtmg_lw_column'):
        instrument(physics_couplers,name)
    cfg=NestedPipelineConfig(case,a.out/'output',a.out/'proof',hours=1,max_dom=1)
    t=time.monotonic();_,bundles,_,_,_,carries=_load_domains(cfg,('d01',))
    n,c=bundles['d01'].namelist,carries['d01'];report['load_s']=time.monotonic()-t
    assert n.mp_physics==a.mp
    om._resolve_operational_suite(n)
    names=conditional_state_leaves_for_mp(a.mp)
    assert all(getattr(c.state,name).dtype==np.float32 for name in names)
    report['selected_carries']={name:list(getattr(c.state,name).shape) for name in names}
    report['work']={name:getattr(n,name) for name in ('mp_physics','dt_s','ra_sw_physics','ra_lw_physics',
        'bl_pbl_physics','sf_sfclay_physics','sf_surface_physics','cu_physics','use_mp_re',
        'radiation_cadence_steps','cumulus_cadence_steps','force_fp64')}
    clock=om.build_clock_base(n);save();print('LOADED',report['load_s'],report['work'],flush=True)
    if a.mp==55:
        from gpuwrf.physics.jensen_ishmael import adapter as producer
        adapter_name='jensen_adapter'
    else:
        from gpuwrf.physics.p3 import adapter as producer
        adapter_name='p3_adapter'
    original_adapter=getattr(producer,adapter_name)
    report['microphysics_itimestep']=[]
    produced={}
    def observed_adapter(*args,**kwargs):
        report['microphysics_itimestep'].append(int(np.asarray(kwargs.get('timestep',1))))
        result=original_adapter(*args,**kwargs)
        updated=result[0] if isinstance(result,tuple) else result
        produced.update({name:np.asarray(getattr(updated,name)).copy() for name in ice_diagnostic_leaves(a.mp)})
        return result
    setattr(producer,adapter_name,observed_adapter)
    report['cache_dir']=os.environ.get('JAX_COMPILATION_CACHE_DIR')
    release_cpu_caches()
    with jax.disable_jit():
        for it in range(1,a.steps+1):
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
            row['ice_diagnostics']={name:dict(min=float(np.asarray(getattr(c.state,name)).min()),
                max=float(np.asarray(getattr(c.state,name)).max()),dtype=str(getattr(c.state,name).dtype))
                for name in ice_diagnostic_leaves(a.mp)}
            # Specified-zone masking can retain the old boundary. Interior
            # values must be the producer values from THIS own step.
            zone=int(n.boundary_config.spec_zone)
            interior=(slice(None),slice(zone,-zone or None),slice(zone,-zone or None))
            row['producer_to_carry_interior_BYTE']={name:
                np.asarray(getattr(c.state,name))[interior].tobytes()==value[interior].tobytes()
                for name,value in produced.items()}
            assert all(row['producer_to_carry_interior_BYTE'].values())
            report['steps'].append(row);save();print(row,flush=True)
            assert bad==0
            release_cpu_caches()
    assert report['microphysics_itimestep']==list(range(1,a.steps+1)),report['microphysics_itimestep']
    setattr(producer,adapter_name,original_adapter)
    path=a.out/'continuation.pkl';write_restart(c,n,n.grid,a.steps,path)
    resumed,*_=read_restart(path)
    assert all(np.asarray(x).tobytes()==np.asarray(y).tobytes()
               for x,y in zip(jax.tree.leaves(c),jax.tree.leaves(resumed),strict=True))
    report['restart_all_leaves_BYTE']=True;path.unlink()
    report['maxrss_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['status']='PASS' if all(row['guards'] is not None and not np.any(np.asarray(row['guards']))
                                 for row in report['steps']) else 'GUARD_REVIEW_REQUIRED'
    save()


if __name__=='__main__':main()
