"""CPU interpretation or CUDA-target Triton lowering; never initializes CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import resource
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.kernels.phys_ntu import full_column
from gpuwrf.physics.ntu.column import PROGNOSTICS,SURFACE


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--mode',choices=('interpret','lower'),default='lower');p.add_argument('--case',type=int,default=0)
    p.add_argument('--levels',type=int,default=12,help='Lower-only shape preflight; interpreted oracle uses original12')
    p.add_argument('--operational',action='store_true',help='Exactly the production adapter output projection')
    p.add_argument('--cpu-opt-level',type=int,choices=(0,1,2,3),default=0);a=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and jax.devices()[0].platform=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    fixtures=Path(__file__).with_name('fixtures');data=np.load(fixtures/'columns_r4.npz')
    def inputs(case):
        f={n:data[f'c{case:02}_s0_{n}'][None,:] for n in ['th']+PROGNOSTICS}
        f.update({n:data[f'c{case:02}_s0_{n}'] for n in SURFACE})
        f.update({n:data[f'c{case:02}_input_{n}'][None,:] for n in ('p','dz','pii','w')})
        if a.levels!=12:
            assert a.mode=='lower','Shape resampling is not oracle evidence'
            for n in ['th']+PROGNOSTICS+['p','dz','pii','w']:
                old=f[n][0];f[n]=np.interp(np.linspace(0,len(old)-1,a.levels+(n=='w')),np.arange(len(old)),old).astype(np.float32)[None,:]
        return f
    fields=inputs(a.case or 8)
    # An ambient-f32 process is intentional: only original DOUBLE work may widen.
    jax.config.update('jax_enable_x64',False)
    start=time.perf_counter()
    from gpuwrf.physics.ntu.adapter import OUTPUT_NAMES
    call=lambda f:full_column(f,np.float32(18),interpret=a.mode=='interpret',output_names=OUTPUT_NAMES if a.operational else None)
    if a.mode=='interpret':
        # Pinned Pallas CPU discharge allocates its own f64 padding while
        # lowering, after the kernel wrapper's trace scope has ended. Keep
        # that validation-only lowering inside the original-DOUBLE scope too.
        with jax.enable_x64(True):
            lowered=jax.jit(call).lower(fields)
            compiler_options={'xla_backend_optimization_level':a.cpu_opt_level}
            compile_start=time.perf_counter();compiled=lowered.compile(compiler_options=compiler_options);compile_s=time.perf_counter()-compile_start
            all_outputs={};scores=[];execute_start=time.perf_counter()
            for case in ([a.case] if a.case else range(1,13)):
                fields=inputs(case)
                for stage in (1,2):
                    outputs=compiled(fields);jax.block_until_ready(outputs)
                    counts={n:int(np.count_nonzero(~np.isfinite(v))) for n,v in outputs.items()}
                    all_outputs.update({f'c{case:02}_s{stage}_{n}':np.asarray(v) for n,v in outputs.items()})
                    scores.append(dict(case=case,stage=stage,nonfinite=counts))
                    fields.update(outputs)
                    print('CASE',case,'STAGE',stage,'NONFINITE',sum(counts.values()),flush=True)
            execute_s=time.perf_counter()-execute_start
        memory=compiled.memory_analysis()
        nonfinite={n:int(sum(row['nonfinite'][n] for row in scores)) for n in outputs}
        np.savez_compressed(a.out.with_suffix('.npz'),**all_outputs)
        result={'mode':a.mode,'backend':'cpu','case':a.case,'elapsed_s':time.perf_counter()-start,'nonfinite':nonfinite,
                'storage_dtypes':{n:str(v.dtype) for n,v in outputs.items()},'ambient_x64_restored':not jax.config.x64_enabled,
                'scores':scores,'compile_s':compile_s,'execute_s':execute_s,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'compiler_options':compiler_options,
                'compiled_memory_bytes':{k:getattr(memory,k) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}}
    else:
        from jax._src.pallas.triton import lowering
        program=jax.make_jaxpr(call)(fields).jaxpr
        params=next(e.params for e in program.eqns if e.primitive.name=='pallas_call')
        def equations(v):
            if hasattr(v,'jaxpr'):yield from equations(v.jaxpr)
            elif hasattr(v,'eqns'):
                for e in v.eqns:
                    yield e
                    for child in e.params.values():yield from equations(child)
            elif isinstance(v,(tuple,list)):
                for child in v:yield from equations(child)
            elif isinstance(v,dict):
                for child in v.values():yield from equations(child)
        wide=[e for e in equations(params['jaxpr']) if any(getattr(getattr(v,'aval',None),'dtype',None)==np.dtype('float64') for v in e.outvars)]
        from collections import Counter
        import re
        origins=Counter(); outside=[]; controls=0
        def audit(value,parent=()):
            nonlocal controls
            if hasattr(value,'jaxpr'):audit(value.jaxpr,parent)
            elif hasattr(value,'eqns'):
                for e in value.eqns:
                    project=tuple(s.split('/src/')[-1] for s in str(e.source_info.traceback).splitlines() if '/src/gpuwrf/' in s)
                    project=project or parent
                    iswide=any(getattr(getattr(v,'aval',None),'dtype',None)==np.dtype('float64') for v in e.outvars)
                    container=e.primitive.name in ('while','scan','cond','jit','pjit')
                    if iswide and container:controls+=1
                    if iswide and not container:
                        origin=project[0] if project else '<unattributed>'
                        origins[origin]+=1
                        allowed=(re.search(r'ntu/math.py:(\d+) \(([^)]*)\)',origin)
                                 and any('('+name in origin for name in ('gamln','dlnx','dlnx2','dpdf','find_rc0')))
                        allowed=allowed or 'aerosol_generated.py:' in origin and '(activa' in origin
                        allowed=allowed or 'phys_ntu.py:' in origin and '(TableRef.__getitem__)' in origin
                        allowed=allowed or 'kernels/_ntu_math.py:' in origin and any('ntu/math.py:' in s or '(activa' in s for s in project)
                        if not allowed:outside.append(dict(primitive=e.primitive.name,stack=project))
                    for child in e.params.values():audit(child,project)
            elif isinstance(value,(tuple,list)):
                for child in value:audit(child,parent)
            elif isinstance(value,dict):
                for child in value.values():audit(child,parent)
        audit(params['jaxpr'])
        native=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
        text=str(native.module);a.out.with_suffix('.ttir').write_text(text)
        result={'mode':a.mode,'backend':'cpu','target':'cuda/sm120','case':a.case,'levels':a.levels,
                'elapsed_s':time.perf_counter()-start,'ttir_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'ttir_bytes':len(text.encode()),'fp64_equations':len(wide),
                'fp64_arithmetic_origins':dict(origins),'fp64_control_containers':controls,
                'fp64_outside_original_DOUBLE':outside,'operational_projection':a.operational,
                'ambient_x64_restored':not jax.config.x64_enabled}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if a.mode=='interpret':assert not any(nonfinite.values()),nonfinite
    else:assert not outside,('FP64 outside source DOUBLE islands',outside[:3])


if __name__=='__main__':main()
