"""CPU compiled-artifact cost and rate identity on real Swiss FDDA operands.

Usage: component_cost.py SOURCE_ROOT ORIGINAL_FDDA_MODULE TF02_ROOT OUTPUT_JSON
No GPU performance claim. Original source is an exact git-show baseline module.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.assimilation import fdda_grid as candidate

root, old_path, ref, output = map(Path,sys.argv[1:5])
assert jax.devices()[0].platform=='cpu'
jax.config.update('jax_enable_x64',True)
spec=importlib.util.spec_from_file_location('original_grid_fdda',old_path)
baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
vals={}
with Dataset(ref/'wrffdda_d01') as ds:
    for n,v in (('u','U'),('v','V'),('t','T'),('q','Q')):
        for s in ('old','new'):vals[f'{n}_{s}']=np.asarray(ds[f'{v}_NDG_{s.upper()}'][0],np.float32)[None]
with Dataset(ref/'repaired/wrfout_d01_2023-01-15_00:00:18') as ds:
    get=lambda n:np.asarray(ds[n][0],np.float32)
    rvrd=np.float32(np.float32(461.6)/np.float32(287.))
    theta=(get('THM')+np.float32(300.))/(np.float32(1.)+rvrd*get('QVAPOR'))
    zw=(get('PH')+get('PHB'))/np.float32(9.81)
    args=tuple(map(jnp.asarray,(get('U'),get('V'),theta,get('QVAPOR'),get('PBLH'),get('HGT'),zw,get('P')+get('PB'))))
results={};exes={};carries={}
for name,module in (('baseline',baseline),('candidate',candidate)):
    inp=module.GridFddaInputs(**{k:jnp.asarray(v) for k,v in vals.items()},
        options=module.GridFddaOptions(guv=3e-4,gt=3e-4,gq=3e-4,interval_m=60,end_h=24),record_steps=(0,))
    carry=module.initial_fdda_carry(inp,jnp.float32,**({'with_history':True} if name=='candidate' else {}))
    def call(c,a,seconds,step,module=module,inp=inp,name=name):
        return module.fdda_driver_tendencies(c,inp,*a[:7],seconds,step,jnp.float32,
            **({'p_phy':a[7]} if name=='candidate' else {}))
    started=time.perf_counter()
    compiled=jax.jit(call).lower(carry,args,jnp.float64(18.),jnp.int32(2)).compile()
    compile_s=time.perf_counter()-started
    result=jax.block_until_ready(compiled(carry,args,jnp.float64(18.),jnp.int32(2)))
    analysis=compiled.memory_analysis()
    results[name]={'compile_s':compile_s,'memory':{k:getattr(analysis,k) for k in
        ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes','generated_code_size_in_bytes')},
        'rates':[np.asarray(x) for x in result.tendencies]}
    exes[name]=compiled;carries[name]=carry
samples={k:[] for k in exes}
for _ in range(5):
    for name in exes:jax.block_until_ready(exes[name](carries[name],args,jnp.float64(18.),jnp.int32(2)))
for _ in range(40):
    for name in exes:
        started=time.perf_counter();jax.block_until_ready(exes[name](carries[name],args,jnp.float64(18.),jnp.int32(2)))
        samples[name].append(time.perf_counter()-started)
rates=[]
for a,b in zip(results['baseline'].pop('rates'),results['candidate'].pop('rates'),strict=True):
    rates.append({'byte_equal':a.tobytes()==b.tobytes(),'nonfinite':int((~np.isfinite(b)).sum()),
                  'different_values':int(np.count_nonzero(a!=b)),'max_abs':float(np.max(np.abs(a-b)))})
for name in exes:results[name]['median_cpu_wall_s']=float(np.median(samples[name]))
receipt={'platform':'cpu','gpu_executed':False,'source_root':str(root),'baseline_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),
         'candidate_sha256':hashlib.sha256((root/'src/gpuwrf/assimilation/fdda_grid.py').read_bytes()).hexdigest(),
         'shape':[44,42,42],'work':'actual Swiss analysis and model entry18s; STEPFG1; g3e-4; drivers only, no forecast',
         'results':results,'rates':rates,'hfx_carry_bytes':44*42*42*4,
         'GPU_launches_compile_peak_runtime':'UNKNOWN; manager GPU requalification required',
         'scope':'CPU static compiled artifact and interleaved component wall only, not S1/GPU speed or full model peak'}
output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
assert all(x['nonfinite']==0 for x in rates)
