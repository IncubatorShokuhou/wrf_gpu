"""CPU init-only cost/state census, exact original lifecycle vs the final fix.

Loads real domains through the same CPU-safe loader as the twin tool. The only
baseline substitution is the exact lifecycle function at the frozen base;
all other production source is identical (checked by the lane source diff).
"""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import time
import numpy as np

source,case,out=map(Path,sys.argv[1:4])
loader=Path(__file__).with_name('step_dump.py')
text=loader.read_text(); prefix=text[:text.index('# Deletion-sensitivity mutants')]
sys.argv=[str(loader),str(source),str(case),str(out.with_suffix('.unused.npz')),'0','']
namespace={'__file__':str(loader),'__name__':'qsqseed_loader'}
exec(compile(prefix,str(loader),'exec'),namespace)
import jax
import jax.numpy as jnp
from gpuwrf.coupling import physics_couplers as C
base=subprocess.check_output(['git','-C',str(source),'show','700a382f8b:src/gpuwrf/coupling/physics_couplers.py'],text=True)
node=next(n for n in ast.parse(base).body if isinstance(n,ast.FunctionDef) and n.name=='_mynn_state_with_first_call_qke')
old_namespace={}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(C.__file__),'exec'),C.__dict__,old_namespace)
baseline=old_namespace['_mynn_state_with_first_call_qke']
carry=namespace['c'];state=carry.state;grid=namespace['bundles']['d01'].namelist.grid
records={}; outputs={}
for tag,function in (('base',baseline),('fixed',C._mynn_state_with_first_call_qke)):
    f=jax.jit(lambda s,first:function(s,grid,first))
    flag=jnp.asarray(True)
    started=time.perf_counter();compiled=f.lower(state,flag).compile();cold=time.perf_counter()-started
    result=compiled(state,flag);jax.block_until_ready(result)
    times=[]
    for _ in range(7):
        started=time.perf_counter();jax.block_until_ready(compiled(state,flag));times.append(time.perf_counter()-started)
    memory=compiled.memory_analysis()
    cycling=jax.jit(lambda s,first:function(s,grid,first,cycling=True)).lower(state,flag).compiler_ir('hlo')
    cycling_sha=hashlib.sha256(cycling.as_hlo_text(print_large_constants=True).encode()).hexdigest()
    outputs[tag]=result
    records[tag]={'cycling_full_hlo_sha256':cycling_sha,'compile_s':cold,'warm_median_s':float(np.median(times)),'warm_samples_s':times,
         'memory':{k:int(getattr(memory,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')},
         'optimized_hlo_sha256':hashlib.sha256(compiled.as_text().encode()).hexdigest()}
    # False/restart/cycling reuse checks are numerical regression evidence;
    # original-WRF fidelity is the separate oracle and two-step twin.
    later=compiled(state,jnp.asarray(False));jax.block_until_ready(later)
    false_times=[]
    for _ in range(7):
        started=time.perf_counter();jax.block_until_ready(compiled(state,jnp.asarray(False)));false_times.append(time.perf_counter()-started)
    records[tag]['nonfirst_warm_median_s']=float(np.median(false_times))
    records[tag]['nonfirst_warm_samples_s']=false_times
    for name in state.__slots__:
        a,b=getattr(state,name),getattr(later,name)
        if a is not None: assert np.asarray(a).dtype==np.asarray(b).dtype and np.asarray(a).shape==np.asarray(b).shape and np.asarray(a).tobytes()==np.asarray(b).tobytes(),('later',tag,name)
assert records['base']['cycling_full_hlo_sha256']==records['fixed']['cycling_full_hlo_sha256']
changed=[];qke_control={}
for name in state.__slots__:
    a,b=getattr(outputs['base'],name),getattr(outputs['fixed'],name)
    if a is None or b is None: assert a is b;continue
    a,b=np.asarray(a),np.asarray(b)
    assert np.isfinite(a).all() and np.isfinite(b).all(),name
    assert a.shape==b.shape and a.dtype==b.dtype,name
    if a.tobytes()!=b.tobytes():changed.append(name)
    if name=='qke':
        qke_control={'byte_equal':a.tobytes()==b.tobytes(),'max_abs':float(abs(a.astype(float)-b).max()),
            'relative_rms_fixed_vs_base':float(np.sqrt(np.mean((a.astype(float)-b)**2))/max(np.sqrt(np.mean(a.astype(float)**2)),1e-30)),
            'changed_values':int(np.count_nonzero(a!=b)),
            'snow_columns':int(np.count_nonzero((np.asarray(state.qs)>1e-6).any(axis=0))),
            'max_input_snow':float(np.max(np.asarray(state.qs)))}
assert set(changed)<= {'qke','el_pbl','qsq'},changed
# Static lifecycle paths must return the original State without even dispatch.
assert C._mynn_state_with_first_call_qke(state,grid,False) is state
assert C._mynn_state_with_first_call_qke(state,grid,True,restart=True) is state
report={'backend':jax.default_backend(),'grid':list(state.theta.shape),'case':str(case),'source':str(source),
        'source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'records':records,'changed_state_leaves':changed,'qke_control':qke_control,'nonfirst_restart_exact':True,'cycling_full_hlo_exact':True,
        'scope':'CPU init component, no GPU cost/forecast speed claim'}
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
