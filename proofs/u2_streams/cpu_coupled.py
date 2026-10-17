"""Real Swiss release step loop with interpreted kernels, OFF/ON three steps.

CPU structural/coupled regression evidence. Independent original CPU-WRF
auxhist parity lives in the separate fixture; this is not station/S1 evidence.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'src'))
sys.path.insert(0,str(repo/'proofs/o1_nlbind'))
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=False)
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
assert jax.devices()[0].platform == 'cpu'
import cpu_safe
cpu_safe.install()
original = pl.pallas_call
def interpret(*args,**kwargs):
    kwargs['interpret']=True
    kwargs.pop('compiler_params',None)
    return original(*args,**kwargs)
pl.pallas_call=interpret
from gpuwrf.contracts import state as state_contract
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig,_load_domains
from gpuwrf.runtime import operational_mode as om
from gpuwrf.streams.runtime import prepare_streams

source=Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
case=out/'case'
case.mkdir()
for name in ('wrfinput_d01','wrfbdy_d01'):
    (case/name).symlink_to(source/name)
(case/'namelist.input').write_bytes((source/'namelist.input').read_bytes())
(case/'gpuwrf.streams.json').write_text(json.dumps({'version':1,'streams':[{
    'name':'airport','domain':'d01','interval_steps':2,'buffer_frames':16,
    'region':{'stations':[{'id':'ctr','i':20,'j':20},{'id':'west','i':9,'j':12}]},
    'variables':{'T2':['instant','min','max','mean'],'RAINNC':['instant','accumulation']}}]}))
config=NestedPipelineConfig(case,out/'output',out/'proof',.015,1)
t0=time.perf_counter()
_,bundles,_,run_start,dt,carries=_load_domains(config,('d01',))
nl=bundles['d01'].namelist
clock=om.build_clock_base(nl)
base=jax.tree.map(jnp.copy,carries['d01'])
runtime,seeded=prepare_streams(config,('d01',),carries,dt,run_start)
record={'backend':'cpu','source':str(repo),'work':'real Swiss42x42x44 dt18; selected Pallas calls interpreted; CENSUS1SLIM0; 3ownsteps OFF/ON',
        'initialization_s':time.perf_counter()-t0,'model_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (repo/'src/gpuwrf/runtime/operational_mode.py',repo/'src/gpuwrf/runtime/operational_state.py',repo/'src/gpuwrf/streams/device.py')}}
results={}
for label,carry in [('off',base),('on',seeded['d01'])]:
    t1=time.perf_counter()
    result=om._advance_chunk(carry,nl,1,clock,n_steps=3,cadence=int(nl.radiation_cadence_steps))
    jax.block_until_ready(result)
    record[label+'_compile_and_3steps_s']=time.perf_counter()-t1
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(result)
               if np.asarray(x).dtype.kind in 'fc')
    results[label]=result
    assert result.census is not None
    record[label+'_guards']=np.asarray(result.census.guards).tolist()
    record[label+'_f2_counts']=np.asarray(result.census.f2_counts).tolist()
    (out/'progress.json').write_text(json.dumps(record,indent=2)+'\n')
left=results['off']
right=results['on'].replace(regional_streams=None)
a,at=jax.tree.flatten(left)
b,bt=jax.tree.flatten(right)
assert at==bt
record['shared_carry_leaves']=len(a)
record['shared_carry_byte_equal']=[np.asarray(x).tobytes()==np.asarray(y).tobytes() for x,y in zip(a,b,strict=True)]
assert all(record['shared_carry_byte_equal']), 'CPU observer changes shared carry; inspect the actual differing leaves'
final=runtime.flush({'d01':results['on']})['d01']
runtime.close()
record['complete_frames']=runtime.report()
record['partial_window_samples']=int(final.regional_streams[0].samples)
assert record['partial_window_samples']==1
assert record['on_guards']==record['off_guards']
record['wall_s']=time.perf_counter()-t0
record['scope']='CPU coupled regression only, not fidelity. Existing guards/F2 events preserved; no global F2 clearance. Actual GPU and pristine-auxhist gates are separate.'
record['resolved_flags']={name:os.environ.get(name) for name in ('GPUWRF_FAST_DEFAULTS','GPUWRF_HISTORY_INSTEP_DIAG','GPUWRF_CENSUS','GPUWRF_CENSUS_SLIM')}
(out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('backend','shared_carry_leaves','partial_window_samples','wall_s')}))
