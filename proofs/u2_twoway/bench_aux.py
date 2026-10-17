"""Actual packed-feedback API: fused vs field dispatch, CPU timing/memory only."""
from pathlib import Path
import json,os,sys,time
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.coupling.boundary_feedback import build_feedback_weights,point_feedback_weights,StateFeedbackWeights
from gpuwrf.coupling.feedback_aux import feedback_group,FLUXES
from gpuwrf.runtime.history_accumulators import PackedFields

assert os.environ['JAX_PLATFORMS']=='cpu' and jax.devices()[0].platform=='cpu'
root=Path(__file__).resolve().parents[2]
with np.load(root/'tests/fixtures/u2_feedback_oracle.npz') as file:
    p,c=map(jnp.asarray,(file['p0'][0],file['c0'][0]))
names=('TH2',*FLUXES,'WA','WT','ZWT','LFMASS','STMASS','RTMASS','WOOD','STBLCP','FASTCP',
       'IVGTYP','ISLTYP','VAR','CON','OA1','OA2','OA3','OA4','OL1','OL2','OL3','OL4','SHDMAX','SHDMIN','SHDAVG')
parent=PackedFields.pack({name:p+i for i,name in enumerate(names)})
child=PackedFields.pack({name:c+i for i,name in enumerate(names)})
mass=build_feedback_weights(parent_grid_ratio=3,i_parent_start=3,j_parent_start=3,
    parent_we=22,parent_sn=20,child_we=36,child_sn=30)
weights=StateFeedbackWeights(mass,mass,mass,point=point_feedback_weights(mass),smooth_option=2)
start=time.perf_counter(); compiled=feedback_group.lower(parent,child,weights).compile();cold=time.perf_counter()-start
fused=compiled(parent,child,weights);sequential=feedback_group.__wrapped__(parent,child,weights)
jax.block_until_ready((fused,sequential))
assert np.array_equal(fused.data,sequential.data),'compiled family must retain every source operator bit'
def bench(fn):
    result=[]
    for _ in range(15):
        start=time.perf_counter()
        for _ in range(10):out=fn(parent,child,weights)
        jax.block_until_ready(out);result.append((time.perf_counter()-start)/10)
    return float(np.median(result))
times={name:bench(fn) for name,fn in (('fused_api_s',feedback_group),('sequential_api_s',feedback_group.__wrapped__))}
mem=compiled.memory_analysis()
record=dict(backend='cpu',no_gpu_claim=True,names=names,fields=len(names),cold_compile_s=cold,
    source_bits_equal=True,times=times,speedup=times['sequential_api_s']/times['fused_api_s'],
    memory={name:getattr(mem,name) for name in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')},
    geometry='22x20 parent,36x30 child,REAL32,3:1,spec_zone1;weights dynamic;all named active operands nonuniform',
    cache='NONE GPUWRF_JAX_CACHE=0')
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
