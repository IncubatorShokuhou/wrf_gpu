"""Initialize the real two-domain mini inputs on CPU before reserving CUDA."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True

assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
source=Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/src')
case=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/cuda_mini_case_v1')
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(source/'src'),str(Path(__file__).resolve().parents[2]/'proofs/o1_nlbind')]
import jax
from jax.experimental import pallas as pl
from jax._src.pallas import pallas_call as pc
original=pc.pallas_call
pc.pallas_call=pl.pallas_call=lambda *a,**kw:original(*a,**{**kw,'interpret':True})
import cpu_safe
cpu_safe.install()
from gpuwrf.contracts import state as sc
sc._gpu_device=lambda:jax.devices('cpu')[0]
from gpuwrf.integration import nested_pipeline as p
from gpuwrf.runtime import domain_tree as dt
import numpy as np
start=time.monotonic()
cfg=p.NestedPipelineConfig(case,out/'unused_outputs',out/'proof',hours=3,max_dom=2)
names=('d01','d02')
hierarchy,bundles,meta,run_start,dts,carries=p._load_domains(cfg,names)
nesting=meta.get('nesting',dict(feedback=False,spec_zone=1,smooth_option=1))
tree=dt.DomainTree.from_domains(hierarchy,bundles,feedback_enabled=nesting['feedback'],
    feedback_spec_zone=nesting['spec_zone'],feedback_smooth_option=nesting['smooth_option'])
for parent,edges in tree.edges.items():
    for edge in edges:carries[edge.child]=dt._operational_force(edge,carries[parent],carries[edge.child])
sizes={name:list(carry.state.qv.shape) for name,carry in carries.items()}
assert sizes=={'d01':[44,16,16],'d02':[44,15,15]},sizes
assert all(np.isfinite(np.asarray(x)).all() for c in carries.values() for x in jax.tree.leaves(c) if hasattr(x,'dtype') and np.issubdtype(x.dtype,np.number))
receipt={'schema':'w44-perf.cuda-mini-input-init.v1','physical_backend':jax.default_backend(),'gpu_executed':False,
         'sizes':sizes,'dt_s':dts,'seconds':time.monotonic()-start,'finite':True,
         'input_manifest_sha256':hashlib.sha256((case/'MINI_INPUTS.json').read_bytes()).hexdigest(),
         'scope':'Actual real-input nested loader and force-down on CPU; no forecast/fidelity qualification'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
