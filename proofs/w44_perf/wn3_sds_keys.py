"""Real WN3 before/after content-walker keys; CPU metadata only, no forecast."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import types

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['GPUWRF_JAX_CACHE']=='0'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
base=Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/src')
case=Path('<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227')
sys.path[:0]=[str(a.source/'src'),str(a.source/'proofs/o1_nlbind')]
a.out.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
rec={'source':str(a.source),'phase':'start','gpu_executed':False,'forecast_steps':0,'keys':[],'errors':[]}
def save():
 rec.update(wall_s=time.monotonic()-started,rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (a.out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
 print(rec['phase'],round(rec['wall_s'],2),flush=True)
save()
try:
 manifest=json.loads(Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/sealed/BUNDLE_MANIFEST.json').read_text())['inputs'][str(case)]
 got={}
 for name in manifest:
  with (case/name).open('rb') as f: got[name]=hashlib.file_digest(f,'sha256').hexdigest()
 assert got==manifest
 rec['input_sha256']=got
 import jax
 from jax.experimental import pallas as pl
 from jax._src.pallas import pallas_call as pc
 raw=pc.pallas_call
 pc.pallas_call=pl.pallas_call=lambda *x,**kw:raw(*x,**{**kw,'interpret':True})
 import cpu_safe
 cpu_safe.install()
 from gpuwrf.runtime import aot_cheap_key as ck
 from gpuwrf.integration import nested_pipeline as pipe,init_kernels as ik
 from gpuwrf.runtime import domain_tree as dt
 from gpuwrf.runtime.aot_key_probe import case_program_keys
 from gpuwrf.contracts import state as sc
 assert all(d.platform=='cpu' for d in jax.devices())
 sc._gpu_device=lambda:jax.devices('cpu')[0]
 spec=importlib.util.spec_from_file_location('base_ck',base/'src/gpuwrf/runtime/aot_cheap_key.py')
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 fixed=ck._walk
 original=types.FunctionType(old._walk.__code__,ck.__dict__,'_walk')
 count=Counter();mode=''
 def compare(label,call):
  global mode
  row={'label':label}
  for mode,walker in [('before',original),('after',fixed)]:
   def spy(h,obj,depth):
    count[(mode,'visits')]+=1
    if isinstance(obj,jax.ShapeDtypeStruct):count[(mode,'sds')]+=1
    return walker(h,obj,depth)
   ck._walk=spy;count.clear()
   row[mode]=call()
   row[mode+'_visits']=count[(mode,'visits')]
   row[mode+'_sds']=count[(mode,'sds')]
  ck._walk=fixed
  assert row['before'] is not None and row['after'] is not None
  row['equal']=row['before']==row['after']
  rec['keys'].append(row);save()
  return row['after']
 def init(fn,*args,**kw):
  compare('init:'+fn.__name__,lambda:ik.radiation_init_key(fn,args,kw))
  return fn(*args,**kw)
 roots=[]
 for root in (base/'src/gpuwrf',a.source/'src/gpuwrf'):
  files=ck._trace_reachable_source_files(root)
  roots.append(ck._source_tree_content_digest(root,files))
 rec['trace_source_fingerprints']=roots;assert roots[0]==roots[1]
 rec['phase']='load';save()
 cfg=pipe.NestedPipelineConfig(case,a.out/'unused_out',a.out/'unused_proof',hours=3,max_dom=3)
 hierarchy,bundles,meta,_,_,carries=pipe._load_domains(cfg,('d01','d02','d03'),_init_kernel_call=init)
 n=meta['nesting'];tree=dt.DomainTree.from_domains(hierarchy,bundles,feedback_enabled=n['feedback'],feedback_spec_zone=n['spec_zone'],feedback_smooth_option=n['smooth_option'])
 for parent,edges in tree.edges.items():
  for edge in edges:carries[edge.child]=dt._operational_force(edge,carries[parent],carries[edge.child])
 rec['phase']='step_keys';save()
 # Fresh per-call factories in the authoritative probe prevent reuse of a key
 # computed by the preceding walker. Require actual walker activity below.
 compare('all_step_programs',lambda:case_program_keys(tree,carries,components=True))
 for row in rec['keys']:
  assert row['equal'] and row['before_sds']==row['after_sds']==0
  assert row['before_visits'] and row['after_visits']
 programs=rec['keys'][-1]['after']
 assert 'fused/d02' in programs and all(v['cheap_key'] for v in programs.values())
 rec['phase']='done';save()
except BaseException as exc:
 rec['errors'].append({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
 rec['phase']='failed';save();raise
