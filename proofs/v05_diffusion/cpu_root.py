"""Bounded CPU nested-root OFF/default/changed regression; not WRF/GPU fidelity."""
import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--input-dir',type=Path,required=True)
p.add_argument('--prepare-only',action='store_true')
a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert subprocess.run(['git','-C',str(a.source),'symbolic-ref','--quiet','HEAD'],capture_output=True).returncode != 0
assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain','--untracked-files=no'])
a.out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.source/'src'),str(a.source/'proofs/w44_params'),str(a.source/'proofs/o1_nlbind')]
started=time.monotonic()
rec=dict(scope=__doc__,gpu_executed=False,platform='cpu',source=str(a.source),
         commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),
         cache='NONE: GPUWRF_JAX_CACHE=0, GPUWRF_NESTED_AOT=0',phase='imports',
         harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         prepare_only=a.prepare_only,stepped_roots={})

def save():
 rec.update(wall_s=time.monotonic()-started,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (a.out/'receipt.json').write_text(json.dumps(rec,indent=2,default=str)+'\n')
 print(rec['phase'],round(rec['wall_s'],1),rec['peak_rss_KiB'],flush=True)

save()
rc=1
try:
 import jax
 import jax.numpy as jnp
 import numpy as np
 from jax.experimental import pallas as pl
 from jax._src.pallas import pallas_call as pc
 original_call=pc.pallas_call
 def cpu_call(*args,**kw):return original_call(*args,**{**kw,'interpret':True})
 pc.pallas_call=pl.pallas_call=cpu_call
 import cpu_safe
 cpu_safe.install()
 from gpuwrf.backends import pallas as facade
 from gpuwrf.contracts import state as sc
 from gpuwrf.integration import nested_pipeline as pipeline
 from gpuwrf.runtime import domain_tree as dt,operational_mode as om
 from gpuwrf import runtime_parameters as rp
 from cold_prepare import domain_operands
 from family_work import defaults
 from product_schedule import product_schedule
 from root_gpu import differences
 assert all(d.platform=='cpu' for d in jax.devices())
 assert Path(om.__file__).resolve().is_relative_to(a.source.resolve())
 facade.cuda_native=lambda:False
 sc._gpu_device=lambda:jax.devices('cpu')[0]
 assert rp.UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES==frozenset({'diffusion_parameters'})
 rp.UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES=frozenset()  # this probe only; shipped refusal unchanged
 names=('d01','d02')
 rec['inputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.input_dir.iterdir()
                if p.is_file() and (p.name.startswith(('wrfinput_','wrfbdy_')) or p.name=='namelist.input')}
 cfg=pipeline.NestedPipelineConfig(a.input_dir,a.out/'unused-output',a.out/'unused-proof',hours=3,max_dom=2)
 rec['phase']='init';save()
 hierarchy,bundles,meta,start,steps,initial=pipeline._load_domains(cfg,names)
 run_kwargs,rec['schedule']=product_schedule(a.input_dir,names,steps,hours=3)
 nesting=meta['nesting']
 rec['dt_s']=steps
 values=defaults('diffusion')
 rec['values']={'on':values,'changed':{**values,'c_s':.5}}
 def clone(carries):return jax.block_until_ready(jax.tree.map(lambda v:v.copy() if isinstance(v,jax.Array) else v,carries))
 def host(carries):return {jax.tree_util.keystr(path):np.asarray(v).copy() for path,v in jax.tree_util.tree_flatten_with_path(carries)[0]}
 def trim():
  gc.collect()
  import ctypes
  ctypes.CDLL(None).malloc_trim(0)
 key_sets={}; baseline=None; on_host=None
 for label,selection in (('off',None),('on',values),('changed',{**values,'c_s':.5})):
  bound=bundles if selection is None else {name:dataclasses.replace(b,namelist=rp.bind_diffusion_parameters(b.namelist,selection)) for name,b in bundles.items()}
  tree=dt.DomainTree.from_domains(hierarchy,bound,feedback_enabled=nesting['feedback'],
       feedback_spec_zone=nesting['spec_zone'],feedback_smooth_option=nesting['smooth_option'])
  key_sets[label]={}
  for domain in names:
   _,operands,key,kw=domain_operands(tree,initial,domain)
   key_sets[label][domain]=key
   rec.setdefault('prepared_work',{}).setdefault(label,{})[domain]=kw
   del operands
  rec['phase']='root_'+label;save()
  if a.prepare_only:
   del tree,bound
   trim()
   continue
  result=dt.run_operational_domain_tree(tree,root_steps=1,carries=clone(initial),initial_own_steps={n:0 for n in names},max_event_tail=0,**run_kwargs)
  jax.block_until_ready(result.carries)
  rec['stepped_roots'][label]=1
  output=host(result.carries)
  assert all(np.isfinite(v).all() for v in output.values())
  rec.setdefault('leaves',{})[label]=len(output)
  rec.setdefault('own_steps',{})[label]=result.own_steps
  rec.setdefault('cpu_chunk_cache_size',{})[label]=om._advance_chunk_fori._cache_size()
  if label=='off':baseline=output
  elif label=='on':
   rec['default_differences']=differences(baseline,output)
   rec['default_byte_exact']=not rec['default_differences']
   on_host=output
  else:
   rec['changed_differences']=differences(on_host,output)
   rec['changed_active']=bool(rec['changed_differences'])
  del result,tree,bound
  if label=='off':jax.clear_caches()
  trim();save()
 rec['same_on_changed_keys']=key_sets['on']==key_sets['changed']
 rec['keys']=key_sets
 if a.prepare_only:
  assert rec['same_on_changed_keys']
  assert key_sets['off'] != key_sets['on']
  rec['phase']='PREPARE_ONLY_PASS';rc=0;save()
  sys.stdout.flush();sys.stderr.flush();os._exit(0)
 rec['same_on_changed_chunk_cache_size']=rec['cpu_chunk_cache_size']['on']==rec['cpu_chunk_cache_size']['changed']
 assert rec['default_byte_exact'], 'ON-at-defaults differs from OFF'
 assert rec['cpu_chunk_cache_size']['on'] > 0, 'no concrete chunk-cache evidence'
 assert rec['changed_active'] and rec['same_on_changed_keys'] and rec['same_on_changed_chunk_cache_size']
 rec['phase']='PASS';rc=0
except BaseException as exc:
 import traceback
 rec.update(phase='FAIL',exception=repr(exc),traceback=traceback.format_exc())
finally:
 save()
 sys.stdout.flush();sys.stderr.flush();os._exit(rc)
