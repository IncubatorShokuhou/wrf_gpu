"""One real PROD two-nest root: OFF, nine default controls, nine changed values.

CPU regression transport/roundoff gate, not CPU-WRF or GPU fidelity. Jarvis2 is
explicit so all nine public controls have active consumers. The unchanged
4e-6 relative field-scale bound is registered before looking at this root.
"""
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
p.add_argument('--require-byte',action='store_true',help='manager W2 fence causal gate: every default leaf must be byte-identical')
a=p.parse_args()
assert a.out.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/v45-lsmparams'))
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert subprocess.run(['git','-C',str(a.source),'symbolic-ref','--quiet','HEAD'],capture_output=True).returncode != 0
assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain','--untracked-files=no'])
a.out.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.source/'src'),str(a.source/'proofs/w44_params'),str(a.source/'proofs/w44_lsmparams'),str(a.source/'proofs/o1_nlbind')]
started=time.monotonic()
rec=dict(scope=__doc__,platform='cpu',gpu_executed=False,source=str(a.source),
         commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),
         cache='NONE: JAX0/NESTED_AOT0',phase='imports',default_bound=4e-6,require_byte=a.require_byte,
         harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
rec['numerical_policy']={k:os.environ.get(k) for k in
    ('GPUWRF_FAST_DEFAULTS','GPUWRF_CENSUS','GPUWRF_JAX_CACHE','GPUWRF_NESTED_AOT','JAX_ENABLE_X64')}

def save():
 rec.update(wall_s=time.monotonic()-started,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (a.out/'receipt.json').write_text(json.dumps(rec,indent=2,default=str)+'\n')
 print(rec['phase'],round(rec['wall_s'],1),rec['peak_rss_KiB'],flush=True)

save();rc=1
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
 from gpuwrf.contracts import state as state_contract
 from gpuwrf.integration import nested_pipeline as pipeline
 from gpuwrf.runtime import domain_tree as dt,operational_mode as om
 from gpuwrf.io.noahmp_options import NoahMPOptions
 from gpuwrf.surface_params import controls as sc
 from cold_prepare import domain_operands
 from product_schedule import product_schedule
 import field_table as ft
 assert all(d.platform=='cpu' for d in jax.devices())
 assert Path(om.__file__).resolve().is_relative_to(a.source.resolve())
 facade.cuda_native=lambda:False
 state_contract._gpu_device=lambda:jax.devices('cpu')[0]
 assert sc.UNAVAILABLE_SURFACE_CONTROLS==frozenset(sc.LSM_CONTROLS)
 sc.UNAVAILABLE_SURFACE_CONTROLS=frozenset()  # process-local research, shipped refusal unchanged
 names=('d01','d02')
 rec['inputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.input_dir.iterdir()
                if p.is_file() and (p.name.startswith(('wrfinput_','wrfbdy_')) or p.name=='namelist.input')}
 cfg=pipeline.NestedPipelineConfig(a.input_dir,a.out/'unused-output',a.out/'unused-proof',hours=3,max_dom=2)
 rec['phase']='init';save()
 hierarchy,bundles,meta,start,steps,initial=pipeline._load_domains(cfg,names)
 # Explicit supported research context, identical in OFF/ON/CHANGED.
 bundles={name:dataclasses.replace(b,namelist=dataclasses.replace(b.namelist,noahmp_options=
              dataclasses.replace(b.namelist.noahmp_options or NoahMPOptions(),opt_crs=2))) for name,b in bundles.items()}
 run_kwargs,rec['schedule']=product_schedule(a.input_dir,names,steps,hours=3)
 rec['options']={n:dataclasses.asdict(b.namelist.noahmp_options) for n,b in bundles.items()}
 nesting=meta['nesting'];rec['dt_s']=steps
 prior={n:sc.prior_controls(b.namelist,sorted(sc.LSM_CONTROLS),land_state=initial[n].noahmp_land) for n,b in bundles.items()}
 factors={n:(.99 if n=='snow_emis' else .01 if n=='hvb' else 1.05) for n in sc.LSM_CONTROLS}
 changed={n:sc.expand_controls(b.namelist,factors,land_state=initial[n].noahmp_land) for n,b in bundles.items()}
 rec['changed_factors']=factors
 def clone(carries):return jax.block_until_ready(jax.tree.map(lambda v:v.copy() if isinstance(v,jax.Array) else v,carries))
 def host(carries):return {jax.tree_util.keystr(path):np.asarray(v).copy() for path,v in jax.tree_util.tree_flatten_with_path(carries)[0]}
 def differences(before,after):
  assert before.keys()==after.keys()
  rows=[]
  for name,x in before.items():
   y=after[name];assert x.dtype==y.dtype and x.shape==y.shape
   if x.tobytes()==y.tobytes():continue
   x64,y64=x.astype(np.float64),y.astype(np.float64)
   delta=np.abs(x64-y64);valid=np.abs(x64)<1e30
   scale=float(np.max(np.abs(x64[valid]),initial=0))
   error=float(delta.max())
   rows.append(dict(leaf=name,cells=int(np.count_nonzero(x!=y)),max_abs=error,field_max=scale,
                    max_abs_over_field_max=error/scale if scale else float('inf')))
  return rows
 def trim():
  gc.collect()
  import ctypes
  ctypes.CDLL(None).malloc_trim(0)
 key_sets={};baseline=None;on_host=None
 for label,selection in (('off',None),('on',prior),('changed',changed)):
  bound=bundles if selection is None else {name:dataclasses.replace(b,namelist=sc.bind_domain_controls(
      b.namelist,selection,name,land_state=initial[name].noahmp_land)) for name,b in bundles.items()}
  tree=dt.DomainTree.from_domains(hierarchy,bound,feedback_enabled=nesting['feedback'],
       feedback_spec_zone=nesting['spec_zone'],feedback_smooth_option=nesting['smooth_option'])
  key_sets[label]={}
  for domain in names:
   _,operands,key,kw=domain_operands(tree,initial,domain)
   key_sets[label][domain]=key
   rec.setdefault('prepared_work',{}).setdefault(label,{})[domain]=kw
   del operands
  rec['phase']='root_'+label;save()
  result=dt.run_operational_domain_tree(tree,root_steps=1,carries=clone(initial),initial_own_steps={n:0 for n in names},max_event_tail=0,**run_kwargs)
  jax.block_until_ready(result.carries)
  output=host(result.carries)
  assert all(np.isfinite(v).all() for v in output.values())
  rec.setdefault('leaves',{})[label]=len(output)
  rec.setdefault('own_steps',{})[label]=result.own_steps
  rec.setdefault('cpu_chunk_cache_size',{})[label]=om._advance_chunk_fori._cache_size()
  rec.setdefault('pre_mask_census',{})[label]={n:(None if c.census is None else
      {key:None if value is None else np.asarray(value).tolist() for key,value in c.census._asdict().items()})
      for n,c in result.carries.items()}
  if label=='off':baseline=output
  elif label=='on':
   rec['default_differences']=differences(baseline,output)
   rec['default_byte_exact']=not rec['default_differences'];on_host=output
   tables={}
   for domain,carry in result.carries.items():
    slots=ft.live_slots(carry)
    ref,cand=(ft.named_fields(v,domain,*slots) for v in (baseline,output))
    for key,value in (("State.xland",carry.state.xland),("State.qfx",carry.state.qfx),
                      ("noahmp_land.smois",carry.noahmp_land.smois)):
     assert cand[key].tobytes()==np.asarray(value).tobytes(),key
    assert ref['State.xland'].tobytes()==cand['State.xland'].tobytes()
    mask=ft.scale_mask(ref,ft.ice_mask(a.input_dir/f'wrfinput_{domain}'))
    tables[domain]=ft.table(ref,cand,mask)
   rec['default_field_tables']=tables
   rec['default_bound_scope']='unchanged w44 land/no-ice named-field scale AND global-leaf guard'
   rec['default_in_registered_class']=all(r['max_abs_over_field_max']<=rec['default_bound']
      for r in rec['default_differences']+[r for rows in tables.values() for r in rows])
  else:
   rec['changed_differences']=differences(on_host,output)
   rec['changed_active']=bool(rec['changed_differences'])
  del result,tree,bound
  if label=='off':jax.clear_caches()
  trim();save()
 rec['keys']=key_sets;rec['same_on_changed_keys']=key_sets['on']==key_sets['changed']
 rec['same_on_changed_chunk_cache_size']=rec['cpu_chunk_cache_size']['on']==rec['cpu_chunk_cache_size']['changed']
 if a.require_byte:
  assert rec['default_byte_exact'],'ON default is not byte-identical to OFF (fence causal gate)'
 else:
  assert rec['default_in_registered_class'],'ON default outside unchanged preregistered 4e-6 class'
 assert rec['changed_active'] and rec['same_on_changed_keys'] and rec['same_on_changed_chunk_cache_size']
 assert rec['cpu_chunk_cache_size']['on']>0
 rec['phase']='PASS';rc=0
except BaseException as exc:
 import traceback
 rec.update(phase='FAIL',exception=repr(exc),traceback=traceback.format_exc())
finally:
 save();sys.stdout.flush();sys.stderr.flush();os._exit(rc)
