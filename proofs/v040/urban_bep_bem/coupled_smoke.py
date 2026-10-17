"""BEP/BEM CPU production-step smoke on an authentic case with urban cells.

The proof can privately register the candidate for qualification while the
public catalog remains closed. This override is recorded and is not CLI
admission evidence. Inputs are symlinked read-only; only a private namelist
changes. CPU Pallas interpret/REAL instruction emulation is not GPU evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

p=argparse.ArgumentParser();p.add_argument('--src',type=Path,required=True);p.add_argument('--case',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--urban',type=int,choices=[2,3],required=True)
p.add_argument('--pbl',type=int,choices=[1,8],required=True);p.add_argument('--steps',type=int,default=3)
p.add_argument('--private-qualification',action='store_true');a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
sys.path.insert(0,str(a.src/'src'));sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'o1_nlbind'))
import jax
import jax.numpy as jnp
import numpy as np
import cpu_safe
cpu_safe.install();assert jax.devices()[0].platform=='cpu'
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as pt
real_pallas_call=pl.pallas_call
def interpret(*args,**kwargs):
    kwargs['interpret']=True;kwargs.pop('compiler_params',None)
    return real_pallas_call(*args,**kwargs)
def inline(asm,*,args,result_shape_dtypes,**kwargs):
    op=asm.split()[0].split('.')[0];x,y=(jnp.asarray(v,jnp.float64) for v in args)
    value={'add':lambda:x+y,'sub':lambda:x-y,'mul':lambda:x*y,'div':lambda:x/y}[op]()
    dtype=result_shape_dtypes[0].dtype
    if jnp.dtype(dtype)==jnp.float32:value=jax.lax.reduce_precision(value,exponent_bits=8,mantissa_bits=23)
    return [value.astype(dtype)]
pl.pallas_call=interpret;pt.elementwise_inline_asm=inline
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om
if a.private_qualification:
    from gpuwrf.io import scheme_catalog
    scheme_catalog._IMPLEMENTED['sf_urban_physics']=frozenset({0,1,2,3})
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
assert str(om.__file__).startswith(str(a.src))
root=Path('<USER_HOME>/wrf_gpu2_lanes/s1-bepbem');started=time.perf_counter()
record=dict(src=str(a.src),case=str(a.case),urban=a.urban,pbl=a.pbl,steps=a.steps,
    private_qualification=a.private_qualification,platform='cpu',pallas='interpret',fast_defaults=os.environ.get('GPUWRF_FAST_DEFAULTS'))
try:
  with tempfile.TemporaryDirectory(dir=root) as tmp:
    tmp=Path(tmp);case=tmp/'case';case.mkdir()
    for name in ('wrfinput_d01','wrfbdy_d01'):
        (case/name).symlink_to((a.case/name).resolve())
        record[name+'_sha256']=hashlib.sha256((a.case/name).read_bytes()).hexdigest()
    text=(a.case/'namelist.input').read_text()
    for key,value in {'sf_urban_physics':a.urban,'bl_pbl_physics':a.pbl,'sf_sfclay_physics':1}.items():
        pattern=rf'^\s*{key}\s*=.*$'
        if re.search(pattern,text,re.M|re.I):text=re.sub(pattern,f' {key}={value},',text,flags=re.M|re.I)
        else:text=text.replace('&physics\n',f'&physics\n {key}={value},\n')
    (case/'namelist.input').write_text(text)
    cfg=pipeline.NestedPipelineConfig(case,tmp/'out',tmp/'proof',hours=1,max_dom=1)
    _,bundles,_,_,_,carries=pipeline._load_domains(cfg,('d01',))
  # Native CUDA-target physics, interpreted/emulated on a CPU just as before
  # the portability facade landed. This is not portable-backend qualification.
  try:
    from gpuwrf.backends import pallas as facade
  except ImportError:
    record['native_facade_emulation']=False
  else:
    facade.cuda_native=lambda:True
    record['native_facade_emulation']=True
  ns=bundles['d01'].namelist;carry=carries['d01'];om._resolve_operational_suite(ns)
  record['urban_columns']=len(ns.urban_bundle.indices);assert record['urban_columns']>0
  record['load_s']=time.perf_counter()-started
  record['urban_carry_bytes']=sum(x.nbytes for x in jax.tree.leaves(carry.urban_bep))
  seed=jax.tree.map(lambda x:np.asarray(x).copy(),carry.urban_bep.state)
  cb=om.build_clock_base(ns);start=time.perf_counter()
  # Match the production chunk entry: one compiled loop containing all own-steps.
  record['phase']='compiling_and_advancing';a.out.write_text(json.dumps(record,indent=2)+'\n')
  print('CPU production chunk',a.steps,'own-steps; load_s',record['load_s'],flush=True)
  carry=om._advance_chunk(carry,ns,1,cb,n_steps=a.steps,cadence=int(ns.radiation_cadence_steps));jax.block_until_ready(carry)
  record['compile_and_steps_s']=time.perf_counter()-start
  record['nonfinite']=sum(int(np.sum(~np.isfinite(np.asarray(x)))) for x in jax.tree.leaves(carry)
                          if np.issubdtype(x.dtype,np.floating))
  record['guards']=None if carry.census is None else int(np.asarray(carry.census.guards).sum())
  record['urban_state_max_change']=max(float(np.max(np.abs(np.asarray(y)-x))) for x,y in zip(jax.tree.leaves(seed),jax.tree.leaves(carry.urban_bep.state)))
  record['urban_drag_nonzero']=int(np.count_nonzero(np.asarray(carry.urban_bep.sources.a_u)))
  record['urban_heat_nonzero']=int(np.count_nonzero(np.asarray(carry.urban_bep.sources.b_t)))
  assert record['nonfinite']==0 and record['guards'] in (None,0),record
  assert record['urban_state_max_change']>0 and record['urban_drag_nonzero']>0 and record['urban_heat_nonzero']>0,record
  record['status']='PASS_CPU_SMOKE_ONLY'
except Exception as e:
  import traceback
  record['status']='FAIL';record['error']=repr(e);record['traceback']=traceback.format_exc()
finally:
  record['wall_s']=time.perf_counter()-started;a.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if record['status']=='FAIL':raise SystemExit(1)
