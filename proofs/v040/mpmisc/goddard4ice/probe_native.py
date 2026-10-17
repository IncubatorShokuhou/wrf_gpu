"""CPU interpretation and deviceless CUDA lowering of the exact column kernel."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'scripts/v025'))
import cpu_guard
import argparse,json,time
import numpy as np
import jax
from jax._src.pallas.triton import lowering
from gpuwrf.physics.goddard4ice.native import native_column
sys.path.insert(0,str(Path(__file__).resolve().parent))
from probe import INPUT,FIELDS,SURFACE

p=argparse.ArgumentParser();p.add_argument('oracle',type=Path);p.add_argument('output',type=Path);p.add_argument('--lower-only',action='store_true');a=p.parse_args()
data=json.loads((a.oracle/'case4.json').read_text());cols=data['columns'];sc=data['scalars']
inp=[np.asarray(cols[key],np.float32)[:,None,None] for key in INPUT]
receipt={'backend':'cpu','GPU_execution':False}
if not a.lower_only:
 start=time.monotonic();result=native_column(*inp,np.float32(sc['DT']),interpret=True,full=True)
 receipt['interpret_seconds']=time.monotonic()-start;receipt['fields']={}
 for name,refname in {**FIELDS,**SURFACE}.items():
  got=np.asarray(result[name]);got=got[:,0,0] if got.ndim==3 else got[0,0]
  ref=np.asarray(cols[refname] if refname in cols else sc[refname],np.float32)
  receipt['fields'][name]={'finite':bool(np.isfinite(got).all()),'maxabs':float(np.max(np.abs(got.astype(float)-ref.astype(float))))}
 print('INTERPRET',receipt,flush=True)
program=jax.make_jaxpr(lambda *args:native_column(*args,np.float32(sc['DT']),full=True))(*inp).jaxpr
for equation in program.eqns:
 if equation.primitive.name=='pallas_call':
  start=time.monotonic();params=equation.params
  result=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
  receipt['cuda_lower']={'status':'PASS','seconds':time.monotonic()-start,'IR_characters':len(str(result.module))}
  break
else:raise AssertionError('no Pallas program found')
a.output.write_text(json.dumps(receipt,indent=2));print(receipt['cuda_lower'],flush=True)
