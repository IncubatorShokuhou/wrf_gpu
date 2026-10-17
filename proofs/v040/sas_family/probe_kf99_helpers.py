"""Independent actual old-KF helper checkpoints -> native JAX component gate."""
from pathlib import Path
import functools
import json
import subprocess
import sys
import time
import numpy as np
from evidence_gate import field_score, require_finite
import jax
import jax.numpy as jnp
from gpuwrf.physics import kf99_thermo as kf

jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
work=Path(sys.argv[1]);mode=sys.argv[2]
dtype=np.float32 if mode=='r4' else np.float64
data=json.loads((work/f'helpers_{mode}.json').read_text())
report={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'mode':mode,'helpers':{}}
functions={'TPMIX':kf.tpmix,'TPDD':kf.tpdd,'ENVIRTHT':kf.envirtht,'CONDLOAD':kf.condload,'DTFRZNEW':kf.dtfrznew}
for name,records in data.items():
    if len(sys.argv)>3 and name!=sys.argv[3]:
        continue
    x=np.array([r['input'] for r in records],dtype=dtype)
    expected=np.array([r['output'] for r in records],dtype=dtype)
    require_finite(input=x, reference=expected)
    fn=jax.jit(jax.vmap(functools.partial(functions[name],mode=mode)))
    start=time.monotonic();result=fn(*x.T);jax.block_until_ready(result);elapsed=time.monotonic()-start
    if name=='ENVIRTHT':actual=np.asarray(result)[:,None]
    else:actual=np.stack(result,axis=1)
    if name=='CONDLOAD':actual=np.concatenate((actual,np.zeros((len(x),2),dtype)),axis=1)
    rec=field_score(actual,expected,lambda: (32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)
                    *np.max(np.abs(expected),axis=0,keepdims=True),operands=(x,))
    rec.update(count=len(records),compile_exec_s=elapsed)
    if rec['max_abs'] is not None:
        rec['max_abs_by_output']=np.abs(actual.astype(np.float64)-expected).max(axis=0).tolist()
        rec['byte_fraction_by_output']=np.mean(actual==expected,axis=0).tolist()
    report['helpers'][name]=rec
    np.savez_compressed(work/f'candidate_{name}_{mode}.npz',actual=actual,expected=expected,input=x)
    (work/f'comparison_{mode}.json').write_text(json.dumps(report,indent=2));print(name,rec,flush=True)
