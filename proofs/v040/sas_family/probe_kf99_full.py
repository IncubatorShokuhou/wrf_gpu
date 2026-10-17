"""Complete distinct KFPARA driver outputs vs pristine old KFCPS reference.

R8: frozen per-column field max relative1e-9. Native REAL4: original rain
32eps relative; subtraction-derived thermal/vapor rates use the source operand
8eps envelope /TIMEC(/PI); fresh condensate32eps field scale. No floors.
"""
from pathlib import Path
import functools
import json
import subprocess
import sys
import time
import numpy as np
from evidence_gate import field_score, require_finite
import jax
from probe_ksas14 import parse
from gpuwrf.physics.cumulus_kf99 import kf99_column

jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
reference,work,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
work.mkdir(parents=True,exist_ok=True)
dtype=np.float32 if mode=='r4' else np.float64
records=[parse(reference/f'case{i}_{mode}_zero.txt') for i in range(1,6)]
inputs={k:np.stack([r[k] for r in records]).astype(dtype) for k in records[0]}
source_environment=json.loads((reference.parent/'SS78_oldKF_trigger'/f'trigger_{mode}.json').read_text())
source_q0=np.array([source_environment[str(i)]['environment']['Q0'] for i in range(1,6)],dtype=dtype)
require_finite(**inputs, source_q0=source_q0)
keys=('T','QV','P','DZ','RHO','W0AVG','U','V','PI')
def column(*args):
    return kf99_column(*args[:8],dtype(inputs['DT'][0]),dtype(inputs['DX'][0]),
                       pi_phy=args[8],mode=mode)
fn=jax.jit(jax.vmap(column))
args=[inputs[k] for k in keys]
start=time.monotonic();exe=fn.lower(*args).compile();compile_s=time.monotonic()-start
actual={k:np.asarray(v) for k,v in exe(*args).items()}
require_finite(**actual)
samples=[]
for _ in range(5):
    start=time.monotonic();jax.block_until_ready(exe(*args));samples.append(time.monotonic()-start)
expected={k:inputs[k] for k in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RAINCV','PRATEC','NCA','TIMEC')}
fields={}
for name,b in expected.items():
    a=actual[name]
    def bound_factory():
        if name in ('NCA','TIMEC'):bound=np.zeros_like(b)
        elif mode=='r8':
            scale=np.max(np.abs(b),axis=1,keepdims=True) if b.ndim==2 else np.abs(b)
            bound=1.e-9*scale
        elif name in ('RAINCV','PRATEC'):bound=32*np.finfo(np.float32).eps*np.abs(b)
        elif name in ('RTHCUTEN','RQVCUTEN'):
            # No tendency is generated on failed triggers. TIMEC is then zero.
            dt=np.where(inputs['TIMEC']>0.,inputs['TIMEC'],1.)[:,None]
            initial=inputs['T'].astype(np.float64) if name=='RTHCUTEN' else source_q0.astype(np.float64)
            factor=dt*inputs['PI'] if name=='RTHCUTEN' else dt
            updated=initial+b.astype(np.float64)*factor
            bound=8*np.finfo(np.float32).eps*(np.abs(initial)+np.abs(updated))/factor
        else:bound=32*np.finfo(np.float32).eps*np.max(np.abs(b),axis=1,keepdims=True)
        return bound
    fields[name]=field_score(a,b,bound_factory,operands=(inputs['TIMEC'],inputs['T'],inputs['PI'],source_q0))

memory=exe.memory_analysis()
report={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'mode':mode,
        'fields':fields,'pass':all(v['pass'] for v in fields.values()) and not np.any(actual['SOURCE_FATAL']),
        'oracle_active':int(np.count_nonzero(inputs['RAINCV'])),'actual_active':int(np.count_nonzero(actual['RAINCV'])),
        'borrow_events':actual['BORROW_EVENTS'].tolist(),'source_fatal':actual['SOURCE_FATAL'].tolist(),
        'water_error_percent':actual['WATER_ERROR_PERCENT'].tolist(),
        'compile_s':compile_s,'warm_median_s':float(np.median(samples)),
        'temp_bytes':memory.temp_size_in_bytes,'argument_bytes':memory.argument_size_in_bytes,'output_bytes':memory.output_size_in_bytes}
np.savez_compressed(work/f'candidate_{mode}.npz',**actual)
(work/f'comparison_{mode}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
