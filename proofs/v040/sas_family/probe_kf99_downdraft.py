"""Literal old-KF downdraft against source checkpoints after original label165."""
from pathlib import Path
import functools
import json
import subprocess
import sys
import numpy as np
from evidence_gate import field_score, require_finite
import jax
from probe_ksas14 import parse
from gpuwrf.physics.cumulus_kf99 import environment,trigger,updraft,prepare_downdraft,downdraft

jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
reference,work,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
dtype=np.float32 if mode=='r4' else np.float64
staged=json.loads((work/f'trigger_{mode}.json').read_text())
report={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'mode':mode,'cases':{}}
def run(t,q,p,dz,rho,w,u,v,lc,dt,dx):
    env=environment(t,q,p,dz,rho,w,u,v,mode=mode)
    tr=trigger(env,lc,dx,mode=mode)
    pl=updraft(env,tr,mode=mode)
    s=prepare_downdraft(env,tr,pl,dt,dx,mode=mode)
    return downdraft(env,tr,s,mode=mode)
fn=jax.jit(run)
names=('LC','LFS','LDB','LDT','LVF','TADVEC','TIMEC','NIC','USR','PEFF','PEFF2','TDER','PPTFLX','CPR','UPDINC','CNDTNF')
for case in range(1,6):
    record=staged[str(case)]['downdraft']
    if not record:continue
    require_finite(scalar_reference=np.asarray(record['SCALARS']))
    raw=parse(reference/f'case{case}_{mode}_zero.txt');lc=np.int32(record['SCALARS'][0])
    require_finite(**raw)
    out=fn(*[raw[k].astype(dtype) for k in ('T','QV','P','DZ','RHO','W0AVG','U','V')],lc,dtype(raw['DT']),dtype(raw['DX']))
    observed=np.array([lc if name=='LC' else out[name] for name in names],dtype)
    expected=np.array(record['SCALARS'],dtype)
    fields={k:field_score(observed[i],expected[i],lambda:0. if i in (0,1,2,3,4,7) else
            (32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)*np.abs(expected[i])) for i,k in enumerate(names)}
    for name,expected in record.items():
        if name=='SCALARS':continue
        b=np.array(expected,dtype)
        a=np.asarray(out[name]);start=int(out['LDB']) if name=='THETED' else 1
        a=a[start:start+len(b)]
        fields[name]=field_score(a,b,lambda:(32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)*np.max(np.abs(b)))
        fields[name]['bad_levels']=[i[0] for i in fields[name]['bad_values']]
    report['cases'][str(case)]={'pass':all(v['pass'] for v in fields.values()),'fields':fields}
    (work/f'comparison_downdraft_{mode}.json').write_text(json.dumps(report,indent=2))
    print(case,[(k,v) for k,v in fields.items() if not v['pass']],flush=True)
