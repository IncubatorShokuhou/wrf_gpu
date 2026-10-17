"""Isolate the old-KF response using original amplitudes on the qualified base fluxes."""
from pathlib import Path
import functools
import json
import subprocess
import sys
import numpy as np
from evidence_gate import field_score, require_finite
import jax
import jax.numpy as jnp
from probe_ksas14 import parse
from gpuwrf.physics.cumulus_kf99 import environment,trigger,updraft,prepare_downdraft,downdraft,feedback

jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
reference,work,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
dtype=np.float32 if mode=='r4' else np.float64
data=json.loads((work/f'trigger_{mode}.json').read_text())
report={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'mode':mode,'cases':{}}
def base(*args):
    env=environment(*args[:8],mode=mode);tr=trigger(env,args[8],args[10],mode=mode)
    s=prepare_downdraft(env,tr,updraft(env,tr,mode=mode),args[9],args[10],mode=mode)
    return env,tr,downdraft(env,tr,s,mode=mode)
basefn=jax.jit(base)
def response(env,tr,s,dx,ainc):
    k=jnp.arange(len(env.t),dtype=jnp.int32);valid=(k>=1)&(k<=s['LTOP'])
    fluxes=('UMF','DMF','DETLQ','DETIC','UDR','UER','DER','DDR')
    scaled=dict(s)
    for name in fluxes:scaled[name]=jnp.where(valid,s[name]*ainc,s[name])
    scaled['TDER']=s['TDER']*ainc;scaled['PPTFLX']=s['PPTFLX']*ainc
    return feedback(env,tr,scaled,dx,mode=mode)
fn=jax.jit(response)
for case in range(1,6):
    rec=data[str(case)]
    if not rec['closure']:continue
    raw=parse(reference/f'case{case}_{mode}_zero.txt');lc=np.int32(rec['downdraft']['SCALARS'][0])
    require_finite(**raw)
    env,tr,s=basefn(*[raw[k].astype(dtype) for k in ('T','QV','P','DZ','RHO','W0AVG','U','V')],lc,dtype(raw['DT']),dtype(raw['DX']))
    attempts=[]
    for row in rec['closure']:
        scal=row['scalars'];require_finite(closure_reference=np.asarray(scal));out=fn(env,tr,s,dtype(raw['DX']),dtype(scal[2]))
        fields={}
        for name,index in (('ABEG',5),('NSTEP',8),('DTIME',9)):
            a=float(out[name]);b=scal[index]
            fields[name]=field_score(a,b,lambda:0. if name=='NSTEP' else (32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)*abs(b))
        for name in ('OMG','FXM','TG','QG'):
            b=np.array(row['profiles'][name],dtype);a=np.asarray(out[name])[1:len(b)+1]
            fields[name]=field_score(a,b,lambda:(32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)*np.max(np.abs(b)))
        attempts.append({'ncount':int(scal[0]),'ainc':scal[2],'fields':fields})
    report['cases'][str(case)]={'attempts':attempts}
    (work/f'comparison_feedback_{mode}.json').write_text(json.dumps(report,indent=2))
    print(case,[(r['ncount'],r['ainc'],[k for k,v in r['fields'].items() if not v['pass']]) for r in attempts],flush=True)
