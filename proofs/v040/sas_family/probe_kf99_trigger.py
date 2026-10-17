"""Pristine old-KF environment and actual candidate-layer trigger checkpoints."""
from pathlib import Path
import json
import functools
import subprocess
import sys
import numpy as np
from evidence_gate import field_score, require_finite
import jax
from probe_ksas14 import parse
from gpuwrf.physics.cumulus_kf99 import environment,trigger

jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
reference,work,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
dtype=np.float32 if mode=='r4' else np.float64
staged=json.loads((work/f'trigger_{mode}.json').read_text())
report={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'mode':mode,'cases':{}}
keys=('lc','kpbl','k','klcl','zlcl','tlcl','tvlcl','tven','tmix','qmix','pmix','zmix','depth','wlcl','thte','thtes','vmflcl','au0')
for case in range(1,6):
    raw=parse(reference/f'case{case}_{mode}_zero.txt')
    require_finite(**raw)
    env=jax.jit(functools.partial(environment,mode=mode))(*[raw[k].astype(dtype) for k in ('T','QV','P','DZ','RHO','W0AVG','U','V')])
    rec=staged[str(case)];fields={}
    for name,value in (('Q0',env.q),('QES',env.qes),('TV0',env.tv),('DP',env.dp),('Z0',env.z),('DZA',env.dza)):
        a=np.asarray(value)[1:];b=np.array(rec['environment'][name],dtype=dtype)
        fields[name]=field_score(a,b,lambda:(32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)*np.max(np.abs(b)))
    levels=field_score(np.array([env.ml,env.l5,env.l4,env.llfc]),np.asarray(rec['environment']['LEVELS']),lambda:0.)
    fields['LEVELS']=levels
    attempts=[]
    for row in rec['triggers']:
        require_finite(trigger_reference=np.asarray(row))
        out=jax.jit(functools.partial(trigger,mode=mode))(env,int(row[0]),dtype(raw['DX']))
        a=np.array([out[k] for k in keys]);b=np.array(row,dtype=dtype)
        def bound():
            value=(32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)*np.abs(b)
            value[:4]=0
            return value
        result=field_score(a,b,bound)
        result.update(active=bool(out['active']),bad_fields=[keys[i[0]] for i in result['bad_values']])
        attempts.append(result)
    report['cases'][str(case)]={'environment':fields,'triggers':attempts}
    (work/f'comparison_{mode}.json').write_text(json.dumps(report,indent=2));print(case,report['cases'][str(case)],flush=True)
