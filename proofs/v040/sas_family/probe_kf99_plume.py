"""All old-KF trigger attempts and accepted updraft profiles vs pristine."""
from pathlib import Path
import functools
import json
import subprocess
import sys
import numpy as np
from evidence_gate import field_score, require_finite
import jax
from probe_ksas14 import parse
from gpuwrf.physics.cumulus_kf99 import environment,trigger,updraft

jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
reference,work,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
dtype=np.float32 if mode=='r4' else np.float64
data=json.loads((work/f'trigger_{mode}.json').read_text())
report={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'mode':mode,'cases':{}}
envfn=jax.jit(functools.partial(environment,mode=mode))
trfn=jax.jit(functools.partial(trigger,mode=mode))
upfn=jax.jit(functools.partial(updraft,mode=mode))
for case in range(1,6):
    raw=parse(reference/f'case{case}_{mode}_zero.txt')
    require_finite(**raw)
    env=envfn(*[raw[k].astype(dtype) for k in ('T','QV','P','DZ','RHO','W0AVG','U','V')])
    rec=data[str(case)];attempts=[]
    for row in rec['plume_scalars']:
        require_finite(plume_reference=np.asarray(row))
        lc=int(row[0]);tr=trfn(env,np.int32(lc),dtype(raw['DX']));out=upfn(env,tr)
        observed=np.array([lc,out['LET'],out['LTOP'],out['KFRZ'],out['CLDHGT'],out['ABE'],out['UPOLD'],out['UPNEW'],out['WTW']],dtype)
        expected=np.array(row,dtype)
        names=('LC','LET','LTOP','KFRZ','CLDHGT','ABE','UPOLD','UPNEW','WTW')
        fields={k:field_score(observed[i],expected[i],lambda:0. if i<4 else
                (32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)*np.abs(expected[i])) for i,k in enumerate(names)}
        records=[r for r in rec['plumes'] if r['LC']==lc]
        assert len(records)<=1
        fields['usable']=field_score(np.asarray(out['usable']),np.asarray(bool(records)),lambda:0.)
        profiles={}
        if records:
            for name,expected in records[0]['profiles'].items():
                b=np.array(expected,dtype=dtype)
                a=np.asarray(out[name])
                a=a[None] if name=='TRPPT' else a[1:len(b)+1]
                profiles[name]=field_score(a,b,lambda:(32*np.finfo(np.float32).eps if mode=='r4' else 1.e-9)*np.max(np.abs(b)))
                profiles[name]['bad_levels']=[i[0] for i in profiles[name]['bad_values']]
        attempts.append({'lc':lc,'fields':fields,'profiles':profiles,
                         'pass':all(f['pass'] for f in (*fields.values(),*profiles.values()))})
    report['cases'][str(case)]={'attempts':attempts}
    (work/f'comparison_plume_{mode}.json').write_text(json.dumps(report,indent=2))
    print(case,[(r['lc'],r['pass'],[k for k,v in r['fields'].items() if not v['pass']],
                  [k for k,v in r['profiles'].items() if not v['pass']]) for r in attempts],flush=True)
