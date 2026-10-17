"""Old OSAS against pristine output and its recorded, uninjected XKT2 draws."""
from pathlib import Path
import functools
import hashlib
import json
import subprocess
import sys
import time
import numpy as np
from evidence_gate import field_score, require_finite

work=Path(sys.argv[1]);mode=sys.argv[2]
import jax
from gpuwrf.physics.cumulus_osas95 import cu_osas95_columns
jax.config.update('jax_enable_x64',True)
assert jax.devices()[0].platform=='cpu'
input_path=Path(__file__).resolve().parents[2]/'v034/scalesas/savepoints/columns_inputs.npz'
with np.load(input_path) as raw:inp=dict(raw)
args=[inp[k] for k in ('T','QV','QC','QI','P','PI','RHO','DZ','U','V','W')]+[inp['P8W'][:,0],inp['XLAND']]
report={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),'mode':mode,'arms':{}}
for dt,stepcu in ((54.,5),(18.,1)):
    name=f'dt{int(dt)}_stepcu{stepcu}_shallow1_{mode}'
    output=work/f'{name}.out';lines=output.read_text().splitlines()
    assert lines[0].startswith('DRAW_XKT2')
    draws=np.fromstring(lines[0].removeprefix('DRAW_XKT2'),sep=' ')
    n,kx=inp['T'].shape
    require_finite(draws=draws)
    assert len(draws)==n and np.all((draws>=0.)&(draws<1.))
    expected={k:[] for k in ('RAINCV','PRATEC','HBOT','HTOP','RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN')}
    for i in range(n):
        header=np.fromstring(lines[1+5*i],sep=' ')
        assert int(header[0])==i+1
        for j,k in enumerate(('RAINCV','PRATEC','HBOT','HTOP')):expected[k].append(header[1+j])
        for j,k in enumerate(('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN')):
            expected[k].append(np.fromstring(lines[2+5*i+j],sep=' '))
    expected={k:np.asarray(v) for k,v in expected.items()}
    fn=jax.jit(functools.partial(cu_osas95_columns,dt=dt,stepcu=stepcu,literals=mode))
    require_finite(**{k:v for k,v in inp.items() if v.dtype.kind in "fiu"}, draws=draws)
    start=time.monotonic();exe=fn.lower(*args,draws=draws).compile();compile_s=time.monotonic()-start
    require_finite(**{k:v for k,v in inp.items() if v.dtype.kind in "fiu"}, draws=draws)
    actual={k:np.asarray(v) for k,v in exe(*args,draws=draws).items()}
    require_finite(**{k:v for k,v in inp.items() if v.dtype.kind in 'fiu'}, draws=draws)
    fields={}
    for k,a in actual.items():
        b=expected[k].astype(a.dtype)
        def bound():
            if k in ('HBOT','HTOP'):return np.zeros_like(b)
            if mode=='r4':return np.spacing(np.maximum(np.abs(a),np.abs(b)))
            scale=np.max(np.abs(b),axis=1,keepdims=True) if b.ndim==2 else np.abs(b)
            return 1.e-9*scale
        fields[k]=field_score(a,b,bound)
        bad=np.zeros(a.shape,bool)
        for index in fields[k]['bad_values']:bad[tuple(index)]=True
        fields[k]['bad_columns']=np.flatnonzero(np.any(bad,axis=1) if bad.ndim==2 else bad).tolist()

    rec={'fields':fields,'all_pass':all(f['pass'] for f in fields.values()),'compile_s':compile_s,
         'actual_active':int(np.count_nonzero(actual['RAINCV'])),'oracle_active':int(np.count_nonzero(expected['RAINCV'])),
         'pristine_output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    report['arms'][name]=rec
    np.savez_compressed(work/f'candidate_{name}.npz',**actual)
    np.savez_compressed(work/f'fixture_{name}.npz',**expected,DRAW_XKT2=draws)
    (work/f'comparison_{mode}.json').write_text(json.dumps(report,indent=2));print(name,json.dumps(rec),flush=True)
