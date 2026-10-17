"""Audit the new no-FMA literal BYTE policy; never loosen the frozen gate."""
import json
import hashlib
import os
import sys
from pathlib import Path

import jax
import numpy as np

from full_compare import inputs
from gpuwrf.physics.pbl_eeps import eeps_column
from gpuwrf.physics.pbl_keps import keps_column

assert jax.devices()[0].platform=='cpu'
rows=json.loads(Path(sys.argv[1]).read_text())['records']
receipt=dict(scope='literal eager REAL32 no-FMA BYTE audit; not GPU/coupled fidelity',
             xla_flags=os.environ.get('XLA_FLAGS',''),fields=0,byte_fields=0,values=0,byte_values=0,differences=[])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt['fixture_sha256']=sha(sys.argv[1])
receipt['port_source_sha256']={name:sha(sys.modules[fn.__module__].__file__)
                              for name,fn in (('eeps',eeps_column),('keps',keps_column))}
source_real32='--source-real32' in sys.argv[3:]
if source_real32:
    from literal_real32 import source_backend,wrap
    receipt['scope']='host literal REAL4/libm evaluation of unchanged port bodies; compiled JAX qualified separately'
    receipt['literal_backend_sha256']=sha(Path(__file__).with_name('literal_real32.py'))
for scheme,fn in ((16,eeps_column),(17,keps_column)):
    for n in sorted({r['n'] for r in rows if r['scheme']==scheme}):
        selected=[r for r in rows if r['scheme']==scheme and r['n']==n]
        batch={k:np.stack([inputs(r)[k] for r in selected]) for k in inputs(selected[0])}
        if source_real32:
            with source_backend():
                results=[fn(**{k:wrap(v) for k,v in inputs(r).items()}) for r in selected]
            got={k:np.stack([x[k] for x in results]) for k in results[0]}
        else:
            with jax.disable_jit():
                got=jax.vmap(lambda x:fn(**x))(batch)
        for i,r in enumerate(selected):
            for name,ref in r['output'].items():
                if (scheme==17 and name=='kpbl') or (scheme==16 and name in ('tpe','prandtl','tpe_adv')):
                    continue
                a=np.atleast_1d(np.asarray(got[name][i]))
                b=np.asarray(ref,np.int32 if name=='kpbl' else np.float32)
                assert a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all(), (
                    scheme,r['name'],r['step'],name,a.shape,b.shape,str(a.dtype),str(b.dtype),a)
                mask=a.view(np.uint32)==b.view(np.uint32)
                receipt['fields']+=1;receipt['byte_fields']+=int(mask.all())
                receipt['values']+=a.size;receipt['byte_values']+=int(mask.sum())
                if not mask.all():
                    def ordered(x):
                        bits=x.view(np.uint32).astype(np.int64)
                        return np.where(bits&0x80000000,0x80000000-(bits&0x7fffffff),0x80000000+bits)
                    receipt['differences'].append(dict(scheme=scheme,name=r['name'],step=r['step'],field=name,
                        different=int((~mask).sum()),max_ulp=int(np.abs(ordered(a)-ordered(b)).max()),
                        max_abs=float(np.abs(a.astype(float)-b.astype(float)).max())))
        receipt['status']='PASS' if receipt['fields']==receipt['byte_fields'] else 'FAIL'
        Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+'\n')
        print(scheme,n,receipt['status'],receipt['byte_fields'],receipt['fields'],flush=True)
if source_real32:
    assert receipt['status']=='PASS','literal REAL4 BYTE gate failed; see receipt'
