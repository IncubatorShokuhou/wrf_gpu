"""Exact-source composition check on both real WN3 MP40 step programs.

Reuses the numerical arm's loader and CPU emulation. Traces only: this is
program/input identity evidence, never a replacement WRF fidelity oracle.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import time

root=Path(sys.argv[1]).resolve(); case=Path(sys.argv[2]).resolve()
out=Path(sys.argv[3]).resolve()
sys.path.insert(0,str(root/'src'))
harness=root/'proofs/s1_nestspec/two_domain_gate.py'
text=harness.read_text().split('observe.wants_carry = True')[0]
sys.argv=[str(harness),str(case),str(out), '0']
namespace={'__file__':str(harness),'__name__':'_identity_loader'}
exec(compile(text,str(harness),'exec'),namespace)
jax=namespace['jax']; np=namespace['np']; pipeline=namespace['pipeline']
from gpuwrf.runtime import operational_mode as om
started=time.monotonic()
config=pipeline.NestedPipelineConfig(case,out.parent/'unused_history',out.parent/'unused_proof',hours=1,max_dom=2)
_,bundles,_,_,_,carries=pipeline._load_domains(config,('d01','d02'))
result={'source':str(root),'backend':'cpu','scope':'JAX-vs-JAX program/input identity only',
        'status':'TRACING','load_s':time.monotonic()-started,'domains':{}}
print('Loaded both domains',result['load_s'],flush=True)
def array_seal(values):
    digest=hashlib.sha256(); signature=[]
    for value in values:
        array=np.asarray(value)
        signature.append((list(array.shape),str(array.dtype)))
        digest.update(array.tobytes())
    return digest.hexdigest(),signature

for name,carry in carries.items():
    nml=bundles[name].namelist; cb=om.build_clock_base(nml)
    graph=jax.make_jaxpr(lambda c:om._advance_chunk_fori(c,nml,1,cb,n_steps=1,
        cadence=int(nml.radiation_cadence_steps)))(carry)
    canonical=re.sub(r'name_and_src_info=[^\]\n]*','',str(graph))
    canonical=re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?','',canonical)
    input_sha,input_signature=array_seal(jax.tree.leaves(carry))
    const_sha,const_signature=array_seal(graph.consts)
    result['domains'][name]={'sha256':hashlib.sha256(canonical.encode()).hexdigest(),
        'chars':len(canonical),'input_leaves':len(input_signature),
        'input_sha256':input_sha,'input_signature':input_signature,
        'constant_sha256':const_sha,'constant_signature':const_signature,
        'timeseries':None if getattr(carry,'timeseries',None) is None else 'ACTIVE',
        'data_assimilation':None if nml.data_assimilation is None else 'ACTIVE'}
    del graph,canonical
    jax.clear_caches()
    result['wall_s']=time.monotonic()-started
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('Sealed domain',name,result['wall_s'],flush=True)
result['status']='COMPLETE'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{n:v for n,v in row.items() if n not in ('input_signature',)}
                  for k,row in result['domains'].items()},indent=2))
