"""Compiled-artifact identity for explicit WRF INTEGER-to-REAL conversion."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import jax
import jax.numpy as jnp
import numpy as np


def canonical_hlo(hlo):
    # New XLA also prints an interned filename/location/stack-frame table
    # before its first computation. It is debug metadata, like metadata={}.
    hlo=re.sub(r'\nFileNames\n.*?(?=\n%|\nENTRY)', '\n',hlo,flags=re.S)
    return re.sub(r' metadata=\{[^}]*\}','',hlo)


def main(old,new,output):
    assert jax.devices()[0].platform=='cpu'
    with np.load(new/'tests/fixtures/s2_gfdl_cloud_v1.npz',allow_pickle=False) as f:
        args=tuple(jnp.asarray(f['grid_'+k]) for k in ('t','p','q','qc','qi','cutop','cubot'))
    ep2=np.float32(np.float32(287.)/np.float32(461.6))
    artifacts=[]
    for i,root in enumerate((old,new)):
        path=root/'src/gpuwrf/physics/ra_gfdleta.py'
        spec=importlib.util.spec_from_file_location('cloud_cast_'+str(i),path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        run=jax.jit(jax.vmap(lambda *x:module.grid_cloud_fraction(*x,ep2=ep2)))
        executable=run.lower(*args).compile()
        hlo=executable.as_text()
        canonical=canonical_hlo(hlo)
        output.with_suffix('.'+('old' if i==0 else 'new')+'.hlo').write_text(hlo)
        artifacts.append(dict(canonical_hlo_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            f64_instructions=sum(bool(re.search(r'\bf64\[',line)) for line in hlo.splitlines()),
            temporary_bytes=executable.memory_analysis().temp_size_in_bytes))
    equal=artifacts[0]['canonical_hlo_sha256']==artifacts[1]['canonical_hlo_sha256']
    report=dict(canonical_optimized_HLO_BYTE=equal,artifacts=artifacts,
        scope='CPU compilation only, metadata stripped; reuse FG25 compiled numerical validation iff identical')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert equal,'Compiled artifact changed; numerical validation cannot be carried forward'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('old','new','output'):p.add_argument(k,type=Path)
    a=p.parse_args();main(a.old,a.new,a.output)
