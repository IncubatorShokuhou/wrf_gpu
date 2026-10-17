"""Compare native GFDL emissivities/cloud overlap to the same port functions."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from literal import load
from gpuwrf.physics import ra_gfdleta as port


def main(native,output):
    assert jax.devices()[0].platform=='cpu'
    literal=load(Path(port.__file__))
    data={k:[] for k in ('amount','index','delta','e2','temperature','water','e3',
                         'number','cloud','bottom','top','overlap')}
    with native.open('rb') as f:
        for _ in range(64):
            for k,n in (('amount',2),('index',2),('delta',2),('e2',2),
                        ('temperature',17),('water',17),('e3',17)):
                value=np.fromfile(f,np.float32,n)
                assert value.size==n and np.isfinite(value).all()
                data[k].append(value)
        for _ in range(9):
            for k,n,dtype in (('number',1,np.int32),('cloud',9,np.float32),
                               ('bottom',9,np.int32),('top',9,np.int32),('overlap',81,np.float32)):
                value=np.fromfile(f,dtype,n)
                assert value.size==n and np.isfinite(value).all()
                data[k].append(value.reshape((9,9),order='F') if k=='overlap' else value)
        assert not f.read(1)
    data={k:np.asarray(v) for k,v in data.items()}
    results={}

    def check(name,value,reference):
        value=np.asarray(value)
        assert value.shape==reference.shape and value.dtype==reference.dtype==np.float32
        assert np.isfinite(value).all() and np.isfinite(reference).all()
        err=np.abs(value.astype(np.float64)-reference.astype(np.float64))
        bound=16.*np.spacing(np.abs(reference)).astype(np.float64)+1.e-12
        results[name]=dict(BYTE=value.tobytes()==reference.tobytes(),
                           outside_16ulp=int(np.count_nonzero(err>bound)),max_abs=float(err.max()))

    for name,keys in (('e2',('amount','index','delta')),('e3',('temperature','water'))):
        args=tuple(data[k] for k in keys)
        fn='emissivity_'+name
        check(name+'_literal',getattr(literal,fn)(*args),data[name])
        check(name+'_compiled',jax.jit(getattr(port,fn))(*map(jnp.asarray,args)),data[name])
    args=tuple(data[k] for k in ('cloud','bottom','top'))+(data['number'][:,0],)
    values=[literal.cloud_overlap(*(x[i] for x in args)) for i in range(9)]
    check('cloud_literal',np.asarray(values),data['overlap'])
    check('cloud_compiled',jax.jit(jax.vmap(port.cloud_overlap))(*map(jnp.asarray,args)),data['overlap'])
    report=dict(results=results,source_sha256=hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        scope='64x2 E2SPEC;64x17 E3V88;CLO89 cloudcount0..8;synthetic routine stress, not forecast parity')
    output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),**data)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.native,a.output)
