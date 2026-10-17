"""First-divergence diagnostic vs private original WRF stage records."""
import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np

from gpuwrf.physics.p3.column import step

NAMES = ('th','qv','qc','nc','qr','nr','t','rho','qvs','qvi','sup','supi',
         'qitot','nitot','qirim','birim','diag_effc','diag_effi','diag_di','diag_vmi','diag_rhoi')
CATS = {'qitot','nitot','qirim','birim','diag_effi','diag_di','diag_vmi','diag_rhoi'}


def records(path):
    content=path.read_bytes();pos=0;result={}
    while pos<len(content):
        tag,it,i,nc,nz=np.frombuffer(content,dtype=np.int32,count=5,offset=pos)
        pos+=20
        arrays={}
        for name in NAMES:
            shape=(nz,nc) if name in CATS else (nz,)
            count=int(np.prod(shape))
            arrays[name]=np.frombuffer(content,dtype=np.float32,count=count,offset=pos).reshape(shape,order='F')
            pos+=4*count
        result[(int(it),int(i))]=arrays
    assert pos==len(content)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--oracle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mp',type=int,default=50);p.add_argument('--compiled',action='store_true')
    a=p.parse_args();assert jax.devices()[0].platform=='cpu'
    with np.load(Path(__file__).with_name('fixtures')/f'columns_mp{a.mp}.npz',allow_pickle=False) as f:
        data={k:f[k] for k in f.files}
    fields={k:v[...,0] for k,v in data.items() if v.ndim==3}
    if a.mp!=52:
        for key in ('qi2','ni2','qir2','bir2','vm2','di2','rhoi2'):fields.pop(key,None)
    if a.mp!=53:fields.pop('qzi',None)
    t=time.monotonic()
    with jax.disable_jit(not a.compiled):
        out,stages=step(fields,18.,1,mp=a.mp,capture_stages=True)
        jax.block_until_ready(out)
    report={'mp':a.mp,'compiled':a.compiled,'elapsed_s':time.monotonic()-t,'stages':{}}
    saved_arrays = {}
    for tag,values in stages.items():
        if tag.startswith('r'):
            from rate_support import records as rate_records
            ref=rate_records(a.oracle/'rates.bin',int(tag[1:]))
        else:
            ref=records(a.oracle/f'stage_{tag}.bin')
        scores={}
        saved_arrays[f'{tag}_column_indices'] = np.array([key[1:] for key in ref if key[0]==1], dtype=np.int32)
        for name,value in values.items():
            value=np.asarray(value)
            actual=np.stack([value[key[1]-1,key[2]-1] if tag.startswith('r') else value[key[1]-1]
                             for key in ref if key[0]==1])
            expected=np.stack([v[name] for key,v in ref.items() if key[0]==1])
            assert np.isfinite(actual).all() and np.isfinite(expected).all(), (tag,name,'nonfinite')
            err=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
            idx=np.unravel_index(np.argmax(err),err.shape)
            scores[name]={'max_abs':float(err.max()),'reference':float(expected[idx]),'candidate':float(actual[idx]),
                          'index':list(map(int,idx)),'different':int((actual!=expected).sum())}
            saved_arrays[f'{tag}_{name}_candidate'] = actual
            saved_arrays[f'{tag}_{name}_reference'] = expected
        report['stages'][tag]=scores
        print(tag,{k:v for k,v in scores.items() if v['max_abs']>1.e-10},flush=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.out.with_suffix('.npz'), **saved_arrays)


if __name__=='__main__':main()
