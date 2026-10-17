"""Independent full physical-field source compiler envelope, no candidate fit."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def read(path):
    with path.open('rb') as f:
        nc,nz=np.fromfile(f,np.int32,2)
        rates=np.fromfile(f,np.float32,nc*nz*3).reshape((nc,nz,3),order='F')
        flux=np.fromfile(f,np.float32,nc*14).reshape((nc,14),order='F')
        assert not f.read(1) and np.isfinite(rates).all() and np.isfinite(flux).all()
    names=('total_rate','lw_rate','sw_rate')
    out={k:rates[:,:,i] for i,k in enumerate(names)}
    out.update({k:flux[:,i] for k,i in dict(glw=0,gsw=1,swdn=2,sw_up_top=4,lw_up_top=5,direct=6,diffuse=7,dni=8).items()})
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('nofma','fma','original','candidate','out'):ap.add_argument(k,type=Path)
    a=ap.parse_args();n,f,o=map(read,(a.nofma,a.fma,a.original))
    assert all(n[k].tobytes()==o[k].tobytes() for k in n),'No-FMA full replay not BYTE'
    envelope={k:float(np.max(np.abs(f[k].astype(np.float64)-n[k].astype(np.float64)))) for k in n}
    with np.load(a.candidate,allow_pickle=False) as c:
        assert np.array_equal(c['columns'],np.arange(40,dtype=np.int32))
        fields={}
        for k in n:
            value=c[k];assert value.dtype==np.float32 and value.shape==n[k].shape and np.isfinite(value).all()
            err=np.abs(value.astype(np.float64)-n[k].astype(np.float64))
            original_allowance=16.*np.spacing(np.abs(n[k])).astype(np.float64)+(1.e-11 if k.endswith('_rate') else 3.e-6)
            bound=envelope[k]+original_allowance
            fields[k]=dict(source_fma_envelope=envelope[k],max_candidate_error=float(err.max()),outside=int(np.count_nonzero(err>bound)))
    a.out.write_text(json.dumps(dict(noFMA_BYTE_original=True,scope='same40inputs; source field-wide Linf compiler envelope plus original allowance',fields=fields,
        all_pass=all(v['outside']==0 for v in fields.values()),hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.nofma,a.fma,a.original,a.candidate)}),indent=2)+'\n')
