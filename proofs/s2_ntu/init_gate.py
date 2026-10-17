"""Literal and fast host initializer checks against original NTU_INIT."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from literal_numpy import LiteralNumpy
from gpuwrf.physics.ntu.initialize import aerosols


def main():
    p=argparse.ArgumentParser();p.add_argument('oracle',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    meta=json.loads((a.oracle/'manifest.json').read_text());file=a.oracle/'initialization.npz'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['npz_sha256']
    data=np.load(file);results={};different=0;hostmax=0
    for ccnty in meta['ccnty']:
        inputs={n:np.stack([data[f'cc{ccnty}_c{i:02}_{n}'] for i in range(1,meta['cases']+1)]) for n in ('p','alt','ph','phb','qv')}
        xland=np.concatenate([data[f'cc{ccnty}_c{i:02}_xland'] for i in range(1,meta['cases']+1)])
        outputs=aerosols(inputs['p'],inputs['alt'],inputs['ph'],inputs['phb'],inputs['qv'],xland,ccnty=ccnty,xp=LiteralNumpy())
        host=aerosols(inputs['p'],inputs['alt'],inputs['ph'],inputs['phb'],inputs['qv'],xland,ccnty=ccnty,xp=np)
        scores={}
        for n,actual in outputs.items():
            truth=np.stack([data[f'cc{ccnty}_c{i:02}_{n}'] for i in range(1,meta['cases']+1)])
            assert np.isfinite(actual).all() and np.isfinite(host[n]).all() and np.isfinite(truth).all()
            itype='uint32' if actual.dtype.itemsize==4 else 'uint64'
            bad=np.asarray(actual).view(itype)!=truth.view(itype);different+=int(bad.sum())
            ulp=np.abs(host[n].view(itype).astype(np.int64)-truth.view(itype).astype(np.int64))
            hostmax=max(hostmax,int(ulp.max()))
            scores[n]={'different':int(bad.sum()),'host_max_ulp':int(ulp.max()),'max_abs':float(np.max(np.abs(actual-truth)))}
        results[str(ccnty)]=scores
    result={'kind':meta['kind'],'literal_different':different,'BYTE':different==0,'host_max_ulp':hostmax,'ccnty':results}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert different==0,'NTU literal initialization differs from original WRF'


if __name__=='__main__':main()
