"""Literal original-libm full-column gate versus pristine MP_NTU output."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from literal_numpy import LiteralNumpy
from gpuwrf.physics.ntu.column import step,PROGNOSTICS,SURFACE


def main():
    p=argparse.ArgumentParser();p.add_argument('fixtures',type=Path);p.add_argument('out',type=Path);p.add_argument('--kind',choices=('r4','r8'),default='r4');a=p.parse_args()
    file=a.fixtures/f'columns_{a.kind}.npz';meta=json.loads((a.fixtures/f'columns_{a.kind}.json').read_text())
    assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['npz_sha256']
    data=np.load(file);xp=LiteralNumpy();ncases=len(meta['cases'])
    fields={name:xp.asarray(np.stack([data[f'c{i:02}_s0_{name}'] for i in range(1,ncases+1)])) for name in ['th']+PROGNOSTICS}
    fields.update({name:xp.asarray(np.concatenate([data[f'c{i:02}_s0_{name}'] for i in range(1,ncases+1)])) for name in SURFACE})
    fields.update({name:xp.asarray(np.stack([data[f'c{i:02}_input_{name}'] for i in range(1,ncases+1)])) for name in ('p','pii','dz','w')})
    report={};different=0
    with np.errstate(all='ignore'):
        for stage in (1,2):
            outputs=step(fields,np.dtype(fields['th'].dtype).type(18),xp=xp)
            scores={}
            for name,actual in outputs.items():
                truth=(np.concatenate if name in SURFACE else np.stack)([data[f'c{i:02}_s{stage}_{name}'] for i in range(1,ncases+1)])
                actual=np.asarray(actual)
                assert np.isfinite(actual).all() and np.isfinite(truth).all(),(stage,name,'nonfinite')
                bad=actual.view('uint32' if actual.dtype.itemsize==4 else 'uint64')!=truth.view('uint32' if truth.dtype.itemsize==4 else 'uint64')
                different+=int(bad.sum());scores[name]={'different':int(bad.sum()),'max_abs':float(np.max(np.abs(actual-truth))),'bad_indices':np.argwhere(bad).tolist()}
            report[str(stage)]=scores;fields.update(outputs)
            print(stage,{k:v['different'] for k,v in scores.items() if v['different']},flush=True)
            a.out.write_text(json.dumps({'kind':a.kind,'different':different,'stages':report},indent=2)+'\n')
    assert different==0,'literal NTU column differs from original WRF'


if __name__=='__main__':main()
