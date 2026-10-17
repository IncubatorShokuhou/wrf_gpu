"""Byte gate for port routines versus independent original Fortran calls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from literal_numpy import LiteralNumpy
from gpuwrf.physics.ntu import scalar_generated as port


def main():
    p=argparse.ArgumentParser();p.add_argument('oracle',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();meta=json.loads((args.oracle/'manifest.json').read_text());data=np.load(args.oracle/'scalars.npz')
    assert hashlib.sha256((args.oracle/'scalars.npz').read_bytes()).hexdigest()==meta['npz_sha256']
    xp=LiteralNumpy();results={}
    with np.errstate(all='ignore'):
        for name,params in meta['signatures'].items():
            active=data[name+'_active'];values=data[name+'_input'][active]
            expected=data[name+'_output'][active]
            outputs=getattr(port,name)({k:xp.asarray(values[:,i]) for i,k in enumerate(params)},xp=xp)
            scores={}
            for i,k in enumerate(params):
                actual=np.asarray(outputs[k]);truth=expected[:,i]
                if not (np.isfinite(actual).all() and np.isfinite(truth).all()):
                    scores[k]={'nonfinite_actual':np.flatnonzero(~np.isfinite(actual)).tolist(),
                               'nonfinite_expected':np.flatnonzero(~np.isfinite(truth)).tolist(),'different':len(actual)}
                    continue
                changed=actual.view('uint32' if actual.dtype.itemsize==4 else 'uint64')!=truth.view('uint32' if truth.dtype.itemsize==4 else 'uint64')
                scores[k]={'different':int(changed.sum()),'max_abs':float(np.max(np.abs(actual-truth))),
                           'rows':np.flatnonzero(changed).tolist(),
                           'actual':actual[changed].tolist(),'expected':truth[changed].tolist()}
            results[name]=scores
            print(name, {k:s['different'] for k,s in scores.items() if s['different']},flush=True)
    result={'kind':meta['kind'],'BYTE':all(s['different']==0 for routine in results.values() for s in routine.values()),'routines':results}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    assert result['BYTE'],'literal NTU scalar routines differ from original WRF'


if __name__=='__main__':main()
