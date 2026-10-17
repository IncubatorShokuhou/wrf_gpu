"""Compare literal CAM-MG helpers to the independent original WRF corpus."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('out',type=Path)
    args=ap.parse_args();sys.path.insert(0,str(args.tree/'src'))
    from compiler_math import compiler_math as xp
    from gpuwrf.physics.cammg import cloud
    from gpuwrf.physics.cammg.gamma import gamma
    from gpuwrf.physics.bl_camuw import EPSILO,LATVAP,LATICE,RH2O,CPAIR,TMELT
    globals=dict(epsqs=np.float64(EPSILO),tmelt=np.float64(TMELT),hlatv=np.float64(LATVAP),
                 hlatf=np.float64(LATICE),rgasv=np.float64(RH2O),cp=np.float64(CPAIR))
    with np.load(args.tree/'proofs/s2_cammg/fixtures/helpers.npz') as z:bank={k:z[k] for k in z.files}
    result={k:[] for k in ['cloud','saturation','gamma']}
    with np.errstate(all='ignore'):
        for row,x in zip(bank['inputs'],bank['gamma_x'],strict=True):
            t,p,land,snow,qv,qc,qi=row
            f=cloud.cal_cldfra_mp_1d(np.float64(0),np.float64(0),qv,qc,qi,t,p,land,snow,_globals=globals,xp=xp)
            result['cloud'].append(f)
            es,qs,gam=cloud.vqsatd_water(np.array([t]),np.array([p]),np.zeros(1),np.zeros(1),np.zeros(1),1,_globals=globals,xp=xp)
            result['saturation'].append([cloud.polysvp(t,0,xp=xp),cloud.polysvp(t,1,xp=xp),es[0],qs[0],gam[0]])
            result['gamma'].append(gamma(x,xp=xp))
    receipt={}
    for name,values in result.items():
        got=np.asarray(values);ref=bank[name]
        assert got.shape==ref.shape,(name,got.shape,ref.shape)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)),name
        bad=got!=ref
        receipt[name]={'BYTE':got.tobytes()==ref.tobytes(),'different':int(np.count_nonzero(bad)),
                       'max_abs':float(np.max(np.abs(got-ref))),
                       'first':np.argwhere(bad)[:8].tolist(),
                       'values':int(got.size)}
    args.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    assert all(v['BYTE'] for v in receipt.values()),'literal original-WRF BYTE gate failed'


if __name__=='__main__':main()
