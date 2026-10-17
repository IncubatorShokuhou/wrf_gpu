"""Original conversion corpus including underflow: test both execution modes."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.p3.qzi import decode,encode


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert jax.devices()[0].platform=='cpu'
    with np.load(a.fixture) as f:x,ref=f['inputs'],f['qzi']
    def call(x):return jnp.stack((decode(x[:,15],x[:,17]),encode(x[:,16],x[:,17])),axis=-1)
    report={}
    for mode in ('eager','jit'):
        out=np.asarray((jax.jit(call) if mode=='jit' else call)(jnp.asarray(x)))
        report[mode]={'BYTE':out.tobytes()==ref.tobytes(),'different':int(np.count_nonzero(out!=ref)),
            'max_abs':float(np.abs(out.astype(np.float64)-ref).max()),'finite':bool(np.isfinite(out).all())}
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
