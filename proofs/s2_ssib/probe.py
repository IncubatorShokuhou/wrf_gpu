#!/usr/bin/env python3
"""CPU shape/compile/oracle diagnostic for the literal SSiB transcription."""
from __future__ import annotations
import argparse
import inspect
import json
import time
from pathlib import Path
import sys
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['GPUWRF_JAX_CACHE']='0'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/v025'))
import cpu_guard
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from gpuwrf.physics._ssib_generated import _ssib
from audit_oracle import fields

INPUT_CARRY={'www1','www2','www3','tc','tgs','td','snoa','snob','ta','bedo','isnow'}
OUTPUT_ALIASES={'xhsflx':'HFX','elaten':'LH','ghtflx':'GRDFLX','xhlflx':'QFX','tgeff':'TGEFF'}
INTEGER={'ix','jx','itime','ivgtyp','isnow','sw_physics','mminlu'}
ARGS=tuple(inspect.signature(_ssib).parameters)

def operands(data,col=1):
    out=[]
    for name in ARGS:
        if name=='ix': value=col
        elif name=='jx': value=1
        elif name=='mminlu': value=0
        elif name=='roff': value=0.
        elif name.startswith('salb'): value=.1 if name in ('salb11','salb12') else .2
        elif name.startswith('radfrac'): value={'radfrac11':.3,'radfrac12':.1,'radfrac21':.4,'radfrac22':.2}[name]
        elif name in INPUT_CARRY: value=data[f'{name.upper()}_IN[{col}]']
        elif f'{name.upper()}[{col}]' in data and name not in OUTPUT_ALIASES and name not in ('q2m','uv10') and not name.startswith('x'): value=data[f'{name.upper()}[{col}]']
        else: value=0.
        if name=='ustar': value=0.
        out.append(jnp.asarray(value,dtype=jnp.int32 if name in INTEGER else jnp.float32))
    return tuple(out)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reference',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--shape-only',action='store_true'); ap.add_argument('--case',type=int,default=1); args=ap.parse_args()
    data=fields((args.reference/f'case_{args.case}.stdout').read_text()); inputs=operands(data)
    start=time.monotonic(); abstract=jax.eval_shape(_ssib,*inputs)
    result={'backend':str(jax.devices()),'shape_seconds':time.monotonic()-start,'outputs':len(abstract),'f64_leaves':sum(a.dtype==jnp.float64 for a in abstract)}
    if not args.shape_only:
        fn=jax.jit(_ssib)
        start=time.monotonic(); compiled=fn.lower(*inputs).compile(); result['compile_seconds']=time.monotonic()-start
        start=time.monotonic(); actual=jax.block_until_ready(compiled(*inputs)); result['first_seconds']=time.monotonic()-start
        diffs=[]
        for name,value in zip(ARGS,actual):
            refname=OUTPUT_ALIASES.get(name,name.upper()); key=f'{refname}[1]'
            if key in data:
                got=float(value); expect=data[key]
                diffs.append({'field':name,'actual':got,'reference':expect,'abs':abs(got-expect)})
        result['differences']=diffs
        result['all_finite']=all(np.all(np.isfinite(np.asarray(x))) for x in actual)
        result['memory']=str(compiled.memory_analysis())
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()
