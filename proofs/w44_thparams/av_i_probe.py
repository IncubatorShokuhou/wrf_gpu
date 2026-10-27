"""Localize AV_I default differences in unchanged ice fall-speed equations.

The pre-fill active-cell equations are original _fall_speeds code objects;
only the private AV_I binding becomes a REAL operand. Compare eager and
compiled results and preserve optimized HLO. This is causal/regression
evidence, not a replacement for the original WRF changed-value oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import FunctionType

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
from gpuwrf.physics import thompson_column as tc

parser=argparse.ArgumentParser()
parser.add_argument("--fixture",type=Path,required=True)
parser.add_argument("--selection",type=Path,required=True)
parser.add_argument("--full-output",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
assert jax.devices()[0].platform=="cpu"
z=dict(np.load(args.fixture,allow_pickle=False))
selected=json.loads(args.selection.read_text())
assert selected["fixture_sha256"]==hashlib.sha256(args.fixture.read_bytes()).hexdigest()
idx=[z["names"].tolist().index(n) for n in selected["selected_columns"]]
f=lambda v:jnp.asarray(z["in_"+v][idx],jnp.float32)
temp,p,qv=f("th")*f("pii"),f("p"),f("qv")
state=tc.ThompsonColumnState(qv=qv,qc=f("qc"),qr=f("qr"),qi=f("qi"),qs=f("qs"),qg=f("qg"),Ni=f("ni"),Nr=f("nr"),
                             T=temp,p=p,rho=tc.density_from_pressure_temperature(p,temp,qv),dz=f("dz8w"),w=f("w"))


def program(state,av_i):
    fn=tc._fall_speeds
    if av_i is not None:
        clone=FunctionType(fn.__code__,{**fn.__globals__,"AV_I":av_i},fn.__name__,fn.__defaults__,fn.__closure__)
        clone.__kwdefaults__=fn.__kwdefaults__
        fn=clone
    return tuple(value.astype(jnp.float32) for value in fn(state,fill_down=lambda value,active:value)[2:4])


args.out.mkdir(parents=True,exist_ok=True)
values=(None,jnp.float32(1493.9))
results={}
for mode in ("eager","compiled"):
    outputs=[]
    for label,value in zip(("static","runtime"),values,strict=True):
        if mode=="compiled":
            fn=jax.jit(program).lower(state,value).compile()
            (args.out/(label+".hlo")).write_text(fn.as_text())
            result=fn(state,value)
        else:
            result=program(state,value)
        arrays=[np.asarray(a) for a in result]
        assert all(np.isfinite(a).all() for a in arrays)
        outputs.append(arrays)
    results[mode]={}
    for name,a,b in zip(("mass","number"),*outputs,strict=True):
        ulp=np.abs(a.view(np.uint32).astype(np.int64)-b.view(np.uint32).astype(np.int64))
        results[mode][name]=dict(byte_exact=a.tobytes()==b.tobytes(),changed_cells=int(np.count_nonzero(ulp)),
                                max_real_ulp=int(ulp.max()),max_abs=float(np.max(np.abs(a-b))))

old=dict(np.load(args.full_output/"off.npz",allow_pickle=False))
new=dict(np.load(args.full_output/"default.npz",allow_pickle=False))
a,b=old["17"],new["17"]
assert a.dtype==b.dtype==np.float64 and np.isfinite(a).all() and np.isfinite(b).all()
widened=all(np.array_equal(x,x.astype(np.float32).astype(np.float64)) for x in (a,b))
assert widened,"ice precipitation is not exact widened REAL32"
real_ulp=np.abs(a.astype(np.float32).view(np.uint32).astype(np.int64)-b.astype(np.float32).view(np.uint32).astype(np.int64))
storage_ulp=np.abs(a.view(np.uint64).astype(np.int64)-b.view(np.uint64).astype(np.int64))
receipt=dict(platform="cpu",gpu_executed=False,default_real=float(np.float32(1493.9)),
             default_bits=int(np.float32(1493.9).view(np.uint32)),changed_real=float(np.float32(2987.8)),
             input_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),fall_speed=results,
             ice_precipitation=dict(exact_widened_real=widened,max_real_ulp=int(real_ulp.max()),
                                    max_storage_ulp=int(storage_ulp.max()),max_abs=float(np.max(np.abs(a-b)))),
             source="phys_thompson_prep._sed_registers carries ppt REAL32; full_column output buffer is f64")
(args.out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
