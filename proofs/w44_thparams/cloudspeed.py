"""Isolate the measured AV_C default-byte failure before choosing a repair."""
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
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
assert jax.devices()[0].platform=="cpu"
z=dict(np.load(args.fixture,allow_pickle=False))
idx=[z["names"].tolist().index(n) for n in json.loads(args.selection.read_text())["selected_columns"]]
f=lambda v:jnp.asarray(z["in_"+v][idx],jnp.float32)
temp,p,qv=f("th")*f("pii"),f("p"),f("qv")
s=tc.ThompsonColumnState(qv=qv,qc=f("qc"),qr=f("qr"),qi=f("qi"),qs=f("qs"),qg=f("qg"),Ni=f("ni"),Nr=f("nr"),
                         T=temp,p=p,rho=tc.density_from_pressure_temperature(p,temp,qv),dz=f("dz8w"),w=f("w"))


def controlled(state, av_c):
    env=dict(vars(tc),AV_C=av_c)
    fn=tc._cloud_water_fall_speed
    return FunctionType(fn.__code__,env,fn.__name__,fn.__defaults__)(state)


args.out.mkdir(parents=True,exist_ok=True)
outputs=[]
for label,fn,inputs in (("static",tc._cloud_water_fall_speed,(s,)),("dynamic",controlled,(s,jnp.float32(tc.AV_C)))):
    compiled=jax.jit(fn).lower(*inputs).compile()
    hlo=compiled.as_text()
    (args.out/(label+".hlo")).write_text(hlo)
    out=np.asarray(compiled(*inputs))
    assert np.isfinite(out).all()
    outputs.append(out)
receipt=dict(platform="cpu",gpu_executed=False,cells=int(np.count_nonzero(outputs[0]!=outputs[1])),
             maxabs=float(np.max(np.abs(outputs[0]-outputs[1]))),
             input_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest())
(args.out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
