"""Causal intervention: freeze only av_r operands in the same native call."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
from gpuwrf.physics import thompson_column as tc
from gpuwrf.physics.thompson_parameters import validate_parameters
from gpuwrf.kernels import phys_thompson_full as full

parser=argparse.ArgumentParser()
parser.add_argument("--fixture",type=Path,required=True)
parser.add_argument("--selection",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
assert jax.devices()[0].platform=="cpu"
z=dict(np.load(args.fixture,allow_pickle=False))
assert all(np.isfinite(v).all() for k,v in z.items() if k.startswith(("in_","wrf_")))
names=json.loads(args.selection.read_text())["selected_columns"]
idx=[z["names"].tolist().index(n) for n in names]
f=lambda name:jnp.asarray(z["in_"+name][idx],jnp.float32)
t,p,qv=f("th")*f("pii"),f("p"),f("qv")
zero=jnp.zeros_like(qv)
state=tc.ThompsonColumnState(qv=qv,qc=f("qc"),qr=f("qr"),qi=f("qi"),qs=f("qs"),qg=f("qg"),
                           Ni=f("ni"),Nr=f("nr"),Ns=zero,Ng=zero,T=t,p=p,
                           rho=tc.density_from_pressure_temperature(p,t,qv),dz=f("dz8w"),w=f("w"))
parameters=validate_parameters({"av_r":4854.})
original=full._parameter_namespace


def frozen(*args,**kwargs):
    env=original(*args,**kwargs)
    # Exactly the default values from the original static namespace. No
    # equation, table, mask, state, or input fixture is replaced.
    env["AV_R"]=tc.AV_R
    for name in ("T1_QR_QC","T1_QR_QI","T2_QR_QI","T2_QR_EV"):
        env[name]=getattr(tc,name)
    env["WRF_REAL_CONSTANTS"]={**env["WRF_REAL_CONSTANTS"],
        **{name:tc.WRF_REAL_CONSTANTS[name] for name in ("T1_QR_QC","T1_QR_QI","T2_QR_QI")}}
    return env


args.out.mkdir(parents=True,exist_ok=True)
results={}
try:
    for label,params in (("off",None),("dynamic",parameters),("frozen",parameters)):
        full._parameter_namespace=frozen if label=="frozen" else original
        jax.clear_caches()  # Import-time/global bindings are captured in traces.
        fn=jax.jit(lambda s,ps:full.full_column(s,18.,interpret=True,**({} if ps is None else {"parameters":ps})))
        result=fn(state,params)
        arrays=[np.asarray(v).copy() for v in jax.tree.leaves(result)]
        assert all(np.isfinite(v).all() for v in arrays),label
        results[label]=arrays
        np.savez_compressed(args.out/(label+".npz"),**{str(i):v for i,v in enumerate(arrays)})
finally:
    full._parameter_namespace=original
    jax.clear_caches()
fields=(*tc.ThompsonColumnState.__slots__,"cloudw","graupel","ice","rain","snow")
rows={}
for label in ("dynamic","frozen"):
    rows[label]=[dict(field=field,byte_exact=a.tobytes()==b.tobytes(),changed=int(np.count_nonzero(a!=b)),
                      maxabs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
                 for field,a,b in zip(fields,results["off"],results[label],strict=True)]
receipt=dict(platform="cpu",gpu_executed=False,fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
             scope="constantness intervention/regression only; original-WRF changed-value gate is AR03",
             field_comparison=rows,restored_all_bytes=all(r["byte_exact"] for r in rows["frozen"]),
             dynamic_changed_fields=sum(not r["byte_exact"] for r in rows["dynamic"]))
(args.out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
assert receipt["restored_all_bytes"] and receipt["dynamic_changed_fields"]>0
