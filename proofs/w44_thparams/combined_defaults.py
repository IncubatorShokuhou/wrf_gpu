"""All-eight default class and a constant-only causal intervention."""
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
from gpuwrf.physics.thompson_parameters import PARAMETERS, validate_parameters
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
parameters=validate_parameters({name:row[0] for name,row in PARAMETERS.items()})
original=full._parameter_namespace


def frozen(*args,**kwargs):
    env=original(*args,**kwargs)
    # Exactly the default values from the original static namespace. No
    # equation, table, mask, state, or input fixture is replaced.
    aliases={"autoconversion_rate":"AU_RT", "autoconversion_threshold":"AU_THR0"}
    for name in PARAMETERS:
        key=aliases.get(name,name.upper())
        env[key]=getattr(tc,key)
    env["NU_C_MP8"]=tc.NU_C_MP8
    env["WRF_REAL_CONSTANTS"]=dict(tc.WRF_REAL_CONSTANTS)
    for name in ("T1_QR_QC","T1_QR_QI","T2_QR_QI","T2_QR_EV"):
        env[name]=getattr(tc,name)
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
from roundoff import FIELDS, compare, ulps
rows=[]
for field,a,b in zip(FIELDS,results["off"],results["dynamic"],strict=True):
    error=np.abs(a.astype(np.float64)-b.astype(np.float64))
    relative=np.divide(error,np.abs(a),out=np.zeros_like(error),where=a!=0)
    relative[(a==0)&(error!=0)]=np.inf
    exact32=(np.array_equal(a,a.astype(np.float32).astype(a.dtype)) and
             np.array_equal(b,b.astype(np.float32).astype(b.dtype)))
    rows.append(dict(field=field,storage_dtype=str(a.dtype),changed_cells=int(np.count_nonzero(a!=b)),
                     max_abs=float(error.max()),max_rel=float(relative.max()),
                     max_storage_ulp=int(ulps(a,b).max()),exact_REAL32_roundtrip=exact32,
                     max_REAL32_ulp=int(ulps(a.astype(np.float32),b.astype(np.float32)).max()) if exact32 else None))
restored=[a.tobytes()==b.tobytes() for a,b in zip(results["off"],results["frozen"],strict=True)]
receipt=dict(platform="cpu",gpu_executed=False,selected_columns=names,
             fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
             parameters={name:row[0] for name,row in PARAMETERS.items()},
             scope="All-eight low-level CPU qualification; experimental names remain refused by runtime binder/adapter",
             cause="Freeze only selected scalar constants and derived REAL prefixes in the same application; no equation, mask, input, table bytes or limiter changed",
             restored_all_bytes=all(restored),restored_leaves=sum(restored),output_leaves=len(restored),
             default_class=rows,generic_roundoff=compare(results["off"],results["dynamic"],nt_c=True),
             E100_floor="pending separate pristine ensemble; this probe does not set a tolerance")
(args.out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
assert receipt["restored_all_bytes"], "constant-only intervention did not close the combined class"
