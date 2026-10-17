"""CPU-only lowering preflight for the exact NWP/helicity GPU programs."""
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from jax._src.pallas.triton import lowering

from gpuwrf.diagnostics.nwp import NwpDiagnostics,update_helicity,update_nwp

ROOT=Path(__file__).resolve().parents[2]
fixture=json.loads((ROOT/"proofs/u2_diag/nwp_oracle.json").read_text())
assert jax.devices()[0].platform == "cpu"
inputs={name:jnp.asarray(value,jnp.float32) for name,value in fixture["inputs"].items()}
state=NwpDiagnostics(*(jnp.asarray(a,jnp.float32) for a in fixture["initial"]))
constants=dict(zip(("rdx","rdy","cf1","cf2","cf3"),inputs["constants"]))
heli_names=("u","v","w","ph","phb","msfux","msfvy","terrain","dn","dnw","fnm","fnp","zx","zy","rdzw")
nwp_names=("u10","v10","w","p8w","znw","qg","rho","ph","phb","refl_10cm")
functions=(lambda d:update_helicity(d,**{name:inputs[name] for name in heli_names},**constants,interpret=False),
           lambda d:update_nwp(d,**{name:inputs[name] for name in nwp_names},itimestep=jnp.int32(2),dt=20.,history_interval=1,interpret=False))
results=[]
for fn in functions:
    jp=jax.make_jaxpr(fn)(state).jaxpr
    for equation in jp.eqns:
        if equation.primitive.name != "pallas_call":
            continue
        params=equation.params
        result=lowering.lower_jaxpr_to_triton_module(params["jaxpr"],params["grid_mapping"],"cuda",120)
        asm=str(result.module)
        name=str(params.get("name_and_src_info",len(results)))
        results.append(dict(kernel=name,lowering="PASS",IR_characters=len(asm)))
target=Path(sys.argv[1])
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(dict(backend="cpu",GPU_execution=False,target="cuda/sm120",results=results),indent=2)+"\n")
print(target.read_text())
