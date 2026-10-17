"""Outside-forecast source comparison of time-entry Smagorinsky XKHH."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

root, forecast, original, output = map(lambda p:Path(p).resolve(),sys.argv[1:])
sys.path[:0]=[str(root/"src"),str(root/"proofs/o1_nlbind")]
import cpu_safe
cpu_safe.install()
import jax
import numpy as np
from jax.experimental import layout,pallas
from gpuwrf.runtime import operational_mode as om
from gpuwrf.kernels.dyn_rk_fp32 import real_metrics
from gpuwrf.io.netcdf_lock import Dataset
assert jax.devices()[0].platform=="cpu"
layout.with_layout_constraint=lambda value,*args,**kw:value
original_call=pallas.pallas_call
def interpret(*args,**kwargs):
    kwargs["interpret"]=True
    kwargs.pop("compiler_params",None)
    return original_call(*args,**kwargs)
pallas.pallas_call=interpret
receipt=json.loads(forecast.read_text())
with gzip.open(receipt["frames"][0]["saved_carry"],"rb") as stream:
    carry,nl=pickle.load(stream)
nl=replace(nl,metrics=real_metrics(nl.metrics))
def coefficient(state,namelist,base):
    return om._diffopt1_dry_forward_tendencies(state,namelist,base_state=base,return_scalar_xkhh=True)[4]
actual=np.asarray(jax.jit(coefficient)(carry.state,nl,carry.base_state))
with Dataset(sorted(original.glob("wrfout_d01_*"))[1]) as ds:
    expected=np.asarray(ds.variables["XKHH"][0])
delta=actual.astype(np.float64)-expected.astype(np.float64)
where=tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(delta)),delta.shape))
spots={}
for index in ((0,19,17),(0,23,24)):
    spots[str(index)]={"port":float(actual[index]),"original":float(expected[index]),"delta":float(delta[index])}
result=dict(source=str(root),backend="cpu",scope="initial-state diffusivity producer vs original first-step field; no trajectory/GPU claim",
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    max_abs=float(np.max(np.abs(delta))),max_index=where,rmse=float(np.sqrt(np.mean(delta**2))),
    port_max=float(actual.max()),original_max=float(expected.max()),nonfinite=int((~np.isfinite(actual)).sum()),
    nonzero_port=int(np.count_nonzero(actual)),nonzero_original=int(np.count_nonzero(expected)),spots=spots)
output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
