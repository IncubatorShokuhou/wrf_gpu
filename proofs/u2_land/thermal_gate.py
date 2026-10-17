"""REAL4 source BYTE and runtime-compiled precision, independent original WRF bounds."""
import importlib.util,json,os,sys
from pathlib import Path
TREE,OUT=map(Path,sys.argv[1:3]);sys.path[:0]=[str(TREE/"src"),str(Path(__file__).parent)]
os.environ["GPUWRF_NOAHMP_NATIVE_REAL"]="1"
import jax
import jax.numpy as jnp
import numpy as np
from literal_real32 import NumpyReal,source_backend
spec=importlib.util.spec_from_file_location("thermal_tests",TREE/"tests/test_u2_thermal_options.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
assert jax.devices()[0].platform=="cpu"
report=dict(platform="cpu",results={})
for name in ("freezing","thermal"):
    codes=[(2,1)] if name=="freezing" else [(t,s) for t in (1,2) for s in (1,2,3)]
    literal=[];compiled=[];expected=[];low=[];high=[]
    for tbot,stc in codes:
        with source_backend():out,cases,fx=mod.matrix(name,tbot,stc,NumpyReal(),literal=True)
        literal.extend(out)
        actual,_,_=mod.matrix(name,tbot,stc)
        compiled.extend(actual)
        expected.extend(c["expected"] for c in cases);low.extend(c["low"] for c in cases);high.extend(c["high"] for c in cases)
    literal,compiled,expected,low,high=(np.asarray(v,np.float32) for v in (literal,compiled,expected,low,high))
    assert np.isfinite(literal).all() and np.isfinite(compiled).all()
    nominal=expected[:,:literal.shape[1]]
    same=literal.view(np.uint32)==nominal.view(np.uint32)
    excess=np.maximum(np.maximum(low.astype(float)-compiled,compiled.astype(float)-high),0.)
    report["results"][name]=dict(literal_values=int(same.size),literal_byte=int(same.sum()),literal_differences=np.argwhere(~same).tolist(),
        compiled_values=int(compiled.size),compiled_within=int((excess==0).sum()),max_excess=float(excess.max()),
        max_abs=float(np.abs(compiled.astype(float)-expected).max()),excess_indices=np.argwhere(excess>0).tolist())
    OUT.write_text(json.dumps(report,indent=2)+"\n");print(name,report["results"][name],flush=True)
