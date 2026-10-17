"""Whole solver literal and input-ULP gates; report every option composition."""
import importlib.util,json,os,sys
from pathlib import Path
TREE,OUT=map(Path,sys.argv[1:3]);sys.path[:0]=[str(TREE/'src'),str(Path(__file__).parent)]
os.environ.update(GPUWRF_NOAHMP_NATIVE_REAL='1',GPUWRF_NOAHMP_COLUMN_KERNELS='0',GPUWRF_NOAHMP_LAYER_SELECT='0')
import jax
import numpy as np
from literal_real32 import NumpyReal,source_backend
spec=importlib.util.spec_from_file_location('advanced_water_tests',TREE/'tests/test_u2_advanced_water.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
assert jax.devices()[0].platform=='cpu'
fx=json.loads((TREE/'proofs/u2_land/advanced_water_oracle.json').read_text())
modes=sorted({tuple(c[k] for k in ('opt_run','opt_inf','opt_infdv','opt_tdrn','drain_layer_opt')) for c in fx['cases']})
if len(sys.argv)>3:modes=[mode for mode in modes if mode[3]==int(sys.argv[3])]
report=dict(platform='cpu',results={})
for mode in modes:
    with source_backend():literal,cases,fx=mod.matrix(mode,NumpyReal())
    compiled,_,_=mod.matrix(mode)
    expected,low,high=(np.asarray([c[k] for c in cases],np.float32) for k in ('expected','low','high'))
    assert literal.dtype==compiled.dtype==np.float32 and np.isfinite(literal).all() and np.isfinite(compiled).all()
    same=literal.view(np.uint32)==expected.view(np.uint32)
    excess=np.maximum(np.maximum(low.astype(float)-compiled,compiled.astype(float)-high),0.)
    row=dict(values=int(same.size),literal_byte=int(same.sum()),literal_differences=np.argwhere(~same).tolist(),
        compiled_within=int((excess==0).sum()),max_excess=float(excess.max()),max_abs=float(np.abs(compiled.astype(float)-expected).max()),excess_indices=np.argwhere(excess>0).tolist())
    report['results'][str(mode)]=row
    OUT.write_text(json.dumps(report,indent=2)+'\n');print(mode,row,flush=True)
