"""Runoff family literal and independently bounded compiled gates."""
import importlib.util,json,os,sys
from pathlib import Path
TREE,OUT=map(Path,sys.argv[1:3]);sys.path[:0]=[str(TREE/'src'),str(Path(__file__).parent)]
os.environ.update(GPUWRF_NOAHMP_NATIVE_REAL='1',GPUWRF_NOAHMP_COLUMN_KERNELS='0')
import jax
import numpy as np
from literal_real32 import NumpyReal,source_backend
spec=importlib.util.spec_from_file_location('runoff_tests',TREE/'tests/test_u2_runoff_options.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
assert jax.devices()[0].platform=='cpu'
report=dict(platform='cpu',results={})
for run,infdv in ((6,0),(7,0),(8,1),(8,2),(8,3)):
    with source_backend():literal,cases,fx=mod.matrix(run,infdv,NumpyReal())
    compiled,_,_=mod.matrix(run,infdv)
    expected,low,high=(np.asarray([c[k] for c in cases],np.float32) for k in ('expected','low','high'))
    assert literal.dtype==compiled.dtype==np.float32 and np.isfinite(literal).all() and np.isfinite(compiled).all()
    same=literal.view(np.uint32)==expected.view(np.uint32)
    excess=np.maximum(np.maximum(low.astype(float)-compiled,compiled.astype(float)-high),0.)
    report['results'][f'{run}/{infdv}']=dict(values=int(same.size),literal_byte=int(same.sum()),literal_differences=np.argwhere(~same).tolist(),
        compiled_within=int((excess==0).sum()),max_excess=float(excess.max()),max_abs=float(np.abs(compiled.astype(float)-expected).max()),excess_indices=np.argwhere(excess>0).tolist())
    OUT.write_text(json.dumps(report,indent=2)+'\n');print(run,infdv,report['results'][f'{run}/{infdv}'],flush=True)
