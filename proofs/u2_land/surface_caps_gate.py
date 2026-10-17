"""Original active cap blocks: literal REAL4 bytes and frozen input-ULP bounds."""
import importlib.util,json,sys
from pathlib import Path
tree,out=map(Path,sys.argv[1:3]);sys.path[:0]=[str(tree/'src'),str(Path(__file__).parent)]
import jax
import numpy as np
from literal_real32 import NumpyReal
assert jax.devices()[0].platform=='cpu'
spec=importlib.util.spec_from_file_location('surface_cap_tests',tree/'tests/test_u2_surface_caps.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
literal,fx=module.matrix(NumpyReal());compiled,_=module.matrix()
expected,low,high=(np.asarray([c[k] for c in fx['cases']],np.float32) for k in ('expected','low','high'))
same=literal.view(np.uint32)==expected.view(np.uint32)
assert literal.dtype==compiled.dtype==np.float32 and np.isfinite(literal).all() and np.isfinite(compiled).all()
excess=np.maximum(np.maximum(low.astype(float)-compiled,compiled.astype(float)-high),0.)
report=dict(platform='cpu',values=int(same.size),literal_byte=int(same.sum()),literal_differences=np.argwhere(~same).tolist(),
    compiled_within=int((excess==0).sum()),max_excess=float(excess.max()),max_abs=float(np.abs(compiled.astype(float)-expected).max()),active_caps=fx['active_caps'])
out.write_text(json.dumps(report,indent=2)+'\n');print(report)
assert same.all(),report
