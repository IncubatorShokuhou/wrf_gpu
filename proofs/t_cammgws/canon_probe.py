"""CPU-only: does MLIR canonicalize/CSE (as run inside Triton's pipeline) merge the
native CAM-MG scf.if ops? Lowers TTIR (sm_120) per variant, runs canonicalize+cse
with the TTIR context, counts scf.if/scf.while before/after. Diagnostic only."""
import json,re,sys,time
from pathlib import Path
import numpy as np
tree=Path(sys.argv[1]);out=Path(sys.argv[2]);variant=sys.argv[3]
sys.path[:0]=[str(tree/'src'),str(tree/'proofs/t_cammgws')]
import jax,jax.numpy as jnp
from jax._src.pallas.triton import lowering
from jaxlib.mlir import passmanager
from variants import install
patch=install(variant)
from gpuwrf.kernels import phys_cammg
S=phys_cammg.SURFACE_FIELDS
with np.load(tree/'proofs/s2_cammg/fixtures/public.npz') as z:
    f={k.split('_input_',1)[1]:jnp.asarray(z[k]).reshape((1,) if k.split('_input_',1)[1] in S else (1,25)) for k in z.files if k.startswith('c01_s1_input_')}
with jax.enable_x64(False):
    pj=jax.make_jaxpr(lambda f,dt,it:phys_cammg.full_column(f,dt,it))(f,jnp.float32(18),jnp.int32(1)).jaxpr
p=next(e for e in pj.eqns if e.primitive.name=='pallas_call').params
res=lowering.lower_jaxpr_to_triton_module(p['jaxpr'],p['grid_mapping'],'cuda',120)
m=res.module
count=lambda t:{op:len(re.findall(r'\b'+re.escape(op)+r'\b',t)) for op in ('scf.if','scf.while','scf.for','scf.yield')}
before=count(str(m))
r={'variant':patch,'before':before}
with m.context:
    for name,pipe in (('canonicalize','builtin.module(canonicalize)'),('canonicalize_cse','builtin.module(canonicalize,cse,canonicalize)')):
        t=time.monotonic()
        try:
            pm=passmanager.PassManager.parse(pipe);pm.run(m.operation);r[name]=count(str(m));r[name+'_s']=time.monotonic()-t
        except Exception as e:r[name]=repr(e)[:300]
out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
