"""CPU-only: write the native CAM-MG CUDA TTIR (sm_120) of a variant WITH debug
locations (for pass_bisect --track). usage: export_ttir.py TREE OUT.ttir VARIANT"""
import sys
from pathlib import Path
import numpy as np
tree=Path(sys.argv[1]);out=Path(sys.argv[2]);variant=sys.argv[3]
sys.path[:0]=[str(tree/'src'),str(tree/'proofs/t_cammgws')]
import jax,jax.numpy as jnp
from jax._src.pallas.triton import lowering
from variants import install
print(install(variant))
from gpuwrf.kernels import phys_cammg
S=phys_cammg.SURFACE_FIELDS
with np.load(tree/'proofs/s2_cammg/fixtures/public.npz') as z:
    f={k.split('_input_',1)[1]:jnp.asarray(z[k]).reshape((1,) if k.split('_input_',1)[1] in S else (1,25)) for k in z.files if k.startswith('c03_s1_input_')}
with jax.enable_x64(False):
    pj=jax.make_jaxpr(lambda f,dt,it:phys_cammg.full_column(f,dt,it))(f,jnp.float32(18),jnp.int32(1)).jaxpr
p=next(e for e in pj.eqns if e.primitive.name=='pallas_call').params
res=lowering.lower_jaxpr_to_triton_module(p['jaxpr'],p['grid_mapping'],'cuda',120)
out.write_text(res.module.operation.get_asm(enable_debug_info=True));print('wrote',out,out.stat().st_size)
