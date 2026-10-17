"""CPU-side: compile the native CAM-MG TTIR with XLA's OWN Triton pipeline (the
cuda PJRT plugin's compile_triton_to_asm; the same compiler the GPU custom call
uses) WITHOUT a visible GPU (CUDA_VISIBLE_DEVICES=''). Returns PTX or the
pipeline error. usage: xla_triton_cpu.py TREE OUTDIR VARIANT [num_stages]"""
import hashlib,io,json,os,sys,time
from pathlib import Path
import numpy as np
tree=Path(sys.argv[1]);out=Path(sys.argv[2]);variant=sys.argv[3]
stages=int(sys.argv[4]) if len(sys.argv)>4 else 3
assert os.environ.get('JAX_PLATFORMS')=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES','x')==''
sys.path[:0]=[str(tree/'src'),str(tree/'proofs/t_cammgws')]
import jax,jax.numpy as jnp
from jax._src.pallas.triton import lowering
from variants import install
patch=install(variant)
from gpuwrf.kernels import phys_cammg
S=phys_cammg.SURFACE_FIELDS
out.mkdir(parents=True,exist_ok=True)
with np.load(tree/'proofs/s2_cammg/fixtures/public.npz') as z:
    f={k.split('_input_',1)[1]:jnp.asarray(z[k]).reshape((1,) if k.split('_input_',1)[1] in S else (1,25)) for k in z.files if k.startswith('c03_s1_input_')}
with jax.enable_x64(False):
    pj=jax.make_jaxpr(lambda f,dt,it:phys_cammg.full_column(f,dt,it))(f,jnp.float32(18),jnp.int32(1)).jaxpr
p=next(e for e in pj.eqns if e.primitive.name=='pallas_call').params
res=lowering.lower_jaxpr_to_triton_module(p['jaxpr'],p['grid_mapping'],'cuda',120)
buf=io.BytesIO();res.module.operation.write_bytecode(buf);ir=buf.getvalue()
report={'variant':patch,'num_stages':stages,'ttir_bytecode_sha256':hashlib.sha256(ir).hexdigest(),'ttir_bytes':len(ir)}
import jax_plugins.xla_cuda13 as plugin
from jax._src import xla_bridge as xb
from jax._src.lib import triton
t=time.monotonic()
if not triton.has_compilation_handler('CUDA'):
    plugin.initialize()
report['handler']=triton.has_compilation_handler('CUDA');report['init_s']=time.monotonic()-t
t=time.monotonic()
try:
    r=triton.compile('cuda',ir,'12.0',num_warps=4,num_ctas=1,num_stages=stages)
    report.update(status='COMPILED',compile_s=time.monotonic()-t,smem_bytes=r.smem_bytes,ptx_bytes=len(r.asm),
                  ptx_sha256=hashlib.sha256(r.asm.encode()).hexdigest())
    (out/'kernel.ptx').write_text(r.asm)
except Exception as e:
    report.update(status='FAILED',compile_s=time.monotonic()-t,error=repr(e)[-3000:])
(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='variant'})[:3000],flush=True)
os._exit(0)
