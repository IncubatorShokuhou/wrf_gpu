"""CPU-only: push the native CAM-MG TTIR (variant) through pip Triton's CUDA
pipeline (GPUTarget cuda/120) stage by stage; save every stage's IR. Diagnostic:
pip Triton may differ from the Triton bundled in XLA. No CUDA context created.
usage: triton_probe.py TREE OUTDIR VARIANT [num_stages]"""
import hashlib,json,os,re,resource,sys,time
from pathlib import Path
import numpy as np
tree=Path(sys.argv[1]);out=Path(sys.argv[2]);variant=sys.argv[3]
stages_opt=int(sys.argv[4]) if len(sys.argv)>4 else 3
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
ttir=out/'native.ttir';ttir.write_text(str(res.module))
report={'variant':patch,'num_stages':stages_opt,'stages':{},'ttir_bytes':ttir.stat().st_size}
def save():(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
save()
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
original=CUDABackend.add_stages
def instrument(self,stages,options,language):
    original(self,stages,options,language)
    for name,fn in list(stages.items()):
        def timed(src,metadata,fn=fn,name=name):
            t=time.monotonic();print('START',name,flush=True)
            result=fn(src,metadata)
            report['stages'][name]=time.monotonic()-t
            if isinstance(result,(str,bytes)) and name!='cubin':
                text=result if isinstance(result,str) else result.decode(errors='replace')
                (out/f'stage_{name}.txt').write_text(text)
            elif name!='cubin':(out/f'stage_{name}.txt').write_text(str(result))
            save();print('DONE',name,report['stages'][name],flush=True)
            return result
        stages[name]=timed
CUDABackend.add_stages=instrument
t=time.monotonic()
try:
    kernel=triton.compile(str(ttir),target=GPUTarget('cuda',120,32),options={'num_warps':4,'num_stages':stages_opt})
    report['status']='COMPILED';report['compile_s']=time.monotonic()-t
    (out/'native.cubin').write_bytes(kernel.asm['cubin'])
except Exception as e:
    report['status']='FAILED';report['error']=repr(e)[-2000:]
report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
save();print(json.dumps({k:v for k,v in report.items() if k!='variant'})[:2500],flush=True)
os._exit(0)
