"""Lower actual Milbrandt column Pallas jaxpr to CUDA TTIR on CPU, without GPU use."""
from pathlib import Path
import argparse
import json
import time
import jax
import jax.numpy as jnp
from jax._src.pallas.triton import lowering
from gpuwrf.physics.milbrandt2mom import column,rounding,kernel


def main():
    p=argparse.ArgumentParser(); p.add_argument('out'); a=p.parse_args()
    assert jax.devices()[0].platform=='cpu'
    # Force native source selection ONLY while tracing, never execute this callable.
    column.cuda_native=rounding.cuda_native=kernel.pl.cuda_native=lambda:True
    shape=lambda s:jax.ShapeDtypeStruct(s,jnp.float32)
    args=((shape((44,2,3)),)*14,shape((44,2,3)),shape((2,3)),shape((45,2,3)),shape(()))
    start=time.monotonic()
    jp=jax.make_jaxpr(kernel.run_fields)(*args).jaxpr
    params=next(e.params for e in jp.eqns if e.primitive.name=='pallas_call')
    result=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
    ir=str(result.module)
    out=Path(a.out); out.with_suffix('.ttir').write_text(ir)
    receipt={'pass':True,'backend':'cpu','GPU_execution':False,'target':'cuda/sm120',
             'work':'actual level-major production ABI 6 columns x44 levels native RN32, all processes; no packed fields',
             'lower_s':time.monotonic()-start,'IR_chars':len(ir),'native_f64_diagnostics':ir.count('f64')}
    out.write_text(json.dumps(receipt,indent=2)+'\n'); print(receipt)

if __name__=='__main__': main()
