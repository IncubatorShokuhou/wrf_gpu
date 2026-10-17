"""Lower the actual CAM-MG Pallas body to CUDA TTIR on CPU only."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--nz',type=int,default=24)
    args=ap.parse_args();sys.path.insert(0,str(args.tree/'src'))
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.triton import lowering
    from gpuwrf.kernels.phys_cammg import full_column,SURFACE_FIELDS
    assert jax.devices()[0].platform=='cpu'
    with np.load(args.tree/'proofs/s2_cammg/fixtures/public.npz') as z:
        fields={k.split('_input_',1)[1]:z[k] for k in z.files if k.startswith('c01_s1_input_')}
    fields={k:jnp.asarray(v).reshape((1,) if k in SURFACE_FIELDS else (1,25)) for k,v in fields.items()}
    if args.nz!=24:
        # Abstract signatures qualify the real 44-level executable shape;
        # they are not constructed meteorological columns or fidelity inputs.
        fields={k:jax.ShapeDtypeStruct((1,) if k in SURFACE_FIELDS else (1,args.nz+1),jnp.float32)
                for k in fields}
    start=time.monotonic()
    with jax.enable_x64(False):
        program=jax.make_jaxpr(lambda f,dt,it:full_column(f,dt,it))(fields,jnp.float32(18),jnp.int32(1)).jaxpr
        assert not jax.config.jax_enable_x64
    params=next(e.params for e in program.eqns if e.primitive.name=='pallas_call')
    arithmetic={'add','sub','mul','div','pow','integer_pow','neg','exp','log','log1p',
                'sqrt','rsqrt','abs','max','min','reduce_sum','reduce_max','reduce_min'}
    def wide(e):
        return e.primitive.name in arithmetic and any(getattr(v.aval,'dtype',None)==np.float64 for v in e.outvars)
    outside=sum(wide(e) for e in program.eqns)
    assert outside==0,'FP64 arithmetic outside original CAM DOUBLE island'
    def walk(jaxpr):
        jaxpr=getattr(jaxpr,'jaxpr',jaxpr)
        for e in jaxpr.eqns:
            yield e
            for value in e.params.values():
                for item in value if isinstance(value,(list,tuple)) else (value,):
                    if hasattr(item,'jaxpr') or hasattr(item,'eqns'):yield from walk(item)
    island=sum(wide(e) for e in walk(params['jaxpr']))
    trace_s=time.monotonic()-start
    result=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
    text=str(result.module);args.out.with_suffix('.ttir').write_text(text)
    receipt=dict(backend='cpu',GPU_execution=False,target='cuda/sm120',physical_levels=args.nz,
                 fp64_arithmetic_outside_CAM=outside,fp64_arithmetic_in_original_CAM_r8=island,
                 dtype_scope='caller x64=False restored; REAL4 public results; original r8 CAM island',
                 E41='pending real GPU executable; TTIR is not cubin evidence',trace_s=trace_s,
                 elapsed_s=time.monotonic()-start,TTIR_characters=len(text),verdict='PASS')
    args.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
