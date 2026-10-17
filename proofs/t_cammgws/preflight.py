"""CPU-only CUDA frontend preflight for a structural CAM-MG variant (t-cammgws).

Lowers the actual native column to CUDA TTIR (sm_120) without a GPU, as
proofs/s2_cammg/lower_column.py does, after installing the variant. Records
TTIR structure counts, FP64 arithmetic outside the original CAM r8 island, and
whether the kernel jaxpr is identical to the unpatched one (line info stripped).
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--variant',required=True);ap.add_argument('--nz',type=int,default=24)
    args=ap.parse_args();here=Path(__file__).resolve().parent
    sys.path[:0]=[str(args.tree/'src'),str(here)]
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.triton import lowering
    assert jax.devices()[0].platform=='cpu'
    from gpuwrf.kernels import phys_cammg
    SURFACE=phys_cammg.SURFACE_FIELDS
    with np.load(args.tree/'proofs/s2_cammg/fixtures/public.npz') as z:
        fields={k.split('_input_',1)[1]:z[k] for k in z.files if k.startswith('c01_s1_input_')}
    if args.nz==24:fields={k:jnp.asarray(v).reshape((1,) if k in SURFACE else (1,25)) for k,v in fields.items()}
    else:fields={k:jax.ShapeDtypeStruct((1,) if k in SURFACE else (1,args.nz+1),jnp.float32) for k in fields}
    def kernel_jaxpr():
        with jax.enable_x64(False):
            program=jax.make_jaxpr(lambda f,dt,it:phys_cammg.full_column(f,dt,it))(fields,jnp.float32(18),jnp.int32(1)).jaxpr
        eqn=next(e for e in program.eqns if e.primitive.name=='pallas_call')
        return program,eqn
    def canon(jaxpr):
        text=str(jaxpr)
        return hashlib.sha256(re.sub(r'\S+\.py:\d+(:\d+)?','',text).encode()).hexdigest()
    start=time.monotonic()
    _,base=kernel_jaxpr();base_sha=canon(base.params['jaxpr'])
    jax.clear_caches()
    from variants import install
    patch=install(args.variant)
    program,eqn=kernel_jaxpr();trace_s=time.monotonic()-start
    arithmetic={'add','sub','mul','div','pow','integer_pow','neg','exp','log','log1p',
                'sqrt','rsqrt','abs','max','min','reduce_sum','reduce_max','reduce_min'}
    wide=lambda e:e.primitive.name in arithmetic and any(getattr(v.aval,'dtype',None)==np.float64 for v in e.outvars)
    outside=sum(wide(e) for e in program.eqns)
    params=eqn.params
    compiler=str(params.get('compiler_params'))
    begin=time.monotonic()
    result=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
    text=str(result.module);lower_s=time.monotonic()-begin
    counts={op:len(re.findall(r'\b'+re.escape(op)+r'\b',text)) for op in ('scf.while','scf.for','scf.if','tt.reduce','tt.load','tt.store')}
    receipt=dict(backend='cpu',GPU_execution=False,target='cuda/sm120',physical_levels=args.nz,variant=patch,
                 kernel_jaxpr_sha256=canon(params['jaxpr']),baseline_kernel_jaxpr_sha256=base_sha,
                 jaxpr_identical_to_baseline=canon(params['jaxpr'])==base_sha,compiler_params=compiler,
                 fp64_arithmetic_outside_CAM=outside,TTIR_characters=len(text),TTIR_sha256=hashlib.sha256(text.encode()).hexdigest(),
                 TTIR_counts=counts,trace_s=trace_s,lower_s=lower_s,
                 verdict='PASS' if outside==0 else 'FAIL',E41='pending real GPU executable; TTIR is not cubin evidence')
    args.out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    assert outside==0


if __name__=='__main__':main()
