"""Deviceless Triton preflight of every native SMS call (no GPU initialization)."""
from pathlib import Path
import sys,json,time,re
import jax
import numpy as np
from jax._src.pallas.triton import lowering
from gpuwrf.backends import pallas as facade
from gpuwrf.dynamics.les3d_smagorinsky import les3d_smagorinsky_memory
from proofs.s1_km5.full_oracle import inputs

assert jax.devices()[0].platform=='cpu'
root=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]);results=[]
real_backend=jax.default_backend
real_cuda_native=facade.cuda_native
try:
    # Trace the product PTX RN path, while the only initialized backend is CPU.
    jax.default_backend=lambda:'gpu'
    facade.cuda_native=lambda:True
    with np.load(root/'data/fixtures/sms3dtke-full-wrf-v1.npz') as z:
        inp,cfg=inputs(z,'real_mixed')
        jp=jax.make_jaxpr(lambda x:les3d_smagorinsky_memory(x,cfg))(inp).jaxpr
    for e in jp.eqns:
        if e.primitive.name!='pallas_call':continue
        p=e.params;t=time.monotonic()
        print('LOWERING_CALL',len(results)+1,flush=True)
        result=lowering.lower_jaxpr_to_triton_module(p['jaxpr'],p['grid_mapping'],'cuda',120)
        asm=str(result.module)
        f64='f64' in asm
        assert not f64,e.params.get('name_and_src_info')
        match=re.search(r'tt.func public @([^\s(]+)',asm)
        results.append(dict(name=match.group(1) if match else 'sms_call_'+str(len(results)+1),IR_characters=len(asm),
                            lowering_s=time.monotonic()-t,f64=f64))
    assert len(results)==16,len(results)
finally:
    jax.default_backend=real_backend
    facade.cuda_native=real_cuda_native
out.write_text(json.dumps(dict(backend='CPU',GPU_execution=False,target='cuda/sm120',
    native_calls=len(results),results=results),indent=1))
print('NATIVE_LOWER_PASS',len(results),flush=True)
