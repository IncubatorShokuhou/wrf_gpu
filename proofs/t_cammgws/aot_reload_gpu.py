"""Fresh-process AOT reload receipt for the stock-Triton CAM-MG custom call (t-cammgws).

export: lower+compile the product native column (pristine c03_s1, 24 levels) with the
        stock-Triton route, record the PTX handed to triton_kernel_call (sha256), serialize
        the XLA executable, execute once, save outputs.
reload: in a FRESH process (no lowering or stock compile allowed), deserialize the blob,
        execute with the same operands, compare every output BYTE with export, and time
        the first launch (driver PTX JIT) and warm launches. CUDA_CACHE_PATH is kept so
        the driver-JIT native image can be extracted and scanned afterwards on CPU.
usage: aot_reload_gpu.py TREE OUT export|reload
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def main():
    tree,out,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
    sys.path[:0]=[str(tree/'src')]
    import jax
    import jax.numpy as jnp
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1' and jax.devices()[0].platform=='gpu'
    from gpuwrf.kernels import phys_cammg, _triton_kernel_call as route
    out.mkdir(parents=True,exist_ok=True)
    with np.load(tree/'proofs/s2_cammg/fixtures/public.npz') as z:
        fields={k.split('_input_',1)[1]:z[k] for k in z.files if k.startswith('c03_s1_input_')}
    inputs={k:jnp.asarray(v).reshape((1,) if k in phys_cammg.SURFACE_FIELDS else (1,25)) for k,v in fields.items()}
    modes=tuple(map(jnp.float32,(1.e9,3.e8,2.e5)))
    args=(inputs,jnp.int32(1),*modes)
    receipt={'mode':mode,'device':str(jax.devices()[0]),'pid':os.getpid(),'cuda_cache':os.environ.get('CUDA_CACHE_PATH')}
    if mode=='export':
        captured={}
        real=route.stock_compile
        def capture(*a,**k):
            result=real(*a,**k);captured['ptx']=result[0];captured['shared']=result[1];captured['name']=result[2]
            return result
        route.stock_compile=capture
        start=time.monotonic()
        lowered=jax.jit(lambda v,it,am,ai,co:phys_cammg.full_column(v,18.,it,accum_mode=am,aitken_mode=ai,
                        coarse_mode=co,interpret=False)).lower(*args)
        text=lowered.as_text();compiled=lowered.compile();receipt['compile_s']=time.monotonic()-start
        assert 'triton_kernel_call' in text and '__gpu$xla.gpu.triton' not in text
        blob=bytes(compiled._executable.xla_executable.serialize())
        (out/'executable.xlaexec').write_bytes(blob);(out/'kernel.ptx').write_text(captured['ptx'])
        receipt.update(executable_sha256=hashlib.sha256(blob).hexdigest(),executable_bytes=len(blob),
                       ptx_sha256=hashlib.sha256(captured['ptx'].encode()).hexdigest(),ptx_bytes=len(captured['ptx']),
                       shared=captured['shared'],kernel=captured['name'],compiler_identity=list(map(str,route.stock_compiler_identity())))
        t=time.perf_counter();values,fatal=compiled(*args);jax.block_until_ready((values,fatal))
        receipt['first_execute_s']=time.perf_counter()-t
        np.savez(out/'export_outputs.npz',**{k:np.asarray(v) for k,v in values.items()},__fatal=np.asarray(fatal))
    else:
        def forbidden(*a,**k):raise AssertionError('reload must not lower/compile the stock kernel')
        route.stock_compile=forbidden
        from jaxlib import _jax
        blob=(out/'executable.xlaexec').read_bytes();dev=jax.devices()[0]
        t=time.monotonic();exe=dev.client.deserialize_executable(blob,_jax.DeviceList((dev,)),None)
        receipt['deserialize_s']=time.monotonic()-t;receipt['executable_sha256']=hashlib.sha256(blob).hexdigest()
        leaves=jax.tree.leaves(args)
        t=time.perf_counter();flat=exe.execute(leaves);jax.block_until_ready(flat)
        receipt['first_execute_s_incl_driver_jit']=time.perf_counter()-t
        warm=[]
        for _ in range(10):
            t=time.perf_counter();jax.block_until_ready(exe.execute(leaves));warm.append(time.perf_counter()-t)
        receipt['warm_median_s']=float(np.median(warm))
        with np.load(out/'export_outputs.npz') as z:
            ref={k:z[k] for k in z.files}
        names=sorted(k for k in ref if k!='__fatal')
        got=dict(zip(names,flat[:-1],strict=True));fatal=np.asarray(flat[-1])
        nonbyte=[k for k in names if np.asarray(got[k]).tobytes()!=ref[k].tobytes()]
        receipt.update(outputs=len(names),nonbyte=nonbyte,fatal=fatal.tolist(),
                       fatal_byte=fatal.tobytes()==ref['__fatal'].tobytes(),
                       rain=float(np.asarray(got['rainncv']).max()),PASS=not nonbyte and fatal.tobytes()==ref['__fatal'].tobytes())
    (out/f'{mode}.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt)[:1500],flush=True)


if __name__=='__main__':main()
