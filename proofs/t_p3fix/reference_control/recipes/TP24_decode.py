from pathlib import Path
import hashlib,json,time,os
assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['CUDA_VISIBLE_DEVICES']==''
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');root=base/'TP24';root.mkdir(exist_ok=True)
raw=base/'TP40_mp52.mlirbc';r=json.loads((base/'TP40_kernel_receipt.json').read_text());assert hashlib.sha256(raw.read_bytes()).hexdigest()==r['kernels'][0]['bytecode_sha256']
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
start=time.monotonic();context=ir.context();ir.load_dialects(context);backend=CUDABackend(GPUTarget('cuda',120,32));backend.load_dialects(context)
mod=ir.parse_mlir_module(str(raw),context);s=str(mod);assert 'p3_mp52_column' in s and 'tt.func' in s
(root/'actual.ttir').write_text(s)
r.update(decode_s=time.monotonic()-start,TTIR_bytes=len(s.encode()),TTIR_sha256=hashlib.sha256(s.encode()).hexdigest(),scope='actual TP12 MP52 embeddedTriton kernel; decoded only, no compile/execute',cache='NONE');(root/'decode_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print('DECODED',r['decode_s'],len(s),r['TTIR_sha256'],flush=True)
