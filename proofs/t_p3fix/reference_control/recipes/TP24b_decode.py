from pathlib import Path
import hashlib,json,time,os
assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['CUDA_VISIBLE_DEVICES']==''
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');root=base/'TP24b';root.mkdir(exist_ok=False)
raw=(base/'TP40_mp52.mlirbc').read_bytes();r=json.loads((base/'TP40_kernel_receipt.json').read_text());assert hashlib.sha256(raw).hexdigest()==r['kernels'][0]['bytecode_sha256']
from jax._src.pallas.triton import lowering
from jax._src.lib.mlir import ir
start=time.monotonic()
with lowering._new_ir_context():
 mod=ir.Module.parse(raw);s=str(mod);assert 'p3_mp52_column' in s and 'tt.func' in s
(root/'actual.ttir').write_text(s);r.update(decode_s=time.monotonic()-start,TTIR_bytes=len(s.encode()),TTIR_sha256=hashlib.sha256(s.encode()).hexdigest(),scope='actual TP12 MP52 customcall decoded by matching JAX dialect; no passes/compile/execute',cache='NONE');(root/'decode_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print('DECODED',r['decode_s'],len(s),r['TTIR_sha256'],flush=True)
