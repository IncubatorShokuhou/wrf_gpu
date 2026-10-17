import os,sys,json,time,hashlib
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
src=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
start=time.monotonic()
print('PRESCREEN',src,hashlib.sha256(src.read_bytes()).hexdigest(),flush=True)
k=triton.compile(str(src),target=GPUTarget('cuda',120,32),options={'num_warps':int(sys.argv[3]) if len(sys.argv)>3 else 4,'enable_fp_fusion':True})
(out/'native.cubin').write_bytes(k.asm['cubin'])
(out/'report.json').write_text(json.dumps({'wall_s':time.monotonic()-start,'metadata':str(k.metadata),'scope':'isolatednativeCPUcompilepassprobe,noGPU'},indent=2)+'\n')
print('DONE',time.monotonic()-start,flush=True)
os._exit(0)
