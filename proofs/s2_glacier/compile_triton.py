"""Compile the actual lowered glacier TTIR for sm120 on CPU; never launch it."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import triton
from triton.backends.compiler import GPUTarget


assert os.environ['JAX_PLATFORMS']=='cpu'
prefix=Path(sys.argv[1]);target=Path(sys.argv[2]);target.mkdir(parents=True,exist_ok=True)
rows=[]
for gla in (1,2):
    source=prefix.with_suffix(f'.gla{gla}.ttir')
    start=time.perf_counter()
    kernel=triton.compile(str(source),target=GPUTarget('cuda',120,32),options={'num_warps':1})
    elapsed=time.perf_counter()-start
    cubin=target/f'glacier{gla}.cubin';cubin.write_bytes(kernel.asm['cubin'])
    ptx=kernel.asm['ptx'];(target/f'glacier{gla}.ptx').write_text(ptx)
    resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cubin)],text=True)
    (target/f'glacier{gla}.resources.txt').write_text(resources)
    rows.append(dict(opt_gla=gla,compile_s=elapsed,metadata=kernel.metadata._asdict(),
                     resources=resources,ptx_f64_lines=sum('.f64' in line for line in ptx.splitlines())))
    (target/'receipt.json').write_text(json.dumps(dict(platform='CPU compiler only',gpu_executed=False,rows=rows),indent=2,default=str)+'\n')
print(json.dumps(rows,indent=2,default=str))
