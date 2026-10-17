"""Compile staged TTIRs without importing JAX or opening a GPU backend."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import triton
from triton.backends.compiler import GPUTarget

assert os.environ['JAX_PLATFORMS']=='cpu'
root=Path(sys.argv[1]);rows=[]
for source in sorted(root.glob('glacier*_stage*.ttir')):
    start=time.perf_counter()
    kernel=triton.compile(str(source),target=GPUTarget('cuda',120,32),options={'num_warps':1})
    elapsed=time.perf_counter()-start
    cubin=source.with_suffix('.cubin');cubin.write_bytes(kernel.asm['cubin']);source.with_suffix('.ptx').write_text(kernel.asm['ptx'])
    resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cubin)],text=True)
    source.with_suffix('.resources.txt').write_text(resources)
    rows.append(dict(name=source.stem,compile_s=elapsed,resources=resources,
                     ptx_f64_lines=sum('.f64' in line for line in kernel.asm['ptx'].splitlines())))
    (root/'receipt.json').write_text(json.dumps(dict(platform='cpu compiler only',gpu_executed=False,rows=rows),indent=2)+'\n')
    print(json.dumps(rows[-1]),flush=True)
