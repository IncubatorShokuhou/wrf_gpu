"""Lower staged Pallas glacier bodies and compile each on CPU for sm120."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.triton import lowering
import triton
from triton.backends.compiler import GPUTarget
from gpuwrf.physics.noahmp.glacier_stages import staged_kernel
from gpuwrf.io.noahmp_glacier_options import GlacierStatic

assert os.environ['JAX_PLATFORMS']=='cpu' and jax.devices()[0].platform=='cpu'
fixture=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
data=json.loads(fixture.read_text());manifest=json.loads((fixture.parent/'G04_oracle/manifest.json').read_text())
rows=[]
lower_only='--lower-only' in sys.argv[3:]
for gla in (1,2):
    cases=[f for f in data['fixtures'] if f['options'][-1]==gla]
    packed=np.asarray([f['input'] for f in cases],np.float32).T;inputs={}
    for field in manifest['fields']:
        if field['intent']=='OUT':continue
        a=packed[field['offset']:field['offset']+field['extent']]
        a=jnp.asarray(a,jnp.int32 if field['integer'] else jnp.float32)
        inputs[field['name']]=[x.reshape(1,-1) for x in a] if field['extent']>1 else a[0].reshape(1,-1)
    config=GlacierStatic(tuple(range(len(cases))),(),(1,len(cases)),opt_gla=gla)
    graph=jax.make_jaxpr(lambda v:staged_kernel(v,config,history=True))(inputs)
    calls=[e.params for e in graph.jaxpr.eqns if e.primitive.name=='pallas_call']
    assert len(calls)==3
    for stage,params in enumerate(calls):
        result=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
        text=str(result.module);path=out/f'glacier{gla}_stage{stage}.ttir';path.write_text(text)
        if lower_only:
            rows.append(dict(opt_gla=gla,stage=stage,ttir_chars=len(text),lowering='PASS'))
            (out/'lowering.json').write_text(json.dumps(dict(platform='cpu',gpu_executed=False,rows=rows),indent=2)+'\n')
            continue
        start=time.perf_counter();kernel=triton.compile(str(path),target=GPUTarget('cuda',120,32),options={'num_warps':1})
        elapsed=time.perf_counter()-start
        cubin=path.with_suffix('.cubin');cubin.write_bytes(kernel.asm['cubin']);path.with_suffix('.ptx').write_text(kernel.asm['ptx'])
        resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(cubin)],text=True)
        path.with_suffix('.resources.txt').write_text(resources)
        rows.append(dict(opt_gla=gla,stage=stage,ttir_chars=len(text),compile_s=elapsed,resources=resources,
            ptx_f64_lines=sum('.f64' in s for s in kernel.asm['ptx'].splitlines())))
        (out/'receipt.json').write_text(json.dumps(dict(platform='cpu',gpu_executed=False,rows=rows),indent=2)+'\n')
        print(json.dumps(rows[-1]),flush=True)
