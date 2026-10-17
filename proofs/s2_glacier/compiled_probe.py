"""Compiled REAL32 checks and CPU Triton preflight, without GPU initialization."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.triton import lowering
from gpuwrf.physics.noahmp.glacier import glacier_column
from gpuwrf.coupling.glacier_surface_hook import glacier_kernel
from gpuwrf.io.noahmp_glacier_options import GlacierStatic


def main(source,out):
    assert jax.devices()[0].platform=='cpu'
    data=json.loads(source.read_text())
    layout=json.loads((source.parent/'G04_oracle/manifest.json').read_text())
    rows=[]
    for gla in (1,2):
        fixtures=[f for f in data['fixtures'] if f['options'][-1]==gla]
        arrays=np.asarray([f['input'] for f in fixtures],np.float32).T
        values={}
        for f in layout['fields']:
            if f['intent']=='OUT':continue
            a=arrays[f['offset']:f['offset']+f['extent']]
            values[f['name']]=list(map(jnp.asarray,a)) if f['extent']>1 else jnp.asarray(a[0],jnp.int32 if f['integer'] else jnp.float32)
        fn=jax.jit(lambda v:glacier_column(v,opt_gla=gla))
        start=time.perf_counter();compiled=fn.lower(values).compile();compile_s=time.perf_counter()-start
        start=time.perf_counter();actual=jax.device_get(compiled(values));execute_s=time.perf_counter()-start
        memory=compiled.memory_analysis()
        fields={};nf=[]
        refs=np.asarray([f['reference'] for f in fixtures],np.float32).T
        for f in layout['fields']:
            if f['intent']=='IN':continue
            a=np.asarray(actual[f['name']],np.float32).reshape(f['extent'],-1)
            r=refs[f['offset']:f['offset']+f['extent']]
            valid=np.ones_like(a,bool)
            if f['name']=='hcpct':
                index=next(f['offset'] for f in layout['fields'] if f['name']=='isnow')
                valid[:3]=np.arange(3)[:,None]>=arrays[index][None,:]+3
            if not np.isfinite(a[valid]).all():nf.append(f['name'])
            scale=np.max(np.abs(r),axis=0)
            ulp=np.spacing(scale)
            fields[f['name']]=dict(max_abs=float(np.max(np.abs(a-r)[valid])),
                max_scale_ulp=float(np.max((np.abs(a-r)/np.maximum(ulp,np.float32(1.e-30)))[valid])))
        # Pallas receives only per-column leaves; lists keep layer arrays in registers.
        inp=jax.tree.map(lambda a:a.reshape(1,-1),values)
        config=GlacierStatic(tuple(range(len(fixtures))),(),(1,len(fixtures)),opt_gla=gla)
        graph=jax.make_jaxpr(lambda v:glacier_kernel(v,config,history=True))(inp)
        params=next(e.params for e in graph.jaxpr.eqns if e.primitive.name=='pallas_call')
        native=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
        ttir=str(native.module);out.with_suffix(f'.gla{gla}.ttir').write_text(ttir)
        rows.append(dict(opt_gla=gla,cases=len(fixtures),compile_s=compile_s,execute_s=execute_s,
            memory=dict(argument_size=memory.argument_size_in_bytes,output_size=memory.output_size_in_bytes,
                        temp_size=memory.temp_size_in_bytes),fields=fields,nonfinite=nf,ttir_chars=len(ttir)))
    out.write_text(json.dumps(dict(platform='cpu',gpu_executed=False,rows=rows),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]))
