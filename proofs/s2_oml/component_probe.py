"""CPU compiled-kind and allocation diagnostics on an immutable source tree.

No GPU timing, launch count, VRAM or compile-readiness claims. Run sequentially
with all other lane JAX work. SOURCE must be the detached measured snapshot.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import re
import subprocess
import sys
import time

SOURCE=Path(sys.argv[1]).resolve()
OUT=Path(sys.argv[2]).resolve()
SELECTED=set(sys.argv[3:])
sys.path[:0]=[str(SOURCE/'src')]
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics.ocean_mixed_layer import OceanCarry,step
from gpuwrf.physics.surface_layer import _surface_layer_impl
from gpuwrf.physics.sfclay_revised_mm5 import SfclayRevisedMM5ColumnState,sfclay_revised_mm5_run
from gpuwrf.physics.sfclay_old_mm5 import sfclay_old_mm5_columns

assert jax.devices()[0].platform=='cpu'
assert jax.config.x64_enabled
assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
data=json.loads((SOURCE/'proofs/s2_oml/fixtures/flux.json').read_text())
record={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        'backend':'cpu','x64_enabled':bool(jax.config.x64_enabled),
        'cache':'disabled','gpu_claim':False,'programs':{}}


def probe(name,fn,*args):
    if SELECTED and name not in SELECTED:return
    start=time.perf_counter()
    compiled=jax.jit(fn).lower(*args).compile()
    compile_s=time.perf_counter()-start
    text=compiled.as_text()
    arithmetic=r'(?:add|subtract|multiply|divide|power|sqrt|rsqrt|exponential|log|sine|cosine|maximum|minimum|reduce|dot|convolution)'
    count=len(re.findall(r'= f64(?:\[[^\]]*\])?[^\n=]*?\b'+arithmetic+r'\(',text))
    total_arithmetic=len(re.findall(r'= f(?:32|64)(?:\[[^\]]*\])?[^\n=]*?\b'+arithmetic+r'\(',text))
    assert total_arithmetic>10,(name,'vacuous HLO parser',total_arithmetic)
    result=compiled(*args);jax.block_until_ready(result)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result))
    assert count==0,(name,count)
    # SurfaceFluxes.t_skin=-1 is static compatibility metadata, not a WRF
    # field. JAX materializes that weak scalar at return; E120 excludes it.
    real_leaves=[x for x in jax.tree.leaves(result) if hasattr(x,'dtype') and np.issubdtype(x.dtype,np.floating)
                 and not (x.shape==() and x.weak_type)]
    weak_metadata=[float(x) for x in jax.tree.leaves(result) if hasattr(x,'dtype') and x.shape==() and x.weak_type]
    assert all(x.dtype==np.float32 for x in real_leaves),[(x.shape,str(x.dtype)) for x in real_leaves]
    memory=compiled.memory_analysis()
    entry={'compile_s':compile_s,'optimized_hlo_sha256':hashlib.sha256(text.encode()).hexdigest(),
           'f64_arithmetic':count,'arithmetic_rows_censused':total_arithmetic,
           'output_float_dtypes':sorted({str(v.dtype) for v in real_leaves}),'weak_static_metadata':weak_metadata,
           'argument_bytes':memory.argument_size_in_bytes,'output_bytes':memory.output_size_in_bytes,
           'temporary_bytes':memory.temp_size_in_bytes}
    record['programs'][name]=entry
    OUT.write_text(json.dumps(record,indent=2)+'\n')
    print(name,entry,flush=True)


shape=(42,42)
f=lambda x:jnp.full(shape,x,jnp.float32)
c=OceanCarry(*(f(x) for x in (302,302,20,20,0,0,297)))
forces={k:f(x) for k,x in zip(('hfx','lh','gsw','glw','u','v','ust','f','emiss'),(250,700,600,380,65,-20,2.3,5e-5,.98),strict=True)}
probe('OML',lambda c,forces:step(c,**forces,dt=18.,gamma=.14,relaxation_time=3600.),c,forces)
for opt in (1,2):
    rows=np.asarray(data[f'mynn_{opt}']['inputs'],np.float32)
    keys=('u','v','temperature','qv','pressure','dz','rho','u2','v2','dz2','mavail','pblh','xland','tsk','psfc','qcg','snowh','znt','ust','mol','qsfc','hfx','qfx')
    inp=dict(zip(keys,rows.T,strict=True))
    fields={k:jnp.asarray(inp[k])[:,None] for k in ('rho','mavail','pblh','xland','psfc','mol','qsfc','hfx','qfx')}
    for dst,src in (('t_skin','tsk'),('roughness_m','znt'),('ustar','ust')):fields[dst]=jnp.asarray(inp[src])[:,None]
    for dst,src in (('u','u'),('v','v'),('theta','temperature'),('t_air','temperature'),('qv','qv'),('p','pressure'),('dz','dz')):fields[dst]=jnp.asarray(inp[src])[:,None,None]
    fields.update(lakemask=jnp.zeros((8,1),jnp.float32),dx_m=jnp.full((8,1),9000.,jnp.float32))
    probe(f'MYNN_{opt}',lambda d,m:_surface_layer_impl(SimpleNamespace(**d),False,jnp.float32,isftcflx=opt,ustm=m,_return_hurricane=True),fields,jnp.full((8,1),.35,jnp.float32))
    for scheme in ('revised','old'):
        d=data[f'{scheme}_{opt}'];a=lambda key:jnp.asarray(d[key],jnp.float32)
        core=[a(k) for k in ('U','V','T','QV','P','DZ')]
        if scheme=='revised':
            state=SfclayRevisedMM5ColumnState(*core,*(a(k) for k in ('PSFC','TSK','XLAND','LAKEMASK','MAVAIL','ZNT_IN','UST_IN','MOL_IN','HFX_IN','QFX_IN','QSFC_IN','PBLH','DX','WATER_DEPTH')))
            probe(f'MM5rev_{opt}',lambda s:sfclay_revised_mm5_run(s,isftcflx=opt),state)
        else:
            args=core+[a(k) for k in ('PSFC','TSK','XLAND','ZNT_IN','UST_IN','MOL_IN','QSFC_IN','PBLH','MAVAIL','LAKEMASK','DX','HFX_IN','QFX_IN')]
            probe(f'MM5old_{opt}',lambda args:sfclay_old_mm5_columns(*args,isftcflx=opt),tuple(args))
record['carry_bytes_per_cell']=32
record['carry_bytes_swiss_42x42']=32*42*42
record['gpu_limits']='GPU launch count, traffic, VRAM, compilation/readiness and forecast D6 remain unmeasured.'
OUT.write_text(json.dumps(record,indent=2)+'\n')
