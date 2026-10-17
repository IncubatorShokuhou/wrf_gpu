"""Changed CPU artifact timing/memory; no GPU performance claim."""
import importlib.util
from pathlib import Path
import json,sys,time,re
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.assimilation import fdda_grid as fg
from gpuwrf.assimilation.fdda_surface import SurfaceGridFddaInputs,SurfaceGridFddaCarry

root=Path(__file__).parents[2]
spec=importlib.util.spec_from_file_location('surface_cost_fixture',root/'tests/v040/sfdda/test_surface_oracle.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert jax.devices()[0].platform=='cpu'
result={'platform':'cpu','gpu_qualified':False,'cases':{}}
for code in (0,1,2):
    d,inp=m.load(f'swiss_sfc{max(code,1)}_plain_r4.npz')
    if code==0:
        opt=fg.GridFddaOptions(**{k:getattr(inp.options,k) for k in fg.GridFddaOptions.__dataclass_fields__})
        inp=fg.GridFddaInputs(**{k:v for k,v in vars(inp).items() if k not in ('surface','surface_steps','options')},options=opt)
    c=fg.initial_fdda_carry(inp,jnp.float32,with_history=True)
    names=('u','v','th_phy','qv','pblh','z_at_w','t_phy','p_phy','regime','znt')
    args=tuple(jnp.asarray(d['frame1_'+n]) for n in names)+(jnp.asarray(d['ht']),jnp.asarray(1800.,jnp.float64),jnp.asarray(101,jnp.int32))
    def step(c,*a):
        u,v,th,q,pblh,zw,t,p,reg,znt,ht,lead,it=a
        kw={'p_phy':p}
        if code:kw['surface_context']=(t,p,reg,znt)
        return fg.fdda_driver_tendencies(c,inp,u,v,th,q,pblh,ht,zw,lead,it,jnp.float32,**kw)
    t=time.perf_counter();exe=jax.jit(step).lower(c,*args).compile();compile_s=time.perf_counter()-t
    jax.block_until_ready(exe(c,*args))
    samples=[]
    for _ in range(20):
        t=time.perf_counter();jax.block_until_ready(exe(c,*args));samples.append(time.perf_counter()-t)
    ma=exe.memory_analysis()
    f64=[line.strip() for line in exe.as_text().splitlines()
         if re.search(r'= f64\[',line) and not re.search(r'\b(parameter|constant)\(',line)]
    result['cases'][str(code)]={'compile_s':compile_s,'median_s':float(np.median(samples)),
        'argument_bytes':ma.argument_size_in_bytes,'temporary_bytes':ma.temp_size_in_bytes,
        'output_bytes':ma.output_size_in_bytes,'code_bytes':ma.generated_code_size_in_bytes,'f64_hlo':f64,
        'surface_resident_bytes':0 if code==0 else sum(np.asarray(a).nbytes for a in jax.tree.leaves(inp.surface)),
        'carry_bytes':sum(np.asarray(a).nbytes for a in jax.tree.leaves(c))}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
