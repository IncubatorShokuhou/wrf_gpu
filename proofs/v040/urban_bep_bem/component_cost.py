"""CPU executable costs only; GPU launches/roofline and forecast cost unknown."""
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import urban_bep_components as k

assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert jax.devices()[0].platform == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
root = Path(__file__).resolve().parent
records = np.load(root/'savepoints/components_fp32.npz')['values']


def inputs(kind):
    rows = records[(records[:,0] == kind) & (records[:,3] == 0)]
    rows = rows[np.lexsort((rows[:,2],rows[:,1]))]
    return jnp.asarray(rows[:,4].reshape(48,-1),dtype=jnp.float32)


x=inputs(0).T
dz,z0,u,v,th,th0,surface,rho,angle,dt,pressure,qv=x
mat=inputs(10)
temp,depth,ala,capacity=(mat[:,a:a+10] for a in (0,10,20,30))
sw,lw,em,alb,sensible=mat[:,40:].T
geo=inputs(11)
z,c,zu,pb,ss=(geo[:,a:b] for a,b in [(0,6),(6,12),(12,30),(30,48),(48,66)])
wf=inputs(12).reshape(48,4,18,7)
hf=inputs(13).reshape(48,2,18,4)
wall=k.WallFlux(*(wf[...,j] for j in range(7)))
flat=k.FlatFlux(*(hf[...,j] for j in range(4)))
bs=jnp.broadcast_to(jnp.array([10.,20.],dtype=jnp.float32),(48,2))
ws=jnp.broadcast_to(jnp.array([15.,12.],dtype=jnp.float32),(48,2))
cases={
 'bep_flat':(k.bep_flux_flat,(dz,z0,u,v,th,th0,surface)),
 'bem_flat':(k.bem_flux_flat,(dz,z0,u,v,th,th0,surface,qv,rho,pressure)),
 'material':(k.bep_soil_temp,(depth,temp,ala,capacity,sw,lw,pressure,dt,em,alb,sensible)),
 'urban_meso':(partial(k.bep_urban_meso,nd=2),(z,z[:,1:]-z[:,:-1],zu[:,:7],pb,ss,bs,ws,wall,flat)),
}
report={'backend':'cpu','cpu_affinity':list(os.sched_getaffinity(0)),
        'columns':48,'gpu_cost':'UNMEASURED','forecast_cost':'UNMEASURED',
        'source_sha256':hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest(),'components':{}}
for name,(fn,args) in cases.items():
    start=time.perf_counter()
    executable=jax.jit(fn).lower(*args).compile()
    compile_s=time.perf_counter()-start
    hlo=executable.as_text()
    assert 'f64[' not in hlo,name
    jax.block_until_ready(executable(*args))
    timings=[]
    for _ in range(7):
        start=time.perf_counter();jax.block_until_ready(executable(*args));timings.append(time.perf_counter()-start)
    memory=executable.memory_analysis()
    report['components'][name]={'compile_s':compile_s,'median_cpu_call_s':statistics.median(timings),
      'argument_bytes':memory.argument_size_in_bytes,'output_bytes':memory.output_size_in_bytes,
      'temporary_bytes':memory.temp_size_in_bytes,'f64_types':0,
      'optimized_hlo_sha256':hashlib.sha256(hlo.encode()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
