"""Real Swiss LCZ/morphology/AHE coupled CPU smoke and evolved restart.

Release kernels run in CPU Pallas interpretation with the established RN32
inline-asm shim and CPU Thomas solver. This is not GPU-binary qualification.
One compiled one-step program serves both uninterrupted and resumed step 2.
"""
from pathlib import Path
import json
import os
import re
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=Path('<USER_HOME>/wrf_gpu2_lanes/u2-lcz')
mode=int(sys.argv[1])
assert mode in (1,2)
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'proofs/o1_nlbind'))
import jax
import jax.numpy as jnp
import numpy as np
import cpu_safe
cpu_safe.install()
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as pt
original=pl.pallas_call
def interpret(*a,**kw):
    kw['interpret']=True;kw.pop('compiler_params',None)
    return original(*a,**kw)
pl.pallas_call=interpret
def cpu_asm(asm,*,args,result_shape_dtypes,**kw):
    op=asm.split()[0].split('.')[0]
    a,b=(jnp.asarray(x,jnp.float64) for x in args)
    exact={'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'div':lambda:a/b}[op]()
    dt=result_shape_dtypes[0].dtype
    if np.dtype(dt)==np.float32:
        exact=jax.lax.reduce_precision(exact,8,23)
    return [exact.astype(dt)]
pt.elementwise_inline_asm=cpu_asm
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om
from gpuwrf.io.restart import write_restart,read_restart
assert jax.devices()[0].platform=='cpu'
assert om.__file__.startswith(str(ROOT))
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
case=OUT/'case_swiss_lcz'
started=time.monotonic()
with tempfile.TemporaryDirectory(dir=OUT) as temp:
    private=Path(temp)/'case';private.mkdir()
    for path in case.iterdir():
        if path.name!='namelist.input': (private/path.name).symlink_to(path)
    text=re.sub(r'distributed_ahe_opt\s*=\s*2',f'distributed_ahe_opt={mode}',(case/'namelist.input').read_text())
    if mode==2:
        text=text.replace('&physics',"&physics\n urban_ahe_file='"+str(OUT/'hourly_ahe.nc')+"',",1)
    (private/'namelist.input').write_text(text)
    cfg=pipeline.NestedPipelineConfig(private,Path(temp)/'out',Path(temp)/'proof',hours=1,max_dom=1)
    _,bundles,_,_,_,carries=pipeline._load_domains(cfg,('d01',))
nl=bundles['d01'].namelist
carry=carries['d01']
assert nl.urban_bundle.use_wudapt_lcz and nl.urban_bundle.ahe_opt==mode
assert carry.urban_slucm.inputs.morphology is not None
assert nl.urban_bundle.table.ahoption==1 and nl.urban_bundle.table.alhoption==1
assert int(np.count_nonzero(np.asarray(carry.urban_slucm.utype)))==12
initial=np.asarray(carry.urban_slucm.state.tc).copy()
clock=om.build_clock_base(nl)
print(json.dumps({'stage':'loaded','mode':mode,'seconds':time.monotonic()-started}),flush=True)
carry=om._advance_chunk(carry,nl,1,clock,n_steps=1,cadence=int(nl.radiation_cadence_steps))
jax.block_until_ready(carry)
print(json.dumps({'stage':'step1','seconds':time.monotonic()-started}),flush=True)
checkpoint=OUT/f'LCZ_SMOKE{mode}.pkl'
write_restart(carry,nl,nl.grid,1,checkpoint)
restored,_,_,step=read_restart(checkpoint,namelist=nl)
assert step==1
# Match nested_pipeline's actual resume path, which commits every restored leaf
# before advance. Otherwise CPU PJRT builds another placement specialization.
restored=pipeline._commit_to_operational_device(restored)
warm_started=time.monotonic()
control=om._advance_chunk(carry,nl,2,clock,n_steps=1,cadence=int(nl.radiation_cadence_steps))
jax.block_until_ready(control)
warm_control_s=time.monotonic()-warm_started
control_host=[np.asarray(x).copy() for x in jax.tree.leaves(control)]
warm_started=time.monotonic()
resumed=om._advance_chunk(restored,nl,2,clock,n_steps=1,cadence=int(nl.radiation_cadence_steps))
jax.block_until_ready(resumed)
warm_resume_s=time.monotonic()-warm_started
resume_host=[np.asarray(x) for x in jax.tree.leaves(resumed)]
assert len(control_host)==len(resume_host)
assert all(np.array_equal(a,b) for a,b in zip(control_host,resume_host))
nonfinite=sum(int((~np.isfinite(x)).sum()) for x in resume_host if np.issubdtype(x.dtype,np.floating))
guards=int(np.asarray(resumed.census.guards).sum())
delta=float(np.max(np.abs(np.asarray(resumed.urban_slucm.state.tc)-initial)))
assert nonfinite==0 and guards==0 and delta>0
result={'source':str(ROOT),'mode':mode,'steps':2,'restart_identical_leaves':len(resume_host),
    'nonfinite':nonfinite,'guards':guards,'canopy_temperature_delta_K':delta,
    'wall_s':time.monotonic()-started,'backend':'cpu','pallas':'interpret','inline_asm':'RN32 emulation',
    'warm_control_step_s':warm_control_s,'resume_step_s':warm_resume_s,
    'theta_dtype':str(resumed.state.theta.dtype),'inputs_bytes':sum(x.nbytes for x in
        jax.tree.leaves(jax.tree.map(np.asarray,resumed.urban_slucm.inputs)))}
(OUT/f'LCZ_SMOKE{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
checkpoint.unlink()
print(json.dumps(result),flush=True)
