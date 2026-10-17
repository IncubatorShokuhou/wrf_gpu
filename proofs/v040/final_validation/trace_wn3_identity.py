"""Real WN3 one-way 3-domain program identity, no forecast or GPU execution."""
from pathlib import Path
import hashlib, inspect, json, os, re, subprocess, sys, time
root,output=map(Path,sys.argv[1:3])
sys.path[:0]=[str(root/'src'),str(root/'proofs/o1_nlbind')]
import cpu_safe
import jax
import jax.numpy as jnp
import numpy as np
cpu_safe.install()
assert os.environ['JAX_PLATFORMS']=='cpu' and jax.devices()[0].platform=='cpu'
from gpuwrf.backends import pallas as facade
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om, domain_tree as dt
from gpuwrf.contracts import state as sc
from jax.experimental import pallas as pl
sc._gpu_device=lambda:jax.devices('cpu')[0]
source=Path('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run')
view=output.parent/'wn3-input-view'
view.mkdir(exist_ok=True)
for name in (p.name for p in source.iterdir() if p.name.startswith(('wrfinput','wrfbdy','wrflowinp')) or p.name=='namelist.input'):
    path=view/name
    if not path.exists():path.symlink_to(source/name)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
inputs={p.name:sha(p) for p in view.iterdir()}
original_pallas=pl.pallas_call
def interpret(*a,**kw):
    kw['interpret']=True;kw.pop('compiler_params',None)
    return original_pallas(*a,**kw)
facade.cuda_native=lambda:False
pl.pallas_call=interpret
t0=time.monotonic()
cfg=pipeline.NestedPipelineConfig(view,output.parent/'unused-out',output.parent/'unused-proof',hours=1,max_dom=3,feedback=False)
hierarchy,bundles,meta,start,dts,carries=pipeline._load_domains(cfg,('d01','d02','d03'))
facade.cuda_native=lambda:True
pl.pallas_call=original_pallas
assert all(c.base_state.alb is None for c in carries.values()) if hasattr(carries['d01'].base_state,'alb') else True
assert all(getattr(c,'feedback_history',None) is None for c in carries.values())
tree=dt.DomainTree.from_domains(hierarchy,bundles,feedback_enabled=False)
report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'source_src_tree':subprocess.check_output(['git','rev-parse','HEAD:src'],cwd=root,text=True).strip(),
    'backend':'cpu','native_facade_at_trace':True,'GPU_executed':False,'steps_executed':0,
    'input_hashes':inputs,'load_s':time.monotonic()-t0,'programs':{},'cache':'NONE'}
def consts(value):
    if hasattr(value,'jaxpr') and hasattr(value,'consts'):
        for item in value.consts:
            a=np.asarray(item);yield {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
        yield from consts(value.jaxpr)
    elif hasattr(value,'eqns'):
        for eq in value.eqns:yield from consts(eq.params)
    elif isinstance(value,dict):
        for key in sorted(value):yield from consts(value[key])
    elif isinstance(value,(list,tuple)):
        for x in value:yield from consts(x)
def trace(name,fn,*args):
    jax.clear_caches()
    jx=jax.make_jaxpr(fn)(*args)
    text=re.sub(r'name_and_src_info=[^\]\n]*','',str(jx))
    text=re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?','',text)
    report['programs'][name]={'sha256':hashlib.sha256(text.encode()).hexdigest(),'chars':len(text),
        'constants':list(consts(jx)),'inputs':[{'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(np.asarray(x).tobytes()).hexdigest()} for x in jax.tree.leaves(args)]}
    report['wall_s']=time.monotonic()-t0
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(name,report['programs'][name]['sha256'],flush=True)
for name in ('d01','d02','d03'):
    nl=bundles[name].namelist
    om._resolve_operational_suite(nl)
    clock=om.build_clock_base(nl)
    trace(name,lambda carry:om._advance_chunk_fori(carry,nl,jnp.int32(1),clock,n_steps=1,cadence=int(nl.radiation_cadence_steps)),carries[name])
fused=dt._operational_fused_cascade_factory(tree)('d02')
assert fused is not None,'actual default d02+d03 program must be covered'
closure=inspect.getclosurevars(fused).nonlocals
fn=closure['fused_jit']
nl2,nl3=(bundles[n].namelist for n in ('d02','d03'))
trace('fused_d02_d03',fn,carries['d02'],(carries['d03'],),jnp.int32(0),(jnp.int32(0),),om.build_clock_base(nl2),(om.build_clock_base(nl3),))
assert inputs=={p.name:sha(p) for p in view.iterdir()}
report['status']='COMPLETE'
report['wall_s']=time.monotonic()-t0
output.write_text(json.dumps(report,indent=2)+'\n')
