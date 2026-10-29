"""First coupled bisection seam: real NML/clock/prior transport, no timestep."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--case',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
sys.path[:0]=[str(a.source/'src'),str(a.source/'proofs/o1_nlbind')]
import jax
import numpy as np
from jax.experimental import pallas as pl
from jax._src.pallas import pallas_call as pc
original=pc.pallas_call
def interpreted(*args,**kw):return original(*args,**{**kw,'interpret':True})
pc.pallas_call=pl.pallas_call=interpreted
import cpu_safe
cpu_safe.install()
from gpuwrf.backends import pallas as facade
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io.noahmp_options import NoahMPOptions
from gpuwrf.runtime.operational_mode import build_clock_base
from gpuwrf.surface_params import controls as sc

assert all(d.platform=='cpu' for d in jax.devices())
facade.cuda_native=lambda:False
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
sc.UNAVAILABLE_SURFACE_CONTROLS=frozenset()  # only this research process
started=time.monotonic();report={'scope':__doc__,'gpu_executed':False,'domains':{}}

def digest(value):
    x=np.asarray(value)
    return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}

def difference(x,y,path='',seen=None):
    if x is y:return []
    if seen is None:seen=set()
    pair=(id(x),id(y))
    if pair in seen:return []
    seen.add(pair)
    if type(x) is not type(y):return [{'path':path,'type_x':type(x).__name__,'type_y':type(y).__name__}]
    if hasattr(x,'shape') and hasattr(x,'dtype'):
        a,b=digest(x),digest(y)
        return [] if a==b else [{'path':path,'x':a,'y':b}]
    if dataclasses.is_dataclass(x):
        return [r for f in dataclasses.fields(x) if f.name!='surface_controls'
                for r in difference(getattr(x,f.name),getattr(y,f.name),path+'.'+f.name,seen)]
    if isinstance(x,dict):
        if x.keys()!=y.keys():return [{'path':path,'keys_x':list(x),'keys_y':list(y)}]
        return [r for key in x for r in difference(x[key],y[key],path+f'[{key}]',seen)]
    if isinstance(x,(tuple,list)):
        if len(x)!=len(y):return [{'path':path,'length_x':len(x),'length_y':len(y)}]
        return [r for i,(a,b) in enumerate(zip(x,y,strict=True)) for r in difference(a,b,path+f'[{i}]',seen)]
    if isinstance(x,(str,int,float,bool,type(None))):
        return [] if x==y else [{'path':path,'x':x,'y':y}]
    slots=getattr(type(x),'__slots__',())
    if slots:
        return [r for key in slots if key!='__weakref__' and hasattr(x,key) and hasattr(y,key)
                for r in difference(getattr(x,key),getattr(y,key),path+'.'+key,seen)]
    try:
        equal=bool(x==y)
    except Exception:
        equal=False
    return [] if equal else [{'path':path,'type':type(x).__name__,'repr_x':repr(x)[:200],'repr_y':repr(y)[:200]}]

cfg=pipeline.NestedPipelineConfig(a.case,a.out.parent/'unused-input-audit-output',a.out.parent/'unused-input-audit-proof',hours=3,max_dom=2)
hierarchy,bundles,meta,start,steps,initial=pipeline._load_domains(cfg,('d01','d02'))
for name,bundle in bundles.items():
    off=dataclasses.replace(bundle.namelist,noahmp_options=dataclasses.replace(bundle.namelist.noahmp_options or NoahMPOptions(),opt_crs=2))
    values=sc.prior_controls(off,sorted(sc.LSM_CONTROLS),land_state=initial[name].noahmp_land)
    on=sc.bind_domain_controls(off,{name:values},name,land_state=initial[name].noahmp_land)
    off_rt=jax.tree.unflatten(*reversed(jax.tree.flatten(off)))
    on_rt=jax.tree.unflatten(*reversed(jax.tree.flatten(on)))
    off_clock,on_clock=build_clock_base(off),build_clock_base(on)
    row={'bind_other_field_differences':difference(off,on),
         'off_roundtrip_differences':difference(off,off_rt),
         'on_roundtrip_differences':difference(on,on_rt),
         'roundtrip_other_field_differences':difference(off_rt,on_rt),
         'clock_other_field_differences':difference(off_clock._replace(surface_controls=None),on_clock._replace(surface_controls=None)),
         'control_names':sorted(values),'clock_control_names':sorted(on_clock.surface_controls),
         'priors_match_targets':{key:digest(value)==digest(sc._target(key,off)) for key,value in values.items()},
         'clock_public_values_exact':{key:digest(value)==digest(on_clock.surface_controls[key]) for key,value in values.items()},
         'options_before':dataclasses.asdict(off.noahmp_options),'options_after':dataclasses.asdict(on_rt.noahmp_options)}
    report['domains'][name]=row
report['wall_s']=time.monotonic()-started
report['all_transport_exact']=all(not r[k] for r in report['domains'].values() for k in
    ('bind_other_field_differences','off_roundtrip_differences','on_roundtrip_differences','roundtrip_other_field_differences','clock_other_field_differences')) and all(all(r[k].values()) for r in report['domains'].values() for k in ('priors_match_targets','clock_public_values_exact'))
a.out.write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps({'all_transport_exact':report['all_transport_exact'],'domains':{n:{k:r[k] for k in r if 'differences' in k} for n,r in report['domains'].items()},'wall_s':report['wall_s']}),flush=True)
