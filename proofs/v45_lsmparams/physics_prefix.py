"""Coupled bisection: actual parent first physics forcing, before RK/acoustics.

Same real case/inputs, no model edits. A separately compiled prefix may change
fusion; use it to localize, never claim original-program transparency/fidelity.
"""
import argparse
import dataclasses
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--case',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--cut',choices=('physics','pre_pbl','canopy_inputs'),default='physics')
p.add_argument('--groups',nargs='*',help='Control subsets as comma-separated keys; OFF is always first')
p.add_argument('--save-arrays',action='store_true',help='Save canopy-input arrays for operand bisection')
a=p.parse_args()
assert not a.save_arrays or a.cut=='canopy_inputs'
assert os.environ['JAX_PLATFORMS']=='cpu'
assert subprocess.run(['git','-C',str(a.source),'symbolic-ref','--quiet','HEAD'],capture_output=True).returncode!=0
sys.path[:0]=[str(a.source/'src'),str(a.source/'proofs/o1_nlbind')]
a.out.mkdir(exist_ok=False,parents=True)
start=time.monotonic();rec={'scope':__doc__,'source_commit':subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),'phase':'imports','gpu_executed':False}
def save():
 rec.update(wall_s=time.monotonic()-start,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (a.out/'receipt.json').write_text(json.dumps(rec,indent=2,default=str)+'\n')
 print(rec['phase'],round(rec['wall_s'],1),rec['peak_rss_KiB'],flush=True)
save();rc=1
try:
 import jax
 import jax.numpy as jnp
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
 from gpuwrf.runtime import operational_mode as om
 from gpuwrf.io.noahmp_options import NoahMPOptions
 from gpuwrf.surface_params import controls as sc
 assert all(d.platform=='cpu' for d in jax.devices())
 facade.cuda_native=lambda:False
 state_contract._gpu_device=lambda:jax.devices('cpu')[0]
 sc.UNAVAILABLE_SURFACE_CONTROLS=frozenset()  # local research process only
 rec['phase']='init';save()
 cfg=pipeline.NestedPipelineConfig(a.case,a.out/'unused-output',a.out/'unused-proof',hours=3,max_dom=1)
 hierarchy,bundles,meta,run_start,dts,initial=pipeline._load_domains(cfg,('d01',))
 base=dataclasses.replace(bundles['d01'].namelist,noahmp_options=dataclasses.replace(bundles['d01'].namelist.noahmp_options or NoahMPOptions(),opt_crs=2))
 prior=sc.prior_controls(base,sorted(sc.LSM_CONTROLS),land_state=initial['d01'].noahmp_land)
 groups=a.groups if a.groups is not None else [','.join(sorted(prior))]
 arms=[('off',base)]
 for i,group in enumerate(groups):
  keys=group.split(',');assert keys and set(keys)<=set(prior)
  values={key:prior[key] for key in keys}
  arms.append((f'on{i}',sc.bind_domain_controls(base,{'d01':values},'d01',land_state=initial['d01'].noahmp_land)))
 om._resolve_operational_suite(base)
 for _,nl in arms[1:]:om._resolve_operational_suite(nl)
 rec['control_groups']=dict(zip((label for label,_ in arms[1:]),groups))
 rec['options']=dataclasses.asdict(base.noahmp_options);rec['dt']=base.dt_s;rec['cut']=a.cut
 class CapturedSurface(Exception):
  def __init__(self,state,carry):self.state,self.carry=state,carry
 class CapturedCanopy(Exception):
  def __init__(self,args):self.args=args
 if a.cut=='canopy_inputs':
  from gpuwrf.physics.noahmp import energy
  original_column_call=energy.column_call
  def capture_canopy(fn,args,**kwargs):
   if kwargs.get('name')=='b_core_noah_canopy_flux':raise CapturedCanopy(args)
   return original_column_call(fn,args,**kwargs)
  energy.column_call=capture_canopy
 if a.cut=='pre_pbl':
  source_lines,first_line=inspect.getsourcelines(om._physics_step_forcing)
  candidates=[first_line+i for i,line in enumerate(source_lines) if 'pbl_entry_state = next_state' in line and '_small_grid_firewall' in line]
  assert len(candidates)==1
  cut_line=candidates[0];rec['cut_source_line']=cut_line
  original_firewall=om._small_grid_firewall
  def capture_surface(values,shape):
   frame=sys._getframe(1)
   if frame.f_code is om._physics_step_forcing.__code__ and frame.f_lineno==cut_line:
    assert len(values)==2
    raise CapturedSurface(*values)
   return original_firewall(values,shape)
  om._small_grid_firewall=capture_surface
 def prefix(carry,nl,step_index,cb):
  lower_fields=None
  if nl.lower_boundary is not None:
   from gpuwrf.io.lower_boundary import lower_boundary_fields
   lower=getattr(cb,'lower_boundary',None)
   lower=nl.lower_boundary if lower is None else lower
   lower_fields=lower_boundary_fields(lower,step_index-jnp.int32(1))
   if nl.noahmp_static is not None:
    nl=dataclasses.replace(nl,noahmp_static=nl.noahmp_static.replace(
      shdfac=lower_fields['VEGFRA'].astype(nl.noahmp_static.shdfac.dtype)/100.))
  lead=(step_index.astype(jnp.float64)-1.)*float(nl.dt_s)
  radiation=jnp.equal(jnp.mod(step_index-1,int(nl.radiation_cadence_steps)),0)
  try:
   result=om._physics_step_forcing(carry,nl,lead,run_radiation=radiation,
              first_timestep=jnp.equal(step_index,1),clock_base=cb,lower_fields=lower_fields)
  except CapturedCanopy as canopy:
   from gpuwrf.physics.noahmp.energy import _vege_flux,_bare_flux
   vege,bare=canopy.args
   return {'vege':dict(zip(inspect.signature(_vege_flux).parameters,vege)),
           'bare':dict(zip(inspect.signature(_bare_flux).parameters,bare))}
  except CapturedSurface as surface:
   state_names=('u','v','w','theta','p_total','p_perturbation','hfx','qfx','t_skin','ustar','mol','qv','pblh','qke','qsfc')
   return {'state':{name:getattr(surface.state,name) for name in state_names},
           'carry':surface.carry,'dry':{},'enabled':jnp.bool_(True)}
  if a.cut!='physics':raise AssertionError('actual production prefix cut was not reached')
  state_names=('u','v','w','theta','p_total','p_perturbation','hfx','qfx','t_skin','ustar','mol','qv','pblh','qke','qsfc')
  return {'state':{name:getattr(result.state,name) for name in state_names},
          'carry':result.carry,'dry':{f.name:getattr(result.dry_tendencies,f.name)
              for f in dataclasses.fields(result.dry_tendencies)},'enabled':result.enabled}
 fn=jax.jit(prefix)
 host={}
 for label,nl in arms:
  carry=jax.block_until_ready(jax.tree.map(lambda v:v.copy() if isinstance(v,jax.Array) else v,initial['d01']))
  cb=om.build_clock_base(nl)
  records=getattr(nl,'record_windows',None)
  if records is not None:carry,cb=records.prepare(carry,cb,1,1)
  rec['phase']='physics_'+label;save()
  result=fn(carry,nl,jnp.int32(1),cb);jax.block_until_ready(result)
  host[label]={jax.tree_util.keystr(path):np.asarray(v).copy() for path,v in jax.tree_util.tree_flatten_with_path(result)[0]}
  nonfinite={key:int(np.count_nonzero(~np.isfinite(v))) for key,v in host[label].items() if not np.isfinite(v).all()}
  rec.setdefault('nonfinite',{})[label]=nonfinite
  # Raw solver operands contain WRF's unused water placeholders. Preserve and
  # report them at the diagnostic cut; completed surface/state cuts stay finite.
  if a.cut!='canopy_inputs':assert not nonfinite
  if a.save_arrays:np.savez(a.out/(label+'.npz'),**host[label])
  rec.setdefault('leaves',{})[label]=len(host[label]);rec.setdefault('cache_size',{})[label]=fn._cache_size()
  if label!='off':
   differences=[]
   for key,x in host['off'].items():
    y=host[label][key]
    assert x.shape==y.shape and x.dtype==y.dtype
    if x.tobytes()==y.tobytes():continue
    same=(x==y)|(np.isnan(x)&np.isnan(y))
    error=np.where(same,0.,np.abs(x.astype(np.float64)-y.astype(np.float64)))
    index=np.unravel_index(error.argmax(),error.shape)
    differences.append({'leaf':key,'cells':int(np.count_nonzero(~same)),
        'max_abs':float(np.nanmax(error)),'off_max':float(np.nanmax(np.abs(x.astype(np.float64)))),
        'worst_index':[int(i) for i in index],'off_at_worst':float(x[index]),'on_at_worst':float(y[index])})
   rec.setdefault('comparisons',{})[label]={'default_byte_exact':not differences,'differences':differences}
   del host[label]
  del result,carry,cb
  jax.clear_caches();gc.collect()
  import ctypes
  ctypes.CDLL(None).malloc_trim(0)
  save()
 rec.update(phase='COMPLETE')
 rc=0
except BaseException as exc:
 import traceback
 rec.update(phase='ERROR',exception=repr(exc),traceback=traceback.format_exc())
finally:
 save();sys.stdout.flush();sys.stderr.flush();os._exit(rc)
