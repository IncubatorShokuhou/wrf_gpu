"""Deviceless HLO/JAXPR REAL carry and explicit WRF DOUBLE-expression audit."""
import argparse,hashlib,json,os,re,sys,traceback
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('output',type=Path);args=ap.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['GPUWRF_JAX_CACHE']=='0'
sys.path[:0]=[str(args.tree/'src'),str(args.tree/'tests')]
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from test_s2_ssib import _inputs,FIXTURE
from gpuwrf.physics.lsm_ssib import ssib_column
with np.load(FIXTURE,allow_pickle=False) as fixture:
 names=fixture['ordinary_1_names'].tolist();frame=dict(zip(names,fixture['ordinary_1'][0,:,0]))
land,forcing=_inputs(frame)
fn=lambda land,forcing:ssib_column(land=land,forcing=forcing)
closed=jax.make_jaxpr(fn)(land,forcing)
source=args.tree/'src/gpuwrf/physics/_ssib_generated.py';lines=source.read_text().splitlines()
wrf=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_ssib.F').read_text().splitlines()
seen=set();r8=[];outside=[];operations=0

def inspect_graph(value):
 global operations
 if hasattr(value,'jaxpr'):value=value.jaxpr
 if hasattr(value,'eqns'):
  if id(value) in seen:return
  seen.add(id(value))
  for eq in value.eqns:
   operations+=1
   wide=any(getattr(getattr(var,'aval',None),'dtype',None)==np.dtype('float64') for var in (*eq.invars,*eq.outvars))
   if wide:
    frames=[]
    if eq.source_info.traceback is not None:
     frames=[(f.f_code.co_filename,n) for f,n in traceback.walk_tb(eq.source_info.traceback.as_python_traceback())]
    generated=[n for file,n in frames if file.endswith('/_ssib_generated.py')]
    explicit=[n for n in generated if 'jnp.float64(' in lines[n-1]]
    sites=[]
    for n in explicit:
     text=next((s for s in reversed(lines[:n]) if s.lstrip().startswith('# WRF:')),None)
     if text:sites.append(text.strip())
    row={'primitive':eq.primitive.name,'generated_lines':generated,'explicit_double_lines':explicit,'wrf_sites':sites}
    r8.append(row)
    if not explicit:outside.append(row)
   inspect_graph(eq.params)
 elif isinstance(value,dict):
  for child in value.values():inspect_graph(child)
 elif isinstance(value,(tuple,list)):
  for child in value:inspect_graph(child)
inspect_graph(closed)
abstract=jax.eval_shape(fn,land,forcing)
hlo=jax.jit(fn).lower(land,forcing).compiler_ir('hlo').as_hlo_text()
report={'scope':'CPU trace/lower only; no GPU execution or optimized cubin claim',
 'platform':jax.devices()[0].platform,'jaxpr_operations':operations,'fp64_typed_operations':len(r8),
 'fp64_outside_explicit_WRF_DOUBLE_expression_sites':len(outside),'unattributed':outside,
 'fp64_sites':r8,'unoptimized_hlo_f64_lines':sum('f64' in s for s in hlo.splitlines()),
 'carry_fp64_leaves':sum(getattr(x,'dtype',None)==np.dtype('float64') for x in jax.tree.leaves(abstract)),
 'generated_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('fp64_sites','unattributed')}))
assert not outside, outside[:3]
assert report['carry_fp64_leaves']==0
