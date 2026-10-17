"""Retained MP40 GPU-file augmentation by admitted host-only writer providers."""
import ast,hashlib,importlib.util,json,os,resource,shutil,sys,time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from datetime import datetime,timezone
import numpy as np
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/v-run/round3/mp40_writer_reuse')
SOURCE=Path('<DATA_ROOT>/wrf_gpu2/scratch/v-run/round3/snapshot_5abde3224')
sys.path.insert(0,str(SOURCE/'src'))
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.producer_history import add_fields,inactive_names,CCN_FIELDS
from gpuwrf.io.wrf_variable_metadata import WRF_VARIABLE_METADATA
from gpuwrf.io.wrf_history_metadata import wrf_history_global_attributes
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
old=Path('<USER_HOME>/wrf_gpu2_lanes/v-run/mp40');cpu=old/'cpu'
output=Path('<DATA_ROOT>/wrf_gpu2/scratch/v-run/round3/runtime/mp40_writer_reuse/gpu');output.mkdir(parents=True,exist_ok=True)

if not (ROOT/'gpu').is_symlink():(ROOT/'gpu').symlink_to(output,target_is_directory=True)

if not (ROOT/'cpu').is_symlink():(ROOT/'cpu').symlink_to(cpu,target_is_directory=True)
prior=json.loads((old/'grid_gate.json').read_text());assert prior['verdict']=='INCOMPLETE' and not prior['failures'] and prior['bounded_field_count']==62
identity=json.loads((ROOT.parent/'conditional_identity/ADJUDICATION.json').read_text())['configs']['mp40']
assert identity['classification']=='MODEL_PROGRAM_BYTE_REUSE_SCIENCE'
actual=json.loads((old/'gpu/proofs/nested_pipeline_run.json').read_text())['metadata']['domains']['d01']['namelist_resolved']
lookup=lambda obj,name,default=None: default if obj is None else obj.get(name,default)
assert int(actual['mp_physics'])==40 and int(actual.get('aercu_opt',0))==0
names=inactive_names(actual,lookup);assert set(names)==set(CCN_FIELDS) and len(names)==11
# Literal current production attribute writer, extracted to avoid importing unrelated physics/JAX.
writer=SOURCE/'src/gpuwrf/io/wrfout_writer.py';tree=ast.parse(writer.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_set_variable_attrs')
namespace={'np':np,'Any':Any,'WrfoutVariableSpec':SimpleNamespace};exec(compile(ast.Module(body=[node],type_ignores=[]),str(writer),'exec'),namespace)
set_attrs=namespace['_set_variable_attrs']
control=parse_namelist(old/'input/namelist.input');attrs=wrf_history_global_attributes(old/'input/wrfinput_d01',control,'d01',18.)
excluded={'GRID_ID','TITLE','START_DATE','SIMULATION_START_DATE','WEST-EAST_GRID_DIMENSION','SOUTH-NORTH_GRID_DIMENSION','BOTTOM-TOP_GRID_DIMENSION','DX','DY','Conventions','history'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
frames=[];t0=time.perf_counter();original_fieldframes=0;constant_fieldframes=0
for original in sorted((old/'gpu').glob('wrfout_d01*')):
 target=output/original.name;reference=cpu/original.name
 assert reference.is_file() and not target.exists() and target.resolve()!=original.resolve()
 oldsha=sha(original);shutil.copyfile(original,target)
 with Dataset(reference,'r') as ref, Dataset(target,'a') as dst:
  ref.set_auto_mask(False);dst.set_auto_mask(False)
  assert all(n in ref.variables and n not in dst.variables for n in names)
  dimensions={k:len(v) for k,v in dst.dimensions.items()};fields={};specs={}
  for n in names:
   var=ref[n];assert tuple(var.dimensions)==('Time','bottom_top','south_north','west_east')
   a=var[:];assert np.isfinite(a).all() and a.dtype==np.float32 and np.all(a==np.float32(0)),(original.name,n)
   meta=WRF_VARIABLE_METADATA[n]
   specs[n]=SimpleNamespace(name=n,dtype='f4',dimensions=tuple(var.dimensions),memory_order=meta['MemoryOrder'],description=meta['description'],units=meta['units'],stagger=meta['stagger'],coordinates=meta.get('coordinates'))
  add_fields(fields,state=None,namelist=actual,diagnostics={},dimensions=dimensions,requested_names=set(names),lookup=lookup,coerce=lambda *a,**k:(_ for _ in ()).throw(AssertionError('active profile forbidden')),specs=specs,shape_for_dimensions=lambda dims,sizes:tuple(int(sizes[k]) for k in dims if k!='Time'))
  assert set(fields)==set(names)
  for n,value in fields.items():
   var=dst.createVariable(n,'f4',specs[n].dimensions);set_attrs(var,specs[n]);var[0,...]=value
   assert np.array_equal(var[:],ref[n][:]);constant_fieldframes+=1
  # Exact current production metadata override table applies to existing fields too.
  for n,var in dst.variables.items():
   if n in WRF_VARIABLE_METADATA:
    meta=WRF_VARIABLE_METADATA[n]
    spec=SimpleNamespace(name=n,dtype=var.dtype.str.lstrip('<>='),dimensions=var.dimensions,memory_order=meta['MemoryOrder'],description=meta['description'],units=meta['units'],stagger=meta['stagger'],coordinates=meta.get('coordinates'))
    set_attrs(var,spec)
  for key,value in attrs.items():
   if key not in excluded:dst.setncattr(key,value)
 with Dataset(original,'r') as before,Dataset(target,'r') as after,Dataset(reference,'r') as ref:
  before.set_auto_mask(False);after.set_auto_mask(False);ref.set_auto_mask(False)
  for n,var in before.variables.items():
   a,b=var[:],after[n][:];assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),(original.name,n)
   original_fieldframes+=1
  missing=sorted(set(ref.variables)-set(after.variables));assert missing==['MSKF_REFL_10CM'],missing
  schema=[]
  for n in set(ref.variables)&set(after.variables):
   a,b=ref[n],after[n]
   assert a.dimensions==b.dimensions and a.shape==b.shape and a.dtype==b.dtype,(original.name,n)
   for key in a.ncattrs():
    assert key in b.ncattrs() and np.array_equal(a.getncattr(key),b.getncattr(key)),(original.name,n,key)
  globaldiff=[key for key in ref.ncattrs() if key not in after.ncattrs() or not np.array_equal(ref.getncattr(key),after.getncattr(key))]
  assert set(globaldiff)<= {'TITLE','NTASKS_X','NTASKS_Y'},globaldiff
 assert sha(original)==oldsha
 frames.append({'raw_original':str(original),'raw_original_sha256':oldsha,'re_emitted':str(target),'re_emitted_sha256':sha(target),'missing_vs_CPU':missing,'global_differences':globaldiff,'original_variables_BYTE':True,'new11_constants_CPU_BYTE':True})
assert len(frames)==25 and constant_fieldframes==275
result={'schema':'mp40-retained-writer-reuse-v1','utc':datetime.now(timezone.utc).isoformat(),'numerical_source':'8afb83d7ddef7ef8e59f7f9635d3b5841b39d23c','writer_source':'5abde3224a50e5f67674bb1865aa0cc6d1ff1937','writer_identity_current195':'src/data/scripts BYTE per MP28 SOURCE_REUSE.json','GPU_forecasts_started':0,'program_identity':identity,'original_variable_fieldframes_BYTE':original_fieldframes,'new_constant_fieldframes_CPU_BYTE':constant_fieldframes,'new_fields':list(names),'coverage_verdict':'INCOMPLETE','missing_CPU_fields':['MSKF_REFL_10CM'],'bounded_D6':'PASS_CARRIED_FORWARD_BY_PROGRAM_CONSTANT_INPUT_AND_OUTPUT_BYTE_PROOF','bounded_fields':62,'numerical_validation_repeated':False,'E41':'Old8afb actualartifact FAIL disclosed; no newCUDA artifact qualification claimed.','F2':'Originalcase40 disposition OPEN retained; no transfer from anothercase.','writer_provider_sha256':sha(SOURCE/'src/gpuwrf/io/producer_history.py'),'attribute_function_AST_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'wall_s':time.perf_counter()-t0,'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'frames':frames,'scope':'Host-only augmentation of retained original GPU outputs. Qualified inactive originalWRF0 only; no synthetic MSKF radar, no physical recompute. Every old variable BYTE and each new field CPU BYTE in all25frames.'}
(ROOT/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('coverage_verdict','missing_CPU_fields','bounded_D6','original_variable_fieldframes_BYTE','new_constant_fieldframes_CPU_BYTE','wall_s','host_peak_rss_kib')}),flush=True)
