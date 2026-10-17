from pathlib import Path
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse,hashlib,importlib.util,json,os,sys,time
import netCDF4
from gpuwrf.io.netcdf_lock import Dataset
p=argparse.ArgumentParser();p.add_argument('--case',choices=['c21','c22','c23','c24'],required=True);p.add_argument('--kind',choices=['gpu','spread'],required=True);a=p.parse_args()
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');mp={'c21':50,'c22':51,'c23':52,'c24':53}[a.case]
assert os.environ['JAX_PLATFORMS']=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
zero=base/'TP32'/f'{mp}a';producer=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')/a.case
receipt=json.loads((zero/'receipt.json').read_text());original=json.loads((producer/'cpu_run.json').read_text())
assert receipt['status']=='COMPLETE' and receipt['frames']==25 and receipt['input_sha256']==original['input_sha256']
root=base/('TP33_zero' if a.kind=='gpu' else 'TP35_spread')/a.case
root.mkdir(parents=True,exist_ok=False);(root/'input').mkdir()
assert not (zero/'namelist.input').is_symlink()
(root/'input/namelist.input').write_bytes((zero/'namelist.input').read_bytes())
(root/'cpu_run.json').write_bytes((zero/'receipt.json').read_bytes());(root/'CPU_OUTPUTS.json').write_bytes((zero/'outputs.json').read_bytes())
(root/'cpu').symlink_to(zero);other=producer/('gpu' if a.kind=='gpu' else 'cpu');(root/'gpu').symlink_to(other)
if a.kind=='spread':
 for row in json.loads((producer/'CPU_OUTPUTS.json').read_text()):
  assert hashlib.sha256((other/row['file']).read_bytes()).hexdigest()==row['sha256']
source=base/'TP33_tools'
path=source/'proofs/v040/final_validation/score.py';spec=importlib.util.spec_from_file_location('frozen_score',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
netCDF4.set_chunk_cache(1<<20,521,.75)
# Reuse read-only handles under the normal process lock; scientific scorer unchanged.
with ExitStack() as stack:
 handles={}
 class Held:
  def __init__(self,ds):self.ds=ds
  def __enter__(self):return self.ds
  def __exit__(self,*exc):return False
 def held(path,mode='r',*args,**kwargs):
  assert mode=='r' and not args and not kwargs
  key=str(path)
  if key not in handles:handles[key]=stack.enter_context(Dataset(key,'r'))
  return Held(handles[key])
 mod.Dataset=held;start=time.monotonic();result=mod.score(source,root,'d01')
record={'scope':'retained real GPU vs compiler-ZERO original CPU-WRF' if a.kind=='gpu' else 'reference uncertainty: original raw CPU-WRF vs compiler-ZERO CPU-WRF; NOT GPU fidelity', 'case':a.case,'kind':a.kind,'utc':datetime.now(timezone.utc).isoformat(),'zero_source_executable_sha256':receipt['source_executable_sha256'],'input_sha256':receipt['input_sha256'],'score_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'compare_source_sha256':hashlib.sha256((source/'scripts/compare_wrfout_grid.py').read_bytes()).hexdigest(),'manifest_sha256':result['manifest_sha256'],'wall_s':time.monotonic()-start,'verdict':result['verdict'],'paired_frames':result['paired_frames'],'bounded_failures':len(result['failures']),'nonfinite':len(result['nonfinite']),'missing_fields':sorted(set(n for r in result['by_frame'] for n in r['missing_vs_cpu'])),'GPU_execution':False,'cache':'NONE; read-only handles closed on return'}
(root/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
