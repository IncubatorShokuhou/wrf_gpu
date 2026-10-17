from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
base=Path('<USER_HOME>/wrf_gpu2_lanes/t-p3fix');root=base/'TP26_zero/WRF';keep=root/'main/wrf.exe';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not Path('/tmp/wrf_gpu2_quiet').exists()
r=json.loads((base/'TP44_repeat.json').read_text());assert r['all_fields_BYTE'];expected=r['source_executable_sha256'];assert sha(keep)==expected
inventory=json.loads((base/'TP48_cache_inventory.json').read_text());deleted=[];total=0
for row in inventory['files']:
 p=Path(row['path']);assert p.is_relative_to(root) and p.suffix in ('.o','.a','.mod') and p!=keep
 if p.exists() or p.is_symlink():
  assert os.lstat(p).st_size==row['bytes'];p.unlink();deleted.append(str(p));total+=row['bytes']
assert sha(keep)==expected
record={'utc':datetime.now(timezone.utc).isoformat(),'scope':'closed owned Fortran object caches only; shared admitted source executable preserved at unchanged path','deleted_object_count':len(deleted),'deleted_object_bytes':total,'shared_executable':str(keep),'shared_executable_sha256':expected,'source_and_config_preserved':True,'snapshots_absent':all(not p.exists() for p in [base/'tp02-tree',base/'tp16-tree']),'duplicate50b_outputs_deleted':not list((base/'TP32/50b').glob('wrfout_d01_*')),'GPU_execution':False}
(base/'TP48_cleanup.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
