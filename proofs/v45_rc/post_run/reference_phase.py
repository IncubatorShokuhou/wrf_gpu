"""Post-GPU scientific adapter API, same scoring/guard code and argv, no GPU access.
The global v9 window flag remains set during this synchronous CPU post phase.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--ready',type=Path,required=True);p.add_argument('--gpu-dir',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
m=json.loads(a.manifest.read_text());ready=json.loads(a.ready.read_text());assert ready['all_model_children_reaped'] and ready['manifest_sha256']==hashlib.sha256(a.manifest.read_bytes()).hexdigest()
for key in ['path','metrics']:
 f=Path(m['rw3_floor'][key]);want=m['rw3_floor']['sha256'if key=='path'else'metrics_sha256'];assert hashlib.sha256(f.read_bytes()).hexdigest()==want
script=Path(m['tools_dir'])/'reference_io.py';spec=importlib.util.spec_from_file_location('post_reference_adapter',script);adapter=importlib.util.module_from_spec(spec);sys.modules[spec.name]=adapter;spec.loader.exec_module(adapter)
auth=json.loads(adapter.GO.read_text());assert auth['in_place_read_only'] and auth['manager_utc']=='2026-10-28T12:47:13Z'
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');index=R/'receipts'/('cpu72_'+m['pin'][:12])/'STAGED_INDEX.json'
if a.phase=='prepare':
 rows=adapter.targets();ag=adapter.load_guard(m['snapshot']);sealed={r['file']:r['sha256']for r in json.loads(adapter.F20.read_text())};result=adapter.prepare(rows,sealed,R/('cpu72_'+m['pin'][:12]),index.parent,ag);adapter.dump(index,result)
else:
 assert a.gpu_dir and a.out
 raise SystemExit(adapter.score_index(index,m['snapshot'],a.gpu_dir,a.out,manifest_path=a.manifest))
