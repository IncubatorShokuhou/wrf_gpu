"""Once-only real WN3 cached fused-path check of the SDS key repair."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SRC=Path(os.environ.get('SDS_CANDIDATE_SOURCE','<USER_HOME>/wrf_gpu2_lanes/w44-perf/snap_sds_2c941de611'))
OUT=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/gpu/SDSWN3')
CASE=Path('<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227')
sys.path.insert(0,str(SRC/'src'))
assert os.environ['GPUWRF_GPU_LOCK_HELD']=='1'
manifest=json.loads(Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/sealed/BUNDLE_MANIFEST.json').read_text())['inputs'][str(CASE)]
got={}
for name in manifest:
 with (CASE/name).open('rb') as f:got[name]=hashlib.file_digest(f,'sha256').hexdigest()
assert got==manifest
import gpuwrf.cli as cli
from gpuwrf.runtime import aot_cheap_key as ck
assert Path(ck.__file__).resolve().is_relative_to(SRC)
import jax
device=jax.devices()[0]
assert device.platform=='gpu'
events=Counter()
jax.monitoring.register_event_listener(lambda event,**kw:events.update([event]))
walk_counts=Counter()
original_walk=ck._walk
def observed_walk(h,obj,depth):
 walk_counts['nodes']+=1
 if isinstance(obj,jax.ShapeDtypeStruct):walk_counts['sds']+=1
 return original_walk(h,obj,depth)
ck._walk=observed_walk
OUT.mkdir(parents=True,exist_ok=False)
t0=time.monotonic()
rc=cli.main(['run','--input-dir',str(CASE),'--output-dir',str(OUT/'wrfout'),
             '--scratch-dir',str(OUT/'scratch'),'--domains-from-namelist','--hours','3','--emit-initial-history'])
files={}
for f in sorted((OUT/'wrfout').glob('wrfout_d0*')):
 with f.open('rb') as src:files[f.name]=hashlib.file_digest(src,'sha256').hexdigest()
baseline=SRC/'proofs/w43_release/receipts/arm_receipts/gpu/A3_wn3_0227_72h/wrfout_sha.txt'
expected={Path(line.split()[1]).name:line.split()[0] for line in baseline.read_text().splitlines() if line.strip()}
from gpuwrf.runtime.domain_tree import NESTED_AOT_STATUS
r={'schema':'w44-perf.wn3_sds_gpu.v1','source_commit':os.environ.get('SDS_CANDIDATE_COMMIT','2c941de611'),'platform':device.platform,'device':device.device_kind,
   'input_sha256':got,'source_key_sha256':hashlib.sha256(Path(ck.__file__).read_bytes()).hexdigest(),
   'rc':rc,'wall_s':time.monotonic()-t0,'cache_events':dict(events),'aot_status':NESTED_AOT_STATUS,
   'wrfout_sha256':files,'files_equal_release':{name:expected.get(name)==sha for name,sha in files.items()},
   'scope':'regression/cache check only; release GPU outputs are not original CPU-WRF truth',
   'memory_stats':device.memory_stats(),'walk_counts':dict(walk_counts)}
(OUT/'receipt.json').write_text(json.dumps(r,indent=2,default=str)+'\n')
assert rc==0 and len(files)==12
assert (NESTED_AOT_STATUS.get('domains') or {}).get('fused/d02',{}).get('loaded') is True,'fused d02+d03 did not load cached program'
assert all(r['files_equal_release'].values()),'release bytes differ; preserve outputs for diagnosis'
assert events['/jax/compilation_cache/cache_misses']==0,'warm cache miss'
print(json.dumps({'rc':rc,'files':len(files),'all_bytes_equal':True,'cache_misses':0}))
