"""Freeze one shared command builder and paired production/mini public entries."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--commit',required=True);a=ap.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert not Path('/tmp/wrf_gpu2_gpu_window').exists()
assert os.statvfs('<DATA_ROOT>').f_bavail*os.statvfs('<DATA_ROOT>').f_frsize>=365*1024**3
repo=Path(__file__).resolve().parents[2];L=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse',a.commit],text=True).strip()
root=L/('cuda_tools_'+commit[:10]);root.mkdir(mode=0o700,exist_ok=False)
names=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','--name-only',commit],text=True).splitlines()
selected=[n for n in names if n.startswith('src/') and n.endswith(('.py','.json'))]
selected += ['scripts/bench/s0_nested_harness.py','scripts/bench_summary.py','scripts/release/bundle_consumer.py',
             'scripts/wave_cubin_gate.py','scripts/e41/cubin_scan.py','scripts/e41/hlo_scan.py',
             'tests/v025/b_thompson/fixtures/exit_columns.npz','tests/v025/b_thompson/fixtures/exit_columns.json']
selected += [n for n in names if n.startswith('proofs/w44_perf/') and n.endswith(('.sh','.py','REPEAT_PLAN.json'))]
payload=subprocess.check_output(['git','-C',str(repo),'archive',commit,*selected])
with tarfile.open(fileobj=io.BytesIO(payload)) as archive:archive.extractall(root,filter='data')
(root/'data').mkdir(exist_ok=True);(root/'data/wrf_pristine').symlink_to('<USER_HOME>/src/wrf_pristine')
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and not p.is_symlink()}
case=L/'cuda_mini_case_v1'
inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in case.iterdir() if p.is_file()}
manifest={'schema':'w44-perf.cuda-mini-seal.v1','commit':commit,'files':files,'mini_inputs':inputs,
 'ram':'OWN sequential MemoryMax12G MemorySwapMax0 cgroup; no shared RAM tickets in GPU entry or postphases',
 'grid_difference':'PROD120x70/267x117 -> real-input subsets16x16/15x15; all44lev,physics,dt54/18,3h,10arms or1hostarm unchanged',
 'cpu_substitution':'real1x44 Thompson at device boundary; real bundle checks/summary/receipt checker and actual E41 scan on existing v043 donor; not CUDA or mini numerical qualification',
 'old_production_launchers':{'repeatcompare':'ae09c4aee94bdf251615a6c52a8409ae7f00636c284563c2cb38d522e10f3181','hostprofile':'7fea49889f71be96ffb50dd1e31f9cf7a47870c73ff0c4a1c859a66340bb626e'}}
seal=root/'CUDA_MINI_MANIFEST.json';seal.write_text(json.dumps(manifest,indent=2)+'\n')
digest=hashlib.sha256(seal.read_bytes()).hexdigest();rows=[]
for phase in ['repeatcompare','hostprofile']:
    for size in ['prod','cuda_mini']:
        name=('_cuda_mini_' if size=='cuda_mini' else '_production_')+phase+'_'+commit[:10]+'.sh'
        path=L/name
        body=('#!/usr/bin/env bash\nset -euo pipefail\n'
              'unset W44_V5_INSIDE W44_RUN_ROOT W44_CPU_PROBE\n'
              'export W44_SEAL_SHA256='+digest+'\n'
              'exec bash '+str(root/'proofs/w44_perf/launch_v5.sh')+' '+phase+' '+size+' "$@"\n')
        with path.open('x') as f:f.write(body)
        path.chmod(0o500)
        rows.append({'phase':phase,'size':size,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cap_minutes':5 if size=='cuda_mini' else 10 if phase=='repeatcompare' else 3})
for p in root.rglob('*'):
    if p.is_file() and not p.is_symlink():p.chmod(0o444)
(L/('CUDA_MINI_PACKET_'+commit[:10]+'.json')).write_text(json.dumps({'tools':str(root),'seal_sha256':digest,'launchers':rows,'status':'UNQUALIFIED until CPU probes and manager CUDA window'},indent=2)+'\n')
print(json.dumps({'tools':str(root),'files':len(files),'launchers':rows},indent=2))
