"""Run real per-phase receipt checks and CPU E41 after the timed payload exits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from netCDF4 import Dataset

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--phase',choices=['repeatcompare','hostprofile'],required=True)
ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--cpu-probe',type=int,required=True)
a=ap.parse_args();here=Path(__file__).resolve().parent
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
plan=json.loads((here/'REPEAT_PLAN.json').read_text())
arms=plan['warmups']+plan['order'] if a.phase=='repeatcompare' else [['v043','H43']]
checked=[]
for version,arm in arms:
    run=a.root/'runs'/arm/'run'
    if a.cpu_probe:
        p=run/'LAUNCH_SMOKE_MODEL.json';r=json.loads(p.read_text())
        assert r['status']=='PASS_CPU_LAUNCH_SMOKE' and r['model_reached']
        assert r['physical_backend']=='cpu' and not r['gpu_executed']
        assert r['finite_output_leaves']==20
        assert r['product_environment']['JAX_PLATFORMS']=='cuda'
        if a.phase=='hostprofile':assert r['host_profile']['total_calls']>0
    else:
        p=run/'receipt.json';r=json.loads(p.read_text())
        assert r['rc']==0 and r['device']['platform']=='gpu'
        ep=json.loads((run.parent/'endpoint.json').read_text());assert ep['valid'] and len(ep['files'])==6
        for output in (run/'wrfout').glob('wrfout_d0*'):
            with Dataset(output) as dataset:
                dataset.set_auto_mask(False)
                for variable in dataset.variables.values():
                    if np.issubdtype(variable.dtype,np.number):
                        assert np.isfinite(np.asarray(variable[:])).all(),(arm,output.name,variable.name)
        if a.phase=='hostprofile':assert r['w44_host_profile']['total_calls']>0
    checked.append({'version':version,'arm':arm,'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
if a.phase=='repeatcompare':
    summary=json.loads((a.root/'BENCH.json').read_text())
    assert len(summary['arms'])==7  # real parser runs even for truthful CPU-only receipts
representatives={version:arm for version,arm in arms}
e41=[]
for version,arm in representatives.items():
    # CPU device/artifact substitution is explicit; scan the donor once rather
    # than repeating the same GPU executable for three CPU source variants.
    if a.cpu_probe and e41:continue
    start=time.monotonic()
    command=[sys.executable,str(here/'cuda_e41.py'),'--root',str(a.root),'--version',version,'--arm',arm,'--cpu-probe',str(a.cpu_probe)]
    with (a.root/('e41_'+version+'.log')).open('w') as log:
        rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'JAX_PLATFORMS':'cpu','PYTHONPATH':str(here.parents[1]/'src')},timeout=180).returncode
    e41.append({'version':version,'rc':rc,'seconds':time.monotonic()-start})
    assert rc==0,('E41',version,rc)
report={'schema':'w44-perf.cuda-launcher-postflight.v1','phase':a.phase,'cpu_probe':bool(a.cpu_probe),
        'cuda_execution_verified':not bool(a.cpu_probe),'checked_arms':checked,'e41':e41,
        'qualification_scope':'Launcher phases only; no forecast fidelity/release/speed qualification',
        'scope':'CPU phase coverage with an explicitly substituted existing GPU E41 donor' if a.cpu_probe else 'actual CUDA run and own artifacts',
        'status':'PASS'}
(a.root/'POSTFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','phase':a.phase,'cpu_probe':bool(a.cpu_probe),'checked_arms':len(checked)}))
