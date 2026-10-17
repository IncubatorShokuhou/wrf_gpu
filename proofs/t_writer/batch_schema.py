"""Sequential, bounded initialized-schema arms; preserve each independent result."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

source,root,output=map(lambda p:Path(p).resolve(),sys.argv[1:4])
output.mkdir(parents=True,exist_ok=True)
commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
matrix=json.loads((root/'MATRIX.json').read_text())
selected=set(sys.argv[4:])
for case in matrix['cases']:
    if selected and case['id'] not in selected:
        continue
    folder=output/case['id']
    if (folder/'inventory.json').exists():
        continue  # explicit resume; completed immutable receipts are never rewritten
    folder.mkdir(parents=True,exist_ok=True)
    if case['status']!='ADMITTED':
        (folder/'inventory.json').write_text(json.dumps(dict(case=case['id'],status='NOT_GPU_ELIGIBLE',reason=case['status']))+'\n')
        continue
    log=folder/'init.log'
    env=dict(os.environ,JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',GPUWRF_FAST_DEFAULTS='1',
             GPUWRF_FULL_WRFOUT_VARIABLES='1',GPUWRF_CENSUS='0',OMP_NUM_THREADS='1',PYTHONPATH=str(source/'src'))
    while Path('/tmp/wrf_gpu2_quiet').exists():
        time.sleep(5)
    with log.open('w') as stream:
        try:
            run=subprocess.run(['bash','-c',
                'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"',
                't-writer-schema',sys.executable,str(source/'proofs/t_writer/initialized_schema.py'),
                str(root/case['id']),str(folder)],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
            rc=run.returncode
        except subprocess.TimeoutExpired:
            rc=124
    if not (folder/'inventory.json').exists():
        (folder/'inventory.json').write_text(json.dumps(dict(case=case['id'],status='INCOMPLETE',rc=rc,log=str(log)))+'\n')
    rec=json.loads((folder/'inventory.json').read_text())
    rec['source_commit']=commit
    (folder/'inventory.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(case['id'],rec['status'],rec.get('error'),rec.get('wall_s'),flush=True)
    stamp=datetime.now(timezone.utc).isoformat()
    line=[stamp,'t-writer','Codex','TW08-'+case['id'],'Real initialization writer exposes planned schema without forecast',
          'immutable'+commit,'CPU schema/producer admission','no emitted gate',
          f"[M CPU]status={rec['status']} rc={rc} wall={rec.get('wall_s')} error={rec.get('error')}",
          'KEEP honest inventory; not scientific validation',
          f'artifact={folder}/inventory.json+init.log;cacheNONE GPUWRF_JAX_CACHE0;workCPU8,9,24,25NI19QUIETtimeout300 FAST1 CENSUS0 noGPU noforecast originalinputreadonly']
    with Path('<USER_HOME>/src/wrf_gpu2/.agent/kernel/EXPERIMENTS.tsv').open('a') as stream:
        stream.write('\t'.join(v.replace('\n',' ') for v in line)+'\n')
