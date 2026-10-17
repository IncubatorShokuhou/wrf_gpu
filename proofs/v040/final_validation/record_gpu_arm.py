"""Host-only launch journal, including bounded arms terminated before finally."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

assert os.environ.get('JAX_PLATFORMS')=='cpu'
ident,started,rc,budget,fraction=sys.argv[1:]
assert len(ident)==3 and ident[0]=='c' and ident[1:].isdigit()
case=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')/ident
inputs=json.loads((case/'inputs.json').read_text())
receipt_path=case/'gpu_run.json'
receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
finished=datetime.now(timezone.utc).isoformat()
result=dict(started_utc=started,finished_utc=finished,process_rc=int(rc),
    timeout_s=int(budget),pool_fraction=float(fraction),
    gpu_process_receipt_present=receipt is not None,
    platform_asserted=None if receipt is None else receipt['platform'],
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    work=inputs,cache=str(case/'cache'),
    scope='actual GPU qualification launch; no scientific PASS inferred')
(case/'gpu_launch.json').write_text(json.dumps(result,indent=2)+'\n')
wall=None if receipt is None else receipt['monotonic_wall_s']
line='\t'.join((finished,'v-prep','GPT Sol','VP-GPU-'+ident,
    'Final composite original-WRF qualification arm',result['source_commit'],
    f'[M]rc{rc} platform{result["platform_asserted"]} wall_s{wall};D6/E41 pending',
    f'artifact={case}/gpu_launch.json,{case}/gpu_run.json;cache={case}/cache;'
    f'work=hours{inputs["hours"]},domains{inputs["domains"]},namelistSHA{inputs["input_sha256"]["namelist.input"]},timeout{budget},pool{fraction}'))+'\n'
fd=os.open('<USER_HOME>/src/wrf_gpu2/.agent/kernel/EXPERIMENTS.tsv',os.O_WRONLY|os.O_APPEND)
try:os.write(fd,line.encode())
finally:os.close(fd)
