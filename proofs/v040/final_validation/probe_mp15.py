"""Isolate original mp15 from c63's nondefault Noah bundle in two own steps."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from prepare import ROOT,RETAINED,PRISTINE,SLOTS,private_executable,sha,write_json
from extra_cases import stage
from cpu_references import append,MAIN
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset

assert os.environ.get('JAX_PLATFORMS')=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
ident=sys.argv[1] if len(sys.argv)>1 else 'p15'
debug=len(sys.argv)>2 and sys.argv[2]=='gdb'
canonical=len(sys.argv)>2 and sys.argv[2]=='canonical'
case=ROOT/ident;assert not case.exists()
groups=parse_namelist(ROOT/'c63/input/namelist.input');groups.pop('noah_mp')
groups['time_control'].update(run_days=0,run_hours=0,run_seconds=36,end_day=15,end_hour=0,end_minute=0,end_second=36,
    history_interval=0,history_interval_s=18)
row=dict(id=ident,geography='Swiss',hours=None,domains=['d01'],
    options={k:groups['physics'][k] for k in SLOTS},extra={},gpu_eligible=False,
    scope='CPU-only diagnostic: mp15 with default Noah, 2ownsteps36s; no24h/GPU qualification')
stage(row,RETAINED/'source_inputs/Swiss',groups);private_executable(case,PRISTINE if canonical else ROOT/'reference_zero')
start=time.monotonic()
with (case/'cpu.log').open('w') as stream:
    cmd=(['gdb','--batch','-ex','set pagination off','-ex','run','-ex','bt 16',
          '-ex','info symbol $pc','-ex','x/48i $pc-96','-ex','info registers',
          '-ex','x/s $rdi','-ex','x/s $rsi','-ex','x/s $rdx','-ex','x/s $r8',
          '--args','./wrf.exe'] if debug else ['./wrf.exe'])
    run=subprocess.run(['timeout','--kill-after=5','180',*cmd],cwd=case/'cpu',stdout=stream,stderr=subprocess.STDOUT)
files=list((case/'cpu').glob('wrfout*'));nonfinite=[];read_errors=[]
for p in files:
    try:
        with Dataset(p) as ds:
            ds.set_auto_mask(False)
            for name,v in ds.variables.items():
                if np.issubdtype(v.dtype,np.number) and not np.isfinite(v[:]).all():nonfinite.append([p.name,name])
    except OSError as exc:read_errors.append([p.name,repr(exc)])
result=dict(rc=None if debug else run.returncode,debugger_rc=run.returncode if debug else None,
    wall_s=time.monotonic()-start,frames=len(files),nonfinite=nonfinite,
    read_errors=read_errors,wrf_success='SUCCESS COMPLETE WRF' in (case/'cpu.log').read_text(errors='replace'),
    scope=row['scope'],original_executable=json.loads((case/'CPU_executable.json').read_text()),input_sha256=row.get('input_sha256'))
write_json(case/'probe.json',result)
for p in (case/'cpu/wrf.exe',case/'wrf.exe'):p.unlink()
append(MAIN/'EXPERIMENTS.tsv','\t'.join((datetime.now(timezone.utc).isoformat(),'v-prep','GPT Sol',
    'VP42-mp15-canonical' if canonical else ('VP41-mp15-gdb' if debug else 'VP40-mp15'),
    'Original-WRF two-factor isolation: remove nondefault Noah bundle from failed c63',json.dumps(result,separators=(',',':')),
    f'artifact={case}/probe.json,{case}/cpu.log;cacheNONE;workCPU16NI19JAXcpuGPUWRF0OMP1timeout180 DT18 twoownsteps36s; no24h/GPUclaim')))
print(json.dumps(result,indent=2))
