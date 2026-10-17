"""Pristine Swiss WRF short-window oracle; never modifies a reference or shared build.

Usage (CPU12, nice19, QUIET guard): python build_short_oracle.py {dfl-default,dfl-endonly,ddfi}
Requires the lane-owned clean RD11 executable and primary oracle prepared for this lane.
"""
from __future__ import annotations
import os
from pathlib import Path
import re
import subprocess
import sys

LANE = Path('<USER_HOME>/wrf_gpu2_lanes/s1-dfi12')
KIND = sys.argv[1]
if KIND not in ('dfl-default','dfl-endonly','ddfi'):
    raise ValueError(KIND)
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
base = LANE/'oracle/opt1'
out = LANE/'oracle'/f'{KIND}-short'
out.mkdir()  # refuse overwrite of an existing oracle
for file in base.iterdir():
    if file.name.startswith(('wrfout','rsl.','wrfinput_initialized','namelist','wrf.stdout','run.pid')):
        continue
    (out/file.name).symlink_to(file.resolve())
text = (base/'namelist.input').read_text()
edits = dict(dfi_opt=2 if KIND=='ddfi' else 1, dfi_cutoff_seconds=1200,
             dfi_bckstop_day=14, dfi_bckstop_hour=23, dfi_bckstop_minute=59, dfi_bckstop_second=24,
             dfi_fwdstop_minute=0, dfi_fwdstop_second=36,
             run_hours=0, run_seconds=60, end_hour=0, end_minute=1, end_second=0)
if KIND == 'dfl-default':
    edits.update(diff_opt_dfi=0,km_opt_dfi=1)
if KIND == 'dfl-endonly':
    edits['run_seconds'] = 0
for key,value in edits.items():
    text = re.sub(rf'(?m)^\s*{key}\s*=.*$',f' {key} = {value},',text)
(out/'namelist.input').write_text(text)
with (out/'wrf.stdout').open('w') as log:
    subprocess.run(['timeout','--kill-after=10s','600',
                    '<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/mpirun',
                    '--bind-to','none','-np','1','./wrf.exe'],cwd=out,stdout=log,stderr=subprocess.STDOUT,
                   env={**os.environ,'OMP_NUM_THREADS':'1'},check=True)
assert 'SUCCESS COMPLETE WRF' in (out/'rsl.out.0000').read_text()
assert (out/'wrfinput_initialized_d01').is_file()
print(KIND,'SUCCESS',out)
