"""Real CPU post phases, own 12GiB virtual-memory cap; no shared RAM ticket."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
p=argparse.ArgumentParser();p.add_argument('role',choices=('e41','checker'));p.add_argument('--tree',type=Path,required=True)
p.add_argument('--cache',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--proof',type=Path)
p.add_argument('--pattern',default='jit__advance_chunk*,jit_fused_jit*');p.add_argument('--tools',type=Path)
p.add_argument('--run',type=Path);p.add_argument('--hours',type=int,default=11);p.add_argument('--report-start',type=int,default=8)
a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
assert os.environ.get('GPUWRF_GPU_LOCK_HELD')!='1','CPU post must run after GPU lock release'
assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
env={**os.environ,'JAX_PLATFORMS':'cpu','GPUWRF_JAX_CACHE':'0','PYTHONPATH':str(a.tree/'src')}
smoke=os.environ.get('TOPO_LAUNCH_SMOKE')=='1'
if a.role=='e41':
    # The driver names the scanned cache (smoke substitutions are explicit and recorded there).
    command=[sys.executable,str(a.tree/'scripts/wave_cubin_gate.py'),str(a.cache),str(a.out),a.pattern]
    if a.proof is not None:command+=['--run-proof',str(a.proof)]
else:
    command=[sys.executable,str(a.tools/'forecast_compare.py'),'--out',str(a.out)]
    if smoke:command+=['--launch-check',str(a.run)]
    else:command+=['--old',str(a.run/'old'),'--new',str(a.run/'new'),'--hours',str(a.hours),'--report-start',str(a.report_start)]
print(json.dumps(dict(role=a.role,command=command,RAM_policy='RLIMIT_AS12GiB; no shared RAM slot',CPU_rehearsal=smoke)),flush=True)
raise SystemExit(subprocess.call(command,env=env,stdin=subprocess.DEVNULL))
