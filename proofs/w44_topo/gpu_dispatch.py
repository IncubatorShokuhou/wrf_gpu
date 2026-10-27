"""Shared JAX-dark, process-group-owned GPU payload; no pool in supervisor."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import sys
from forecast_driver import load_group
p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--seconds',type=float,required=True);a,command=p.parse_known_args()
if command and command[0]=='--':command=command[1:]
assert command and os.environ['JAX_PLATFORMS']=='cuda'
smoke=os.environ.get('TOPO_LAUNCH_SMOKE')=='1'
group=load_group()
for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(number,group.stop_signal)
def cgroup_limit(name):
    """Effective cgroup-v2 limit (memory.max / memory.swap.max) of this process (None = unlimited)."""
    relative=Path('/proc/self/cgroup').read_text().strip().split('::',1)[-1]
    folder=Path('/sys/fs/cgroup'+relative)
    limits=[]
    while str(folder).startswith('/sys/fs/cgroup') and folder!=Path('/sys/fs/cgroup'):
        value=(folder/name).read_text().strip() if (folder/name).exists() else 'max'
        if value!='max':limits.append(int(value))
        folder=folder.parent
    return min(limits) if limits else None
# E233: the driver starts every model child in its own systemd user scope, no swap
# (smoke 12G: the REAL model runs on CPU; RLIMIT_AS breaks JAX's virtual reservations).
memory,swap=cgroup_limit('memory.max'),cgroup_limit('memory.swap.max')
assert memory is not None and memory<=(12 if smoke else 16)*1024**3 and swap==0,('model child needs its own scope',memory,swap)
normal=[a.lock,'--label','w44-topo','--timeout','1','--',*command]
actual=command if smoke else normal
out=Path(command[command.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
result=group.run_group(actual,dict(os.environ),out/'DEVICE_PAYLOAD.log',a.seconds,group.utc_now()+a.seconds+1)
result.update(production_command=normal,CPU_rehearsal=smoke,cgroup_memory_max=memory,cgroup_swap_max=swap,ended_utc=datetime.now(timezone.utc).isoformat())
(out/'DEVICE_PHASE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
raise SystemExit(result['rc'])
