"""Light, read-only cgroup accounting; no CUDA client or model imports."""
import json, os, re, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/v-run/round4_qhmass')
assert os.environ['JAX_PLATFORMS']=='cpu'
names=['mp18','mp18_cu5','mp24','mp26']; scopes={}
with (ROOT/'RAM_SAMPLES.jsonl').open('a',buffering=1) as out:
    while not all((ROOT/n/'GPU_DONE.json').exists() for n in names):
        for name in names:
            log=ROOT/name/'gpu_cuda.log'
            if name not in scopes and log.exists():
                with log.open() as f: head=f.read(300)
                m=re.search(r'pid=(\d+)',head)
                if m:
                    try: scopes[name]=Path('/sys/fs/cgroup')/Path(f'/proc/{m[1]}/cgroup').read_text().split('0::',1)[1].strip().lstrip('/')
                    except FileNotFoundError: pass
        for scope in set(scopes.values()):
            try:
                values={n:(scope/n).read_text().strip() for n in ('memory.current','memory.peak','memory.max','memory.swap.current','memory.events')}
                out.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'scope':str(scope),**values})+'\n')
            except FileNotFoundError: pass
        time.sleep(5)
