"""Two independent Swiss arms under one official lock; no parent CUDA client."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-run/round3/mp28_followup')
plan = json.loads((ROOT / 'RUN_PLAN.json').read_text())
source = Path(plan.get('source', plan['configs'][0]['source']))
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
fd = int(os.environ['GPUWRF_GPU_LOCK_FD'])
lock_stat = os.stat(os.environ['GPUWRF_GPU_LOCK_FILE'])
assert fd > 2 and (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (lock_stat.st_dev, lock_stat.st_ino)
memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free,memory.total',
                                  '--format=csv,noheader,nounits'], text=True).strip()
free = int(memory.split(',')[0])
assert free >= 12000, memory  # prior measured Swiss peaks 1.6-3.4 GiB per PID
print('GPU batch memory free,total MiB:', memory, flush=True)
assert 1 <= len(sys.argv[1:]) <= 2
jobs = []
for name in sys.argv[1:]:
    config = next(c for c in plan['configs'] if c['config'] == name)
    case = ROOT / name
    runtime = Path(config.get('runtime', str(case)))
    env = dict(os.environ, JAX_PLATFORMS='cuda', GPUWRF_JAX_CACHE='1', GPUWRF_CENSUS='1',
        GPUWRF_AOT_SLIM_MODULE='0', GPUWRF_RRTMG_MP_RE=config['GPUWRF_RRTMG_MP_RE'],
        JAX_COMPILATION_CACHE_DIR=str(runtime / 'cache/jax'), GPUWRF_CACHE=str(runtime / 'cache'),
        GPUWRF_XLA_AUTOTUNE_CACHE_DIR=str(runtime / 'cache/autotune'),
        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(source / 'src'), GPUQUAL_CASE=str(case), GPUQUAL_SOURCE=str(source),
        GPUQUAL_RUNTIME=str(runtime))
    cmd = ['timeout', '--signal=TERM', '--kill-after=15', '1170', sys.executable,
           str(ROOT / 'tools/gpu_cli.py'), 'run', '--input-dir', str(case / 'input'),
           '--output-dir', str(runtime / 'gpu'), '--hours', '24']
    (case / 'launch.json').write_text(json.dumps({'argv': cmd, 'official_lock_label': 'v-run',
        'concurrent_group': sys.argv[1:], 'source_rev': config['source_rev'],
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {k: env[k] for k in ('JAX_PLATFORMS', 'GPUWRF_CENSUS',
            'GPUWRF_AOT_SLIM_MODULE', 'GPUWRF_RRTMG_MP_RE', 'JAX_COMPILATION_CACHE_DIR',
            'GPUWRF_CACHE', 'GPUWRF_XLA_AUTOTUNE_CACHE_DIR', 'XLA_PYTHON_CLIENT_PREALLOCATE')}}, indent=2) + '\n')
    with (case / 'gpu_cuda.log').open('w') as log:
        job = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
    jobs.append((case, job))
failed = False
while jobs:
    for case, job in list(jobs):
        rc = job.poll()
        if rc is None:
            continue
        failed |= rc != 0
        (case / 'GPU_DONE.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
            'rc': rc, 'pid': job.pid}) + '\n')
        print(case.name, 'finished rc', rc, flush=True)
        jobs.remove((case, job))
    if jobs:
        time.sleep(1)
sys.exit(1 if failed else 0)
