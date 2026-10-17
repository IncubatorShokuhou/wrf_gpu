"""Bounded private serial WRF run; receipt is not a 4-rank benchmark."""
import json
import os
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

case = Path(sys.argv[1]).resolve()
budget = int(sys.argv[2]) if len(sys.argv) > 2 else 5400
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert len(os.sched_getaffinity(0)) == 1 and next(iter(os.sched_getaffinity(0))) in (0,1,2,3,16,17,18,19,8,9,24,25,12,13,28,29)
assert not Path('/tmp/wrf_gpu2_quiet').exists()
started = datetime.now(timezone.utc).isoformat()
t0 = time.perf_counter()
with (case / 'cpu.log').open('w') as log:
    run = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=10', str(budget), './wrf.exe'],
                           cwd=case / 'cpu', stdout=log, stderr=subprocess.STDOUT,
                           start_new_session=True)
    def terminate(signum, frame):
        if run.poll() is None:
            os.killpg(run.pid, signal.SIGCONT)
            os.killpg(run.pid, signal.SIGTERM)
            try:
                run.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(run.pid, signal.SIGKILL)
                run.wait()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    paused = False
    quiet_pauses = []
    while run.poll() is None:
        quiet = Path('/tmp/wrf_gpu2_quiet').exists()
        if quiet and not paused:
            os.killpg(run.pid, signal.SIGSTOP)
            paused = True
            quiet_pauses.append({'start_s': time.perf_counter() - t0})
        elif not quiet and paused:
            os.killpg(run.pid, signal.SIGCONT)
            paused = False
            quiet_pauses[-1]['end_s'] = time.perf_counter() - t0
        time.sleep(.5)
wall = time.perf_counter() - t0
frames = sorted((case / 'cpu').glob('wrfout_d01_*'))
text = (case / 'cpu.log').read_text(errors='replace')
receipt = {'schema': 'gpuqual-cpu-run-v1', 'started_utc': started,
           'finished_utc': datetime.now(timezone.utc).isoformat(), 'rc': run.returncode,
           'wall_s': wall,
           's_per_fc_h': wall / 24 if run.returncode == 0 and len(frames) == 25 and 'SUCCESS COMPLETE WRF' in text else None,
           'cpu_cores': sorted(os.sched_getaffinity(0)), 'ranks': 1,
           'maxrss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
           'benchmark_scope': 'serial WRF on shared nice19 core, not released 4-rank baseline',
           'frame_count': len(frames), 'success_marker': 'SUCCESS COMPLETE WRF' in text,
           'frame_names': [p.name for p in frames]}
receipt['quiet_pauses'] = quiet_pauses
receipt['timeout_s'] = budget
(case / 'cpu_run.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
sys.exit(0 if run.returncode == 0 and len(frames) == 25 and receipt['success_marker'] else 1)
