"""Noninteractive, bounded launcher for the immutable qWT prototype.

No JAX/CUDA imports here. The official GPU-lock wrapper and every child run in
a private session/process group. Cleanup covers other groups in that session
(nested timeout/setpgrp), including TERM-ignoring children retaining the flock.
"""
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE = Path('<USER_HOME>/wrf_gpu2_lanes/w44-budget')
SNAP = BASE / 'snapshot_p09'
PIN = '3fa48d159766101770a787fb9947c0622c7c4456'
RUNNER_SHA = '306913415ca3a0f159f69247adaea27052e4183a71edfeb2e0cc5eb9377e985a'
RU = Path('<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh')
CAP = 1800
GPU_CAP = 1320
E41_CAP = 300
CPU_ENV = dict(os.environ, JAX_PLATFORMS='cpu', GPUWRF_JAX_CACHE='0')
SIGNALLED = 0


def boot():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def real_utc():
    return int(subprocess.check_output(['bash', str(RU)], env=CPU_ENV, timeout=8, text=True).strip())


def utc_epoch(text):
    value = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.utcoffset() != dt.timedelta(0):
        raise ValueError('GA times must explicitly use UTC')
    return value.timestamp()


def write_json(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def children(leader):
    """Only this supervisor's descendants or members of its private session."""
    rows = {}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            tail = p.read_text().rsplit(')', 1)[1].split()
            rows[int(p.parent.name)] = (tail[0], int(tail[1]), int(tail[2]), int(tail[3]))
        except (OSError, ValueError, IndexError):
            continue
    owned = {leader}
    while True:
        more = {pid for pid, (_, parent, _, sid) in rows.items()
                if parent == os.getpid() or parent in owned or sid == leader}
        if more <= owned:
            break
        owned |= more
    return {pid: row for pid, row in rows.items() if pid in owned and row[0] != 'Z'}


def reap():
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break


def cleanup(process):
    # PR_SET_CHILD_SUBREAPER also adopts double-fork/setsid descendants, so a
    # group-changing helper cannot escape cleanup by orphaning itself.
    seen = set()
    for sig, seconds in ((signal.SIGTERM, 3), (signal.SIGKILL, 5)):
        end = boot() + seconds
        while True:
            live = children(process.pid)
            seen.update(live)
            if not live:
                break
            for group in {row[2] for row in live.values()}:
                if group != os.getpgrp():
                    try:
                        os.killpg(group, sig)
                    except ProcessLookupError:
                        pass
            for pid in live:  # cover a descendant joining the caller's group
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if boot() >= end:
                break
            time.sleep(.05)
        if process.poll() is not None:
            reap()
    process.wait(timeout=1)
    reap()
    remaining = children(process.pid)
    return dict(signalled_pids=sorted(seen), live_remaining=sorted(remaining))


def run_group(command, log, limit, *, env=None):
    start = boot()
    with open(log, 'w') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        row = dict(pid=process.pid, pgid=process.pid, sid=process.pid, cap_s=limit)
        timed_out = False
        try:
            while process.poll() is None:
                if SIGNALLED or boot() - start >= limit:
                    timed_out = not SIGNALLED
                    break
                time.sleep(.1)
        finally:
            original_rc = process.poll()
            row['cleanup'] = cleanup(process)
        child_rc = 128 - original_rc if original_rc is not None and original_rc < 0 else original_rc
        row.update(elapsed_s=boot() - start, timed_out=timed_out,
                   rc=128 + SIGNALLED if SIGNALLED else 124 if timed_out else child_rc)
        if row['cleanup']['live_remaining']:
            row['rc'] = 125
        elif original_rc == 0 and row['cleanup']['signalled_pids']:
            row['rc'] = 125  # successful parent must not leave background work
        return row


GPU_BODY = r'''
set -euo pipefail
mkdir -p "$RUN/cache/jax" "$RUN/cache/triton" "$RUN/cache/cuda" "$RUN/tmp"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$RUN/gpu_start.txt"
journalctl -k --no-pager 2>/dev/null | rg -i hibernat | tail -n 8 > "$RUN/hibernation.txt" || true
exec env -u JAX_PLATFORMS GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=1 \
  GPUWRF_JAX_CACHE_DIR="$RUN/cache/jax" JAX_COMPILATION_CACHE_DIR="$RUN/cache/jax" \
  JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
  GPUWRF_XLA_AUTOTUNE_CACHE_DIR="$RUN/cache/jax/xla_gpu_per_fusion_autotune_cache_dir" \
  TRITON_CACHE_DIR="$RUN/cache/triton" CUDA_CACHE_PATH="$RUN/cache/cuda" TMPDIR="$RUN/tmp" \
  XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=true \
  XLA_PYTHON_CLIENT_MEM_FRACTION=.22 GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF \
  PYTHONPATH="$SNAP/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u "$SNAP/proofs/w44_budget/coupled_probe.py" \
  --input-dir <DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227 \
  --out "$RUN/probe" --max-dom 3
'''


def main():
    global SIGNALLED
    # Linux child subreaper; no capabilities or privilege changes required.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot install child subreaper')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda number, _frame: globals().__setitem__('SIGNALLED', number))
    run = BASE / ('NIGHT_' + str(time.time_ns()) + '_' + str(os.getpid()))
    run.mkdir()  # atomic Python mkdir, never a shell claim/lock
    record = dict(schema='wrf_gpu2.w44_budget.night.v1', run=str(run), cap_s=CAP,
                  source_commit=PIN, launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  start_env=os.environ.get('GA_START_UTC'), end_env=os.environ.get('GA_END_UTC'),
                  gpu_started=False, phases={}, verdict='PREFLIGHT', rc=2)
    receipt = run / 'night_receipt.json'
    write_json(receipt, record)
    rc = 2
    start_boot = boot()
    try:
        start = utc_epoch(os.environ['GA_START_UTC'])
        end = utc_epoch(os.environ['GA_END_UTC'])
        now = real_utc()
        record.update(start_real_utc=now, remaining_s=end - now)
        if end <= start or now < start or end - now < CAP:
            rc = 4
            record['verdict'] = 'REFUSED_WINDOW'
            return rc
        record['free_bytes'] = {path: shutil.disk_usage(path).free for path in ('/', '<DATA_ROOT>')}
        if record['free_bytes']['/'] < 20 * 2**30 or record['free_bytes']['<DATA_ROOT>'] < 200 * 2**30:
            rc = 5
            record['verdict'] = 'REFUSED_ACTUAL_DISK_FLOOR'
            return rc
        deadline = start_boot + CAP - 20  # reserve group teardown/receipt time
        if os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1':
            raise ValueError('sequencer must call outside any existing GPU lock')
        if subprocess.check_output(['git', '-C', str(SNAP), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
            raise ValueError('immutable source pin changed')
        if subprocess.check_output(['git', '-C', str(SNAP), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise ValueError('dirty immutable snapshot')
        probe = SNAP / 'proofs/w44_budget/coupled_probe.py'
        if hashlib.sha256(probe.read_bytes()).hexdigest() != RUNNER_SHA:
            raise ValueError('immutable numerical runner changed')
        if not (SNAP / 'data/wrf_pristine').exists():
            raise ValueError('missing pristine link')
        if Path('/tmp/wrf_gpu2_quiet').exists():
            rc = 3
            record['verdict'] = 'REFUSED_QUIET'
            return rc
        if SIGNALLED:
            rc = 128 + SIGNALLED
            record['verdict'] = 'CANCELLED_BEFORE_GPU'
            return rc
        env = dict(CPU_ENV, SNAP=str(SNAP), RUN=str(run), OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        # No nested timeout: the outer supervisor owns the complete tree.
        command = [str(SNAP / 'scripts/with_gpu_lock.sh'), '--label', 'w44-budget',
                   '--timeout', '1', '--', 'bash', '-c', GPU_BODY]
        record['verdict'] = 'GPU_REQUESTED'
        write_json(receipt, record)
        row = run_group(command, run / 'gpu.log', min(GPU_CAP, deadline - boot()), env=env)
        record['phases']['gpu'] = row
        record['gpu_started'] = (run / 'gpu_start.txt').exists()
        rc = row['rc']
        (run / 'probe.rc').write_text(str(rc) + '\n')
        write_json(receipt, record)
        if record['gpu_started'] and not SIGNALLED:
            while not SIGNALLED and Path('/tmp/wrf_gpu2_quiet').exists() and deadline - boot() >= E41_CAP + 1:
                time.sleep(.25)
            if SIGNALLED or Path('/tmp/wrf_gpu2_quiet').exists() or deadline - boot() < E41_CAP:
                record['phases']['e41'] = dict(rc=3, verdict='PENDING_WINDOW_OR_QUIET')
                rc = rc or (128 + SIGNALLED if SIGNALLED else 3)
            else:
                command = ['taskset', '-c', '8,9,12,24,25,28', 'nice', '-n', '19',
                    str(SNAP / 'scripts/with_ram_slot.sh'), '--label', 'w44-budget', '--max', '12G',
                    '--timeout', '1', '--', sys.executable, str(SNAP / 'scripts/wave_cubin_gate.py'),
                    str(run / 'cache/jax'), str(run / 'e41')]
                gate = run_group(command, run / 'e41.log', E41_CAP, env=env)
                record['phases']['e41'] = gate
                rc = rc or gate['rc']
        elif record['gpu_started']:
            record['phases']['e41'] = dict(rc=3, verdict='PENDING_SIGNAL')
        (run / 'e41.rc').write_text(str(record['phases'].get('e41', {}).get('rc', 3)) + '\n')
        result = run / 'probe/result.json'
        record['probe_verdict'] = json.loads(result.read_text()).get('verdict') if result.exists() else 'INCOMPLETE'
        if rc == 0 and record['probe_verdict'] != 'FIRST_STEP_AND_3H_BYTE':
            rc = 2
        record['verdict'] = 'COMPLETED' if rc == 0 else 'FAILED_OR_INCOMPLETE'
        record['retained'] = 'Actual cold cache/probe outputs retained pending manager evidence review; no silent deletion before E41 or failed-gate disposition.'
        return rc
    except (KeyError, ValueError, OSError, subprocess.SubprocessError) as error:
        record['error'] = str(error)
        record['verdict'] = 'REFUSED_OR_ERROR'
        return rc
    finally:
        record.update(rc=rc, elapsed_s=boot() - start_boot)
        write_json(receipt, record)
        (run / 'launcher.rc').write_text(str(rc) + '\n')
        print(json.dumps(dict(receipt=str(receipt), rc=rc, verdict=record['verdict'])), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
