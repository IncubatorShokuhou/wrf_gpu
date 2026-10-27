"""Immutable manager-sequenced native shadow oracle. No GPU self-start watcher.

The parent stays JAX-dark. Every child starts a new session/process group;
normal completion, timeout, signal and exception all clean that whole group.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

LANE = Path('<USER_HOME>/wrf_gpu2_lanes/w44-topo')
REQUEST = LANE / 'GPU_REQUEST.json'
REQUEST_SHA = '7738fa8e3d6e1f5d267b87899b16b2923b31fee627b6eeead9c92230d45c2bf6'
REAL = Path('<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh')
CAP = 180


def utc_now():
    return int(subprocess.check_output([str(REAL)], text=True, timeout=3).strip())


def epoch(text):
    stamp = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError('GA window must use explicit UTC')
    return stamp.timestamp()


def live_members(pgid):
    result = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdecimal():
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != 'Z':
                    result.append(int(entry.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    return result


def stop_group(proc):
    pgid = proc.pid
    assert pgid != os.getpgrp(), 'refuse parent process-group signal'
    found = live_members(pgid)
    if found:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        limit = time.monotonic() + 3
        while live_members(pgid) and time.monotonic() < limit:
            time.sleep(.05)
        if live_members(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        raise RuntimeError('owned child group did not terminate')
    if live_members(pgid):
        raise RuntimeError(f'owned descendants remain after group kill: pgid={pgid} pids={live_members(pgid)}')
    return found


def run_group(command, env, log_path, seconds, hard_end):
    began = time.monotonic()
    proc = None
    result = {'command': command, 'rc': None, 'pgid': None, 'group_cleaned': False}
    with log_path.open('w') as log:
        try:
            proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result['pgid'] = proc.pid
            while proc.poll() is None:
                # REAL UTC includes suspend time, unlike CLOCK_MONOTONIC.
                if time.monotonic() - began >= seconds or utc_now() >= hard_end:
                    result['rc'] = 124
                    break
                time.sleep(.2)
            if result['rc'] is None:
                result['rc'] = proc.returncode
        finally:
            if proc is not None:
                result['stragglers_at_cleanup'] = stop_group(proc)
                result['group_cleaned'] = True
            result['wall_s'] = time.monotonic() - began
    return result


def preflight():
    assert hashlib.sha256(REQUEST.read_bytes()).hexdigest() == REQUEST_SHA, 'request changed'
    request = json.loads(REQUEST.read_text())
    tree = Path(request['model_tree'])
    assert subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip() == request['model_pin']
    assert not subprocess.check_output(['git', '-C', str(tree), 'status', '--porcelain', '--untracked-files=no'])
    assert (tree / 'data/wrf_pristine').resolve() == Path('<USER_HOME>/src/wrf_pristine')
    tools = Path(request['wrapper']).parent
    for name, digest in request['tools']['files'].items():
        assert hashlib.sha256((tools / name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256(Path(request['manifest']).read_bytes()).hexdigest() == request['manifest_sha256']
    return request, tree, tools


def selftest(directory):
    lock = directory / 'child.lock'
    ready = directory / 'child.ready'
    # A grandchild deliberately inherits this PRIVATE test flock. No GPU lock.
    code = ('import fcntl,os,pathlib,subprocess,sys,time;'
            'f=open(sys.argv[1],"w");fcntl.flock(f,fcntl.LOCK_EX);'
            'subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],pass_fds=(f.fileno(),));'
            'pathlib.Path(sys.argv[2]).write_text("ready");time.sleep(30)')
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
    result = run_group([sys.executable, '-c', code, str(lock), str(ready)], env,
                       directory / 'group.log', 2, utc_now() + 10)
    assert result['rc'] == 124 and result['group_cleaned'] and ready.exists(), result
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return result


def stop_signal(signum, _frame):
    raise InterruptedError(f'night launcher signal {signum}')


def hibernation_record(path):
    try:
        return run_group(['bash', '-c',
            "journalctl -k --since '-3h' --no-pager 2>&1 | /usr/bin/grep -iE 'hibernat|permission|denied' || true"],
            {**os.environ, 'JAX_PLATFORMS': 'cpu'}, path, 2, utc_now() + 3)
    except Exception as exc:
        # Accounting uncertainty is diagnostic; it cannot turn a safe physics
        # arm into an invented failure or an invented no-hibernation claim.
        return {'error': repr(exc), 'known': False}


def main():
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    if Path('/tmp/wrf_gpu2_quiet').exists():
        print('QUIET', flush=True)
        return 3
    if os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1':
        raise ValueError('night launcher must acquire its own lock')
    request, tree, tools = preflight()
    if sys.argv[1:] == ['--check-window']:
        start, end, current = epoch(os.environ['GA_START_UTC']), epoch(os.environ['GA_END_UTC']), utc_now()
        return 0 if current >= start and end - current >= CAP else 5
    if sys.argv[1:] == ['--dry-run']:
        directory = LANE / ('night_dry_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + str(os.getpid()))
        directory.mkdir()
        env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
        dry = run_group([sys.executable, str(tools / 'gpu_probe.py'), '--manifest', request['manifest'],
            '--out', str(directory), '--dry-run'], env, directory / 'dry.log', 15, utc_now() + 20)
        assert dry['rc'] == 0
        group = selftest(directory)
        (directory / 'receipt.json').write_text(json.dumps({'dry': dry, 'private_flock_group_test': group,
            'GPU_started': False, 'status': 'PASS'}, indent=2) + '\n')
        print('DRY PASS: 24-case host fixture; timeout killed own child+grandchild; private flock reacquired; no GPU', flush=True)
        return 0
    if sys.argv[1:]:
        raise ValueError('only --dry-run/--check-window are supported')
    start, end, current = epoch(os.environ['GA_START_UTC']), epoch(os.environ['GA_END_UTC']), utc_now()
    if current < start or end - current < CAP:
        print('REFUSE: outside GA window or less than full 3-minute cap remains', flush=True)
        return 5
    deadline = min(end, current + CAP)
    directory = LANE / 'night_receipts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True)
    result = {'source': request['model_pin'], 'tools': request['tools_commit'], 'cap_minutes': 3,
              'scope': '24-case native shadow operator oracle/timing; NOT the whole WN3 3h forecast gate',
              'window': {'start': os.environ['GA_START_UTC'], 'end': os.environ['GA_END_UTC']},
              'gpu': None, 'e41': None, 'passed': False}
    (directory / 'START.json').write_text(json.dumps(result, indent=2) + '\n')
    cache = directory / 'cache'
    for name in ('jax', 'triton', 'cuda', 'tmp'):
        (cache / name).mkdir(parents=True)
    env = {**os.environ, 'JAX_PLATFORMS': 'cuda', 'GPUWRF_JAX_CACHE': '1',
           'GPUWRF_AOT_SLIM_MODULE': '0',
           'GPUWRF_JAX_CACHE_DIR': str(cache / 'jax'), 'JAX_COMPILATION_CACHE_DIR': str(cache / 'jax'),
           'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS': '0', 'JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES': '0',
           'TRITON_CACHE_DIR': str(cache / 'triton'), 'CUDA_CACHE_PATH': str(cache / 'cuda'),
           'TMPDIR': str(cache / 'tmp'), 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'PYTHONPATH': str(tree / 'src'), 'OMP_NUM_THREADS': '4'}
    env.pop('XLA_FLAGS', None)
    try:
        result['hibernation_before'] = hibernation_record(directory / 'hibernation_before.log')
        result['gpu'] = run_group([str(tree / 'scripts/with_gpu_lock.sh'), '--label', 'w44-topo',
            '--timeout', '1', '--', sys.executable, str(tools / 'gpu_probe.py'), '--manifest', request['manifest'],
            '--out', str(directory / 'operator')], env, directory / 'gpu.log', 120, deadline - 45)
        print(f"GPU END rc={result['gpu']['rc']}; entire owned process group cleaned", flush=True)
        result['hibernation_after'] = hibernation_record(directory / 'hibernation_after.log')
        if Path('/tmp/wrf_gpu2_quiet').exists():
            result['deferred'] = 'QUIET; E41 cache retained'
            return 3
        env.update(JAX_PLATFORMS='cpu', GPUWRF_JAX_CACHE='0')
        remaining = deadline - utc_now() - 12
        if remaining <= 0:
            result['deferred'] = 'deadline; E41 cache retained'
            return 5
        result['e41'] = run_group([sys.executable, str(tree / 'scripts/wave_cubin_gate.py'),
            str(cache / 'jax'), str(directory / 'e41.json'), 'jit_one*'], env,
            directory / 'e41.log', remaining, deadline - 12)
        result['passed'] = result['gpu']['rc'] == 0 and result['e41']['rc'] == 0
        if result['passed'] and not Path('/tmp/wrf_gpu2_quiet').exists():
            # Delete ONLY generated regular files beneath this exact new run's
            # cache; refuse links and never traverse reference/case directories.
            assert cache.resolve().parent == directory.resolve()
            for path in sorted(cache.rglob('*'), key=lambda x: len(x.parts), reverse=True):
                if path.is_symlink():
                    raise ValueError('unexpected symlink in private disposable cache')
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            cache.rmdir()
            result['disposable_cache_cleaned'] = True
        return result['gpu']['rc'] or result['e41']['rc']
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result['ended_utc'] = datetime.now(timezone.utc).isoformat()
        (directory / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
