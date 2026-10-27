"""Bounded, immutable WS11 GPU component launcher for the manager sequencer.

CPU supervisor; never imports JAX. Cleanup helpers are reused from w44-budget's
frozen, CPU-tested night supervisor (private copy, exact hash in the seal).
An actual GPU component PASS is required before scheduling the whole forecast.
"""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from launch_preflight import cuda_environment

BASE = Path('<USER_HOME>/wrf_gpu2_lanes/w44-warmstart')
SEAL = BASE / 'NIGHT_COMPONENT_SEAL_V3.json'
FIXED_RESULT = BASE / 'NIGHT_COMPONENT_RESULT.json'
CAP = 600
GPU_CAP = 240
E41_CAP = 240
CPU_ENV = dict(os.environ, JAX_PLATFORMS='cpu', GPUWRF_JAX_CACHE='0',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def child_environment(snap,run):
    cache=run/'cache/jax'
    env=cuda_environment(CPU_ENV,PYTHONPATH=str(snap/'src'),
        GPUWRF_JAX_CACHE='1',GPUWRF_JAX_CACHE_DIR=str(cache),
        JAX_COMPILATION_CACHE_DIR=str(cache),JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0',
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES='-1',
        TMPDIR=str(run/'tmp'),TRITON_CACHE_DIR=str(run/'cache/triton'),CUDA_CACHE_PATH=str(run/'cache/cuda'),
        XLA_PYTHON_CLIENT_PREALLOCATE='false',GPUWRF_FAST_DEFAULTS='1')
    env.update({n:'1'for n in ('GPUWRF_MYNN_REAL_CONSTANTS','GPUWRF_MYNN_PSIG_CLAMP',
        'GPUWRF_MYNN_ELB_MF','GPUWRF_MYNN_SGS_MIXING_RATIO','GPUWRF_MYNN_PHY_EXNER','GPUWRF_MYNN_FP32_COLUMNS')})
    return env


def child_command(seal,run):
    return [sys.executable,seal['runner'],str(run/'probe'),seal['oracle']]


def main():
    record = dict(schema='w44-warmstart.night-component.v1', cap_s=CAP,
                  gpu_started=False, phases={}, verdict='PREFLIGHT', rc=2)
    run = BASE / ('NIGHT_COMPONENT_' + str(time.time_ns()))
    run.mkdir()
    receipt = run / 'receipt.json'
    record['receipt_path'] = str(receipt)
    write(receipt, record)
    write(FIXED_RESULT, record)
    rc = 2
    supervisor = None
    try:
        # Missing/short windows refuse before loading a GPU-facing command.
        def epoch(value):
            value = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
            if value.utcoffset() != dt.timedelta(0):
                raise ValueError('explicit UTC window required')
            return value.timestamp()
        start = epoch(os.environ['GA_START_UTC']); end = epoch(os.environ['GA_END_UTC'])
        now = int(subprocess.check_output(['bash', '<USER_HOME>/wrf_gpu2_lanes/manager/real_utc.sh'],
                                         env=CPU_ENV, timeout=8, text=True).strip())
        record.update(window_start=start, window_end=end, real_utc=now)
        if now < start or end <= start or end-now < CAP:
            rc = 4; record['verdict'] = 'REFUSED_WINDOW'; return rc
        if os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1':
            raise ValueError('call outside an existing GPU lock')
        if Path('/tmp/wrf_gpu2_quiet').exists():
            rc = 3; record['verdict'] = 'REFUSED_QUIET'; return rc
        seal = json.loads(SEAL.read_text()); snap = Path(seal['snapshot'])
        for name, expected in seal['files'].items():
            if sha(name) != expected:
                raise ValueError('immutable file changed: '+name)
        commit = subprocess.check_output(['git', '-C', str(snap), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != seal['commit'] or subprocess.check_output(
                ['git', '-C', str(snap), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise ValueError('immutable numerical snapshot changed')
        if not (snap/'data/wrf_pristine').exists():
            raise ValueError('missing pristine WRF link')
        if shutil.disk_usage('<DATA_ROOT>').free < 200*2**30 or shutil.disk_usage(BASE).free < 20*2**30:
            rc=5; record['verdict']='REFUSED_DISK_FLOOR'; return rc
        spec = importlib.util.spec_from_file_location('reused_night_groups', seal['supervisor'])
        supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor)
        import ctypes
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise OSError('cannot install child subreaper')
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, lambda number, _frame: setattr(supervisor, 'SIGNALLED', number))
        cache = run / 'cache/jax'; cache.mkdir(parents=True)
        for directory in ('tmp','cache/triton','cache/cuda'):(run/directory).mkdir()
        env=child_environment(snap,run)
        # GPU initialization occurs solely inside this official lock wrapper.
        command = [str(snap/'scripts/with_gpu_lock.sh'), '--label', 'w44-warmstart', '--timeout', '1', '--',
                   *child_command(seal,run)]
        record.update(source_commit=commit, command=command, verdict='GPU_REQUESTED')
        write(receipt, record)
        record['phases']['gpu'] = supervisor.run_group(command, run/'gpu.log', GPU_CAP, env=env)
        record['gpu_started'] = (run/'probe/PLATFORM.json').exists()
        rc = record['phases']['gpu']['rc']
        # Scan actual cold native executables even if the numerical gate failed.
        if record['gpu_started'] and not supervisor.SIGNALLED:
            if Path('/tmp/wrf_gpu2_quiet').exists():
                record['phases']['e41'] = dict(rc=3, verdict='PENDING_QUIET')
            else:
                command = ['taskset', '-c', '8,9,12', 'nice', '-n', '19',
                    str(snap/'scripts/with_ram_slot.sh'), '--label', 'w44-warmstart-component-E41',
                    '--max', '6G', '--timeout', '30', '--', 'bash', '-c',
                    'if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; exec "$@"',
                    'w44-e41', sys.executable, str(snap/'scripts/wave_cubin_gate.py'), str(cache), str(run/'e41'),
                    'jit__lambda*']
                host_env=dict(CPU_ENV,PYTHONPATH=str(snap/'src'),CUDA_VISIBLE_DEVICES='')
                record['phases']['e41'] = supervisor.run_group(command, run/'e41.log', E41_CAP, env=host_env)
            rc = rc or record['phases']['e41']['rc']
        else:
            record['phases']['e41'] = dict(rc=3, verdict='NO_GPU_OR_SIGNAL')
        result = run/'probe/result.json'
        record['component'] = json.loads(result.read_text()) if result.exists() else None
        if rc == 0 and (not record['component'] or not record['component']['passed']
                        or record['component']['backend'] != 'gpu'):
            rc=2
        record['verdict'] = 'GPU_ORIGINAL_INIT_PASS' if rc == 0 else 'FAILED_OR_INCOMPLETE'
        # The cache is retained until the actual E41 gate has completed; a
        # pending scan is never silently presented as a completed cold arm.
        record['cache'] = str(cache)
        record['cache_disposition'] = 'CLEAN_AFTER_E41_REVIEW; never needed by whole-model forecast'
        return rc
    except (KeyError, ValueError, OSError, subprocess.SubprocessError) as exc:
        record.update(error=str(exc), verdict='REFUSED_OR_ERROR')
        rc = 4 if isinstance(exc, KeyError) and str(exc).strip("'").startswith('GA_') else 2
        return rc
    finally:
        record.update(rc=rc, ended_monotonic=time.monotonic())
        write(receipt, record)
        write(FIXED_RESULT, record)
        print(json.dumps({'receipt':str(receipt), 'rc':rc, 'verdict':record['verdict']}), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
