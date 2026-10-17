"""Actual GPU process receipt; exact source/input identities and sampled VRAM."""
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

case = Path(os.environ['GPUQUAL_CASE'])
runtime = Path(os.environ.get('GPUQUAL_RUNTIME', str(case)))
source = Path(os.environ['GPUQUAL_SOURCE']).resolve()
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
fd = int(os.environ['GPUWRF_GPU_LOCK_FD'])
assert fd > 2 and os.fstat(fd).st_ino == os.stat(os.environ['GPUWRF_GPU_LOCK_FILE']).st_ino
assert not Path('/tmp/wrf_gpu2_quiet').exists()
now = datetime.now(timezone.utc)
nightly = json.loads(Path('<DATA_ROOT>/alisios/state/nightly_9nest/LAST_RUN.json').read_text())
assert nightly['issue'] == '20261014_12z' and nightly['verdict'] == 'DONE'
assert now < datetime(2026, 10, 15, 18, 40, tzinfo=timezone.utc), now
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
for var, suffix in [('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cache/cuda'),
                    ('TRITON_CACHE_DIR', 'cache/triton'), ('XDG_CACHE_HOME', 'cache/xdg')]:
    p = runtime / suffix
    p.mkdir(parents=True, exist_ok=True)
    os.environ[var] = str(p)
actual_input = Path(sys.argv[sys.argv.index('--input-dir') + 1]).resolve()
assert actual_input == (case / 'input').resolve()
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in actual_input.iterdir() if p.is_file()}
expected = json.loads((case / 'inputs.json').read_text())
assert all(hashes[k] == v['GPU_sha256'] for k, v in expected['matched_inputs'].items())
t0 = time.perf_counter()
import gpuwrf
assert Path(gpuwrf.__file__).resolve().is_relative_to(source / 'src')
import jax
assert jax.devices()[0].platform == 'gpu', jax.devices()
print(f'GPUQUAL platform=gpu config={case.name} pid={os.getpid()} source={revision} device={jax.devices()[0]}', flush=True)
stop = threading.Event()
samples = []
def sample():
    while not stop.is_set():
        p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        for line in p.stdout.splitlines():
            fields = line.split(',')
            if len(fields) == 2 and fields[0].strip() == str(os.getpid()):
                try:
                    samples.append([time.perf_counter() - t0, int(fields[1].strip())])
                except ValueError:
                    pass
        stop.wait(1)
thread = threading.Thread(target=sample, daemon=True)
thread.start()
rc, error = 1, None
try:
    from gpuwrf.cli import main
    rc = main(sys.argv[1:])
except BaseException as exc:
    error = repr(exc)
    raise
finally:
    stop.set()
    thread.join(timeout=6)
    wall = time.perf_counter() - t0
    receipt = {'schema': 'gpuqual-round2-process-v1', 'started_utc': now.isoformat(),
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'source': str(source), 'source_rev': revision, 'source_status': subprocess.check_output(
            ['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(),
        'imported_gpuwrf': gpuwrf.__file__, 'platform': 'gpu', 'device': str(jax.devices()[0]),
        'launcher': str(Path(__file__).resolve()),
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rc': rc, 'error': error, 'actual_cli_input_dir': str(actual_input),
        'actual_input_sha256': hashes, 'wall_s': wall,
        'cold_s_per_fc_h': wall / 24 if rc == 0 else None,
        'timing_scope': 'cold concurrent Swiss qualification, shared CPU/IO; not S1 or isolated benchmark',
        'peak_process_vram_mib_sampled': max((x[1] for x in samples), default=None),
        'host_peak_rss_kib': __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
        'vram_samples': samples, 'argv': sys.argv[1:],
        'cache': os.environ.get('JAX_COMPILATION_CACHE_DIR'),
        'runtime': str(runtime), 'tmp': os.environ.get('TMPDIR'),
        'measurement_flags': {k: os.environ.get(k) for k in ('JAX_PLATFORMS',
            'GPUWRF_FAST_DEFAULTS', 'GPUWRF_CENSUS', 'GPUWRF_STRICT_GUARDS',
            'GPUWRF_AOT_SLIM_MODULE', 'GPUWRF_RRTMG_MP_RE', 'XLA_PYTHON_CLIENT_PREALLOCATE')}}
    (case / 'gpu_run.json').write_text(json.dumps(receipt, indent=2) + '\n')
sys.exit(rc)
