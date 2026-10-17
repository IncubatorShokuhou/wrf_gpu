"""CLI qualification entry: GPU assertion, process VRAM sampling and cold wall."""
import json
import hashlib
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

case = Path(os.environ['GPUQUAL_CASE'])
source = Path(os.environ['GPUQUAL_SOURCE'])
launcher_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for variable, suffix in (('TMPDIR', 'tmp'), ('CUDA_CACHE_PATH', 'cache/cuda'),
                         ('TRITON_CACHE_DIR', 'cache/triton'), ('XDG_CACHE_HOME', 'cache/xdg')):
    location = case / suffix
    location.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(location)
now = datetime.now(timezone.utc)
assert 11 * 60 + 30 <= now.hour * 60 + now.minute < 18 * 60 + 40, now
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
t0 = time.perf_counter()
started = now.isoformat()
import gpuwrf  # release defaults and cache setup must precede backend creation
import jax

assert jax.devices()[0].platform == 'gpu', jax.devices()
print(f'GPUQUAL platform=gpu pid={os.getpid()} device={jax.devices()[0]}', flush=True)
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
actual_input = Path(sys.argv[sys.argv.index('--input-dir') + 1]).resolve()
assert actual_input == (case / 'input').resolve()
input_hashes = {name: hashlib.sha256((actual_input / name).read_bytes()).hexdigest()
                for name in ('namelist.input', 'wrfinput_d01', 'wrfbdy_d01')}
rc = 1
error = None
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
    receipt = {'schema': 'gpuqual-gpu-run-v1', 'started_utc': started,
               'finished_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
               'source': str(source), 'source_rev': subprocess.check_output(
                   ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
               'source_status': subprocess.check_output(
                   ['git', '-C', str(source), 'status', '--porcelain'], text=True).strip(),
               'launcher': str(Path(__file__).resolve()), 'launcher_sha256': launcher_sha256,
               'platform': 'gpu', 'device': str(jax.devices()[0]), 'rc': rc, 'error': error,
               'actual_cli_input_dir': str(actual_input), 'actual_input_sha256': input_hashes,
               'measurement_flags': {k: os.environ.get(k) for k in
                    ('JAX_PLATFORMS', 'GPUWRF_FAST_DEFAULTS', 'GPUWRF_CENSUS',
                     'GPUWRF_STRICT_GUARDS', 'GPUWRF_AOT_SLIM_MODULE', 'GPUWRF_RRTMG_MP_RE')},
               'wall_s': wall, 'cold_s_per_fc_h': wall / 24 if rc == 0 else None,
               'timing_scope': 'cold, shared CPU/IO; not an S1 warm benchmark',
               'peak_process_vram_mib_sampled': max((x[1] for x in samples), default=None),
               'vram_samples': samples,
               'cache': os.environ.get('JAX_COMPILATION_CACHE_DIR'), 'argv': sys.argv[1:]}
    (case / 'gpu_run.json').write_text(json.dumps(receipt, indent=2) + '\n')
sys.exit(rc)
