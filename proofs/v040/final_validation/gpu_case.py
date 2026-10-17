"""One actual GPU CLI process, with immutable-source and input provenance."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone

case = Path(sys.argv[1]).resolve()
source = Path(__file__).resolve().parents[3]
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
fd = int(os.environ['GPUWRF_GPU_LOCK_FD'])
assert fd >= 3 and os.fstat(fd).st_ino == os.stat(os.environ['GPUWRF_GPU_LOCK_FILE']).st_ino
assert subprocess.run(['git', '-C', str(source), 'symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode != 0
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
assert not (case / 'gpu_run.json').exists(), 'Preserve prior attempts; never overwrite a receipt'
inputs = json.loads((case / 'inputs.json').read_text())
assert inputs.get('gpu_eligible', True), 'Reference-only control; do not run this GPU row'
assert not inputs['admission_errors']
cpu = json.loads((case / 'cpu_run.json').read_text())
assert cpu['status'] == 'COMPLETE', 'No actual complete original-WRF reference'
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
actual = {p.name: sha(p) for p in (case / 'input').iterdir() if p.is_file()}
assert actual == cpu['actual_input_sha256'] == inputs['input_sha256']
assert not list((case / 'gpu').glob('wrfout_*')), 'Preserve previous forecast frames'
started = datetime.now(timezone.utc).isoformat()
t0 = time.monotonic()
import gpuwrf
import jax
assert jax.devices()[0].platform == 'gpu', jax.devices()
print(f'FINALVAL platform=gpu pid={os.getpid()} device={jax.devices()[0]}', flush=True)
try:
    from jaxlib import _jax
    target = _jax.get_topology_for_devices(jax.devices()).target_config
    (case / 'target_config.pbtxt').write_text(str(target))
except Exception as exc:
    (case / 'target_config_capture_error.json').write_text(json.dumps(dict(error=repr(exc)))+'\n')
stop = threading.Event()
vram_samples = []
def sample_vram():
    while not stop.is_set():
        try:
            r = subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            for line in r.stdout.splitlines():
                fields=line.split(',')
                if len(fields)==2 and fields[0].strip()==str(os.getpid()):
                    vram_samples.append([time.monotonic()-t0,int(fields[1].strip())])
        except (subprocess.SubprocessError,ValueError):
            pass  # unknown memory remains unknown, never fabricated zero
        stop.wait(1)
sampler=threading.Thread(target=sample_vram,daemon=True)
sampler.start()
rc, error = 1, None
argv = ['run', '--input-dir', str(case / 'input'), '--output-dir', str(case / 'gpu'),
        '--proof-dir', str(case / 'gpu_proof'), '--scratch-dir', str(case / 'gpu_scratch'),
        '--hours', str(inputs['hours']), '--domains-from-namelist']
if len(inputs['domains']) > 1:
    argv.append('--emit-initial-history')
# Native single-domain CLI automatically emits its initial history and rejects
# the explicit nested-only flag (cli.py:930/1205).
try:
    from gpuwrf.cli import main
    rc = main(argv)
except BaseException as exc:
    error = repr(exc)
    raise
finally:
    stop.set()
    sampler.join(timeout=6)
    receipt = dict(schema='v040-final-gpu-run-v1', case=case.name, platform=jax.devices()[0].platform,
        device=str(jax.devices()[0]), pid=os.getpid(), started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(), monotonic_wall_s=time.monotonic() - t0,
        rc=rc, error=error, source=str(source), source_commit=subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        launcher_sha256=sha(__file__), actual_input_dir=str(case / 'input'), actual_input_sha256=actual,
        original_cpu_receipt_sha256=sha(case / 'cpu_run.json'),
        lock_label=os.environ['GPUWRF_GPU_LOCK_LABEL'], lock_fd=fd,
        sampled_process_vram_mib_peak=max((v[1] for v in vram_samples),default=None),
        vram_sampling_scope='once/sec process samples, can miss short peaks; pool fraction is not total VRAM bound',
        vram_samples=vram_samples,
        environment={k: os.environ.get(k) for k in ('JAX_PLATFORMS', 'GPUWRF_FAST_DEFAULTS',
            'GPUWRF_CENSUS', 'GPUWRF_CENSUS_SLIM', 'GPUWRF_AOT_SLIM_MODULE',
            'GPUWRF_RRTMG_MP_RE', 'GPUWRF_JAX_CACHE_DIR', 'GPUWRF_XLA_AUTOTUNE_CACHE_DIR',
            'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_CLIENT_MEM_FRACTION')},
        argv=argv, timing_scope='cold shared-CPU/I/O composite qualification; not warm S1 benchmark')
    (case / 'gpu_run.json').write_text(json.dumps(receipt, indent=2) + '\n')
sys.exit(rc)
