"""One actual GPU CLI process for a v-prep composite case (c87 Jensen / c88 NTU).

Short compile-readiness arm: cold whole-program compile + N forecast hours on the
original case inputs (read-only symlinks into the v-prep case), private caches.
Must run under scripts/with_gpu_lock.sh from an immutable detached tree.
Not a D6/S1 result; the 24 h qualification belongs to v-prep.
usage: gpu_arm.py <vprep_case_dir> <arm_dir> <hours>
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

case = Path(sys.argv[1]).resolve()
arm = Path(sys.argv[2]).resolve()
hours = int(sys.argv[3])
source = Path(__file__).resolve().parents[2]
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
assert subprocess.run(['git', '-C', str(source), 'symbolic-ref', '--quiet', 'HEAD'],
                      capture_output=True).returncode != 0, 'detached immutable tree required'
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', 'src'], text=True).strip()
arm.mkdir(parents=True, exist_ok=False)
inputs = json.loads((case / 'inputs.json').read_text())
assert inputs['status'] == 'ADMITTED' and not inputs['admission_errors']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


(arm / 'input').mkdir()
for p in sorted((case / 'input').iterdir()):
    (arm / 'input' / p.name).symlink_to(p.resolve())
actual = {p.name: sha(p) for p in (arm / 'input').iterdir()}
assert actual == inputs['input_sha256'], actual
started = datetime.now(timezone.utc).isoformat()
t0 = time.monotonic()
import gpuwrf  # noqa: E402,F401
import jax  # noqa: E402
assert jax.devices()[0].platform == 'gpu', jax.devices()
print(f'ARM platform=gpu pid={os.getpid()} device={jax.devices()[0]} case={case.name}', flush=True)
stop = threading.Event()
vram = []


def sample():
    while not stop.is_set():
        try:
            r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            for line in r.stdout.splitlines():
                f = line.split(',')
                if len(f) == 2 and f[0].strip() == str(os.getpid()):
                    vram.append([round(time.monotonic() - t0, 2), int(f[1].strip())])
        except (subprocess.SubprocessError, ValueError):
            pass
        stop.wait(1)


threading.Thread(target=sample, daemon=True).start()
argv = ['run', '--input-dir', str(arm / 'input'), '--output-dir', str(arm / 'gpu'),
        '--proof-dir', str(arm / 'gpu_proof'), '--scratch-dir', str(arm / 'gpu_scratch'),
        '--hours', str(hours), '--domains-from-namelist']
rc, error = 1, None
try:
    from gpuwrf.cli import main
    rc = main(argv)
except BaseException as exc:  # receipt first, then re-raise
    error = repr(exc)
    raise
finally:
    stop.set()
    frames = sorted(p.name for p in (arm / 'gpu').glob('wrfout_*')) if (arm / 'gpu').exists() else []
    receipt = dict(case=case.name, work=inputs['options'], platform=jax.devices()[0].platform,
                   device=str(jax.devices()[0]), pid=os.getpid(), started_utc=started,
                   finished_utc=datetime.now(timezone.utc).isoformat(), monotonic_wall_s=time.monotonic() - t0,
                   rc=rc, error=error, hours=hours, frames=frames, source=str(source),
                   source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                                         text=True).strip(),
                   launcher_sha256=sha(__file__), input_sha256=actual,
                   lock_label=os.environ.get('GPUWRF_GPU_LOCK_LABEL'),
                   gpu_arm_cpus=os.environ.get('GPUWRF_GPU_ARM_CPUS'),
                   sampled_process_vram_mib_peak=max((v[1] for v in vram), default=None), vram_samples=vram,
                   environment={k: os.environ.get(k) for k in (
                       'JAX_PLATFORMS', 'GPUWRF_FAST_DEFAULTS', 'GPUWRF_CENSUS', 'GPUWRF_RRTMG_MP_RE',
                       'GPUWRF_JAX_CACHE_DIR', 'GPUWRF_XLA_AUTOTUNE_CACHE_DIR', 'TRITON_CACHE_DIR',
                       'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_CLIENT_MEM_FRACTION')},
                   argv=argv, scope='cold compile-readiness arm (whole program + N h); not D6/S1')
    (arm / 'gpu_run.json').write_text(json.dumps(receipt, indent=2) + '\n')
sys.exit(rc)
