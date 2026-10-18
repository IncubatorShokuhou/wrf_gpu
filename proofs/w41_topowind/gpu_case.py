"""One GPU CLI process for a topo_wind composite case (lane w41-topowind), modelled on
proofs/v040/final_validation/gpu_case.py: under the official lock, immutable clean detached
snapshot, release-default CLI (no flag file), 24 h, asserted GPU, receipt gpu_run.json.
usage (from the snapshot, inside scripts/with_gpu_lock.sh): python gpu_case.py <case_dir> [hours]
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

case = Path(sys.argv[1]).resolve()
hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
source = Path(__file__).resolve().parents[2]
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1', 'run under scripts/with_gpu_lock.sh'
fd = int(os.environ['GPUWRF_GPU_LOCK_FD'])
assert fd >= 3 and os.fstat(fd).st_ino == os.stat(os.environ['GPUWRF_GPU_LOCK_FILE']).st_ino
assert subprocess.run(['git', '-C', str(source), 'symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode != 0
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
assert not (case / 'gpu_run.json').exists(), 'preserve prior attempts; never overwrite a receipt'
assert not list((case / 'gpu').glob('wrfout_*')), 'preserve previous forecast frames (E42)'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = {p.name: sha(p) for p in sorted((case / 'input').iterdir()) if p.is_file()}
text = (case / 'input/namelist.input').read_text().lower()
max_dom = int(next(line.split('=')[1].split(',')[0] for line in text.splitlines() if line.strip().startswith('max_dom')))
started = datetime.now(timezone.utc).isoformat()
t0 = time.monotonic()
sys.path.insert(0, str(source / 'src'))
import gpuwrf  # noqa: E402  (XLA/x64/command-buffer configuration before the backend)
import jax  # noqa: E402

assert Path(gpuwrf.__file__).resolve().is_relative_to(source / 'src'), gpuwrf.__file__   # E159

assert jax.devices()[0].platform == 'gpu', jax.devices()
print(f'W41TOPO platform=gpu pid={os.getpid()} device={jax.devices()[0]}', flush=True)
argv = ['run', '--input-dir', str(case / 'input'), '--output-dir', str(case / 'gpu'),
        '--proof-dir', str(case / 'gpu_proof'), '--scratch-dir', str(case / 'gpu_scratch'),
        '--hours', str(hours), '--domains-from-namelist']
if max_dom > 1:
    argv.append('--emit-initial-history')   # single-domain CLI emits t=0 itself and rejects the flag (cli.py:930)
rc, error = 1, None
try:
    from gpuwrf.cli import main
    rc = main(argv)
except BaseException as exc:
    error = repr(exc)
    raise
finally:
    receipt = dict(schema='w41-topowind-gpu-run-v1', case=case.name, platform=jax.devices()[0].platform,
                   device=str(jax.devices()[0]), pid=os.getpid(), started_utc=started,
                   finished_utc=datetime.now(timezone.utc).isoformat(), monotonic_wall_s=time.monotonic() - t0,
                   rc=rc, error=error, source=str(source),
                   source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
                   launcher_sha256=sha(__file__), input_sha256=inputs, argv=argv,
                   lock_label=os.environ.get('GPUWRF_GPU_LOCK_LABEL'),
                   environment={k: os.environ.get(k) for k in ('JAX_PLATFORMS', 'GPUWRF_FAST_DEFAULTS', 'GPUWRF_CENSUS',
                                                             'GPUWRF_JAX_CACHE_DIR', 'JAX_COMPILATION_CACHE_DIR')},
                   timing_scope='cold composite qualification on a shared box; not an S1 benchmark')
    (case / 'gpu_run.json').write_text(json.dumps(receipt, indent=2) + '\n')
sys.exit(rc)
