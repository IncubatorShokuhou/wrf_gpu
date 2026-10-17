"""One locked segment of the manager-approved six-hour Goddard gate."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

source, output = map(Path, sys.argv[1:3])
source, output = source.resolve(), output.resolve()
case = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c45')
assert os.environ['GPUWRF_GPU_LOCK_HELD'] == '1'
fd = int(os.environ['GPUWRF_GPU_LOCK_FD'])
assert os.fstat(fd).st_ino == os.stat(os.environ['GPUWRF_GPU_LOCK_FILE']).st_ino
assert subprocess.run(['git', '-C', str(source), 'symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode
assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
assert output == Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms/c45/six-hour')
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
cpu = json.loads((case / 'cpu_run.json').read_text())
inputs = json.loads((case / 'inputs.json').read_text())
actual = {p.name: sha(p) for p in (case / 'input').iterdir() if p.is_file()}
assert cpu['status'] == 'COMPLETE' and actual == cpu['actual_input_sha256'] == inputs['input_sha256']
sys.path.insert(0, str(source / 'src'))
import jax
assert jax.devices()[0].platform == 'gpu'
print('GOD5_SEGMENT platform=gpu device=' + str(jax.devices()[0]), flush=True)
from gpuwrf.runtime.restart_store import RestartStore
store = RestartStore(output / 'checkpoints', 8 << 30, 2, 120 << 30)
verified = store.verified()
resume = verified[-1][1] if verified else None
assert resume is not None or not list((output / 'gpu').glob('wrfout_*'))
segment = len(list(output.glob('segment-*.json'))) + 1
receipt = dict(segment=segment, started_utc=datetime.now(timezone.utc).isoformat(),
    platform=jax.devices()[0].platform, device=str(jax.devices()[0]), pid=os.getpid(),
    source=str(source), source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
    launcher_sha256=sha(__file__), original_cpu_receipt_sha256=sha(case / 'cpu_run.json'),
    actual_input_sha256=actual, gate_hours=6, checkpoint_interval_steps=200,
    resume_checkpoint=str(resume) if resume else None,
    resume_manifest_sha256=sha(resume / 'manifest.json') if resume else None,
    lock_label=os.environ['GPUWRF_GPU_LOCK_LABEL'], lock_token=os.environ['GPUWRF_GPU_LOCK_TOKEN'],
    cache=str(output / 'cache'), timing_scope='checkpointed functional gate, not S1')
path = output / f'segment-{segment:02d}.json'
path.write_text(json.dumps(receipt, indent=2) + '\n')
argv = ['run', '--input-dir', str(case / 'input'), '--output-dir', str(output / 'gpu'),
    '--proof-dir', str(output / 'proof'), '--scratch-dir', str(output / 'scratch'),
    '--hours', '6', '--domains-from-namelist', '--checkpoint-dir', str(store.root),
    '--checkpoint-interval-steps', '200', '--checkpoint-max-generations', '2',
    '--checkpoint-reserve-bytes', str(store.reserve_bytes)]
if resume is not None:
    argv += ['--resume-checkpoint', str(resume)]
receipt['argv'] = argv
path.write_text(json.dumps(receipt, indent=2) + '\n')
t0 = time.monotonic()
try:
    from gpuwrf.cli import main
    rc = main(argv)
    receipt['rc'] = rc
finally:
    receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.monotonic()-t0)
    path.write_text(json.dumps(receipt, indent=2) + '\n')
sys.exit(rc)
