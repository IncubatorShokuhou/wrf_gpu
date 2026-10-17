"""Actual locked GPU smoke on original WN3 inputs; regression, not fidelity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from prepare import ROOT, sha

source = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[3]
case = ROOT / 'WN3-smoke'
plan = json.loads((case / 'PLAN.json').read_text())
assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
fd = int(os.environ['GPUWRF_GPU_LOCK_FD'])
assert fd >= 3 and os.fstat(fd).st_ino == os.stat(os.environ['GPUWRF_GPU_LOCK_FILE']).st_ino
assert subprocess.run(['git','-C',str(source),'symbolic-ref','--quiet','HEAD'],capture_output=True).returncode != 0
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
assert {p.name:sha(p) for p in (case/'input').iterdir()} == plan['input_sha256']
assert not (case/'gpu_run.json').exists() and not list((case/'gpu').glob('wrfout*'))
import gpuwrf
assert str(Path(gpuwrf.__file__).resolve()).startswith(str(source / 'src'))
import jax
assert jax.devices()[0].platform == 'gpu'
print(f'WN3_SMOKE platform=gpu device={jax.devices()[0]} pid={os.getpid()}',flush=True)
try:
    from jaxlib import _jax
    target=_jax.get_topology_for_devices(jax.devices()).target_config
    (case/'target_config.pbtxt').write_text(str(target))
except Exception as exc:
    (case/'target_config_capture_error.json').write_text(json.dumps(dict(error=repr(exc)))+'\n')
started=datetime.now(timezone.utc).isoformat();clock=time.monotonic();rc=1
argv=['run','--input-dir',str(case/'input'),'--output-dir',str(case/'gpu'),
    '--proof-dir',str(case/'gpu_proof'),'--scratch-dir',str(case/'gpu_scratch'),
    '--hours',str(plan.get('forecast_hours',3)),'--domains-from-namelist','--emit-initial-history']
try:
    from gpuwrf.cli import main
    rc=main(argv)
finally:
    receipt=dict(schema='v040-wn3-actual-gpu-smoke-v1',source_commit=subprocess.check_output(
        ['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        platform=jax.devices()[0].platform,device=str(jax.devices()[0]),rc=rc,
        started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
        monotonic_wall_s=time.monotonic()-clock,argv=argv,plan_sha256=sha(case/'PLAN.json'),
        launcher_sha256=sha(__file__),lock_label=os.environ['GPUWRF_GPU_LOCK_LABEL'],
        environment={k:os.environ.get(k) for k in ('GPUWRF_FAST_DEFAULTS','GPUWRF_CENSUS','GPUWRF_RRTMG_MP_RE',
            'GPUWRF_JAX_CACHE_DIR','GPUWRF_XLA_AUTOTUNE_CACHE_DIR','XLA_CLIENT_MEM_FRACTION')},
        scope='GPU-vs-released-GPU byte regression only; no fidelity or S1 claim')
    (case/'gpu_run.json').write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(rc)
