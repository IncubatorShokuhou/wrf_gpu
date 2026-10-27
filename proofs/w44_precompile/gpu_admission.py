"""Live GPU admission check only. Never relabel metadata or compile a fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--artifact', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
started = time.monotonic()
assert os.environ['GPUWRF_GPU_LOCK_HELD'] == '1'
assert os.environ['GPUWRF_GPU_LOCK_LABEL'] == 'w44-precompile'
assert os.environ['JAX_PLATFORMS'] == 'gpu'
manifest = json.loads((Path(__file__).parent / 'receipts/ARTIFACTS.json').read_text())
for name, expected in manifest['files'].items():
    with (a.artifact / name).open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == expected, name
assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'], text=True).strip() == manifest['source_commit']
assert not a.out.exists(), 'preserve the previous admission receipt'
sys.path.insert(0, str(a.source / 'src'))
from gpuwrf.runtime.gpu_preflight import gpu_lock_status
assert gpu_lock_status()['ok']
import jax
from gpuwrf.runtime import aot_executable as aotx
assert Path(aotx.__file__).resolve().is_relative_to(a.source.resolve())
dev = jax.devices()[0]
assert dev.platform == 'gpu'
assert not subprocess.check_output(['git', '-C', str(a.source), 'status', '--porcelain', '--untracked-files=no'])
with (a.artifact / 'program.meta').open('rb') as f:
    meta = pickle.load(f)
blob = (a.artifact / 'program.xlaexec').read_bytes()
assert hashlib.sha256(blob).hexdigest() == meta.blob_sha256
live = aotx.target_fingerprint(dev)
rec = dict(scope=__doc__, platform=dev.platform, gpu_backend_initialized=True, stepping_executed=False,
           source_commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'], text=True).strip(),
           saved=meta.fingerprint, live=live, blob_sha256=meta.blob_sha256,
           differences={k:[v,live.get(k)] for k,v in meta.fingerprint.items() if v != live.get(k)},
           qualified=False)
t = time.monotonic()
try:
    call = aotx.load(blob, meta, dev=dev)  # strict default fingerprint check
except Exception as exc:
    rec.update(loaded=False, exception=f'{type(exc).__name__}: {exc}', verdict='NOT_QUALIFIED_LOAD_REFUSED')
else:
    rec.update(loaded=True, verdict='LOAD_ONLY_SCIENTIFIC_GATES_PENDING')
rec.update(load_s=time.monotonic()-t, process_to_result_s=time.monotonic()-started)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(rec, indent=2)+'\n')
print(json.dumps(rec, indent=2), flush=True)
# Both outcomes are an honest admission result; neither claims BYTE qualification.
