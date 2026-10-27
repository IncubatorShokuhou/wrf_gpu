"""Compare a preserved CPU AOT target against an attested release GPU target.

CPU only. Neither a driver query nor GPU execution; the reference is a sealed
historical target. The same-XLA-flags control isolates the CompileOnly tag.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--meta', type=Path, required=True)
p.add_argument('--bundle-manifest', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
from gpuwrf.runtime.aot_executable import fingerprint_matches
with a.meta.open('rb') as f:
    meta = pickle.load(f)
bundle = json.loads(a.bundle_manifest.read_text())
reference = bundle['toolchain']['target_fingerprints'][0]
control = {**reference, 'xla_flags': meta.fingerprint['xla_flags']}
different = {k: [v, control.get(k)] for k, v in meta.fingerprint.items() if v != control.get(k)}
assert different == {'cuda_driver': ['PJRT C API\nNVIDIA GeForce RTX 5090', 'PJRT C API\ncuda 13000']}
result = dict(scope=__doc__, saved=meta.fingerprint, reference=reference,
              reference_manifest_sha256=hashlib.sha256(a.bundle_manifest.read_bytes()).hexdigest(),
              meta_sha256=hashlib.sha256(a.meta.read_bytes()).hexdigest(),
              ordinary_fingerprint_matches=fingerprint_matches(meta.fingerprint, reference),
              same_flags_control_matches=fingerprint_matches(meta.fingerprint, control),
              isolated_differences=different, physical_backend='cpu', gpu_executed=False)
assert result['same_flags_control_matches'] is False
a.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'matches': False, 'cause': 'CompileOnly cuda_driver target tag differs', 'gpu_executed': False}))
