"""Run the pinned producer's E156 probe on a verified, read-only git archive.

Only archive provenance and the output root differ from the original harness.
No lowering, initialization, controls, or compiler-option code is modified.
"""
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v45-seam')
SOURCE = ROOT / 'snapshot'
COMMIT = sys.argv.pop(1)
assert len(COMMIT) == 40 and all(c in '0123456789abcdef' for c in COMMIT)
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert os.sched_getaffinity(0) <= {12, 28}

# Verify every archived tracked blob against the sealed git ls-tree manifest.
for record in (ROOT / 'source_git_manifest.bin').read_bytes().split(b'\0'):
    if not record:
        continue
    meta, relative = record.split(b'\t', 1)
    mode, kind, expected = meta.split()
    path = SOURCE / os.fsdecode(relative)
    content = os.fsencode(os.readlink(path)) if mode == b'120000' else path.read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    assert actual == expected.decode(), str(path)
del content

harness = SOURCE / 'proofs/v45_lsmparams/deviceless.py'
code = harness.read_text()
original_sha = hashlib.sha256(code.encode()).hexdigest()
assert original_sha == 'f3561ab8fb638e2aad54245e4bcb0b333b91311c017bb72cf2f2970ee1d14f3d'
changes = {
    "Path('<USER_HOME>/wrf_gpu2_lanes/v45-lsmparams')": repr(ROOT),
    "assert subprocess.run(['git', '-C', str(a.source), 'symbolic-ref', '--quiet', 'HEAD'], capture_output=True).returncode != 0":
        "assert a.source.resolve() == Path('<USER_HOME>/wrf_gpu2_lanes/v45-seam/snapshot')",
    "dirty = subprocess.check_output(['git', '-C', str(a.source), 'status', '--porcelain', '--untracked-files=no'])":
        "dirty = b''  # All archived blobs verified against git object IDs above.",
    "subprocess.check_output(\n    ['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip()": repr(COMMIT),
}
for old, new in changes.items():
    assert code.count(old) == 1, old
    code = code.replace(old, new)
# repr(Path) would require PosixPath in the producer namespace.
code = code.replace("PosixPath('<USER_HOME>/wrf_gpu2_lanes/v45-seam')",
                    "Path('<USER_HOME>/wrf_gpu2_lanes/v45-seam')")
sys.argv[0] = str(harness)
exec(compile(code, str(harness), 'exec'), {'__name__': '__main__', '__file__': str(harness)})
