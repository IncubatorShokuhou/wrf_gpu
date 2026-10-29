"""Read-only compact-blob and cache-key policy proof; CPU only, no deserialize."""
import ast
import hashlib
import json
import os
import pickle
import subprocess
from pathlib import Path

assert os.environ['JAX_PLATFORMS'] == 'cpu'
ROOT = Path(os.environ['V05_AOTFP_SOURCE'])
from gpuwrf.runtime import aot_executable as ax, aot_cheap_key as ck


def base(rel):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', '286e2442a9:' + rel])


def definitions(data):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(data).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


rel = 'src/gpuwrf/runtime/aot_executable.py'
a, b = definitions(base(rel)), definitions((ROOT / rel).read_bytes())
changed = [name for name in a if a[name] != b[name]]
assert changed == ['fingerprint_matches', 'load'], changed
assert base('src/gpuwrf/runtime/aot_cheap_key.py') == (ROOT / 'src/gpuwrf/runtime/aot_cheap_key.py').read_bytes()
assert base('src/gpuwrf/__init__.py') == (ROOT / 'src/gpuwrf/__init__.py').read_bytes()
closure = ck._trace_reachable_source_files(ROOT / 'src/gpuwrf')
assert closure is not None
changed_rel = ['cli.py', 'runtime/aot_executable.py', 'runtime/aot_precompile.py']
# The source-key closure must not include any orchestration files changed here.
closure_paths = {str(p) for p in closure}
assert not any(str(ROOT / 'src/gpuwrf' / p) in closure_paths for p in changed_rel)
bundle = Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/jax/aot')
for path in sorted(bundle.glob('*/d01/k_*.meta')):
    meta_bytes = path.read_bytes()
    meta = pickle.loads(meta_bytes)
    # This real release kept rich trace labels but compacted the retained HLO
    # module. Attest that actual slim/thunk artifact, not a guessed label mode.
    if meta.retained_module == 'slim':
        break
else:
    raise AssertionError('no real compact domain blob')
blob_path = path.with_suffix('.xlaexec')
blob = blob_path.read_bytes()
digest = hashlib.sha256(blob).hexdigest()
assert digest == meta.blob_sha256
assert ax.fingerprint_matches(meta.fingerprint, meta.fingerprint)
assert ax._gpu_blob_format(blob) == meta.executable_format
# Compare the original predicate on the identical saved target, without GPU init.
namespace = {}
node = next(n for n in ast.parse(base(rel)).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint_matches')
exec('from __future__ import annotations\n' + ast.unparse(node), namespace)
assert namespace['fingerprint_matches'](meta.fingerprint, meta.fingerprint)
assert blob_path.read_bytes() == blob and path.read_bytes() == meta_bytes
print(json.dumps(dict(verdict='PASS', changed_executable_definitions=changed,
    serializer_ast='IDENTICAL', target_fingerprint_ast='IDENTICAL', cheap_key_source='BYTE_IDENTICAL',
    package_version='UNCHANGED', source_key_closure_count=len(closure), changed_paths_in_closure=0,
    real_compact_blob=str(blob_path), blob_sha256=digest, blob_bytes=len(blob),
    trace_metadata=meta.trace_metadata, retained_module=meta.retained_module,
    executable_format=meta.executable_format,
    meta_sha256=hashlib.sha256(meta_bytes).hexdigest(), cheap_key=meta.cheap_key,
    old_and_new_saved_target_match=True, blob_and_meta_after='BYTE_IDENTICAL',
    scope='read-only static/host evidence; no GPU deserialize or forecast claim'), indent=2))
