"""Real-input queue preparation and default traced-source identity; no backend."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
import jax._src.xla_bridge as xb
attempts = []
def refuse(*args, **kwargs):
    attempts.append(str((args, kwargs)))
    raise AssertionError('backend initialized during queue preparation')
xb.get_backend = refuse
from gpuwrf.cli import main
from gpuwrf.runtime import aot_cheap_key as ck

started = time.monotonic()
package = Path(__file__).resolve().parents[2] / 'src/gpuwrf'
old = a.baseline / 'src/gpuwrf'
current_files = ck._trace_reachable_source_files(package)
old_files = ck._trace_reachable_source_files(old)
assert current_files and old_files
relative = [str(f.relative_to(package)) for f in current_files]
assert relative == [str(f.relative_to(old)) for f in old_files]
assert all((old / r).read_bytes() == (package / r).read_bytes() for r in relative)
closure = ck._source_tree_content_digest(package, current_files)
assert closure == ck._source_tree_content_digest(old, old_files)
a.out.mkdir(parents=True, exist_ok=False)
manifest_path = Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/sealed/BUNDLE_MANIFEST.json')
sealed = json.loads(manifest_path.read_text())
wn3 = Path('<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases/wn3_0227')
expected = sealed['inputs'][str(wn3)]
from gpuwrf.runtime.precompile_cli import sha_file
assert {name: sha_file(wn3 / name) for name in expected} == expected
cases = {'PROD': Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/prod_case/input'), 'WN3': wn3}
receipts = {}
for name, case in cases.items():
    manifest = a.out / f'{name}.json'
    manifest.write_text(json.dumps({'variants': [{'name': name.lower(), 'namelist': str(case / 'namelist.input')}]}))
    t = time.monotonic()
    rc = main(['precompile', '--namelist-variants', str(manifest), '--input-dir', str(case),
               '--cache-dir', str(a.out / f'{name}_cache'), '--work-dir', str(a.out / f'{name}_work'), '--hours', '3'])
    assert rc == 0
    assert not (a.out / f'{name}_cache').exists()
    report = json.loads((a.out / f'{name}_work/report.json').read_text())
    assert report['variants'][0]['status'] == 'QUEUED'
    receipts[name] = dict(wall_s=time.monotonic()-t, report=report)
result = dict(scope=__doc__, trace_closure_sha256=closure, trace_closure_files=len(relative),
              model_source_BYTE=True, baseline=str(a.baseline), candidate=str(package),
              inherited_release_jaxpr={'Swiss':'b977712f', 'PROD':'321aa651'},
              numerical_revalidation='not repeated; exact traced-source identity, CLI additions only',
              backend_attempts=attempts, wn3_input_sha256=expected,
              reference_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              queues=receipts, wall_s=time.monotonic()-started,
              peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
assert not attempts
(a.out / 'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
