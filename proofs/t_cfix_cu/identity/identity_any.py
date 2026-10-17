"""Run a tree's own unchanged official release probe (proofs/s1_port/release_identity.py) with lane-local temp
files and NO expected-hash assertion: records the release jaxpr sha for cross-tree comparison.
usage: identity_any.py SOURCE_ROOT OUTPUT_JSON [CASE]"""
from pathlib import Path
import hashlib, json, sys
root, output = (Path(s).resolve() for s in sys.argv[1:3])
probe = root / 'proofs/s1_port/release_identity.py'
source = probe.read_text()
assert source.count('<USER_HOME>/wrf_gpu2_lanes/s1-port') == 1
source = source.replace('<USER_HOME>/wrf_gpu2_lanes/s1-port', str(output.parent))
exec(compile(source, str(probe), 'exec'), {'__file__': str(probe), '__name__': '__main__'})
receipt = json.loads(output.read_text())
receipt['lane_probe'] = {'upstream_sha256': hashlib.sha256(probe.read_bytes()).hexdigest(), 'root': str(root),
                         'only_change': 'lane-local temporary path; no expected-hash assertion'}
output.write_text(json.dumps(receipt, indent=2) + '\n')
print('RELEASE_SHA', receipt['arms']['release']['sha'], receipt.get('case'))
