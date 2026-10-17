"""CPU final-tree release identities and c03 selected-program reuse proof.

Execute the unchanged official release_identity.py. A private mount redirects
only its fixed scratch directory into this lane's approved disk spillover.
No model execution, GPU access, or scientific trajectory comparison occurs.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

candidate, baseline, output, scratch = map(lambda x: Path(x).resolve(), sys.argv[1:])
assert os.environ['JAX_PLATFORMS'] == 'cpu'
output.mkdir(parents=True, exist_ok=True)
scratch.mkdir(parents=True, exist_ok=True)
expected = {
    'swiss': 'f672a5739cb96e388b6b7a40ef80337bf81bdc14ef15268f3476e09d79fe055d',
    'prod': '6fc715a2bd630934c4cf0f94427df19a527cbf429011c9ba5efe4d712744d27b',
}
jobs = [('swiss', candidate, '<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu'),
        ('prod', candidate, '<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725'),
        ('c03_base', baseline, '<USER_HOME>/wrf_gpu2_lanes/v-prep/c03/input'),
        ('c03_final', candidate, '<USER_HOME>/wrf_gpu2_lanes/v-prep/c03/input'),
        ('c26_final', candidate, '<USER_HOME>/wrf_gpu2_lanes/v-prep/c26/input')]
rows = {}
for name, tree, case in jobs:
    if Path('/tmp/wrf_gpu2_quiet').exists():
        raise SystemExit(3)
    script = tree / 'proofs/s1_port/release_identity.py'
    commit = subprocess.check_output(['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(tree), 'status', '--porcelain'], text=True)
    artifact = output / (name + '.json')
    with (output / (name + '.log')).open('w') as log:
        subprocess.run(['timeout', '300', 'bwrap', '--bind', '/', '/', '--dev-bind', '/dev', '/dev', '--bind', str(scratch),
                        '<USER_HOME>/wrf_gpu2_lanes/s1-port', sys.executable, str(script),
                        str(tree), str(artifact), case], stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(artifact.read_text())['arms']['release']
    assert 'error' not in result, result
    if name in expected:
        assert result['sha'] == expected[name], result
    rows[name] = dict(commit=commit, source=str(tree), case=case,
                      official_probe_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                      artifact=str(artifact), **result)
    (output / 'receipt.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(name, result['sha'], result['chars'], flush=True)
assert rows['c03_base']['sha'] == rows['c03_final']['sha'], 'SF4 changed the c03 selected program'
assert rows['c03_base']['chars'] == rows['c03_final']['chars']
rows['verdict'] = 'EXACT defaults and selected c03 program; c26 full-step trace succeeds'
(output / 'receipt.json').write_text(json.dumps(rows, indent=2) + '\n')
