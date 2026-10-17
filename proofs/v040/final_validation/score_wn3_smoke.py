"""Compare every released full-file byte hash; never reinterpret a mismatch."""
import json
import os
from prepare import ROOT, sha, write_json

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
case=ROOT/'WN3-smoke';plan=json.loads((case/'PLAN.json').read_text())
rows=[]
for old in plan['baseline_frames']:
    path=case/'gpu'/old['filename']
    digest=sha(path) if path.exists() else None
    rows.append(dict(file=old['filename'],domain=old['nest'],tau_h=old['tau_h'],
        expected_sha256=old['sha256'],actual_sha256=digest,equal=digest==old['sha256']))
result=dict(scope='GPU-vs-released-GPU byte regression only; NOT CPU-WRF fidelity',
    verdict='PASS' if all(r['equal'] for r in rows) else 'FAIL',
    equal=sum(r['equal'] for r in rows),expected=12,frames=rows,
    all_program_identity_gate='Separate prerequisite, not inferred from smoke')
write_json(case/'BYTE_REGRESSION.json',result);print(json.dumps(result,indent=2))
