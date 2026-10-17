"""Apply 12:59Z owner decisions to existing receipts, without rerunning WRF."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from prepare import ROOT, sha, write_json
from gpuwrf.io.namelist_parser import parse_namelist
from score import reference_schedule

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
case = ROOT / 'c49'
path = case / 'cpu_run.json'
old = case / 'cpu_run_25_frame_refusal.json'
receipt = json.loads(path.read_text())
if not old.exists():
    assert receipt['status'] == 'FAILED'
    old.write_bytes(path.read_bytes())
frames = json.loads((case / 'CPU_OUTPUTS.json').read_text())
assert sha(case / 'CPU_OUTPUTS.json') == receipt['output_manifest_sha256']
stamps = [datetime.strptime(r['times'][0], '%Y-%m-%d_%H:%M:%S') for r in frames]
_, labels, seconds = reference_schedule(parse_namelist(case / 'input/namelist.input'), stamps, 24)
assert len(stamps) == 24 and receipt['rc'] == 0 and receipt['success_marker']
assert not receipt['nonfinite'] and not any('inventory_error' in r for r in frames)
receipt.update(status='COMPLETE', expected_frames_per_domain=24,
    owner_scope_decision='MAIN MAILBOX 2026-10-14T12:59:16Z: natural DFL24 actual Times accepted',
    prior_count_refusal=str(old), prior_count_refusal_sha256=sha(old),
    receipt_audit_utc=datetime.now(timezone.utc).isoformat(),
    actual_seconds=seconds, comparison_labels=labels,
    scope='Metadata-only acceptance; original bytes, inputs, timing and all-field inventory unchanged')
write_json(path, receipt)
matrix_path = ROOT / 'MATRIX.json'
matrix = json.loads(matrix_path.read_text())
row = next(r for r in matrix['cases'] if r['id'] == 'c63')
row.update(gpu_eligible=False,
    exclusion='Owner12:59Z: MP15 experimental, column-oracle-qualified; upstream prehistory SIGSEGV; no coupled reference')
write_json(matrix_path, matrix)
print(json.dumps(dict(c49=receipt['status'], frames=len(stamps), first=frames[0]['times'],
    last=frames[-1]['times'], c63_gpu_eligible=row['gpu_eligible']), indent=2))
