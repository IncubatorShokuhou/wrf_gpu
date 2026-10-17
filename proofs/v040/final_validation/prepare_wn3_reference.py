"""Reuse the accepted original WN3 prefix:75 unchanged CPU history frames."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
from prepare import ROOT, sha, write_json
from cpu_references import inventory
from gpuwrf.io.namelist_parser import parse_namelist
from score import reference_schedule

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
case=ROOT/'WN3-smoke';plan=json.loads((case/'PLAN.json').read_text())
assert not (case/'gpu_run.json').exists()
base=Path('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run')
assert {p.name:sha(p) for p in (case/'input').iterdir()} == plan['input_sha256']
rsl=base/'rsl.out.0000';assert 'SUCCESS COMPLETE WRF' in rsl.read_text(errors='replace')
view=case/'cpu';view.mkdir(exist_ok=True)
for domain in plan['domains']:
    frames=sorted(base.glob(f'wrfout_{domain}_*'))[:25]
    assert len(frames)==25
    for p in frames:
        target=view/p.name
        if not target.exists():target.symlink_to(p)
frames,nonfinite,complete=inventory(case,plan['domains'],24)
assert complete and not nonfinite and len(frames)==75
nl=parse_namelist(case/'input/namelist.input')
for domain in plan['domains']:
    stamps=[datetime.strptime(f['times'][0],'%Y-%m-%d_%H:%M:%S') for f in frames if f['file'].startswith('wrfout_'+domain)]
    reference_schedule(nl,stamps,24)
    assert all('OUTPUT FROM WRF V4.' in f['globals']['TITLE'] for f in frames if f['file'].startswith('wrfout_'+domain))
write_json(case/'CPU_OUTPUTS.json',frames)
receipt=dict(schema='v040-reused-original-wn3-reference-v1',case='WN3-smoke',status='COMPLETE',
    actual_input_sha256=plan['input_sha256'],input_sha256=plan['input_sha256'],
    frame_count=75,nonfinite=[],success_marker=True,rc=None,monotonic_wall_s=None,
    output_manifest=str(case/'CPU_OUTPUTS.json'),output_manifest_sha256=sha(case/'CPU_OUTPUTS.json'),
    original_rsl=str(rsl),original_rsl_sha256=sha(rsl),
    executable=dict(original='Previously accepted original operational CPU-WRF WN3 reference',original_sha256=None,
        historical_binary_unavailable=True),
    scope='Reuse first24h of successful original72h reference; no CPU model rerun; historical returncode/binarySHA unknown, never fabricated',
    audited_utc=datetime.now(timezone.utc).isoformat())
write_json(case/'cpu_run.json',receipt)
plan.update(forecast_hours=24,original_cpu_receipt_sha256=sha(case/'cpu_run.json'),
    qualification='Actual24h original-CPU-WRF D6/full-field gate plus first3h/12-frame released-GPU byte regression; distinct verdicts',
    schedule='Standalone official lock; cold24h arm timeout1800s; hard end19:15Z')
write_json(case/'PLAN.json',plan)
print(json.dumps(receipt,indent=2))
