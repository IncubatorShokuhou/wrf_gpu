"""Pin authentic WN3 inputs and retained released byte hashes for a 3h smoke."""
import json
import os
from pathlib import Path
from prepare import ROOT, sha, write_json
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.namelist_binding import namelist_honour_report

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
baseline = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate/arms0227/W3R/twin_manifest_20260227_18z_72h.json')
old = json.loads(baseline.read_text())
base = Path('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run')
case = ROOT / 'WN3-smoke'
inp = case / 'input'
inp.mkdir(parents=True, exist_ok=True)
assert not (case / 'gpu_run.json').exists()
expected = dict(old['input_sha256'])
expected['namelist.input'] = old['gpu_input_attestation']['files']['namelist.input']['sha256']
for name, digest in expected.items():
    assert sha(base / name) == digest, name
    target = inp / name
    if not target.exists():
        target.symlink_to(base / name)
    assert sha(target) == digest
groups = parse_namelist(inp / 'namelist.input')
errors, warnings = namelist_honour_report(groups, ('d01','d02','d03'), replay=False)
assert not errors, errors
frames = [dict(r, filename=Path(r['path']).name) for r in old['frames'] if r['tau_h'] <= 3]
assert len(frames) == 12
for domain in ('d01','d02','d03'):
    assert [r['tau_h'] for r in frames if r['nest'] == domain] == [0,1,2,3]
manifest = dict(schema='v040-wn3-release-smoke-plan-v1', hours=3, domains=['d01','d02','d03'],
    input_sha256=expected, admission_errors=errors, admission_warnings=warnings,
    baseline_manifest=str(baseline), baseline_manifest_sha256=sha(baseline),
    baseline_gpu_commit=old['gpu_commit'], baseline_frames=frames,
    baseline_scope='Actual v0.3.3 cf873 W3R; later v0.3.4 carry requires its independent identity certificate.',
    qualification='GPU-vs-released-GPU byte regression only; not original-WRF fidelity.',
    acceptance='All12 full-file SHA256 equal; if unequal, fail carry and preserve differences. No numeric tolerance substitution.',
    carry_requirements=['All three domains and fused compiled programs/constants exact to accepted release',
        'Resolved physics, inputs, cadence and source/provider assets exact',
        'Preserve original D6/station limitations; no global historical PASS inferred'],
    original_reference='Existing original-WRF WN3 qualification receipts remain authoritative; smoke is not a replacement',
    clock='dt54/18/6; 3h minimum (200 root steps), sealed original actual history Times',
    schedule='Standalone official v-prep lock; <=60min cold-setup or <=30min warm smoke; hard end19:15Z')
write_json(case / 'PLAN.json', manifest)
print(json.dumps(manifest, indent=2))
