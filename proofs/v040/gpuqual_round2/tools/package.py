"""Keep small reviewable receipts in git and hash complete external evidence."""
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-run')
DEST = Path(sys.argv[1]) / 'proofs/v040/gpuqual_round2'
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
plan = json.loads((ROOT / 'PLAN.json').read_text())
status = json.loads((ROOT / 'STATUS.json').read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

DEST.mkdir(exist_ok=True)
rows = []
for config in plan['configs']:
    name = config['config']
    label = config.get('display_name', name)
    case, out = ROOT / name, DEST / label
    out.mkdir(exist_ok=True)
    raw_f2 = case / 'gpu/f2_negative_summary.json'
    disclosure = case / 'F2_DISCLOSURE.json'
    if (case / 'grid_gate.json').exists() and raw_f2.exists() and not disclosure.exists():
        domains = json.loads(raw_f2.read_text())['domains']
        records = [record for domain in domains.values() for record in domain['records']]
        write(disclosure, {'config': name, 'phase_field_rows': len(records),
            'negative_phase_field_visits': sum(r['negative_count'] for r in records),
            'interior_phase_field_visits': sum(r['interior_count'] for r in records),
            'nonfinite_phase_field_visits': sum(r['nonfinite_count'] for r in records),
            'nonzero_records': [{'phase': r['phase'], 'field': r['field'],
                'visits': r['negative_count'], 'interior': r['interior_count'],
                'minimum': r['minimum_negative']} for r in records if r['negative_count']],
            'interpretation': 'Repeated phase-field visits; no blanket every-event/F2 closure.'})
    row = dict(config=label, artifact_config=name, required=config.get('required', True),
               **status['done'].get(name, {'verdict': 'NOT_RUN'}))
    rows.append(row)
    artifacts = {}
    for file in ('gpu_run.json', 'launch.json', 'inputs.json', 'cleanup.json', 'GPU_DONE.json', 'F2_DISCLOSURE.json'):
        if (case / file).exists():
            shutil.copyfile(case / file, out / file)
    for origin, dest in [('input/namelist.input', 'namelist.input'),
                         ('e41/wave_gate.json', 'E41.json'), ('e41/PIN_MAP.json', 'E41_PIN_MAP.json'),
                         ('gpu/f2_negative_summary.json', 'f2_negative_summary.json')]:
        if (case / origin).exists():
            shutil.copyfile(case / origin, out / dest)
    original_receipt = (Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/verified_receipt.json')
        if name in ('release_default', 'release_default_re1') else Path('<USER_HOME>/wrf_gpu2_lanes/t-fdda/TF02/full24_run.json')
        if name == 'fdda1' else Path('<USER_HOME>/wrf_gpu2_lanes/s1-gpuqual') / name / 'cpu_run.json')
    shutil.copyfile(original_receipt, out / 'cpu_run.json')
    for path in [original_receipt, *[case / n for n in ('grid_gate.json', 'compare.json',
            'gpu_cuda.log', 'score.log', 'e41.log')]]:
        if path.exists():
            artifacts[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    write(out / 'provenance.json', {'source_rev': plan['source_rev'],
        'matched_inputs': config['matched_inputs'], 'original_CPU_reference': config['reference'],
        'raw_artifact_root': str(case), 'artifacts': artifacts,
        'reference_reuse': config['reference_reuse']})
    if not (case / 'grid_gate.json').exists():
        write(out / 'D6.json', row)
        continue
    d6 = json.loads((case / 'grid_gate.json').read_text())
    headers = d6.pop('headers_by_frame')
    d6['header_audit'] = {'frames': len(headers),
        'missing_attrs': sorted({a for h in headers for a in h['missing_vs_cpu']}),
        'differing_attrs': sorted({a['name'] for h in headers for a in h['differences']})}
    for f in d6['fields'].values():
        leads = f.pop('by_lead')
        finite = [v for v in leads if v.get('rmse') is not None]
        f['worst_rmse_lead'] = max(finite, key=lambda x: x['rmse']) if finite else None
        f['worst_max_abs_lead'] = max(finite, key=lambda x: x['max_abs']) if finite else None
    d6['raw_full_receipt'] = {'path': str(case / 'grid_gate.json'),
                            **artifacts[str(case / 'grid_gate.json')]}
    write(out / 'D6.json', d6)
summary = {'utc': datetime.now(timezone.utc).isoformat(), 'source_rev': plan['source_rev'],
    'rows': [r for r in rows if r['required']],
    'diagnostic_variants': [r for r in rows if not r['required']],
    'required_config_count': sum(r['required'] for r in rows),
    'pending': status['pending'], 'complete': not status['pending'],
    'all_required_arms_processed': not status['pending'],
    'all_required_D6_coverage_pass': all(r['verdict'] == 'PASS' for r in rows if r['required']),
    'required_D6_counts': {verdict: sum(r['required'] and r['verdict'] == verdict for r in rows)
        for verdict in sorted({r['verdict'] for r in rows if r['required']})},
    'release_default_E41': next(r.get('E41', 'UNKNOWN') for r in rows if r['config'] == 'release_default'),
    'F2_status': 'OPEN; raw negative visits disclosed, narrow known-issue scope does not clear every arm',
    'timing_scope': 'cold concurrent Swiss qualification; not warm S1 or isolated throughput benchmark',
    'F2_scope': 'Raw counters disclosed separately; D6 does not justify negative visits.',
    'E41_policy': 'Experimental flags disclosed; release default hard gate remains binding.',
    'manifest_sha256': '9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc'}
write(DEST / 'summary.json', summary)
shutil.copyfile(ROOT / 'PLAN.json', DEST / 'PLAN.json')
for file in ('SOURCE_IDENTITY.json', 'REFERENCE_FINITE_REUSE.json', 'CPU_VALIDATION_FIXED.json', 'MP_RE_SCOPE_PROOF.json', 'STORAGE_MIGRATION.json', 'SUSPEND_SCOPE.json', 'HIBERNATION_20261014.log', 'SOURCE_CLEANUP.json', 'METADATA_DISCREPANCIES.json', 'FINAL_AUDIT.json'):
    if (ROOT / file).exists():
        shutil.copyfile(ROOT / file, DEST / file)
lines = ['| Config | D6 + coverage | Bounded D6 | Frames | Missing CPU fields | E41 | Cold concurrent s/fc-h | Sampled VRAM MiB |',
         '|---|---|---|---:|---|---|---:|---:|']
for r in rows:
    if not r['required']:
        continue
    lines.append(f'| {r["config"]} | {r["verdict"]} | {r.get("bounded_D6", "—")} | '
        f'{r.get("paired_frames", "—")} | {", ".join(r.get("missing_CPU_fields", [])) or "—"} | '
        f'{r.get("E41", "UNKNOWN")} | {r.get("cold_s_per_fc_h", "—")} | {r.get("sampled_peak_vram_MiB", "—")} |')
lines += ['', 'Every numerical bound is the unchanged frozen D6 bound, assessed at every hourly frame 0–24.',
          f'All required arms processed: {summary["all_required_arms_processed"]}. '
          f'All required D6 coverage gates pass: {summary["all_required_D6_coverage_pass"]}.',
          'Missing fields prevent full qualification. Unbounded fields are report-only.',
          'Times include cold compilation and shared concurrent work; they are not S1 speed claims.',
          'Raw F2 events remain separate evidence and do not become justified through a D6 pass.', '']
lines += ['An additional Swiss MP_RE=0 diagnostic is preserved separately. It is not the release-default qualification.', '']
(DEST / 'summary.md').write_text('\n'.join(lines))
checks = {str(p.relative_to(DEST)): {'sha256': sha(p), 'bytes': p.stat().st_size}
          for p in DEST.rglob('*') if p.is_file() and p.name != 'checksums.json'}
write(DEST / 'checksums.json', checks)
print('Packaged', len(rows), 'rows; pending', status['pending'])
