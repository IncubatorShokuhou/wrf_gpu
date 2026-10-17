"""Bind the current GPU evidence without promoting incomplete/historical arms."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def record(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) if path.exists() else None


def device_accounting(directory):
    """Report retained device observations; never infer a justified F2 verdict."""
    census_path = directory / 'census.json'
    extrema_path = directory / 'f2_negative_summary.json'
    census, extrema = read(census_path), read(extrema_path)
    domains = {}
    for name, domain in census.get('domains', {}).items():
        observed = extrema.get('domains', {}).get(name, {}).get('records', [])
        events = domain.get('f2', {}).get('events', [])
        guards = domain.get('guards', {})
        def worst(moisture):
            candidates = [r for r in observed if r.get('minimum_negative') and
                          (r['field'] in {'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'qh'}) == moisture]
            item = min(candidates, key=lambda r: r['minimum_negative']['value'], default=None)
            return {k: item[k] for k in ('phase', 'field', 'minimum_negative')} if item else None
        domains[name] = dict(
            actual_work=domain.get('actual_work'),
            negative_phase_visits=sum(r.get('negative_count', 0) for r in observed) if observed else None,
            interior_negative_phase_visits=sum(r.get('interior_count', 0) for r in observed) if observed else None,
            raw_nonfinite_phase_visits=sum(r.get('nonfinite_count', 0) for r in observed) if observed else None,
            event_counts={kind: sum(e.get('count', 0) for e in events if e.get('event') == kind)
                          for kind in sorted({e['event'] for e in events})},
            nonzero_guards={key: value for key, value in guards.items()
                            if any(value.values())},
            worst_moisture_negative=worst(True),
            worst_number_negative=worst(False),
            full_output_set=domain.get('full_output_set'))
    return dict(census=record(census_path), extrema=record(extrema_path), domains=domains,
                disposition='OPEN: observed counts/extrema only; phase visits are not unique cells. '
                            'Producer full_output_set is not proof of a complete forecast clock. '
                            'A matched-control justification and F2 decision are separate.')


rows = []
qh_policy = read(Path(__file__).parent / 'TOMORROW_GPU_PLAN.json').get('qh_mass_rerun', {})
qh_science_verified = set(qh_policy.get('science_verified_on_final_source',
                                      qh_policy.get('qualified_on_final_source', [])))
qh_provisional = set(qh_policy.get('cases', [])) - qh_science_verified
for item in read(ROOT / 'MATRIX.json')['cases']:
    if not item.get('gpu_eligible', True):
        continue
    ident = item['id']
    case = ROOT / ident
    launch = read(case / 'gpu_launch.json')
    reuse = read(case / 'GPU_REUSE.json')
    run_path = Path(reuse['producer_receipt']) if reuse else case / 'gpu_run.json'
    run = read(run_path)
    qualification = read(case / 'QUALIFICATION.json')
    native_gate = read(case / 'e41/wave_gate.json')
    accepted_scope = read(case / 'ADMITTED_SCOPE_GATE.json')
    domains = {}
    for domain in item['domains']:
        path = case / f'D6_{domain}.json'
        gate = read(path)
        domains[domain] = dict(
            receipt=record(path), verdict=gate.get('verdict'),
            paired_frames=gate.get('paired_frames'), coverage_complete=gate.get('coverage_complete'),
            bounded_misses=len(gate['failures']) if 'failures' in gate else None,
            raw_output_nonfinite=len(gate['nonfinite']) if 'nonfinite' in gate else None,
            missing_cpu_fields=sorted({name for frame in gate.get('by_frame', [])
                                      for name in frame.get('missing_vs_cpu', [])}))
    publication = read(case / 'WRITER_RECHECK.json')
    raw_domains = None
    if publication.get('verdict') == 'PASS' and publication.get('forecast_receipt_sha256') == (record(run_path) or {}).get('sha256'):
        raw_domains, domains = domains, {}
        for domain in item['domains']:
            sealed_gate = publication['domains'][domain]
            path = Path(sealed_gate['path'])
            assert record(path)['sha256'] == sealed_gate['sha256']
            gate = read(path)
            domains[domain] = dict(receipt=record(path), verdict=gate['verdict'],
                paired_frames=gate['paired_frames'], coverage_complete=gate['coverage_complete'],
                bounded_misses=len(gate['failures']), raw_output_nonfinite=len(gate['nonfinite']),
                missing_cpu_fields=sorted({name for frame in gate['by_frame']
                                          for name in frame['missing_vs_cpu']}))
    error_lines = []
    if (case / 'gpu.log').exists():
        error_lines = [line for line in (case / 'gpu.log').read_text().splitlines()
                       if 'gpuwrf: error:' in line]
    expected = 24 if ident == 'c49' else 25
    complete = run.get('rc') == 0 and all(d['paired_frames'] == expected for d in domains.values())
    full_d6 = complete and all(d['verdict'] == 'PASS' for d in domains.values())
    rows.append(dict(
        case=ident, options=item['options'], domains=domains,
        raw_measured_domains_before_writer_recheck=raw_domains,
        host_writer_recheck=publication if raw_domains is not None else None,
        host_writer_recheck_receipt=record(case / 'WRITER_RECHECK.json'),
        original_cpu_receipt=record(case / 'cpu_run.json'),
        reference_of_record=read(case / 'REFERENCE_OF_RECORD.json') or None,
        reference_of_record_receipt=record(case / 'REFERENCE_OF_RECORD.json'),
        matched_inputs=record(case / 'inputs.json'),
        launch=record(case / 'gpu_launch.json'), actual_gpu_receipt=record(run_path),
        reuse_receipt=record(case / 'GPU_REUSE.json'), reused=bool(reuse),
        current_main_identity=reuse.get('current_main_identity'),
        source_commit=run.get('source_commit', run.get('source_rev', launch.get('source_commit'))),
        platform=run.get('platform'), process_rc=run.get('rc', launch.get('process_rc')),
        forecast_complete=complete, full_d6_pass=full_d6,
        qh_mass_provisional=ident in qh_provisional,
        qh_mass_science_verified=ident in qh_science_verified,
        accepted_scope_gate=accepted_scope or None,
        accepted_scope_receipt=record(case / 'ADMITTED_SCOPE_GATE.json'),
        execution_errors=error_lines,
        resident_nonfinite_failure=any('NonFiniteStateError' in line for line in error_lines),
        e41=reuse.get('e41') if reuse else qualification.get('e41', native_gate.get('e41_gate')),
        external_e41=reuse.get('external_e41') if reuse else None,
        local_pin_map=record(case / 'e41/PIN_MAP.json'),
        device_accounting=device_accounting(case / 'gpu'),
        cache_remaining=(case / 'cache').exists(),
        cold_shared_wall_s=run.get('monotonic_wall_s', run.get('wall_s')),
        sampled_vram_peak_mib=run.get('sampled_process_vram_mib_peak', run.get('peak_process_vram_mib_sampled'))))

result = dict(
    schema='v040-v-prep-gpu-status-v1', utc=datetime.now(timezone.utc).isoformat(),
    reporter_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    scope='Each result is evidence at its cited measured source. No later-main qualification, S1 speed claim, or F2 pre-mask closure is inferred.',
    prepared_cpu_manifest=record(Path(__file__).parent / 'CPU_REFERENCES.json'),
    active_composites=len(rows), full_forecasts=sum(r['forecast_complete'] for r in rows),
    full_d6_passes=sum(r['full_d6_pass'] for r in rows),
    full_d6_passes_without_qh_provisional=sum(r['full_d6_pass'] and not r['qh_mass_provisional'] for r in rows),
    accepted_short_scope_passes=sum(bool(r['accepted_scope_gate'])
        and r['accepted_scope_gate'].get('accepted_D6') == 'PASS'
        and not r['full_d6_pass'] for r in rows),
    qh_mass_provisional_cases=sorted(qh_provisional),
    reused_forecasts=sum(r['reused'] for r in rows),
    resident_nonfinite_cases=[r['case'] for r in rows if r['resident_nonfinite_failure']],
    e41_counts=dict(Counter(str(r['e41']) for r in rows)),
    caches_remaining=[r['case'] for r in rows if r['cache_remaining']],
    wn3_full_24h=read(Path(__file__).parent / 'WN3_GPU14_D6.json') or {
        'status': 'PENDING; no completed release-case receipt available'},
    rows=rows)
target = Path(__file__).parent / 'GPU_STATUS.json'
target.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
