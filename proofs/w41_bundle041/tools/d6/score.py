"""Frozen D6, every CPU field/frame, and raw finite audits of BOTH arms."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

FROZEN_SHA = '9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc'


def reference_schedule(nl, stamps, hours):
    """Check WRF alarms on the requested clock, including DFL's filtered start."""
    def date(group, prefix):
        return datetime(*(int(group.get(prefix+'_'+k,0) if k in ('minute','second')
                              else group[prefix + '_' + k]) for k in
                          ('year', 'month', 'day', 'hour', 'minute', 'second')))
    tc = {k: v[0] if isinstance(v, list) else v for k, v in nl['time_control'].items()}
    start = date(tc, 'start')
    dt = float(nl['domains']['time_step'])
    initial = start
    dfi = nl.get('dfi_control', {})
    if dfi.get('dfi_opt') == 1:
        initial += (date(dfi, 'dfi_fwdstop') - start) / 2
    count=int((start+timedelta(hours=hours)-initial).total_seconds()//3600)+1
    assert len(stamps) == count
    assert stamps[0] == initial, 'Original-WRF initial/filtered clock mismatch'
    for hour, stamp in enumerate(stamps[1:], 1):
        late = (stamp - initial - timedelta(hours=hour)).total_seconds()
        assert 0 <= late < dt + .01, 'Incomplete original-WRF alarm coverage'
    return start, list(range(count)), [(s - start).total_seconds() for s in stamps]


def audit_pair(cpu_path, gpu_path):
    with Dataset(cpu_path) as cpu, Dataset(gpu_path) as gpu:
        missing = sorted(set(cpu.variables) - set(gpu.variables))
        schema, nonfinite = [], []
        for name in sorted(set(cpu.variables) & set(gpu.variables)):
            a, b = cpu[name], gpu[name]
            if a.dimensions != b.dimensions or a.shape != b.shape:
                schema.append(dict(field=name, cpu_dimensions=a.dimensions, gpu_dimensions=b.dimensions,
                                   cpu_shape=a.shape, gpu_shape=b.shape))
        for label, ds in (('cpu', cpu), ('gpu', gpu)):
            ds.set_auto_mask(False)
            for name, var in ds.variables.items():
                if np.issubdtype(var.dtype, np.number):
                    n = int(np.count_nonzero(~np.isfinite(var[:])))
                    if n:
                        nonfinite.append(dict(arm=label, field=name, count=n))
        header_missing = sorted(set(cpu.ncattrs()) - set(gpu.ncattrs()))
        header_differences = [dict(name=k, cpu=str(cpu.getncattr(k)), gpu=str(gpu.getncattr(k)))
                              for k in cpu.ncattrs() if k in gpu.ncattrs()
                              and not np.array_equal(cpu.getncattr(k), gpu.getncattr(k))]
        return dict(cpu_file=str(cpu_path), gpu_file=str(gpu_path),
                    cpu_field_count=len(cpu.variables), gpu_field_count=len(gpu.variables),
                    missing_vs_cpu=missing, schema_mismatches=schema, nonfinite=nonfinite,
                    missing_header_attributes=header_missing, header_differences=header_differences)


def score(source, case, domain, hours=24):
    spec = importlib.util.spec_from_file_location('final_grid_compare', source / 'scripts/compare_wrfout_grid.py')
    compare = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compare
    spec.loader.exec_module(compare)
    compare.Dataset = Dataset
    manifest = source / 'proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest == FROZEN_SHA, 'Frozen scientific tolerances changed'
    # Root dt54 does not divide one hour: real WRF alarms can occur at
    # 01:00:18/02:00:36. Require the SEALED original times, not invented exact
    # hour labels. Exact CPU/GPU Times and all25 observations remain binding.
    from gpuwrf.io.namelist_parser import parse_namelist
    nl = parse_namelist(case / 'input/namelist.input')
    root_dt = float(nl['domains']['time_step'])
    cpu_receipt = json.loads((case / 'cpu_run.json').read_text())
    sealed_path = case / 'CPU_OUTPUTS.json'
    assert hashlib.sha256(sealed_path.read_bytes()).hexdigest() == cpu_receipt['output_manifest_sha256']
    sealed = [r for r in json.loads(sealed_path.read_text()) if r['file'].startswith(f'wrfout_{domain}_')]
    stamps = [datetime.strptime(r['times'][0], '%Y-%m-%d_%H:%M:%S') for r in sealed]
    start, expected_leads, actual_seconds = reference_schedule(nl, stamps, hours)
    expected_frames=len(expected_leads)
    for r in sealed:
        assert hashlib.sha256((case / 'cpu' / r['file']).read_bytes()).hexdigest() == r['sha256']
    args = argparse.Namespace(cpu_dir=case / 'cpu', gpu_dir=case / 'gpu', domain=domain,
        init=start.strftime('%Y-%m-%d_%H:%M:%S'), min_lead=0, max_lead=hours, vars=None, tolerance_json=manifest,
        boundary_width=5, no_spatial_splits=True, progress=50)
    report = compare.build_report(args)
    (case / f'compare_{domain}.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    pairing = report['pairing']
    rows = []
    for pair in pairing['pairs']:
        row = audit_pair(pair['cpu_file'], pair['gpu_file'])
        row['lead_h'] = pair['lead_h']
        rows.append(row)
    failures, incomplete, fields = [], [], {}
    for name, field in report['field_summaries'].items():
        if name == 'Times':
            if not field['all_equal']:
                incomplete.append(dict(field=name, reason='Times mismatch'))
            continue
        leads = field.get('by_lead', [])
        bad = [dict(field=name, lead_h=x['lead_h'], checks=x['tolerance_result'].get('failures', []))
               for x in leads if x.get('tolerance_result', {}).get('pass') is False]
        failures.extend(bad)
        if field.get('compared_lead_count') != expected_frames or field.get('missing_leads') or field.get('incompatible_leads'):
            incomplete.append(dict(field=name, reason='missing/incompatible frames'))
        fields[name] = dict(verdict='FAIL' if bad else
                            ('PASS' if field.get('tolerance_result', {}).get('supplied') else 'REPORT_ONLY'),
                            overall=field.get('overall'), by_lead=leads)
    nonfinite = [dict(lead_h=r['lead_h'], **n) for r in rows for n in r['nonfinite']]
    coverage_ok = (pairing['common_leads_h'] == expected_leads
        and pairing['cpu_file_count'] == expected_frames and pairing['gpu_file_count'] == expected_frames
        and not pairing['unmatched_cpu_count'] and not pairing['unmatched_gpu_count']
        and not incomplete and not report['inventory']['incompatible_common']
        and all(not r['missing_vs_cpu'] and not r['schema_mismatches'] for r in rows))
    result = dict(schema='v040-final-grid-gate-v1', case=case.name, domain=domain,
                  reference='original CPU-WRF; GPU-vs-GPU is never fidelity evidence',
                  manifest_sha256=digest,
                  verdict='FAIL' if failures or nonfinite else ('PASS' if coverage_ok else 'INCOMPLETE'),
                  coverage_complete=coverage_ok, paired_frames=len(rows), by_frame=rows,
                  expected_original_cpu_leads_h=expected_leads,
                  actual_original_cpu_seconds=actual_seconds,
                  failures=failures, nonfinite=nonfinite, incomplete=incomplete, fields=fields,
                  headers='every frame audited; differences disclosed separately from numerical D6',
                  note='Unbounded fields are REPORT_ONLY; field inventory alone is not numerical qualification')
    (case / f'D6_{domain}.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('case', type=Path)
    p.add_argument('--domains', nargs='+', default=['d01'])
    a = p.parse_args()
    results = [score(a.source.resolve(), a.case.resolve(), d) for d in a.domains]
    print(json.dumps({r['domain']: r['verdict'] for r in results}))
    sys.exit(0 if all(r['verdict'] == 'PASS' for r in results) else 1)
