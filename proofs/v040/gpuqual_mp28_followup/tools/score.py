"""Full frozen-D6 grid score plus fail-closed 25-frame/CPU-field inventory."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
source = Path(sys.argv[1]).resolve()
case = Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location('gpuqual_compare', source / 'scripts/compare_wrfout_grid.py')
compare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compare
spec.loader.exec_module(compare)
compare.Dataset = Dataset  # unchanged scorer, process-wide NetCDF serialization
manifest = source / 'proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json'
args = argparse.Namespace(cpu_dir=case / 'cpu', gpu_dir=case / 'gpu', domain='d01',
                          init=None, min_lead=0, max_lead=24,
                          vars=None, tolerance_json=manifest, boundary_width=5,
                          no_spatial_splits=True, progress=50)
report = compare.build_report(args)
(case / 'compare.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
compare.write_markdown(report, case / 'compare.md')

pairing = report['pairing']
coverage = []
headers = []
nonfinite = []
for pair in pairing['pairs']:
    with Dataset(pair['cpu_file']) as cpu, Dataset(pair['gpu_file']) as gpu:
        missing = sorted(set(cpu.variables) - set(gpu.variables))
        schema = []
        for name in sorted(set(cpu.variables) & set(gpu.variables)):
            a, b = cpu[name], gpu[name]
            if a.dimensions != b.dimensions or a.shape != b.shape:
                schema.append({'field': name, 'cpu_dims': a.dimensions, 'gpu_dims': b.dimensions,
                               'cpu_shape': a.shape, 'gpu_shape': b.shape})
        # Global metadata audited on every frame, including scheme controls.
        missing_attrs = sorted(set(cpu.ncattrs()) - set(gpu.ncattrs()))
        different_attrs = []
        for name in cpu.ncattrs():
            if name in gpu.ncattrs() and not np.array_equal(cpu.getncattr(name), gpu.getncattr(name)):
                different_attrs.append({'name': name, 'cpu': str(cpu.getncattr(name)),
                                        'gpu': str(gpu.getncattr(name))})
        coverage.append({'lead_h': pair['lead_h'], 'cpu_field_count': len(cpu.variables),
                         'gpu_field_count': len(gpu.variables), 'missing_vs_cpu': missing,
                         'schema_mismatches': schema})
        headers.append({'lead_h': pair['lead_h'], 'cpu_global_count': len(cpu.ncattrs()),
                        'missing_vs_cpu': missing_attrs, 'differences': different_attrs})
        # Include GPU-only diagnostics; finite WRF sentinels are not NaN/Inf.
        gpu.set_auto_mask(False)
        for name, var in gpu.variables.items():
            if np.issubdtype(var.dtype, np.number):
                count = int(np.count_nonzero(~np.isfinite(var[:])))
                if count:
                    nonfinite.append({'lead_h': pair['lead_h'], 'field': name, 'count': count})

fields = {}
failures = []
incomplete = []
for name, row in report['field_summaries'].items():
    if name == 'Times':
        if not row['all_equal']:
            incomplete.append({'field': name, 'reason': 'Times mismatch'})
        continue
    leads = row.get('by_lead', [])
    for lead in leads:
        result = lead.get('tolerance_result', {})
        if result.get('pass') is False:
            failures.append({'field': name, 'lead_h': lead['lead_h'],
                             'checks': result.get('failures', []), 'rmse': lead.get('rmse'),
                             'max_abs': lead.get('max_abs')})
    if row.get('compared_lead_count') != 25 or row.get('missing_leads') or row.get('incompatible_leads'):
        incomplete.append({'field': name, 'reason': 'missing/incompatible frame',
                           'compared': row.get('compared_lead_count')})
    fields[name] = {'verdict': 'FAIL' if any(x['field'] == name for x in failures) else
                   ('PASS' if row.get('tolerance_result', {}).get('supplied') else 'REPORT_ONLY'),
                   'rmse': row.get('overall', {}).get('rmse'),
                   'max_abs': row.get('overall', {}).get('max_abs'),
                   'frame_count': row.get('compared_lead_count'),
                   'by_lead': [{k: lead.get(k) for k in ('lead_h', 'rmse', 'max_abs', 'tolerance_result')}
                               for lead in leads]}
failures.sort(key=lambda r: (r['lead_h'], r['field']))
coverage_ok = (pairing['common_leads_h'] == list(range(25)) and
               pairing['cpu_file_count'] == 25 and pairing['gpu_file_count'] == 25 and
               not pairing['unmatched_cpu_count'] and not pairing['unmatched_gpu_count'] and
               not incomplete and not report['inventory']['incompatible_common'] and
               all(not x['missing_vs_cpu'] and not x['schema_mismatches'] for x in coverage))
summary = {'schema': 'gpuqual-grid-gate-v1', 'config': case.name,
           'reference': 'original CPU-WRF, matched namelist; execution mode in original CPU receipt',
           'manifest': str(manifest), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
           'verdict': 'FAIL' if failures or nonfinite else ('PASS' if coverage_ok else 'INCOMPLETE'),
           'coverage_complete': coverage_ok, 'paired_frames': pairing['paired_file_count'],
           'cpu_field_count': report['inventory']['cpu_variable_count'],
           'bounded_field_count': sum(x['verdict'] != 'REPORT_ONLY' for x in fields.values()),
           'first_failure': failures[0] if failures else None, 'failures': failures,
           'incomplete': incomplete, 'by_frame': coverage, 'headers_by_frame': headers,
           'raw_gpu_nonfinite': nonfinite,
           'fields': fields, 'note': 'D6 bounded fields PASS only if every hourly frame passes; unbounded inventoried fields REPORT_ONLY'}
(case / 'grid_gate.json').write_text(json.dumps(summary, indent=1, allow_nan=False) + '\n')
print(json.dumps({k: summary[k] for k in ('config', 'verdict', 'coverage_complete',
                                        'paired_frames', 'cpu_field_count', 'first_failure')}, indent=2))
