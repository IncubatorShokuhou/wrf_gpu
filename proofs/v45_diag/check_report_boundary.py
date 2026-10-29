"""F1 finite-reader deletion control and scoped CPU cost, no model execution."""
import ast
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc

sys.path.insert(0, 'scripts/v025')
import cpu_guard
from gpuwrf.optimize import diagnostics as D

lane = Path('<USER_HOME>/wrf_gpu2_lanes/v45-diag')
if sys.argv[1] == 'mutation':
    import pytest
    # Exactly the effect of deleting the finite check after float conversion.
    D._finite_report_number = float
    raise SystemExit(pytest.main(['-q', 'tests/test_v45_diag_profiles.py::test_wave_entry_rejects_one_nonfinite_per_numeric_leaf',
                                 '--basetemp', str(lane/'pytest_mutation'), '--junitxml='+str(lane/'F1_mutation.xml')]))

paths = sorted((lane/'pytest').rglob('profiles.json'))
valid = next(p for p in paths if 'test_known_answer_profiles' in str(p))
report = json.loads(valid.read_bytes())
old_source = subprocess.check_output(['git', '-C', '<USER_HOME>/src/wrf_gpu2_wt/v45-diag', 'show',
                                      'ce959f691c:src/gpuwrf/optimize/diagnostics.py'], text=True)
tree = ast.parse(old_source)
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'wave_report_entry')
scope = dict(D.__dict__)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<ce959f691c-wave-report-entry>', 'exec'), scope)
before = scope['wave_report_entry']
assert before(valid) == D.wave_report_entry(valid)
controls = []
bad_path = lane/'F1_bad_report.json'
for name, value in (('nan', float('nan')), ('inf', float('inf')), ('neginf', -float('inf'))):
    original = report['rows'][0]['metrics']['tke_budget_residual']['max_abs']
    report['rows'][0]['metrics']['tke_budget_residual']['max_abs'] = value
    bad_path.write_text(json.dumps(report, allow_nan=True))
    unsafe = before(bad_path)
    assert unsafe['status'].startswith('COMPLETE')
    try:
        D.wave_report_entry(bad_path)
    except ValueError as exc:
        assert 'nonfinite' in str(exc)
    else:
        raise AssertionError('F1 still exports a complete report')
    controls.append(dict(fault=name, old_status=unsafe['status'], old_max_is_finite=math.isfinite(unsafe['max_abs_tke_residual_m2_s3']),
                         new_result='ValueError before reduction/export'))
    report['rows'][0]['metrics']['tke_budget_residual']['max_abs'] = original
    bad_path.unlink()

timings = {'old': [], 'fixed': []}
for n in range(31):
    for label, fn in ([('old', before), ('fixed', D.wave_report_entry)] if n % 2 else [('fixed', D.wave_report_entry), ('old', before)]):
        start = time.perf_counter()
        fn(valid)
        timings[label].append(time.perf_counter()-start)
peaks = {}
tracemalloc.start()
for label, fn in [('old', before), ('fixed', D.wave_report_entry)]:
    gc.collect()
    current = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    fn(valid)
    peaks[label] = tracemalloc.get_traced_memory()[1]-current
tracemalloc.stop()

def count_numbers(value):
    if isinstance(value, dict): return sum(map(count_numbers, value.values()))
    if isinstance(value, list): return sum(map(count_numbers, value))
    return int(type(value) in (int, float))

row = dict(candidate='144fba6310', baseline='ce959f691c', helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           scope='Offline wave-entry read/export of WRF-shaped 4-row/4-level fixture; shared CPU12/28 nice19, no model/GPU, not forecast performance',
           numeric_leaves=count_numbers(report), one_per_leaf_faults=3*count_numbers(report), exponent_overflow_controls=2,
           baseline_f1_controls=controls, valid_entry_equal=True,
           json_bytes=valid.stat().st_size, paired_reads=31, median_wall_s={k:statistics.median(v) for k,v in timings.items()},
           python_incremental_peak_bytes=peaks, host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           forecast_work='unchanged; reader is offline and outside AOT closure', cache='NONE')
(lane/'F1_probe.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
print(json.dumps(row, allow_nan=False))
