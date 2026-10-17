"""Run fixture-free original test assertions without competing pytest temp roots.

Owner permits concurrent CPU JAX processes on distinct cores. Only the mutant
uses pytest; these assertions consume no pytest fixtures or shared basetemp.
"""
from pathlib import Path
import json
import os
import runpy
import sys
import time
import traceback
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/v025'),
               str(ROOT / 'proofs/o1_nlbind')]
import cpu_guard
import jax
import cpu_safe

assert jax.devices()[0].platform == 'cpu'
cpu_safe.install()
mode, destination = sys.argv[1:]
out = Path(destination)
report = dict(mode=mode, source=str(ROOT), platform='cpu',
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              affinity=sorted(os.sched_getaffinity(0)), checks=[])
if mode == 'upstream':
    source = runpy.run_path(str(ROOT / 'tests/dynamics/test_sms3dtke_upstream.py'))
    calls = [('test_sms_upstream_pristine', (key,))
             for key in ('periodic_convective', 'zero_tke_stable')]
elif mode == 'v013':
    source = runpy.run_path(str(ROOT / 'tests/test_v013_operational_smoke.py'))
    calls = [('test_ra_lw_operational_runs_and_cools', (value,)) for value in (1, 3, 4, 31)]
    calls += [(name, ()) for name in ('test_sw_and_lw_are_selected_independently',
        'test_sf_surface_noahmp_operational_runs_and_advances_land',
        'test_sf_surface_noahclassic_operational_runs_and_advances_land',
        'test_sf_surface_slab_operational_runs_and_advances_land',
        'test_sf_surface_pleim_xiu_operational_runs_and_advances_land')]
else:
    raise ValueError(mode)
for name, args in calls:
    record = dict(test=name, parameters=args, status='RUNNING')
    report['checks'].append(record)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('START', name, args, flush=True)
    start = time.monotonic()
    try:
        source[name](*args)
        record['status'] = 'PASS'
    except BaseException as exc:
        record.update(status='INCOMPLETE' if isinstance(exc, KeyboardInterrupt) else 'FAIL',
                      traceback=traceback.format_exc())
        raise
    finally:
        record['seconds'] = time.monotonic() - start
        out.write_text(json.dumps(report, indent=2) + '\n')
        print(record['status'], name, args, record['seconds'], flush=True)
