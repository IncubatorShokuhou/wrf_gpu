#!/usr/bin/env python3
"""Independent syntax fixtures and critic regressions, never execute harnesses."""
import hashlib
import json
from pathlib import Path
import sys

from build_inventory import entry, enrich, git, safe_open


def fixture(text):
    return enrich(entry('FIXTURE', 'fixture.py', text, ['<DATA_ROOT>/alisios/']), text)


def check():
    e = fixture('''from pathlib import Path
BASE = Path('<DATA_ROOT>/alisios/runs/20260502_18z')
T_A = '2026-05-04_12:00:00'
WRONG = '2026-11-16'
frame = BASE / f'wrfout_d01_{T_A}'
''')
    assert e['days'] == ['2026-05-02', '2026-05-04']
    assert e['checks']['in_B1']['value'] == 'UNKNOWN'
    e = fixture('''from pathlib import Path
BASE = Path('<DATA_ROOT>/alisios/runs/20260502_18z')
for d in (1, 2):
    f = BASE / f'wrfout_d{d:02d}_2026-05-03_00:00:00'
''')
    assert e['days'] == ['2026-05-02', '2026-05-03']
    assert any('/wrfout_d01_' in p for p in e['derived_paths'])
    e = fixture('''from pathlib import Path
ROOT = Path('<DATA_ROOT>/alisios/wrf_gen')
CASES = ('20260227_18z_a1', '20260614_18z_a1')
def reader(case):
    source = ROOT / f'wg_{case}' / 'run/run'
def main():
    result = [reader(case) for case in CASES]
''')
    assert e['days'] == ['2026-02-27', '2026-06-14']
    e = fixture('''from pathlib import Path
RAW = Path('<DATA_ROOT>/alisios/state/manager/runs/administrative_20261008/sonde_before20260701.txt')
VALID = ['20260228']
with guard.guarded_session({str(RAW): VALID}, ['20260227'], OUT):
    pass
''')
    assert e['days'] == ['2026-02-28']
    assert e['checks']['on_or_after_2026-09-15']['value'] == 'UNKNOWN'
    obj = {'measurement': {'source': '<DATA_ROOT>/alisios/runs/20260725_18z/rsl.out.0000',
                          'window': {'start_time': '2026-07-26_00:00:00', 'end_time': '2026-08-01_18:00:00'}},
           'unrelated': {'start_time': '2026-11-16_00:00:00', 'end_time': '2026-11-18_00:00:00'},
           'start_utc': '2026-10-08T01:00:00Z', 'end_utc': '2026-10-08T01:01:00Z'}
    e = enrich(entry('FIXTURE', 'fixture.json', json.dumps(obj), [obj['measurement']['source']]), json.dumps(obj))
    assert e['days'] == ['2026-07-25', '2026-07-26', '2026-07-27', '2026-07-28', '2026-07-29', '2026-07-30', '2026-07-31', '2026-08-01']
    assert all(v['value'] is False for v in e['checks'].values())
    assert len(e['run_timestamp_evidence']) == 2
    print('PASS: joins/fstrings, comprehension/call parameters, guard bindings, same-object windows, administrative/synthetic isolation, UTC execution metadata')

    if len(sys.argv) > 1:
        data = json.loads(safe_open(sys.argv[1], max_bytes=64 * 1024 * 1024))
        review = json.loads(safe_open('<USER_HOME>/wrf_gpu2_lanes/cr-w41-readaudit/CR02_CONTEXT_FINDINGS.json'))
        for c in review['date_corrections']:
            e = next(e for e in data[1:] if e['source'] == c['source'] and
                     (not c.get('historical_literal') or e['path_evidence'][0]['path'] == c['historical_literal']))
            assert set(c['missing_evidenced_days']) <= set(e['days']), (c['source'], e['days'])
            assert e['context_source_sha256'] == c['source_sha256'], c['source']
            assert e['context_date_evidence'], c['source']
        for c in review['missing_run_timestamps']:
            e = next(e for e in data[1:] if e['source'] == c['source'])
            assert e['context_source_sha256'] == c['source_sha256']
            assert {c['start_utc'], c['end_utc']} <= {x['value'] for x in e['run_timestamp_evidence']}
        cloud = next(e for e in data[1:] if e['source'] == 'tests/a1_da/test_obs.py')
        assert all(v['value'] is False for v in cloud['checks'].values())
        assert all(v['value'] != True for e in data[1:] for v in e['checks'].values())
        print('PASS: all22 critic date cases, exact21 source hashes, all6 completed plot start/end receipts; cloud single-day FALSE; no administrative positive range')


if __name__ == '__main__':
    check()
