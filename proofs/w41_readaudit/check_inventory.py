#!/usr/bin/env python3
"""Small independent edge/safety checks for the static audit, no payload data."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('build_inventory.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def check():
    c = audit.checks(['2026-08-10', '2026-08-22', '2026-11-16', '2026-12-13'], False)
    assert c['in_E32']['value'] is True and c['in_B1']['value'] is True
    c = audit.checks(['2026-08-09', '2026-08-23'], False)
    assert c['in_E32']['value'] is False
    assert audit.checks([], True)['in_B1']['value'] == 'UNKNOWN'
    p = audit.path_evidence('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260914_18z_a1/run/run', 'run_hours = 72; init_time_utc = "2026-09-15T00:00:00Z"')
    assert p['checks']['on_or_after_2026-09-15']['value'] == 'UNKNOWN'
    assert p['possible_horizons'][0]['days'] == ['2026-09-15', '2026-09-16', '2026-09-17', '2026-09-18']
    assert audit.path_evidence('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260914_18z_a1/run/run', 'run_hours=72')['possible_horizons'] == []
    assert audit.path_evidence('<DATA_ROOT>/alisios/runs/20260914_18z/wrfout_d01_2026-09-15_00:00:00', '')['checks']['on_or_after_2026-09-15']['value'] is True
    p = audit.path_evidence('<DATA_ROOT>/alisios/state/manager/runs/gpu_input_store_20261004/20260501_18z', '')
    assert p['forecast_or_valid_days'] == ['2026-05-01']
    assert p['unclassified_path_date_tokens'] == ['2026-10-04']
    assert p['checks']['on_or_after_2026-09-15']['value'] == 'UNKNOWN'
    assert audit.paths_in('P = ("<DATA_ROOT>/alisios/runs/20260725_18z/"\n "cpu_case")') == ['<DATA_ROOT>/alisios/runs/20260725_18z/cpu_case']
    stamps = audit.stamp_evidence(json.dumps({'valid_time_utc': '2026-04-09T00:00:00Z', 'audited_utc': '2026-10-14T13:51:44Z'}))
    assert stamps == [{'key': '$.audited_utc', 'value': '2026-10-14T13:51:44Z'}]
    # Parent and leaf symlinks must fail, even when the target is harmless.
    with tempfile.TemporaryDirectory(prefix='safety_', dir='<USER_HOME>/wrf_gpu2_lanes/w41-readaudit') as d:
        root = Path(d)
        (root / 'real').mkdir()
        (root / 'real' / 'a.json').write_text('{}')
        (root / 'link').symlink_to(root / 'real', target_is_directory=True)
        (root / 'leaf.json').symlink_to(root / 'real' / 'a.json')
        assert audit.safe_open(str(root / 'real' / 'a.json')) == '{}'
        for path in (root / 'link' / 'a.json', root / 'leaf.json'):
            try:
                audit.safe_open(str(path))
            except OSError:
                pass
            else:
                raise AssertionError('symlink was followed: ' + str(path))
    # Prefix rejection happens before any open of a disallowed root.
    try:
        audit.safe_open('<DATA_ROOT>/alisios/NEVER_OPEN')
    except ValueError:
        pass
    else:
        raise AssertionError('forbidden root admitted')
    print('PASS: inclusive ranges, UNKNOWN extents, horizon, adjacent literals, run timestamps, parent/leaf symlink refusal, root refusal')


if __name__ == '__main__':
    check()
