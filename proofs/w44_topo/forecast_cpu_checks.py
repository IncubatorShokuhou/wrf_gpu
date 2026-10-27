"""CPU-only launcher controls; fabricated IO fixtures are NOT forecast evidence."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LANE = Path('<USER_HOME>/wrf_gpu2_lanes/w44-topo')


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    reader, harness, driver = (module(name) for name in ('forecast_compare', 'forecast_harness', 'forecast_driver'))
    import jax
    import numpy as np
    from netCDF4 import Dataset
    from gpuwrf.integration import nested_pipeline as pipeline
    assert jax.devices()[0].platform == 'cpu'
    report = dict(status='PASS', scope='CPU launcher/IO controls only; NOT numerical forecast evidence', GPU_started=False)
    with tempfile.TemporaryDirectory(dir=LANE) as temporary:
        root = Path(temporary)
        original = pipeline._PerDomainWrfoutWriter.__call__
        calls = []
        pipeline._PerDomainWrfoutWriter.__call__ = lambda *args: calls.append(args) or {'ok': True}
        try:
            harness.install_output_observer(root / 'observer')
            writer = SimpleNamespace(run_start=datetime(2026, 2, 28), dt_by_domain={'d01': 54})
            rad = SimpleNamespace(swnorm=np.array([10, 20], np.float32), swup_topographic=np.array([3, 4], np.float32))
            carry = SimpleNamespace(radiation_diagnostics=rad)
            observed = pipeline._PerDomainWrfoutWriter.__call__(writer, 'd01', 67, carry)
            assert observed == {'ok': True} and len(calls) == 1
            with np.load(root / 'observer/d01_2026-02-28_01:00:18.npz') as captured:
                np.testing.assert_array_equal(captured['GSW'], np.array([7, 16], np.float32))
            for field in ('swnorm', 'swup_topographic'):
                before = getattr(rad, field).copy()
                for value in (np.nan, np.inf, -np.inf):
                    getattr(rad, field)[0] = value
                    try:
                        pipeline._PerDomainWrfoutWriter.__call__(writer, 'd01', 68, carry)
                    except ValueError as exc:
                        assert 'nonfinite' in str(exc)
                    else:
                        raise AssertionError('nonfinite observer input survived')
                setattr(rad, field, before)
            assert len(calls) == 1, 'observer must refuse nonfinite before real writer'
        finally:
            pipeline._PerDomainWrfoutWriter.__call__ = original
        report['real_callback_observer'] = 'PASS: timestamp/held GSW/original callback and six finite controls'
        reader.selftest(root / 'reader')
        report['reader_ulp_and_nonfinite_controls'] = 'PASS'
        for side in ('old', 'new'):
            folder = root / 'full' / side
            (folder / 'wrfout').mkdir(parents=True)
            (folder / 'held_surface').mkdir()
            for domain in ('d01', 'd02', 'd03'):
                for hour in range(12):
                    stamp = f'2026-02-28_{hour:02d}:00:00'
                    with Dataset(folder / f'wrfout/wrfout_{domain}_{stamp}', 'w') as ds:
                        ds.createDimension('n', 2)
                        ds.createVariable('SWDOWN', 'f4', ('n',))[:] = 0 if hour < 8 else 100
                        ds.createVariable('T2', 'f4', ('n',))[:] = 280 + int(side == 'new' and hour >= 8)
                    np.savez(folder / f'held_surface/{domain}_{stamp}.npz', GSW=np.array([0, 0] if hour < 8 else [80, 80], np.float32))
            (folder / 'receipt.json').write_text(json.dumps({'rc': 0, 'device': {'platform': 'gpu'}}))
            (folder / 'cli_stdout.json').write_text(json.dumps({'all_domains_finite': True, 'all_outputs_present': True}))
        result = reader.compare(root / 'full/old', root / 'full/new')
        assert result['frames'] == 12 and result['finite_checked_frames'] == 36
        assert all(r['T2']['mean'] == 1 and r['T2']['rmse'] == 1 for r in result['aggregates'].values())
        missing = root / 'full/new/held_surface/d03_2026-02-28_11:00:00.npz'
        missing.unlink()
        try:
            reader.compare(root / 'full/old', root / 'full/new')
        except ValueError as exc:
            assert 'GSW' in str(exc)
        else:
            raise AssertionError('missing daylight GSW survived')
        report['daylight_coverage'] = 'PASS: 36 checked/12 daylight, expected mean/RMSE, missing capture rejected'
        report['private_flock_process_group'] = driver.load_group().selftest(root)
        assert report['private_flock_process_group']['rc'] == 124
        now = datetime.now(timezone.utc)
        windows = dict(future=(now + timedelta(minutes=1), now + timedelta(hours=1), 5),
            short=(now - timedelta(minutes=1), now + timedelta(minutes=39), 5),
            expired=(now - timedelta(hours=1), now - timedelta(minutes=1), 5),
            valid=(now - timedelta(minutes=1), now + timedelta(minutes=41), 0))
        for name, (start, end, expected) in windows.items():
            env = {**os.environ, 'GA_START_UTC': start.isoformat(), 'GA_END_UTC': end.isoformat()}
            rc = subprocess.run([sys.executable, str(HERE / 'forecast_driver.py'), '--check-window'], env=env,
                                stdin=subprocess.DEVNULL, timeout=10).returncode
            assert rc == expected, (name, rc)
        report['four_UTC_windows'] = 'PASS'
    (LANE / 'TP30_CPU.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
