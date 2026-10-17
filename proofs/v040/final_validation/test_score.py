"""Deletion-sensitive coverage and NaN/Inf controls for the final reader."""
import importlib.util
from pathlib import Path
import sys
from datetime import datetime, timedelta
import numpy as np
import pytest
from gpuwrf.io.netcdf_lock import Dataset

spec = importlib.util.spec_from_file_location('vprep_score', Path(__file__).with_name('score.py'))
score = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = score
spec.loader.exec_module(score)


def make(path, values, extra=True):
    with Dataset(path, 'w') as d:
        d.createDimension('x', 2)
        d.createVariable('T2', 'f4', ('x',))[:] = values
        if extra:
            d.createVariable('EXTRA_CPU_FIELD', 'f4', ('x',))[:] = [3, 4]
        d.TITLE = 'original WRF control'


def test_inventory_uses_every_cpu_field(tmp_path):
    a, b = tmp_path / 'cpu.nc', tmp_path / 'gpu.nc'
    make(a, [280, 281])
    make(b, [280, 281], extra=False)
    row = score.audit_pair(a, b)
    assert row['missing_vs_cpu'] == ['EXTRA_CPU_FIELD']


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('arm', ['cpu', 'gpu'])
def test_both_arms_reject_each_nonfinite(tmp_path, value, arm):
    a, b = tmp_path / 'cpu.nc', tmp_path / 'gpu.nc'
    make(a, [280, value] if arm == 'cpu' else [280, 281])
    make(b, [280, value] if arm == 'gpu' else [280, 281])
    assert score.audit_pair(a, b)['nonfinite'] == [dict(arm=arm, field='T2', count=1)]


def test_later_frame_header_is_read(tmp_path):
    a, b = tmp_path / 'cpu.nc', tmp_path / 'gpu.nc'
    make(a, [280, 281])
    make(b, [280, 281])
    with Dataset(b, 'a') as d:
        d.TITLE = 'wrong later frame'
    assert score.audit_pair(a, b)['header_differences'][0]['name'] == 'TITLE'


@pytest.mark.parametrize('dfi', [0, 1])
@pytest.mark.parametrize('omit_zero_clock_fields', [False,True])
def test_reference_alarm_clock_and_missing_hour(dfi,omit_zero_clock_fields):
    start = datetime(2023, 1, 15)
    tc = dict(zip(('start_year','start_month','start_day','start_hour','start_minute','start_second'),
                  (2023,1,15,0,0,0)))
    if omit_zero_clock_fields:
        tc.pop('start_minute');tc.pop('start_second')
    nl = dict(time_control=tc, domains=dict(time_step=54))
    if dfi:
        nl['dfi_control'] = dict(dfi_opt=1, dfi_fwdstop_year=2023, dfi_fwdstop_month=1,
            dfi_fwdstop_day=15, dfi_fwdstop_hour=0, dfi_fwdstop_minute=18, dfi_fwdstop_second=0)
    initial=start+timedelta(minutes=9 if dfi else 0)
    count=24 if dfi else 25
    stamps = [initial] + [initial + timedelta(seconds=((h*3600+53)//54)*54) for h in range(1,count)]
    assert score.reference_schedule(nl, stamps, 24)[1] == list(range(count))
    stamps[8] += timedelta(hours=1)
    with pytest.raises(AssertionError, match='alarm coverage'):
        score.reference_schedule(nl, stamps, 24)
