"""BD94: actual WRF REAL constant/column bits; OFF and fp64 sources unchanged."""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gpuwrf.io import wrfout_writer as output
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.netcdf_lock import Dataset

FIXTURE = Path(__file__).parent / 'fixtures' / 'writer_rvovrd_real.npz'
PROVENANCE = FIXTURE.with_suffix('.json')
META = json.loads(PROVENANCE.read_text())


def _validate_oracle(data):
    for name, value in data.items():
        array = np.asarray(value)
        if array.dtype.kind == 'f':
            assert np.isfinite(array).all(), f'non-finite oracle field {name}'


@pytest.fixture(scope='module')
def oracle():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == META['fixture_sha256']
    with np.load(FIXTURE, allow_pickle=False) as data:
        result = {name: data[name].copy() for name in data.files}
    _validate_oracle(result)
    return result


def bits(array):
    source = np.asarray(array)
    assert np.isfinite(source).all(), 'non-finite bit operand'
    real = np.asarray(source, dtype=np.float32)
    assert np.isfinite(real).all(), 'non-finite REAL bit operand'
    return real.view(np.uint32)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('where', ['actual', 'reference', 'both'])
def test_bit_gate_rejects_non_finite(bad, where):
    actual = np.ones(3, dtype=np.float32)
    reference = actual.copy()
    if where in ('actual', 'both'):
        actual[0] = bad
    if where in ('reference', 'both'):
        reference[0] = bad
    with pytest.raises(AssertionError, match='non-finite'):
        np.testing.assert_array_equal(bits(actual), bits(reference))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('field', ['theta', 'qv', 'dry', 'T'])
def test_oracle_gate_rejects_non_finite(oracle, field, bad):
    poisoned = {name: value.copy() for name, value in oracle.items()}
    poisoned[field].flat[0] = bad
    with pytest.raises(AssertionError, match='non-finite oracle'):
        _validate_oracle(poisoned)


def test_pristine_ratio_and_sensitive_real_operands(oracle):
    assert bits(output.RVOVRD_REAL).item() == META['oracle_ratio_bits'] == 1070456530
    assert bits(output.RVOVRD).item() == META['legacy_ratio_bits'] == 1070456529
    old = oracle['theta'] / (1.0 + (461.6 / 287.0) * oracle['qv'])
    assert np.count_nonzero(bits(old) != bits(oracle['dry'])) == META['sensitive_selected_cells'] > 0


@pytest.mark.parametrize('flag', [False, True], ids=['OFF','ON'])
@pytest.mark.parametrize('theta_dtype,qv_dtype', [
    (np.float32,np.float32), (np.float64,np.float64), (np.float32,np.float64),
])
def test_real_order_and_legacy_source_bits(oracle, monkeypatch, flag, theta_dtype, qv_dtype):
    monkeypatch.setattr(output, '_WRITER_RVOVRD_REAL', flag)
    state = {'theta': oracle['theta'].astype(theta_dtype), 'qv': oracle['qv'].astype(qv_dtype)}
    # Exercise the writer's actual host coercion before its source-dtype decision.
    theta = output._field_array(state, ('theta','THETA'), oracle['theta'].shape)
    qv = output._field_array(state, ('QVAPOR','qv','qvapor'), oracle['qv'].shape)
    actual = output._theta_dry_for_output(state, theta, qv)
    expected = oracle['dry'] if flag and theta_dtype == qv_dtype == np.float32 else theta / (1.0 + (461.6/287.0)*qv)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_array_equal(bits(actual), bits(expected))


def test_none_alias_does_not_hide_real_source(oracle, monkeypatch):
    monkeypatch.setattr(output, '_WRITER_RVOVRD_REAL', True)
    state = {'theta':None, 'THETA':oracle['theta'], 'QVAPOR':None, 'qv':oracle['qv']}
    np.testing.assert_array_equal(bits(output._theta_dry_for_output(state, oracle['theta'], oracle['qv'])),bits(oracle['dry']))


@pytest.mark.parametrize('setting,expected', [(None,False), ('0',False), ('1',True)])
def test_fresh_import_default_and_explicit_flag(setting, expected):
    env = dict(os.environ, JAX_PLATFORMS='cpu', GPUWRF_FAST_DEFAULTS='0', GPUWRF_JAX_CACHE='0',
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    env.pop('GPUWRF_WRITER_RVOVRD_REAL',None)
    if setting is not None:
        env['GPUWRF_WRITER_RVOVRD_REAL'] = setting
    result = subprocess.run([sys.executable,'-c','from gpuwrf.io.wrfout_writer import _WRITER_RVOVRD_REAL; print(int(_WRITER_RVOVRD_REAL))'],env=env,text=True,capture_output=True,check=True,timeout=90)
    assert int(result.stdout.strip()) == expected


def real_payload_case(row, tmp_path, subset):
    path = Path(row['file'])
    if not path.exists():
        pytest.skip('original WN3/PROD CPU-WRF operands are not mounted')
    names = ('THM','QVAPOR','U','V','W','P','PB','PH','PHB','MU','MUB','T2','TSK','PSFC','Q2','U10','V10','HFX','LH','UST')
    with Dataset(path) as ds:
        raw = {n:np.asarray(ds[n][0]).copy() for n in names}
    state = dict(raw,theta=raw['THM']+np.float32(300),qv=raw['QVAPOR'],p_total=raw['P']+raw['PB'],ph_total=raw['PH']+raw['PHB'],mu_total=raw['MU']+raw['MUB'])
    diagnostics = {n:raw[n] for n in ('T2','TSK','PSFC','Q2','U10','V10','HFX','LH','UST')}
    source = Gen2Run(path.parent).grid(row['domain'])
    grid = source.as_grid_spec()
    stamp = datetime.strptime(path.name[11:],'%Y-%m-%d_%H:%M:%S')
    kwargs = dict(domain=row['domain'],domain_authority=output.bind_wrfout_domain_authority(row['domain'],source,grid),run_start=stamp,valid_time=stamp,lead_hours=0,diagnostics=diagnostics,variable_subset=('T','THM','QVAPOR') if subset else None,full_variable_set=not subset)
    return raw,state,grid,kwargs


@pytest.mark.parametrize('row', META['rows'], ids=lambda r:Path(r['file']).name)
@pytest.mark.parametrize('subset', [False,True], ids=['full','subset'])
def test_actual_history_paths_equal_pristine_real_columns(row, subset, oracle, monkeypatch, tmp_path):
    monkeypatch.setattr(output, '_WRITER_RVOVRD_REAL', True)
    raw,state,grid,kwargs = real_payload_case(row,tmp_path,subset)
    before = {n:a.tobytes() for n,a in state.items()}
    prepared = output.prepare_wrfout_payload(state,grid,SimpleNamespace(use_theta_m=1),tmp_path/'history.nc',**kwargs)
    selected = np.concatenate([prepared.fields['T'][:,y,x] for y,x in row['columns_yx']])
    start, count = row['offset'], row['count']
    expected = oracle['T'][start:start+count]
    assert np.isfinite(selected).all() and np.isfinite(expected).all()
    np.testing.assert_array_equal(bits(selected),bits(expected))
    assert before == {n:a.tobytes() for n,a in state.items()}
    np.testing.assert_array_equal(bits(prepared.fields['THM']),bits(state['theta']-np.float32(300)))
    np.testing.assert_array_equal(bits(prepared.fields['QVAPOR']),bits(raw['QVAPOR']))


@pytest.mark.parametrize('subset', [False,True], ids=['full','subset'])
@pytest.mark.parametrize('mixed', [False,True], ids=['fp64','mixed'])
def test_flag_preserves_entire_legacy_source_payload(subset, mixed, monkeypatch, tmp_path):
    _,state,grid,kwargs = real_payload_case(META['rows'][0],tmp_path,subset)
    state['theta'] = state['theta'].astype(np.float32 if mixed else np.float64)
    state['QVAPOR'] = state['qv'] = state['qv'].astype(np.float64)
    snapshots = []
    before = {n:a.tobytes() for n,a in state.items()}
    for flag in (False,True):
        monkeypatch.setattr(output,'_WRITER_RVOVRD_REAL',flag)
        payload = output.prepare_wrfout_payload(state,grid,SimpleNamespace(use_theta_m=1),tmp_path/'legacy.nc',**kwargs)
        assert all(np.isfinite(a).all() for a in payload.fields.values())
        snapshots.append({n:(a.dtype.str,a.shape,a.tobytes()) for n,a in payload.fields.items()})
    assert snapshots[0] == snapshots[1]
    assert before == {n:a.tobytes() for n,a in state.items()}
