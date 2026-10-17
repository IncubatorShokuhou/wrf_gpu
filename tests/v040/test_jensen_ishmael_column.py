"""Unchanged pristine public WRF caller, literal complete composed scheme."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'proofs/s2_mpmore/jensen55'
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,DATA/file)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
libm=module('column_libm','libm_numpy.py');reference=module('column_tables','reference_tables.py')


@pytest.fixture(scope='module')
def backend(tmp_path_factory):return libm.aggregation_backend(tmp_path_factory.mktemp('original_libmvec'))


@pytest.mark.parametrize('mode',['fp32','fp64'])
@pytest.mark.parametrize('case',range(1,18))
def test_complete_original_public_caller_literal_BYTE(mode,case,backend):
    from gpuwrf.physics.jensen_ishmael.column import mp_jensen_ishmael
    folder=DATA/'fixtures';manifest=json.loads((folder/'manifest.json').read_text())
    record=next(r for r in manifest['rows'] if (r['mode'],r['case'])==(mode,case))
    path=folder/f'{mode}_{case}.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'] and record['finite']
    original=json.loads(path.read_text());R=np.float32 if mode=='fp32' else np.float64
    inputs={k:np.asarray(v,R) for k,v in original['input'].items()}
    with np.errstate(all='ignore'):
        got,precip=mp_jensen_ishmael(inputs,reference.load(ROOT,mode),R(original['dt']),original['it'],xp=backend)
    failures=[]
    for group,actual in [('output',got),('precipitation',precip)]:
        assert set(actual)==set(original[group]),(group,set(actual),set(original[group]))
        for name,value in actual.items():
            expected=np.asarray(original[group][name],R);observed=np.asarray(value)
            assert np.isfinite(observed).all() and observed.dtype==expected.dtype,(name,observed.dtype,expected.dtype)
            if observed.tobytes()!=expected.tobytes():
                bad=np.flatnonzero(observed!=expected);failures.append((name,len(bad),bad[:8]))
    assert not failures,(mode,case,failures)
