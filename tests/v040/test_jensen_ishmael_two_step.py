"""Two pristine public steps activate native evolved-habit aggregation."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'proofs/s2_mpmore/jensen55'
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,DATA/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
libm=module('two_step_libm','libm_numpy.py')


@pytest.fixture(scope='module')
def backend(tmp_path_factory):return libm.aggregation_backend(tmp_path_factory.mktemp('original_libmvec'))


@pytest.mark.parametrize('mode',['fp32','fp64'])
@pytest.mark.parametrize('case',range(1,18))
def test_two_original_public_steps_literal_BYTE(mode,case,backend):
    from gpuwrf.physics.jensen_ishmael.column import mp_jensen_ishmael
    from gpuwrf.physics.jensen_ishmael.tables import get_tables
    folder=DATA/'two_step';manifest=json.loads((folder/'manifest.json').read_text())
    row=next(r for r in manifest['rows'] if (r['mode'],r['case'])==(mode,case));p=folder/f'{mode}_{case}.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and row['first_step_BYTE'] and row['finite']
    original=json.loads(p.read_text());R=np.float32 if mode=='fp32' else np.float64
    fields={n:np.asarray(a,R) for n,a in original['input'].items()};data=get_tables(R)
    with np.errstate(all='ignore'):
        first,precip=mp_jensen_ishmael(fields,data,R(original['dt']),original['initial_it'],xp=backend)
        final,accum=mp_jensen_ishmael(first,data,R(original['dt']),original['final_it'],rain=precip['rain'],snow=precip['snow'],xp=backend)
    for actual,expected in ((first,original['first_output']),(precip,original['first_precipitation']),
                            (final,original['output']),(accum,original['precipitation'])):
        assert set(actual)==set(expected)
        for n,a in actual.items():
            ref=np.asarray(expected[n],R);a=np.asarray(a)
            assert np.isfinite(a).all() and a.dtype==ref.dtype
            assert a.tobytes()==ref.tobytes(),(n,np.flatnonzero(a!=ref)[:8])


def test_actual_evolved_REAL4_habits_activate_aggregation_mass_transfer():
    with np.load(DATA/'two_step/fp32_aggregation.npz') as data:steps=data['steps'];rates=data['values']
    assert np.count_nonzero(np.any(rates[steps==1,:3]!=0,axis=1))==0
    assert np.count_nonzero(np.any(rates[steps==2,:3]!=0,axis=1))==245
