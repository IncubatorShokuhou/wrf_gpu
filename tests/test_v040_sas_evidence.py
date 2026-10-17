"""Single-field adverse controls for the actual shared SAS evidence gates."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/'proofs/v040/sas_family'
sys.path.insert(0,str(ROOT))
from evidence_gate import field_score, promoted_fields, require_finite
from probe_ksas14 import compare_fields

NATIVE=('HBOT','HTOP','RAINCV','PRATEC','RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN','RUCUTEN','RVCUTEN')
KF=('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RAINCV','PRATEC','NCA','TIMEC')


def inputs(names):
    d={name:np.ones((2,3) if name.endswith('TEN') else (2,),np.float64) for name in names}
    d.update({name:np.ones((2,3)) for name in ('T','QV','QC','QI','U','V','PI')})
    d.update(DT=np.full(2,18.),STEPCU=np.ones(2))
    return d


@pytest.mark.parametrize('mode',('r4','r8'))
@pytest.mark.parametrize('field',NATIVE)
@pytest.mark.parametrize('side',('candidate','reference','both'))
@pytest.mark.parametrize('bad',(np.nan,np.inf,-np.inf))
def test_native_single_bad_field_never_passes_and_is_counted(mode,field,side,bad):
    ref=inputs(NATIVE);actual={k:ref[k].copy() for k in NATIVE}
    for values in ([actual] if side=='candidate' else [ref] if side=='reference' else [actual,ref]):
        values[field].flat[0]=bad
    result=compare_fields(actual,ref,mode)[field]
    assert not result['pass'] and result['failed_values']>0


@pytest.mark.parametrize('field',KF)
@pytest.mark.parametrize('side',('candidate','reference','both'))
@pytest.mark.parametrize('bad',(np.nan,np.inf,-np.inf))
def test_old_kf_primary_scorer_rejects_every_bad_field(field,side,bad):
    ref={k:v for k,v in inputs(KF).items() if k in KF};actual={k:v.copy() for k,v in ref.items()}
    for values in ([actual] if side=='candidate' else [ref] if side=='reference' else [actual,ref]):
        values[field].flat[0]=bad
    result=promoted_fields(actual,ref,exact=('NCA','TIMEC'))[field]
    assert not result['pass'] and result['failed_values']>0


@pytest.mark.parametrize('field',KF)
@pytest.mark.parametrize('side',('candidate','reference','both'))
@pytest.mark.parametrize('bad',(np.nan,np.inf,-np.inf))
def test_actual_old_kf_primary_body_rejects_substituted_nonfinite(monkeypatch,field,side,bad):
    import test_v040_kf99 as primary
    with np.load(ROOT/'fixtures/kf99_r8.npz') as raw:
        reference={k:v.copy() for k,v in raw.items()}
    actual={k:reference[k].copy() for k in KF}
    actual.update(SOURCE_FATAL=np.zeros(5,bool),BORROW_EVENTS=np.zeros(5,np.int32))
    for values in ([actual] if side=='candidate' else [reference] if side=='reference' else [actual,reference]):
        values[field].flat[0]=bad
    class Fixture:
        def __enter__(self):return reference
        def __exit__(self,*args):pass
    monkeypatch.setattr(primary.np,'load',lambda *a,**k:Fixture())
    # Substitute outputs at the compiled-function boundary: run the real
    # primary assertion body without paying for another numerical model run.
    monkeypatch.setattr(primary.jax,'jit',lambda fn:lambda *args:actual)
    with pytest.raises((AssertionError,ValueError)):
        primary.test_full_promoted_pristine_old_kfcps_ten_outputs()


@pytest.mark.parametrize('bad',(np.nan,np.inf,-np.inf))
def test_auxiliary_operands_are_rejected_before_bound_evaluation(bad):
    def forbidden():raise AssertionError('bound evaluated on corrupt operands')
    result=field_score(np.ones(3),np.ones(3),forbidden,operands=(np.array([bad]),))
    assert not result['pass'] and result['failed_values']==3
    with pytest.raises(ValueError,match='failed_values=1'):
        require_finite(input=np.array([bad]))


@pytest.mark.parametrize('bad',(np.nan,np.inf,-np.inf,-1.))
def test_nonfinite_or_negative_derived_bounds_fail_with_counts(bad):
    result=field_score(np.ones(3),np.ones(3),lambda:bad)
    assert not result['pass'] and result['failed_values']==3


def test_finite_overflowing_derived_difference_fails():
    result=field_score(np.array([1.e308]),np.array([-1.e308]),lambda:1.)
    assert not result['pass'] and result['failed_values']==1


@pytest.mark.parametrize('mode',('r4','r8'))
def test_healthy_native_scores_keep_frozen_arithmetic(mode):
    ref=inputs(NATIVE);actual={k:ref[k].copy() for k in NATIVE}
    actual['RQVCUTEN'][0,0]+=1.e-5
    scores=compare_fields(actual,ref,mode)
    assert scores['HBOT']['pass'] and scores['RAINCV']['pass']
    assert not scores['RQVCUTEN']['pass'] and scores['RQVCUTEN']['failed_values']==1
