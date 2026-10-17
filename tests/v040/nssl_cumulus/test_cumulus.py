"""Literal pristine-WRF calcnfromcuten caller and full-driver qualification.

Tolerances were frozen in the manifest before building/running the candidate.
The original module is compiled unchanged; print-only pre/post savepoints are
admitted only after the final-output BYTE gate against that original module.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.nssl2mom import cleanup, driver
from gpuwrf.physics.nssl2mom.constants import get_constants
from gpuwrf.physics.nssl2mom.indices import FP32, FP64, LNC, LNI, LNR, LNS, NA

jax.config.update('jax_enable_x64', True)
FIX = Path(__file__).resolve().parents[3] / 'proofs/v040/nssl_cumulus/fixtures'
MODES = {'fp32':FP32, 'fp64':FP64}
NAMES = dict(th='TH',qv='QV',qc='QC',qr='QR',qi='QI',qs='QS',qg='QH',qh='QHL',
             qndrop='CCW',qnr='CRW',qni='CCI',qns='CSW',qng='CHW',qnh='CHL',qnn='CN',qvolg='VHW',qvolh='VHL')


def load(mode, case):
    return json.loads((FIX / f'{mode}_{case}.json').read_text())


def fields(sp, mode):
    c, R = sp['columns'], MODES[mode].nR
    out = {k:np.asarray(c[v+'_IN'],R) for k,v in NAMES.items()}
    out.update({k:np.asarray(c[v],R) for k,v in dict(pii='PII',p='P',rho='DN',dz='DZ',w='W').items()})
    cu = {q:np.asarray(c[q.upper()+'CUTEN'],R) for q in ('qc','qr','qi','qs')}
    return out,cu


def stack(sp, stage, mode):
    return np.concatenate([np.zeros((1,40)),np.asarray(sp['stages'][stage])]).astype(MODES[mode].nR)


def test_fixture_integrity():
    m = json.loads((FIX/'manifest.json').read_text())
    for name, sha in m['fixtures'].items():
        data = (FIX/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha
        assert json.loads(data)['E26'] == 'BYTE'
        assert json.loads(data)['source_sha256'] == m['source_sha256']


@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('case',range(18,24))
def test_cumulus_number_stage(mode,case):
    sp = load(mode,case)
    R = MODES[mode].nR
    _f,cu = fields(sp,mode)
    before = stack(sp,'PRECU',mode)
    mass = {q:R(sp['scalars']['DT'])*v for q,v in cu.items()}
    fn = jax.jit(lambda an,dn:cleanup.calcnfromcuten(an,dn,mass,get_constants(mode),MODES[mode]))
    result = np.asarray(fn(before,np.asarray(sp['columns']['DN'],R)))
    ref = stack(sp,'POSTCU',mode)
    np.testing.assert_allclose(result,ref,rtol=2e-6 if mode=='fp32' else 1e-12,atol=1e-30)
    assert np.isfinite(result).all()
    if case in (18,19,20):
        assert np.all(ref[LNR]>before[LNR]) and np.all(ref[LNS]>before[LNS])
    # WRF caller populates no cloud/ice NUMBER slots in ancuten.
    np.testing.assert_array_equal(result[[LNC,LNI]],before[[LNC,LNI]])
    if case in (21,23):
        np.testing.assert_array_equal(result,before)
    if case==22:
        inactive = np.arange(1,41)%4 != 0
        np.testing.assert_array_equal(result[:,inactive],before[:,inactive])


@lru_cache(None)
def column(mode):
    return jax.jit(lambda f,first,cu:driver.nssl2mom_column(
        f,18.,get_constants(mode),MODES[mode],first_step=first,diag=True,cumulus_tendencies=cu))


@lru_cache(None)
def run(mode,case):
    sp = load(mode,case)
    f,cu = fields(sp,mode)
    if not sp['scalars']['CU_USED']:
        cu = None
    return jax.tree_util.tree_map(np.asarray,column(mode)(f,np.bool_(sp['scalars']['ITIMESTEP']==1),cu))


@pytest.mark.parametrize('case',range(18,25))
def test_full_driver_fp64(case):
    sp = load('fp64',case)
    out,precip,diag = run('fp64',case)
    for k,v in NAMES.items():
        ref = np.asarray(sp['columns'][v+'_OUT'])
        scale = max(np.max(np.abs(ref)),np.max(np.abs(sp['columns'][v+'_IN'])))
        assert np.isfinite(out[k]).all(),k
        np.testing.assert_allclose(out[k],ref,rtol=1e-10,atol=1e-13*scale,err_msg=k)
    for k in ('rainncv','snowncv','grplncv','hailncv'):
        np.testing.assert_allclose(precip[k],sp['scalars'][k.upper()],rtol=1e-9,atol=1e-15)
    for k in ('re_cloud','re_ice','re_snow','dbz'):
        np.testing.assert_allclose(diag[k],sp['columns'][k.upper()+'_OUT'],rtol=1e-10,atol=1e-13)


@pytest.mark.parametrize('case',range(18,25))
def test_full_driver_fp32(case):
    sp, sp64 = load('fp32',case),load('fp64',case)
    out,precip,_ = run('fp32',case)
    mass_of = dict(qndrop='QC',qnr='QR',qni='QI',qns='QS',qng='QH',qnh='QHL',qvolg='QH',qvolh='QHL')
    for k,v in NAMES.items():
        r32,r64 = (np.asarray(s['columns'][v+'_OUT']) for s in (sp,sp64))
        p = out[k]
        assert np.isfinite(p).all()
        assert p.dtype == np.float32
        atol = 1e-6*max(np.max(np.abs(r32)),np.max(np.abs(r64)),1e-30)
        bad = (np.abs(p-r32)>2e-3*np.abs(r32)+atol)&(np.abs(p-r64)>2e-3*np.abs(r64)+atol)
        if k in mass_of:
            mk=mass_of[k]
            mass=np.maximum.reduce([np.abs(out[next(n for n,vv in NAMES.items() if vv==mk)]),
                                    np.abs(sp['columns'][mk+'_OUT']),np.abs(sp64['columns'][mk+'_OUT'])])
            bad &= mass>=1e-10
        assert not bad.any(),(k,np.where(bad),p[bad],r32[bad],r64[bad])
    for k in ('rainncv','snowncv','grplncv','hailncv'):
        r32,r64 = sp['scalars'][k.upper()],sp64['scalars'][k.upper()]
        assert min(abs(precip[k]-r32),abs(precip[k]-r64))<=2e-3*max(abs(r32),abs(r64))+1e-12


def test_cloud_ice_only_is_exact_caller_noop():
    # Same pristine sounding, cu_used=1 cloud/ice-only versus cu_used=0.
    for mode in MODES:
        a,b = run(mode,23),run(mode,24)
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            np.testing.assert_array_equal(x,y)


def test_stage_mutant_is_killed(monkeypatch):
    sp = load('fp64',18)
    old = stack(sp,'PRECU','fp64')
    ref = stack(sp,'POSTCU','fp64')
    monkeypatch.setattr(cleanup,'calcnfromcuten',lambda an,*args:an)
    _f,cu = fields(sp,'fp64')
    got = cleanup.calcnfromcuten(old,sp['columns']['DN'],cu,get_constants('fp64'),FP64)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(got,ref,rtol=1e-12,atol=1e-30)
