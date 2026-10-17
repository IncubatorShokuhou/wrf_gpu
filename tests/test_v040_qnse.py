"""Standalone QNSE columns against staged, unmodified WRF (not a forecast)."""
from __future__ import annotations

import inspect
import json
import hashlib
from contextlib import nullcontext
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.pbl_qnse import qnsepbl_column
from gpuwrf.physics.sfclay_qnse import qnsesfc_column
from gpuwrf.physics.qnse_edmf import mfshconvpbl_column
from gpuwrf.physics import pbl_qnse, sfclay_qnse, qnse_edmf

ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / 'proofs/s2_qnse/oracle.json'
sys.path.insert(0,str(ROOT/'proofs/s2_qnse'))
from real4_literal import real4_backend
FUNCTIONS = {'pbl': qnsepbl_column, 'sfc': qnsesfc_column, 'mf':mfshconvpbl_column}
COMPILED = {name: jax.jit(fn) for name, fn in FUNCTIONS.items()}


def load():
    return json.loads(ORACLE.read_text())


def kwargs(case, scheme, dtype):
    fn = FUNCTIONS[scheme]
    names = set(inspect.signature(fn).parameters)
    section = {'pbl':'INPUT','sfc':'INPUT_SFC','mf':'INPUT_MF'}[scheme]
    result = {k: jnp.asarray(v, dtype=dtype) for k,v in case[section].items() if k in names}
    if scheme == 'sfc' and case[section]['scm_force_flux'] < 0:
        result['scm_force_flux'] = None
    return result


def oracle_case(scheme,index,mode='fp32'):
    data=json.loads(ORACLE.with_name('mf_oracle.json').read_text()) if scheme=='mf' else load()
    return data['cases'][mode][index]


@pytest.mark.parametrize('scheme', ('pbl','sfc','mf'))
@pytest.mark.parametrize('index', range(15))
def test_pristine_wrf_real4_literal_bytes(scheme,index):
    case=oracle_case(scheme,index)
    inp=kwargs(case,scheme,jnp.float32)
    with real4_backend(pbl_qnse,sfclay_qnse,qnse_edmf):
        result=FUNCTIONS[scheme](**inp)
    truth=case[{'pbl':'OUTPUT','sfc':'OUTPUT_SFC','mf':'OUTPUT_MF'}[scheme]]
    compared=0
    for field, value in result.items():
        assert np.isfinite(np.asarray(value)).all(), (scheme,index,field)
        assert value.dtype==(np.int32 if field=='KPBL' else np.float32)
        if field in truth:
            compared+=1
            assert np.asarray(value).tobytes()==np.asarray(truth[field],value.dtype).tobytes(),(scheme,index,field)
    assert compared >= {'pbl':15,'sfc':30,'mf':19}[scheme]


def compiled_columns(scheme,interpret=True):
    from gpuwrf.kernels.phys_qnse_columns import qnse_columns
    cases=[oracle_case(scheme,k) for k in range(15)]
    cols=[kwargs(case,scheme,jnp.float32) for case in cases]
    if scheme=='sfc':
        for col in cols:
            if col['scm_force_flux'] is None:col['scm_force_flux']=jnp.float32(1.)
    inputs={key:jnp.stack([col[key] for col in cols],axis=-1) for key in cols[0]}
    return jax.jit(lambda a:qnse_columns(scheme,a,interpret=interpret,block=16))(inputs)


@pytest.mark.parametrize('scheme',('pbl','sfc'))
@pytest.mark.parametrize('interpret',(True,None),ids=('interpreter','portable'))
def test_compiled_real32_inside_independent_input_ulp_bound(scheme,interpret):
    records=json.loads(ORACLE.with_name('enclosures.json').read_text())['records'][scheme]
    result=compiled_columns(scheme,interpret)
    for field,value in result.items():
        for k,record in enumerate(records):
            if field not in record:continue
            lo,hi=np.asarray(record[field]['lo']),np.asarray(record[field]['hi'])
            assert np.isfinite(lo).all() and np.isfinite(hi).all(),(scheme,k,field,'unbounded')
            actual=np.asarray(value)[...,k]
            assert np.isfinite(actual).all() and ((lo<=actual)&(actual<=hi)).all(),(scheme,k,field,actual,lo,hi)


@pytest.mark.parametrize('interpret',(True,None),ids=('interpreter','portable'))
def test_compiled_mf_matches_disclosed_rounding_artifact(interpret):
    # Regression ONLY. The pristine BYTE test above establishes source fidelity.
    # The raw compiled MF WRF-spread gate FAIL is deliberately retained in the
    # packet, admitted as an experimental deviation by the standing wave policy.
    # No output-derived number becomes a numerical acceptance tolerance here.
    packet=json.loads(ORACLE.with_name('compiler_rounding.json').read_text())
    assert packet['raw_gate']=='FAIL' and packet['outside_spread_cells']>0
    assert packet['gpu_executed'] is False
    result=compiled_columns('mf',interpret)
    assert set(result)==set(packet['compiled_outputs'])
    for field,value in result.items():
        actual=np.asarray(value)
        expected=np.asarray(packet['compiled_outputs'][field],actual.dtype)
        assert np.isfinite(actual).all()
        assert actual.tobytes()==expected.tobytes(),field


@pytest.mark.parametrize('scheme', ('pbl','sfc'))
def test_real32_lower_contains_no_double(scheme):
    case = load()['cases']['fp32'][0]
    ir = str(COMPILED[scheme].lower(**kwargs(case,scheme,jnp.float32)).compiler_ir('stablehlo'))
    assert 'f64' not in ir


def test_oracle_regime_census():
    data = load()
    assert len(data['regimes']) == 15
    for mode in ('fp32','fp64'):
        cases = data['cases'][mode]
        assert len(cases) == 15
        assert np.any(np.diff(cases[0]['INPUT']['theta']) < 0.)
        assert np.all(np.diff(cases[1]['INPUT']['theta']) > 0.)
        assert np.count_nonzero(cases[6]['INPUT']['qc']) == 5
        assert cases[5]['INPUT']['snow'] > 0.
        assert cases[8]['INPUT']['corf'] < 0.
        assert cases[10]['INPUT']['chklowq'] == 0.
        assert cases[11]['INPUT']['ct'] != 0.
        for field in ('RUBLTEN','RVBLTEN','RTHBLTEN','RQVBLTEN','RQCBLTEN'):
            assert np.count_nonzero(cases[11]['INPUT'][field]) == 16
        assert cases[0]['INPUT']['hfx'] > 0.
        assert cases[1]['INPUT']['hfx'] < 0.
        assert cases[12]['INPUT_SFC']['scm_force_flux'] == 1.
        assert cases[13]['INPUT_SFC']['scm_force_flux'] == -1.
        assert cases[14]['INPUT_SFC']['itimestep'] == 1.


def test_oracle_driver_identity_and_precision_metadata():
    data = load()
    assert data['driver_sha256'] == hashlib.sha256(ORACLE.with_name('oracle.f90').read_bytes()).hexdigest()
    assert set(data['cases']) == {'fp32','fp64'}
    assert set(data['source_sha256']) == {
        'share/module_model_constants.F', 'phys/module_sf_qnsesfc.F', 'phys/module_bl_qnsepbl.F'}
    assert 'mfshconv=0' in data['scope']
    assert data['tolerances']['fp64']['absolute'] == 5e-10


def test_surface_tables_are_host_values_between_traces():
    from gpuwrf.physics.sfclay_qnse import _tables
    for dtype in ('float32','float64'):
        a,b,spacing,maximum = _tables(dtype)
        assert isinstance(a,np.ndarray) and isinstance(b,np.ndarray)
        assert a.dtype == np.dtype(dtype) and b.dtype == np.dtype(dtype)
        assert a.shape == b.shape == (10001,)
        assert 2.29 < maximum < 2.31
        assert spacing > 0.


@pytest.mark.parametrize('scheme', ('pbl','sfc'))
def test_second_independent_jit_does_not_leak_tracers(scheme):
    # E206: host cache must not retain JAX values from a first trace.
    inp = kwargs(load()['cases']['fp32'][6],scheme,jnp.float32)
    for _ in range(2):
        traced = jax.make_jaxpr(FUNCTIONS[scheme])(**inp)
        assert traced.jaxpr.outvars


def test_pbl_shape_errors():
    inp = kwargs(load()['cases']['fp32'][0],'pbl',jnp.float32)
    with pytest.raises(ValueError,match='profile shape'):
        qnsepbl_column(**{**inp,'v':jnp.zeros(3,jnp.float32)})
    with pytest.raises(ValueError,match='interfaces'):
        qnsepbl_column(**{**inp,'p_int':jnp.zeros(3,jnp.float32)})


@pytest.mark.parametrize('theta,rt',((295.,.005),(268.,.012),(295.,.018)))
def test_independent_thermodynamic_enclosure_covers_real4_corners(theta,rt):
    from itertools import product
    from affine_bounds import affine_backend,affine_inputs
    args=dict(p=np.float32(97000.),thl=np.float32(theta),rt=np.float32(rt))
    c=qnse_edmf._constants('float32')
    with affine_backend(qnse_edmf):
        bounds=qnse_edmf._thermo_liquid(**affine_inputs(args),c=c)
    assert all(np.isfinite(b.lo).all() and np.isfinite(b.hi).all() for b in bounds)
    for directions in product((-1,1),repeat=3):
        corner={key:np.nextafter(value,np.float32(-np.inf if sign<0 else np.inf))
                for (key,value),sign in zip(args.items(),directions,strict=True)}
        with real4_backend(qnse_edmf):values=qnse_edmf._thermo_liquid(**corner,c=c)
        for value,bound in zip(values,bounds,strict=True):
            value=np.asarray(value)
            assert np.isfinite(value).all()
            assert ((bound.lo<=value)&(value<=bound.hi)).all()
