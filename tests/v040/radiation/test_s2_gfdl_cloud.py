"""Native ETARA caller and RADTN cloud qualification; not solver admission."""
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import ra_gfdleta as port

ROOT=Path(__file__).resolve().parents[3]
FIXTURE=ROOT/'tests/fixtures/s2_gfdl_cloud_v1.npz'


@pytest.fixture(scope='module')
def data():
    with np.load(FIXTURE,allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def literal():
    spec=importlib.util.spec_from_file_location('gfdl_real4',ROOT/'proofs/s2_flggfdl/literal.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper.load(Path(port.__file__))


def check(value,reference,byte):
    value=np.asarray(value)
    assert value.shape==reference.shape and value.dtype==reference.dtype==np.float32
    assert np.isfinite(value).all() and np.isfinite(reference).all()
    if byte:
        assert value.tobytes()==reference.tobytes()
    else:
        error=np.abs(value.astype(np.float64)-reference.astype(np.float64))
        allowance=16.*np.spacing(np.abs(reference)).astype(np.float64)+1.e-12
        assert np.all(error<=allowance),f'{np.count_nonzero(error>allowance)} outside;max {error.max()}'


def test_fixture_source_provenance(data):
    receipt=json.loads(FIXTURE.with_suffix('.json').read_text())
    assert receipt['final_BYTE'] and receipt['columns']==40 and receipt['levels']==44
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==receipt['fixture_sha256']
    assert np.count_nonzero(data['grid_cloud'])==receipt['cloudy_levels']==125
    tables=json.loads((ROOT/'proofs/s2_flggfdl/FG24_tables.json').read_text())
    asset=Path(port.__file__).with_name('gfdleta_tables_v1.npz')
    assert hashlib.sha256(asset.read_bytes()).hexdigest()==tables['asset_sha256']
    with np.load(asset,allow_pickle=False) as f:
        for key,digest in tables['prior_array_sha256'].items():
            assert hashlib.sha256(f[key].tobytes()).hexdigest()==digest


@pytest.mark.parametrize('compiled',[False,True])
def test_etara_preprocessing(data,compiled):
    fields=data['in_fields'];surface=data['in_sfc']
    args=tuple(fields[:,:,i] for i in (0,1,12,5,6,9))+(surface[:,8],surface[:,9])
    kwargs=dict(qi=fields[:,:,8])
    if compiled:
        run=jax.jit(port.preprocess_column)
        value=run(*map(jnp.asarray,args),qi=jnp.asarray(kwargs['qi']))
    else:
        value=literal().preprocess_column(*args,**kwargs)
    for key,candidate in value.items():check(candidate,data['pre_'+key],not compiled)


@pytest.mark.parametrize('compiled',[False,True])
def test_active_ferrier_grid_cloud(data,compiled):
    # WRF module_model_constants R_d/R_v as REAL, not an alternate epsilon.
    ep2=np.float32(np.float32(287.)/np.float32(461.6))
    args=tuple(data['grid_'+k] for k in ('t','p','q','qc','qi','cutop','cubot'))
    if compiled:
        value=jax.jit(jax.vmap(lambda *x:port.grid_cloud_fraction(*x,ep2=ep2)))(*map(jnp.asarray,args))
    else:
        run=literal().grid_cloud_fraction
        value=np.asarray([run(*(a[i] for a in args),ep2=ep2) for i in range(40)])
    check(value,data['grid_cloud'],not compiled)
