"""Original WRF REAL4 truth, not GPU-vs-GPU regression."""
from collections import namedtuple
from contextlib import contextmanager
from functools import partial
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.noahmp import groundwater as gw, water_hydro as water

PROOF = Path(__file__).resolve().parents[1] / 'proofs/s2_mmf'
spec = importlib.util.spec_from_file_location('mmf_literal', PROOF / 'literal_real4.py')
literal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(literal)


class LiteralLax:
    @staticmethod
    def fori_loop(start, end, body, state):
        for i in range(start, end): state = body(i, state)
        return state

    @staticmethod
    def while_loop(cond, body, state):
        for _ in range(501):
            if not bool(cond(state)): return state
            state = body(state)
        raise AssertionError('original cold-start iteration limit exceeded')


@contextmanager
def literal_body():
    old = gw.jnp, gw.lax, water.jnp
    gw.jnp, gw.lax, water.jnp = literal.LiteralNumpy(), LiteralLax(), literal.LiteralNumpy()
    try: yield
    finally: gw.jnp, gw.lax, water.jnp = old


@pytest.fixture(scope='module')
def source():
    with np.load(PROOF / 'original_wrf_real4.npz', allow_pickle=False) as data:
        yield data


def group(data, prefix):
    return {k[len(prefix)+1:]: data[k] for k in data.files if k.startswith(prefix+'/')}


def parameters(data, prefix):
    values = group(data, prefix)
    return namedtuple('OriginalTables', values)(**values)


def grid_args(data, tag):
    f = group(data, f'grid/{tag}/input')
    static = gw.GroundwaterStatic(*(f[n] for n in ('FDEPTH','AREA','HGT','EQZWT','RIVERBED','RIVERCOND','PEXP','SMOISEQ')))
    carry = gw.GroundwaterCarry(*(f.get(n.upper(), np.zeros_like(f['ZWT'])) for n in gw.GroundwaterCarry._fields))
    args = (static,carry,f['SMOIS'],f['SH2O'],f['SMCWTD'],f['DZS'],f['ISLTYP'],
        (f['XLAND']<1.5)&(f['SEAICE']<.5)&(f['IVGTYP']!=15),f['IVGTYP']==13,
        parameters(data,'tables'),np.float32(150))
    return args


def grid_fields(result):
    c,smc,sh2o,aux = result
    return dict(SMOIS=smc,SH2O=sh2o,SMCWTD=aux,**{n.upper():v for n,v in zip(c._fields,c,strict=True)})


@pytest.mark.parametrize('tag', ['real','rise','fall','frozen_fall','spring'])
def test_literal_spatial_vs_pristine(source, tag):
    args = jax.tree.map(lambda a: literal.real4(np.asarray(a)), grid_args(source,tag))
    with literal_body(): actual = grid_fields(gw.wtable(*args))
    for n,b in group(source,f'grid/{tag}/expected').items():
        a = np.asarray(actual[n]); assert np.isfinite(a).all() and np.isfinite(b).all()
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), (tag,n)


def test_compiled_spatial_independent_original_bounds(source):
    kernel = jax.jit(gw.wtable)
    for tag in ('real','rise','fall','frozen_fall','spring'):
        actual = grid_fields(kernel(*jax.tree.map(jnp.asarray,grid_args(source,tag))))
        for n,b in group(source,f'grid/{tag}/expected').items():
            a = np.asarray(actual[n]); bound = source[f'bounds/grid/{tag}/accepted/{n}']
            assert np.isfinite(a).all() and np.isfinite(bound).all()
            assert np.all(np.abs(a.astype(np.float64)-b) <= bound), (tag,n)


def init_args(data):
    f = group(data,'init/input')
    return tuple(f[n] for n in ('FDEPTH','AREA','HGT','EQZWT','RECHCLIM','RIVERBED','SMOIS','SH2O','DZS','ISLTYP','IVGTYP')) + (
        parameters(data,'init/tables'),np.float32(150))


def init_fields(result):
    s,c,smc,sh2o,aux = result
    return dict(SMOIS=smc,SH2O=sh2o,SMCWTD=aux,**{n.upper():v for n,v in zip(s._fields,s,strict=True)},
                **{n.upper():v for n,v in zip(c._fields,c,strict=True)})


def test_literal_cold_start_vs_pristine(source):
    args = jax.tree.map(lambda a: literal.real4(np.asarray(a)),init_args(source))
    with literal_body(): actual = init_fields(gw.initialize(*args))
    for n,b in group(source,'init/expected').items():
        a = np.asarray(actual[n]); assert np.isfinite(a).all() and a.tobytes()==b.tobytes(), n


def column_args(data, option, regime):
    prefix=f'column/{option}/{regime}/arg'
    return tuple(list(data[f'{prefix}/{k}']) if k in (*range(2,8),*range(10,16))
                 else data[f'{prefix}/{k}'] for k in range(19))


def column_fields(result):
    sh,sm,aux,wt,deep,rech,runsf,qd,runsub=result
    return dict(SH2O=np.stack(sh).reshape(4,16),SMOIS=np.stack(sm).reshape(4,16),
        SMCWTD=aux,ZWT=wt,DEEPRECH=np.float32(1.e-5)+deep,
        RECH=np.float32(.125)+rech*np.float32(1000),RUNSF=runsf,QDRAIN=qd,RUNSUB=runsub)


@pytest.mark.parametrize('option', [1,2])
@pytest.mark.parametrize('regime', ['dry','rain','burst','frozen','saturated'])
def test_literal_full_soil_column_vs_pristine(source, option, regime):
    args=jax.tree.map(lambda a:literal.real4(np.asarray(a)),column_args(source,option,regime))
    with literal_body(): actual=column_fields(gw._soilwater5_core(*args,opt_inf=option))
    for n,b in group(source,f'column/{option}/{regime}/expected').items():
        a=np.asarray(actual[n]).reshape(b.shape)
        assert np.isfinite(a).all() and a.tobytes()==b.tobytes(),(option,regime,n)


def test_original_activity_is_nonvacuous(source):
    for name in ('QLAT','QRF','RECH'):
        assert np.count_nonzero(source['grid/real/expected/'+name]) > 1000
    assert np.count_nonzero(source['grid/spring/expected/QSPRING']) > 100
    assert np.count_nonzero(source['column/1/burst/expected/RUNSF']) > 0


def test_compiled_initializer_and_soil_independent_original_bounds(source):
    with np.load(PROOF / 'original_native_bounds.npz', allow_pickle=False) as bounds:
        actual = init_fields(jax.jit(gw.initialize)(*jax.tree.map(jnp.asarray,init_args(source))))
        for n,b in group(source,'init/expected').items():
            a=np.asarray(actual[n]);limit=bounds['init/bound/'+n]
            assert np.isfinite(a).all() and np.isfinite(limit).all()
            assert np.all(np.abs(a.astype(np.float64)-b)<=limit),n
        for option in (1,2):
            kernel=jax.jit(partial(gw._soilwater5_core,opt_inf=option))
            for regime in ('dry','rain','burst','frozen','saturated'):
                actual=column_fields(kernel(*jax.tree.map(jnp.asarray,column_args(source,option,regime))))
                for n,b in group(source,f'column/{option}/{regime}/expected').items():
                    a=np.asarray(actual[n]).reshape(b.shape)
                    limit=bounds[f'column/{option}/{regime}/bound/{n}']
                    assert np.isfinite(a).all() and np.isfinite(limit).all()
                    assert np.all(np.abs(a.astype(np.float64)-b)<=limit),(option,regime,n)


def test_rosr12_original_reciprocal_order():
    with np.load(PROOF / 'rosr12_original.npz', allow_pickle=False) as data:
        args=[[literal.real4(v) for v in data[n]] for n in ('a','b','c','d')]
        actual=np.stack(gw._rosr12_4(*args))
        assert actual.tobytes()==data['expected'].tobytes()
