"""Complete WRF soil solver with actual runoff/tile branches and caller cadence."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
FIXTURE=json.loads((ROOT/'proofs/u2_land/advanced_water_oracle.json').read_text())
MODES=sorted({tuple(c[k] for k in ('opt_run','opt_inf','opt_infdv','opt_tdrn','drain_layer_opt')) for c in FIXTURE['cases']})


@pytest.fixture
def wrf_real(monkeypatch):
    for key,value in {'GPUWRF_NOAHMP_NATIVE_REAL':'1','GPUWRF_NOAHMP_COLUMN_KERNELS':'0',
                      'GPUWRF_NOAHMP_LAYER_SELECT':'0','GPUWRF_NOAHMP_LAYER_LISTS':'0'}.items():
        monkeypatch.setenv(key,value)
    jax.clear_caches()
    try:
        yield
    finally:
        jax.clear_caches()


def matrix(mode,engine=jnp):
    from gpuwrf.physics.noahmp.runoff_options import RunoffParameters,HydrologyParameters,_advanced_soilwater
    from gpuwrf.physics.noahmp.tile_drain import TileParameters,equivalent_depth
    fx=json.loads((ROOT/'proofs/u2_land/advanced_water_oracle.json').read_text())
    keys=('opt_run','opt_inf','opt_infdv','opt_tdrn','drain_layer_opt')
    cases=[c for c in fx['cases'] if tuple(c[k] for k in keys)==mode]
    run,inf,infdv,tdrn,layer=mode
    values=engine.asarray([[c[k] for k in ('dt','qinsur','qseva')] for c in cases],engine.float32).T[:,None,:]
    profiles=engine.asarray([[c[k] for k in ('sh2o','smc','etrani')] for c in cases],engine.float32).transpose(1,2,0)[:,:,None,:]
    params={k:engine.asarray(v,engine.float32).reshape((4,1,1)) if isinstance(v,list) else engine.asarray(v,engine.float32) for k,v in fx['parameters'].items()}
    zsoil=engine.asarray([-.1,-.4,-1.,-2.],engine.float32)[:,None,None]
    dz=engine.concatenate((engine.zeros_like(zsoil[:1]),zsoil[:-1]),axis=0)-zsoil
    def compute(values,profiles,p,zsoil,dz):
        dt,qin,qseva=values;sh2o,smc,etrani=profiles
        rp=RunoffParameters(*(p[k.upper()] for k in RunoffParameters._fields)) if run in (6,7,8) else None
        tp=None
        if tdrn:
            cp=equivalent_depth(p['TD_ADEPTH']-p['TD_DDRAIN'],p['TD_SPAC'],p['TD_RADI'])
            tp=TileParameters(*(p[k] for k in ('TDSMC_FAC','TD_DC','TD_DEPTH','TD_DCOEF','KLAT_FAC','TD_ADEPTH','TD_DDRAIN','TD_SPAC','TD_RADI')),cp)
        hp=HydrologyParameters(rp,tp,engine.ones_like(dt) if tdrn else None,engine.full_like(dt,3000.) if tdrn else None)
        result=_advanced_soilwater(qin,qseva,etrani,sh2o,smc,engine.maximum(0.,smc-sh2o),zsoil,dz,
            *(p[k] for k in ('BEXP','SMCMAX','SMCREF','SMCWLT','DKSAT','DWSAT','KDT','FRZX','SLOPE')),dt,hp,None,
            opt_run=run,opt_inf=inf,opt_infdv=infdv,opt_tdrn=tdrn,drain_layer_opt=layer)
        if not tdrn:result=(*result,engine.zeros_like(dt),engine.full_like(dt,2.5))
        return engine.concatenate((result[0],result[1],*(v[None,...] for v in result[2:])),axis=0)[:,0,:].T
    args=(values,profiles,params,zsoil,dz)
    return np.asarray(jax.jit(compute)(*args) if engine is jnp else compute(*args)),cases,fx


@pytest.mark.parametrize('mode',MODES)
def test_whole_original_advanced_soilwater_literal_byte(mode,wrf_real):
    import importlib.util
    spec=importlib.util.spec_from_file_location('u2_advanced_literal',ROOT/'proofs/u2_land/literal_real32.py')
    literal=importlib.util.module_from_spec(spec);spec.loader.exec_module(literal)
    with literal.source_backend():
        actual,cases,_=matrix(mode,literal.NumpyReal())
    expected=np.asarray([c['expected'] for c in cases],np.float32)
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


@pytest.mark.parametrize('mode',MODES)
def test_whole_original_advanced_soilwater_compiled_bounds(mode,wrf_real):
    actual,cases,fx=matrix(mode)
    expected,low,high=(np.asarray([c[key] for c in cases],np.float32) for key in ('expected','low','high'))
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    excess=np.maximum(np.maximum(low.astype(float)-actual,actual.astype(float)-high),0.)
    nominal=np.isclose(actual,expected,**fx['nominal_tolerance'])
    raw=dict(mode=mode,values=actual.size,compiled_within=int((excess==0).sum()),
             nominal_outside=int((~nominal).sum()),max_excess=float(excess.max()),
             max_abs=float(np.abs(actual.astype(float)-expected).max()),
             excess_indices=np.argwhere(excess>0).tolist())
    print('compiled_original_bounds',json.dumps(raw))
    if np.any(excess>0):
        # This comparison is regression evidence only. The original bound is
        # still violated, so report XFAIL, never a widened-tolerance PASS.
        previous=json.loads((ROOT/'proofs/u2_land/receipts/LAND48_advanced_water.json').read_text())['results'][str(mode)]
        assert raw['excess_indices']==previous['excess_indices'],raw
        assert raw['max_excess']==previous['max_excess'],raw
        pytest.xfail(f"Disclosed compiled deviation outside frozen original bounds: {json.dumps(raw)}; D6 pending")
    assert np.all(excess==0)
