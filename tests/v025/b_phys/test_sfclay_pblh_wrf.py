"""BP102: WRF's supplied PBLH, including zero, determines convective gust."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import surface_layer as SL
from gpuwrf.physics.fp32 import surface_layer_real as REAL
from test_sfclay_snow_andreas import view, IN_COLS, THRESHOLDS

DATA=json.loads((Path(__file__).parent/'fixtures/sfclay_pblh_wrf.json').read_text())


@pytest.fixture(autouse=True)
def flags(monkeypatch):
    monkeypatch.setenv('GPUWRF_MYNN_SFC_WSPD','1')
    monkeypatch.setenv('GPUWRF_MYNN_FLTV_WRF','1')
    monkeypatch.setattr(SL,'_MYNN_PSIQ_FLUX_WRF',True)
    jax.clear_caches()
    yield
    jax.clear_caches()


def finite(*groups):
    for role,values in groups:
        for name,value in values.items():
            assert np.isfinite(value).all(),f'{role} {name} nonfinite'


def values(diag):
    finite(('candidate', {str(i):np.asarray(x) for i,x in enumerate(jax.tree.leaves(diag))}))
    out={n:np.asarray(getattr(diag,n)).reshape(-1) for n in
         ('hfx','lh','qsfc','mol','znt','chs2','cqs2','t2','q2')}
    out.update(ust=np.asarray(diag.fluxes.ustar).reshape(-1),
        wspd=np.asarray(diag.fluxes.wspd).reshape(-1),
        qfx=np.asarray(diag.fluxes.qv_flux*diag.fluxes.rhosfc).reshape(-1))
    return out


def check(actual,reference,inputs):
    finite(('candidate',actual),('reference',reference),('input',inputs))
    # WSPD is the deletion-sensitive primitive: native has exact equality,
    # retained DOUBLE can differ only by rounding to original REAL4.
    spacing=np.spacing(np.asarray(reference['wspd'],np.float32)).astype(float)
    assert np.all(abs(actual['wspd']-reference['wspd'])<=4*spacing),'WRF WSPD >4 REAL32 ULP'
    for n in actual.keys()&reference.keys():
        if n=='wspd':continue
        # Preserve the existing pristine full-surface field tolerances.
        # QFX uses the same 3% bar as LH, with an 8-ULP precision allowance.
        limit=THRESHOLDS.get(n,.03 if n=='qfx' else .01)
        floor=8*np.abs(np.spacing(reference[n])) if n=='qfx' else 1e-6
        bound=np.maximum(limit*np.abs(reference[n]),floor)
        assert np.all(np.abs(actual[n]-reference[n])<=bound),n


@pytest.mark.parametrize('native',[True,False],ids=['native','retained'])
def test_real_zero_pblh_surface_memory_matches_original(monkeypatch,native):
    monkeypatch.setattr(REAL,'_NATIVE_REAL',native)
    monkeypatch.setattr(SL,'_MYNN_SFC_PBLH_WRF',True)
    rows=[dict(r) for r in DATA['inputs']['rows']]
    assert len(rows)==24 and all(r['pblh']==0 for r in rows)
    for i,raw in enumerate(DATA['reference']):
        reference={n:np.asarray(x,np.float32) for n,x in raw.items()}
        inputs={n:np.asarray([r[n] for r in rows],np.float32) for n in IN_COLS}
        finite(('input',inputs),('reference',reference))
        with jax.enable_x64(True):
            out=SL.surface_layer_with_diagnostics(view(rows)._replace(dx_m=9000.),first_timestep=i==0)
        actual=values(out);check(actual,reference,inputs)
        for j,row in enumerate(rows):
            for n in ('ust','mol','qsfc','hfx','qfx','znt'):row[n]=float(actual[n][j])


@pytest.mark.parametrize('pblh',[1.,50.,1000.])
@pytest.mark.parametrize('native',[True,False])
def test_positive_pblh_and_default_off_preserve_program(monkeypatch,pblh,native):
    monkeypatch.setattr(REAL,'_NATIVE_REAL',native)
    rows=[dict(r,pblh=pblh) for r in DATA['inputs']['rows']]
    state=view(rows)._replace(dx_m=9000.)
    monkeypatch.setattr(SL,'_MYNN_SFC_PBLH_WRF',False)
    off=SL.surface_layer_with_diagnostics(state)
    monkeypatch.setattr(SL,'_MYNN_SFC_PBLH_WRF',True)
    on=SL.surface_layer_with_diagnostics(state)
    for a,b in zip(jax.tree.leaves(off),jax.tree.leaves(on)):
        finite(('off',{'leaf':np.asarray(a)}),('on',{'leaf':np.asarray(b)}))
        np.testing.assert_array_equal(a,b)


def test_resolved_flag_splits_cache_key(monkeypatch):
    from gpuwrf.runtime.aot_cheap_key import module_const_env_hash
    monkeypatch.setattr(SL,'_MYNN_SFC_PBLH_WRF',False);off=module_const_env_hash()
    monkeypatch.setattr(SL,'_MYNN_SFC_PBLH_WRF',True)
    assert module_const_env_hash()!=off


@pytest.mark.parametrize('role',['input','candidate','reference'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_gate_refuses_nonfinite_before_reduction(role,bad):
    a={'wspd':np.ones(24,np.float32)};b={'wspd':a['wspd'].copy()};c={'pblh':np.zeros(24,np.float32)}
    (c['pblh'] if role=='input' else a['wspd'] if role=='candidate' else b['wspd'])[0]=bad
    with pytest.raises(AssertionError,match='nonfinite'):check(a,b,c)


GATED_FIELDS=(
    [('input',n) for n in IN_COLS]
    +[('reference',n) for n in DATA['reference'][0]]
    +[('candidate',n) for n in ('hfx','lh','qsfc','mol','znt','chs2','cqs2','t2','q2','ust','wspd','qfx')]
)
@pytest.mark.parametrize('role,field',GATED_FIELDS)
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_every_gated_field_rejects_nonfinite(role,field,bad):
    reference={n:np.asarray(x,np.float32) for n,x in DATA['reference'][0].items()}
    candidate={n:reference[n].copy() for n in ('hfx','lh','qsfc','mol','znt','chs2','cqs2','t2','q2','ust','wspd','qfx')}
    inputs={n:np.asarray([r[n] for r in DATA['inputs']['rows']],np.float32) for n in IN_COLS}
    {'input':inputs,'reference':reference,'candidate':candidate}[role][field][0]=bad
    with pytest.raises(AssertionError,match='nonfinite'):check(candidate,reference,inputs)
