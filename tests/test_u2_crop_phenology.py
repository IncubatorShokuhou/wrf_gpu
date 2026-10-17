"""Original crop PHENOLOGY/SFLX caller: no LAI floor; old INTEGER PGS."""
import json
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def matrix(engine=jnp):
    from gpuwrf.physics.noahmp.phenology import _crop_phenology
    from gpuwrf.physics.noahmp.types import NoahMPPhenology
    fx=json.loads((ROOT/'proofs/u2_land/crop_phenology_oracle.json').read_text());cases=fx['cases']
    keys=('snowh','lai','sai','shdfac','shdmax','hvt','hvb')
    values=engine.asarray([[c[k] for c in cases] for k in keys],engine.float32)[:,None,:]
    integers=engine.asarray([[c[k] for c in cases] for k in ('croptype','pgs','vegtyp')],engine.int32)[:,None,:]
    def compute(values,integers):
        snowh,lai,sai,shdfac,shdmax,hvt,hvb=values;crop,pgs,veg=integers
        land=SimpleNamespace(snowh=snowh,lai=lai,sai=sai)
        static=SimpleNamespace(ivgtyp=veg,parameters=SimpleNamespace(hvt=hvt,hvb=hvb),shdmax=shdmax,shdfac=shdfac)
        base=NoahMPPhenology(*(engine.zeros_like(lai) for _ in range(6)))
        return engine.stack(_crop_phenology(land,static,base,crop,pgs),axis=-1)[0]
    return np.asarray(jax.jit(compute)(values,integers) if engine is jnp else compute(values,integers)),cases,fx


def test_original_crop_phenology_and_fveg(monkeypatch):
    monkeypatch.setenv('GPUWRF_NOAHMP_NATIVE_REAL','1')
    jax.clear_caches()
    try:
        actual,cases,fx=matrix()
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        np.testing.assert_allclose(actual,[c['expected'] for c in cases],**fx['nominal_tolerance'])
    finally:
        jax.clear_caches()
