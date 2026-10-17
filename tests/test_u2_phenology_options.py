"""Non-carbon vegetation choices use original PHENOLOGY and FVEG caller references."""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def matrix(dveg,engine=jnp):
    from gpuwrf.physics.noahmp.phenology import noahmp_phenology_table
    filename="phenology_dynamic_oracle.json" if dveg in (2,5,6) else "phenology_oracle.json"
    fx=json.loads((ROOT/"proofs/u2_land"/filename).read_text())
    cases=[case for case in fx["cases"] if case["dveg"]==dveg]
    r=lambda key:engine.asarray([case[key] for case in cases],engine.float32).reshape(1,-1)
    # Pre-gathered monthly tables are a supported original phenology interface.
    params=SimpleNamespace(laim=engine.asarray([case["laim"] for case in cases],engine.float32).T[:,None,:],
                           saim=engine.asarray([case["saim"] for case in cases],engine.float32).T[:,None,:],
                           hvt=r("hvt"),hvb=r("hvb"))
    values=engine.stack([r(key) for key in ("snowh","tv","lat","julian","lai","sai","shdfac","shdmax")],axis=0)
    def compute(values):
        snowh,tv,lat,julian,lai,sai,shdfac,shdmax=values
        land=SimpleNamespace(snowh=snowh,tv=tv,lai=lai,sai=sai)
        forcing=SimpleNamespace(julian=julian,yearlen=engine.asarray(365.,engine.float32))
        static=SimpleNamespace(parameters=params,ivgtyp=engine.asarray([[case["vegtyp"] for case in cases]],engine.int32),
                               lat=lat,shdfac=shdfac,shdmax=shdmax)
        result=noahmp_phenology_table(land,forcing,static,dveg=dveg)
        return engine.stack(result,axis=-1)[0]
    result=jax.jit(compute)(values) if engine is jnp else compute(values)
    return np.asarray(result),cases,fx


@pytest.mark.parametrize("dveg",[1,2,3,4,5,6,7,8,9])
def test_pristine_phenology_and_caller_runtime_inputs(monkeypatch,dveg):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL","1")
    jax.clear_caches()
    actual,cases,fx=matrix(dveg)
    expected=np.asarray([case["expected"] for case in cases])
    assert actual.dtype==np.float32 and np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual,expected,**fx["nominal_tolerance"])


def test_vegetation_selection_admission():
    from gpuwrf.io.noahmp_options import selected_options,option_refusals
    for code in (1,2,3,5,6,7,8,9):
        assert selected_options({"noah_mp":{"dveg":code}}).dveg==code
        assert not option_refusals({"noah_mp":{"dveg":code}})
    for code in (0,10):assert option_refusals({"noah_mp":{"dveg":code}})
    assert selected_options({"noah_mp":{"dveg":4}}) is None
