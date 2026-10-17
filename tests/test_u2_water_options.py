"""Complete original SOILWATER choices, including 3/6 fine-step cadence."""
import json,os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


def matrix(run,inf,engine=jnp):
    from gpuwrf.physics.noahmp.water_hydro import _soilwater
    fx=json.loads((ROOT/"proofs/u2_land/water_oracle.json").read_text())
    cases=[c for c in fx["cases"] if (c["opt_run"],c["opt_inf"])==(run,inf)]
    p=fx["parameters"]
    values=engine.asarray([[c[k] for k in ("dt","qinsur","qseva")] for c in cases],engine.float32).T[:,None,:]
    arrays=engine.asarray([[c[k] for k in ("sh2o","smc","etrani")] for c in cases],engine.float32).transpose(1,2,0)[:,:,None,:]
    params=engine.asarray([p[k] for k in ("bexp","smcmax","smcwlt","dksat","dwsat")],engine.float32)[:,:,None,None]
    constants=engine.asarray([p[k] for k in ("kdt","frzx","slope")],engine.float32)
    zsoil=engine.asarray(fx["zsoil"],engine.float32)[:,None,None]
    dz=engine.concatenate((engine.zeros_like(zsoil[:1]),zsoil[:-1]),axis=0)-zsoil
    def compute(arrays,values,params,zsoil,dz,constants):
        sh2o,smc,etrani=arrays;dt,qin,qseva=values;bexp,smcmax,smcwlt,dksat,dwsat=params
        sice=engine.maximum(0.,smc-sh2o)
        result=_soilwater(qin,qseva,etrani,sh2o,smc,sice,zsoil,dz,engine.zeros_like(qin),bexp,smcmax,smcmax,smcwlt,dksat,dwsat,
                          *constants,dt,opt_inf=inf,opt_run=run)
        return engine.concatenate((result[0],result[1],*(x[None] for x in result[2:])),axis=0)[:,0,:].T
    out=jax.jit(compute)(arrays,values,params,zsoil,dz,constants) if engine is jnp else compute(arrays,values,params,zsoil,dz,constants)
    return np.asarray(out),cases,fx


@pytest.mark.parametrize("run",[3,4])
@pytest.mark.parametrize("inf",[1,2])
@pytest.mark.parametrize("lists",[False,True])
def test_complete_pristine_water_runtime_operands(monkeypatch,run,inf,lists):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL","1")
    monkeypatch.setenv("GPUWRF_NOAHMP_LAYER_SELECT","1" if lists else "0")
    monkeypatch.setenv("GPUWRF_NOAHMP_LAYER_LISTS","1" if lists else "0")
    jax.clear_caches()
    actual,cases,fx=matrix(run,inf)
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c["expected"] for c in cases],**fx["nominal_tolerance"])


def test_water_options_bound():
    from gpuwrf.io.noahmp_options import selected_options,option_refusals
    selected=selected_options({"noah_mp":{"opt_run":4,"opt_inf":2}})
    assert (selected.opt_run,selected.opt_inf)==(4,2)
    assert not option_refusals({"noah_mp":{"opt_run":4,"opt_inf":2}})
