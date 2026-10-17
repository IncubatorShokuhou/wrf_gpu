"""Original complete equilibrium-water table option on real runtime operands."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


def matrix(inf,engine=jnp):
    from gpuwrf.physics.noahmp.groundwater_options import _equilibrium_soilwater
    fx=json.loads((ROOT/"proofs/u2_land/equilibrium_oracle.json").read_text())
    cases=[c for c in fx["cases"] if c["opt_inf"]==inf];p=fx["parameters"]
    values=engine.asarray([[c[k] for k in ("dt","qinsur","qseva")] for c in cases],engine.float32).T[:,None,:]
    arrays=engine.asarray([[c[k] for k in ("sh2o","smc","etrani")] for c in cases],engine.float32).transpose(1,2,0)[:,:,None,:]
    params=engine.asarray([p[k] for k in ("bexp","smcmax","smcwlt","dksat","dwsat","psisat")],engine.float32)[:,:,None,None]
    constants=engine.asarray([p[k] for k in ("fsatmx","timean")],engine.float32)
    zsoil=engine.asarray(fx["zsoil"],engine.float32)[:,None,None]
    dz=engine.concatenate((engine.zeros_like(zsoil[:1]),zsoil[:-1]),axis=0)-zsoil
    def compute(arrays,values,params,zsoil,dz,constants):
        sh2o,smc,et=arrays;dt,qin,qseva=values;bexp,smcmax,smcwlt,dksat,dwsat,psisat=params
        result=_equilibrium_soilwater(qin,qseva,et,sh2o,smc,engine.maximum(0.,smc-sh2o),zsoil,dz,bexp,smcmax,smcwlt,dksat,dwsat,psisat,
                                      dt,*constants,opt_inf=inf)
        return engine.concatenate((result[0],result[1],*(v[None] for v in result[2:])),axis=0)[:,0,:].T
    args=(arrays,values,params,zsoil,dz,constants)
    out=jax.jit(compute)(*args) if engine is jnp else compute(*args)
    return np.asarray(out),cases,fx


@pytest.mark.parametrize("inf",[1,2])
def test_original_equilibrium_water_system(inf):
    actual,cases,fx=matrix(inf)
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c["expected"] for c in cases],**fx["nominal_tolerance"])


def test_equilibrium_selection_uses_actual_table():
    from gpuwrf.io.noahmp_options import selected_options,option_refusals
    opts=selected_options({"noah_mp":{"opt_run":2}})
    assert (opts.opt_run,opts.fsatmx,opts.timean)==(2,.38,10.5)
    assert not option_refusals({"noah_mp":{"opt_run":2}})
