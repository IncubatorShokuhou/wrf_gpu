"""Original prognostic groundwater trajectory with only the source DOUBLE island."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def matrix(engine=jnp):
    from gpuwrf.physics.noahmp.groundwater_options import GroundwaterState,groundwater_step
    fx=json.loads((ROOT/"proofs/u2_land/groundwater_oracle.json").read_text());cases=fx["cases"];p=fx["parameters"]
    values=engine.asarray([[c[k] for k in ("dt","fcrmax")] for c in cases],engine.float32).T[:,None,:]
    initial=engine.asarray([c["state"] for c in cases],engine.float32).T[:,None,:]
    arrays=engine.asarray([[c[k] for k in ("sh2o","sice","wcnd")] for c in cases],engine.float32).transpose(1,2,0)[:,:,None,:]
    params=engine.asarray([p[k] for k in ("bexp","smcmax","psisat","dksat")],engine.float32)[:,:,None,None]
    zsoil=engine.asarray(fx["zsoil"],engine.float32)[:,None,None]
    timean=engine.asarray(p["timean"],engine.float32)
    def compute(initial,arrays,values,params,zsoil,timean):
        state=GroundwaterState(*initial);sh2o,sice,wcnd=arrays;dt,fcr=values;bexp,smcmax,psisat,dksat=params
        out=[]
        for _ in range(fx["steps"]):
            state,sh2o,qin,qdis=groundwater_step(state,sh2o,sice,zsoil,wcnd,fcr,bexp,smcmax,psisat,dksat,timean,dt)
            out.append(engine.concatenate((engine.stack(state),sh2o,qin[None,...],qdis[None,...]),axis=0)[:,0,:].T)
        return engine.stack(out,axis=1)
    args=(initial,arrays,values,params,zsoil,timean)
    out=jax.jit(compute)(*args) if engine is jnp else compute(*args)
    return np.asarray(out),cases,fx


def test_original_groundwater_runtime_trajectory():
    actual,cases,fx=matrix()
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c["expected"] for c in cases],**fx["nominal_tolerance"])
