"""Liu crop against original WRF 4.7.1 multi-step trajectories."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def matrix(engine=jnp):
    from gpuwrf.physics.noahmp.crop import CropCarbonState,CropParameters,crop_flux
    fx=json.loads((ROOT/'proofs/u2_land/crop_oracle.json').read_text());cases=fx['cases']
    values=engine.asarray([[c[k] for k in ('dt','tv','t2m','julian','psn','btran','foln','lai','sai')] for c in cases],engine.float32)
    state_values=engine.asarray([c['state'] for c in cases],engine.float32)
    params=engine.asarray([[c['parameters'][k] for k in fx['scalar_keys']] for c in cases],engine.float32).T
    arrays=engine.asarray([[c['parameters'][k] for k in fx['array_keys']] for c in cases],engine.float32).transpose(1,2,0)
    def compute(values,state_values,params,arrays):
        dt,tv,t2m,julian,psn,btran,foln,lai,sai=values.T
        p=CropParameters(*params,*arrays)
        state=CropCarbonState(*state_values.T,engine.zeros_like(tv,dtype=engine.int32))
        # CARBON_CROP:10400-10402 sequential root weighting, with fixture's
        # actual identical three root layers and 1 m root depth.
        wroot=engine.zeros_like(tv)
        for dz in (.1,.3,.6):wroot=wroot+engine.asarray(.2,engine.float32)/engine.asarray(.45,engine.float32)*engine.asarray(dz,engine.float32)/engine.asarray(1.,engine.float32)
        outputs=[]
        for _ in range(fx['steps']):
            state,lai,sai,diag=crop_flux(state,p,lai,sai,psn,btran,foln,tv,tv,wroot,t2m,julian,dt)
            outputs.append(engine.stack((*state[:8],state.pgs.astype(engine.float32),lai,sai,*diag.values()),axis=-1))
        return engine.stack(outputs,axis=1)
    inputs=(values,state_values,params,arrays)
    return np.asarray(jax.jit(compute)(*inputs) if engine is jnp else compute(*inputs)),cases,fx


def test_original_liu_crop_runtime_trajectories():
    actual,cases,fx=matrix()
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c['expected'] for c in cases],**fx['nominal_tolerance'])
