"""Dynamic carbon model against original CARBON/CO2FLUX multistep trajectories."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def matrix(engine=jnp):
    from gpuwrf.physics.noahmp.carbon import CarbonState,CarbonParameters,carbon_flux,root_zone_water
    fx=json.loads((ROOT/"proofs/u2_land/carbon_oracle.json").read_text());cases=fx["cases"]
    values=engine.asarray([[c[k] for k in ("dt","tv","psn","btran","foln","lai","sai")] for c in cases],engine.float32)
    state_values=engine.asarray([c["state"] for c in cases],engine.float32)
    params=engine.asarray([[c["parameters"][k] for k in fx["parameter_keys"]] for c in cases],engine.float32)
    smc=engine.asarray([c["smc"] for c in cases],engine.float32).T
    smcmax=engine.asarray([c["smcmax"] for c in cases],engine.float32).T
    stc=engine.asarray([c["stc"] for c in cases],engine.float32).T
    nroot=engine.asarray([c["nroot"] for c in cases],engine.int32)
    veg=engine.asarray([c["veg"] for c in cases],engine.int32)
    zsoil=engine.asarray([-.1,-.4,-1.,-2.],engine.float32)[:,None]
    dz=engine.asarray([.1,.3,.6,1.],engine.float32)[:,None]
    def compute(values,state_values,params,smc,smcmax,stc,nroot,veg,zsoil,dz):
        dt,tv,psn,btran,foln,lai,sai=values.T
        p=CarbonParameters(*params.T);state=CarbonState(*state_values.T)
        no_veg=(veg==13)|(veg==15)|(veg==16)|(veg==17)
        wroot=root_zone_water(smc,smcmax,dz,zsoil,nroot,no_veg)
        igs=engine.where(tv>p.tmin,engine.ones_like(tv),engine.zeros_like(tv))
        outputs=[]
        for _ in range(fx["steps"]):
            state,lai,sai,diag=carbon_flux(state,p,lai,sai,psn,btran,foln,tv,stc[3],wroot,igs,veg==2,no_veg,dt)
            outputs.append(engine.stack((*state,lai,sai,*diag.values()),axis=-1))
        return engine.stack(outputs,axis=1)
    inputs=(values,state_values,params,smc,smcmax,stc,nroot,veg,zsoil,dz)
    return np.asarray(jax.jit(compute)(*inputs) if engine is jnp else compute(*inputs)),cases,fx


def test_original_dynamic_carbon_runtime_trajectories():
    actual,cases,fx=matrix()
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c["expected"] for c in cases],**fx["nominal_tolerance"])
