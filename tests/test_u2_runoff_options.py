"""Original runoff family with runtime operands, including persistent FACC."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def matrix(run,infdv,engine=jnp):
    from gpuwrf.physics.noahmp.runoff_options import RunoffParameters,vic_surface,xaj_surface,dynamic_vic
    fx=json.loads((ROOT/'proofs/u2_land/runoff_oracle.json').read_text())
    cases=[c for c in fx['cases'] if (c['opt_run'],c['opt_infdv'])==(run,infdv)]
    values=engine.asarray([[c[k] for k in ('dt','qinsur')] for c in cases],engine.float32).T
    profiles=tuple(engine.asarray([c[k] for c in cases],engine.float32).T for k in ('smc','sh2o','fcr'))
    p={k.lower():engine.asarray([c['parameters'][k] for c in cases],engine.float32).T for k in fx['parameter_fields']}
    zsoil=engine.asarray([-.1,-.4,-1.,-2.],engine.float32)[:,None]
    def compute(values,smc,sh2o,fcr,p,zsoil):
        dt,qin=values;facc=engine.full_like(qin,1.e-6)
        params=RunoffParameters(*(p[k] for k in RunoffParameters._fields))
        outputs=[]
        for _ in range(fx['steps']):
            if run==6:inf,ro=vic_surface(params,smc,p['smcmax'],zsoil,qin,dt)
            elif run==7:inf,ro=xaj_surface(params,smc,p['smcmax'],p['smcref'],zsoil,fcr,qin,dt)
            else:inf,ro,facc=dynamic_vic(params,smc,sh2o,smc-sh2o,zsoil,p['smcmax'],p['bexp'],p['dksat'],p['dwsat'],qin,dt,facc,opt_infdv=infdv)
            outputs.append(engine.stack((inf,ro,facc),axis=-1))
        return engine.stack(outputs,axis=1)
    inputs=(values,*profiles,p,zsoil)
    return np.asarray(jax.jit(compute)(*inputs) if engine is jnp else compute(*inputs)),cases,fx


def test_original_runtime_runoff_family():
    for run,infdv in ((6,0),(7,0),(8,1),(8,2),(8,3)):
        actual,cases,fx=matrix(run,infdv)
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        np.testing.assert_allclose(actual,[c['expected'] for c in cases],**fx['nominal_tolerance'])
