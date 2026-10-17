"""Original CANRES and twenty SFCDIF2 caller sweeps, runtime operands."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


def matrix(name,engine=jnp):
    from gpuwrf.physics.noahmp.canopy_options import jarvis_resistance,sfcdif2
    fx=json.loads((ROOT/"proofs/u2_land/canopy_oracle.json").read_text())["groups"][name]
    cases=fx["cases"]
    values=engine.asarray([[c[k] for k in fx["fields"]] for c in cases],engine.float32)
    snow=engine.asarray([c.get("snow",0) for c in cases],engine.int32)
    def compute(values):
        if name=="jarvis":return engine.stack(jarvis_resistance(*values.T),axis=-1)
        z0,thz0,thlm,speed,zlm,akms,akhs,rlmo,wstar,ustar,czil=values.T
        outputs=[]
        for it in range(1,21):
            akms,akhs,rlmo,wstar,ustar=sfcdif2(it,z0,thz0,thlm,speed,zlm,akms,akhs,rlmo,wstar,ustar,czil)
            akms,akhs=akms/speed,akhs/speed
            akms=engine.where(snow==1,engine.minimum(.01,akms),akms)
            akhs=engine.where(snow==1,engine.minimum(.01,akhs),akhs)
            outputs.append(engine.stack((akms,akhs,rlmo,wstar,ustar),axis=-1))
        return engine.concatenate(outputs,axis=-1)
    return np.asarray(jax.jit(compute)(values) if engine is jnp else compute(values)),cases,fx


@pytest.mark.parametrize("name",["sfcdif2","jarvis"])
def test_pristine_canopy_runtime_operands(name):
    actual,cases,fx=matrix(name)
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c["expected"] for c in cases],**fx["nominal_tolerance"])


def test_new_canopy_choices_are_bound():
    from gpuwrf.io.noahmp_options import selected_options,option_refusals
    selected=selected_options({"noah_mp":{"opt_crs":2,"opt_sfc":2}})
    assert selected.opt_crs==selected.opt_sfc==2
    assert not option_refusals({"noah_mp":{"opt_crs":2,"opt_sfc":2}})
