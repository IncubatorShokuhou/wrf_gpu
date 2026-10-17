"""New soil stress/resistance choices against staged pristine ENERGY blocks."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.noahmp.surface_options import soil_moisture_stress,evaporation_resistance

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("btr",[1,2,3])
@pytest.mark.parametrize("rsf",[1,2,3,4])
def test_pristine_energy_branches_runtime_inputs(btr,rsf):
    fx=json.loads((ROOT/"proofs/u2_land/surface_oracle.json").read_text())
    cases=[case for case in fx["cases"] if case["opt_btr"]==btr and case["opt_rsf"]==rsf]
    p=fx["parameters"]
    def compute(sh2o,fsno):
        gx=soil_moisture_stress(sh2o,*[jnp.asarray(p[key],jnp.float32)[:,None]
            for key in ("smcwlt","smcref","smcmax","psisat","bexp")],opt_btr=btr)
        resistance=evaporation_resistance(sh2o[0],p["smcmax"][0],p["smcwlt"][0],p["bexp"][0],fx["zsoil"][0],p["rsurf_exp"],fsno,
                                          opt_rsf=rsf,rsurf_snow=p["rsurf_snow"])
        return jnp.concatenate((gx,resistance[None,:]),axis=0).T
    sh=jnp.asarray([case["sh2o"] for case in cases],jnp.float32).T
    snow=jnp.asarray([case["fsno"] for case in cases],jnp.float32)
    actual=np.asarray(jax.jit(compute)(sh,snow))
    expected=np.asarray([case["expected"] for case in cases])
    assert actual.dtype==np.float32 and np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual,expected,**fx["nominal_tolerance"])


def test_nondefault_options_are_bound_and_defaults_stay_absent():
    from gpuwrf.io.noahmp_options import selected_options,option_refusals
    selected=selected_options({"noah_mp":{"opt_btr":3,"opt_rsf":4}})
    assert selected.opt_btr==3 and selected.opt_rsf==4 and selected.rsurf_snow==50.
    assert selected_options({"noah_mp":{"opt_btr":1,"opt_rsf":1}}) is None
    assert not option_refusals({"noah_mp":{"opt_btr":3,"opt_rsf":4}})
    assert option_refusals({"noah_mp":{"opt_btr":4}})
