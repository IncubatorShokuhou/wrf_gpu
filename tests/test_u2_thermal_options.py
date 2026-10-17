"""Original WRF thermal system and nonlinear freezing branch on runtime inputs."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def matrix(name,tbot=2,stc=1,engine=jnp,literal=False):
    from gpuwrf.physics.noahmp.soil_thermo import _hrt,noahmp_soil_thermo
    from gpuwrf.physics.noahmp.surface_options import supercooled_water_koren
    fx=json.loads((ROOT/"proofs/u2_land/thermal_oracle.json").read_text())["groups"][name]
    cases=fx["cases"]
    if name=="freezing":
        values=engine.asarray([[c[k] for k in ("temperature","smc","sh2o","bexp","smcmax","psisat")] for c in cases],engine.float32)
        def compute(values):return supercooled_water_koren(*values.T)[:,None]
        actual=(jax.jit(compute)(values) if engine is jnp else compute(values))
    else:
        cases=[c for c in cases if (c["opt_tbot"],c["opt_stc"])==(tbot,stc)]
        scalars=engine.asarray([[c[k] for k in ("ssoil","tbot","zbot","dt")] for c in cases],engine.float32).T[:,None,:]
        arrays=engine.asarray([[c[k] for k in ("stc","df","hcpct","zsnso")] for c in cases],engine.float32).transpose(1,2,0)[:,:,None,:]
        isnow=engine.asarray([[c["isnow"] for c in cases]],engine.int32)
        def compute(arrays,scalars):
            temperature,df,hcpct,zsnso=arrays
            ssoil,deep,zbot,dt=scalars
            coefficients=_hrt(temperature,deep,zbot,df,hcpct,ssoil,zsnso,isnow,opt_tbot=tbot,opt_stc=stc)
            if literal:return engine.concatenate(coefficients,axis=0)[:,0,:].T
            dz=engine.concatenate((engine.zeros_like(zsnso[:1]),zsnso[:-1]),axis=0)-zsnso
            changed=noahmp_soil_thermo(temperature,df,hcpct,ssoil,deep,zsnso,dz,isnow,dt,opt_tbot=tbot,opt_stc=stc)
            return engine.concatenate((*coefficients,changed),axis=0)[:,0,:].T
        actual=jax.jit(compute)(arrays,scalars) if engine is jnp else compute(arrays,scalars)
    return np.asarray(actual),cases,fx


@pytest.mark.parametrize("tbot",[1,2])
@pytest.mark.parametrize("stc",[1,2,3])
def test_original_thermal_system_runtime_operands(monkeypatch,tbot,stc):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL","1")
    jax.clear_caches()
    actual,cases,fx=matrix("thermal",tbot,stc)
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c["expected"] for c in cases],**fx["nominal_tolerance"])


def test_original_frh2o_runtime_operands():
    actual,cases,fx=matrix("freezing")
    assert actual.dtype==np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual,[c["expected"] for c in cases],**fx["nominal_tolerance"])


def test_thermal_options_admission():
    from gpuwrf.io.noahmp_options import selected_options,option_refusals
    opts=selected_options({"noah_mp":{"opt_tbot":1,"opt_stc":3,"opt_frz":2}})
    assert (opts.opt_tbot,opts.opt_stc,opts.opt_frz)==(1,3,2)
    assert not option_refusals({"noah_mp":{"opt_tbot":1,"opt_stc":2,"opt_frz":2}})
