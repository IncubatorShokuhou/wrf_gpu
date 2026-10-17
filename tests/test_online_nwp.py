"""Independent pristine-WRF operator fixture, resets and active hail/helicity."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.diagnostics.nwp import (
    NwpDiagnostics,helicity_metrics,history_fields,update_helicity,update_nwp,
    validate_restart_nwp,
)

FIXTURE=Path(__file__).resolve().parents[1]/"proofs/u2_diag/nwp_oracle.json"


def oracle():
    fixture=json.loads(FIXTURE.read_text())
    assert fixture["precision"] == "WRF REAL=4, DOUBLE=8"
    assert all(len(value)==64 for value in fixture["sources"].values())
    operands={name:jnp.asarray(value,jnp.float32) for name,value in fixture["inputs"].items()}
    state=NwpDiagnostics(*(jnp.asarray(a,jnp.float32) for a in fixture["initial"]))
    return fixture,operands,state


def helicity_arguments(inputs):
    names=("u","v","w","ph","phb","msfux","msfvy","terrain","dn","dnw","fnm","fnp","zx","zy","rdzw")
    result={name:inputs[name] for name in names}
    result.update(dict(zip(("rdx","rdy","cf1","cf2","cf3"),inputs["constants"])))
    return result


def nwp_arguments(fixture,inputs):
    names=("u10","v10","w","p8w","znw","qg","rho","ph","phb","refl_10cm")
    result={name:inputs[name] for name in names}
    result.update(dt=fixture["dt"],history_interval=fixture["history_interval"])
    return result


def test_pristine_sequential_helicity_nwp_hail():
    fixture,inputs,state=oracle()
    heli=jax.jit(lambda d:update_helicity(d,**helicity_arguments(inputs)))
    nwp=jax.jit(lambda d,step:update_nwp(d,itimestep=step,**nwp_arguments(fixture,inputs)))
    expected=np.asarray(fixture["expected"],np.float32)
    assert expected[:,0].max()>.1 and expected[:,9:11].max()>.01
    for i,step in enumerate(fixture["itimestep"]):
        state=heli(state)
        np.testing.assert_allclose(state.up_heli_max,expected[i,0],rtol=5e-6,atol=1e-6)
        np.testing.assert_allclose(state.uh,expected[i,1],rtol=5e-6,atol=1e-6)
        state=nwp(state,jnp.int32(step))
        actual=np.asarray(state)
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        for index in (0,1,2,4,5,6,7,8):
            np.testing.assert_array_max_ulp(actual[index],expected[i,index+2],maxulp=4)
        for index in (3,9):
            np.testing.assert_allclose(actual[index],expected[i,index+2],rtol=5e-6,atol=1e-6)
    # Step1/4/7 reset happens AFTER helicity has run, not at its entry.
    assert np.all(expected[[0,3,6],5] == 0)
    assert np.any(expected[2,5] > 0)
    assert np.all(expected[:,4] <= 0)  # W_DN_MAX remains negative, not speed.


def test_pristine_split_geopotential_metrics():
    _,inputs,_=oracle()
    zx,zy,rdzw=helicity_metrics(inputs["ph"],inputs["phb"],rdx=inputs["constants"][0],rdy=inputs["constants"][1])
    for actual,name in ((zx,"zx"),(zy,"zy"),(rdzw,"rdzw")):
        np.testing.assert_array_max_ulp(actual,inputs[name],maxulp=0)


def test_nwp_history_and_restart_contract():
    _,_,state=oracle()
    fields=history_fields(state)
    assert tuple(fields)==tuple(name.upper() for name in NwpDiagnostics._fields[:-1])
    assert "UH" not in fields and history_fields(None)=={}
    validate_restart_nwp(state,True,1,state.uh.shape)
    validate_restart_nwp(None,False,0)
    for damaged,enabled,version,shape in ((None,True,1,None),(state,True,0,None),
            (state._replace(uh=None),True,1,None),(state._replace(w_mean=state.w_mean.astype(jnp.float64)),True,1,None),
            (state,True,1,(3,5))):
        with pytest.raises(ValueError,match="NWP diagnostics.*E78"):
            validate_restart_nwp(damaged,enabled,version,shape)


def test_nwp_invalid_cadence_refused():
    fixture,inputs,state=oracle()
    args=nwp_arguments(fixture,inputs)
    with pytest.raises(ValueError,match="history interval"):
        update_nwp(state,itimestep=1,**{**args,"history_interval":.01})

