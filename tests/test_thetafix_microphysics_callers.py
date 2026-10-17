"""Two actual Swiss caller steps against pristine WRF scheme+prep/finish.

The seam gate injects independently frozen WRF solver results and verifies the
real adapter's input/output mapping, separately from column-kernel accuracy.
Both steps advance the State produced by the preceding call (E85).
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling import scan_adapters as adapters

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT/"proofs/t_thetafix"
SCHEMES = {1:"kessler",3:"wsm3",4:"wsm5",6:"wsm6",10:"morrison",13:"sbu_ylin",14:"wdm5",97:"goddard"}
FIELDS = ("theta","qv","qc","qr","qi","qs","qg","Ni","Ns","Nr","Ng","Nn","Nc")


def assert_real_theta(actual,expected,label):
    """REAL4 boundary rounding, including the retained fp64 dispatcher island.

    Use two representable REAL4 spacings, rather than the rounded decimal
    6e-5: at 256--512 K two spacings are 6.103515625e-5 K. The independently
    compiled original REAL4/REAL8 callers show that exact two-ULP spread.
    """
    actual,expected = np.asarray(actual),np.asarray(expected)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(),label
    limit = 2*np.spacing(expected.astype(np.float32)).astype(np.float64)
    error = np.abs(actual.astype(np.float64)-expected)
    assert np.all(error <= limit), f"{label}: max error={error.max()}, max REAL4 bound={limit.max()}"


@pytest.fixture(scope="module")
def fixture():
    path = PROOF/"swiss_two_step.npz"
    manifest = json.loads((PROOF/"manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    assert manifest["steps"] == 2 and manifest["nz"] == 44
    with np.load(path) as data:
        yield {k: data[k] for k in data.files}


def state_for(data, option):
    from types import SimpleNamespace
    ncol,nz = data["input_theta"].shape
    grid = SimpleNamespace(nz=nz,ny=1,nx=ncol)
    state = State(**{n:jnp.zeros(shape,DEFAULT_DTYPES.dtype_for(n))
                     for n,shape in _state_field_shapes(grid,mp_physics=option).items()})
    back = lambda a: jnp.asarray(a.T[:,None,:],jnp.float32)
    updates = {n:back(data["input_"+n]) for n in FIELDS if getattr(state,n) is not None}
    updates.update(p_total=back(data["input_p"]),ph_total=back(data["input_ph"]),
                   w=back(data["input_w"]),xland=jnp.ones((1,ncol),jnp.float32))
    return state.replace(**updates)


@pytest.mark.parametrize("option",SCHEMES)
@pytest.mark.parametrize("through_dispatch",(False,True),ids=("direct","dispatch"))
def test_two_step_pristine_swiss_caller(monkeypatch,fixture,option,through_dispatch):
    """Original scheme outputs + original REAL4 moist finish, all eight paths."""
    state = state_for(fixture,option)
    reference = fixture[f"mp{option}_out"]
    calls = []
    riming_seam_outputs = []
    def original_wrf_solver(*args):
        step = len(calls)
        assert step < 2
        calls.append(step)
        expected = reference[step]
        assert_real_theta(args[0],expected[:,13,:],f"mp{option}: dry TH_PHY input, step{step+1}")
        np.testing.assert_allclose(args[1],state.qv[:,0,:].T,rtol=0,atol=0)
        replacements = {n:jnp.asarray(expected[:,i,:],jnp.float32)
                        for i,n in enumerate(FIELDS) if getattr(state,n) is not None}
        replacements["theta"] = jnp.asarray(expected[:,14,:],jnp.float32)
        # WDM5's CCN output is a diagnostic threaded into the State by its caller.
        diagnostics = {"Nn":replacements.pop("Nn")} if option==14 else {}
        if option == 13:
            # Mapping-only sentinel for the newly consumed output channel.
            # Thermal/mass references remain the frozen pristine WRF arrays;
            # these distinct riming values make no WRF-fidelity claim.
            riming = np.linspace(.01 + .01*step, .98 + .01*step,
                expected[:,13,:].size, dtype=np.float32).reshape(expected[:,13,:].shape)
            riming_seam_outputs.append(riming)
            diagnostics["ri3d"] = jnp.asarray(riming)
        return PhysicsTendency(state_replacements=replacements,diagnostics=diagnostics)
    tendency = "morrison_tendency" if option==10 else SCHEMES[option]+"_physics_tendency"
    monkeypatch.setattr(adapters,tendency,original_wrf_solver)
    for step in range(2):
        run = adapters.MP_SCAN_ADAPTERS[option] if through_dispatch else getattr(adapters,SCHEMES[option]+"_adapter")
        state = run(state,18.)
        if option == 13:
            np.testing.assert_array_equal(state.rimi,
                riming_seam_outputs[step].T[:,None,:])
        for i,name in enumerate(FIELDS):
            actual = getattr(state,name)
            if actual is None:
                continue
            expected = reference[step,:,i,:].T[:,None,:]
            assert np.isfinite(actual).all() and np.isfinite(expected).all(),name
            if name == "theta":
                assert_real_theta(actual,expected,f"mp{option} {name} step{step+1}")
            else:
                np.testing.assert_array_equal(actual,expected,err_msg=f"mp{option} {name} step{step+1}")
        assert state.theta.dtype == jnp.float32
    assert calls == [0,1]


@pytest.mark.parametrize("option",SCHEMES)
def test_real_swiss_inputs_exercise_moist_theta(fixture,option):
    reference = fixture[f"mp{option}_out"]
    assert np.isfinite(reference).all()
    assert np.max(np.abs(fixture["input_theta"]-reference[0,:,13,:])) > .1
    assert np.max(np.abs(reference[1,:,0,:]-reference[1,:,14,:])) > .1
    # All schemes actually change moisture or dry temperature on these columns.
    assert np.max(np.abs(reference[0,:,14,:]-reference[0,:,13,:])) > 1.e-4


@pytest.mark.parametrize("option",SCHEMES)
def test_two_step_real_dispatcher_thermal_and_mass(fixture,option):
    """Execute the production adapter and its real JAX solver for both calls.

    This caller gate uses the existing Kessler REAL4 absolute bands (2e-3 K,
    1e-7 kg/kg, 2e-4 mm), at least as strict as the existing WSM6, Morrison,
    YLin, WDM5 and Goddard REAL4 physical bands. It does not replace any
    isolated solver's fp64 machine-precision gate. Number concentrations are
    explicitly outside this thermal/mass gate; all outputs must be finite.
    Original WDM5's REAL4/REAL8 number spread is preserved in manifest_fp64.
    """
    state = state_for(fixture,option)
    run = jax.jit(lambda s: adapters.MP_SCAN_ADAPTERS[option](s,18.))
    for step in range(2):
        state = run(state)
        assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(state))
        ref = fixture[f"mp{option}_out"][step]
        for i,name in enumerate(FIELDS[:7]):
            actual = getattr(state,name)
            if actual is None:
                continue
            np.testing.assert_allclose(np.asarray(actual)[:,0,:].T,ref[:,i,:],rtol=0,
                atol=2.e-3 if name=="theta" else 1.e-7,err_msg=f"mp{option} {name} step{step+1}")
        total = sum(np.asarray(getattr(state,name))[0] for name in
                    ("rain_acc","snow_acc","graupel_acc","ice_acc","hail_acc")
                    if getattr(state,name) is not None)
        np.testing.assert_allclose(total,fixture[f"mp{option}_precip"][step,:,0],rtol=0,atol=2.e-4)
    jax.clear_caches()
