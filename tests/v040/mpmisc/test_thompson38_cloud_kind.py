"""Original active cloud savepoint and declared DOUBLE slope contract."""
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_hail import native


def test_original_cloud_velocity_and_double_slope():
    root=Path(__file__).resolve().parents[3]
    original=json.loads((root/"proofs/v040/mpmisc/thompson38/original_cloud_sed_kind.json").read_text())["original"]
    coeff=SimpleNamespace(**{name:jnp.asarray(value) for name,value in native._cloud_coefficients().items()})
    state=SimpleNamespace(rho=jnp.ones((1,),jnp.float32),qc=jnp.asarray([original["rc(k)"]],jnp.float32),
                          Nc=jnp.asarray([original["nc(k)"]],jnp.float32),w=jnp.zeros((1,),jnp.float32))
    fn=native.AERO._cloud_water_fall_speeds_aero
    env=dict(fn.__globals__,_rho_correction=lambda _:jnp.asarray([original["rhof(k)"]],jnp.float32))
    frozen=FunctionType(fn.__code__,env,fn.__name__,fn.__defaults__,fn.__closure__)
    frozen.__kwdefaults__=fn.__kwdefaults__
    with jax.disable_jit():
        mass,number=frozen(state,coeff,wrf_hail=True)
    np.testing.assert_array_equal(np.asarray(mass),np.asarray([original["vtck(k)"]],np.float32))
    np.testing.assert_array_equal(np.asarray(number),np.asarray([original["vtc"]],np.float32))
    assert mass.dtype==number.dtype==jnp.float32
    def has_original_double_inverse(enabled):
        graph=jax.make_jaxpr(lambda q,n:frozen(SimpleNamespace(**dict(vars(state),qc=q,Nc=n)),coeff,wrf_hail=enabled))(state.qc,state.Nc)
        return any(eq.primitive.name=="div" and any(getattr(v.aval,"dtype",None)==np.dtype("float64") for v in eq.outvars) for eq in graph.jaxpr.eqns)
    assert has_original_double_inverse(True)
    assert not has_original_double_inverse(False)  # Original missing-island mutant.
