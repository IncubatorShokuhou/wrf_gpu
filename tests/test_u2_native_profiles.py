"""Byte-level ABI checks, separate from pristine-WRF numerical qualification."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics.noahmp.land_columns import layered_column_call,layer_row,stack_layers


@pytest.mark.parametrize('layers',(4,8))
def test_native_profile_planes_preserve_layer_order_and_bits(layers):
    x=np.arange(layers*6,dtype=np.float32).reshape(layers,2,3)
    x[0,0,0]=-0.
    def identity(value):return stack_layers([layer_row(value,k) for k in range(layers)])
    actual=jax.jit(lambda value:layered_column_call(identity,(value,),grid_shape=(2,3),name='u2_profile_identity'))(jnp.asarray(x))
    assert np.asarray(actual).tobytes()==x.tobytes()
