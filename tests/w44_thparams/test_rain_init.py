"""Pin original WRF REAL initializer bits, including its approximate WGAMMA."""
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics.thompson_rain_parameters import rain_parameters

# Fresh original wrf_constants.F90 FCOPTIM outputs, AR01 (2427/4854/9708).
REFERENCE = {
    2427.: (1177727968, 1311462459, 1104172975),
    4854.: (1186116576, 1319851067, 1108562439),
    9708.: (1194505184, 1328239675, 1112561583),
}


def test_rain_initializer_original_bits_and_one_value_independent_trace():
    traces=[]
    def evaluate(value):
        traces.append(1)
        return rain_parameters(value)
    fn=jax.jit(evaluate)
    for value,bits in REFERENCE.items():
        expected=np.array(bits,np.uint32)
        eager=np.asarray(rain_parameters(jnp.float32(value)))
        compiled=np.asarray(fn(jnp.float32(value)))
        assert eager.dtype==compiled.dtype==np.float32
        assert np.isfinite(eager).all() and np.isfinite(compiled).all()
        np.testing.assert_array_equal(eager.view(np.uint32),expected)
        error=np.abs(compiled.view(np.uint32).astype(np.int64)-expected.astype(np.int64))
        assert np.all(error <= np.array([1,0,0])),error  # measured scalar constant-association class
    assert traces==[1]
