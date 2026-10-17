"""The scoped P3 memory lowering retains Pallas CPU reference semantics."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from gpuwrf.kernels import _p3_memory as memory


def _roundtrip(load, store, values, indices, mask):
    def kernel(source, ix, pred, out):
        k = jnp.arange(8, dtype=jnp.int32)
        table_index = plt.load(ix.at[k])
        active = plt.load(pred.at[k])
        # Initialize all slots to distinguish masked stores from zero writes.
        plt.store(out.at[k], jnp.full((8,), -7., jnp.float32))
        value = load(source.at[table_index], mask=active, other=13.)
        store(out.at[k], value, mask=(k % 2) == 0)

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
                          interpret=True)(values, indices, mask)


def test_masked_gather_and_store_keep_exact_bits():
    # Signed zero, subnormals and large magnitudes check that the memory
    # lowering performs no floating-point arithmetic or conversions.
    values = np.array([0x80000000, 0x00000001, 0x3f800001, 0xbf800001,
                       0x00800000, 0x7f000000, 0x00000000, 0x40400000], np.uint32).view(np.float32)
    indices = np.array([7, 6, 0, 1, 4, 2, 5, 3], np.int32)
    mask = np.array([True, False, True, True, False, True, True, False])
    baseline = _roundtrip(plt.load, plt.store, values, indices, mask)
    actual = _roundtrip(memory.load, memory.store, values, indices, mask)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32),
                                  np.asarray(baseline).view(np.uint32))
    assert np.asarray(actual)[4] == 13.
    np.testing.assert_array_equal(np.asarray(actual)[1::2], np.full(4, -7., np.float32))


def test_scalar_integer_load_and_float_store():
    def kernel(source, out):
        value = memory.load(source.at[()])
        memory.store(out.at[()], value.astype(jnp.float32))

    call = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((), jnp.float32),
                          interpret=True)
    assert np.asarray(call(np.int32(19))).tobytes() == np.float32(19).tobytes()
