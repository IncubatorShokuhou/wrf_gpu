"""MP53 must use the scoped memory path, without changing other variants."""
import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl

import gpuwrf.backends.pallas as facade
from gpuwrf.kernels.phys_p3 import full_column
from gpuwrf.physics.p3.adapter import output_mapping
from gpuwrf.physics.p3.process_support import _expand, MP53_PREDICATE_RESHAPE


def test_predicate_singleton_reshape_is_scoped_and_exact():
    import numpy as np
    values = np.array([True, False, True])
    control = jax.make_jaxpr(lambda x: _expand(x, 2))(values)
    token = MP53_PREDICATE_RESHAPE.set(True)
    try:
        program = jax.make_jaxpr(lambda x: _expand(x, 2))(values)
        np.testing.assert_array_equal(_expand(values, 2), values[:, None, None])
        assert [e.primitive.name for e in program.jaxpr.eqns] == ['reshape']
        # Integer table indices retain the established lowering.
        ints = jax.make_jaxpr(lambda x: _expand(x, 1))(values.astype(np.int32))
        assert [e.primitive.name for e in ints.jaxpr.eqns] == ['broadcast_in_dim']
    finally:
        MP53_PREDICATE_RESHAPE.reset(token)
    assert [e.primitive.name for e in control.jaxpr.eqns] == ['broadcast_in_dim']
    assert not MP53_PREDICATE_RESHAPE.get()


@pytest.mark.parametrize('mp,scoped,warps', [(50, False, 2), (51, False, 2),
                                           (52, True, 4), (53, True, 4)])
def test_variant_memory_and_launch_geometry(monkeypatch, mp, scoped, warps):
    monkeypatch.setattr(facade, 'cuda_native', lambda: True)
    monkeypatch.setattr(facade, 'pallas_call', pl.pallas_call)
    names = ['qv', 'qc', 'qr', 'qi', 'ni', 'nr', 'nc', 'qir', 'bir',
             'th_old', 'qv_old', 'th', 'p', 'dz']
    if mp == 52:
        names += ['qi2', 'ni2', 'qir2', 'bir2']
    if mp == 53:
        names += ['qzi']
    fields = {n: jax.ShapeDtypeStruct((1, 44), jnp.float32) for n in names}
    outputs = (*output_mapping(mp), 'th', 'rainncv', 'snowncv', 'ze')
    with jax.enable_x64(True):
        program = jax.make_jaxpr(lambda x: full_column(
            x, jnp.float32(18), jnp.int32(1), mp=mp, output_names=outputs))(fields)
    call = next(e for e in program.jaxpr.eqns if e.primitive.name == 'pallas_call')
    primitives = {e.primitive.name for e in call.params['jaxpr'].eqns}
    assert ('p3_global_load' in primitives) is scoped
    assert ('p3_global_store' in primitives) is scoped
    assert call.params['compiler_params'].num_warps == warps
