"""WDM5/7 use the reviewed WDM6 Nn column-view firewall (c10/c18).

Whole-program CUDA compilation is measured separately: CPU identity here is
regression evidence, not a pristine-WRF or GPU fidelity claim.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.kernels import layout_pin
from test_t_cfix_cu_wdm6_small_grid_firewall import _barriers, _stub
from test_t_wsmdz_wsm_callers import _prep_case


SCHEMES = ((14, "wdm5", 6), (26, "wdm7", 8))


@pytest.mark.parametrize("mp,name,nn_index", SCHEMES)
def test_only_nn_column_view_is_fenced(monkeypatch, mp, name, nn_index):
    grid, state = _prep_case(2, 3, jnp.float32, mp)
    nn = jnp.arange(state.qv.size, dtype=jnp.float32).reshape(state.qv.shape) * 1e6 + 1e8
    state = state.replace(_cast=False, Nn=nn)
    seen, fences = {}, []

    def spy(value, shape):
        out = layout_pin.small_grid_firewall(value, shape)
        fences.append((tuple(shape), out))
        return out

    monkeypatch.setattr(sa, name + "_physics_tendency", _stub(seen))
    monkeypatch.setattr(sa, "_small_grid_firewall", spy)
    adapter = getattr(sa, name + "_adapter")
    traced = jax.make_jaxpr(lambda s: adapter(s, 18.0, grid))(state)
    assert len(fences) == 1 and fences[0][0] == (2, 3)
    assert seen["args"][nn_index] is fences[0][1]
    assert _barriers(traced.jaxpr) == 1
    adapter(state, 18.0, grid)
    expected = np.moveaxis(np.asarray(nn), 0, -1).reshape(6, 3)
    np.testing.assert_array_equal(seen["args"][nn_index], expected)


@pytest.mark.parametrize("mp,name,nn_index", SCHEMES)
def test_large_grid_program_is_unchanged(monkeypatch, mp, name, nn_index):
    assert 64 * 65 > layout_pin.SMALL_GRID_COLUMNS
    grid, state = _prep_case(64, 65, jnp.float32, mp)
    monkeypatch.setattr(sa, name + "_physics_tendency", _stub({}))
    adapter = getattr(sa, name + "_adapter")
    fenced = jax.make_jaxpr(lambda s: adapter(s, 18.0, grid))(state)
    monkeypatch.setattr(sa, "_small_grid_firewall", lambda v, shape: v)
    plain = jax.make_jaxpr(lambda s: adapter(s, 18.0, grid))(state)
    assert _barriers(fenced.jaxpr) == 0
    assert str(fenced) == str(plain)


@pytest.mark.parametrize("mp,name,nn_index", SCHEMES)
@pytest.mark.parametrize("return_precipitation", (False, True))
@pytest.mark.parametrize("use_mp_re", (0, 1), ids=("radii-off", "radii-on"))
def test_cpu_outputs_bitwise_equal(monkeypatch, mp, name, nn_index, return_precipitation, use_mp_re):
    grid, state = _prep_case(2, 3, jnp.float32, mp)
    state = state.ensure_conditional_leaves(mp_physics=mp, use_mp_re=use_mp_re)
    # Nonuniform CCN, warm/cold levels and active cloud/rain avoid an inert
    # comparison. The WDM7 hail leaf is allocated by the real State contract.
    state = state.replace(
        _cast=False, theta=jnp.broadcast_to(jnp.array([310., 285., 265.], dtype=jnp.float32)[:, None, None], state.theta.shape),
        qc=jnp.full_like(state.qv, 2e-4), qr=jnp.full_like(state.qv, 5e-5),
        Nn=jnp.arange(state.qv.size, dtype=jnp.float32).reshape(state.qv.shape) * 1e6 + 1e8,
        Nc=jnp.full_like(state.qv, 1e8), Nr=jnp.full_like(state.qv, 1e3))
    assert state.theta.dtype == state.qv.dtype == jnp.float32
    # Use the actual operational dispatch, including its documented fp64 island
    # for these opt-in schemes. The State enters and leaves with REAL32 carry.
    adapter = sa.MP_SCAN_ADAPTERS[mp]
    assert adapter.__wrapped__ is getattr(sa, name + "_adapter")
    fenced = jax.jit(lambda s: adapter(s, 18., grid, return_precipitation=return_precipitation))(state)
    monkeypatch.setattr(sa, "_small_grid_firewall", lambda v, shape: v)
    plain = jax.jit(lambda s: adapter(s, 18., grid, return_precipitation=return_precipitation))(state)
    left, tree = jax.tree.flatten(fenced)
    right, other_tree = jax.tree.flatten(plain)
    assert tree == other_tree
    for a, b in zip(left, right):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    result = fenced[0] if return_precipitation else fenced
    assert result.theta.dtype == result.qv.dtype == jnp.float32
    assert np.any(np.asarray(result.qc) != np.asarray(state.qc)), "microphysics must act"
