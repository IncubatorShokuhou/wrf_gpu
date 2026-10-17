"""c11 (mp16 WDM6 / cu96 / bl17, Swiss 42x42) cold-compile fix: the WDM6 Nn column view rides one small-grid firewall.

On the c11 program the column-major Nn (root flow_dep fence layout) column-view transpose was the hero of ONE kInput
fusion that absorbed the WDM6 column body (1613 optimized-HLO instructions, 23 outputs; ptxas > 57 min / 25.9 GiB
RSS, v-prep 2026-10-16). Fencing that one operand keeps the transpose its own fusion (proofs/t_cfix_cu/c11_ptxas).
The whole-module fusion effect is only visible in the whole-program compile (E40); these tests pin the wiring:
  * small grids: exactly the Nn argument of wdm6_physics_tendency is the firewall output, and the traced adapter
    gains exactly one optimization_barrier;
  * grids above SMALL_GRID_COLUMNS trace nothing new (no barrier);
  * the Nn operand values equal an independently built column view;
  * the firewall is an identity op: CPU outputs bitwise equal with it removed.
"""
from __future__ import annotations

import jax
import jax.extend
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.kernels import layout_pin
from test_t_wsmdz_wsm_callers import _prep_case


def _barriers(jaxpr):
    count = 0
    for eqn in jaxpr.eqns:
        count += eqn.primitive.name == "optimization_barrier"
        for sub in jax.extend.core.jaxprs_in_params(eqn.params):
            count += _barriers(sub)
    return count


def _stub(seen):
    def tendency(*args):
        seen["args"] = args
        return PhysicsTendency(state_replacements={"theta": args[0]}, accumulator_increments={}, diagnostics={})
    return tendency


def test_small_grid_fences_exactly_the_nn_column_view(monkeypatch):
    grid, state = _prep_case(2, 3, jnp.float32, 16)
    seen, fenced = {}, []
    real = layout_pin.small_grid_firewall

    def spy(value, horizontal_shape):
        out = real(value, horizontal_shape)
        fenced.append((tuple(horizontal_shape), out))
        return out

    monkeypatch.setattr(sa, "wdm6_physics_tendency", _stub(seen))
    monkeypatch.setattr(sa, "_small_grid_firewall", spy)
    jaxpr = jax.make_jaxpr(lambda s: sa.wdm6_adapter(s, 18.0, grid))(state)
    assert [shape for shape, _ in fenced] == [(2, 3)]
    assert seen["args"][7] is fenced[0][1], "the firewall output must be the Nn (8th) WDM6 argument"
    assert _barriers(jaxpr.jaxpr) == 1


def test_nn_argument_values_unchanged(monkeypatch):
    grid, state = _prep_case(2, 3, jnp.float32, 16)
    nn = jnp.arange(np.prod(state.qv.shape), dtype=state.qv.dtype).reshape(state.qv.shape) * 1.0e6 + 1.0e8
    state = state.replace(_cast=False, Nn=nn)
    seen = {}
    monkeypatch.setattr(sa, "wdm6_physics_tendency", _stub(seen))
    sa.wdm6_adapter(state, 18.0, grid)
    nz, ny, nx = state.qv.shape
    expect = np.moveaxis(np.asarray(nn), 0, -1).reshape(ny * nx, nz)  # independent (ncol, nlev) column view
    np.testing.assert_array_equal(np.asarray(seen["args"][7]), expect)


def test_large_grid_traces_no_barrier(monkeypatch):
    ny, nx = 64, 65
    assert ny * nx > layout_pin.SMALL_GRID_COLUMNS
    grid, state = _prep_case(ny, nx, jnp.float32, 16)
    monkeypatch.setattr(sa, "wdm6_physics_tendency", _stub({}))
    jaxpr = jax.make_jaxpr(lambda s: sa.wdm6_adapter(s, 18.0, grid))(state)
    assert _barriers(jaxpr.jaxpr) == 0


def test_firewall_is_value_identity_on_cpu(monkeypatch):
    grid, state = _prep_case(2, 3, jnp.float32, 16)
    state = state.replace(_cast=False, qc=jnp.full_like(state.qv, 2.0e-4), qr=jnp.full_like(state.qv, 5.0e-5),
                          Nn=jnp.full_like(state.qv, 1.0e8), Nc=jnp.full_like(state.qv, 1.0e8),
                          Nr=jnp.full_like(state.qv, 1.0e3))
    fenced = jax.jit(lambda s: sa.wdm6_adapter(s, 18.0, grid))(state)
    monkeypatch.setattr(sa, "_small_grid_firewall", lambda value, horizontal_shape: value)
    plain = jax.jit(lambda s: sa.wdm6_adapter(s, 18.0, grid))(state)
    for a, b in zip(jax.tree.leaves(fenced), jax.tree.leaves(plain)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert float(jnp.max(jnp.abs(fenced.qc - state.qc))) > 0.0, "WDM6 must act on this column"
