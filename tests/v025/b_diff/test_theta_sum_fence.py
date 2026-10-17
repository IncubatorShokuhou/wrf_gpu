"""BD93: output-only fusion fence on the REAL theta stage sum (E41: d01-only program STACK 632 B).

The fence must (a) act only on f32 tendencies (f64 legacy graph untouched), (b) be a pure identity on values, and
(c) stay wired at both cut points of ``_augment_large_step_tendencies`` (after the coupled theta tendency, and on the
rk_addtend_dry result before the root relax_bdy_dry add).  Bitwise neutrality of the compiled GPU programs is a GPU
acceptance (3-nest F1' frames), not claimed here.
"""
import ast
import inspect
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import operational_mode as op


def _barriers(fn, x):
    return sum(e.primitive.name == "optimization_barrier" for e in jax.make_jaxpr(fn)(x).jaxpr.eqns)


def _real_operand(dtype):
    rng = np.random.default_rng(93)
    x = (rng.standard_normal((4, 5, 6)) * 1e-3).astype(dtype)
    assert np.isfinite(x).all()
    return jnp.asarray(x)


def test_fence_is_a_barrier_on_f32_only():
    x32 = _real_operand(np.float32)
    assert _barriers(op._real_theta_sum_fence, x32) == 1
    with jax.enable_x64(True):
        x64 = _real_operand(np.float64)
        assert _barriers(op._real_theta_sum_fence, x64) == 0
        assert op._real_theta_sum_fence(x64) is x64


def _bitwise_equal_finite(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):   # E200: never let NaN/Inf payloads compare equal
        raise ValueError("non-finite operand in a bitwise identity check")
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_fence_values_are_identity_bitwise():
    x = _real_operand(np.float32)
    y = jax.jit(lambda a: op._real_theta_sum_fence(a) + a)(x)
    ref = jax.jit(lambda a: a + a)(x)
    assert _bitwise_equal_finite(y, ref)


def test_fence_wired_at_both_cut_points():
    src = textwrap.dedent(inspect.getsource(op._augment_large_step_tendencies))
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_real_theta_sum_fence"]
    args = sorted(ast.unparse(c.args[0]) for c in calls)
    # th_t before rk_addtend_dry; merged.theta before + bdy_relax.t in both relax branches (with / without the w lane)
    assert args == ["merged.theta", "merged.theta", "th_t"], args


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", ["left", "right", "both"])
def test_identity_check_rejects_non_finite(bad, where):
    x = np.asarray(_real_operand(np.float32)).copy()
    y = x.copy()
    if where in ("left", "both"):
        x[0, 0, 0] = bad
    if where in ("right", "both"):
        y[0, 0, 0] = bad
    with pytest.raises(ValueError):
        _bitwise_equal_finite(x, y)
