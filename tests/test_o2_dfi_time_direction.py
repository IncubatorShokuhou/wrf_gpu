"""WRF ``sign(1,time_step)`` upwinding for the TDFI backward stage (gpuwrf.dynamics.time_direction).

WRF's flux statement functions are ``flux5 = flux6 - sign(1,time_step)*sign(1.,ua)*c5`` and
``flux3 = flux4 + sign(1,time_step)*sign(1.,ua)*c3`` (module_advect_em.F).  An advection operator
``T(v, s) = A(v) + s*B(|v|)`` (A odd in the transporting velocity v, B even) therefore satisfies the
exact identity ``T(v, -1) == -T(-v, +1)``: the backward stage equals the forward operator with
reversed velocities, negated.  The identity fails as soon as one stencil site ignores the time-step
sign (deletion-sensitive), and it does not hold without the context (``B != 0`` on rough fields).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from gpuwrf.dynamics.time_direction import time_step_sign, upwind_sign, wrf_time_step_sign  # noqa: E402


def _rng_fields(nz=6, ny=9, nx=11, seed=0):
    rng = np.random.default_rng(seed)
    field = rng.normal(size=(nz, ny, nx)).astype(np.float32) * 3.0 + 280.0
    ru = rng.normal(size=(nz, ny, nx)).astype(np.float32) * 5e4
    rv = rng.normal(size=(nz, ny, nx)).astype(np.float32) * 5e4
    rom = rng.normal(size=(nz + 1, ny, nx)).astype(np.float32) * 2e2
    return field, ru, rv, rom


def test_context_defaults_forward_and_restores():
    assert time_step_sign() == 1
    x = jnp.asarray([-1.0, 0.0, 2.0])
    assert upwind_sign(x) is x  # forward: the very same object, no extra op in any trace
    with wrf_time_step_sign(-18.0):
        assert time_step_sign() == -1
        np.testing.assert_array_equal(upwind_sign(x), -x)
        with wrf_time_step_sign(18.0):
            assert time_step_sign() == 1
        assert time_step_sign() == -1
    assert time_step_sign() == 1


def test_legacy_periodic_flux5_identity():
    from gpuwrf.dynamics.flux_advection import flux5_face_periodic

    field, ru, _rv, _rom = _rng_fields()
    fwd_neg = np.asarray(flux5_face_periodic(jnp.asarray(field), jnp.asarray(-ru), 2))
    plain = np.asarray(flux5_face_periodic(jnp.asarray(field), jnp.asarray(ru), 2))
    with wrf_time_step_sign(-18.0):
        bck = np.asarray(flux5_face_periodic(jnp.asarray(field), jnp.asarray(ru), 2))
    # face value (not times velocity): flux5(q, v, s=-1) == flux5(q, -v, s=+1)
    np.testing.assert_allclose(bck, fwd_neg, rtol=0, atol=1e-4)
    assert np.abs(bck - plain).max() > 1e-3


def test_native_scalar_flux_kernel_identity_interpret():
    from gpuwrf.kernels.dyn_flux_fp32 import advect_scalar_flux_fp32

    field, ru, rv, rom = _rng_fields(seed=1)
    nz, ny, nx = field.shape
    msf = np.ones((ny, nx), np.float32)
    rdzw = np.full((nz,), -25.0, np.float32)
    fzm = np.full((nz + 1,), 0.5, np.float32)
    fzp = np.full((nz + 1,), 0.5, np.float32)

    def call(sign_ru):
        return np.asarray(advect_scalar_flux_fp32(
            jnp.asarray(field), jnp.asarray(sign_ru * ru), jnp.asarray(sign_ru * rv), jnp.asarray(sign_ru * rom),
            jnp.asarray(msf), jnp.asarray(rdzw), jnp.asarray(fzm), jnp.asarray(fzp), 1 / 3000.0, 1 / 3000.0,
            interpret=True))

    forward = call(1.0)
    forward_reversed = call(-1.0)
    with wrf_time_step_sign(-18.0):
        backward = call(1.0)
    scale = np.abs(forward).max()
    np.testing.assert_allclose(backward, -forward_reversed, rtol=0, atol=2e-6 * scale)
    assert np.abs(backward + forward).max() > 1e-3 * scale  # the upwind term is really present


def test_native_w_momentum_kernel_identity_interpret():
    from gpuwrf.kernels.dyn_momflux_fp32 import advect_w_fp32

    rng = np.random.default_rng(2)
    nz, ny, nx = 6, 9, 11
    w = rng.normal(size=(nz + 1, ny, nx)).astype(np.float32)
    ru = rng.normal(size=(nz, ny, nx + 1)).astype(np.float32) * 5e4
    rv = rng.normal(size=(nz, ny + 1, nx)).astype(np.float32) * 5e4
    rom = rng.normal(size=(nz + 1, ny, nx)).astype(np.float32) * 2e2
    msf = np.ones((ny, nx), np.float32)
    rdn = np.full((nz,), -25.0, np.float32)
    fzm = np.full((nz,), 0.5, np.float32)
    fzp = np.full((nz,), 0.5, np.float32)

    def call(sign_v):
        return np.asarray(advect_w_fp32(jnp.asarray(w), jnp.asarray(sign_v * ru), jnp.asarray(sign_v * rv),
                                        jnp.asarray(sign_v * rom), jnp.asarray(msf), jnp.asarray(rdn),
                                        jnp.asarray(fzm), jnp.asarray(fzp), 1 / 3000.0, 1 / 3000.0,
                                        interpret=True))

    forward_reversed = call(-1.0)
    with wrf_time_step_sign(-18.0):
        backward = call(1.0)
    scale = np.abs(forward_reversed).max()
    np.testing.assert_allclose(backward, -forward_reversed, rtol=0, atol=2e-6 * scale)


def test_step_enters_the_backward_context_for_negative_dt(monkeypatch):
    from gpuwrf.runtime import operational_mode as om

    seen = []
    original = om._physics_boundary_step_with_limiter_diagnostics

    def spy(*args, **kwargs):
        seen.append(time_step_sign())
        return "traced"

    monkeypatch.setattr(om, "_physics_boundary_step_with_limiter_diagnostics", spy)

    class _NL:
        dt_s = -18.0

    assert original(None, _NL(), 3, run_radiation=False) == "traced"
    assert seen == [-1] and time_step_sign() == 1
