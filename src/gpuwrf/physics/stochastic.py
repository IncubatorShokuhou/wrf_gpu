"""Standalone REAL arithmetic from pristine WRF dyn_em/module_stoch.F.

These source-qualified building blocks are NOT operational stochastic physics.
The caller must supply WRF's Gaussian innovations; this module does not replace
the compiler-specific RANDOM_NUMBER stream with JAX's unrelated PRNG. No runtime
imports these helpers and all stochastic namelist switches remain refused.

Array views are WRF domain extents without halos, transposed from (i,k,j) to
(k,j,i). The last k slot is the unused top slot in WRF tendency calls; the last
i/j slots are the domain's staggered edges. This is not a State adapter.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def _real(*arrays):
    if any(jnp.dtype(a.dtype) != jnp.dtype("float32") for a in arrays):
        raise ValueError("WRF stochastic arithmetic requires REAL float32 operands")


def update_coefficients(sine, cosine, amplitude, alpha, sine_noise, cosine_noise):
    """UPDATE_STOCH symmetry and AR(1), given WRF's accepted innovations.

    Inputs/outputs use (j,i). Noise is drawn by WRF in j-outer/i-inner order,
    sine first, cosine second, rejecting |z| >= 3. Seed evolution is outside
    this helper's qualified boundary.
    """
    _real(sine, cosine, amplitude, sine_noise, cosine_noise)
    if len(sine.shape) != 2 or any(a.shape != sine.shape for a in (cosine, amplitude, sine_noise, cosine_noise)):
        raise ValueError("spectral arrays must have the same two-dimensional shape")
    ny, nx = sine.shape
    if min(nx, ny) < 2:
        raise ValueError("spectral domain must have at least two points per axis")
    j, i = np.indices((ny, nx))
    lower = j > ny // 2
    jj = np.where(lower, ny - j, j)
    mirror = lower | ((j == 0) & (i > nx // 2))
    ii = np.where(mirror & (i != 0), nx - i, i)
    sign = jnp.asarray(np.where(mirror, -1, 1), dtype=jnp.float32)
    a = jnp.asarray(alpha, dtype=jnp.float32)
    keep = jnp.float32(1) - a
    return (keep * sine + sign * amplitude * sine_noise[jj, ii],
            keep * cosine + amplitude * cosine_noise[jj, ii])


def spectral_field(sine, cosine, *, component="T", dx=1., dy=1.,
                   vertical_cosine=None, vertical_sine=None, vertical_amplitude=None):
    """RAND_PERT_UPDATE spectral arithmetic and unnormalized FFTPACK synthesis.

    ``component`` is T (scalar), U (-d/dy) or V (+d/dx), with the exact WRF
    signs. U and V must use the SAME already-updated streamfunction carry.
    Optional vertical arrays have (k,j,i), amplitude (k,). WRF's imaginary
    update intentionally reads the newly assigned real value.
    """
    _real(sine, cosine)
    if sine.ndim != 2 or cosine.shape != sine.shape or min(sine.shape) < 2:
        raise ValueError("sine/cosine must have the same two-dimensional shape")
    if component not in {"T", "U", "V"}:
        raise ValueError("component must be T, U or V")
    ny, nx = sine.shape
    rpi = jnp.float32(3.141592653589793)
    if component == "T":
        real, imag = cosine, sine
    else:
        size = ny if component == "U" else nx
        # WRF keeps the EVEN Nyquist mode positive (unlike fftfreq).
        index = np.arange(size)
        wave = jnp.asarray(np.where(index <= size // 2, index, index - size), dtype=jnp.float32)
        if component == "U":
            multiplier = jnp.float32(2) * rpi / (jnp.float32(ny) * jnp.float32(dy)) * wave[:, None]
            real, imag = multiplier * sine, -multiplier * cosine
        else:
            multiplier = jnp.float32(2) * rpi / (jnp.float32(nx) * jnp.float32(dx)) * wave[None, :]
            real, imag = -multiplier * sine, multiplier * cosine
    supplied = (vertical_cosine is not None, vertical_sine is not None, vertical_amplitude is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("supply all three vertical structure operands together")
    if all(supplied):
        _real(vertical_cosine, vertical_sine, vertical_amplitude)
        if (vertical_cosine.ndim != 3 or vertical_cosine.shape[1:] != sine.shape
                or vertical_sine.shape != vertical_cosine.shape
                or vertical_amplitude.shape != (vertical_cosine.shape[0],)):
            raise ValueError("vertical structure needs (k,j,i) arrays and (k,) amplitudes")
        amp = vertical_amplitude[:, None, None]
        real = amp * (real * vertical_cosine - imag * vertical_sine)
        imag = amp * (real * vertical_sine + imag * vertical_cosine)
    else:
        real, imag = real[None, ...], imag[None, ...]
    value = jnp.asarray(real + jnp.complex64(1j) * imag, dtype=jnp.complex64)
    # Match WRF's x transform followed by y, including REAL intermediates.
    value = jnp.fft.ifft(value, axis=-1) * jnp.float32(nx)
    value = jnp.fft.ifft(value, axis=-2) * jnp.float32(ny)
    return value.real.astype(jnp.float32)


def skebs_tendencies(u, v, theta, u_pattern, v_pattern, theta_pattern, mu, mub, c1h, c2h):
    """UPDATE_STOCH_TEN with WRF's mass coupling, top and edge exclusions."""
    _real(u, v, theta, u_pattern, v_pattern, theta_pattern, mu, mub, c1h, c2h)
    if u.ndim != 3 or v.shape != u.shape or theta.shape != u.shape:
        raise ValueError("tendencies need common (k,j,i) WRF memory extents")
    nz, ny, nx = u.shape
    if (mu.shape != (ny, nx) or mub.shape != mu.shape or c1h.shape != (nz,) or c2h.shape != (nz,)
            or u_pattern.ndim != 3 or u_pattern.shape[1:] != (ny, nx)
            or v_pattern.shape != u_pattern.shape or theta_pattern.shape != u_pattern.shape
            or u_pattern.shape[0] < 1):
        raise ValueError("mass, metric or stochastic pattern shape mismatch")
    k = jnp.minimum(jnp.arange(nz), u_pattern.shape[0] - 1)
    c1 = c1h[:, None, None]
    mass = c1 * mu + (c1 * mub + c2h[:, None, None])
    return (u.at[:-1, :-1, :].add((u_pattern[k] * mass)[:-1, :-1, :]),
            v.at[:-1, :, :-1].add((v_pattern[k] * mass)[:-1, :, :-1]),
            theta.at[:-1, :-1, :-1].add((theta_pattern[k] * mass)[:-1, :-1, :-1]))


def sppt_tendencies(u, v, theta, qv, pattern, *, stddev, cutoff):
    """PERTURB_PHYSICS_TEND for the first-RK physics-tendency call.

    WRF caps the pattern in place, excludes its final k slot, then multiplies
    u/v/theta and vapor only. Hydrometeor tendencies are not operands. Returned
    pattern includes WRF's untouched final stochastic/top slots.
    """
    _real(u, v, theta, qv, pattern)
    if u.ndim != 3 or any(a.shape != u.shape for a in (v, theta, qv)):
        raise ValueError("tendencies need common (k,j,i) WRF memory extents")
    nz, ny, nx = u.shape
    if pattern.ndim != 3 or pattern.shape[1:] != (ny, nx) or pattern.shape[0] < 2:
        raise ValueError("SPPT needs at least two stochastic k slots and matching horizontal extents")
    limit = jnp.float32(stddev) * jnp.float32(cutoff)
    capped = pattern.at[:min(nz, pattern.shape[0]) - 1].set(
        jnp.clip(pattern[:min(nz, pattern.shape[0]) - 1], -limit, limit))
    k = jnp.minimum(jnp.arange(nz), pattern.shape[0] - 2)
    factor = jnp.float32(1) + capped[k]
    return (u.at[:-1, :-1, :].multiply(factor[:-1, :-1, :]),
            v.at[:-1, :, :-1].multiply(factor[:-1, :, :-1]),
            theta.at[:-1, :-1, :-1].multiply(factor[:-1, :-1, :-1]),
            qv.at[:-1, :-1, :-1].multiply(factor[:-1, :-1, :-1]), capped)
