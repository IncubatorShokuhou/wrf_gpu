"""WRF rand_seed and the GNU Fortran xoshiro256** REAL(4) stream profile.

WRF uses processor-defined RANDOM_NUMBER, so there is no compiler-independent
"WRF RNG". This profile matches the staged GNU Fortran 14.3 oracle, including
the int32 overflow in WRF's rand_seed and RANDOM_SEED's eight-word mapping.
The mathematical xoshiro256** recurrence is by Blackman/Vigna (public domain);
the GNU seed mapping is documented by libgfortran/intrinsics/random.c.
"""
from datetime import datetime
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

_KEYS = (0xbd0c5b6e50c2df49, 0xd46061cd46e1df38, 0xbb4f4d4ed6103544, 0x114a583d0756ad39)


def _key(value):
    # Triton's signless MLIR integer constant builder rejects Python uint64
    # values above INT64_MAX. Two 32-bit words preserve the identical bits.
    return jnp.uint64(value & 0xffffffff) | (jnp.uint64(value >> 32) << jnp.uint64(32))


def seed_array(iseed, nens, start: datetime):
    """Original rand_seed INTEGER(4) expression order, GNU eight-word seed."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) or not -(2**31) <= v < 2**31 for v in (iseed, nens)):
        raise ValueError("iseed and nens must be WRF INTEGER(4)")
    wrap = lambda value: (int(value) + 2**31) % 2**32 - 2**31
    fctime = wrap(start.year * (start.month*100 + start.day) + start.hour)
    mix = fctime + wrap(iseed*nens*1000000)
    mod = lambda divisor: (abs(mix) % divisor) * (-1 if mix < 0 else 1)
    group = [wrap(iseed+nens*1000000), mod(19211), mod(71209), mod(11279)]
    return np.asarray(group*2, dtype=np.int32)


def _from_seed(seed):
    return tuple((seed[7-2*i].astype(jnp.uint32).astype(jnp.uint64)
                  | (seed[6-2*i].astype(jnp.uint32).astype(jnp.uint64) << jnp.uint64(32))) ^ _key(key)
                 for i, key in enumerate(_KEYS))


def _to_seed(state):
    words = []
    for value, key in zip(state, _KEYS):
        value = value ^ _key(key)
        words.extend((value.astype(jnp.uint32), (value >> jnp.uint64(32)).astype(jnp.uint32)))
    return jnp.stack(words[::-1]).astype(jnp.int32)


def _uniform(state):
    s0, s1, s2, s3 = state
    v = s1 * jnp.uint64(5)
    v = ((v << jnp.uint64(7)) | (v >> jnp.uint64(57))) * jnp.uint64(9)
    t = s1 << jnp.uint64(17)
    s2, s3 = s2 ^ s0, s3 ^ s1
    s1, s0 = s1 ^ s2, s0 ^ s3
    s2 = s2 ^ t
    s3 = (s3 << jnp.uint64(45)) | (s3 >> jnp.uint64(19))
    # The high 24 bits form an exactly representable REAL32 integer.
    value = (v >> jnp.uint64(40)).astype(jnp.float32) * jnp.float32(2**-24)
    return (s0, s1, s2, s3), value


def _gaussian(state):
    def body(carry):
        state, _z, _accepted = carry
        state, x = _uniform(state)
        state, y = _uniform(state)
        x, y = jnp.float32(2)*x-jnp.float32(1), jnp.float32(2)*y-jnp.float32(1)
        radius = x*x + y*y
        valid = (radius > jnp.float32(0)) & (radius < jnp.float32(1))
        # Invalid proposals are rejected. Avoid a constant-return cond branch
        # (unsupported by current Triton lowering) without changing any accepted
        # radius or the original rejection decisions.
        accepted_radius = jnp.where(valid,radius,jnp.float32(1))
        z = jnp.sqrt(jnp.float32(-2)*jnp.log(accepted_radius)/accepted_radius)*x
        return state, z, valid & (jnp.abs(z) < jnp.float32(3))
    state, z, _accepted = jax.lax.while_loop(lambda carry: ~carry[2], body,
                                           (state, jnp.float32(0), jnp.bool_(False)))
    return state, z


def gaussian_innovations(seed, shape):
    """Return sine/cosine innovations and the exact resumed GNU seed array.

    One device kernel owns WRF's serial stream and rejection decisions. Loops
    remain inside the Pallas kernel; there are no per-draw launches or host
    transfers. CPU interpretation uses the same recurrence and control flow.
    This is an experimental kernel; actual GPU performance is not yet measured.
    """
    if seed.shape != (8,) or seed.dtype != jnp.int32:
        raise ValueError("GNU WRF RNG profile requires eight INTEGER(4) seed words")
    ny, nx = shape
    if min(ny, nx) < 1:
        raise ValueError("innovation shape must be positive")
    count = 2*nx*ny
    def kernel(seed_ref, noise_ref, next_ref):
        # Scalar Ref loads lower directly to Triton. Loading a vector then
        # slicing it introduces a Pallas `slice` primitive unsupported on GPU.
        state = _from_seed(seed_ref)
        def draw(index, state):
            state, value = _gaussian(state)
            noise_ref[index] = value
            return state
        state = jax.lax.fori_loop(0, count, draw, state)
        for i,(value,key) in enumerate(zip(state,_KEYS)):
            value = value ^ _key(key)
            next_ref[7-2*i] = value.astype(jnp.uint32).astype(jnp.int32)
            next_ref[6-2*i] = (value >> jnp.uint64(32)).astype(jnp.uint32).astype(jnp.int32)
    noise, next_seed = pl.pallas_call(kernel, out_shape=(jax.ShapeDtypeStruct((count,), jnp.float32),
        jax.ShapeDtypeStruct((8,), jnp.int32)), grid=(), name="wrf_stochastic_gnu_rng",
        interpret=jax.default_backend() == "cpu")(seed)
    noise = noise.reshape((ny, nx, 2))
    return noise[..., 0], noise[..., 1], next_seed
