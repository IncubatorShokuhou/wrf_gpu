"""WRF rand_seed and the GNU Fortran xoshiro256** REAL(4) stream profile.

WRF uses processor-defined RANDOM_NUMBER, so there is no compiler-independent
"WRF RNG". This profile matches the staged GNU Fortran 14.3 oracle, including
the int32 overflow in WRF's rand_seed and RANDOM_SEED's eight-word mapping.
The mathematical xoshiro256** recurrence is by Blackman/Vigna (public domain);
the GNU seed mapping is documented by libgfortran/intrinsics/random.c.
"""
from datetime import datetime
from functools import lru_cache
import math
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

_KEYS = (0xbd0c5b6e50c2df49, 0xd46061cd46e1df38, 0xbb4f4d4ed6103544, 0x114a583d0756ad39)
# Characteristic polynomial of the GF(2)-linear xoshiro256 state transition (bit k = x^k), found by
# Berlekamp-Massey and pinned by the jump-ahead tests. T^n s = (x^n mod C)(T) s.
_CHARPOLY = 0x10003c03c3f3ecb1904b4edcf26259f850280002bcefd1a5e9d116f2bb0f0f001
_ACCEPT = .78   # P(radius<1)*P(|z|<3) = 0.7834; proposals are generated with a >= 13 sigma margin


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


def _advance(state):
    """The xoshiro256 state transition of :func:`_uniform` without the output scrambler."""
    s0, s1, s2, s3 = state
    t = s1 << jnp.uint64(17)
    s2, s3 = s2 ^ s0, s3 ^ s1
    s1, s0 = s1 ^ s2, s0 ^ s3
    s2 = s2 ^ t
    s3 = (s3 << jnp.uint64(45)) | (s3 >> jnp.uint64(19))
    return (s0, s1, s2, s3)


def _proposal(state):
    """One polar proposal: always two uniforms; WRF keeps it only when accepted."""
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


def _gaussian(state):
    def body(carry):
        return _proposal(carry[0])
    state, z, _accepted = jax.lax.while_loop(lambda carry: ~carry[2], body,
                                           (state, jnp.float32(0), jnp.bool_(False)))
    return state, z


def _check(seed, shape):
    if seed.shape != (8,) or seed.dtype != jnp.int32:
        raise ValueError("GNU WRF RNG profile requires eight INTEGER(4) seed words")
    ny, nx = shape
    if min(ny, nx) < 1:
        raise ValueError("innovation shape must be positive")
    return ny, nx


def gaussian_innovations_serial(seed, shape):
    """WRF's serial stream in ONE single-program kernel (reference and exact fallback)."""
    ny, nx = _check(seed, shape)
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


def _mulmod(a, b):
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a >> 256 & 1:
            a ^= _CHARPOLY
    return result


@lru_cache(maxsize=16)
def _plan(count):
    """Lanes, proposals per lane and host jump polynomials x^(2*block*lane) mod C (4 uint64 words)."""
    needed = math.ceil(count/_ACCEPT + 8*math.sqrt(count) + 64)
    lanes = max(1, min(4096, math.ceil(needed/24)))
    block = math.ceil(needed/lanes)
    step = 2                      # x^(2*block): square-and-multiply from x
    for bit in bin(2*block)[3:]:
        step = _mulmod(step, step)
        if bit == "1":
            step = _mulmod(step, 2)
    table, poly = [], 1
    for _ in range(lanes):
        table.append([(poly >> (64*w)) & (2**64-1) for w in range(4)])
        poly = _mulmod(poly, step)
    return lanes, block, np.asarray(table, dtype=np.uint64)


def gaussian_innovations(seed, shape):
    """Return sine/cosine innovations and the exact resumed GNU seed array.

    WRF's serial stream is reproduced exactly but generated in parallel: every
    proposal consumes exactly two uniforms whether accepted or not, so lane l
    starts at uniform 2*block*l (GF(2) jump-ahead from the seed), each lane draws
    its block of proposals with the serial kernel's arithmetic (:func:`_proposal`),
    and the first ``2*nx*ny`` accepted proposals are compacted in order. The
    resumed seed is the state after the last proposal used. If the generated
    proposals ever held too few accepted values the serial kernel runs instead.
    """
    ny, nx = _check(seed, shape)
    count = 2*nx*ny
    lanes, block, table = _plan(count)
    total = lanes*block
    interpret = jax.default_backend() == "cpu"

    def lanes_kernel(seed_ref, poly_ref, z_ref, ok_ref, start_ref):
        lane = pl.program_id(0)
        state = _from_seed(seed_ref)
        acc = tuple(jnp.uint64(0) for _ in range(4))
        for word in range(4):
            bits = poly_ref[lane, word]
            def horner(i, carry, bits=bits):
                acc, state = carry
                mask = jnp.uint64(0) - ((bits >> i.astype(jnp.uint64)) & jnp.uint64(1))
                acc = tuple(a ^ (x & mask) for a, x in zip(acc, state))
                return acc, _advance(state)
            acc, state = jax.lax.fori_loop(0, 64, horner, (acc, state))
        for w, value in enumerate(acc):
            start_ref[lane, w] = value
        def draw(j, state):
            state, z, ok = _proposal(state)
            z_ref[lane*block + j] = z
            ok_ref[lane*block + j] = ok.astype(jnp.int32)
            return state
        jax.lax.fori_loop(0, block, draw, acc)

    z, ok, starts = pl.pallas_call(lanes_kernel, out_shape=(
        jax.ShapeDtypeStruct((total,), jnp.float32), jax.ShapeDtypeStruct((total,), jnp.int32),
        jax.ShapeDtypeStruct((lanes, 4), jnp.uint64)), grid=(lanes,),
        name="wrf_stochastic_gnu_rng_lanes", interpret=interpret)(seed, jnp.asarray(table))
    accepted = ok.astype(bool)
    index = jnp.nonzero(accepted, size=count, fill_value=0)[0]
    last = index[count-1].astype(jnp.int32)

    def final_kernel(last_ref, starts_ref, next_ref):
        last = last_ref[0]
        lane = last // block
        state = tuple(starts_ref[lane, w] for w in range(4))
        state = jax.lax.fori_loop(0, 2*(last - lane*block + 1), lambda _i, s: _advance(s), state)
        for i,(value,key) in enumerate(zip(state,_KEYS)):
            value = value ^ _key(key)
            next_ref[7-2*i] = value.astype(jnp.uint32).astype(jnp.int32)
            next_ref[6-2*i] = (value >> jnp.uint64(32)).astype(jnp.uint32).astype(jnp.int32)

    next_seed = pl.pallas_call(final_kernel, out_shape=jax.ShapeDtypeStruct((8,), jnp.int32), grid=(),
        name="wrf_stochastic_gnu_rng_resume", interpret=interpret)(last[None], starts)
    noise = z[index].reshape((ny, nx, 2))

    def serial(_):
        sine, cosine, resumed = gaussian_innovations_serial(seed, shape)
        return sine, cosine, resumed
    return jax.lax.cond(jnp.sum(ok) >= count, lambda _: (noise[..., 0], noise[..., 1], next_seed), serial, None)
