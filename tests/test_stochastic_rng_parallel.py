"""Parallel exact WRF/GNU Gaussian stream == the serial single-program kernel (lane w41-stoch).

The serial kernel is the u2-stoch reference qualified against original CPU-WRF seed words; the parallel
kernel must reproduce its innovations and resumed seed bit for bit, including the fallback path.
"""
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import stochastic_rng as rng


def _step(s):
    m = 2**64 - 1
    s0, s1, s2, s3 = s
    t = (s1 << 17) & m
    s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t
    return (s0, s1, s2, ((s3 << 45) | (s3 >> 19)) & m)


def test_jump_table_equals_sequential_stepping():
    lanes, block, table = rng._plan(2*43*43)
    start = (0x0123456789abcdef, 0xfedcba9876543210, 0x0f1e2d3c4b5a6978, 0x8796a5b4c3d2e1f0)
    for lane in (0, 1, 2, lanes - 1):
        poly = sum(int(w) << (64*i) for i, w in enumerate(table[lane]))
        acc, state = (0, 0, 0, 0), start
        for bit in range(256):
            if poly >> bit & 1:
                acc = tuple(a ^ x for a, x in zip(acc, state))
            state = _step(state)
        expect = start
        for _ in range(2*block*lane):
            expect = _step(expect)
        assert acc == expect, lane


@pytest.mark.parametrize("iseed,nens,shape", [(811, 7, (43, 43)), (53, 1, (12, 17)), (217, 3, (1, 1)),
                                              (317, 2146, (20, 33)), (171, 11, (71, 121))])
def test_parallel_stream_is_bitwise_serial(iseed, nens, shape):
    seed = jnp.asarray(rng.seed_array(iseed, nens, datetime(2023, 1, 15, 0)))
    serial = jax.jit(rng.gaussian_innovations_serial, static_argnums=1)(seed, shape)
    parallel = jax.jit(rng.gaussian_innovations, static_argnums=1)(seed, shape)
    for a, b in zip(serial, parallel):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a.view(np.uint32) if a.dtype == np.float32 else a,
                              b.view(np.uint32) if b.dtype == np.float32 else b)


def test_chained_calls_and_fallback_are_bitwise(monkeypatch):
    seed = jnp.asarray(rng.seed_array(811, 7, datetime(2023, 1, 15, 0)))
    s_seed = p_seed = seed
    for _ in range(3):   # SKEBS T then W share one stream: resumed seeds must chain exactly
        *s_out, s_seed = rng.gaussian_innovations_serial(s_seed, (9, 14))
        *p_out, p_seed = rng.gaussian_innovations(p_seed, (9, 14))
        assert np.array_equal(np.asarray(s_seed), np.asarray(p_seed))
        assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(s_out, p_out))
    # Too few generated proposals must take the exact serial fallback.
    monkeypatch.setattr(rng, "_ACCEPT", 50.)
    rng._plan.cache_clear()
    try:
        assert rng._plan(2*9*14)[0]*rng._plan(2*9*14)[1] < 2*9*14
        got = rng.gaussian_innovations(seed, (9, 14))
        ref = rng.gaussian_innovations_serial(seed, (9, 14))
        assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(got, ref))
    finally:
        monkeypatch.undo()
        rng._plan.cache_clear()
