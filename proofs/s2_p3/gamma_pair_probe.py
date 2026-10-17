"""Measure a REAL32-pair intrinsic against original Fortran, without fp64."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.p3.process_support import _rounded_real as rn, _r


def add(a, b):
    s = rn('add', a[0], b[0])
    v = rn('sub', s, a[0])
    e = rn('add', rn('sub', a[0], rn('sub', s, v)), rn('sub', b[0], v))
    e = rn('add', e, rn('add', a[1], b[1]))
    h = rn('add', s, e)
    return h, rn('sub', e, rn('sub', h, s))


def product(a, b):
    p = rn('mul', a, b)
    def split(x):
        c = rn('mul', x, _r(4097))
        hi = rn('sub', c, rn('sub', c, x))
        return hi, rn('sub', x, hi)
    ah, al = split(a)
    bh, bl = split(b)
    e = rn('add', rn('sub', rn('mul', ah, bh), p), rn('mul', ah, bl))
    e = rn('add', e, rn('mul', al, bh))
    return p, rn('add', e, rn('mul', al, bl))


def mul(a, b):
    p, e = product(a[0], b[0])
    e = rn('add', e, rn('add', rn('mul', a[0], b[1]), rn('mul', a[1], b[0])))
    h = rn('add', p, e)
    return h, rn('sub', e, rn('sub', h, p))


def div(a, b):
    q = rn('div', a[0], b[0])
    qb = mul((q, _r(0)), b)
    residual = add(a, (-qb[0], -qb[1]))
    correction = rn('div', rn('add', *residual), b[0])
    return add((q, _r(0)), (correction, _r(0)))


def pair(x):
    hi = np.float32(x)
    return _r(hi), _r(float(x) - float(hi))


def gamma_pair(x):
    # Cody/Stoltz rational, same coefficients as original Morrison GAMMA;
    # extra REAL32 residual words preserve intrinsic, not prognostic precision.
    ps = (-1.7161851388654949, 24.76565080557592, -379.80425647094564,
          629.3311553128184, 866.9662027904132, -31451.272968848367,
          -36144.41341869117, 66456.14382024054)
    qs = (-30.8402300119739, 315.35062697960416, -1015.1563674902191,
          -3107.771671572311, 22538.11842098015, 4755.846277527881,
          -134659.9598649693, -115132.25967555348)
    n = jnp.floor(x).astype(jnp.int32) - 1
    y = rn('sub', x, n.astype(jnp.float32))
    z = (rn('sub', y, _r(1)), _r(0))
    numerator = (jnp.zeros_like(x), _r(0))
    denominator = (jnp.ones_like(x), _r(0))
    for a, b in zip(ps, qs):
        numerator = mul(add(numerator, pair(a)), z)
        denominator = add(mul(denominator, z), pair(b))
    result = add(div(numerator, denominator), pair(1))
    for k in range(22):
        following = mul(result, (rn('add', y, _r(k)), _r(0)))
        result = tuple(jnp.where(k < n, new, old) for new, old in zip(following, result))
    return rn('add', *result)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--oracle', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    assert jax.devices()[0].platform == 'cpu'
    x = np.fromfile(args.oracle / 'input.bin', np.float32)
    reference = np.fromfile(args.oracle / 'output.bin', np.float32)
    start = time.monotonic()
    actual = np.asarray(jax.jit(gamma_pair)(jnp.asarray(x)))
    ulp = np.abs(actual.view(np.int32).astype(np.int64) - reference.view(np.int32).astype(np.int64))
    report = {'samples': len(x), 'different': int(np.count_nonzero(ulp)), 'max_ulp': int(ulp.max()),
              'worst_argument': float(x[ulp.argmax()]), 'compile_first_s': time.monotonic() - start,
              'all_finite': bool(np.isfinite(actual).all()), 'arithmetic': 'REAL32 only'}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
