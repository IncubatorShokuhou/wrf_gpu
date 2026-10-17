"""WRF's own +-1 ULP input envelope for the surface-layer oracle (lane t-cfix-sfc, no JAX).

For every scheme and both chained calls, the pristine zero build is re-run on the
same columns with every REAL input independently nudged by -1/0/+1 ULP (seeded,
DRAWS draws).  The per-output [min, max] over the draws (plus the unperturbed and
FMA builds) is the independently derived band a compiled port must hit: differences
of libm/compiler arithmetic of the order of one input rounding cannot be told apart
from WRF itself.  Discrete outputs (REGIME) are excluded from widening.
usage: python ulp_envelope.py <lane_dir>   -> <lane>/oracle/envelope.npz
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_oracle import IN, OUT, load, call  # noqa: E402

LANE = Path(sys.argv[1])
DRAWS = 256
# Integer-valued or bookkeeping inputs are not REAL operands.
FIXED = {'xland', 'lakemask', 'dx', 'regime'}


def nudge(inp, rng):
    out = inp.copy()
    for j, name in enumerate(IN):
        if name in FIXED:
            continue
        step = rng.integers(-1, 2, size=inp.shape[0]).astype(np.float32)
        out[:, j] = np.nextafter(inp[:, j], np.where(step > 0, np.inf, -np.inf).astype(np.float32))
        out[:, j] = np.where(step == 0, inp[:, j], out[:, j])
    return out


def main():
    d = dict(np.load(LANE / 'oracle' / 'oracle.npz'))
    rng = np.random.default_rng(20261016)
    env = {}
    for scheme in (91, 7, 3):
        fn = load(scheme, 'zero')
        for c, inp in ((1, d['inputs']), (2, d[f'sf{scheme}_zero_inp2'])):
            stack = [d[f'sf{scheme}_zero_out{c}'], d[f'sf{scheme}_fma_out{c}']]
            for _ in range(DRAWS):
                stack.append(call(fn, scheme, nudge(inp, rng), c))
            stack = np.stack(stack)
            env[f'sf{scheme}_lower{c}'] = stack.min(0)
            env[f'sf{scheme}_upper{c}'] = stack.max(0)
    np.savez_compressed(LANE / 'oracle' / 'envelope.npz', draws=DRAWS, **env)
    print('envelope written', DRAWS)


if __name__ == '__main__':
    main()
