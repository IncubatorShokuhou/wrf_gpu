"""WRF's own +-1 ULP input envelope for the MYJSFC oracle (lane t-cfix-sfc, no JAX).

Per chained call c (operands = the zero build's own call-c inputs), DRAWS re-runs of the
pristine zero build with every REAL profile/surface operand nudged by -1/0/+1 ULP, plus
the FMA build; per-output [min, max] is the band.  Discrete XLAND untouched.
usage: python envelope_myj.py <lane_dir>   -> <lane>/oracle/envelope_myj.npz
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_oracle_myj import SFC, load, call  # noqa: E402

LANE = Path(sys.argv[1])
DRAWS = 128


def nudge(a, rng, fixed=()):
    step = rng.integers(-1, 2, size=a.shape)
    out = np.where(step > 0, np.nextafter(a, np.float32(np.inf)), np.where(step < 0, np.nextafter(a, np.float32(-np.inf)), a))
    for j in fixed:
        out[..., j] = a[..., j]
    return out.astype(np.float32)


def main():
    d = dict(np.load(LANE / 'oracle' / 'oracle_myj.npz'))
    fn = load('zero')
    rng = np.random.default_rng(20261017)
    env = {}
    for c in (1, 2, 3):
        stack = [d[f'zero_out{c}'], d[f'fma_out{c}']]
        for _ in range(DRAWS):
            stack.append(call(fn, nudge(d['prof'], rng), nudge(d[f'zero_sfc{c}'], rng, (SFC.index('xland'),)), c))
        stack = np.stack(stack)
        env[f'lower{c}'], env[f'upper{c}'] = stack.min(0), stack.max(0)
    np.savez_compressed(LANE / 'oracle' / 'envelope_myj.npz', draws=DRAWS, **env)
    print('envelope written', DRAWS)


if __name__ == '__main__':
    main()
