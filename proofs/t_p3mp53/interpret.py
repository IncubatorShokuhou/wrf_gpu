"""Regression only: execute the actual P3 memory primitives on CPU.

Run separately for the baseline and candidate, then compare every saved byte.
Both start from frozen original-WRF operands; this does not replace the WRF
oracle or the forecast D6 gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('tree', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--levels', type=int, choices=(12, 44), default=44)
    a = p.parse_args()
    sys.path.insert(0, str(a.tree / 'src'))
    import gpuwrf  # noqa: F401
    import jax
    import numpy as np
    from gpuwrf.kernels.phys_p3 import full_column
    assert jax.devices()[0].platform == 'cpu'
    fixture = a.tree / 'proofs/s2_p3/fixtures/columns_mp53.npz'
    with np.load(fixture, allow_pickle=False) as f:
        data = dict(f)
    fields = {k: v[..., 0] for k, v in data.items() if v.ndim == 3 }
    assert {v.shape for v in fields.values()} == {(8, 12)}
    if a.levels == 44:
        # Extend the top profile only for the compiler/regression shape test.
        # The independent original-WRF fidelity fixture stays untouched.
        fields = {k: np.pad(v, ((0, 0), (0, 32)), mode='edge')
                  for k, v in fields.items()}
    call = jax.jit(lambda x, it: full_column(
        x, np.float32(18), it, mp=53, interpret=True))
    saved, steps = {}, []
    for it in (1, 2, 3):
        start = time.monotonic()
        result = call(fields, np.int32(it))
        jax.block_until_ready(result)
        elapsed = time.monotonic() - start
        for name, value in result.items():
            value = np.asarray(value)
            assert value.dtype == np.float32 and np.isfinite(value).all(), name
            saved[f'{name}__{it}'] = value
        steps.append({'it': it, 'wall_s': elapsed, 'outputs': len(result)})
        fields = {k: result.get(k, v) for k, v in fields.items()}
        print(it, elapsed, len(result), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out.with_suffix('.npz'), **saved)
    report = {
        'scope': 'CPU Pallas interpret byte regression, not fidelity/performance',
        'source': subprocess.check_output(
            ['git', '-C', str(a.tree), 'rev-parse', 'HEAD'], text=True).strip(),
        'platform': str(jax.devices()[0]), 'levels': a.levels, 'steps': steps,
        'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'outputs': {k: {'shape': v.shape, 'dtype': str(v.dtype),
                       'sha256': hashlib.sha256(v.tobytes()).hexdigest()}
                    for k, v in saved.items()},
    }
    a.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
