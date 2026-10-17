"""Actual native-Pallas gate against independently saved pristine P3 columns."""
import argparse
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np

from gpuwrf.kernels.phys_p3 import full_column

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'proofs/s2_p3'))
from comparison import checked_inventory, compare_field


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mp', type=int, default=50)
    a = p.parse_args()
    assert jax.devices()[0].platform == 'gpu'
    with np.load(ROOT / f'proofs/s2_p3/fixtures/columns_mp{a.mp}.npz') as data:
        data = dict(data)
    checked_inventory(data, 'original WRF')
    fields = {k: v[..., 0] for k, v in data.items() if v.ndim == 3}
    if a.mp != 52:
        for k in ('qi2', 'ni2', 'qir2', 'bir2', 'vm2', 'di2', 'rhoi2'):
            fields.pop(k, None)
    if a.mp != 53:
        fields.pop('qzi', None)
    run = jax.jit(lambda f, it: full_column(f, np.float32(18), it, mp=a.mp))
    report = {'platform': str(jax.devices()[0]), 'source': str(ROOT), 'mp': a.mp,
              'scope': 'native GPU vs original-WRF column fixtures; strict scores, not D6', 'steps': []}
    saved = {}
    for it in (1, 2, 3):
        start = time.monotonic()
        out = run(fields, np.int32(it))
        jax.block_until_ready(out)
        elapsed = time.monotonic()-start
        checked_inventory(out, f'GPU.step{it}', mp=a.mp, reference=data)
        scores = {k: compare_field(k, np.asarray(v), data[k][..., it]) for k, v in out.items()}
        report['steps'].append({'step': it, 'wall_s': elapsed, 'scores': scores})
        print(it, elapsed, {k: (v['strict_bad'], v['max_abs']) for k, v in scores.items()}, flush=True)
        for k, v in out.items():
            saved[f'{k}__{it}'] = np.asarray(v)
        fields = {k: out.get(k, v) for k, v in fields.items()}
        a.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(a.out.with_suffix('.npz'), **saved)
    report['memory_analysis'] = str(run.lower(fields, np.int32(3)).compile().memory_analysis())
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
