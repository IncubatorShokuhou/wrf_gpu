"""Isolate literal saturation and initial Exner against original WRF records."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from compare_stages import records
from gpuwrf.physics.p3.process_support import qv_sat
from gpuwrf.physics.p3.constants import CONSTANTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--strict-math', action='store_true')
    a = p.parse_args()
    assert jax.devices()[0].platform == 'cpu'
    reference = records(a.oracle/'stage_1.bin')
    saved = {name: np.stack([fields[name] for (it, i), fields in reference.items() if it == 1])
             for name in ('t', 'qvs', 'qvi', 'supi')}
    with np.load(Path(__file__).with_name('fixtures')/'columns_mp50.npz') as data:
        pressure, theta, old_qv = (data[key][..., 0] for key in ('p', 'th_old', 'qv_old'))
    f = lambda x: jnp.asarray(x, jnp.float32)
    def calculate(t, pressure, theta, vapor, original_qvi):
        exner = (pressure * f(1e-5)) ** (f(CONSTANTS['rd']) * f(CONSTANTS['inv_cp']))
        ti = theta * exner
        qvi = qv_sat(t, pressure, 1)
        return {'t_recomputed': ti,
                'qvs': qv_sat(t, pressure, 0), 'qvi': qvi,
                'qvi_recomputed': qv_sat(ti, pressure, 1),
                'supi': vapor/qvi - f(1), 'supi_division_only': vapor/original_qvi - f(1)}
    report = {'backend': 'cpu', 'scores': {}}
    for mode in ('eager', 'jit'):
        with jax.disable_jit(mode == 'eager'):
            options = {'xla_cpu_enable_fast_math': False} if a.strict_math else None
            result = jax.jit(calculate, compiler_options=options)(
                f(saved['t']), f(pressure), f(theta), f(old_qv), f(saved['qvi']))
            jax.block_until_ready(result)
        scores = {}
        for name, value in result.items():
            expected = saved['t'] if name == 't_recomputed' else saved[name.replace('_recomputed', '').replace('_division_only', '')]
            value = np.asarray(value)
            assert np.isfinite(value).all()
            error = np.abs(value.astype(np.float64) - expected)
            bad = value != expected
            scores[name] = {'max_abs': float(error.max()), 'different': int(bad.sum()),
                            'indices': np.argwhere(bad).tolist(),
                            'candidate': value[bad].tolist(), 'reference': expected[bad].tolist()}
            if name.startswith('qvi'):
                candidate_sup = np.float32(old_qv/value - np.float32(1))
                scores[name]['cooper_mask_differences'] = np.argwhere(
                    (candidate_sup >= np.float32(.05)) != (saved['supi'] >= np.float32(.05))).tolist()
        report['scores'][mode] = scores
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({mode: {name: {k:v for k,v in score.items() if k in ('max_abs', 'different', 'cooper_mask_differences')}
                            for name,score in scores.items()} for mode,scores in report['scores'].items()}), flush=True)


if __name__ == '__main__':
    main()
