"""Byte regression on all staged WRF QNSE regimes; not a fidelity substitute."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scheme', choices=('sfc', 'pbl', 'mf'), required=True)
    p.add_argument('--baseline', type=Path)
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0, str(a.tree / 'src'))
    import gpuwrf  # noqa: F401
    import jax
    import jax.numpy as jnp
    import numpy as np
    assert jax.devices()[0].platform == 'cpu'
    spec = importlib.util.spec_from_file_location('qnse_cases', a.tree / 'tests/test_v040_qnse.py')
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    from gpuwrf.kernels.phys_qnse_columns import qnse_columns
    operands = [cases.kwargs(cases.oracle_case(a.scheme, k), a.scheme, jnp.float32)
                for k in range(15)]
    if a.scheme == 'sfc':
        for col in operands:
            if col['scm_force_flux'] is None:
                col['scm_force_flux'] = jnp.float32(1.)
    fields = {k: jnp.stack([col[k] for col in operands], axis=-1) for k in operands[0]}
    t = time.monotonic()
    result = jax.jit(lambda x: qnse_columns(a.scheme, x, interpret=True, block=16))(fields)
    result = {k: np.asarray(v) for k, v in result.items()}
    assert all(np.isfinite(v).all() for v in result.values())
    count = sum(v.size for v in result.values())
    if a.baseline:
        with np.load(a.baseline, allow_pickle=False) as baseline:
            assert set(baseline) == set(result)
            for key, value in result.items():
                assert value.dtype == baseline[key].dtype and value.shape == baseline[key].shape
                assert value.tobytes() == baseline[key].tobytes(), key
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out.with_suffix('.npz'), **result)
    report = dict(source=subprocess.check_output(['git', '-C', str(a.tree), 'rev-parse', 'HEAD'], text=True).strip(),
                  scheme=a.scheme, scope='CPU Pallas interpreter byte regression, not GPU/fidelity',
                  regimes=15, outputs=len(result), values=count, elapsed_s=time.monotonic()-t,
                  baseline=str(a.baseline), verdict='BYTE' if a.baseline else 'BASELINE',
                  hashes={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in result.items()})
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
