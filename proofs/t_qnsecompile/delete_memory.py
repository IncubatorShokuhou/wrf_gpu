"""Genuine assertion control: remove typed memory from production QNSE."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0, str(a.tree / 'src'))
    import gpuwrf  # noqa: F401
    import pytest
    from jax.experimental.pallas import triton as pt
    import gpuwrf.kernels.phys_qnse_columns as columns
    spec = importlib.util.spec_from_file_location('controls', a.tree / 'tests/test_t_qnsecompile.py')
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    columns.load, columns.store = pt.load, pt.store
    try:
        with pytest.MonkeyPatch.context() as patch:
            controls.test_actual_qnse_native_uses_typed_memory('sfc', patch)
    except AssertionError as exc:
        assert 'QNSE production loads lost typed lowering' in str(exc)
        report = dict(verdict='KILLED', genuine_assertions=1, setup_errors=0,
                      mutation='replace actual production QNSE load/store with stock Pallas',
                      assertion=str(exc), traceback=traceback.format_exc())
    else:
        raise AssertionError('Production memory deletion survived')
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(report['verdict'], report['assertion'], flush=True)


if __name__ == '__main__':
    main()
