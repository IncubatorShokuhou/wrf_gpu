"""Original-WRF compiled oracle under its explicit release REAL/no-FMA CPU mode."""
from pathlib import Path
import argparse
import inspect
import json
import os
import sys

import jax
import numpy as np
from gpuwrf._fast_defaults import FAST_DEFAULTS_STATUS
from gpuwrf.physics import mynn_sgs_cloud as S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=int, choices=(0, 1), required=True)
    parser.add_argument("--delete-barrier", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == "cpu" and FAST_DEFAULTS_STATUS["enabled"]
    assert os.environ["XLA_FLAGS"] == "--xla_cpu_max_isa=AVX"
    root = Path(__file__).resolve().parents[2]
    assert Path(S.__file__).resolve() == root / "src/gpuwrf/physics/mynn_sgs_cloud.py"
    sys.path.insert(0, str(root / "tests/v045"))
    from test_mynn_cloudpdf import _compiled_gate, FIXTURE
    with np.load(FIXTURE) as data:
        _compiled_gate(data, args.option)
        result = dict(healthy_passed=True, option=args.option, backend="cpu", release_real=True,
                      xla_flags=os.environ["XLA_FLAGS"], mutant_rejected=None)
        if args.delete_barrier:
            source = inspect.getsource(S.esat_blend)
            guarded = "jax.lax.optimization_barrier((_T0C4 - _F4(6.0)) - _TICE4)"
            assert source.count(guarded) == 1
            namespace = {}
            exec(compile(source.replace(guarded, "((_T0C4 - _F4(6.0)) - _TICE4)"),
                         "<reciprocal-fold>", "exec"), S.__dict__, namespace)
            S.esat_blend = namespace["esat_blend"]
            jax.clear_caches()
            try:
                _compiled_gate(data, args.option)
            except AssertionError as exc:
                result.update(mutant_rejected=True, mutant_violations=exc.args[0])
            else:
                raise AssertionError("reciprocal-fold deletion survived the healthy compiled gate")
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
