"""Release REAL4 cold moments versus the frozen original-WRF witness.

CPU-only; run in a fresh GPUWRF_FAST_DEFAULTS=1 process, like the existing
w44_qsqseed witness. The normal CPU pytest process selects legacy constants.
"""
from pathlib import Path
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import mynn_pbl as P
from gpuwrf._fast_defaults import FAST_DEFAULTS_STATUS


def evaluate(path):
    assert jax.default_backend() == "cpu" and FAST_DEFAULTS_STATUS["enabled"]
    rows = []
    with np.load(path) as packet:
        arrays, scalars, truth = (packet[name] for name in ("arrays", "scalars", "truth"))
        for dx in sorted(set(scalars[:, -1])):
            ix = np.flatnonzero(scalars[:, -1] == dx)
            a = arrays[ix]
            value = lambda n: jnp.asarray(a[:, :, n], jnp.float32)
            zero = jnp.zeros_like(value(0))
            state = P.MynnPBLColumnState(value(0), value(1), value(2), value(3), value(4), value(5),
                value(6), value(7), value(8), zero, zero, zero, qc=value(9), qi=value(10), qs=value(11),
                qsq=zero, exner=value(13))
            with jax.enable_x64(False):
                fun = jax.jit(lambda s, u, land: P.mynn_coldstart_init_columns(s, u, float(dx), land,
                    return_moments=True, return_cloudpdf_moments=True))
                args = state, jnp.asarray(scalars[ix, 0]), jnp.asarray(scalars[ix, 4])
                start = time.perf_counter()
                compiled = fun.lower(*args).compile()
                compile_s = time.perf_counter() - start
                out = compiled(*args)
                jax.block_until_ready(out)
            for slot, name, expected_slot in ((0, "QKE", 0), (2, "EL", 1), (3, "QSQ", 2),
                                              (4, "TSQ", 3), (5, "COV", 4)):
                actual = np.asarray(out[slot], np.float64)
                expected = truth[ix, :, expected_slot].astype(np.float64)
                assert np.isfinite(actual).all() and np.isfinite(expected).all()
                # Same pre-registered .2% gate, unchanged after the FAST0 failure.
                tolerance = 1.e-12 + .002 * np.abs(expected)
                ratio = np.abs(actual - expected) / tolerance
                rows.append(dict(dx=float(dx), field=name, bad=int(np.count_nonzero(ratio > 1)),
                    max_gate=float(ratio.max()), max_abs=float(np.abs(actual - expected).max()),
                    truth_nonzero=int(np.count_nonzero(expected)), columns=len(ix), compile_s=compile_s))
    return dict(passed=all(row["bad"] == 0 for row in rows), rows=rows, backend="cpu", fast_defaults=True)


if __name__ == "__main__":
    report = evaluate(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
    assert report["passed"], report
