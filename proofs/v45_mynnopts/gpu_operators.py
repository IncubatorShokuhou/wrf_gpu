"""Bounded real-column PDF operator arm, shared by CPU sequencer smoke/GPU.

This is a component oracle/latency arm. It never claims coupled forecast,
S3, roofline, E41 or release qualification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import mynn_sgs_cloud as S
from operator_artifacts import check_contract, usage, export_native, native_gate, cleanup_caches, RUN_CAP_BYTES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--option", type=int, choices=(0, 1), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    args.cache_root = args.cache_root.resolve()
    allowed = args.out.resolve().parent / "cache_runs"
    if (not args.cache_root.is_relative_to(allowed)
            or args.cache_root.name != f"pdf{args.option}"):
        raise ValueError("cache root must be this output's private cache_runs/<run>/pdfN")
    args.out.unlink(missing_ok=True)
    caches = check_contract(args.cache_root, args.backend)
    jax.config.update("jax_compilation_cache_dir", caches["JAX_COMPILATION_CACHE_DIR"])
    jax.config.update("jax_enable_compilation_cache", args.backend == "gpu")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_compilation_cache_max_size", 64 * 1024**2)
    try:
        _measure(args, caches)
    finally:
        cleanup_caches(args.cache_root)


def _measure(args, caches):
    assert jax.devices()[0].platform == args.backend
    root = Path(__file__).resolve().parents[2]
    module = root / "src/gpuwrf/physics/mynn_sgs_cloud.py"
    assert Path(S.__file__).resolve() == module.resolve(), S.__file__
    sys.path.insert(0, str(root / "tests/v045"))
    from test_mynn_cloudpdf import _violations
    fixture = root / "tests/fixtures/o5_n333/mynn_cloudpdf_pristine_v1.npz"
    arms = []
    with np.load(fixture) as data:
        arrays = {key[3:]: jnp.asarray(data[key], jnp.float32) for key in data.files if key.startswith("in_")}
        kwargs = {name: arrays[key] for name, key in (("theta", "th"), ("p", "p"), ("exner", "exner"),
            ("dz", "dz"), ("thl", "thl"), ("qw", "qw"), ("tsq", "tsq"), ("qsq", "qsq"),
            ("cov", "cov"), ("el", "el"), ("sh", "sh"))}
        for option in (args.option,):
            with jax.enable_x64(False):
                fun = jax.jit(lambda values: S.mym_condensation_cloudpdf01(option, **values))
                started = time.perf_counter()
                lowered = fun.lower(kwargs)
                compiled = lowered.compile()
                compile_s = time.perf_counter() - started
                native = {"status": "NOT-APPLICABLE (CPU)"}
                e41 = "NOT-APPLICABLE (CPU)"
                if args.backend == "gpu":
                    path, native = export_native(compiled, lowered, args.cache_root, jax.devices()[0])
                    usage(args.cache_root)
                    e41 = native_gate(path, root)
                out = compiled(kwargs)
                jax.block_until_ready(out)
                *_, sigma = S._cloudpdf01_columns(option, **kwargs)
                _, alp, _ = S._alp_bet(arrays["th"] * arrays["exner"], arrays["p"], arrays["exner"])
                qsl = S._alp_bet(arrays["th"] * arrays["exner"], arrays["p"], arrays["exner"])[0]
                q1 = (arrays["qw"][:, :-1] - qsl[:, :-1]) / sigma
                bad = _violations(data, option, [np.asarray(value, np.float64) for value in out],
                                  np.asarray(sigma, np.float64), np.asarray(alp[:, :-1], np.float64),
                                  np.asarray(q1, np.float64))
                assert not any(bad.values()), (option, bad)
                for _ in range(3):
                    jax.block_until_ready(compiled(kwargs))
                times = []
                for _ in range(20):
                    started = time.perf_counter()
                    jax.block_until_ready(compiled(kwargs))
                    times.append(time.perf_counter() - started)
            memory = compiled.memory_analysis()
            arms.append(dict(option=option, violations=bad, compile_s=compile_s,
                median_s=float(np.median(times)), columns=384, levels=44,
                native_export=native, e41=e41,
                memory={name: int(getattr(memory, name)) for name in (
                    "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")}))
    report = dict(backend=args.backend, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        source_root=str(root), module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(), jax_version=jax.__version__,
        scope=__doc__, arms=arms, finite=True,
        e41=arms[0]["e41"],
        cache_contract=dict(paths=caches, persistent_enabled=args.backend == "gpu",
                            actual_jax_cache_dir=jax.config.jax_compilation_cache_dir,
                            arm_bytes=usage(args.cache_root), run_bytes=usage(args.cache_root.parent, RUN_CAP_BYTES),
                            run_cap_bytes=RUN_CAP_BYTES, cleanup="after export+scan; native/E41 evidence retained"),
        coupled_forecast_qualified=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
