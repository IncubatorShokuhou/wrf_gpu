"""First BYTE screen: export an existing qWT result from native whole MYNN.

Only the last return of _advance_native is extended; every original expression,
call and output is retained. This is a component experiment, not yet the full
coupled-forecast proof. No production source is modified by this runner.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time


def observed_advance(module):
    """Add qWT to the existing return, without recomputing any original work."""
    source = inspect.getsource(module._advance_native)
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
    assert len(returns) == 1 and isinstance(returns[0].value, ast.Tuple)
    assert len(returns[0].value.elts) == 2
    returns[0].value.elts.append(ast.Name(id="_qwt", ctx=ast.Load()))
    ast.fix_missing_locations(tree)
    namespace = dict(module.__dict__)
    exec(compile(tree, inspect.getfile(module._advance_native), "exec"), namespace)
    return namespace[function.name], hashlib.sha256(source.encode()).hexdigest()


def compare(left, right):
    import numpy as np
    left, right = np.asarray(left), np.asarray(right)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError("non-finite input/output in BYTE comparison")
    assert (left.shape, left.dtype) == (right.shape, right.dtype)
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return dict(byte_equal=left.tobytes() == right.tobytes(), shape=list(left.shape),
                dtype=str(left.dtype), different_values=int(np.count_nonzero(left != right)),
                max_abs=float(delta.max(initial=0)), rms=float(np.sqrt(np.mean(delta * delta))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--lower-only", action="store_true")
    parser.add_argument("--max-columns", type=int, default=0, help="CPU dry test only; 0 keeps all real columns")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    if args.platform == "cpu" and Path("/tmp/wrf_gpu2_quiet").exists():
        raise RuntimeError("QUIET")
    if args.platform == "gpu" and os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
        raise RuntimeError("official GPU lock required")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.kernels import phys_mynn_columns as N
    from gpuwrf.physics import mynn_pbl as P
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    assert jax.devices()[0].platform == args.platform, jax.devices()
    meta = json.loads((args.inputs / "inputs.json").read_text())
    input_path = args.inputs / "columns.npz"
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == meta["input_sha256"]
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {n: data[n] for n in data.files}
    assert all(np.isfinite(a).all() for a in arrays.values()), "non-finite workload"
    if args.max_columns:
        assert args.platform == "cpu", "GPU screen must use the complete actual domain"
        arrays = {n: a[:args.max_columns] if a.ndim else a for n, a in arrays.items()}
    state = P.MynnPBLColumnState(**{n: jnp.asarray(arrays["state_" + n])
            for n in P.MynnPBLColumnState.__slots__ if "state_" + n in arrays})
    flux = SurfaceFluxes(**{n: jnp.asarray(arrays["flux_" + n])
                          for n in SurfaceFluxes._fields if "flux_" + n in arrays})
    observed, source_sha = observed_advance(N)
    args.out.mkdir(parents=True, exist_ok=True)
    record = dict(schema="wrf_gpu2.w44_budget.prototype.v1", platform=args.platform,
                  scope="whole native MYNN component; not the full coupled program; regression only, not WRF fidelity",
                  inputs_sha256=meta["input_sha256"], original_advance_sha256=source_sha,
                  shape=list(state.theta.shape), dt=meta["dt"], dx=meta["dx"], edmf=True,
                  observed_output="qWT already returned by predictor; final unused top not yet normalized",
                  caches={k: os.environ.get(k) for k in ("GPUWRF_JAX_CACHE", "GPUWRF_JAX_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR")},
                  resolved_env={k: v for k, v in os.environ.items() if k.startswith("GPUWRF_MYNN")}, arms={})
    results, programs = {}, {}
    # Separate callable identities, one source/input set and shared cache.
    for arm in ("off", "on"):
        advance = N._advance_native if arm == "off" else observed
        def step(s, f, advance=advance):
            with jax.enable_x64(False):
                return advance(s, meta["dt"], False, f, True, meta["dx"])
        start = time.perf_counter()
        lowered = jax.jit(step).lower(state, flux)
        hlo = lowered.compiler_ir("hlo").as_hlo_text()
        arm_record = dict(raw_hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                          lower_s=time.perf_counter() - start)
        (args.out / f"{arm}.hlo.txt").write_text(hlo)
        if not args.lower_only:
            compiled = lowered.compile()
            programs[arm] = compiled
            results[arm] = jax.block_until_ready(compiled(state, flux))
            arm_record["compile_first_s"] = time.perf_counter() - start
            arm_record["memory"] = vars(compiled.memory_analysis()) if hasattr(compiled.memory_analysis(), "__dict__") else str(compiled.memory_analysis())
            print(arm, arm_record, flush=True)
        record["arms"][arm] = arm_record
        (args.out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if not args.lower_only:
        fields = {n: compare(getattr(results["off"][0], n), getattr(results["on"][0], n))
                  for n in state.__slots__ if getattr(results["off"][0], n) is not None}
        fields["pblh"] = compare(results["off"][1], results["on"][1])
        diag = np.asarray(results["on"][2])
        assert np.isfinite(diag).all(), "non-finite qWT diagnostic"
        record.update(fields=fields, byte_equal=all(v["byte_equal"] for v in fields.values()),
                      byte_fields=sum(v["byte_equal"] for v in fields.values()), total_fields=len(fields),
                      diagnostic=dict(shape=list(diag.shape), nonzero_values=int(np.count_nonzero(diag)),
                                      max_abs=float(np.max(np.abs(diag))),
                                      sha256=hashlib.sha256(diag.tobytes()).hexdigest()))
        # Save no observer values on failure; the summary is the stop evidence.
        if record["byte_equal"]:
            np.savez(args.out / "accepted_outputs.npz", qWT=diag)
        times = {arm: [] for arm in programs}
        for arm in ("off", "on", "on", "off") * 3:
            start = time.perf_counter()
            jax.block_until_ready(programs[arm](state, flux))
            times[arm].append((time.perf_counter() - start) * 1000)
        record["wall_ms"] = {arm: dict(samples=values, median=statistics.median(values))
                             for arm, values in times.items()}
    record["verdict"] = "LOWERED_ONLY" if args.lower_only else ("COMPONENT_BYTE" if record["byte_equal"] else "STOP_NONBYTE")
    (args.out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("platform", "verdict", "shape")}), flush=True)


if __name__ == "__main__":
    main()
