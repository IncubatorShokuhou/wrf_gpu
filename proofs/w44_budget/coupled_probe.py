"""Strict first-step qWT prototype gate on a real production domain.

The optional leaf is seeded after the real loader's precision enforcement. OFF
and ON execute _advance_chunk_fori with the actual namelist/clock, traced counts,
cadence and import-resolved donation. No new diagnostic is a prognostic input.
Any non-BYTE old output is a STOP. This experiment does not enable checkpoint
or writer support for the unfinished feature.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-dom", type=int, default=1)
    parser.add_argument("--cpu-lower", action="store_true")
    parser.add_argument("--cpu-compile", action="store_true", help="Small real domain only, capped externally")
    parser.add_argument("--skip-full-forecast", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    if args.cpu_lower or args.cpu_compile:
        if Path("/tmp/wrf_gpu2_quiet").exists():
            raise RuntimeError("QUIET")
        assert os.environ.get("JAX_PLATFORMS") == "cpu"
    else:
        assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1", "official GPU lock required"
    import jax
    import jax.numpy as jnp
    import numpy as np
    from gpuwrf.backends import pallas as facade
    from gpuwrf.contracts import state as S
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime import operational_mode as O
    from jax.experimental import pallas as pl
    from prototype import compare
    platform = "cpu" if args.cpu_lower or args.cpu_compile else "gpu"
    assert jax.devices()[0].platform == platform
    assert Path(O.__file__).resolve().is_relative_to(root)
    assert os.environ.get("GPUWRF_MYNN_FP32_COLUMNS") == "1", "prototype requires native MYNN"
    args.out.mkdir(parents=True, exist_ok=True)
    record = dict(schema="wrf_gpu2.w44_budget.coupled_proto.v1", platform=platform,
                  source_root=str(root), source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  source_tree=subprocess.check_output(["git", "rev-parse", "HEAD:src/gpuwrf"], cwd=root, text=True).strip(),
                  input_dir=str(args.input_dir.resolve()), max_dom=args.max_dom,
                  scope="first real d01 production step; ON/OFF regression identity only, not original-WRF fidelity",
                  caches={k: os.environ.get(k) for k in ("GPUWRF_JAX_CACHE", "GPUWRF_JAX_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR")},
                  arms={})
    record["input_files"] = {p.name: dict(path=str(p.resolve()), bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(args.input_dir.iterdir()) if p.is_file()}
    output = args.out / "result.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    original_pallas = pl.pallas_call
    original_facade = facade.cuda_native
    if platform == "cpu":
        S._gpu_device = lambda: jax.devices("cpu")[0]
        def interpreted(*a, **kw):
            kw["interpret"] = True
            kw.pop("compiler_params", None)
            return original_pallas(*a, **kw)
        pl.pallas_call = interpreted
        facade.cuda_native = lambda: False
        sys.path.insert(0, str(root / "proofs/o1_nlbind"))
        import cpu_safe
        cpu_safe.install()
    config = pipeline.NestedPipelineConfig(args.input_dir, args.out / "init_out",
        args.out / "init_proof", hours=3, max_dom=args.max_dom)
    domains = tuple(f"d{i:02d}" for i in range(1, args.max_dom + 1))
    start = time.perf_counter()
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(config, domains)
    record["init_s"] = time.perf_counter() - start
    nl, initial = bundles["d01"].namelist, carries["d01"]
    record.update(dt=float(nl.dt_s), shape=list(initial.state.theta.shape),
                  cadence=int(nl.radiation_cadence_steps), steps=1)
    O._resolve_operational_suite(nl)
    clock = O.build_clock_base(nl)
    # Keep interpretation for a CPU-only type check/compile. GPU lowering uses
    # untouched product dispatch; no pretending CPU Pallas is native CUDA.
    programs, results = {}, {}
    for arm in ("off", "on"):
        carry = jax.tree.map(lambda a: jnp.array(a, copy=True), initial)
        if arm == "on":
            carry = carry.replace(state=carry.state.replace(mynn_qwt=jnp.zeros_like(carry.state.qv, dtype=jnp.float32)))
        jax.clear_caches()
        start = time.perf_counter()
        lowered = O._advance_chunk_fori.lower(carry, nl, 1, clock,
            n_steps=1, cadence=int(nl.radiation_cadence_steps))
        hlo = lowered.compiler_ir("hlo").as_hlo_text()
        arm_record = dict(lower_s=time.perf_counter() - start,
                          raw_hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest())
        if not args.cpu_lower:
            program = lowered.compile()
            programs[arm] = program
            # Match the original jitted function's complete dynamic signature.
            results[arm] = jax.block_until_ready(program(carry, nl, 1, clock,
                n_steps=1, cadence=int(nl.radiation_cadence_steps)))
            arm_record["compile_first_s"] = time.perf_counter() - start
            stat = program.memory_analysis()
            arm_record["memory"] = {n: int(getattr(stat, n)) for n in
                ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
        record["arms"][arm] = arm_record
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(arm, json.dumps(arm_record), flush=True)
    pl.pallas_call, facade.cuda_native = original_pallas, original_facade
    if args.cpu_lower:
        record["verdict"] = "CPU_LOWER_ONLY"
    else:
        candidate = results["on"].replace(state=results["on"].state.replace(mynn_qwt=None))
        left, ld = jax.tree.flatten_with_path(results["off"])
        right, rd = jax.tree.flatten_with_path(candidate)
        assert ld == rd, "original carry structure changed"
        fields = {str(path): compare(a, b) for (path, a), (other, b) in zip(left, right, strict=True)
                  if not (path != other)}
        assert len(fields) == len(left), "carry path mismatch"
        qwt = np.asarray(results["on"].state.mynn_qwt)
        assert np.isfinite(qwt).all() and np.count_nonzero(qwt), "qWT observer missing/inactive"
        record.update(fields=fields, byte_fields=sum(v["byte_equal"] for v in fields.values()), total_fields=len(fields),
                      qwt=dict(nonzero=int(np.count_nonzero(qwt)), max_abs=float(np.max(np.abs(qwt))), nbytes=qwt.nbytes))
        record["verdict"] = "FIRST_STEP_BYTE" if record["byte_fields"] == len(fields) else "STOP_NONBYTE"
        if platform == "gpu":
            record["device_memory_stats"] = jax.devices()[0].memory_stats()
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("platform", "verdict", "shape")}), flush=True)
    if record["verdict"] == "STOP_NONBYTE":
        raise SystemExit(2)
    if platform == "gpu" and not args.skip_full_forecast:
        from gpuwrf import cli
        original_load = pipeline._load_domains
        original_segment = pipeline.run_operational_domain_tree
        current = {}
        final = {}
        def loaded(*a, **kw):
            values = original_load(*a, **kw)
            if current["on"]:
                values = (*values[:-1], {name: c.replace(state=c.state.replace(
                    mynn_qwt=jnp.zeros_like(c.state.qv, dtype=jnp.float32)))
                    for name, c in values[-1].items()})
            return values
        def segment(*a, **kw):
            start = time.perf_counter()
            result = original_segment(*a, **kw)
            jax.block_until_ready(result.carries)
            current["segments"].append(dict(root_steps=int(kw["root_steps"]),
                wall_s=time.perf_counter() - start))
            final.clear()
            final.update(result.carries)
            return result
        pipeline._load_domains, pipeline.run_operational_domain_tree = loaded, segment
        full = {}
        baseline_sha = None
        try:
            # Fill both keys first, then interleave the warm pair. The original
            # init/step JAX caches and autotune directory remain shared.
            for label, on in (("cold_off", False), ("cold_on", True),
                              ("warm_on", True), ("warm_off", False)):
                current.clear()
                current.update(on=on, segments=[])
                out = args.out / label
                start = time.perf_counter()
                rc = cli.main(["run", "--input-dir", str(args.input_dir),
                    "--output-dir", str(out / "wrfout"), "--scratch-dir", str(out / "scratch"),
                    "--hours", "3", "--max-dom", str(args.max_dom)])
                assert rc == 0, f"CLI {label} rc={rc}"
                rows = {}
                for domain, carry in final.items():
                    old = carry.replace(state=carry.state.replace(mynn_qwt=None))
                    for path, leaf in jax.tree.flatten_with_path(old)[0]:
                        array = np.asarray(leaf)
                        assert np.isfinite(array).all(), f"non-finite {label}/{domain}/{path}"
                        rows[f"{domain}/{path}"] = hashlib.sha256(array.tobytes()).hexdigest()
                files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((out / "wrfout").glob("wrfout_d*")) if p.is_file()}
                arm = dict(wall_s=time.perf_counter() - start, segments=list(current["segments"]),
                           terminal_leaf_sha256=rows, wrfout_sha256=files, files=len(files))
                if baseline_sha is None:
                    baseline_sha = rows
                    baseline_files = files
                arm["terminal_byte_equal"] = rows == baseline_sha
                arm["wrfout_byte_equal"] = files == baseline_files
                full[label] = arm
                record["full_3h"] = full
                output.write_text(json.dumps(record, indent=2) + "\n")
                if not arm["terminal_byte_equal"] or not arm["wrfout_byte_equal"]:
                    record["verdict"] = "STOP_NONBYTE_3H"
                    output.write_text(json.dumps(record, indent=2) + "\n")
                    raise SystemExit(2)
        finally:
            pipeline._load_domains, pipeline.run_operational_domain_tree = original_load, original_segment
        record["verdict"] = "FIRST_STEP_AND_3H_BYTE"
        output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
