"""Thompson nested-root gate, based on w44-params' pinned 8a443ffeaf harness.

Prime and measure in separate processes (E179). Copies, initialization, key
inspection and host comparisons stay outside the timed root ranges. This is a
component/root cost, not an S1 whole-forecast claim.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from phase_journal import PhaseJournal, write_receipt
from product_schedule import product_schedule


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("prime", "measure"), required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--mini", action="store_true", help="CUDA mini: 5-column Thompson model swap, not a grid crop")
    parser.add_argument("--reps", type=int, default=12)
    return parser.parse_args()


def mini_measure(args, source, receipt, journal):
    """CUDA mini warm phase (manager 21:48Z): MODEL SWAP, not a grid crop.

    Loads the two prepared 5-column Thompson programs (OFF/ON) from the mini
    cache, refuses any new compile, times reps under the product NVTX labels
    and proves a changed Nt_c changes output. No PROD root/fidelity claim.
    """
    cpu = args.cpu_smoke  # the mini's CPU all-phase probe runs this same body
    assert args.phase == "measure" and not (cpu and args.profile)
    if cpu:
        assert os.environ.get("JAX_PLATFORMS") == "cpu" and os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1"
    else:
        assert os.environ.get("JAX_PLATFORMS") == "cuda" and os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    import sys
    sys.path.insert(0, str(source / "src"))
    import jax
    import numpy as np
    from cold_prepare import cpu_operands, prepare
    from gpuwrf.physics.thompson_parameters import validate_parameters
    from gpuwrf._fast_defaults import apply_fast_path_defaults
    from roundoff import compare
    apply_fast_path_defaults()
    assert jax.devices()[0].platform == ("cpu" if cpu else "gpu"), jax.devices()
    receipt.update(gpu_executed=not cpu, platform=jax.devices()[0].platform, device=str(jax.devices()[0]),
                   mini=True, cpu_smoke=cpu,
                   scope="CUDA mini: 5-column Thompson MODEL SWAP (not a grid crop); launcher/AOT/E41-cubin/NVTX evidence only")
    calls, outputs, timings, statuses = {}, {}, {}, {}
    for variant in ("off", "on"):
        fn, operands, key = cpu_operands(source, variant, interpret=cpu, namespace="cuda-mini-")
        call, status = prepare("d01", fn, operands, key, journal)
        assert status["loaded"], "CUDA mini warm must load the prepared program (no new compile)"
        calls[variant], statuses[variant] = (call, operands), status
        with journal.phase("warm_mini_first_" + variant):
            outputs[variant] = jax.block_until_ready(call(*operands))
    if args.profile:
        cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        assert cuda.cudaProfilerStart() == 0
    try:
        for variant, (call, operands) in calls.items():
            rows = []
            for _ in range(args.reps):
                start = time.perf_counter()
                with jax.profiler.TraceAnnotation("THPARAMS:root:" + variant):
                    jax.block_until_ready(call(*operands))
                rows.append(time.perf_counter() - start)
            timings[variant] = dict(reps=len(rows), median_s=statistics.median(rows), max_s=max(rows))
    finally:
        if args.profile:
            assert cuda.cudaProfilerStop() == 0
    call, operands = calls["on"]
    changed = {k: float(np.asarray(v)) for k, v in operands[1].items()}
    changed["nt_c"] = 50.e6
    other = jax.block_until_ready(call(operands[0], validate_parameters(changed)))
    changed_active = any(np.asarray(a).tobytes() != np.asarray(b).tobytes()
                         for a, b in zip(jax.tree.leaves(outputs["on"]), jax.tree.leaves(other), strict=True))
    assert changed_active, "Nt_c=50e6 must change the mini output"
    default = compare([np.asarray(x) for x in jax.tree.leaves(outputs["off"])],
                      [np.asarray(x) for x in jax.tree.leaves(outputs["on"])], nt_c=True)
    receipt.update(status="PASS", aot=statuses, timings=timings, changed_active=True, default_roundoff=default,
                   default_byte_exact=default.get("byte_exact"),
                   environment={k: os.environ.get(k) for k in (
                       "JAX_PLATFORMS", "GPUWRF_FAST_DEFAULTS", "GPUWRF_JAX_CACHE", "GPUWRF_AOT_SLIM_MODULE",
                       "GPUWRF_CENSUS", "GPUWRF_RRTMG_MP_RE", "XLA_PYTHON_CLIENT_ALLOCATOR",
                       "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION")})
    name = "measure_cpu_smoke" if cpu else "measure_mini" + ("_profile" if args.profile else "")
    write_receipt(args.out / (name + ".json"), receipt)


def main():
    args = arguments()
    source = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])
    assert args.reps > 0 and (args.input_dir / "wrfinput_d01").is_file()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"source": str(source), "pin": pin, "phase": args.phase, "profile": args.profile, "family": "thompson_parameters",
               "input_dir": str(args.input_dir.resolve()), "cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
               "scope": "PROD 2-nest first and ordinary root replay; not forecast/station fidelity or S1"}
    journal = PhaseJournal(args.out / (args.phase + ("_profile" if args.profile else "") + "_phases.jsonl"))
    if args.dry_run:
        assert os.environ["JAX_PLATFORMS"] == "cpu"
        receipt.update(gpu_executed=False, dry_run=True)
        (args.out / "dry.json").write_text(json.dumps(receipt, indent=2))
        return
    if args.mini:
        return mini_measure(args, source, receipt, journal)
    if args.cpu_smoke:
        assert args.phase == "measure" and not args.profile
        assert os.environ.get("JAX_PLATFORMS") == "cpu"
        assert os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1"
        import sys
        sys.path.insert(0, str(source / "src"))
        import jax
        import numpy as np
        from cold_prepare import cpu_operands, prepare
        from gpuwrf.physics.thompson_parameters import validate_parameters
        from roundoff import compare
        assert jax.devices()[0].platform == "cpu"
        outputs = {}
        for variant in ("off", "on"):
            fn, operands, key = cpu_operands(source, variant)
            call, status = prepare("launch_smoke_" + variant, fn, operands, key, journal)
            assert status["loaded"], "run CPU preparation smoke first"
            with journal.phase("warm_mini_model_" + variant):
                outputs[variant] = jax.block_until_ready(call(*operands))
            if variant == "on":
                changed = {k: float(np.asarray(v)) for k, v in operands[1].items()}
                changed["nt_c"] = 50.e6
                with journal.phase("warm_mini_model_changed"):
                    other = jax.block_until_ready(call(operands[0], validate_parameters(changed)))
                assert any(np.asarray(a).tobytes() != np.asarray(b).tobytes()
                           for a, b in zip(jax.tree.leaves(outputs[variant]), jax.tree.leaves(other), strict=True))
        default = compare([np.asarray(x) for x in jax.tree.leaves(outputs["off"])],
                          [np.asarray(x) for x in jax.tree.leaves(outputs["on"])], nt_c=True)
        assert default["within_roundoff"], default
        _, schedule = product_schedule(args.input_dir, ("d01", "d02"), {"d01":54., "d02":18.}, hours=3)
        receipt.update(status="PASS", platform="cpu", gpu_executed=False, cpu_smoke=True,
                       default_roundoff=default, changed_active=True, schedule=schedule,
                       environment={k:os.environ.get(k) for k in (
                           "JAX_PLATFORMS", "GPUWRF_FAST_DEFAULTS", "GPUWRF_JAX_CACHE",
                           "GPUWRF_AOT_SLIM_MODULE", "GPUWRF_CENSUS", "GPUWRF_RRTMG_MP_RE")},
                       scope="CPU launcher and real Thompson mini model; product scheduling separately proven; no root/GPU performance claim")
        write_receipt(args.out / "measure_cpu_smoke.json", receipt)
        return
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    import sys
    sys.path.insert(0, str(source / "src"))
    import jax
    import numpy as np
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.runtime import domain_tree as dt
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime_parameters import bind_domain_parameters
    from gpuwrf._fast_defaults import FAST_PATH_DEFAULTS

    assert jax.devices()[0].platform == "gpu", jax.devices()
    assert Path(om.__file__).resolve().is_relative_to(source / "src")
    receipt.update(gpu_executed=True, platform="gpu", device=str(jax.devices()[0]))
    receipt["resolved_release_flags"] = {name: os.environ.get(name) for name in FAST_PATH_DEFAULTS}
    receipt["environment"] = {name: os.environ.get(name) for name in (
        "JAX_PLATFORMS", "GPUWRF_FAST_DEFAULTS", "GPUWRF_AOT_SLIM_MODULE", "GPUWRF_CENSUS",
        "GPUWRF_RRTMG_MP_RE", "XLA_FLAGS", "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION")}
    cfg = NestedPipelineConfig(args.input_dir, args.out / "unused-output", args.out / "unused-proof",
                               hours=3, max_dom=2)
    names = ("d01", "d02")
    with journal.phase("load_domains"):
        hierarchy, bundles, meta, start, steps, initial = _load_domains(cfg, names)
    run_kwargs, schedule = product_schedule(args.input_dir, names, steps, hours=3)
    receipt["schedule"] = schedule
    write_receipt(args.out / "progress.json", receipt)
    inputs = {p for pattern in ("wrfinput_d??", "wrfbdy_d??", "wrflowinp_d??", "namelist.input")
              for p in args.input_dir.glob(pattern) if p.is_file()}
    receipt["work"] = {"dt_s": steps, "start": start.isoformat(), "meta": meta,
                       "input_sha256": {p.name: file_sha256(p) for p in sorted(inputs)}}
    values = {"nt_c": 100.e6, "tno": 5., "ato": .304, "av_c": 31694600.}
    changed_values = {**values, "nt_c": 50.e6}
    selections = {"off": None, "on": values, "changed": changed_values}
    trees, runtimes = {}, {}
    for label, controls in selections.items():
        bound = bundles if controls is None else {
            name: dataclasses.replace(b, namelist=bind_domain_parameters(
                b.namelist, {name: {"thompson_parameters": controls}}, name))
            for name, b in bundles.items()}
        trees[label] = dt.DomainTree.from_domains(hierarchy, bound)
        runtimes[label] = dt._prepare_operational_domain_tree_runtime(trees[label])
        family = "thompson_parameters"
        receipt.setdefault("resolved_parameters", {})[label] = {
            name: {key: float(np.asarray(value)) for key, value in (getattr(b.namelist, family) or {}).items()}
            for name, b in bound.items()}
    process_native_before = {str(p) for p in Path(receipt["cache"]).rglob("*.xlaexec")}

    def clone(carry):
        result = jax.tree.map(lambda a: a.copy() if isinstance(a, jax.Array) else a, carry)
        return jax.block_until_ready(result)

    def run(label, inputs, own):
        result = dt.run_operational_domain_tree(trees[label], root_steps=1, carries=inputs,
                    initial_own_steps=own, prepared_runtime=runtimes[label], max_event_tail=0,
                    **run_kwargs)
        jax.block_until_ready(result.carries)
        return result

    def host(carry):
        return {jax.tree_util.keystr(path): np.asarray(value).copy()
                for path, value in jax.tree_util.tree_flatten_with_path(carry)[0]}

    def differences(reference, actual):
        assert reference.keys() == actual.keys()
        rows = []
        for name in reference:
            a, b = reference[name], actual[name]
            assert a.shape == b.shape and a.dtype == b.dtype
            assert np.isfinite(a).all() and np.isfinite(b).all(), name
            if a.tobytes() == b.tobytes():
                continue
            row = {"leaf": name, "dtype": str(a.dtype), "changed": int(np.count_nonzero(a != b))}
            if a.dtype == np.float32:
                def ordered(x):
                    bits = x.view(np.uint32)
                    return np.where(bits & np.uint32(0x80000000), ~bits, bits | np.uint32(0x80000000)).astype(np.int64)
                row["max_ulp"] = int(np.max(np.abs(ordered(a) - ordered(b))))
            row["max_abs"] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            rows.append(row)
        return rows

    own0 = {name: 0 for name in names}
    with journal.phase("first_off"):
        first = run("off", clone(initial), own0)
    receipt["off_aot"] = json.loads(json.dumps(dt.NESTED_AOT_STATUS))
    assert set(receipt["off_aot"]["domains"]) == {"d01", "d02"}, "PROD program inventory mismatch"
    write_receipt(args.out / "progress.json", receipt)
    baseline = host(first.carries)
    with journal.phase("first_on"):
        default = run("on", clone(initial), own0)
    delta = differences(baseline, host(default.carries))
    receipt["default_difference"] = delta
    receipt["default_byte_exact"] = not delta
    receipt["default_one_ulp_class"] = all(row.get("max_ulp", 2) <= 1 for row in delta)
    receipt["default_two_ulp_class"] = all(row.get("max_ulp", 3) <= 2 for row in delta)
    receipt["on_aot"] = json.loads(json.dumps(dt.NESTED_AOT_STATUS))
    write_receipt(args.out / "progress.json", receipt)
    native_before = {str(p) for p in Path(receipt["cache"]).rglob("*.xlaexec")}
    with journal.phase("first_changed"):
        changed = run("changed", clone(initial), own0)
    receipt["changed_difference"] = differences(host(default.carries), host(changed.carries))
    receipt["changed_aot"] = json.loads(json.dumps(dt.NESTED_AOT_STATUS))
    # A real PROD root may have no active cold/low-cloud cells. The separate
    # real-column GPU oracle must prove each knob's effect; never fabricate
    # clouds in the production root used for identity and cost measurements.
    receipt["changed_root_active"] = bool(receipt["changed_difference"])
    on_keys = {name: row.get("cheap_key") for name, row in receipt["on_aot"]["domains"].items()}
    changed_keys = {name: row.get("cheap_key") for name, row in receipt["changed_aot"]["domains"].items()}
    receipt["same_aot_keys"] = bool(on_keys and all(on_keys.values()) and on_keys == changed_keys)
    receipt["changed_new_native_blobs"] = sorted({str(p) for p in Path(receipt["cache"]).rglob("*.xlaexec")} - native_before)
    receipt["changed_loaded"] = all(row.get("loaded") for row in receipt["changed_aot"]["domains"].values())
    result_path = args.out / (args.phase + ("_profile" if args.profile else "") + ".json")
    result_path.write_text(json.dumps(receipt, indent=2, default=str))
    assert receipt["same_aot_keys"], "AOT key changed across parameter values"
    assert receipt["changed_loaded"] and not receipt["changed_new_native_blobs"], "changed values compiled a native program"
    # A non-BYTE result is not automatically a shelving verdict. The manager's
    # binding rule requires measured constant-folding/association attribution,
    # few-ULP magnitudes, changed-value WRF parity and one AOT key. Preserve the
    # differences for that separate adjudication; timing does not certify it.
    receipt["default_roundoff_review_required"] = not receipt["default_byte_exact"]
    del default, changed, baseline
    # Ordinary root warmup also primes carry/boundary shape variants.
    for label in ("off", "on"):
        with journal.phase("ordinary_" + label):
            ordinary = run(label, clone(first.carries), first.own_steps)
        if label == "off":
            ordinary_reference = host(ordinary.carries)
        else:
            ordinary_delta = differences(ordinary_reference, host(ordinary.carries))
            receipt["ordinary_default_difference"] = ordinary_delta
            receipt["ordinary_default_byte_exact"] = not ordinary_delta
            receipt["ordinary_default_two_ulp_class"] = all(row.get("max_ulp", 3) <= 2 for row in ordinary_delta)
            receipt["default_roundoff_review_required"] |= bool(ordinary_delta)
            del ordinary_reference
        receipt.setdefault("ordinary_aot", {})[label] = json.loads(json.dumps(dt.NESTED_AOT_STATUS))
        window = inspect.getclosurevars(runtimes[label].advance).nonlocals["_advance_windowed"]
        calls = inspect.getclosurevars(window).nonlocals["_aot_calls"]
        memory = {}
        for (domain, key), call in calls.items():
            executable = getattr(call, "loaded_executable", None)
            if executable is None:
                memory[domain + ":" + key] = {"status": "UNKNOWN_NOT_LOADED"}
            else:
                stats = executable.get_compiled_memory_stats()
                memory[domain + ":" + key] = {name: getattr(stats, name) for name in (
                    "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes",
                    "alias_size_in_bytes", "generated_code_size_in_bytes")}
        receipt.setdefault("compiled_memory", {})[label] = memory
    if args.phase == "measure":
        samples = {"off": [], "on": []}
        if args.profile:
            cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
            assert cuda.cudaProfilerStart() == 0
        try:
            for rep in range(args.reps):
                for label in (("off", "on") if rep % 2 == 0 else ("on", "off")):
                    inputs = clone(first.carries)
                    with jax.profiler.TraceAnnotation("THPARAMS:root:" + label):
                        t0 = time.perf_counter()
                        run(label, inputs, first.own_steps)
                        samples[label].append(time.perf_counter() - t0)
        finally:
            if args.profile:
                assert cuda.cudaProfilerStop() == 0
        receipt["samples_s"] = samples
        receipt["median_s"] = {name: statistics.median(rows) for name, rows in samples.items()}
        receipt["on_over_off"] = receipt["median_s"]["on"] / receipt["median_s"]["off"]
    receipt["combined_variants_device_memory_stats"] = jax.devices()[0].memory_stats()
    receipt["combined_variants_host_vmhwm_kb"] = int(next(
        line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:")))
    receipt["new_native_process_blobs"] = sorted({str(p) for p in Path(receipt["cache"]).rglob("*.xlaexec")} - process_native_before)
    receipt["all_warm_root_programs_loaded"] = all(
        row.get("loaded") for status in (receipt["off_aot"], receipt["on_aot"],
                                         *receipt["ordinary_aot"].values()) for row in status["domains"].values())
    inventory = {name: {} for status in (receipt["off_aot"], receipt["on_aot"], receipt["changed_aot"],
                                         *receipt["ordinary_aot"].values()) for name in status["domains"]}
    (args.out / "program_inventory.json").write_text(json.dumps({
        "pin": pin, "scope": receipt["scope"], "metadata": {"nested_aot": {"domains": inventory}}}, indent=2))
    result_path.write_text(json.dumps(receipt, indent=2, default=str))
    if args.phase == "measure":
        assert receipt["all_warm_root_programs_loaded"] and not receipt["new_native_process_blobs"], "warm root compiled/missed AOT"


if __name__ == "__main__":
    main()
