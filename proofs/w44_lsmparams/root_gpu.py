"""Short actual nested-root gate; run only through the official GPU lock.

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
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--family", choices=("lsm", "jarvis"), default="lsm")
    # Grid/length only: the CUDA mini runs the Swiss single-domain case with max_dom 1.
    parser.add_argument("--max-dom", type=int, choices=(1, 2), default=2)
    parser.add_argument("--hours", type=int, default=3)
    return parser.parse_args()


def alignment(input_dir, max_dom, hours):
    """E225: the run length must be a whole number of steps on every domain (exact rationals)."""
    from fractions import Fraction
    import re
    text = (input_dir / "namelist.input").read_text()

    def values(key):
        match = re.search(rf"(?mi)^\s*{key}\s*=\s*([^\n/]*)", text)
        return [v.strip() for v in match.group(1).split(",") if v.strip()] if match else []
    dt = Fraction(int(values("time_step")[0]))
    num, den = values("time_step_fract_num"), values("time_step_fract_den")
    if num and den and int(den[0]):
        dt += Fraction(int(num[0]), int(den[0]))
    ratios = [int(r) for r in values("parent_time_step_ratio")] or [1]
    steps = {"d01": dt}
    for d in range(2, max_dom + 1):
        steps[f"d0{d}"] = steps["d01"] / ratios[d - 1]
    bad = [name for name, step in steps.items() if (hours * 3600) % step]
    if bad:
        raise SystemExit(f"E225 refused before model work: {hours} h is not a whole number of steps on {bad}")
    return {name: str(step) for name, step in steps.items()}


def main():
    args = arguments()
    source = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])
    assert args.reps > 0 and (args.input_dir / "wrfinput_d01").is_file()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"source": str(source), "pin": pin, "phase": args.phase, "profile": args.profile, "family": args.family,
               "input_dir": str(args.input_dir.resolve()), "cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
               "max_dom": args.max_dom, "hours": args.hours,
               "alignment_dt_s": alignment(args.input_dir, args.max_dom, args.hours),
               "scope": "first and ordinary nested-root replay; not forecast/station fidelity or S1"}
    if args.dry_run:
        assert os.environ["JAX_PLATFORMS"] == "cpu"
        receipt.update(gpu_executed=False, dry_run=True)
        (args.out / "dry.json").write_text(json.dumps(receipt, indent=2))
        return
    platform = "gpu"
    if os.environ.get("W2_LAUNCH_SMOKE_ROOT"):
        # Private launch smoke: the REAL model path below, CPU device only (substituted after the
        # product child env is recorded and checked). Never writes a GPU/native PASS.
        from launch_smoke_model import substitute_device
        receipt.update(substitute_device())
        platform = "cpu"
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    import sys
    sys.path.insert(0, str(source / "src"))
    import jax
    import numpy as np
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.runtime import domain_tree as dt
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.surface_params import controls as sc
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf._fast_defaults import FAST_PATH_DEFAULTS

    assert jax.devices()[0].platform == platform, jax.devices()
    assert Path(om.__file__).resolve().is_relative_to((source / "src").resolve())
    receipt.update(gpu_executed=platform == "gpu", platform=platform, device=str(jax.devices()[0]))
    receipt["resolved_release_flags"] = {name: os.environ.get(name) for name in FAST_PATH_DEFAULTS}
    receipt["environment"] = {name: os.environ.get(name) for name in (
        "JAX_PLATFORMS", "GPUWRF_FAST_DEFAULTS", "GPUWRF_AOT_SLIM_MODULE", "GPUWRF_CENSUS",
        "GPUWRF_RRTMG_MP_RE", "XLA_FLAGS", "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "CUDA_VISIBLE_DEVICES")}
    cfg = NestedPipelineConfig(args.input_dir, args.out / "unused-output", args.out / "unused-proof",
                               hours=args.hours, max_dom=args.max_dom)
    names = ("d01", "d02")[:args.max_dom]
    hierarchy, bundles, meta, start, steps, initial = _load_domains(cfg, names)
    inputs = {p for pattern in ("wrfinput_d??", "wrfbdy_d??", "wrflowinp_d??", "namelist.input")
              for p in args.input_dir.glob(pattern) if p.is_file()}
    receipt["work"] = {"dt_s": steps, "start": start.isoformat(), "meta": meta,
                       "input_sha256": {p.name: file_sha256(p) for p in sorted(inputs)}}
    if args.family == "jarvis":
        # All compared arms use Jarvis on the SAME initialized case state.
        # This is a root regression/cost probe, not a cold-init fidelity claim.
        bundles = {name: dataclasses.replace(b, namelist=dataclasses.replace(
            b.namelist, noahmp_options=dataclasses.replace(
                b.namelist.noahmp_options or NoahMPOptions(), opt_crs=2))) for name, b in bundles.items()}
    parameter_names = ["rsmin"] if args.family == "jarvis" else sorted(set(sc.LSM_CONTROLS) - {"rsmin"})
    values = {name: sc.prior_controls(b.namelist, parameter_names, land_state=initial[name].noahmp_land)
              for name, b in bundles.items()}
    multipliers = {name: 1. for name in parameter_names}
    multipliers["rsmin" if args.family == "jarvis" else "hvt"] = 1.1
    changed_values = {name: sc.expand_controls(b.namelist, multipliers, land_state=initial[name].noahmp_land)
                      for name, b in bundles.items()}
    selections = {"off": None, "on": values, "changed": changed_values}
    trees, runtimes = {}, {}
    for label, controls in selections.items():
        bound = bundles if controls is None else {
            name: dataclasses.replace(b, namelist=sc.bind_domain_controls(
                b.namelist, controls, name, land_state=initial[name].noahmp_land))
            for name, b in bundles.items()}
        trees[label] = dt.DomainTree.from_domains(hierarchy, bound)
        runtimes[label] = dt._prepare_operational_domain_tree_runtime(trees[label])
        receipt.setdefault("resolved_parameters", {})[label] = {
            name: {"signature": sc.signature(b.namelist.surface_controls or {}),
                   "digest": sc.controls_digest(b.namelist.surface_controls),
                   "parameter_bytes": sum(np.asarray(v).nbytes for v in (b.namelist.surface_controls or {}).values()),
                   "opt_crs": (b.namelist.noahmp_options.opt_crs if b.namelist.noahmp_options is not None else 1)}
            for name, b in bound.items()}

    def clone(carry):
        result = jax.tree.map(lambda a: a.copy() if isinstance(a, jax.Array) else a, carry)
        return jax.block_until_ready(result)

    def run(label, inputs, own):
        result = dt.run_operational_domain_tree(trees[label], root_steps=1, carries=inputs,
                    initial_own_steps=own, prepared_runtime=runtimes[label], max_event_tail=0)
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
            if a.dtype in (np.float32, np.float64):
                def ordered(x):
                    kind = np.uint32 if x.dtype == np.float32 else np.uint64
                    bits, sign = x.view(kind), kind(1 << (x.itemsize * 8 - 1))
                    return np.where(bits & sign, ~bits, bits | sign)
                oa, ob = ordered(a), ordered(b)
                row["max_ulp"] = int(np.max(np.maximum(oa, ob) - np.minimum(oa, ob)))
            # Distinct names: rebinding ``reference`` here clobbered the dict on the
            # first differing leaf (IndexError, found by the LS58 real-path smoke).
            ref64, cand64 = a.astype(np.float64), b.astype(np.float64)
            error = np.abs(ref64 - cand64)
            nonzero = ref64 != 0
            row["max_abs"] = float(np.max(error))
            row["max_rel_nonzero_reference"] = float(np.max(error[nonzero] / np.abs(ref64[nonzero]), initial=0))
            row["zero_reference_changed_cells"] = int(np.count_nonzero((~nonzero) & (error != 0)))
            rows.append(row)
        return rows

    own0 = {name: 0 for name in names}
    first = run("off", clone(initial), own0)
    receipt["off_aot"] = json.loads(json.dumps(dt.NESTED_AOT_STATUS))
    baseline = host(first.carries)
    default = run("on", clone(initial), own0)
    default_host = host(default.carries)
    delta = differences(baseline, default_host)
    receipt["default_difference"] = delta
    receipt["default_byte_exact"] = not delta
    receipt["default_one_ulp_class"] = all(row.get("max_ulp", 2) <= 1 for row in delta)
    # Manager ruling 2026-10-25 05:12Z/05:16Z: record the per-field table (max abs / field max, cells, ULP;
    # packed history split by name) with the same code as the CPU LS71 table. Record only: the pre-registered
    # bound in receipts/GPU_ONDEFAULT_GATE.json is applied by the reader, never inside the measured run.
    import field_table
    receipt["default_field_table"] = field_table.domain_tables(baseline, default_host, first.carries, args.input_dir)
    del default_host
    receipt["on_aot"] = json.loads(json.dumps(dt.NESTED_AOT_STATUS))
    native_before = {str(p) for p in Path(receipt["cache"]).rglob("*.xlaexec")}
    changed = run("changed", clone(initial), own0)
    receipt["changed_difference"] = differences(host(default.carries), host(changed.carries))
    receipt["changed_aot"] = json.loads(json.dumps(dt.NESTED_AOT_STATUS))
    assert receipt["changed_difference"], "changed parameter did not affect the real root"
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
        run(label, clone(first.carries), first.own_steps)
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
        if args.profile and platform == "gpu":
            cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
            assert cuda.cudaProfilerStart() == 0
        try:
            for rep in range(args.reps):
                for label in (("off", "on") if rep % 2 == 0 else ("on", "off")):
                    inputs = clone(first.carries)
                    with jax.profiler.TraceAnnotation("LSMPARAMS:" + label):
                        t0 = time.perf_counter()
                        run(label, inputs, first.own_steps)
                        samples[label].append(time.perf_counter() - t0)
        finally:
            if args.profile and platform == "gpu":
                assert cuda.cudaProfilerStop() == 0
        receipt["samples_s"] = samples
        receipt["median_s"] = {name: statistics.median(rows) for name, rows in samples.items()}
        receipt["on_over_off"] = receipt["median_s"]["on"] / receipt["median_s"]["off"]
    receipt["combined_variants_device_memory_stats"] = jax.devices()[0].memory_stats()
    receipt["combined_variants_host_vmhwm_kb"] = int(next(
        line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:")))
    result_path.write_text(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
