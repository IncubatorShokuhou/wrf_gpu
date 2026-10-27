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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_journal import PhaseJournal, write_receipt
from product_schedule import product_schedule
from launch_environment import select_device


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
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--family", choices=("diffusion", "mynn"), default="diffusion")
    return parser.parse_args()


def differences(reference, actual):
    """Finite, byte-sensitive differences for every carried field in a root."""
    import numpy as np

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
        before, after = a.astype(np.float64), b.astype(np.float64)
        error = np.abs(before - after)
        nonzero = before != 0
        row["max_abs"] = float(np.max(error))
        row["max_rel_nonzero_reference"] = float(np.max(error[nonzero] / np.abs(before[nonzero]), initial=0))
        row["zero_reference_changed_cells"] = int(np.count_nonzero((~nonzero) & (error != 0)))
        rows.append(row)
    return rows


def runtime_calls(runtime):
    """Read this runtime's own regular/fused caches, including fast-path hits."""
    window = inspect.getclosurevars(runtime.advance).nonlocals["_advance_windowed"]
    calls = dict(inspect.getclosurevars(window).nonlocals["_aot_calls"])
    if runtime.fused_cascade is not None:
        for program in inspect.getclosurevars(runtime.fused_cascade).nonlocals["cache"].values():
            if program is None:
                continue
            inner = inspect.getclosurevars(program).nonlocals["_fused_call"]
            scope = inspect.getclosurevars(inner).nonlocals
            for call, key in scope["cached_calls"].values():
                calls[(scope["aot_name"], key)] = call
    return {address: call for address, call in calls.items() if call is not None}


def main():
    args = arguments()
    source = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])
    assert args.reps > 0 and (args.input_dir / "wrfinput_d01").is_file()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"source": str(source), "pin": pin, "phase": args.phase, "profile": args.profile, "family": args.family,
               "input_dir": str(args.input_dir.resolve()), "cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
               "scope": "PROD 2-nest first and ordinary root replay; not forecast/station fidelity or S1"}
    journal = PhaseJournal(args.out / (args.phase + ("_profile" if args.profile else "") + "_phases.jsonl"))
    if args.dry_run:
        assert os.environ["JAX_PLATFORMS"] == "cpu"
        receipt.update(gpu_executed=False, dry_run=True)
        (args.out / "dry.json").write_text(json.dumps(receipt, indent=2))
        return
    receipt.update(select_device(args.cpu_smoke))
    write_receipt(args.out / 'started.json', receipt)
    import sys
    sys.path.insert(0, str(source / "src"))
    if args.cpu_smoke:
        from warm_smoke import run
        run(args, source, receipt, journal)
        return
    import jax
    import numpy as np
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.runtime import domain_tree as dt
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime_parameters import MYNN, bind_diffusion_parameters, bind_mynn_parameters
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
    with journal.phase('load_domains'):
        hierarchy, bundles, meta, start, steps, initial = _load_domains(cfg, names)
    run_kwargs, receipt['product_schedule'] = product_schedule(args.input_dir, names, steps, hours=3)
    nesting = meta.get("nesting", {"feedback": bool(cfg.feedback), "spec_zone": 1, "smooth_option": 1})
    inputs = {p for pattern in ("wrfinput_d??", "wrfbdy_d??", "wrflowinp_d??", "namelist.input")
              for p in args.input_dir.glob(pattern) if p.is_file()}
    receipt["work"] = {"dt_s": steps, "start": start.isoformat(), "meta": meta,
                       "input_sha256": {p.name: file_sha256(p) for p in sorted(inputs)}}
    values = {"c_s": .25, "diff_6th_factor": .12, "dampcoef": .2, "zdamp": 5000.,
              "w_alpha": .3, "w_beta": 1., "w_crit_cfl": 1.}
    if args.family == "diffusion":
        for bundle in bundles.values():
            for name in ("c_s", "diff_6th_factor", "dampcoef", "zdamp"):
                assert float(getattr(bundle.namelist, name)) == values[name]
        changed_values = {**values, "c_s": .5}
        bind = bind_diffusion_parameters
    else:
        values = {name: spec[0] for name, spec in MYNN.items()}
        changed_values = {name: value * 1.1 for name, value in values.items()}
        bind = bind_mynn_parameters
    selections = {"off": None, "on": values, "changed": changed_values}
    trees, runtimes = {}, {}
    for label, controls in selections.items():
        bound = bundles if controls is None else {
            name: dataclasses.replace(b, namelist=bind(b.namelist, controls))
            for name, b in bundles.items()}
        trees[label] = dt.DomainTree.from_domains(hierarchy, bound, feedback_enabled=nesting["feedback"],
            feedback_spec_zone=nesting["spec_zone"], feedback_smooth_option=nesting["smooth_option"])
        runtimes[label] = dt._prepare_operational_domain_tree_runtime(trees[label])
        family = "diffusion_parameters" if args.family == "diffusion" else "mynn_parameters"
        receipt.setdefault("resolved_parameters", {})[label] = {
            name: {key: float(np.asarray(value)) for key, value in (getattr(b.namelist, family) or {}).items()}
            for name, b in bound.items()}

    def clone(carry):
        result = jax.tree.map(lambda a: a.copy() if isinstance(a, jax.Array) else a, carry)
        return jax.block_until_ready(result)

    def run(label, inputs, own):
        result = dt.run_operational_domain_tree(trees[label], root_steps=1, carries=inputs,
                    initial_own_steps=own, prepared_runtime=runtimes[label], max_event_tail=0, **run_kwargs)
        jax.block_until_ready(result.carries)
        return result

    def host(carry):
        return {jax.tree_util.keystr(path): np.asarray(value).copy()
                for path, value in jax.tree_util.tree_flatten_with_path(carry)[0]}

    own0 = {name: 0 for name in names}
    with journal.phase('first_off'):
        first = run("off", clone(initial), own0)
    receipt["off_aot"] = json.loads(json.dumps(dt.NESTED_AOT_STATUS))
    assert all(row.get('loaded') for row in receipt['off_aot']['domains'].values()), 'OFF must load prepared programs'
    baseline = host(first.carries)
    with journal.phase('first_on'):
        default = run("on", clone(initial), own0)
    delta = differences(baseline, host(default.carries))
    receipt["default_difference"] = delta
    receipt["default_byte_exact"] = not delta
    receipt["default_one_ulp_class"] = all(row.get("max_ulp", 2) <= 1 for row in delta)
    receipt["on_aot"] = json.loads(json.dumps(dt.NESTED_AOT_STATUS))
    assert all(row.get('loaded') for row in receipt['on_aot']['domains'].values()), 'ON must load prepared programs'
    native_before = {str(p) for p in Path(receipt["cache"]).rglob("*.xlaexec")}
    with journal.phase('first_changed'):
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
        with journal.phase('ordinary_' + label):
            run(label, clone(first.carries), first.own_steps)
        # Global NESTED_AOT_STATUS can retain another variant's fused fast-path
        # row. Inspect actual cached callables for per-variant memory instead.
        calls = runtime_calls(runtimes[label])
        receipt.setdefault("retained_program_addresses", {})[label] = sorted(domain + ":" + key for domain, key in calls)
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
                    with jax.profiler.TraceAnnotation("PARAMS:" + label):
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
    result_path.write_text(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
