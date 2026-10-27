"""Native Thompson component gate; launch only under the official GPU lock.

Run --phase prime and --phase measure in separate processes with the same
explicit cache directories. This measures synchronized component wall time,
not device time, a forecast root, S1, or operational AOT-key reuse. The caller
must run E41 on the cold cache before deleting it. --dry-run never imports JAX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile


CACHE_ENV = ("JAX_COMPILATION_CACHE_DIR", "GPUWRF_JAX_CACHE_DIR",
             "GPUWRF_XLA_AUTOTUNE_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH")
DEFAULTS = {"nt_c": 100.e6, "tno": 5., "ato": .304, "av_c": 31694600.}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("prime", "measure"), required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--ncol", type=int, default=24)
    parser.add_argument("--profile", action="store_true", help="CUDA capture range around warm timings only")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def persistent_entries(cache):
    """Persistent executable entries only; access-time files are not compiles."""
    return sorted(str(path.relative_to(cache)) for path in cache.rglob("*")
                  if path.is_file() and (path.name.endswith("-cache") or path.suffix == ".xlaexec"))


def main():
    args = arguments()
    source = Path(__file__).resolve().parents[2]
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert args.reps > 0 and args.ncol >= 5
    assert not args.profile or args.phase == "measure", "profiling requires the warm measure phase"
    assert args.fixture.is_file()
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"]), "dirty tracked source"
    relative_tool = Path(__file__).resolve().relative_to(source).as_posix()
    committed_tool = subprocess.check_output(["git", "-C", str(source), "show", f"{pin}:{relative_tool}"])
    assert hashlib.sha256(committed_tool).hexdigest() == sha(Path(__file__)), "tool must be committed at source pin"
    with zipfile.ZipFile(args.fixture) as archive:
        required = {name + ".npy" for name in ("names", "dt", *(
            "in_" + name for name in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "dz8w", "w")))}
        assert required <= set(archive.namelist()), "incomplete column fixture"
    caches = {name: os.environ.get(name) for name in CACHE_ENV}
    assert all(value and Path(value).is_absolute() for value in caches.values()), "explicit absolute cache paths required"
    assert Path(caches["JAX_COMPILATION_CACHE_DIR"]).resolve() == Path(caches["GPUWRF_JAX_CACHE_DIR"]).resolve()
    receipt = dict(source=str(source), pin=pin, tool_sha256=sha(Path(__file__)),
                   fixture=str(args.fixture.resolve()), fixture_sha256=sha(args.fixture),
                   phase=args.phase, profile=args.profile, pid=os.getpid(), cache=caches, gpu_executed=False,
                   scope="Native column component; no forecast/root/S1/AOT-key claim")
    receipt["environment"] = {name: os.environ.get(name) for name in (
        "JAX_PLATFORMS", "GPUWRF_FAST_DEFAULTS", "GPUWRF_JAX_CACHE", "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION")}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        assert os.environ.get("JAX_PLATFORMS") == "cpu"
        assert "jax" not in sys.modules and "gpuwrf" not in sys.modules
        receipt.update(status="DRY_RUN", jax_imported=False, numerical_validation=False)
        with (args.out / "dry-run.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
        print(json.dumps(receipt, indent=2))
        return

    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1", "official GPU lock required before JAX import"
    assert os.environ.get("GPUWRF_JAX_CACHE") == "1"
    assert os.environ.get("GPUWRF_FAST_DEFAULTS") == "1"
    run_dir = args.out / (args.phase + ("_profile" if args.profile else ""))
    assert not run_dir.exists(), "refuse to overwrite a component arm"
    if args.phase == "measure":
        prime = json.loads((args.out / "prime/receipt.json").read_text())
        assert prime["status"] == "PASS" and prime["pid"] != os.getpid(), "prime must finish in another process"
        for key in ("pin", "fixture_sha256", "cache", "tool_sha256", "environment"):
            assert prime[key] == receipt[key], f"prime/measure mismatch: {key}"
    run_dir.mkdir()
    cache = Path(caches["JAX_COMPILATION_CACHE_DIR"])
    before_process = persistent_entries(cache)
    try:
        sys.path.insert(0, str(source / "src"))
        import jax
        import jax.numpy as jnp
        import numpy as np
        from gpuwrf._fast_defaults import apply_fast_path_defaults, FAST_PATH_DEFAULTS
        apply_fast_path_defaults()
        from gpuwrf.kernels.phys_thompson_full import full_column
        from gpuwrf.physics import thompson_column as tc
        from gpuwrf.physics.thompson_parameters import validate_parameters

        device = jax.devices()[0]
        assert device.platform == "gpu", jax.devices()
        receipt.update(gpu_executed=True, platform=device.platform, device=str(device), jax_version=jax.__version__,
                       resolved_flags={name: os.environ.get(name) for name in FAST_PATH_DEFAULTS})
        z = dict(np.load(args.fixture, allow_pickle=False))
        assert all(np.isfinite(value).all() for key, value in z.items() if key.startswith(("in_", "wrf_")))
        candidates = np.flatnonzero(z["dt"] == 18.)
        assert len(candidates) >= args.ncol
        seeds = [int(candidates[np.argmax(np.max(z["in_" + v][candidates], axis=1))])
                 for v in ("qc", "qr", "qi", "qs", "qg")]
        spaced = candidates[np.linspace(0, len(candidates) - 1, args.ncol, dtype=int)].tolist()
        idx = np.array(list(dict.fromkeys(seeds + spaced))[:args.ncol])
        f = lambda name: jnp.asarray(z["in_" + name][idx], jnp.float32)
        temp, p, qv = f("th") * f("pii"), f("p"), f("qv")
        zero = jnp.zeros_like(qv)
        state = tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"),
                                      Ni=f("ni"), Nr=f("nr"), Ns=zero, Ng=zero, T=temp, p=p,
                                      rho=tc.density_from_pressure_temperature(p, temp, qv), dz=f("dz8w"), w=f("w"))
        receipt.update(ncol=len(idx), nz=state.qv.shape[-1], dt=18., selected_columns=z["names"][idx].tolist(),
                       input_nonzero={v: int(np.count_nonzero(z["in_" + v][idx])) for v in ("qc", "qr", "qi", "qs", "qg")})
        assert all(receipt["input_nonzero"].values())
        traces = []

        def program(state, parameters):
            traces.append(1)
            kwargs = {} if parameters is None else {"parameters": parameters}
            return full_column(state, 18., interpret=False, **kwargs)

        fn = jax.jit(program)
        controls = {"off": None, "default": validate_parameters(DEFAULTS),
                    "nt50": validate_parameters({**DEFAULTS, "nt_c": 50.e6}),
                    "nt150": validate_parameters({**DEFAULTS, "nt_c": 150.e6}),
                    "tno": validate_parameters({**DEFAULTS, "tno": 10.}),
                    "ato": validate_parameters({**DEFAULTS, "ato": .608}),
                    "av_c": validate_parameters({**DEFAULTS, "av_c": 63389200.})}
        jax.block_until_ready((state, controls))
        results = {}
        for label, parameters in controls.items():
            if label == "nt50":
                before_changed = persistent_entries(cache)
            started = time.perf_counter()
            result = jax.block_until_ready(fn(state, parameters))
            receipt.setdefault("first_call_s", {})[label] = time.perf_counter() - started
            arrays = [np.asarray(value).copy() for value in jax.tree.leaves(result)]
            assert all(np.isfinite(value).all() for value in arrays), f"nonfinite {label} output"
            results[label] = arrays
            np.savez_compressed(run_dir / (label + ".npz"), **{str(i): value for i, value in enumerate(arrays)})
        receipt.update(output_leaves=len(results["off"]), traces=len(traces),
                       changed_new_persistent_entries=sorted(set(persistent_entries(cache)) - set(before_changed)))
        receipt["default_differences"] = [dict(leaf=i, cells=int(np.count_nonzero(a != b)), maxabs=float(np.max(np.abs(a - b))))
            for i, (a, b) in enumerate(zip(results["off"], results["default"], strict=True)) if a.tobytes() != b.tobytes()]
        receipt["default_byte_exact"] = not receipt["default_differences"]
        from roundoff import compare
        receipt["default_roundoff"] = compare(results["off"], results["default"], nt_c=True)
        receipt["changed_leaves"] = {name: sum(a.tobytes() != b.tobytes() for a, b in zip(results["default"], results[name], strict=True))
                                    for name in ("nt50", "nt150", "tno", "ato", "av_c")}
        assert receipt["default_roundoff"]["within_roundoff"], receipt["default_roundoff"]
        assert len(traces) == 2 and not receipt["changed_new_persistent_entries"], "changed values specialized the program"
        assert all(receipt["changed_leaves"].values()), "inactive parameter control"
        # Existing on-device pre-mask counters are a separate diagnostic
        # executable. Its prognostic outputs must be BYTE to the measured
        # program before its counts can describe that program (E205).
        def program_events(state, parameters):
            kwargs = {} if parameters is None else {"parameters": parameters}
            return full_column(state, 18., interpret=False, return_events=True, **kwargs)

        event_fn = jax.jit(program_events)
        receipt["events"] = {}
        for label, parameters in controls.items():
            out, ppt, counters = jax.block_until_ready(event_fn(state, parameters))
            arrays = [np.asarray(value) for value in jax.tree.leaves((out, ppt))]
            assert all(np.isfinite(value).all() for value in arrays), label
            counts = np.asarray(counters)
            exact = all(a.tobytes() == b.tobytes() for a, b in zip(results[label], arrays, strict=True))
            receipt["events"][label] = {"prognostic_byte_exact": exact,
                "totals": dict(zip(("input_nonfinite", "inadmissible", "pre_mask_nonfinite"), counts.sum(axis=0).tolist()))}
            assert exact, "pre-mask observer changed prognostics; counters unqualified"
            assert receipt["events"][label]["totals"]["input_nonfinite"] == 0
            assert receipt["events"][label]["totals"]["pre_mask_nonfinite"] == 0
        receipt["inadmissible_events_need_review"] = any(row["totals"]["inadmissible"] for row in receipt["events"].values())
        for label in ("off", "default"):
            compiled = fn.lower(state, controls[label]).compile()
            stats = compiled.memory_analysis()
            receipt.setdefault("compiled_memory", {})[label] = {name: getattr(stats, name) for name in (
                "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes", "generated_code_size_in_bytes")}
            jax.block_until_ready(compiled(state, controls[label]))
        receipt["new_persistent_entries"] = sorted(set(persistent_entries(cache)) - set(before_process))
        if args.phase == "measure":
            assert not receipt["new_persistent_entries"], "warm process created new persistent executable entries"
            samples = {"off": [], "default": []}
            if args.profile:
                import ctypes
                cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
                assert cuda.cudaProfilerStart() == 0
            try:
                for rep in range(args.reps):
                    for label in (("off", "default") if rep % 2 == 0 else ("default", "off")):
                        # Host conversion/validation and operand creation are outside the timed ranges.
                        tag = "on" if label == "default" else "off"
                        with jax.profiler.TraceAnnotation("THPARAMS:component:" + tag):
                            started = time.perf_counter()
                            result = fn(state, controls[label])
                            jax.block_until_ready(result)
                            samples[label].append(time.perf_counter() - started)
            finally:
                if args.profile:
                    assert cuda.cudaProfilerStop() == 0
            receipt.update(samples_s=samples, median_s={label: statistics.median(values) for label, values in samples.items()})
            receipt["on_over_off"] = receipt["median_s"]["default"] / receipt["median_s"]["off"]
        assert len(traces) == 2
        receipt["combined_variants_device_memory_stats"] = device.memory_stats()
        receipt["host_vmhwm_kb"] = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                                            if line.startswith("VmHWM:")))
        receipt["status"] = "PASS"
    except Exception as error:
        receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
