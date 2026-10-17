"""CPU-only changed-artifact oracle/cost receipt for online Thompson radar.

Run from an immutable snapshot, JAX_PLATFORMS=cpu, GPUWRF_JAX_CACHE=0,
CPU29 nice19. Timing is operator-only CPU evidence, never GPU speed/forecast
qualification. Memory analysis is for this CPU compiled artifact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.diagnostics.thompson_radar import thompson_reflectivity, final_graupel_number


def main(output):
    assert jax.devices()[0].platform == "cpu"
    root = Path(__file__).resolve().parents[2]
    fixture = Path(__file__).with_name("radar_oracle.npz")
    manifest = json.loads(fixture.with_name("radar_oracle_manifest.json").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    with np.load(fixture) as data:
        arrays = data["inputs"].copy()
        expected = data["dbz"].copy()
        bottom_expected = data["bottom"].copy()
        ng_inputs, ng_expected = data["ng_inputs"].copy(), data["balanced_ng"].copy()
    args = tuple(jnp.asarray(arrays[:, k]) for k in (0, 2, 3, 4, 5, 6, 8, 9))
    receipt = {"platform": "cpu", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "fixture_sha256": manifest["fixture_sha256"], "scope": "original operator parity; CPU changed-artifact timing only; no coupled-forecast/GPU claim",
               "cache": "GPUWRF_JAX_CACHE=0; no persistent JAX cache"}
    for label, kdiag, truth in (("full", None, expected), ("bottom", 1, bottom_expected)):
        fn = jax.jit(lambda *a: thompson_reflectivity(*a, ke_diag=kdiag))
        start = time.perf_counter()
        compiled = fn.lower(*args).compile()
        cold = time.perf_counter() - start
        actual = np.asarray(compiled(*args))
        error = float(np.max(np.abs(actual - truth)))
        assert error <= manifest["tolerance"]["max_absolute_dbz"], (label, error)
        times = []
        for _ in range(7):
            start = time.perf_counter()
            compiled(*args).block_until_ready()
            times.append(time.perf_counter() - start)
        memory = compiled.memory_analysis()
        receipt[label] = {"max_abs_dbz": error, "cold_compile_s": cold, "warm_median_s": float(np.median(times)),
                          "columns": len(arrays), "levels": arrays.shape[-1],
                          "argument_bytes": memory.argument_size_in_bytes,
                          "output_bytes": memory.output_size_in_bytes, "temporary_bytes": memory.temp_size_in_bytes}
    actual_ng = np.asarray(jax.jit(final_graupel_number)(*(jnp.asarray(ng_inputs[:, k]) for k in range(3))))
    ulp = np.abs(actual_ng.view("i4").astype("i8") - ng_expected.view("i4").astype("i8"))
    receipt["final_ng"] = {"max_ulp": int(ulp.max()), "max_abs": float(np.max(np.abs(actual_ng - ng_expected)))}
    assert ulp.max() <= manifest["tolerance"]["ng_max_ulp"]
    # A real 42x42 horizontal footprint with the qualified active-regime column
    # population repeated for bounded component timing, not a forecast surrogate.
    ncol = 42 * 42
    large = tuple(jnp.tile(a, (int(np.ceil(ncol / len(a))), 1))[:ncol] for a in args)
    fn = jax.jit(thompson_reflectivity)
    start = time.perf_counter()
    compiled = fn.lower(*large).compile()
    cold = time.perf_counter() - start
    compiled(*large).block_until_ready()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        compiled(*large).block_until_ready()
        times.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    receipt["cpu_42x42_component"] = {"warm_median_s": float(np.median(times)), "cold_compile_s": cold,
                                      "argument_bytes": memory.argument_size_in_bytes,
                                      "output_bytes": memory.output_size_in_bytes,
                                      "temporary_bytes": memory.temp_size_in_bytes,
                                      "source_population": "24 original history-field +16 active constructed columns repeated; not forecast timing"}
    receipt["context_liveness"] = {"extra_real_3d_arrays": 2, "bytes_42x42x44": 2 * 42 * 42 * 44 * 4,
                                    "resident_output_bytes_42x42x44": 42 * 42 * 44 * 4,
                                    "gpu_shape": "one column per Pallas program; width64 for44 levels;50 serial bins; no bins x domain tensor",
                                    "gpu_compile_nsight_cubin_gate": "pending manager GPU window"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
