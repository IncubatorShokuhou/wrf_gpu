"""Changed bottom-only artifact parity/byte and CPU cost receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.diagnostics.thompson_radar import thompson_reflectivity


def main(output):
    assert jax.devices()[0].platform == "cpu"
    root = Path(__file__).resolve().parents[2]
    receipt = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
               "platform": "cpu", "scope": "changed bottom-only artifact; full operator unchanged; no forecast/GPU timing",
               "cache": "GPUWRF_JAX_CACHE=0"}
    fn = jax.jit(lambda *a: thompson_reflectivity(*a, ke_diag=1))
    for label, name in (("original40", "radar_oracle.npz"), ("active_bottom8", "radar_bottom_oracle.npz")):
        with np.load(Path(__file__).with_name(name)) as data:
            a, expected = data["inputs"], data["bottom"]
        args = tuple(jnp.asarray(a[:, k]) for k in (0, 2, 3, 4, 5, 6, 8, 9))
        start = time.perf_counter()
        compiled = fn.lower(*args).compile()
        compile_s = time.perf_counter() - start
        actual = np.asarray(compiled(*args))
        error = float(np.max(np.abs(actual - expected)))
        assert error <= 5e-5
        if label == "original40":
            # The previous qualified bottom artifact was BYTE to this pristine
            # fixture (RAD03), so this carries forward its exact output identity.
            np.testing.assert_array_equal(actual, expected)
        times = []
        for _ in range(7):
            start = time.perf_counter()
            compiled(*args).block_until_ready()
            times.append(time.perf_counter() - start)
        memory = compiled.memory_analysis()
        receipt[label] = {"max_abs_dbz": error, "cold_compile_s": compile_s,
                          "warm_median_s": float(np.median(times)), "temporary_bytes": memory.temp_size_in_bytes,
                          "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest()}
        if label == "original40":
            population = args
    # Changed bottom-only cost at the same footprint/population as RAD03.
    ncol = 42 * 42
    args = tuple(jnp.tile(a, (int(np.ceil(ncol / len(a))), 1))[:ncol] for a in population)
    start = time.perf_counter()
    compiled = fn.lower(*args).compile()
    cold = time.perf_counter() - start
    compiled(*args).block_until_ready()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        compiled(*args).block_until_ready()
        times.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    receipt["cpu_42x42_bottom_component"] = {"warm_median_s": float(np.median(times)), "cold_compile_s": cold,
                                            "temporary_bytes": memory.temp_size_in_bytes,
                                            "argument_bytes": memory.argument_size_in_bytes,
                                            "output_bytes": memory.output_size_in_bytes,
                                            "population": "same24real+16constructed column population as RAD03"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
