"""Manager-scheduled changed-artifact timing: replay the first CAM step in process.

Each timed replay uses exactly the same immutable input carry and step number,
so radiation executes every time. Compilation and the first call are excluded.
This is a coupled Swiss component measurement, not a forecast speed claim.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    assert os.environ.get("JAX_PLATFORMS") == "cuda"
    sys.path.insert(0, str(a.snapshot / "src"))
    import gpuwrf  # noqa: F401
    import jax
    assert jax.devices()[0].platform == "gpu"
    sys.path.insert(0, str(a.snapshot / "scripts"))
    import gpu_smoke_v034 as smoke
    from gpuwrf.runtime import domain_tree
    from gpuwrf.runtime import aot_executable
    from gpuwrf.runtime import operational_mode
    donation = operational_mode._CARRY_DONATE
    from capture_gpu import capture_inputs
    from jax._src import compiler

    a.out.mkdir(parents=True, exist_ok=True)
    call = domain_tree.run_operational_domain_tree
    measured = []
    replay_overhead = []
    bindings = []
    serialize = aot_executable.serialize

    def retain_aot_hlo(compiled, **kwargs):
        # Observe the exact compiled object BEFORE AOT metadata compaction.
        # The delegate's returned bytes/meta are forwarded unchanged. This
        # binds full optimized HLO to the actual compact AOT blob, rather than
        # inferring HLO identity from equal native ELF alone (E188).
        hlo = compiled._executable.xla_executable.hlo_modules()[0].to_string().encode()
        blob, meta = serialize(compiled, **kwargs)
        digest = hashlib.sha256(blob).hexdigest()
        path = a.out / f"aot-{digest}.optimized.hlo.gz"
        path.write_bytes(gzip.compress(hlo, mtime=0))
        binding = {"aot_blob_sha256": digest,
                   "optimized_hlo_sha256": hashlib.sha256(hlo).hexdigest(),
                   "optimized_hlo_file": str(path), "delegate_output_unchanged": True}
        existing = next((b for b in bindings if b["aot_blob_sha256"] == digest), None)
        if existing is None:
            bindings.append(binding)
        else:
            assert existing == binding, "one AOT blob bound to conflicting optimized HLO"
        (a.out / "aot_hlo_bindings.json").write_text(json.dumps(bindings, indent=2) + "\n")
        return blob, meta

    def replay(*args, **kwargs):
        seed = dict(kwargs["initial_own_steps"])
        # Retain the real release donation setting. A master copy stays outside
        # donated calls; each replay receives fresh copies prepared and waited
        # BEFORE its timer. Copies preserve values, dtypes and committed layout.
        clone = lambda c: jax.tree.map(lambda v: v.copy() if isinstance(v, jax.Array) else v, c)
        master = clone(kwargs["carries"]) if not measured and donation else None
        if master is not None:
            jax.block_until_ready(master)
        result = call(*args, **kwargs)
        jax.block_until_ready(result.carries)
        if not measured:
            window_start = time.perf_counter()
            for _ in range(9):
                replay_kwargs = kwargs
                if donation:
                    fresh = clone(master)
                    jax.block_until_ready(fresh)
                    replay_kwargs = {**kwargs, "carries": fresh}
                start = time.perf_counter()
                repeated = call(*args, **replay_kwargs)
                jax.block_until_ready(repeated.carries)
                measured.append(time.perf_counter() - start)
                assert dict(repeated.own_steps) == dict(result.own_steps)
            replay_overhead.append(time.perf_counter() - window_start)
            assert kwargs["initial_own_steps"] == seed == {"d01": 0}
            record = {"platform": jax.devices()[0].platform, "device": str(jax.devices()[0]),
                      "scope": "warm coupled CAM first-step replay, identical input and radiation cadence",
                      "replay_s": measured, "median_s": statistics.median(measured),
                      "initial_own_steps": seed, "result_own_steps": dict(result.own_steps),
                      "carry_donation": donation, "input_clone_preparation_excluded": donation,
                      "memory_scope": "benchmark allocator includes master/replay input copies; not forecast peak",
                      "compilation_excluded": True, "forecast_speed_claim": False,
                      "allocator_memory_stats": jax.devices()[0].memory_stats()}
            (a.out / "timing.json").write_text(json.dumps(record, indent=2) + "\n")
        return result

    domain_tree.run_operational_domain_tree = replay
    aot_executable.serialize = retain_aot_hlo
    original = compiler.backend_compile_and_load
    compiler.backend_compile_and_load = capture_inputs(a.out / "compiler_inputs", original)
    try:
        status = smoke.child(argparse.Namespace(child="ra3_cam", out=str(a.out / "run"),
                                                cpu_dry_run=False, trace_only=False, steps=3))
        # The smoke's outer first-step clock surrounds our replay wrapper.
        # Disclose that raw clock and remove the measured extra replay window.
        path = a.out / "run/child.json"
        record = json.loads(path.read_text())
        record["raw_first_step_with_replays_s"] = record["step_s"][0]
        record["replay_window_s"] = replay_overhead[0]
        record["step_s"][0] -= replay_overhead[0]
        record["compile_plus_first_step_s"] = record["step_s"][0]
        path.write_text(json.dumps(record, indent=2) + "\n")
        return status
    finally:
        compiler.backend_compile_and_load = original
        domain_tree.run_operational_domain_tree = call
        aot_executable.serialize = serialize


if __name__ == "__main__":
    raise SystemExit(main())
