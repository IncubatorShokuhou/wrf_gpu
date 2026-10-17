"""Compile captured CAM StableHLO on CPU for the captured CUDA target, no GPU."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mlir", type=Path, required=True)
    p.add_argument("--options", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    a.out.mkdir(parents=True, exist_ok=False)
    for key, name in (("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"), ("CUDA_CACHE_PATH", "cuda")):
        directory = a.out / name
        directory.mkdir()
        os.environ[key] = str(directory)
    import gpuwrf  # noqa: F401
    import jax
    from jax._src.interpreters import mlir
    from jax.experimental import topologies
    from jaxlib import xla_client as xc
    from jaxlib.mlir import ir
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import wave_cubin_gate as scan
    from e41 import hlo_scan

    assert all(d.platform == "cpu" for d in jax.devices())
    raw = a.mlir.read_bytes()
    if a.mlir.suffix == ".gz":
        raw = gzip.decompress(raw)
    original_options = a.options.read_bytes()
    options = xc.CompileOptions.ParseFromString(original_options)
    # Only diagnostics differ; normal tuning is reused, never performed on GPU.
    diagnostic = {"xla_dump_to": str(a.out / "dump"),
                  "xla_gpu_dump_autotune_results_to": str(a.out / "autotune_results.pb"),
                  "xla_gpu_require_complete_aot_autotune_results": True}
    options.env_option_overrides = [(k, v) for k, v in options.env_option_overrides
                                    if k not in diagnostic] + list(diagnostic.items())
    target = a.target.read_bytes()
    topology = topologies.get_topology_desc(platform="cuda", target_config=target.decode(), topology="1x1x1")
    device = topology.devices[0]
    receipt = {"mlir_sha256": hashlib.sha256(raw).hexdigest(),
               "original_options_sha256": hashlib.sha256(original_options).hexdigest(),
               "effective_options_sha256": hashlib.sha256(options.SerializeAsString()).hexdigest(),
               "diagnostic_overrides": diagnostic,
               "target_sha256": hashlib.sha256(target).hexdigest(),
               "physical_backend": "cpu", "virtual_target": str(device),
               "scope": "deviceless native static prescreen; no GPU execution/performance claim"}
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    start = time.monotonic()
    with mlir.make_ir_context():
        module = ir.Module.parse(raw)
        executable = device.client.compile(module, xc.DeviceList(tuple(topology.devices)), options)
    receipt["compile_s"] = time.monotonic() - start
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    blob = executable.serialize()
    native_dir = a.out / "cache/aot/captured"
    native_dir.mkdir(parents=True)
    (native_dir / "d01.xlaexec").write_bytes(blob)
    hlo = executable.hlo_modules()[0].to_string()
    (a.out / "optimized.hlo.gz").write_bytes(gzip.compress(hlo.encode(), mtime=0))
    receipt.update(native_bytes=len(blob), native_sha256=hashlib.sha256(blob).hexdigest(),
                   optimized_hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest())
    try:
        memory = executable.get_compiled_memory_stats()
        receipt["memory"] = {key: getattr(memory, key) for key in (
            "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    except Exception as exc:
        if "UNIMPLEMENTED: buffer_assignment is not supported" not in str(exc):
            raise
        receipt["memory_api"] = str(exc)
        reports = list((a.out / "dump").glob("*after_optimizations-memory-usage-report.txt"))
        if len(reports) != 1:
            raise RuntimeError("missing/ambiguous XLA allocation report") from exc
        match = re.search(r"Total bytes: (\d+)", reports[0].read_text())
        if not match:
            raise RuntimeError("XLA allocation report has no total bytes") from exc
        receipt["memory"] = {"total_allocation_bytes": int(match.group(1)), "source": str(reports[0])}
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    gate = scan.run_gate(a.out / "cache", a.out / "e41")
    hlo_result = hlo_scan.scan_text(hlo)
    transpose_rows = [r for r in hlo_result["transpose_hero_fusions"] if r["kind"] == "kInput"]
    receipt["kinput_transpose_over500"] = [(r["fusion"], r["n_instr"]) for r in transpose_rows if r["n_instr"] > 500]
    # CompileOnly marks the exact real-GPU kInput bodies as kLoop (measured
    # 658/1165/513 matches). Require all transpose bodies <=500 as a stronger
    # local prescreen; the merged scanner's formal native verdict is unchanged.
    receipt["all_transpose_prescreen_over500"] = [r for r in hlo_result["transpose_hero_fusions"] if r["n_instr"] > 500]
    receipt["real_gpu_kind_prescreen"] = "FAIL" if receipt["all_transpose_prescreen_over500"] else "PASS"
    receipt["hlo_stats"] = {"computations": hlo_result["computations"],
                            "n_fusions": hlo_result["n_fusions"],
                            "kinput_transpose_fusions": len(transpose_rows),
                            "max_kinput_transpose_instr": max((r["n_instr"] for r in transpose_rows), default=0)}
    receipt["stack_gate"] = gate["hard_stack_gate"]
    receipt["max_stack_B"] = max(e["max_stack_B"] for r in gate["entries"] for e in r["elfs"])
    receipt["conservative_flags"] = gate["conservative_flags"]
    receipt["hlo_transpose_gate"] = "FAIL" if receipt["kinput_transpose_over500"] else "PASS"
    receipt["native_gate"] = gate["e41_gate"]
    receipt["coverage"] = gate["coverage"]
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if gate["hard_stack_gate"] == "FAIL" or receipt["kinput_transpose_over500"] or receipt["all_transpose_prescreen_over500"]:
        return 2
    return 0 if gate["e41_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
