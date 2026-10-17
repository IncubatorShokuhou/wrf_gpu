"""Read-only source/SHA/coverage audit of v-prep's exact C7 GPU receipts; no GPU execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

output = Path(sys.argv[1]).resolve()
base = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/C5_353a85f8a1_20261016")
head = "353a85f8a11abf4a12dcd78bad77ec7dc682a019"
result = {"head": head, "scope": "read-only review of actual-GPU receipts; no new GPU run",
          "read_utc": datetime.now(timezone.utc).isoformat(), "cold": {}, "full24": {}, "transfer": {}}
for case in ("Swiss", "PROD", "WN3-0227"):
    folder = base / "cold" / case
    run_path, gate_path = folder / "gpu_run.json", folder / "e41/wave_gate.json"
    run, gate = json.loads(run_path.read_text()), json.loads(gate_path.read_text())
    assert run["source_commit"] == head and run["platform"] == "gpu" and run["rc"] == 0
    assert gate["e41_gate"] == gate["hard_stack_gate"] == gate["coverage"]["status"] == "PASS"
    assert gate["coverage"]["n_discovered"] == gate["coverage"]["n_scanned"] == gate["n_entries"]
    assert set(gate["coverage"]["aot_programs"]) == set(gate["coverage"]["required_run_programs"])
    stacks = []
    for entry in gate["entries"]:
        assert not entry["kinput_transpose_over500"]
        assert entry["kinput_transpose_status"].startswith("MEASURED")
        for elf in entry["elfs"]:
            assert not elf["stack_over256"] and not elf["nonleaf_over1k"] and not elf["callgraph_unknown"]
            stacks.append(elf["max_stack_B"])
    result["cold"][case] = {"source": head, "gpu_rc": 0, "E41": "PASS",
                            "programs": gate["coverage"]["required_run_programs"], "entries": gate["n_entries"],
                            "max_stack_B": max(stacks),
                            "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
                            "gate_sha256": hashlib.sha256(gate_path.read_bytes()).hexdigest()}
    a_path, b_path = folder / "CUDA_TRACE.json", base / "baseline" / case / "CUDA_TRACE.json"
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    assert a["source_commit"] == head and b["source_commit"].startswith("e7e37390cc")
    assert a["profiled_root_steps"] == b["profiled_root_steps"]
    def copies(row):
        return {r["kind"]: (r["count"], r["bytes"]) for r in row["copies"]}
    def apis(row, key):
        return {r["name"]: r["count"] for r in row[key]}
    assert copies(a) == copies(b)
    assert apis(a, "host_blocking_synchronize_API") == apis(b, "host_blocking_synchronize_API")
    assert apis(a, "host_callback_API") == apis(b, "host_callback_API")
    result["transfer"][case] = {"roots": a["profiled_root_steps"], "copy_count_bytes_equal": True,
                                "synchronization_counts_equal": True, "callback_counts_equal": True,
                                "candidate_sha256": hashlib.sha256(a_path.read_bytes()).hexdigest(),
                                "baseline_sha256": hashlib.sha256(b_path.read_bytes()).hexdigest()}
    full = base / "full24" / case
    if (full / "gpu_run.json").exists():
        full_run = json.loads((full / "gpu_run.json").read_text())
        assert full_run["source_commit"] == head and full_run["platform"] == "gpu" and full_run["rc"] == 0
        rows = {}
        for path in sorted(full.glob("D6_d*.json")):
            d = json.loads(path.read_text())
            assert d["verdict"] == "PASS" and d["coverage_complete"] and d["paired_frames"] == 25
            assert not d["failures"] and not d["nonfinite"] and not d["incomplete"]
            assert "original CPU-WRF" in d["reference"]
            rows[path.name] = {"verdict": "PASS", "frames": 25, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        result["full24"][case] = rows or "PENDING"
    else:
        result["full24"][case] = "PENDING"
result["timing_status"] = "Matched ABAB <=1.03 gate is separate and pending at audit start; do not accept unmatched +5.73pct or infer pass from attribution."
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"GPU_E41": list(result["cold"]), "host_transfer_equal": list(result["transfer"]),
                  "full24": result["full24"], "timing": "pending matched ABAB"}), flush=True)
