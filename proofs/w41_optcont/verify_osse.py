"""Score the GPU OSSE smoke against the pre-registered gates (proofs/w41_optcont/REGISTERED_GATES.json).

  python proofs/w41_optcont/verify_osse.py ROOT OUT.json      (CPU, seconds; reads only ROOT)
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

GATES = json.loads((Path(__file__).parent/"REGISTERED_GATES.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, out):
    root = Path(root)
    run = root/"run"
    result = json.loads((run/"result.json").read_text())
    ledger = json.loads((run/"ledger.json").read_text())
    devices, sources = {}, {}
    for key, entry in sorted(ledger["members"].items()):
        proof = json.loads((Path(entry["files"][0]["path"]).parent/"proofs"/"nested_pipeline_run.json").read_text())
        # The run proof names the device as 'cuda:0' and the GPU model under gpu_architecture.name (fix 2026-10-17
        # 23:5xZ: the first scoring read only 'device' and reported G5 false for RTX 5090 members).
        arch = proof.get("gpu_architecture") or {}
        devices[key] = f"{proof.get('device')} {arch.get('name')}" if str(proof.get("device", "")).startswith("cuda") else None
        doms = (proof.get("metadata") or {}).get("nested_aot", {}).get("domains") or {}
        sources[key] = {d: r.get("source") for d, r in doms.items()}
    program, twin, ev = result["program"], result["twin"], result["evaluation"]
    start = datetime.fromisoformat((root/"gpu_start_utc.txt").read_text().strip().replace("Z", "+00:00"))
    end = datetime.fromisoformat((root/"gpu_end_utc.txt").read_text().strip().replace("Z", "+00:00"))
    gpu_minutes = (end-start).total_seconds()/60
    osse = GATES["osse"]
    verdicts = {
        "G1_recovery": twin["verdict"] == "RECOVERED",
        "G2_misfit": all(ev[s]["posterior"]["normalised_misfit"] < ev[s]["prior"]["normalised_misfit"]
                         for s in ("train", "test")),
        "G3_one_program": (program["verdict"] == "ONE_PROGRAM" and program["cold_or_unknown_members"] == ["truth|d15"]
                           and program["warm_members"] == program["members"]-1),
        "G4_budget": result["scheduled_case_hours"] == osse["planned_case_hours"] == result["planned_case_hours"]
                     and gpu_minutes <= 20.0,
        "G5_device": bool(devices) and all(d and "5090" in str(d) for d in devices.values()),
    }
    summary = {"scope": "GPU OSSE smoke, Swiss d01, SYNTHETIC held-out date (same initial state); twin, no skill claim",
               "gates_registered": GATES["registered_utc"], "verdicts": verdicts,
               "all_pass": all(verdicts.values()), "gpu_minutes_lock_wall": round(gpu_minutes, 2),
               "truth": twin["truth_multipliers"], "estimate": result["estimate_multipliers"],
               "abs_log_error": twin["abs_log_error"], "posterior_sd_log": result["posterior_sd_log"],
               "misfit_history": [h["normalised_misfit"] for h in result["iterations"]],
               "evaluation": ev, "program": program, "devices": sorted(set(map(str, devices.values()))),
               "aot_sources": {k: v for k, v in sources.items()},
               "member_walls_s": {k: e.get("wall_s") for k, e in ledger["members"].items()},
               "scheduled_case_hours": result["scheduled_case_hours"], "planned_case_hours": result["planned_case_hours"],
               "result_sha256": sha(run/"result.json"), "ledger_sha256": sha(run/"ledger.json")}
    Path(out).write_text(json.dumps(summary, indent=1)+"\n")
    print(json.dumps({"verdicts": verdicts, "estimate": result["estimate_multipliers"], "gpu_minutes": summary["gpu_minutes_lock_wall"]}))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(verify(*sys.argv[1:]))
