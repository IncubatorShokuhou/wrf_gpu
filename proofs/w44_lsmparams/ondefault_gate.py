"""Apply the pre-registered GPU ON-default gate (receipts/GPU_ONDEFAULT_GATE.json) to root_gpu receipts.

Reads default_field_table of every given prime/measure receipt; a field passes when max abs / field max is
finite and <= the bound. Exit 0 = every field of every domain in class, 1 = stop and report (no re-tuning).
Usage: ondefault_gate.py --gate receipts/GPU_ONDEFAULT_GATE.json <receipt.json> [...]
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--gate", type=Path, required=True)
p.add_argument("receipts", type=Path, nargs="+")
a = p.parse_args()
gate = json.loads(a.gate.read_text())
bound = float(gate["bound_max_abs_over_field_max"])
report = {"gate": str(a.gate), "gate_sha256": hashlib.sha256(a.gate.read_bytes()).hexdigest(), "bound": bound,
          "receipts": []}
for path in a.receipts:
    value = json.loads(path.read_text())
    tables = value.get("default_field_table")
    entry = {"receipt": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "platform": value.get("platform"), "gpu_executed": value.get("gpu_executed"),
             "family": value.get("family"), "phase": value.get("phase"), "domains": {}}
    if not tables:
        entry["verdict"] = "MISSING_TABLE"
    else:
        for domain, table in tables.items():
            above = [r for r in table["rows"] if not (math.isfinite(r["max_abs_over_field_max"])
                                                      and r["max_abs_over_field_max"] <= bound)]
            entry["domains"][domain] = {"fields": len(table["rows"]), "max_ratio": table["max_ratio"],
                                        "above": [{k: r.get(k) for k in ("field", "cells", "ulp_median", "ulp_max",
                                                  "max_abs", "field_abs_max", "max_abs_over_field_max")} for r in above]}
        entry["verdict"] = "IN_CLASS" if not any(d["above"] for d in entry["domains"].values()) else "STOP"
    report["receipts"].append(entry)
report["verdict"] = "IN_CLASS" if all(r["verdict"] == "IN_CLASS" for r in report["receipts"]) else "STOP"
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["verdict"] == "IN_CLASS" else 1)
