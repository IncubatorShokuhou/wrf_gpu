"""Apply the pre-authorized pristine-ensemble floor to retained measurements.

Writes a successor report; preserves all measured arrays/field and energy
results and the original failed ratio verdicts without re-running the model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import math

p = argparse.ArgumentParser()
p.add_argument("--report", type=Path, required=True)
p.add_argument("--floor", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
report, floor = json.loads(a.report.read_text()), json.loads(a.floor.read_text())
assert report["oracle_sha256"] == floor["oracle_sha256"]
changes, original_failures, failures = [], 0, 0
for name, arm in report["arms"].items():
    indexed = {r["name"]: r for r in floor["arms"][name]["columns"]}
    for row in arm["closure"]:
        prior = row["water_pass"]
        original_failures += not prior
        absolute = indexed[row["column"]]["floor_mm"]
        native, pristine = row["native_water_mm"], row["pristine_water_mm"]
        assert all(math.isfinite(x) and x >= 0 for x in (absolute, native, pristine))
        limit = max(1.5 * pristine, absolute)
        row.update(original_ratio_pass=prior, water_floor_mm=absolute, water_limit_mm=min(limit, .1),
                   water_pass=bool(native <= limit and native < .1))
        failures += not row["water_pass"]
        changes.append({"parameter": name, **row})
    arm["closure_pass"] = all(r["water_pass"] and r["energy_pass"] for r in arm["closure"])
report["water_rule"] = floor["binding_rule"]
report["manager_gate_amendment"] = {"authority": floor["manager_authority"],
    "original_report": str(a.report), "original_report_sha256": hashlib.sha256(a.report.read_bytes()).hexdigest(),
    "floor_report": str(a.floor), "floor_sha256": hashlib.sha256(a.floor.read_bytes()).hexdigest(),
    "members_per_column": floor["members"], "original_ratio_failures": original_failures,
    "remaining_water_failures": failures, "model_reexecuted": False}
report["pass"] = report["one_key"] and report["traces"] == 2 and all(
    arm["field_pass"] and arm["closure_pass"] for arm in report["arms"].values())
a.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"pass": report["pass"], "original_ratio_failures": original_failures,
                  "remaining_failures": failures, "columns": changes}), flush=True)
assert report["pass"]
