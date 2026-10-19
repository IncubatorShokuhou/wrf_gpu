"""Worst-lead RMSE of the D6-bounded headline fields per domain from score.py's D6_dXX.json (report helper).
usage: python3 d6_summary.py CASE OUT.json"""
import json
import sys
from pathlib import Path

case, out = Path(sys.argv[1]), Path(sys.argv[2])
FIELDS = ("T2", "PSFC", "U10", "V10", "RAINNC", "T", "U", "V", "W", "QVAPOR")
res = {}
for dom in ("d01", "d02", "d03"):
    p = case / f"D6_{dom}.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    inv = json.loads((case / f"compare_{dom}.json").read_text()).get("inventory", {})
    row = {"verdict": d["verdict"], "paired_frames": d["paired_frames"], "failures": len(d["failures"]),
           "nonfinite": len(d["nonfinite"]), "incomplete": len(d["incomplete"]), "coverage_complete": d["coverage_complete"],
           "fields": len(d["fields"]), "inventory": {k: (len(v) if isinstance(v, list) else v) for k, v in inv.items()},
           "worst_lead_rmse": {}}
    for n in FIELDS:
        f = d["fields"].get(n)
        if not f:
            continue
        worst = max(f["by_lead"], key=lambda r: r["rmse"])
        row["worst_lead_rmse"][n] = {"rmse": worst["rmse"], "lead_h": worst.get("lead_h"), "verdict": f["verdict"]}
    res[dom] = row
out.write_text(json.dumps(res, indent=1) + "\n")
for dom, row in res.items():
    print(dom, row["verdict"], row["paired_frames"], "fail", row["failures"], "nonfinite", row["nonfinite"],
          {n: (round(v["rmse"], 4), v["lead_h"]) for n, v in row["worst_lead_rmse"].items()})
