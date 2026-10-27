"""Per-lead RMSE curves (CPU twin C and GPU X vs the same CPU reference) from w43-teacher scorer JSONs (own lane outputs)."""
import json, sys
S = "<USER_HOME>/wrf_gpu2_lanes/w43-teacher/score"
FIELDS = ["W", "U10", "V10", "RAINNC", "T2", "PSFC"]
PAIRS = {"0227": ("C_0227", "wn3_20260227"), "era5_0302": ("C_era5_20240302", "era5_20240302")}
out = {}
for case, (c, x) in PAIRS.items():
    for dom in ("d01", "d02", "d03"):
        dc = json.load(open(f"{S}/{c}/x/{dom}.json"))["field_summaries"]
        dx = json.load(open(f"{S}/{x}/x/{dom}.json"))["field_summaries"]
        for f in FIELDS:
            bc = {r["lead_h"]: r["rmse"] for r in dc[f]["by_lead"]}
            bx = {r["lead_h"]: r["rmse"] for r in dx[f]["by_lead"]}
            out[f"{case}/{dom}/{f}"] = {"C": bc, "X": bx}
json.dump(out, open(sys.argv[1], "w"), indent=0)
