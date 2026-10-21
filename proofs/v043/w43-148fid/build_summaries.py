"""Assemble per-question JSONs (Q1..Q6, SCHEMES) from q_dXX.json, q2_restart.json, q1_parent_footprint.json, scorer docs."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
DOMS = ["d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09"]
q = {d: json.load(open(H / f"q_{d}.json")) for d in DOMS}
lab = "[M] guarded reads A-148/ADD1; derived from analysis/data/dXX.npz; RMSE reproduces the ALISIOS scorer exactly (max |diff| 0.0 on W/U10/V10/RAINNC/T2, all leads)"
# Q1
onset = {}
for d in DOMS:
    q4 = {r["lead"]: r for r in q[d]["Q4"]["by_lead"]}
    on1 = [L for L, r in q4.items() if max(r["cpu_max_hourly"], r["gpu_max_hourly"]) >= 1.0]
    onset[d] = {"rain_ge1mmh_first_lead": on1[0] if on1 else None,
                "first_exceed": {f: q[d]["Q1"][f]["first_exceed"] for f in ("W", "U10", "V10", "RAINNC")}}
q1 = {"label": lab, "question": "Q1 timeline/hierarchy", "hierarchy": {"parent_id": [1, 1, 2, 2, 2, 2, 2, 2, 2], "dx_km": [9, 3, 1, 1, 1, 1, 1, 1, 1],
      "dt_s": [54, 18, 6, 6, 6, 6, 6, 6, 6], "feedback": 0, "physics": "WN3 default suite on all domains (mp8, RRTMG 4/4, sfclay5, NoahMP4, MYNN5, KF d01 only)"},
      "onset_vs_divergence": onset, "parent_footprint": json.load(open(H / "q1_parent_footprint.json")),
      "region_by_lead": {d: {f: q[d]["Q1"][f]["by_lead"] for f in ("W", "U10", "V10", "RAINNC", "T2", "PSFC") if f in q[d]["Q1"]} for d in DOMS},
      "scorer_timeline": json.load(open(H / "q1_timeline_rmse.json"))}
json.dump(q1, open(H / "Q1.json", "w"), indent=0)
json.dump({"label": "[M] guarded R05_q2: CPU a2 (uninterrupted) vs CPU a3 (restart from tau39 wrfrst) at tau40", **json.load(open(H / "q2_restart.json"))}, open(H / "Q2.json", "w"), indent=0)
json.dump({"label": lab, "mask": "heavy = hourly RAINNC increment >= 1 mm/h in CPU or GPU this or previous hour, dilated 3 km; steep = |grad HGT| > 0.10",
           "by_domain": {d: q[d]["Q3"] for d in DOMS}}, open(H / "Q3.json", "w"), indent=0)
pool = {}
for d in DOMS:
    q4 = {r["lead"]: r for r in q[d]["Q4"]["by_lead"]}
    for L in (24, 48, 72):
        p = pool.setdefault(L, {"cpu_ge10": 0, "gpu_ge10": 0, "cpu_ge25": 0, "gpu_ge25": 0})
        for k in p:
            p[k] += q4[L][k]
json.dump({"label": lab, "pooled_d02_d09": pool, "by_domain": {d: q[d]["Q4"] for d in DOMS}}, open(H / "Q4.json", "w"), indent=0)
json.dump({"label": lab, "by_domain": {d: q[d]["Q5"] for d in DOMS}}, open(H / "Q5.json", "w"), indent=0)
print("ok")
