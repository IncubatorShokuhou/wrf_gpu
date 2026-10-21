import json, glob, os, sys
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "docs")
pre = "state__manager__runs__mgr_wave_20261009__nightly-engine-148-a3__B__"
F = ["T2", "PSFC", "U10", "V10", "W", "U", "V", "T", "QVAPOR", "QCLOUD", "RAINNC", "PH", "MU", "HFX", "SWDOWN"]
LEADS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36, 42, 48, 60, 72]
out = {}
for dom in ["d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09"]:
    p = D + "/" + pre + ("fidelity24__" if dom == "d01" else "fidelity72_d02_d09__") + dom + ".json"
    d = json.load(open(p))
    fs = d["field_summaries"]
    out[dom] = {}
    for f in F:
        if f not in fs: continue
        bl = {r["lead_h"]: r for r in fs[f]["by_lead"]}
        out[dom][f] = {L: (bl[L]["rmse"], bl[L]["bias"]) for L in LEADS if L in bl}
json.dump(out, open(os.path.join(os.path.dirname(D), "..", "q1_timeline_rmse.json"), "w"), indent=0)
for f in F:
    print(f"\n{f:7s}", " ".join(f"{L:>6d}" for L in LEADS))
    for dom in out:
        if f in out[dom]:
            print(f"{dom:7s}", " ".join(f"{out[dom][f][L][0]:6.3g}" if L in out[dom][f] else "     -" for L in LEADS))
