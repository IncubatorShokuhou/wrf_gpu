"""GPU-vs-GPU impact census (regression/impact evidence, NOT fidelity): every wrfout variable of arm X vs reference
arm R, per domain and lead. Exact byte equality per variable, max |diff|, RMSE, and land/water split for key fields.

usage: python census.py OUT_JSON REF_DIR:LABEL X_DIR:LABEL [X_DIR:LABEL ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

KEY = ("T2", "U10", "V10", "TSK", "SST", "PSFC", "Q2", "RAINNC", "HFX", "LH", "TSLB", "SMOIS", "T", "U", "V", "W",
       "QVAPOR", "PH", "MU", "VEGFRA", "SHDFAC", "TMN")
LEADS = (1, 6, 24)


def files(d):
    out = {}
    for p in sorted(Path(d).glob("wrfout_d0*")):
        dom = p.name[7:10]
        out.setdefault(dom, []).append(p)
    return out


def stats(a, b, mask=None):
    d = a.astype(np.float64) - b.astype(np.float64)
    if mask is not None:
        m = np.broadcast_to(mask, d.shape)
        d = d[m]
    if d.size == 0:
        return None
    return {"rmse": float(np.sqrt(np.mean(d * d))), "max_abs": float(np.abs(d).max()), "mean": float(d.mean()),
            "n_diff": int(np.count_nonzero(d))}


out_json = Path(sys.argv[1])
ref_dir, ref_label = sys.argv[2].split(":")
result = {"label": "GPU-vs-GPU impact census (not fidelity)", "reference": ref_label, "arms": {}}
ref_files = files(ref_dir)
for spec in sys.argv[3:]:
    xd, xl = spec.split(":")
    arm = {}
    for dom, rlist in sorted(ref_files.items()):
        xmap = {p.name: p for p in files(xd).get(dom, [])}
        per_lead = []
        for lead, rp in enumerate(rlist, start=1):
            xp = xmap.get(rp.name)
            if xp is None:
                per_lead.append({"file": rp.name, "missing": True})
                continue
            rec = {"file": rp.name, "lead_index": lead, "identical_vars": 0, "differing": {}}
            with Dataset(rp) as R, Dataset(xp) as X:
                land = np.asarray(R["LANDMASK"][0]) > 0.5 if "LANDMASK" in R.variables else None
                for v in R.variables:
                    if v not in X.variables or v == "Times":
                        continue
                    a, b = np.asarray(X[v][:]), np.asarray(R[v][:])
                    if a.shape != b.shape:
                        rec["differing"][v] = {"shape_mismatch": [list(a.shape), list(b.shape)]}
                        continue
                    if a.dtype.kind not in "fiu":
                        same = np.array_equal(a, b)
                    else:
                        same = np.array_equal(a, b, equal_nan=True)
                    if same:
                        rec["identical_vars"] += 1
                        continue
                    s = stats(a, b)
                    if v in KEY and land is not None and a.shape[-2:] == land.shape and lead in LEADS:
                        s["land"] = stats(a, b, land)
                        s["water"] = stats(a, b, ~land)
                    rec["differing"][v] = s
            per_lead.append(rec)
        arm[dom] = per_lead
    result["arms"][xl] = arm
    for dom, per_lead in sorted(arm.items()):
        nd = [len(r.get("differing", {})) for r in per_lead]
        print(f"{xl} vs {ref_label} {dom}: frames {len(per_lead)} differing-vars per frame {nd}")
        for r in per_lead:
            if r.get("lead_index") in LEADS:
                for v in ("T2", "U10", "V10", "TSK", "SST", "TSLB"):
                    s = r["differing"].get(v)
                    if s:
                        print(f"   lead {r['lead_index']:>2} {v:5s} rmse {s['rmse']:.3g} max {s['max_abs']:.3g}"
                              + (f" land {s['land']['rmse']:.3g} water {s['water']['rmse']:.3g}" if s.get('land') else ""))
out_json.write_text(json.dumps(result, indent=1))
