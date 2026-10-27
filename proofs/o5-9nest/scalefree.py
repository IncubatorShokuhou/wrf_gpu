"""Scale-free realization distance per field/domain/lead: pattern correlation r and RMSE/std(reference).

#148 X (GPU v0.4.2 vs CPU 2x2) from the verified w43-148fid extracts; CPU-vs-CPU twins C: 0420 (results/cpu_twin_0420.json),
0227 and ERA5 E-C (w43-teacher scorer JSONs: pearson_r per lead; their std is not stored, so only r there).
Two runs that are both valid realizations decorrelate with lead; if GPU-vs-CPU on #148 decorrelated MORE than CPU-vs-CPU on
other convective days at the same lead, that would point to (a). Regime differs between cases, so this is [I] context only.

usage: JAX_PLATFORMS=cpu python scalefree.py <npz_dir> <out.json>
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEACH = "<USER_HOME>/wrf_gpu2_lanes/w43-teacher/score"
FIELDS = ["W", "U10", "V10", "RAINNC"]


def main(npz_dir, out_path):
    out = {"x148": {}, "c0420": {}, "c0227": {}, "cera5": {}, "x0227": {}, "xera5": {}}
    for dom in ("d02", "d04", "d05", "d06"):
        z = dict(np.load(Path(npz_dir) / f"{dom}.npz"))
        for f in FIELDS:
            rows = {}
            for i, L in enumerate(z["leads"].tolist()):
                c = z[f"cpu_{f}"][i].astype(np.float64).ravel()
                g = z[f"gpu_{f}"][i].astype(np.float64).ravel()
                if c.std() == 0 or g.std() == 0:
                    continue
                rows[L] = {"r": float(np.corrcoef(c, g)[0, 1]), "rmse_over_std": float(np.sqrt(np.mean((g - c) ** 2)) / c.std())}
            out["x148"][f"{dom}/{f}"] = rows
        del z
    t = json.load(open(HERE / "results" / "cpu_twin_0420.json"))
    for dom in t:
        for f in FIELDS:
            out["c0420"][f"{dom}/{f}"] = {int(L): {"r": v[f].get("pearson_r"),
                                                   "rmse_over_std": (v[f]["rmse"] / v[f]["std_C"]) if v[f].get("std_C") else None}
                                         for L, v in t[dom].items() if v[f].get("pearson_r") is not None}
    for key, src in (("c0227", "C_0227"), ("x0227", "wn3_20260227"), ("cera5", "C_era5_20240302"), ("xera5", "era5_20240302")):
        for dom in ("d02", "d03"):
            fs = json.load(open(f"{TEACH}/{src}/x/{dom}.json"))["field_summaries"]
            for f in FIELDS:
                out[key][f"{dom}/{f}"] = {r["lead_h"]: {"r": r["pearson_r"]} for r in fs[f]["by_lead"] if r["pearson_r"] is not None}
    Path(out_path).write_text(json.dumps(out, indent=0))

    def summ(d, lo, hi, k="r"):
        v = [x[k] for L, x in d.items() if lo <= int(L) <= hi and x.get(k) is not None]
        return (round(min(v), 3), round(st.median(v), 3)) if v else None
    print("field  window   x148 d04/d06 (min,med r)         c0420 d03   c0227 d03   cera5 d03   x0227 d03   xera5 d03")
    for f in FIELDS:
        for lo, hi in ((12, 24), (25, 48), (49, 72)):
            print(f, f"τ{lo}-{hi}", summ(out["x148"][f"d04/{f}"], lo, hi), summ(out["x148"][f"d06/{f}"], lo, hi),
                  summ(out["c0420"].get(f"d03/{f}", {}), lo, hi), summ(out["c0227"][f"d03/{f}"], lo, hi),
                  summ(out["cera5"][f"d03/{f}"], lo, hi), summ(out["x0227"][f"d03/{f}"], lo, hi), summ(out["xera5"][f"d03/{f}"], lo, hi))


if __name__ == "__main__":
    main(*sys.argv[1:3])
