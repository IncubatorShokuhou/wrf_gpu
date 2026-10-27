"""Convective intensity symmetry GPU vs CPU over all 1 km leaves (d03-d09), all leads with active convection
(max W > 5 m/s in either run): fraction of leads with GPU max W > CPU max W, median ratio of p99.9 |W| and of hourly
rain p99, per leaf and pooled. Realization -> ~0.5 / ~1; a systematic GPU intensity deficit (#149-like) -> < 0.5 / < 1.
usage: JAX_PLATFORMS=cpu python intensity_symmetry.py <npz_dir> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
out = {}
pooled = {"w_gt": [], "w_ratio": [], "r_ratio": [], "r_gt": []}
for dom in ("d03", "d04", "d05", "d06", "d07", "d08", "d09"):
    z = np.load(Path(sys.argv[1]) / f"{dom}.npz")
    wc, wg = z["cpu_W"], z["gpu_W"]
    rc, rg = np.diff(z["cpu_RAINNC"], axis=0), np.diff(z["gpu_RAINNC"], axis=0)
    rows = []
    for i, L in enumerate(z["leads"].tolist()):
        mc, mg = float(wc[i].max()), float(wg[i].max())
        if max(mc, mg) <= 5.0:
            continue
        pc, pg = float(np.percentile(np.abs(wc[i]), 99.9)), float(np.percentile(np.abs(wg[i]), 99.9))
        row = {"lead": L, "maxW_cpu": mc, "maxW_gpu": mg, "p999W_cpu": pc, "p999W_gpu": pg}
        if i > 0:
            qc, qg = float(np.percentile(rc[i - 1], 99)), float(np.percentile(rg[i - 1], 99))
            row.update(rain1h_p99_cpu=qc, rain1h_p99_gpu=qg)
            if max(qc, qg) > 1.0:
                pooled["r_ratio"].append(qg / qc if qc > 0 else np.inf); pooled["r_gt"].append(qg > qc)
        pooled["w_gt"].append(mg > mc); pooled["w_ratio"].append(pg / pc)
        rows.append(row)
    gt = [r["maxW_gpu"] > r["maxW_cpu"] for r in rows]
    out[dom] = {"n_active_leads": len(rows), "frac_gpu_maxW_gt": float(np.mean(gt)) if rows else None,
                "median_p999W_ratio": float(np.median([r["p999W_gpu"] / r["p999W_cpu"] for r in rows])) if rows else None,
                "sum_maxW_cpu": float(sum(r["maxW_cpu"] for r in rows)), "sum_maxW_gpu": float(sum(r["maxW_gpu"] for r in rows)), "rows": rows}
    print(dom, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[dom].items() if k != "rows"})
    del z, wc, wg
n = len(pooled["w_gt"])
k = int(sum(pooled["w_gt"]))
from math import comb
p2 = min(1.0, 2 * sum(comb(n, j) for j in range(0, min(k, n - k) + 1)) / 2 ** n)
rr = [x for x in pooled["r_ratio"] if np.isfinite(x)]
out["pooled"] = {"n_active_leads": n, "frac_gpu_maxW_gt": k / n, "binomial_two_sided_p": p2,
                 "median_p999W_ratio": float(np.median(pooled["w_ratio"])),
                 "n_rain_leads": len(pooled["r_gt"]), "frac_gpu_rain1h_p99_gt": float(np.mean(pooled["r_gt"])),
                 "median_rain1h_p99_ratio": float(np.median(rr))}
print("pooled", out["pooled"])
Path(sys.argv[2]).write_text(json.dumps(out, indent=0))
