"""GPU vs original CPU-WRF CLDFRA / radiation agreement for the icloud=3 composite arm (report only).

usage: JAX_PLATFORMS=cpu python cldfra_report.py CASE OUT.json
CLDFRA, SWDOWN and GLW are unbounded in the frozen D6 manifest (hydrometeor_and_cloud / radiation
attribution families), so this is disclosure, not a gate. Pairs CASE/cpu and CASE/gpu wrfout files by
name (sealed original Times), per domain and lead: domain means, bias, RMSE, pattern correlation,
cloudy (> 0.01) and partial (0 < cf < 1) cell fractions, column-max CLDFRA RMSE.
"""
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CASE, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def stats(c, g):
    c, g = np.asarray(c, np.float64), np.asarray(g, np.float64)
    d = g - c
    corr = float(np.corrcoef(c.ravel(), g.ravel())[0, 1]) if c.std() > 0 and g.std() > 0 else None
    return {"cpu_mean": float(c.mean()), "gpu_mean": float(g.mean()), "bias": float(d.mean()),
            "rmse": float(np.sqrt((d * d).mean())), "corr": corr, "nonfinite_gpu": int((~np.isfinite(g)).sum())}


out = {"case": str(CASE), "note": "report only: CLDFRA/SWDOWN/GLW have no frozen D6 bound", "domains": {}}
for dom in ("d01", "d02", "d03"):
    rows = []
    for cf in sorted((CASE / "cpu").glob(f"wrfout_{dom}_*")):
        gf = CASE / "gpu" / cf.name
        if not gf.exists():
            rows.append({"file": cf.name, "missing_gpu": True})
            continue
        with Dataset(cf) as c, Dataset(gf) as g:
            cc, gc = np.asarray(c["CLDFRA"][0]), np.asarray(g["CLDFRA"][0])
            row = {"file": cf.name, "CLDFRA": stats(cc, gc),
                   "cloudy_frac": [float((cc > .01).mean()), float((gc > .01).mean())],
                   "partial_frac": [float(((cc > 0) & (cc < 1)).mean()), float(((gc > 0) & (gc < 1)).mean())],
                   # icloud=2 (cal_cldfra2) is binary: any value other than 0/1 in either arm is a wiring defect
                   "nonbinary_count": [int(((cc != 0) & (cc != 1)).sum()), int(((gc != 0) & (gc != 1)).sum())],
                   "colmax_CLDFRA": stats(cc.max(axis=0), gc.max(axis=0))}
            for name in ("SWDOWN", "GLW"):
                row[name] = stats(c[name][0], g[name][0])
        rows.append(row)
    out["domains"][dom] = rows
    fc = [r for r in rows[1:] if "CLDFRA" in r]
    print(dom, "frames", len(rows), "forecast CLDFRA rmse max %.4f mean %.4f | colmax rmse mean %.4f | GLW rmse mean %.2f"
          % (max(r["CLDFRA"]["rmse"] for r in fc), np.mean([r["CLDFRA"]["rmse"] for r in fc]),
             np.mean([r["colmax_CLDFRA"]["rmse"] for r in fc]), np.mean([r["GLW"]["rmse"] for r in fc])), flush=True)
OUT.write_text(json.dumps(out, indent=1) + "\n")
