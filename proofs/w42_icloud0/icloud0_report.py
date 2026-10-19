"""GPU vs original CPU-WRF icloud=0 radiation report (w42-icloud0; disclosure, the gate is the frozen D6 in post.sh).

usage: JAX_PLATFORMS=cpu python icloud0_report.py CASE OUT.json
Checks per domain and frame: history CLDFRA and CLDT are 0 in BOTH arms after the first radiation call (WRF
radiation_driver.F:1309-1318 with icloud=0; the t=0 frame is wrfinput's CLDFRA); radiation is cloud-free in both arms:
SWDOWN vs SWDOWNC, GLW vs LWDNBC and SWCF/LWCF (absent fields reported as such). GPU-CPU stats (mean, bias, RMSE,
pattern correlation) for SWDOWN, GLW, SWDDIR, SWDDIF, OLR (unbounded radiation fields) and the cloud species.
"""
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CASE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RAD = ("SWDOWN", "GLW", "SWDDIR", "SWDDIF", "OLR", "SWDNB", "LWDNB")
CLEAR = (("SWDOWN", "SWDOWNC"), ("GLW", "LWDNBC"), ("SWDNB", "SWDNBC"), ("LWDNB", "LWDNBC"))
SPECIES = ("QCLOUD", "QICE", "QSNOW")


def stats(c, g):
    c, g = np.asarray(c, np.float64), np.asarray(g, np.float64)
    d = g - c
    corr = float(np.corrcoef(c.ravel(), g.ravel())[0, 1]) if c.std() > 0 and g.std() > 0 else None
    return {"cpu_mean": float(c.mean()), "gpu_mean": float(g.mean()), "bias": float(d.mean()),
            "rmse": float(np.sqrt((d * d).mean())), "corr": corr, "nonfinite_gpu": int((~np.isfinite(g)).sum())}


def field(ds, name):
    return np.asarray(ds[name][0], np.float64) if name in ds.variables else None


out = {"case": str(CASE), "note": "report only: CLDFRA and radiation fluxes have no frozen D6 bound", "domains": {}}
for dom in ("d01", "d02", "d03"):
    rows = []
    for k, cf in enumerate(sorted((CASE / "cpu").glob(f"wrfout_{dom}_*"))):
        gf = CASE / "gpu" / cf.name
        if not gf.exists():
            rows.append({"file": cf.name, "missing_gpu": True})
            continue
        with Dataset(cf) as c, Dataset(gf) as g:
            row = {"file": cf.name, "frame": k}
            for arm, ds in (("cpu", c), ("gpu", g)):
                row[f"{arm}_cldfra_max"] = float(np.max(field(ds, "CLDFRA")))
                cldt = field(ds, "CLDT")
                row[f"{arm}_cldt_max"] = None if cldt is None else float(np.max(cldt))
                row[f"{arm}_clear_minus_all_max"] = {
                    f"{a}-{b}": (None if field(ds, a) is None or field(ds, b) is None
                                 else float(np.max(np.abs(field(ds, a) - field(ds, b))))) for a, b in CLEAR}
                row[f"{arm}_cf_max"] = {n: (None if field(ds, n) is None else float(np.max(np.abs(field(ds, n)))))
                                        for n in ("SWCF", "LWCF")}
            row["rad"] = {n: stats(field(c, n), field(g, n)) for n in RAD if n in c.variables and n in g.variables}
            row["species"] = {n: stats(field(c, n), field(g, n)) for n in SPECIES
                              if n in c.variables and n in g.variables}
            rows.append(row)
    fc = [r for r in rows if r.get("frame", 0) > 0 and not r.get("missing_gpu")]
    out["domains"][dom] = {
        "frames": rows,
        "forecast_frames": len(fc),
        "cldfra_zero_both_arms_all_forecast_frames": bool(fc) and all(
            r["cpu_cldfra_max"] == 0.0 and r["gpu_cldfra_max"] == 0.0 for r in fc),
        "clouds_present_in_microphysics": bool(fc) and any(
            r["species"].get("QCLOUD", {}).get("cpu_mean", 0) > 0 for r in fc),
        "worst": {n: {"rmse": max(r["rad"][n]["rmse"] for r in fc), "abs_bias": max(abs(r["rad"][n]["bias"]) for r in fc)}
                  for n in RAD if fc and all(n in r["rad"] for r in fc)},
    }
OUT.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({d: {k: v for k, v in s.items() if k != "frames"} for d, s in out["domains"].items()}))
