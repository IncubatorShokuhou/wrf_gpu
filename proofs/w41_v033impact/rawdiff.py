"""Raw input differences of the fields that feed the v0.3.x baked d02/d03 namelist leaves (no gpuwrf import)."""
import json, sys
import numpy as np
from netCDF4 import Dataset
C = "<DATA_ROOT>/wrf_gpu2/scratch/w41-v033impact/cases"
pairs = [("wn3_0227", "g2_20260707"), ("g2_20260702", "g2_20260707")]
WI = ["TMN", "VEGFRA", "SHDMAX", "SHDMIN", "ALBBCK", "LAI", "HGT", "MUB", "PHB", "PB", "T_INIT", "MAPFAC_M", "XLAT",
      "XLONG", "LANDMASK", "XLAND", "IVGTYP", "ISLTYP", "LU_INDEX", "SST", "TSK", "SNOALB", "ALBEDO", "EMISS",
      "VAR", "VAR_SSO", "OC12D", "SEAICE", "SINALPHA", "COSALPHA", "ZNU", "C1H", "C2H", "P_TOP", "Q2", "TSLB", "SMOIS"]
WL = ["SST", "VEGFRA", "ALBBCK", "SEAICE"]
out = {}
for a, b in pairs:
    rec = {}
    for d in ("d02", "d03"):
        with Dataset(f"{C}/{a}/wrfinput_{d}") as A, Dataset(f"{C}/{b}/wrfinput_{d}") as B:
            land = np.asarray(B["LANDMASK"][0]) > 0.5
            for v in WI:
                if v not in A.variables or v not in B.variables:
                    rec[f"{d}/wrfinput/{v}"] = "absent"; continue
                x, y = np.asarray(A[v][0], np.float64), np.asarray(B[v][0], np.float64)
                dd = x - y
                r = {"n_diff": int(np.count_nonzero(dd)), "n": int(dd.size), "max_abs": float(np.abs(dd).max()),
                     "mean": float(dd.mean())}
                if dd.shape == land.shape:
                    r["mean_land"] = float(dd[land].mean()) if land.any() else None
                    r["mean_water"] = float(dd[~land].mean()) if (~land).any() else None
                rec[f"{d}/wrfinput/{v}"] = r
        with Dataset(f"{C}/{a}/wrflowinp_{d}") as A, Dataset(f"{C}/{b}/wrflowinp_{d}") as B:
            xl = np.asarray(B["XLAND"][0]) if "XLAND" in B.variables else None
            for v in WL:
                x, y = np.asarray(A[v][:], np.float64), np.asarray(B[v][:], np.float64)
                dd = x - y
                r = {"n_diff": int(np.count_nonzero(dd)), "n": int(dd.size), "max_abs": float(np.abs(dd).max()),
                     "mean": float(dd.mean()), "per_record_max_abs": [round(float(np.abs(q).max()), 4) for q in dd]}
                if v == "SST":
                    water = ~land
                    valid = (y > 250) & (y < 350) & water[None]
                    r["water_mean_by_record"] = [round(float(q[w].mean()), 4) for q, w in zip(dd, valid)]
                    r["water_absmax_by_record"] = [round(float(np.abs(q[w]).max()), 4) for q, w in zip(dd, valid)]
                rec[f"{d}/wrflowinp/{v}"] = r
    out[f"{a}-{b}"] = rec
json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, rec in out.items():
    print("==", k)
    for f, r in rec.items():
        if isinstance(r, dict) and r["n_diff"]:
            print(f"  {f:28s} ndiff {r['n_diff']:>8}/{r['n']:<8} maxabs {r['max_abs']:.4g} mean {r['mean']:.4g}"
                  + (f" land {r.get('mean_land')} water {r.get('mean_water')}" if 'mean_land' in r else ""))
        elif isinstance(r, dict):
            print(f"  {f:28s} IDENTICAL")
        else:
            print(f"  {f:28s} {r}")
