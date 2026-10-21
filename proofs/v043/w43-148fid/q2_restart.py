"""Q2 CPU restart continuity: CPU a2 (uninterrupted to tau40) vs CPU a3 (restarted from the tau39 wrfrst) at tau40,
per domain and variable, inside one GuardedRead session (admission A-148). Also GPU vs CPU a3 at tau40 for scale.

usage: python analysis/q2_restart.py <label>  -> analysis/q2_restart.json
"""
import glob, json, os, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gread

S = "<DATA_ROOT>/alisios/scratch"
A2 = S + "/n148-cpuref-20261018_12z/cpu"
A3 = S + "/n148-cpuref-20261018_12z/cpu_a3"
GPU = S + "/n148-gpu-v042/runs/warm72/wrfout"
VARS = ["T2", "U10", "V10", "PSFC", "RAINNC", "W", "U", "V", "T", "QVAPOR", "QRAIN", "QCLOUD", "PH", "MU", "TSK", "SMOIS", "HFX"]


def one(pattern):
    m = sorted(glob.glob(pattern))
    return m[0] if len(m) == 1 else None


def main(label):
    import netCDF4
    trip = {}
    for k in range(1, 10):
        dom = f"d0{k}"
        a2 = one(f"{A2}/wrfout_{dom}_2026-10-20_10:00:*")
        a3 = one(f"{A3}/wrfout_{dom}_2026-10-20_10:00:*")
        g = one(f"{GPU}/wrfout_{dom}_2026-10-20_10:00:*")
        trip[dom] = (a2, a3, g)
    targets = sorted({os.path.realpath(p) for t in trip.values() for p in t if p})
    gread.use_add()
    sess, rdir = gread.session(targets, label)
    res = {}
    with sess:
        for dom, (a2, a3, g) in trip.items():
            row = {"files": [os.path.basename(p) if p else None for p in (a2, a3, g)]}
            if not (a2 and a3):
                res[dom] = row
                continue
            with netCDF4.Dataset(a2) as n2, netCDF4.Dataset(a3) as n3:
                n2.set_auto_mask(False); n3.set_auto_mask(False)
                ng = netCDF4.Dataset(g) if g else None
                if ng is not None:
                    ng.set_auto_mask(False)
                common = sorted(set(n2.variables) & set(n3.variables))
                ident = [v for v in common if np.array_equal(n2.variables[v][:], n3.variables[v][:], equal_nan=n2.variables[v].dtype.kind == "f")]
                row["n_common_vars"] = len(common)
                row["n_bitwise_identical"] = len(ident)
                row["differing_vars"] = [v for v in common if v not in ident]
                row["times"] = [str(netCDF4.chartostring(n2.variables["Times"][:])[0]), str(netCDF4.chartostring(n3.variables["Times"][:])[0])]
                for v in VARS:
                    if v not in n2.variables:
                        continue
                    x2 = n2.variables[v][:].astype(np.float64); x3 = n3.variables[v][:].astype(np.float64)
                    e = {"a3_minus_a2_rmse": float(np.sqrt(np.mean((x3 - x2) ** 2))), "a3_minus_a2_maxabs": float(np.abs(x3 - x2).max())}
                    if ng is not None and v in ng.variables:
                        xg = ng.variables[v][:].astype(np.float64)
                        if xg.shape == x3.shape:
                            e["gpu_minus_a3_rmse"] = float(np.sqrt(np.mean((xg - x3) ** 2)))
                            e["gpu_minus_a2_rmse"] = float(np.sqrt(np.mean((xg - x2) ** 2)))
                    row[v] = e
                if ng is not None:
                    ng.close()
            res[dom] = row
    gread.close_receipt(sess, rdir)
    json.dump(res, open(HERE / "q2_restart.json", "w"), indent=1)
    for dom, r in res.items():
        print(dom, r.get("n_bitwise_identical"), "/", r.get("n_common_vars"), "T2", r.get("T2"), "W", r.get("W"))


if __name__ == "__main__":
    main(sys.argv[1])
