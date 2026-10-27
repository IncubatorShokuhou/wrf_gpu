"""Guarded spot-check (A-O5-9NEST-148): the w43-148fid derived extracts equal the admitted wrfouts bit-for-bit on sampled
frames/fields, and the files' sha256 equal the admission list. Values are compared in memory only."""
import json, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gread148 as g

NPZ = Path("<DATA_ROOT>/wrf_gpu2/worktrees/w43-148fid/analysis/data")
S = "<DATA_ROOT>/alisios/scratch"
CPU = S + "/n148-cpuref-20261018_12z/cpu_reference_a2_a3"
GPU = S + "/n148-gpu-v042/runs/warm72/wrfout"
T0 = datetime(2026, 10, 18, 18)
SAMPLES = [("d06", 24), ("d04", 59), ("d02", 12), ("d06", 12), ("d04", 36), ("d05", 42)]
if len(sys.argv) > 2:  # e.g. d03:6,d07:5 LABEL
    SAMPLES = [(x.split(":")[0], int(x.split(":")[1])) for x in sys.argv[1].split(",")]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "S01b_npz_spotcheck"
FIELDS2 = ["RAINNC", "U10", "V10", "GRAUPELNC", "QRAINXY"]


def path_for(root, dom, lead):
    for p in os.listdir(root):
        if p.startswith(f"wrfout_{dom}_"):
            t = datetime.strptime(p.split("_", 2)[2], "%Y-%m-%d_%H:%M:%S")
            if abs((t - T0).total_seconds() - 3600 * lead) < 60:
                return os.path.realpath(os.path.join(root, p))
    raise FileNotFoundError((root, dom, lead))


def main():
    import hashlib, netCDF4
    targets = {(side, d, L): path_for(CPU if side == "cpu" else GPU, d, L) for d, L in SAMPLES for side in ("cpu", "gpu")}
    npz = {d: dict(np.load(NPZ / f"{d}.npz")) for d in {d for d, _ in SAMPLES}}  # derived reads BEFORE the session
    sess, rdir, listed = g.session(sorted(set(targets.values())), LABEL)
    res = {"checks": 0, "mismatch": [], "sha_ok": 0, "sha_bad": []}
    with sess:
        for (side, d, L), p in targets.items():
            with open(p, "rb") as fh:
                h = hashlib.sha256(fh.read()).hexdigest()
            if h == listed[p]["sha256"]:
                res["sha_ok"] += 1
            else:
                res["sha_bad"].append(p)
            z = npz[d]
            i = z["leads"].tolist().index(L)
            with netCDF4.Dataset(p) as nc:
                nc.set_auto_mask(False)
                for f in FIELDS2:
                    a = nc.variables[f][0].astype(np.float32)
                    res["checks"] += 1
                    if not np.array_equal(a, z[f"{side}_{f}"][i]):
                        res["mismatch"].append([side, d, L, f])
                for f in ("QRAIN", "QNRAIN", "QGRAUP"):
                    a = nc.variables[f][0].max(axis=0).astype(np.float32)
                    res["checks"] += 1
                    if not np.array_equal(a, z[f"{side}_{f}_colmax"][i]):
                        res["mismatch"].append([side, d, L, f + "_colmax"])
                a = nc.variables["W"][0].astype(np.float32)
                res["checks"] += 1
                if not np.array_equal(a, z[f"{side}_W"][i]):
                    res["mismatch"].append([side, d, L, "W"])
    g.close_receipt(sess, rdir, {"result": {k: (v if k != "mismatch" else v) for k, v in res.items()}})
    print(json.dumps(res))


if __name__ == "__main__":
    main()
