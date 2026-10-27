"""Per-lead difference statistics (rmse, max_abs, bias, pearson r) of a candidate vs an ALISIOS reference, two-phase so that the
reference is read only inside a guarded session (CPU only, summary numbers only):
  phase 1 (outside):  pair_stats.py prep CAND_DIR DOMAIN VARS(,) OUT.pkl            -> candidate arrays (our files)
  phase 2 (in session via guarded_run.py): pair_stats.py stats REF_DIR CAND.pkl OUT.json
Statistics use the frozen scorer's conventions for these numbers (float64, all cells, finite pairs only; E200: non-finite -> abort).
"""
import glob, json, os, pickle, sys

import numpy as np

if sys.argv[1] == "prep":
    from netCDF4 import Dataset
    cand, dom, vs, out = sys.argv[2], sys.argv[3], sys.argv[4].split(","), sys.argv[5]
    rec = {}
    for p in sorted(glob.glob(f"{cand}/wrfout_{dom}_*")):
        with Dataset(p) as ds:
            rec[os.path.basename(p)] = {v: np.ma.asarray(ds[v][:], dtype=np.float64).filled(np.nan) for v in vs}
    pickle.dump({"dom": dom, "vars": vs, "frames": rec}, open(out, "wb"), protocol=4)
    print("prepared", len(rec))
elif sys.argv[1] == "stats":
    from netCDF4 import Dataset   # inside the session: the guarded wrapper
    ref, pk, out = sys.argv[2], sys.argv[3], sys.argv[4]
    c = pickle.load(open(pk, "rb"))
    res = {"ref": ref, "dom": c["dom"], "by_frame": {}}
    for i, (n, arrs) in enumerate(sorted(c["frames"].items())):
        with Dataset(os.path.join(ref, n)) as ds:
            row = {}
            for v in c["vars"]:
                a = np.ma.asarray(ds[v][:], dtype=np.float64).filled(np.nan); b = arrs[v]
                assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), (n, v)
                d = b - a
                sa, sb = a.std(), b.std()
                r = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else None
                row[v] = {"rmse": float(np.sqrt((d * d).mean())), "max_abs": float(np.abs(d).max()), "bias": float(d.mean()),
                          "r": r, "ref_max": float(a.max()), "cand_max": float(b.max())}
            res["by_frame"][n] = row
    json.dump(res, open(out, "w"), indent=1)
    print("stats", len(res["by_frame"]))
