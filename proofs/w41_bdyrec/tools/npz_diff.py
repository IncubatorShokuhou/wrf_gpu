"""Bitwise diff of two step_compare dumps. usage: python npz_diff.py A.npz B.npz [--with-bdy]"""
import json
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
with_bdy = "--with-bdy" in sys.argv
keys = sorted(set(a.files) | set(b.files))
diff, missing, compared, skipped = [], [], 0, 0
for key in keys:
    if not with_bdy and "_bdy" in key:
        skipped += 1
        continue
    if key not in a.files or key not in b.files:
        missing.append(key)
        continue
    x, y = a[key], b[key]
    compared += 1
    if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
        entry = {"leaf": key, "shape": [list(x.shape), list(y.shape)]}
        if x.shape == y.shape and np.issubdtype(x.dtype, np.floating):
            entry["max_abs"] = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            entry["n_diff"] = int((x != y).sum())
        diff.append(entry)
print(json.dumps({"compared": compared, "skipped_bdy": skipped, "missing": missing, "n_diff_leaves": len(diff),
                  "diff": diff[:20]}, indent=1))
