"""Bitwise comparison of two gpu_ntu_run.py NPZ outputs (CPU, NumPy only).
usage: ntu_npz_compare.py A.npz B.npz OUT_JSON"""
import json
import sys
import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert sorted(a.files) == sorted(b.files)
diff = {}
for k in a.files:
    x, y = a[k], b[k]
    if x.tobytes() != y.tobytes():
        bad = (x.view(np.uint32) != y.view(np.uint32)) if x.dtype == np.float32 else (x != y)
        both_nan = np.isnan(x) & np.isnan(y) if x.dtype.kind == 'f' else np.zeros_like(bad)
        diff[k] = dict(n=int(bad.sum()), nan_bits_only=int((bad & both_nan).sum()),
                       max_abs=float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64))[bad & ~both_nan])) if (bad & ~both_nan).any() else 0.0)
r = dict(arrays=len(a.files), bitwise_equal=not diff, differing=diff)
json.dump(r, open(sys.argv[3], 'w'), indent=2)
print(json.dumps(r)[:600])
