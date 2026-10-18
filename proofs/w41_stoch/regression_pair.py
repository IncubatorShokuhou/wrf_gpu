"""GPU-vs-GPU REGRESSION check (never fidelity evidence): per frame and variable, are two GPU runs of the
same namelist byte-identical? Used for st4 (serial RNG) vs st4p (parallel RNG). usage: regression_pair.py A B out.json"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import netCDF4
a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows, differ, only = [], {}, {}
for fa in sorted(a.glob('wrfout_d01_*')):
    fb = b / fa.name
    with netCDF4.Dataset(fa) as x, netCDF4.Dataset(fb) as y:
        x.set_auto_mask(False); y.set_auto_mask(False)
        names = sorted(set(x.variables) | set(y.variables))
        same = 0
        for n in names:
            if n not in x.variables or n not in y.variables:
                only.setdefault(n, 'A' if n in x.variables else 'B'); continue
            u, v = np.asarray(x[n][:]), np.asarray(y[n][:])
            if u.shape == v.shape and u.tobytes() == v.tobytes():
                same += 1
            else:
                d = float(np.max(np.abs(u.astype(np.float64) - v.astype(np.float64)))) if u.shape == v.shape and u.dtype.kind in 'fi' else None
                r = differ.setdefault(n, dict(frames=0, max_abs=0.0)); r['frames'] += 1
                if d is not None: r['max_abs'] = max(r['max_abs'], d)
        rows.append(dict(file=fa.name, identical=same, compared=len([n for n in names if n in x.variables and n in y.variables])))
res = dict(schema='w41-stoch-gpu-regression-v1', label='GPU-vs-GPU regression only; not fidelity', a=str(a), b=str(b),
           frames=len(rows), all_identical=not differ, differing=differ, only_in=only, by_frame=rows)
out.write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps(dict(frames=len(rows), all_identical=not differ, n_differing=len(differ), only_in=only,
                      worst=sorted(differ.items(), key=lambda kv: -kv[1]['max_abs'])[:8]), indent=1))
