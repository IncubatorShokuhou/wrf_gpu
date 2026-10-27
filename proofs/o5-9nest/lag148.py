"""Root boundary-clock signature on #148 (o5-coast FINDING 15:27Z, method of their a4_mu_lag.py, PSFC only):
per ring 0..9 + interior, fit mean_ring(GPU - CPU)(τ) = lag * mean_ring(dPSFC_cpu/dt)(τ) over early frames (chaos small),
dPSFC/dt = central hourly difference / 7200 s. Bug signature: d01 rings 1-9 lag ~ +1 root dt (54 s), ring 0 ~ 0, d02 ~ +30 s.
usage: python lag148.py <npz_dir> <max_lead> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np


def rings(ny, nx):
    y, x = np.mgrid[0:ny, 0:nx]
    return np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x))


out = {}
for dom in ("d01", "d02", "d04", "d05", "d06"):
    z = np.load(Path(sys.argv[1]) / f"{dom}.npz")
    leads = z["leads"].tolist()
    c, g = z["cpu_PSFC"].astype(np.float64), z["gpu_PSFC"].astype(np.float64)
    rg = rings(*c.shape[1:])
    res = {}
    for name, m in [(f"ring{r}", rg == r) for r in range(10)] + [("interior", rg >= 10)]:
        d, t = [], []
        for i, L in enumerate(leads):
            if 1 <= L <= int(sys.argv[2]) and 0 < i < len(leads) - 1 and leads[i + 1] - leads[i - 1] == 2:
                d.append((g[i] - c[i])[m].mean())
                t.append(((c[i + 1] - c[i - 1]) / 7200.0)[m].mean())
        d, t = np.array(d), np.array(t)
        lag = float((d * t).sum() / (t * t).sum())
        r = float(np.corrcoef(d, t)[0, 1]) if d.std() > 0 and t.std() > 0 else None
        res[name] = {"lag_s": round(lag, 2), "r": None if r is None else round(r, 3), "mean_abs_diff_Pa": round(float(np.abs(d).mean()), 4), "n": len(d)}
    out[dom] = res
    print(dom, " ".join(f"{k}:{v['lag_s']}s(r{v['r']})" for k, v in res.items()))
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
