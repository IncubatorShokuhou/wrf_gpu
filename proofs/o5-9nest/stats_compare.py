"""Distribution check per failing field/domain/lead: two valid realizations differ point-wise but share statistics.

For W (all levels), U10, V10 and hourly RAINNC increments: spatial std, max |x| (W: also max updraft / downdraft), and the
grid-scale fraction gs = var(x - box3(x)) / var(x) (box3 = 3x3 horizontal mean, edge-replicated), i.e. how much of the
variance sits at ~2-3 dx. A GPU field with systematically larger grid-scale fraction or variance than CPU-WRF at the same
leads would be numerical noise / an engine difference (a); matching statistics with different positions = realization (b).

usage: JAX_PLATFORMS=cpu python stats_compare.py <npz_dir> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROWS = {"d04": ["W", "U10", "V10", "RAIN1H"], "d06": ["W", "U10", "V10", "RAIN1H"], "d05": ["RAIN1H"], "d02": ["RAIN1H"]}


def box3(a):
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    s = np.zeros_like(a)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            s += p[..., dy:dy + a.shape[-2], dx:dx + a.shape[-1]]
    return s / 9.0


def stats(x):
    x = x.astype(np.float64)
    v = x.var()
    hp = x - box3(x)
    return {"std": float(np.sqrt(v)), "maxabs": float(np.abs(x).max()), "max": float(x.max()), "min": float(x.min()),
            "gs_frac": float(hp.var() / v) if v > 0 else None, "p99abs": float(np.percentile(np.abs(x), 99))}


def main(npz_dir, out_path):
    out = {}
    for dom, fields in ROWS.items():
        z = dict(np.load(Path(npz_dir) / f"{dom}.npz"))
        leads = z["leads"].tolist()
        for f in fields:
            rows = {}
            for i, L in enumerate(leads):
                if f == "RAIN1H":
                    if i == 0:
                        continue
                    c = z["cpu_RAINNC"][i] - z["cpu_RAINNC"][i - 1]
                    g = z["gpu_RAINNC"][i] - z["gpu_RAINNC"][i - 1]
                else:
                    c, g = z[f"cpu_{f}"][i], z[f"gpu_{f}"][i]
                if not (np.isfinite(c).all() and np.isfinite(g).all()):
                    raise ValueError("non-finite")
                rows[L] = {"cpu": stats(c), "gpu": stats(g)}
            out[f"{dom}/{f}"] = rows
        del z
    Path(out_path).write_text(json.dumps(out, indent=0))
    for key, rows in out.items():
        ls = sorted(rows)
        sr = np.array([rows[L]["gpu"]["std"] / rows[L]["cpu"]["std"] for L in ls if rows[L]["cpu"]["std"] > 0])
        gc = np.array([rows[L]["cpu"]["gs_frac"] for L in ls if rows[L]["cpu"]["gs_frac"] is not None and rows[L]["gpu"]["gs_frac"] is not None])
        gg = np.array([rows[L]["gpu"]["gs_frac"] for L in ls if rows[L]["cpu"]["gs_frac"] is not None and rows[L]["gpu"]["gs_frac"] is not None])
        print(f"{key:10s} std GPU/CPU median {np.median(sr):.3f} p10 {np.percentile(sr,10):.3f} p90 {np.percentile(sr,90):.3f} "
              f"frac>1 {np.mean(sr>1):.2f} | grid-scale frac CPU med {np.median(gc):.3f} GPU med {np.median(gg):.3f} frac GPU>CPU {np.mean(gg>gc):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
