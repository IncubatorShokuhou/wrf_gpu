"""Q1/Q3/Q4/Q5 analysis on analysis/data/<dom>.npz (derived, guarded extraction). CPU only, numpy only.

usage: python analysis/analyze.py d02 d04 d05 d06 ...   -> analysis/q_<dom>.json
Conventions: rings counted from the outer edge of the mass grid (ring 0 = WRF spec zone for nests, spec_zone=1;
rings 1..4 = relax zone (spec_bdy_width 5); interior = ring >= 5). W RMSE over all levels/cells like the scorer.
"""
import json, sys
import numpy as np
from pathlib import Path
from scipy import ndimage

HERE = Path(__file__).resolve().parent
TOL = {"W": 0.3, "U10": 1.5, "V10": 1.5, "RAINNC": 1.0, "T2": 1.5, "PSFC": 120.0}
DX_KM = {"d01": 9.0, "d02": 3.0}


def rings(shape):
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    return np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x))


def rmse(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a))) if a.size else None


def mass_err(z, field):
    """squared error per mass column (W: summed over levels, de-staggered count kept for the scorer mean)."""
    d = (z[f"gpu_{field}"].astype(np.float64) - z[f"cpu_{field}"].astype(np.float64))
    return d


def analyze(dom):
    z = dict(np.load(HERE / "data" / f"{dom}.npz"))
    leads = z["leads"].tolist()
    shp = z["HGT"].shape
    rg = rings(shp)
    dx = DX_KM.get(dom, 1.0)
    hgt = z["HGT"]
    land = z["LANDMASK"] > 0.5
    gy, gx = np.gradient(hgt, dx * 1000.0)
    slope = np.hypot(gy, gx)
    steep = slope > 0.10
    out = {"dom": dom, "leads": leads, "shape": list(shp), "n_land": int(land.sum()), "n_steep": int(steep.sum())}
    regions = {"ring0": rg == 0, "relax1_4": (rg >= 1) & (rg <= 4), "interior": rg >= 5}

    # ---- Q1 per-lead RMSE by region
    q1 = {}
    for f in ("U10", "V10", "T2", "PSFC", "RAINNC", "W"):
        if f"gpu_{f}" not in z:
            continue
        d = mass_err(z, f)
        rows = []
        for i, L in enumerate(leads):
            di = d[i]
            row = {"lead": L, "all": rmse(di)}
            for r, m in regions.items():
                row[r] = rmse(di[..., m]) if f != "W" else rmse(di[:, m])
            rows.append(row)
        first = {}
        for frac in (0.1, 0.5, 1.0):
            for r in ("all", "ring0", "relax1_4", "interior"):
                hit = [row["lead"] for row in rows if row[r] is not None and row[r] > frac * TOL[f]]
                first[f"{r}>{frac}tol"] = hit[0] if hit else None
        q1[f] = {"by_lead": rows, "first_exceed": first}
    out["Q1"] = q1

    # ---- rain masks (hourly increments)
    rc, rg_ = z["cpu_RAINNC"].astype(np.float64), z["gpu_RAINNC"].astype(np.float64)
    inc_c = np.diff(rc, axis=0, prepend=rc[:1])
    inc_g = np.diff(rg_, axis=0, prepend=rg_[:1])
    r_km = 3.0
    it = max(1, int(round(r_km / dx)))
    struct = ndimage.generate_binary_structure(2, 1)

    # ---- Q3: error share near heavy-rain cells (either run, >= 1 mm/h this or previous hour), W/U10/V10
    q3 = {}
    for f in ("W", "U10", "V10"):
        d = mass_err(z, f)
        rows = []
        for i, L in enumerate(leads):
            heavy = (inc_c[i] >= 1.0) | (inc_g[i] >= 1.0)
            if i > 0:
                heavy |= (inc_c[i - 1] >= 1.0) | (inc_g[i - 1] >= 1.0)
            near = ndimage.binary_dilation(heavy, struct, iterations=it) if heavy.any() else heavy
            if f == "W":
                se = (d[i] ** 2).sum(axis=0)   # per column
                nlev = d.shape[1]
                tot = se.sum()
                out_rmse = float(np.sqrt(se[~near].sum() / (nlev * max(1, (~near).sum())))) if (~near).any() else None
                st_share = float(se[steep].sum() / tot) if tot > 0 else None
            else:
                se = d[i] ** 2
                tot = se.sum()
                out_rmse = rmse(d[i][~near])
                st_share = float(se[steep].sum() / tot) if tot > 0 else None
            rows.append({"lead": L, "rmse_all": rmse(d[i]), "near_cells": int(near.sum()),
                         "near_area_frac": float(near.mean()),
                         "sq_err_share_near": float(se[near].sum() / tot) if tot > 0 else None,
                         "rmse_outside_near": out_rmse, "outside_within_tol": (out_rmse is not None and out_rmse <= TOL[f]),
                         "sq_err_share_steep": st_share, "steep_area_frac": float(steep.mean()),
                         "sq_err_share_land": float(se[land].sum() / tot) if tot > 0 else None,
                         "land_area_frac": float(land.mean())})
        q3[f] = rows
    out["Q3"] = q3

    # ---- Q4: RAINNC totals, intensity, partition, displacement
    q4 = []
    for i, L in enumerate(leads):
        row = {"lead": L, "cpu_mean": float(rc[i].mean()), "gpu_mean": float(rg_[i].mean()),
               "ratio_gpu_cpu": float(rg_[i].mean() / rc[i].mean()) if rc[i].mean() > 0 else None,
               "cpu_max": float(rc[i].max()), "gpu_max": float(rg_[i].max()),
               "cpu_max_hourly": float(inc_c[i].max()), "gpu_max_hourly": float(inc_g[i].max()),
               "cpu_ge10": int((rc[i] >= 10).sum()), "gpu_ge10": int((rg_[i] >= 10).sum()),
               "cpu_ge25": int((rc[i] >= 25).sum()), "gpu_ge25": int((rg_[i] >= 25).sum()),
               "rmse": rmse(rg_[i] - rc[i])}
        for k in ("GRAUPELNC", "SNOWNC"):
            if f"cpu_{k}" in z:
                row[f"cpu_{k}_mean"] = float(z[f"cpu_{k}"][i].mean())
                row[f"gpu_{k}_mean"] = float(z[f"gpu_{k}"][i].mean())
        q4.append(row)
    # displacement: best integer shift (<= 10 cells) of the GPU accumulation vs CPU at chosen leads
    disp = []
    for L in (12, 24, 36, 48, 72):
        if L not in leads:
            continue
        i = leads.index(L)
        a, b = rc[i], rg_[i]
        if a.max() <= 0:
            continue
        S = 10
        best = (rmse((b - a)[rg >= S + 1]), 0, 0)
        for sy in range(-S, S + 1):
            for sx in range(-S, S + 1):
                bs = np.roll(np.roll(b, sy, 0), sx, 1)
                m = rg >= S + 1
                e = rmse((bs - a)[m])
                if e < best[0]:
                    best = (e, sy, sx)
        m = rg >= S + 1
        disp.append({"lead": L, "rmse_interior_noshift": rmse((b - a)[m]), "rmse_best_shift": best[0],
                     "shift_cells_y_x": [best[1], best[2]], "shift_km": float(np.hypot(best[1], best[2]) * dx),
                     "intensity_ratio_p99": float(np.percentile(b, 99) / np.percentile(a, 99)) if np.percentile(a, 99) > 0 else None,
                     "cpu_p99": float(np.percentile(a, 99)), "gpu_p99": float(np.percentile(b, 99))})
    out["Q4"] = {"by_lead": q4, "displacement": disp}

    # ---- Q5: zero-field integrity per frame
    q5 = {}
    for f in ("QRAIN_colmax", "QNRAIN_colmax", "QGRAUP_colmax", "GRAUPELNC", "QRAINXY", "RAINNC"):
        if f"gpu_{f}" not in z:
            continue
        g, c = z[f"gpu_{f}"], z[f"cpu_{f}"]
        rows = []
        for i, L in enumerate(leads):
            cn = c[i] != 0
            rows.append({"lead": L, "gpu_nonzero": int((g[i] != 0).sum()), "gpu_max": float(g[i].max()),
                         "cpu_nonzero": int(cn.sum()), "cpu_max": float(c[i].max()),
                         "cpu_nonzero_ring0": int((cn & (rg == 0)).sum()), "cpu_nonzero_relax": int((cn & regions["relax1_4"]).sum()),
                         "cpu_nonzero_interior": int((cn & (rg >= 5)).sum()),
                         "cpu_gt_R1": int((c[i] > 1e-12).sum()), "gpu_gt_R1": int((g[i] > 1e-12).sum())})
        q5[f] = {"frames_gpu_all_zero": [r["lead"] for r in rows if r["gpu_nonzero"] == 0],
                 "frames_cpu_all_zero": [r["lead"] for r in rows if r["cpu_nonzero"] == 0],
                 "frames_gpu_zero_cpu_nonzero": [r["lead"] for r in rows if r["gpu_nonzero"] == 0 and r["cpu_nonzero"] > 0],
                 "gpu_nonzero_frames": sum(r["gpu_nonzero"] > 0 for r in rows), "by_lead": rows}
    out["Q5"] = q5
    json.dump(out, open(HERE / f"q_{dom}.json", "w"), indent=0)
    return out


if __name__ == "__main__":
    for dom in sys.argv[1:]:
        o = analyze(dom)
        print(dom, "done")
