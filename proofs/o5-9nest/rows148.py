"""#148 per-row numbers from the w43-148fid derived extracts (d0X.npz: cpu_* = CPU-WRF 2x2 ref, gpu_* = GPU v0.4.2 warm72).

Per frozen D6 row: X(τ) (RMSE like the ALISIOS scorer: all cells; W all levels), breach leads (X > limit), the C a CPU twin
would need for FLOOR-LIMITED under the teacher A3 form (every breach X <= 4/3*C -> C >= 0.75*X), and for RAINNC the
scale-free X/mean(CPU) and heavy-cell counts. Interior trace rows: GPU/CPU graupel aloft and rain near the flagged cells.

usage: JAX_PLATFORMS=cpu python rows148.py <npz_dir> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

LIMIT = {"RAINNC": 1.0, "W": 0.3, "U10": 1.5, "V10": 1.5}
ROWS = [  # (stage, dom, field) from TO_WRF_GPU2_148_FIDELITY_FAIL_20261020.md; stage 24 = leads <= 24
    (24, "d02", "RAINNC"), (24, "d04", "RAINNC"), (24, "d05", "RAINNC"), (24, "d06", "RAINNC"), (24, "d06", "W"),
    (72, "d02", "RAINNC"), (72, "d04", "RAINNC"), (72, "d04", "U10"), (72, "d04", "V10"), (72, "d04", "W"),
    (72, "d05", "RAINNC"), (72, "d06", "RAINNC"), (72, "d06", "U10"), (72, "d06", "V10"), (72, "d06", "W")]


def rmse(d):
    d = np.asarray(d, np.float64)
    return float(np.sqrt(np.mean(d * d)))


def finite(*arrs):
    for a in arrs:
        if not np.isfinite(a).all():
            raise ValueError("non-finite operand")


def main(npz_dir, out_path):
    npz_dir = Path(npz_dir)
    cache, out = {}, {"rows": [], "rain": {}, "interior_traces": {}}
    for stage, dom, f in ROWS:
        z = cache.setdefault(dom, dict(np.load(npz_dir / f"{dom}.npz")))
        leads = z["leads"].tolist()
        c, g = z[f"cpu_{f}"], z[f"gpu_{f}"]
        finite(c, g)
        X = {L: rmse(g[i].astype(np.float64) - c[i]) for i, L in enumerate(leads) if L <= stage}
        br = {L: x for L, x in X.items() if x > LIMIT[f]}
        out["rows"].append({"stage": stage, "dom": dom, "field": f, "limit": LIMIT[f],
                            "breach_leads": sorted(br), "max_X": max(br.values()) if br else None,
                            "argmax_lead": max(br, key=br.get) if br else None,
                            "C_needed_for_A3": 0.75 * max(br.values()) if br else None, "X_by_lead": X})
    for dom in ("d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09"):
        p = npz_dir / f"{dom}.npz"
        if not p.exists():
            continue
        z = cache.setdefault(dom, dict(np.load(p)))
        leads = z["leads"].tolist()
        c, g = z["cpu_RAINNC"].astype(np.float64), z["gpu_RAINNC"].astype(np.float64)
        rows = []
        for i, L in enumerate(leads):
            mc, mg = float(c[i].mean()), float(g[i].mean())
            x = rmse(g[i] - c[i])
            rows.append({"lead": L, "mean_cpu": mc, "mean_gpu": mg, "ratio_mean": mg / mc if mc > 0 else None,
                         "X": x, "X_over_mean_cpu": x / mc if mc > 0 else None,
                         "cells_ge10_cpu": int((c[i] >= 10).sum()), "cells_ge10_gpu": int((g[i] >= 10).sum()),
                         "max_cpu": float(c[i].max()), "max_gpu": float(g[i].max())})
        out["rain"][dom] = rows
    # interior trace rows: d06 GRAUPELNC τ24 (2 cells), d06 QRAINXY τ72 (1 cell)
    z = cache["d06"]
    leads = z["leads"].tolist()
    it = {}
    for name, lead in (("GRAUPELNC", 24), ("QRAINXY", 72)):
        i = leads.index(lead)
        c, g = z[f"cpu_{name}"][i], z[f"gpu_{name}"][i]
        cells = list(zip(*np.nonzero(c > 0)))
        row = {"lead": lead, "cpu_cells": [[int(a), int(b)] for a, b in cells], "cpu_max": float(c.max()), "gpu_max": float(g.max()),
               "gpu_nonzero_frames": int(sum(float(z[f"gpu_{name}"][k].max()) > 0 for k in range(len(leads)))),
               "cpu_nonzero_frames": int(sum(float(z[f"cpu_{name}"][k].max()) > 0 for k in range(len(leads))))}
        if name == "GRAUPELNC":
            qg_c, qg_g = z["cpu_QGRAUP_colmax"], z["gpu_QGRAUP_colmax"]
            row["qgraup_aloft_first_lead"] = {
                "cpu": next((L for k, L in enumerate(leads) if qg_c[k].max() > 0), None),
                "gpu": next((L for k, L in enumerate(leads) if qg_g[k].max() > 0), None)}
            row["qgraup_aloft_max_upto_lead"] = {"cpu": float(qg_c[: i + 1].max()), "gpu": float(qg_g[: i + 1].max())}
            row["graupelnc_first_nonzero_lead"] = {
                "cpu": next((L for k, L in enumerate(leads) if z["cpu_GRAUPELNC"][k].max() > 0), None),
                "gpu": next((L for k, L in enumerate(leads) if z["gpu_GRAUPELNC"][k].max() > 0), None)}
        else:
            inc_c = z["cpu_RAINNC"][i] - z["cpu_RAINNC"][i - 1]
            inc_g = z["gpu_RAINNC"][i] - z["gpu_RAINNC"][i - 1]
            row["last_hour_rain_at_cells"] = {"cpu": [float(inc_c[a, b]) for a, b in cells], "gpu": [float(inc_g[a, b]) for a, b in cells]}
            row["last_hour_rain_domain_max"] = {"cpu": float(inc_c.max()), "gpu": float(inc_g.max())}
            row["last_hour_rain_cells_gt0"] = {"cpu": int((inc_c > 0).sum()), "gpu": int((inc_g > 0).sum())}
        it[name] = row
    out["interior_traces"] = it
    Path(out_path).write_text(json.dumps(out, indent=1))
    for r in out["rows"]:
        print(r["stage"], r["dom"], r["field"], "breaches", len(r["breach_leads"]), "maxX", r["max_X"], "@", r["argmax_lead"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
