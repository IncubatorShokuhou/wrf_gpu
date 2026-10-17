"""Localize c03 precipitation differences against retained original CPU-WRF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset


def analyze(case: Path, output: Path) -> dict:
    rows = []
    previous = None
    for cpu in sorted((case / "cpu").glob("wrfout_d01_*")):
        gpu = case / "gpu" / cpu.name
        with Dataset(cpu) as ref, Dataset(gpu) as cand:
            a = np.asarray(ref.variables["RAINNC"][0], dtype=np.float64)
            b = np.asarray(cand.variables["RAINNC"][0], dtype=np.float64)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            ny, nx = a.shape
            y, x = np.indices(a.shape)
            ring = np.minimum.reduce([y, x, ny - 1 - y, nx - 1 - x])
            delta = b - a
            total = float(np.sum(delta * delta))
            top = np.argsort((delta * delta).ravel())[-10:][::-1]
            row = {"time": cpu.name, "rmse_mm": float(np.sqrt(np.mean(delta**2))),
                   "bias_mm": float(delta.mean()), "cpu_max_mm": float(a.max()),
                   "gpu_max_mm": float(b.max()), "regions": {}, "top_cells": []}
            for label, mask in [("ring0", ring == 0), ("ring1to4", (ring >= 1) & (ring <= 4)),
                                ("interior5", ring >= 5)]:
                row["regions"][label] = {"cells": int(mask.sum()),
                    "rmse_mm": float(np.sqrt(np.mean(delta[mask]**2))),
                    "sse_fraction": float(np.sum(delta[mask]**2) / total) if total else 0.0}
            for idx in top:
                iy, ix = np.unravel_index(idx, a.shape)
                row["top_cells"].append({"y": int(iy), "x": int(ix), "ring": int(ring[iy, ix]),
                    "cpu_mm": float(a[iy, ix]), "gpu_mm": float(b[iy, ix]),
                    "hgt_m": float(ref.variables["HGT"][0, iy, ix])})
            row["other_fields"] = {}
            for name in ("RAINC", "SNOWNC", "GRAUPELNC", "QRAIN", "QSNOW", "QGRAUP", "QVAPOR", "T2", "PBLH"):
                if name not in ref.variables or name not in cand.variables:
                    continue
                ra = np.asarray(ref.variables[name][0], dtype=np.float64)
                rb = np.asarray(cand.variables[name][0], dtype=np.float64)
                assert np.isfinite(ra).all() and np.isfinite(rb).all(), name
                row["other_fields"][name] = {"cpu_max": float(ra.max()), "gpu_max": float(rb.max()),
                    "bias": float(np.mean(rb-ra)), "rmse": float(np.sqrt(np.mean((rb-ra)**2)))}
            if previous is not None:
                da, db = a - previous[0], b - previous[1]
                row["hour_increment"] = {"cpu_mean_mm": float(da.mean()), "gpu_mean_mm": float(db.mean()),
                    "rmse_mm": float(np.sqrt(np.mean((db-da)**2)))}
            previous = a, b
            rows.append(row)
    result = {"reference": "original CPU-WRF c03", "case": str(case), "frames": len(rows),
              "rows": rows, "max_rmse_mm": max(r["rmse_mm"] for r in rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.case, args.output)
    print(json.dumps({"frames": result["frames"], "max_rmse_mm": result["max_rmse_mm"]}))
