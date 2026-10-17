"""Read retained original CPU-WRF/GPU outputs; no forecast or reference mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    fields = ("T2", "TSK", "HFX", "LH", "UST", "PBLH", "SWDOWN", "GLW", "T", "U", "V", "QVAPOR", "CLDFRA", "QCLOUD", "QICE", "QSNOW")
    result = {"case": str(args.case), "reference": "original CPU-WRF", "frames": []}
    cpu_files = sorted((args.case / "cpu").glob("wrfout_d01_*"))
    for lead, cpu_path in enumerate(cpu_files):
        if lead > 6:
            break
        gpu_path = args.case / "gpu" / cpu_path.name
        row = {"lead_h": lead, "cpu": str(cpu_path), "gpu": str(gpu_path), "fields": {}}
        with Dataset(cpu_path) as cpu, Dataset(gpu_path) as gpu:
            cpu.set_auto_mask(False)
            gpu.set_auto_mask(False)
            for field in fields:
                if field not in cpu.variables or field not in gpu.variables:
                    continue
                a = np.asarray(cpu[field][0], dtype=np.float64)
                b = np.asarray(gpu[field][0], dtype=np.float64)
                if a.ndim == 3 and field not in ("CLDFRA", "QCLOUD", "QICE", "QSNOW"):
                    a, b = a[0], b[0]  # lowest atmospheric level
                delta = b - a
                location = np.unravel_index(np.abs(delta).argmax(), delta.shape)
                row["fields"][field] = {
                    "rmse": float(np.sqrt(np.mean(delta * delta))), "bias": float(delta.mean()),
                    "max_abs": float(np.abs(delta).max()), "argmax": [int(x) for x in location],
                    "cpu_argmax": float(a[location]), "gpu_argmax": float(b[location]),
                    "cpu_nonzero": int(np.count_nonzero(a)), "gpu_nonzero": int(np.count_nonzero(b)),
                    "cpu_range": [float(a.min()), float(a.max())],
                    "gpu_range": [float(b.min()), float(b.max())],
                    "interior_rmse_b5": float(np.sqrt(np.mean(delta[..., 5:-5, 5:-5] ** 2))),
                }
        result["frames"].append(row)
    nml = args.case / "input" / "namelist.input"
    result["namelist_sha256"] = hashlib.sha256(nml.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for frame in result["frames"]:
        print(frame["lead_h"], {k: round(v["rmse"], 6) for k, v in frame["fields"].items()}, flush=True)


if __name__ == "__main__":
    main()
