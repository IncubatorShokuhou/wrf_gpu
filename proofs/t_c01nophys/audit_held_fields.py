"""Audit original CPU-WRF's surface field lifetime without altering outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset

FIELDS = ("PSFC", "T2", "Q2", "U10", "V10", "TSK", "TH2", "HFX", "QFX", "LH", "UST")


def read(variable):
    value = variable[0] if variable.dimensions[0] == "Time" else variable[:]
    if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
        raise ValueError("masked reference")
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite reference")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with Dataset(args.case / "input/wrfinput_d01") as ds:
        initial = {name: read(ds[name]) for name in FIELDS if name in ds.variables}
    rows = []
    for path in sorted((args.case / "cpu").glob("wrfout_d01_*")):
        with Dataset(path) as ds:
            row = {"frame": path.name, "fields": {}}
            for name in FIELDS:
                if name not in ds.variables:
                    continue
                value = read(ds[name])
                baseline = initial.get(name, np.zeros_like(value))
                delta = value.astype(np.float64) - baseline.astype(np.float64)
                row["fields"][name] = {
                    "input_present": name in initial,
                    "byte_equal": value.dtype == baseline.dtype and value.tobytes() == baseline.tobytes(),
                    "rmse": float(np.sqrt(np.mean(delta * delta))),
                    "max_abs": float(np.max(np.abs(delta))),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
            rows.append(row)
    summary = {name: {
        "frames": sum(name in row["fields"] for row in rows),
        "all_byte_equal": all(row["fields"][name]["byte_equal"] for row in rows if name in row["fields"]),
        "max_abs": max(row["fields"][name]["max_abs"] for row in rows if name in row["fields"]),
    } for name in FIELDS if any(name in row["fields"] for row in rows)}
    result = {"case": str(args.case), "scope": "original CPU-WRF vs input/allocator zero; no GPU regression treated as truth", "summary": summary, "rows": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
