"""Read-only hourly c28 CPU-WRF/GPU attribution; history fluxes may be fallback diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    fields = ("T2", "TSK", "SWDOWN", "GLW", "OLR", "SWUPT", "SWDNT", "LWUPB", "LWDNB",
              "HFX", "LH", "CLDFRA", "RTHRATEN", "RTHRATENLW", "RTHRATENSW", "QCLOUD", "QICE", "QSNOW")
    result = {"case": str(args.case), "reference": "original CPU-WRF", "cache": "NONE",
              "scope": "retained trajectories; flux history provenance must be checked separately", "frames": []}
    for cpu in sorted((args.case / "cpu").glob("wrfout_d01_*")):
        gpu = args.case / "gpu" / cpu.name
        row = {"time": cpu.name.removeprefix("wrfout_d01_"), "fields": {}}
        with Dataset(cpu) as c, Dataset(gpu) as g:
            for name in fields:
                if name not in c.variables or name not in g.variables:
                    continue
                a, b = (np.asarray(ds.variables[name][:], dtype=np.float64) for ds in (c, g))
                assert np.isfinite(a).all() and np.isfinite(b).all(), (cpu, name)
                # Exclude the fixed 5-cell specified/relaxation ring for attribution only.
                for label, sl in (("all", (..., slice(None), slice(None))),
                                  ("interior", (..., slice(5, -5), slice(5, -5)))):
                    x, y = a[sl], b[sl]
                    d = y - x
                    row["fields"][f"{name}/{label}"] = {
                        "cpu_mean": float(x.mean()), "gpu_mean": float(y.mean()),
                        "bias": float(d.mean()), "rmse": float(np.sqrt(np.mean(d*d))),
                        "max_abs": float(np.max(np.abs(d))), "cpu_max": float(x.max()), "gpu_max": float(y.max())}
        result["frames"].append(row)
    result["source_qualification_sha256"] = hashlib.sha256((args.case / "QUALIFICATION.json").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["frames"]:
        print(row["time"], " ".join(f"{f}:bias={row['fields'][f+'/interior']['bias']:.3g},rms={row['fields'][f+'/interior']['rmse']:.3g}"
              for f in ("T2", "TSK", "SWDOWN", "GLW", "CLDFRA") if f+"/interior" in row["fields"]), flush=True)


if __name__ == "__main__":
    main()
