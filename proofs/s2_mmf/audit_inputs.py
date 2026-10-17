"""Inventory MMF geography before choosing a pristine-WRF oracle case.

This is a qualification diagnostic, not an MMF implementation or an admission
gate for the forecast CLI. A populated input still needs provenance and a
pristine coupled oracle. Reads use the production NetCDF process lock.

JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src python \
    proofs/s2_mmf/audit_inputs.py --output receipt.json wrfinput_d01 ...
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise SystemExit("run this diagnostic with JAX_PLATFORMS=cpu")

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset


GEOGRAPHY = ("FDEPTH", "EQZWT", "RECHCLIM", "RIVERBED")
GEOG_SUBDIRS = ("f", "wtd", "recharge", "riverbed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def census_input(path: Path) -> dict:
    result = {"path": str(path.resolve()), "sha256": sha256(path), "fields": {}}
    reasons = []
    with Dataset(path) as dataset:
        raw_land = np.ma.asarray(dataset["LANDMASK"][0])
        if np.ma.count_masked(raw_land) or not np.isfinite(raw_land.compressed()).all():
            reasons.append("masked/nonfinite LANDMASK")
        land = np.ma.filled(raw_land == 1, False)
        result["land_cells"] = int(np.count_nonzero(land))
        for name in GEOGRAPHY:
            if name not in dataset.variables:
                result["fields"][name] = {"missing": True}
                reasons.append(f"missing {name}")
                continue
            variable = dataset[name]
            data = np.ma.asarray(variable[0])
            valid = data.compressed()
            stats = {
                "shape": list(data.shape), "dtype": str(data.dtype),
                "units": str(getattr(variable, "units", "")),
                "masked": int(np.ma.count_masked(data)),
                "nonfinite": int(np.count_nonzero(~np.isfinite(valid))),
                "nonzero": int(np.count_nonzero(valid)),
                "finite_min": float(valid[np.isfinite(valid)].min())
                if np.isfinite(valid).any() else None,
                "finite_max": float(valid[np.isfinite(valid)].max())
                if np.isfinite(valid).any() else None,
            }
            if data.shape != land.shape:
                reasons.append(f"{name} shape differs from LANDMASK")
            else:
                land_data = np.ma.asarray(data[land])
                stats["positive_land_cells"] = int(np.count_nonzero(
                    np.ma.filled(land_data > 0, False)))
            if stats["masked"] or stats["nonfinite"]:
                reasons.append(f"masked/nonfinite {name}")
            result["fields"][name] = stats
        if not result["land_cells"]:
            reasons.append("no land columns for an active MMF oracle")
        if result["fields"]["FDEPTH"].get("positive_land_cells", 0) == 0:
            reasons.append("no positive FDEPTH on land: lateral conductivity inactive")
    result["qualification_input_status"] = (
        "BLOCKED" if reasons else "CANDIDATE_REQUIRES_PROVENANCE_AND_PRISTINE_ORACLE")
    result["reasons"] = reasons
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geog-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        parser.error("run this diagnostic with JAX_PLATFORMS=cpu")
    root = Path(__file__).resolve().parents[2]
    git = lambda *arguments: subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True).strip()
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "gpuwrf_tree": git("rev-parse", "HEAD:src/gpuwrf"),
        "model_source_dirty": bool(git("status", "--porcelain", "--", "src/gpuwrf")),
        "scope": "input inventory only; no model compile or forecast",
        "cache": "NONE", "gpu_executed": False,
        "inputs": [census_input(path) for path in args.inputs],
        "geog_roots": [{
            "path": str(path),
            "datasets": {name: {"directory_exists": (path / "groundwater" / name).is_dir(),
                                "index_exists": (path / "groundwater" / name / "index").is_file()}
                         for name in GEOG_SUBDIRS},
        } for path in args.geog_root],
    }
    blocked = any(item["qualification_input_status"] == "BLOCKED" for item in report["inputs"])
    report["status"] = "BLOCKED_INPUTS" if blocked else "INPUT_CANDIDATES_ONLY"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {len(report['inputs'])} inputs; {args.output}")
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
