"""Freeze small unmodified land-state crops from original CPU-WRF outputs.

Run from repo root; never executes WRF or changes a reference file.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.cycling.core import CORE, MP_REQUIRED, MP_OPTIONAL, sha256

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-diag/cpu_reference_3h")
INPUT = ROOT / "examples/switzerland_d01/wrfinput_d01"
arrays, metadata = {}, {"source": "original pristine CPU-WRF Swiss hourly outputs", "sources": []}
names = (*CORE, *MP_REQUIRED, *MP_OPTIONAL, "CANWAT", "FVEG", "XLAT", "XLONG", "HGT", "LU_INDEX",
         "IVGTYP", "ISLTYP", "LANDMASK", "XLAND", "LAKEMASK", "SEAICE", "ZS", "DZS")
with Dataset(INPUT) as ds:
    ice = np.argwhere(np.asarray(ds["IVGTYP"][0]) == int(ds.ISICE))[0]
    y, x = max(0, int(ice[0]) - 2), max(0, int(ice[1]) - 2)
metadata["crop_yx"] = [y, y + 6, x, x + 6]
for hour in (0, 1, 2, 3):
    path = INPUT if hour == 0 else REFERENCE / f"wrfout_d01_2023-01-15_0{hour}:00:00"
    with Dataset(path) as ds:
        prefix = "target" if hour == 0 else f"source{hour}"
        attrs = {name: ds.getncattr(name) for name in ("DX", "DY", "MAP_PROJ", "GRID_ID", "MMINLU", "NUM_LAND_CAT", "ISWATER", "ISICE", "SF_SURFACE_PHYSICS")}
        entry = {"path": str(path), "sha256": sha256(path), "prefix": prefix, "attrs": attrs, "variables": {}}
        for name in names:
            if name not in ds.variables:
                continue
            var = ds[name]
            raw = np.asarray(var[0])
            cropped = raw[..., y:y + 6, x:x + 6] if raw.ndim >= 2 else raw
            key = f"{prefix}__{name}"
            arrays[key] = cropped
            entry["variables"][name] = {"dimensions": list(var.dimensions), "units": getattr(var, "units", ""),
                                         "sha256": hashlib.sha256(np.ascontiguousarray(cropped).tobytes()).hexdigest()}
        metadata["sources"].append(entry)
out = ROOT / "tests/fixtures/cycling"
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "swiss_cpu_land.npz", **arrays)
metadata["fixture_sha256"] = sha256(out / "swiss_cpu_land.npz")
(out / "swiss_cpu_land.json").write_text(json.dumps(metadata, indent=2, default=lambda v: v.item()) + "\n")
print(json.dumps({"fixture_bytes": (out / "swiss_cpu_land.npz").stat().st_size, "arrays": len(arrays), "crop_yx": metadata["crop_yx"]}))
