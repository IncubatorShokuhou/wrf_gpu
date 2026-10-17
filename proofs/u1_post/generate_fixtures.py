"""Extract small unmodified real-history fields; no synthetic expected products.

Run from repo root on CPU25. Sources are preserved; output carries source SHA256,
crop and UTC samples in manifest.json. Numeric fields remain source float32.
"""
import hashlib
import json
from pathlib import Path

import xarray as xr

DEST = Path("tests/fixtures/post")
DEST.mkdir(parents=True, exist_ok=True)
CPU = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
GPU = Path("proofs/m20/tost_run/gpu_wrfout/case1_L2")
FIELDS = "P PB T PH PHB HGT QVAPOR T2 Q2 PSFC U V U10 V10 COSALPHA SINALPHA CLDFRA RAINC RAINNC RAINSH I_RAINC I_RAINNC SNOWH SNOW SNOWNC GRAUPELNC HAILNC XLAT XLONG Times".split()
manifest = {}
for label, paths in (
    ("cpu", [CPU/f"wrfout_d01_2023-01-15_{hour:02d}:00:00" for hour in (6, 7, 9)]),
    ("gpu_surface", [GPU/f"wrfout_d02_2026-05-29_{hour:02d}:00:00" for hour in (19, 20)]),
    ("supersaturated_surface", [CPU/f"wrfout_d01_2023-01-15_{hour:02d}:00:00" for hour in (6, 7, 9)]),
):
    manifest[label] = []
    for i, source in enumerate(paths):
        with xr.open_dataset(source, decode_times=False) as raw:
            names = FIELDS if label != "supersaturated_surface" else "PSFC T2 Q2 U10 V10 XLAT XLONG Times".split()
            selected = raw[[v for v in names if v in raw]]
            y, x = (19, 33) if label == "cpu" else (20, 20)
            crop = {d: slice(y if d.startswith("south") else x,
                            (y if d.startswith("south") else x) + (3 if d.endswith("_stag") else 2))
                    for d in selected.dims if d in ("south_north", "west_east", "south_north_stag", "west_east_stag")}
            selected = selected.isel(crop).load()
        target = DEST/f"{label}_{i}.nc"
        selected.to_netcdf(target)
        manifest[label].append({"source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "fixture": str(target), "fixture_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "crop": {k: [v.start, v.stop] for k, v in crop.items()}, "sizes": dict(selected.sizes)})
(DEST/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps({key: len(val) for key, val in manifest.items()}))
