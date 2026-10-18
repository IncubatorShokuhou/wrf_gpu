#!/usr/bin/env python3
"""Extract WRF's aer_opt=1 Tegen aerosol climatology (run/aerosol*.formatted) into a deterministic NPZ asset.

WRF reads the tables in ``module_physics_init.F::aerosol_in`` with list-directed REAL reads in the loop order
``ks (6 types) / m (12 months) / j (46 lat) / k (12 levels) / i (72 lon)``; the asset keeps that order as the C
array ``aerodin[ks, m, j, k, i]``.  ``--gfortran-dump`` optionally checks every value bitwise against a stream
dump written by gfortran's own list-directed reader (proofs/w41_aerosol/read_tegen.f90).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from extract_wrf_cam_ozone import _array_sha, _read_f4, _sha256, _write_deterministic_npz

SHAPE = (6, 12, 46, 12, 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrf-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gfortran-dump", type=Path, default=None)
    args = parser.parse_args()

    run = args.wrf_root.resolve() / "run"
    sources = {
        "pressure": run / "aerosol_plev.formatted",
        "latitude": run / "aerosol_lat.formatted",
        "longitude": run / "aerosol_lon.formatted",
        "aod_per_pa": run / "aerosol.formatted",
    }
    if any(not path.is_file() for path in sources.values()):
        raise SystemExit(f"missing WRF Tegen aerosol source: {sources}")
    pressure = _read_f4(sources["pressure"])
    latitude = _read_f4(sources["latitude"])
    longitude = _read_f4(sources["longitude"])
    values = _read_f4(sources["aod_per_pa"])
    if pressure.shape != (12,) or latitude.shape != (46,) or longitude.shape != (72,) or values.size != np.prod(SHAPE):
        raise SystemExit(f"unexpected Tegen dimensions {pressure.shape} {latitude.shape} {longitude.shape} {values.shape}")
    gfortran = None
    if args.gfortran_dump is not None:
        dump = np.fromfile(args.gfortran_dump, dtype="<f4")
        reference = np.concatenate([pressure, latitude, longitude, values])
        if dump.shape != reference.shape or dump.view(np.uint32).tobytes() != reference.view(np.uint32).tobytes():
            raise SystemExit("Tegen parse differs from the gfortran list-directed REAL read")
        gfortran = {"path": str(args.gfortran_dump.resolve()), "sha256": _sha256(args.gfortran_dump), "bitwise": True}
    arrays = {
        "pressure_hpa": pressure,  # aerosol_in reads millibar; the *100 conversion is commented out in WRF
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "aerodin": values.reshape(SHAPE),  # AOD per Pa, [type, month, lat, level, lon]
    }
    if not all(np.isfinite(v).all() and v.dtype == np.float32 for v in arrays.values()):
        raise SystemExit("Tegen asset contains invalid values")
    if not (np.all(np.diff(pressure) > 0) and np.all(np.diff(latitude) > 0) and np.all(np.diff(longitude) > 0)):
        raise SystemExit("Tegen coordinates are not strictly increasing")
    _write_deterministic_npz(args.output.resolve(), arrays)
    record = {
        "schema": "wrf-tegen-aerosol-v1",
        "description": ("WRF v4.7.1 run/aerosol*.formatted (aer_opt=1 Tegen/ECMWF 6-type AOD per Pa, 12 months x 12 "
                        "pressure levels x 46 lat x 72 lon); float32 values = gfortran list-directed REAL reads"),
        "asset": {
            "path": str(args.output.resolve()),
            "sha256": _sha256(args.output.resolve()),
            "arrays": {name: {"shape": list(v.shape), "dtype": v.dtype.str, "payload_sha256": _array_sha(v)}
                       for name, v in arrays.items()},
        },
        "wrf_sources": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in sources.items()},
        "gfortran_read_check": gfortran,
    }
    args.manifest.resolve().write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["asset"], sort_keys=True))


if __name__ == "__main__":
    main()
