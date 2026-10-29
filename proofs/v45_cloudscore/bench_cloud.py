"""CPU cost proxy: 72 reads of one archived real WRF frame + cloud metrics.

No real MSG values. Synthetic centres coincide with the WRF mass cells and CLM
is invented clear sky. Repeated weather/OS cache make this a workload proxy,
not an observed 72-hour forecast scoring time or scientific skill evidence.
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
import pandas as pd

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.optimize.cloud import Footprints, channel_diagnostics, superobserve, verify, wrf_columns
from gpuwrf.optimize.data import identity
from gpuwrf.verify.wrfout import _field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrfout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=72)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("frames must be positive")
    source = identity(args.wrfout)
    started = time.perf_counter()
    with Dataset(args.wrfout) as ds:
        lat, lon, mf = (_field(ds, k, 0) for k in ("XLAT", "XLONG", "MAPFAC_M"))
        op = Footprints.build(lat, lon, float(ds.DX)*float(ds.DY)/mf**2, lat.ravel(), lon.ravel())
        op.identity(lat.ravel().astype("f4"), lon.ravel().astype("f4"))
        setup_s = time.perf_counter()-started
        values, per_frame = [], []
        for _ in range(args.frames):
            start = time.perf_counter()
            columns, rho, dz = wrf_columns(ds, 0, return_geometry=True)
            values.append(op.apply(columns >= 10))
            channel_diagnostics(ds, 0, op, rho, dz, columns, op.weight > 0, {})
            per_frame.append(time.perf_counter()-start)
        shape = ds.variables["QCLOUD"].shape
    started = time.perf_counter()
    values = np.array(values)
    report, *_ = verify(values, np.zeros(values.shape, dtype="u1"), np.ones(values.shape, bool), op.weight,
                         lat.ravel(), lon.ravel(), radii_km=[0, 5, 10])
    metrics_s = time.perf_counter()-started
    # Explicitly synthetic bounded EKI scope: 8x22 model centres, 50-km blocks.
    # It measures superobs machinery, not an admitted real correlation length.
    start = time.perf_counter()
    subset = np.arange(lat.size).reshape(lat.shape)[:8, :22].ravel()
    slots = pd.date_range("2026-01-01T01:00:00Z", periods=args.frames, freq="h")
    keys = [(t.isoformat(), int(p)) for t in slots for p in subset]
    proxy_report = {"footprints": {"assigned_area_m2": op.weight.tolist()}}
    px = values[:, subset].ravel()
    sx, sy, sk = superobserve(proxy_report, px, np.zeros_like(px), keys,
        {"superobs": {"correlation_length_km": 50., "contract": "SYNTHETIC_PERFORMANCE_ONLY"}},
        {"lat": lat.ravel(), "lon": lon.ravel(), "regions": {}})
    superobs_s = time.perf_counter()-start
    output = {"schema": "gpuwrf.optimize.cloud.cost.v1", "source": source, "frames": args.frames,
              "model_shape": list(shape), "pixels": int(lat.size), "setup_s": setup_s,
              "column_footprint_s": sum(per_frame), "frame_median_s": float(np.median(per_frame)),
              "metrics_s": metrics_s, "superobs_s": superobs_s, "superobs_m": len(sx), "superobs_native_samples": len(px),
              "superobs_scope": "synthetic first8x22 mass centres,50km blocks; not a real admitted correlation length",
              "proxy_72_frames_s": setup_s+sum(per_frame)+metrics_s+superobs_s,
              "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "counts": report["counts"], "cache": "no JAX; warm OS/NetCDF caches, repeated frame",
              "work": "QCLOUD+QICE dry CWP10g/m2, EPSG6933, FSS0/5/10km, all synthetic daytime",
              "claim": "[M] repeated-frame CPU workload; [I] cost proxy for hourly 72h at this grid; no real MSG/forecast skill"}
    with open(args.output, "x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
