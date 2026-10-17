"""Read WRF history without dask; CF-annotated, atomically published output."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr

from gpuwrf.io.netcdf_lock import NETCDF_LOCK


def _times(ds):
    if "Time" in ds.coords and np.issubdtype(ds.Time.dtype, np.datetime64):
        values = ds.Time.values.astype("datetime64[ns]")
    elif "Times" in ds:
        a = ds.Times.values
        texts = []
        for row in a:
            if np.ndim(row):
                text = b"".join(row).decode() if a.dtype.kind == "S" else "".join(row)
            else:
                text = row.decode() if isinstance(row, bytes) else str(row)
            texts.append(text.replace("_", "T"))
        values = np.array(texts, dtype="datetime64[ns]")
    elif "Times" in ds.attrs and ds.sizes.get("Time") == 1:
        values = np.array([str(ds.attrs["Times"]).replace("_", "T")], dtype="datetime64[ns]")
    else:
        raise ValueError("WRF history needs Times or a decoded datetime Time coordinate")
    if len(values) == 0 or len(values) != ds.sizes.get("Time") or np.isnat(values).any():
        raise ValueError("invalid WRF timestamps")
    return values


def read_wrfout(paths, variables=None):
    """Load selected fields, close each input immediately, sort by actual UTC time.

    Multiple domains, duplicate times, moving grids and inconsistent field sets are
    refused. No restart-reset guessing or filename-based temporal inference.
    """
    if not paths:
        raise ValueError("at least one wrfout path is required")
    frames, reference, names, attributes = [], None, None, None
    for path in paths:
        with NETCDF_LOCK, xr.open_dataset(path, decode_times=False) as raw:
            times = _times(raw)
            keep = set(raw.data_vars) if variables is None else set(variables)
            keep |= {"XLAT", "XLONG", "HGT"}
            d = raw[[k for k in raw.data_vars if k in keep and k != "Times"]].load()
            d = d.assign_coords(Time=times)
        current_names = set(d.data_vars)
        grid = {k: d[k].isel(Time=0).values if "Time" in d[k].dims else d[k].values
                for k in ("XLAT", "XLONG", "HGT") if k in d}
        if reference is None:
            reference, names, attributes = grid, current_names, dict(d.attrs)
        if current_names != names:
            raise ValueError("inconsistent field set across history files")
        if grid.keys() != reference.keys() or any(not np.array_equal(grid[k], reference[k]) for k in grid):
            raise ValueError("history grid changed; process one stationary domain at a time")
        for key in ("GRID_ID", "MAP_PROJ", "BUCKET_MM", "SIMULATION_START_DATE"):
            if d.attrs.get(key) != attributes.get(key):
                raise ValueError(f"inconsistent history attribute {key}")
        for k in grid:
            if "Time" in d[k].dims and not np.array_equal(d[k].values, np.broadcast_to(grid[k], d[k].shape)):
                raise ValueError(f"moving history grid: {k}")
        frames.append(d)
    try:
        result = xr.concat(frames, dim="Time", join="exact", data_vars="all", coords="minimal", compat="equals").sortby("Time")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"incompatible WRF history grids: {exc}") from exc
    if np.any(np.diff(result.Time.values) <= np.timedelta64(0, "ns")):
        raise ValueError("duplicate or non-increasing history timestamps")
    result.attrs["post_source_files"] = "\n".join(str(Path(p).resolve()) for p in paths)
    return result


def write_products(ds, path):
    """Write beside destination then replace; never leave a partial destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".nc", dir=path.parent)
    os.close(fd)
    try:
        ds.to_netcdf(tmp, engine="netcdf4")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
