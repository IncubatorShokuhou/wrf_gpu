"""One-time-slice-at-a-time products on identical static WRF mass grids."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .members import identity, read_members
from .statistics import finite, levels, products

GRID_ATTRS = ("GRID_ID", "MAP_PROJ", "DX", "DY", "TRUELAT1", "TRUELAT2", "STAND_LON", "MOAD_CEN_LAT")


def inventory(member):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.verify.wrfout import _times

    frames, issues, domains = {}, set(), set()
    for path in member["files"]:
        with Dataset(path) as ds:
            issue = getattr(ds, "SIMULATION_START_DATE", getattr(ds, "START_DATE", None))
            domain = getattr(ds, "GRID_ID", None)
            if issue is None or domain is None:
                raise ValueError("each wrfout requires forecast issue and GRID_ID")
            issues.add(str(issue))
            domains.add(domain)
            for index, time in enumerate(_times(ds)):
                if time in frames:
                    raise ValueError(f"{member['id']}: duplicate valid time {time}")
                frames[time] = (path, index)
    if len(issues) != 1 or len(domains) != 1 or not frames:
        raise ValueError("select exactly one issue and domain per member")
    return frames, next(iter(issues)), next(iter(domains))


def field(ds, name, index):
    var = ds.variables[name]
    if var.dimensions and var.dimensions[0] == "Time":
        return finite(var[index], name), var.dimensions[1:]
    return finite(var[:], name), var.dimensions


def frame(path, index, variable):
    from gpuwrf.io.netcdf_lock import Dataset

    with Dataset(path) as ds:
        values, dims = field(ds, variable, index)
        units = str(getattr(ds.variables[variable], "units", "")).strip()
        allowed = (("south_north", "west_east"), ("bottom_top", "south_north", "west_east"))
        if dims not in allowed or not units:
            raise ValueError("products require a 2D/3D mass-grid scalar with explicit units")
        if variable in ("U10", "V10", "WDIR10", "WDIR"):
            raise ValueError("grid wind components/directions require prior earth-frame/circular processing; use station verify")
        coords = {}
        for name in ("XLAT", "XLONG", *(("ZNU",) if len(dims) == 3 else ())):
            coords[name] = field(ds, name, index)
        if coords["XLAT"][0].shape != values.shape[-2:] or coords["XLONG"][0].shape != values.shape[-2:]:
            raise ValueError("latitude/longitude shape mismatch")
        if len(dims) == 3 and coords["ZNU"][0].shape != (values.shape[0],):
            raise ValueError("vertical coordinate mismatch")
        attrs = {key: ds.getncattr(key) for key in GRID_ATTRS if key in ds.ncattrs()}
    return values, dims, units, coords, attrs


def write_products(member_path, variable, output, *, alignment="exact", thresholds=(0,), quantiles=(0.1, 0.5, 0.9)):
    from gpuwrf.io.netcdf_lock import Dataset

    if alignment not in ("exact", "intersection"):
        raise ValueError("alignment must be exact or intersection")
    thresholds = levels(thresholds, "thresholds")
    quantiles = levels(quantiles, "quantiles", probability=True)
    members = read_members(member_path)
    inventories = [inventory(member) for member in members]
    if len({item[2] for item in inventories}) != 1:
        raise ValueError("members must use the same domain")
    time_sets = [set(item[0]) for item in inventories]
    if alignment == "exact" and any(times != time_sets[0] for times in time_sets[1:]):
        raise ValueError("member valid times differ; select intersection explicitly for lagged products")
    times = sorted(set.intersection(*time_sets))
    if not times:
        raise ValueError("no common valid times")
    provenance = [{"id": member["id"], "issue": inv[1], "excluded_times": len(inv[0])-len(times),
                   "files": [identity(p) for p in member["files"]]} for member, inv in zip(members, inventories)]
    output = Path(output)
    if output.exists():
        raise ValueError("output exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".ensemble-", suffix=".nc", dir=output.parent)
    os.close(fd)
    reference = None
    try:
        with Dataset(temp, "w") as target:
            target.createDimension("Time", len(times))
            target.createDimension("DateStrLen", 19)
            target.createDimension("quantile", len(quantiles))
            target.createDimension("threshold", len(thresholds))
            target.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array(
                [list(t.strftime("%Y-%m-%d_%H:%M:%S")) for t in times], dtype="S1")
            target.createVariable("quantile_level", "f8", ("quantile",))[:] = quantiles
            target.createVariable("exceedance_threshold", "f8", ("threshold",))[:] = thresholds
            target.setncatts({"schema": "gpuwrf.ensemble.products.v1", "member_count": len(members),
                             "source_variable": variable, "alignment": alignment, "event": "X > threshold",
                             "spread_ddof": 1, "quantile_method": "linear", "member_provenance": json.dumps(provenance)})
            for t, time in enumerate(times):
                arrays = []
                for inv in inventories:
                    path, index = inv[0][time]
                    values, dims, units, coords, attrs = frame(path, index, variable)
                    if reference is None:
                        reference = (dims, units, coords, attrs, values.shape)
                        for dim, size in zip(dims, values.shape):
                            target.createDimension(dim, size)
                        for name, (coord, coord_dims) in coords.items():
                            target.createVariable(name, "f8", coord_dims)[:] = coord
                        target.setncatts(attrs)
                        target.variables["exceedance_threshold"].units = units
                        for name, extra, unit in (("mean", (), units), ("spread", (), units),
                                                  ("quantile", ("quantile",), units), ("probability", ("threshold",), "1")):
                            var = target.createVariable(name, "f8", ("Time",)+extra+dims)
                            var.units = unit
                    rdims, runits, rcoords, rattrs, rshape = reference
                    if (dims != rdims or units != runits or attrs != rattrs or values.shape != rshape
                            or any(coords[key][1] != rcoords[key][1] or not np.array_equal(coords[key][0], rcoords[key][0]) for key in coords)):
                        raise ValueError("member/frame grid, projection, vertical coordinates or units differ")
                    arrays.append(values)
                result = products(np.stack(arrays), thresholds=thresholds, quantiles=quantiles)
                for name, value in result.items():
                    target.variables[name][t] = value
        # Atomic publication with no overwrite, including a concurrent publisher.
        os.link(temp, output)
    finally:
        Path(temp).unlink(missing_ok=True)
    return {"schema": "gpuwrf.ensemble.products.v1", "members": len(members), "frames": len(times),
            "output": str(output), "alignment": alignment}
