"""Atomic local export writers, optional ecCodes/SciPy for GRIB2."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from gpuwrf.post.io import write_products

# WMO GRIB2 discipline 0, category/number, fixed surface type/value.
GRIB = {
    "temperature_2m": (0, 0, 103, 2), "surface_pressure": (3, 0, 1, 0),
    "mixing_ratio_2m": (1, 2, 103, 2),
    "eastward_wind_10m": (2, 2, 103, 10), "northward_wind_10m": (2, 3, 103, 10),
    "terrain_height": (3, 5, 1, 0), "temperature": (0, 0, 100, None),
    "mixing_ratio": (1, 2, 100, None), "eastward_wind": (2, 2, 100, None),
    "northward_wind": (2, 3, 100, None), "geopotential_height": (3, 5, 100, None),
}


def _grib(ds, path, grid=None):
    try:
        import eccodes as ec
    except ImportError as exc:
        raise ValueError("GRIB2 requires pip install 'gpuwrf[export]'") from exc
    lat, lon = ds.latitude.values, ds.longitude.values
    if grid is None:
        if not (np.allclose(lat, lat[:, :1], atol=1e-6, rtol=0)
                and np.allclose(lon, lon[:1], atol=1e-6, rtol=0)):
            raise ValueError("GRIB2 subset needs regular lat/lon or explicit --grib-grid WEST EAST SOUTH NORTH NX NY")
        gx, gy = lon[0], lat[:, 0]
    else:
        west, east, south, north, nx, ny = grid
        if (not np.isfinite(grid).all() or nx != int(nx) or ny != int(ny)
                or min(nx, ny) < 2 or not (-180 <= west < east <= 180 and -90 <= south < north <= 90)):
            raise ValueError("invalid --grib-grid WEST EAST SOUTH NORTH NX NY")
        gx, gy = np.linspace(west, east, int(nx)), np.linspace(south, north, int(ny))
    if len(gx) < 2 or len(gy) < 2 or np.any(np.diff(gx) <= 0) or np.any(np.diff(gy) <= 0):
        raise ValueError("GRIB2 grid must increase east/north with at least two points per axis")
    if not np.allclose(np.diff(gx), np.diff(gx)[0], atol=1e-6, rtol=0) or not np.allclose(np.diff(gy), np.diff(gy)[0], atol=1e-6, rtol=0):
        raise ValueError("GRIB2 subset requires uniform lat/lon spacing")
    xx, yy = np.meshgrid(gx, gy)
    if any(k not in GRIB for k in ds.data_vars if k != "crs"):
        raise ValueError("field outside GRIB2 subset")
    if "pressure" in ds.coords and np.any(ds.pressure.values != np.round(ds.pressure.values)):
        raise ValueError("GRIB2 subset requires pressure levels in whole Pa")
    with open(path, "wb") as output:
        for it, stamp in enumerate(ds.time.values):
            text = np.datetime_as_string(stamp, unit="s")
            if text[17:19] != "00" or stamp != stamp.astype("datetime64[m]"):
                raise ValueError("GRIB2 subset supports whole-minute UTC timestamps only")
            for name, da in ds.data_vars.items():
                if name == "crs":
                    continue
                cat, number, surface, value = GRIB[name]
                for iz in range(da.sizes.get("pressure", 1)):
                    slab = da.isel(time=it, **({"pressure": iz} if "pressure" in da.dims else {})).values
                    if grid is not None:
                        from scipy.interpolate import griddata
                        slab = griddata(np.column_stack([lon.ravel(), lat.ravel()]), slab.ravel(), (xx, yy), method="linear")
                    if not np.isfinite(slab).all():
                        raise ValueError("GRIB2 target outside source convex hull; extrapolation refused")
                    gid = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
                    try:
                        keys = {"Ni": len(gx), "Nj": len(gy), "shapeOfTheEarth": 1,
                                "scaleFactorOfRadiusOfSphericalEarth": 0, "scaledValueOfRadiusOfSphericalEarth": 6370000,
                                "latitudeOfFirstGridPointInDegrees": float(gy[0]), "longitudeOfFirstGridPointInDegrees": float(gx[0] % 360),
                                "latitudeOfLastGridPointInDegrees": float(gy[-1]), "longitudeOfLastGridPointInDegrees": float(gx[-1] % 360),
                                "iDirectionIncrementInDegrees": float(gx[1]-gx[0]), "jDirectionIncrementInDegrees": float(gy[1]-gy[0]),
                                "jScansPositively": 1, "iScansNegatively": 0, "uvRelativeToGrid": 0,
                                "dataDate": int(text[:10].replace("-", "")), "dataTime": int(text[11:16].replace(":", "")),
                                "centre": 65535, "subCentre": 65535, "generatingProcessIdentifier": 255, "significanceOfReferenceTime": 2,
                                "typeOfGeneratingProcess": 13, "forecastTime": 0, "discipline": 0, "parameterCategory": cat, "parameterNumber": number,
                                "typeOfFirstFixedSurface": surface, "scaleFactorOfFirstFixedSurface": 0,
                                "scaledValueOfFirstFixedSurface": int(ds.pressure.values[iz]) if value is None else value,
                                "packingType": "grid_ieee", "precision": 2}
                        for key, val in keys.items():
                            ec.codes_set(gid, key, val)
                        ec.codes_set_values(gid, slab.ravel())
                        ec.codes_write(gid, output)
                    finally:
                        ec.codes_release(gid)


def write(ds, path, fmt, chunk=128, grib_grid=None):
    path = Path(path)
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if fmt not in {"netcdf", "zarr", "grib2"}:
        raise ValueError("unknown export format")
    if fmt == "netcdf":
        # CF section 5 forbids fill attributes on dimension coordinates.
        # Share array buffers, but keep caller attributes/encoding unchanged.
        ds = ds.copy(deep=False)
        for name, coord in ds.coords.items():
            if coord.dims == (name,):
                coord.attrs = {key: value for key, value in coord.attrs.items()
                               if key not in {"_FillValue", "missing_value"}}
                coord.encoding = {key: value for key, value in coord.encoding.items()
                                  if key not in {"_FillValue", "missing_value"}}
                coord.encoding["_FillValue"] = None
        write_products(ds, path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "zarr":
        if path.exists():
            raise ValueError("Zarr destination exists; use a new directory")
        temp = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
        try:
            encoding = {name: {"chunks": tuple(1 if d in {"time", "pressure"} else min(chunk, ds.sizes[d]) for d in da.dims)}
                        for name, da in ds.variables.items() if da.ndim}
            # Zarr v2 on disk for broad xarray/client compatibility (zarr 3 library).
            ds.to_zarr(temp, mode="w", encoding=encoding, zarr_format=2, consolidated=True)
            os.rename(temp, path)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    else:
        fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        try:
            _grib(ds, temp, grib_grid)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
