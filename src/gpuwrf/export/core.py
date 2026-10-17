"""Explicit CF surface/isobaric products; no model or JAX imports."""
from __future__ import annotations

import numpy as np
import xarray as xr

from gpuwrf.post.common import PROFILE, field, earth_wind, pressure, temperature
from gpuwrf.post.io import read_wrfout

# WRF name: output name, CF standard name, units. Q2/QVAPOR are MIXING RATIO.
FIELDS = {
    "T2": ("temperature_2m", "air_temperature", "K"),
    "PSFC": ("surface_pressure", "surface_air_pressure", "Pa"),
    "Q2": ("mixing_ratio_2m", "humidity_mixing_ratio", "1"),
    "U10": ("eastward_wind_10m", "eastward_wind", "m s-1"),
    "V10": ("northward_wind_10m", "northward_wind", "m s-1"),
    "HGT": ("terrain_height", "surface_altitude", "m"),
    "T": ("temperature", "air_temperature", "K"),
    "QVAPOR": ("mixing_ratio", "humidity_mixing_ratio", "1"),
    "U": ("eastward_wind", "eastward_wind", "m s-1"),
    "V": ("northward_wind", "northward_wind", "m s-1"),
    "PH": ("geopotential_height", "geopotential_height", "m"),
}
PROFILES = {"T", "QVAPOR", "U", "V", "PH"}


def mapping(attrs):
    """CF Appendix F, WRF's spherical 6370000 m Earth."""
    proj = int(attrs.get("MAP_PROJ", -1))
    common = {"earth_radius": 6370000.}
    def attr(key):
        value = float(attrs[key])
        if not np.isfinite(value):
            raise ValueError(f"nonfinite projection attribute {key}")
        return value
    if proj == 1:
        common.update(grid_mapping_name="lambert_conformal_conic",
                      standard_parallel=[attr("TRUELAT1"), attr("TRUELAT2")],
                      longitude_of_central_meridian=attr("STAND_LON"),
                      latitude_of_projection_origin=attr("MOAD_CEN_LAT"))
    elif proj == 2:
        common.update(grid_mapping_name="polar_stereographic",
                      straight_vertical_longitude_from_pole=attr("STAND_LON"),
                      latitude_of_projection_origin=90. if attr("TRUELAT1") > 0 else -90.,
                      standard_parallel=attr("TRUELAT1"))
    elif proj == 3:
        common.update(grid_mapping_name="mercator", standard_parallel=attr("TRUELAT1"),
                      longitude_of_projection_origin=attr("STAND_LON"))
    elif proj == 6 and float(attrs.get("POLE_LAT", 90)) == 90 and float(attrs.get("POLE_LON", 0)) == 0:
        common.update(grid_mapping_name="latitude_longitude")
    else:
        raise ValueError(f"unsupported MAP_PROJ={proj} (rotated grids are not exported)")
    return common


def log_pressure_interp(p, a, levels):
    """Linear in log(P), strictly monotonic columns, no extrapolation."""
    levels = np.asarray(levels, dtype=float) * 100.
    if (levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all()
            or (levels <= 0).any() or len(np.unique(levels)) != len(levels)):
        raise ValueError("pressure levels must be unique positive finite hPa values")
    delta = np.diff(levels)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError("pressure levels must be strictly monotonic (increasing or decreasing order)")
    if np.any(np.diff(p, axis=1) >= 0):
        raise ValueError("pressure must strictly decrease with model level")
    if np.any(levels.max() > p[:, 0]) or np.any(levels.min() < p[:, -1]):
        raise ValueError("pressure level outside a column; below-ground/extrapolation refused")
    result = np.empty((p.shape[0], len(levels), *p.shape[2:]))
    for t, y, x in np.ndindex(p.shape[0], *p.shape[2:]):
        result[t, :, y, x] = np.interp(np.log(levels), np.log(p[t, ::-1, y, x]), a[t, ::-1, y, x])
    return result


def required(names):
    keep = set(names) | {"XLAT", "XLONG", "HGT"}
    if set(names) & {"U", "V", "U10", "V10"}:
        keep |= {"COSALPHA", "SINALPHA"}
        if set(names) & {"U10", "V10"}:
            keep |= {"U10", "V10"}
        if set(names) & {"U", "V"}:
            keep |= {"U", "V"}
    if set(names) & PROFILES:
        keep |= {"P", "PB"}
    if "PH" in names:
        keep |= {"PH", "PHB"}
    return keep


def convert(ds, names, levels=None, region=None):
    names = list(dict.fromkeys(names))
    if not names or set(names) - FIELDS.keys():
        raise ValueError(f"choose WRF variables from {', '.join(FIELDS)}")
    if bool(set(names) & PROFILES) != bool(levels):
        raise ValueError("--pressure-levels is required exactly when selecting profile fields T/QVAPOR/U/V/PH")
    lat, lon = field(ds, "XLAT"), field(ds, "XLONG")
    if np.any(abs(lat) > 90) or np.any(abs(lon) > 180):
        raise ValueError("invalid latitude/longitude")
    if not np.array_equal(lat, np.broadcast_to(lat[0], lat.shape)) or not np.array_equal(lon, np.broadcast_to(lon[0], lon.shape)):
        raise ValueError("moving grids are not supported")
    # Crop before derived calculations; input loading is variable-selective.
    if region is not None:
        west, east, south, north = region
        if not np.isfinite(region).all() or not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("region must be WEST EAST SOUTH NORTH; dateline crossing unsupported")
        inside = (lon[0] >= west) & (lon[0] <= east) & (lat[0] >= south) & (lat[0] <= north)
        yy, xx = np.where(inside)
        if not len(yy):
            raise ValueError("region contains no grid centres")
        y0, y1, x0, x1 = yy.min(), yy.max()+1, xx.min(), xx.max()+1
        ds = ds.isel(south_north=slice(y0, y1), west_east=slice(x0, x1),
                     **({"south_north_stag": slice(y0, y1+1)} if "south_north_stag" in ds.dims else {}),
                     **({"west_east_stag": slice(x0, x1+1)} if "west_east_stag" in ds.dims else {}))
        lat, lon = field(ds, "XLAT"), field(ds, "XLONG")
    crs = mapping(ds.attrs)
    result = xr.Dataset(coords={"time": ("time", ds.Time.values, {"standard_name": "time", "axis": "T"}),
                                "latitude": (("y", "x"), lat[0], {"standard_name": "latitude", "units": "degrees_north"}),
                                "longitude": (("y", "x"), lon[0], {"standard_name": "longitude", "units": "degrees_east"})})
    result["crs"] = xr.DataArray(np.int32(0), attrs=crs)
    if crs["grid_mapping_name"] != "latitude_longitude":
        try:
            from pyproj import CRS, Transformer
        except ImportError as exc:
            raise ValueError("projected export requires pip install 'gpuwrf[export]'") from exc
        target = CRS.from_cf(crs)
        x, y = Transformer.from_crs(target.geodetic_crs, target, always_xy=True).transform(lon[0], lat[0])
        # WRF's float32 geographical coordinates have metre-scale rounding.
        if not np.allclose(x, x[:1], atol=10, rtol=0) or not np.allclose(y, y[:, :1], atol=10, rtol=0):
            raise ValueError("lat/lon inconsistent with rectilinear WRF projection")
        result = result.assign_coords(x=("x", np.mean(x, axis=0), {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}),
                                      y=("y", np.mean(y, axis=1), {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}))
    if levels:
        result = result.assign_coords(pressure=("pressure", np.asarray(levels)*100.,
            {"standard_name": "air_pressure", "units": "Pa", "positive": "down", "axis": "Z"}))
        p = pressure(ds)
    wind = {}
    if set(names) & {"U10", "V10"}:
        wind["U10"], wind["V10"] = earth_wind(ds, field(ds, "U10"), field(ds, "V10"), {})
    if set(names) & {"U", "V"}:
        u = field(ds, "U", ("Time", "bottom_top", "south_north", "west_east_stag"))
        v = field(ds, "V", ("Time", "bottom_top", "south_north_stag", "west_east"))
        wind["U"], wind["V"] = earth_wind(ds, .5*(u[..., :-1]+u[..., 1:]), .5*(v[:, :, :-1]+v[:, :, 1:]), {})
    for name in names:
        if name in wind:
            a = wind[name]
        elif name == "T":
            if int(ds.attrs.get("USE_THETA_M", 0)) != 0:
                raise ValueError("T is moist theta (USE_THETA_M); dry-theta conversion refused")
            a = temperature(ds)
        elif name == "PH":
            a = (field(ds, "PH", ("Time", "bottom_top_stag", "south_north", "west_east"))
                 + field(ds, "PHB", ("Time", "bottom_top_stag", "south_north", "west_east"))) / 9.81
            a = .5*(a[:, :-1]+a[:, 1:])
        else:
            a = field(ds, name, PROFILE if name in PROFILES else ("Time", "south_north", "west_east"))
        if name in {"QVAPOR", "Q2"} and np.any(a < 0):
            raise ValueError(f"negative {name} mixing ratio")
        if name in PROFILES:
            a = log_pressure_interp(p, a, levels)
        out, standard, units = FIELDS[name]
        dims = ("time", "pressure", "y", "x") if name in PROFILES else ("time", "y", "x")
        result[out] = xr.DataArray(a, dims=dims, attrs={"standard_name": standard, "units": units,
            "grid_mapping": "crs", "cell_methods": "time: point", "wrf_source_variable": name})
        result[out].encoding["coordinates"] = "latitude longitude"
        if name in {"T2", "Q2", "U10", "V10"}:
            height = "height_2m" if name in {"T2", "Q2"} else "height_10m"
            result.coords[height] = xr.DataArray(2. if height == "height_2m" else 10., attrs={"standard_name": "height", "units": "m", "positive": "up"})
            result[out].encoding["coordinates"] = f"latitude longitude {height}"
    for name, da in result.variables.items():
        if da.dtype.kind in "fiu" and not np.isfinite(da.values).all():
            raise ValueError(f"nonfinite exported field {name}")
    result.attrs = {"Conventions": "CF-1.10", "title": "WRF offline surface/isobaric export",
                    "source": "WRF history; spherical Earth radius 6370000 m",
                    "history": "gpuwrf export; destagger, rotate winds, interpolate linearly in log pressure",
                    "region_sampling": "smallest rectangular index envelope of grid centres inside bbox; no masking",
                    "input_files": ds.attrs.get("post_source_files", "")}
    return result


def load(paths, names, levels=None, region=None):
    return convert(read_wrfout(paths, required(names)), names, levels, region)


def raw_subset(paths, names, region=None):
    """Preserve selected WRF arrays/attrs without claiming a CF conversion."""
    ds = read_wrfout(paths, set(names) | {"XLAT", "XLONG"})
    missing = set(names) - set(ds.variables)
    if missing:
        raise ValueError(f"missing history fields: {sorted(missing)}")
    if region is not None:
        west,east,south,north = region
        if not np.isfinite(region).all() or not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("invalid region; dateline crossing unsupported")
        lat,lon = field(ds,"XLAT")[0],field(ds,"XLONG")[0]
        yy,xx = np.where((lon>=west)&(lon<=east)&(lat>=south)&(lat<=north))
        if not len(yy):
            raise ValueError("region contains no grid centres")
        slices={"south_north":slice(yy.min(),yy.max()+1),"west_east":slice(xx.min(),xx.max()+1)}
        for dim in tuple(slices):
            if dim+"_stag" in ds.dims:
                sl=slices[dim]
                slices[dim+"_stag"]=slice(sl.start,sl.stop+1)
        ds=ds.isel(slices)
    for name,da in ds.variables.items():
        if da.dtype.kind in "fiu" and not np.isfinite(da.values).all():
            raise ValueError(f"nonfinite history field {name}")
    ds.attrs.pop("Conventions",None)
    ds.attrs["export_sampling"]="WRF native grid and units; rectangular index envelope; no CF conversion"
    return ds
