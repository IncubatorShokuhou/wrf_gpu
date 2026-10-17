"""Validated WRF fields and shared offline interpolation helpers."""
from __future__ import annotations

import numpy as np
import xarray as xr

SURFACE = ("Time", "south_north", "west_east")
PROFILE = ("Time", "bottom_top", "south_north", "west_east")


def field(ds, name, dims=SURFACE):
    if name not in ds:
        raise ValueError(f"missing WRF field {name}")
    da = ds[name]
    if set(da.dims) != set(dims):
        raise ValueError(f"{name}: expected dimensions {dims}, got {da.dims}")
    a = np.asarray(da.transpose(*dims).values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: NaN/Inf or missing input")
    return a


def positive(a, name, *, zero=False):
    if np.any(a < 0 if zero else a <= 0):
        raise ValueError(f"{name}: invalid {'negative' if zero else 'non-positive'} values")
    return a


def variable(values, units, long_name, dims=SURFACE, **attrs):
    return xr.DataArray(values, dims=dims, attrs={
        "units": units, "long_name": long_name, "cell_methods": "time: point", **attrs,
    })


def pressure(ds):
    return positive(field(ds, "P", PROFILE) + field(ds, "PB", PROFILE), "pressure")


def temperature(ds):
    # WRF dry potential temperature perturbation and ARW constants (Rd/Cp).
    p = pressure(ds)
    return positive((field(ds, "T", PROFILE) + 300.) * (p / 100000.) ** (287. / 1004.5), "temperature")


def height_agl(ds):
    dims = ("Time", "bottom_top_stag", "south_north", "west_east")
    ph = field(ds, "PH", dims) + field(ds, "PHB", dims)
    z = (ph[:, :-1] + ph[:, 1:]) / (2. * 9.81) - field(ds, "HGT")[:, None]
    if np.any(z <= 0) or np.any(np.diff(z, axis=1) <= 0):
        raise ValueError("geopotential: mass heights must be positive and strictly increasing AGL")
    return z


def interp_height(z, value, targets):
    """Linear component interpolation; reject extrapolation for any column."""
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 1 or not np.isfinite(targets).all() or len(targets) == 0:
        raise ValueError("heights must be a nonempty finite sequence")
    if np.any(np.diff(z, axis=1) <= 0):
        raise ValueError("height profile must strictly increase")
    if np.any(targets.min() < z[:, 0]) or np.any(targets.max() > z[:, -1]):
        raise ValueError("requested height lies outside a column; extrapolation refused")
    out = np.empty((z.shape[0], len(targets), *z.shape[2:]))
    for t, y, x in np.ndindex(z.shape[0], *z.shape[2:]):
        out[t, :, y, x] = np.interp(targets, z[t, :, y, x], value[t, :, y, x])
    return out


def earth_wind(ds, u, v, options):
    if "COSALPHA" in ds and "SINALPHA" in ds:
        c, s = field(ds, "COSALPHA"), field(ds, "SINALPHA")
        if not np.allclose(c*c + s*s, 1., atol=2e-5, rtol=0):
            raise ValueError("COSALPHA/SINALPHA are not a unit rotation")
        if u.ndim == 4:
            c, s = c[:, None], s[:, None]
        return u*c - v*s, v*c + u*s
    if options.get("wind_frame") == "earth":
        return u, v
    raise ValueError("earth-relative wind requires COSALPHA/SINALPHA, or --option wind_frame=\"earth\" for known earth-relative inputs")


def profile_wind(ds, options):
    u = field(ds, "U", ("Time", "bottom_top", "south_north", "west_east_stag"))
    v = field(ds, "V", ("Time", "bottom_top", "south_north_stag", "west_east"))
    u, v = earth_wind(ds, (u[..., :-1] + u[..., 1:]) / 2., (v[:, :, :-1] + v[:, :, 1:]) / 2., options)
    z = height_agl(ds)
    if np.any(z[:, 0] <= 10):
        raise ValueError("lowest mass height must exceed the 10 m wind anchor")
    u10, v10 = earth_wind(ds, field(ds, "U10"), field(ds, "V10"), options)
    return (np.concatenate([np.full_like(z[:, :1], 10.), z], axis=1),
            np.concatenate([u10[:, None], u], axis=1),
            np.concatenate([v10[:, None], v], axis=1))


def speed_direction(u, v):
    speed = np.hypot(u, v)
    direction = np.mod(np.degrees(np.arctan2(-u, -v)), 360.)
    return speed, np.where(speed == 0, 0., direction)


def metpy():
    try:
        import metpy.calc as calc
        from metpy.units import units
    except ImportError as exc:
        raise ValueError("this product needs the optional dependency: pip install 'gpuwrf[post]'") from exc
    return calc, units
