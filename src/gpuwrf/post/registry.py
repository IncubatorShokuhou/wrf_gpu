"""Small product registry shared with optional application modules."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import json
from typing import Callable, Mapping

import numpy as np
import xarray as xr

from . import products, thermo


@dataclass(frozen=True)
class Product:
    runner: Callable
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()


PRODUCTS: dict[str, Product] = {}


def register(name, runner, required=(), optional=()):
    if name in PRODUCTS:
        raise ValueError(f"product already registered: {name}")
    PRODUCTS[name] = Product(runner, tuple(required), tuple(optional))


ROTATION = ("COSALPHA", "SINALPHA")
PRESSURE = ("P", "PB")
HEIGHT = ("PH", "PHB", "HGT")
SURFACE_THERMO = ("PSFC", "T2", "Q2")
PROFILE_THERMO = (*PRESSURE, *HEIGHT, *SURFACE_THERMO, "T", "QVAPOR")
PROFILE_WIND = (*HEIGHT, "U", "V", "U10", "V10")
register("wind-speed10", products.speed10, ("U10", "V10"))
register("wind10", products.wind10, ("U10", "V10"), ROTATION)
register("wind-height", products.wind_height, PROFILE_WIND, ROTATION)
register("cloud", products.cloud, (*PRESSURE, "CLDFRA"))
register("precip", products.precip, ("RAINNC", "RAINC"), ("RAINSH", "I_RAINNC", "I_RAINC"))
register("precip-type", products.precip_type, ("RAINNC", "SNOWNC", "GRAUPELNC"), ("HAILNC", "I_RAINNC"))
register("snow", products.snow, ("SNOWH", "SNOW", "SNOWNC"))
register("thermal", thermo.thermal, (*SURFACE_THERMO, "U10", "V10"))
register("instability", thermo.instability, PROFILE_THERMO)
register("srh", thermo.srh, PROFILE_WIND, ROTATION)
register("freezing", thermo.freezing, PROFILE_THERMO)


def load_application_products():
    """Optional u1-app extension: apps.register_products() registers once itself."""
    if importlib.util.find_spec("gpuwrf.post.apps") is not None:
        importlib.import_module("gpuwrf.post.apps").register_products()
    if importlib.util.find_spec("gpuwrf.post.radar") is not None:
        importlib.import_module("gpuwrf.post.radar").register_products()


def required_inputs(names):
    load_application_products()
    unknown = set(names)-PRODUCTS.keys()
    if unknown:
        raise ValueError(f"unknown products: {', '.join(sorted(unknown))}; available: {', '.join(sorted(PRODUCTS))}")
    return set().union(*(set(PRODUCTS[n].required + PRODUCTS[n].optional) for n in names))


def run_products(ds, names, options: Mapping | None = None):
    """Evaluate named products. Every required numeric input/output must be finite.

    Undefined physical levels/windows use documented finite on-disk fill values
    and explicit status/valid flags; unexpected NaN is always an error.
    """
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("at least one product is required")
    required_inputs(names)
    options = dict(options or {})
    if ds.sizes.get("Time", 0) == 0 or "Time" not in ds.coords or not np.issubdtype(ds.Time.dtype, np.datetime64) or np.isnat(ds.Time.values).any():
        raise ValueError("products require a finite datetime64 Time coordinate (use read_wrfout)")
    if np.any(np.diff(ds.Time.values) <= np.timedelta64(0, "ns")):
        raise ValueError("timestamps must strictly increase")
    outputs, metadata = [], {}
    for name in names:
        spec = PRODUCTS[name]
        missing = set(spec.required)-set(ds.variables)
        if missing:
            raise ValueError(f"{name}: missing WRF fields {', '.join(sorted(missing))}")
        for key in spec.required + spec.optional:
            if key in ds and np.issubdtype(ds[key].dtype, np.number) and not np.isfinite(ds[key].values).all():
                raise ValueError(f"{key}: NaN/Inf or missing input")
        part = spec.runner(ds, options)
        if "Time" in part.dims:
            if "Time" not in part.coords:
                if part.sizes["Time"] != ds.sizes["Time"]:
                    raise ValueError(f"{name}: reduced Time output must supply its sample timestamps")
                part = part.assign_coords(Time=ds.Time)
            if (not np.issubdtype(part.Time.dtype, np.datetime64) or np.isnat(part.Time.values).any()
                    or not np.isin(part.Time.values, ds.Time.values).all()
                    or np.any(np.diff(part.Time.values) <= np.timedelta64(0, "ns"))):
                raise ValueError(f"{name}: output Time must be an increasing subset of input timestamps")
        for key, da in part.data_vars.items():
            if np.issubdtype(da.dtype, np.number) and not np.isfinite(da.values).all():
                raise ValueError(f"{name}/{key}: non-finite diagnostic")
        outputs.append(part)
        metadata[name] = dict(part.attrs)
    result = xr.merge(outputs, compat="equals", join="exact")
    if "Time" not in result.coords:
        raise ValueError("product output must contain sample Time")
    for old, new, unit, std in (("XLAT", "latitude", "degrees_north", "latitude"), ("XLONG", "longitude", "degrees_east", "longitude")):
        if old in ds:
            if not np.isfinite(ds[old].values).all():
                raise ValueError(f"{old}: NaN/Inf coordinate")
            geography = ds[old].sel(Time=result.Time) if "Time" in ds[old].dims else ds[old]
            result = result.assign_coords({new: geography.copy(deep=False)})
            result[new].attrs = {"units": unit, "standard_name": std}
    # Geographic arrays carry the source Time coordinate. Set metadata after
    # attaching them, otherwise xarray can replace the annotated Time object.
    result.Time.attrs = {"standard_name": "time", "long_name": "UTC history sample time", "axis": "T"}
    if "time_bounds" in result:
        result.Time.attrs["bounds"] = "time_bounds"
        encoding = {"units": "seconds since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"}
        result.Time.encoding.update(encoding)
        result.time_bounds.encoding.update(encoding)
    for key, da in result.data_vars.items():
        if "Time" in da.dims and "bounds" not in da.dims:
            da.attrs.setdefault("sampling", "instantaneous history sample" if da.attrs.get("cell_methods", "time: point") == "time: point" else "history sample window")
        if key != "time_bounds":
            da.attrs.setdefault("units", "1")
    result.attrs = {
        "Conventions": "CF-1.10", "title": "gpuwrf offline forecast products",
        "source": "CPU-WRF or gpuwrf history; model step program unchanged",
        "products": ",".join(names), "post_options": json.dumps(options, sort_keys=True, allow_nan=False),
        "product_metadata": json.dumps(metadata, sort_keys=True, default=str, allow_nan=False),
        "source_files": ds.attrs.get("post_source_files", "in-memory WRF dataset"),
        "sampling_limitation": "History sampling cannot recover sub-interval maxima, gusts, or precipitation timing.",
    }
    return result
