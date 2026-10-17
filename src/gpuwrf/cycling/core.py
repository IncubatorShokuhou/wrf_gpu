"""Exact-time, same-grid WRF land-state transplantation (no interpolation).

Names/units: pristine WRF 4.7.1 Registry.EM_COMMON, module_sf_noahmpdrv.F
INOUT state (lines 224-280). The restart branch at :2008 preserves this state;
the ordinary cold initializer reconstructs it. No arithmetic repairs are made.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

MARKER = "GPUWRF_LAND_CYCLE"
CORE = ("SMOIS", "SH2O", "TSLB", "SNOW", "SNOWH", "SNOWC", "TSK")
MP_REQUIRED = ("ISNOW", "TSNO", "SNICE", "SNLIQ", "ZSNSO", "SNEQVO", "ALBOLD",
               "TV", "TG", "TAH", "EAH", "CANLIQ", "CANICE", "FWET", "LAI", "XSAI")
MP_OPTIONAL = ("TAUSS", "SMCWTD", "CM", "CH", "ALBEDO", "EMISS", "ZNT", "QSFC",
               "WA", "WT", "ZWT", "LFMASS", "RTMASS", "STMASS", "WOOD", "STBLCP", "FASTCP")
VEGETATION = frozenset(("TV", "TAH", "EAH", "FWET", "LAI", "XSAI", "LFMASS",
                        "RTMASS", "STMASS", "WOOD", "STBLCP", "FASTCP"))
SOIL = frozenset(("SMOIS", "SH2O", "TSLB"))
SNOW_LAYERS = frozenset(("TSNO", "SNICE", "SNLIQ"))
SURFACE_DIMS = ("south_north", "west_east")
LAND_LEAVES = {"TSLB": "tslb", "SMOIS": "smois", "SH2O": "sh2o", "SMCWTD": "smcwtd",
               "ISNOW": "isnow", "TSNO": "tsno", "SNICE": "snice", "SNLIQ": "snliq",
               "ZSNSO": "zsnso", "SNOWH": "snowh", "SNOW": "sneqv", "SNEQVO": "sneqvo",
               "TAUSS": "tauss", "ALBOLD": "albold", "TV": "tv", "TG": "tg", "TAH": "tah",
               "EAH": "eah", "CANLIQ": "canliq", "CANICE": "canice", "FWET": "fwet",
               "LAI": "lai", "XSAI": "sai", "CM": "cm", "CH": "ch", "TSK": "t_skin",
               "QSFC": "qsfc", "ZNT": "znt", "EMISS": "emiss", "ALBEDO": "albedo"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _times(ds):
    if "Times" not in ds.variables:
        raise ValueError("Times is required; timestamps are never inferred from filenames")
    result = []
    for row in ds["Times"][:]:
        stamp = b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").strip("\0 ")
        try:
            datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S")
        except ValueError as exc:
            raise ValueError(f"invalid WRF Times value {stamp!r}") from exc
        result.append(stamp)
    return result


def _read(ds, name, index):
    if name not in ds.variables:
        raise ValueError(f"required variable {name} is missing")
    var = ds[name]
    if "Time" in var.dimensions and var.dimensions[0] != "Time":
        raise ValueError(f"{name}: Time must be the first dimension")
    values = np.ma.asarray(var[index] if var.dimensions[:1] == ("Time",) else var[:])
    data = np.asarray(values.data)
    if data.dtype.kind not in "fiu" or not np.isfinite(data).all():
        raise ValueError(f"{name}: non-finite or non-numeric values")
    return data, np.ma.getmaskarray(values)


def _dims(var):
    return var.dimensions[1:] if var.dimensions[:1] == ("Time",) else var.dimensions


def _exact(source, target, name, index):
    a, ma = _read(source, name, index)
    b, mb = _read(target, name, 0)
    if ma.any() or mb.any() or _dims(source[name]) != _dims(target[name]) or not np.array_equal(a, b):
        raise ValueError(f"grid/categories/soil layers differ: {name}")
    return b


def _grid(source, target, index):
    required = ("DX", "DY", "MAP_PROJ", "GRID_ID", "MMINLU", "NUM_LAND_CAT",
                "ISWATER", "ISICE", "SF_SURFACE_PHYSICS")
    optional = ("CEN_LAT", "CEN_LON", "MOAD_CEN_LAT", "STAND_LON", "TRUELAT1", "TRUELAT2",
                "POLE_LAT", "POLE_LON", "ISLAKE", "ISURBAN", "ISOILWATER")
    for name in required + optional:
        if name in required or name in source.ncattrs() or name in target.ncattrs():
            if name not in source.ncattrs() or name not in target.ncattrs():
                raise ValueError(f"grid metadata missing: {name}")
            a, b = source.getncattr(name), target.getncattr(name)
            if isinstance(a, (float, np.floating)) and not np.isfinite(a):
                raise ValueError(f"non-finite grid metadata: {name}")
            if a != b:
                raise ValueError(f"grid/categories metadata differ: {name}")
    if int(target.SF_SURFACE_PHYSICS) not in (2, 4):
        raise ValueError("cycling supports Noah (2) and Noah-MP (4) soil/snow state only")
    for name in ("XLAT", "XLONG", "HGT", "LU_INDEX", "IVGTYP", "ISLTYP", "ZS", "DZS"):
        _exact(source, target, name, index)
    landmask = _exact(source, target, "LANDMASK", index)
    xland = _exact(source, target, "XLAND", index)
    if not np.isin(landmask, (0, 1)).all() or not np.isin(xland, (1, 2)).all():
        raise ValueError("invalid LANDMASK/XLAND")
    if not np.array_equal(landmask == 1, xland == 1):
        raise ValueError("LANDMASK and XLAND disagree")
    land = landmask == 1
    for name in ("LAKEMASK", "SEAICE"):
        if name in source.variables or name in target.variables:
            value = _exact(source, target, name, index)
            if name == "LAKEMASK":
                if not np.isin(value, (0, 1)).all():
                    raise ValueError("invalid LAKEMASK")
                land &= value == 0
    if land.shape != (len(target.dimensions["south_north"]), len(target.dimensions["west_east"])):
        raise ValueError("land mask shape differs from grid")
    if not land.any():
        raise ValueError("no land cells to cycle")
    ncat = int(target.NUM_LAND_CAT)
    for name in ("LU_INDEX", "IVGTYP", "ISLTYP"):
        data, _ = _read(target, name, 0)
        if (data < 1).any() or not np.equal(data, np.floor(data)).all():
            raise ValueError(f"invalid categories: {name}")
        if name != "ISLTYP" and (data > ncat).any():
            raise ValueError(f"invalid categories: {name}")
    for name in ("ZS", "DZS"):
        data, _ = _read(target, name, 0)
        if data.ndim != 1 or len(data) != len(target.dimensions["soil_layers_stag"]) or (data <= 0).any():
            raise ValueError(f"invalid soil layers: {name}")
    return land


def field_mask(ds, name, index, land, shape):
    """Only land is cycled; canopy has no state where WRF FVEG is zero.

    WRF glacier driver writes missing-value sentinels for canopy/groundwater;
    preserve the new input at these physically inactive cells (never clamp).
    """
    mask = land.copy()
    if name in VEGETATION:
        veg, missing = _read(ds, "FVEG", index)
        if veg.shape != land.shape or missing[land].any() or ((veg[land] < 0) | (veg[land] > 1)).any():
            raise ValueError("invalid FVEG canopy mask")
        mask &= veg > 0
    if name in VEGETATION or name in ("WA", "WT"):
        categories, _ = _read(ds, "IVGTYP", index)
        mask &= categories != int(ds.ISICE)
    return np.broadcast_to(mask, shape)


def _field(ds, name, index, land):
    data, missing = _read(ds, name, index)
    dims = (("soil_layers_stag",) if name in SOIL else
            ("snow_layers_stag",) if name in SNOW_LAYERS else
            ("snso_layers_stag",) if name == "ZSNSO" else ()) + SURFACE_DIMS
    if _dims(ds[name]) != dims or data.shape[-2:] != land.shape:
        raise ValueError(f"{name}: incompatible field dimensions")
    mask = field_mask(ds, name, index, land, data.shape)
    if missing[mask].any():
        raise ValueError(f"{name}: missing values on active cells")
    bounds = (0, 1e8)
    if name in ("SMOIS", "SH2O", "SMCWTD", "SNOWC", "FWET", "ALBOLD", "ALBEDO", "EMISS"):
        bounds = (0, 1)
    elif name in ("TSLB", "TSK", "TV", "TG", "TAH"):
        bounds = (150, 350)
    elif name == "TSNO":
        counts, _ = _read(ds, "ISNOW", index)
        active = np.arange(data.shape[0])[:, None, None] - 2 >= counts[None] + 1
        values = data[mask & active]
        if ((values < 150) | (values > 350)).any():
            raise ValueError("TSNO: active snow temperature outside sanity bounds")
        bounds = (0, 350)
    elif name == "ISNOW":
        if not np.isin(data[mask], (-3, -2, -1, 0)).all():
            raise ValueError("ISNOW: invalid active snow-layer count")
        bounds = (-3, 0)
    elif name == "ZSNSO":
        bounds = (-10000, 0)
    elif name in ("LAI", "XSAI"):
        bounds = (0, 100)
    if ((data[mask] < bounds[0]) | (data[mask] > bounds[1])).any():
        raise ValueError(f"{name}: outside sanity bounds {bounds}")
    return data, mask


def _units(var):
    # Equivalent WRF water mass/area and depth units; no numerical conversion.
    value = str(getattr(var, "units", "")).strip().lower()
    return {"mm": "kg m-2", "kg/m2": "kg m-2", "m3/m3": "m3 m-3", "pa": "pa"}.get(value, value)


def _digest(data, mask):
    return hashlib.sha256(np.ascontiguousarray(data[mask]).tobytes()).hexdigest()


def validate_columns(ds, index, land, *, noahmp):
    """Host-only admission, before publication or any model-state mutation.

    Pristine noahmplsm.F:826-833,6478-6533 supplies the geometry, inactive
    zeros and bulk sums. Bounds use gamma_n=n*u/(1-n*u) for the original REAL
    operations, not observed port error; see proofs/u2_cycling/COLUMN_BOUNDS.md.
    Only comparisons are made. Valid input storage bytes are never changed.
    """
    moisture, _ = _field(ds, "SMOIS", index, land)
    liquid, _ = _field(ds, "SH2O", index, land)
    if np.any(liquid[:, land] > moisture[:, land]):
        raise ValueError("SH2O exceeds total SMOIS")
    if not noahmp:
        return
    names = ("ISNOW", "ZSNSO", "TSNO", "SNICE", "SNLIQ", "SNOW", "SNOWH")
    fields = {name: _field(ds, name, index, land)[0] for name in names}
    dzs, missing = _read(ds, "DZS", index)
    if dzs.shape != (4,) or missing.any() or np.any(dzs <= 0):
        raise ValueError("soil/snow column: invalid DZS")
    if fields["ZSNSO"].shape != (7,) + land.shape or any(
            fields[n].shape != (3,) + land.shape for n in ("TSNO", "SNICE", "SNLIQ")):
        raise ValueError("soil/snow column: requires 3 snow and 4 soil layers")

    def unit_roundoff(*arrays):
        return max((np.finfo(a.dtype).eps / 2 for a in arrays if a.dtype.kind == "f"),
                   default=np.finfo(np.float32).eps / 2)

    def gamma(n, u):
        return (n * u) / (1 - n * u)

    z = fields["ZSNSO"].astype(np.float64)
    previous = np.concatenate((np.zeros((1,) + land.shape), z[:-1]), axis=0)
    thickness = previous - z
    active = np.arange(3)[:, None, None] >= 3 + fields["ISNOW"][None]
    active_land = active & land[None]
    inactive_land = ~active & land[None]
    for name in ("TSNO", "SNICE", "SNLIQ"):
        if np.any(fields[name][inactive_land] != 0):
            raise ValueError(f"soil/snow column: inactive snow {name} must be zero")
    if np.any(z[:3][inactive_land] != 0):
        raise ValueError("soil/snow column: inactive snow ZSNSO must be zero")
    if np.any(thickness[:3][active_land] <= 0) or np.any(thickness[3:, land] <= 0):
        raise ValueError("soil/snow column: active snow/soil thickness must be positive")

    # ZSOIL cumulative construction <=4 REAL adds; differencing its endpoints
    # adds one rounding. Final ZSNSO accumulation contributes gamma_1*|z_k|.
    # Float64 host evaluation gets its own two-operation error allowance.
    u_geometry = unit_roundoff(fields["ZSNSO"], dzs)
    u_host = np.finfo(np.float64).eps / 2
    dzs64 = dzs.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        soil_depth = np.cumsum(dzs64)
    if not np.isfinite(soil_depth).all():
        raise ValueError("soil/snow column: non-finite cumulative DZS")
    soil_previous = np.concatenate(([0.0], soil_depth[:-1]))
    soil_bound = (gamma(5, u_geometry) * (soil_depth + soil_previous))[:, None, None]
    soil_bound = soil_bound + gamma(1, u_geometry) * np.abs(z[3:])
    soil_bound += gamma(2, u_host) * (np.abs(z[3:]) + np.abs(previous[3:]) + dzs64[:, None, None])
    if np.any((np.abs(thickness[3:] - dzs64[:, None, None]) > soil_bound)[:, land]):
        raise ValueError("soil/snow column: soil thickness disagrees with DZS")

    ice = fields["SNICE"].astype(np.float64)
    water = fields["SNLIQ"].astype(np.float64)
    layer_mass = ice + water
    if np.any(layer_mass[active_land] <= 0):
        raise ValueError("soil/snow column: active snow layer mass must be positive")
    mass = np.sum(layer_mass, axis=0)
    bulk = fields["SNOW"].astype(np.float64)
    height = fields["SNOWH"].astype(np.float64)
    layered = (fields["ISNOW"] < 0) & land
    u_mass = unit_roundoff(fields["SNICE"], fields["SNLIQ"], fields["SNOW"])
    # SNOWWATER: <=6 adds; SNOW_INIT: division, two products, two depth
    # subtractions, and <=3 layer sums (<=8 operations). No fixed epsilon floor.
    mass_bound = gamma(8, u_mass) * (mass + bulk) + gamma(6, u_host) * mass
    if np.any((np.abs(mass - bulk) > mass_bound)[layered]):
        raise ValueError("soil/snow column: layered mass disagrees with SNOW")
    u_depth = unit_roundoff(fields["ZSNSO"], fields["SNOWH"])
    depth = -z[2]
    depth_bound = (gamma(6, u_depth) + gamma(2, u_host)) * (np.abs(depth) + height)
    if np.any((np.abs(depth - height) > depth_bound)[layered]):
        raise ValueError("soil/snow column: layered depth disagrees with SNOWH")
    # With ISNOW=0, thin snow remains in bulk SWE/height, not the three slots;
    # its soil interfaces start at the soil surface, without the bulk height.
    if np.any(((bulk == 0) != (height == 0))[land]):
        raise ValueError("soil/snow column: SNOW and SNOWH must vanish together")


def cycle(source, wrfinput, output):
    """Write a new wrfinput, embedding the complete transfer/provenance record.

    Existing files are never overwritten. The output is published atomically
    after all validation and writes, so failures leave both inputs untouched.
    Transfers stream one field at a time; admission holds bounded land/snow
    column arrays. No atmosphere arrays are materialized.
    """
    source, wrfinput, output = (Path(p).resolve() for p in (source, wrfinput, output))
    if output in (source, wrfinput) or output.exists():
        raise ValueError("output must be a new file, distinct from both inputs")
    source_sha, input_sha = sha256(source), sha256(wrfinput)
    with Dataset(source) as src, Dataset(wrfinput) as dst:
        target_times = _times(dst)
        if len(target_times) != 1:
            raise ValueError("new wrfinput must have exactly one initialization time")
        time = target_times[0]
        matches = [i for i, stamp in enumerate(_times(src)) if stamp == time]
        if len(matches) != 1:
            raise ValueError(f"source must contain exactly one frame at new init time {time}")
        index = matches[0]
        land = _grid(src, dst, index)
        names = list(CORE)
        if int(dst.SF_SURFACE_PHYSICS) == 4:
            for dim, size in (("soil_layers_stag", 4), ("snow_layers_stag", 3), ("snso_layers_stag", 7)):
                if dim not in dst.dimensions or len(dst.dimensions[dim]) != size:
                    raise ValueError(f"Noah-MP requires {size} {dim} layers")
            names += list(MP_REQUIRED)
            names += [n for n in MP_OPTIONAL if n in src.variables or n in dst.variables]
        if "CANWAT" in src.variables or "CANWAT" in dst.variables:
            names.append("CANWAT")
        records = {}
        for name in names:
            data, mask = _field(src, name, index, land)
            if any(k in src[name].ncattrs() for k in ("scale_factor", "add_offset", "least_significant_digit", "_QuantizeBitGroomNumberOfSignificantDigits")):
                raise ValueError(f"{name}: packed/quantized cycling fields are unsupported")
            if name in dst.variables:
                var = dst[name]
                _read(dst, name, 0)  # reject non-finite template state as well
                if _dims(var) != _dims(src[name]) or var.shape[1:] != data.shape or var.dimensions[:1] != ("Time",):
                    raise ValueError(f"{name}: target dimensions differ")
                if var.dtype != src[name].dtype or _units(var) != _units(src[name]):
                    raise ValueError(f"{name}: target dtype/units differ (no conversion)")
                if any(k in var.ncattrs() for k in ("scale_factor", "add_offset", "least_significant_digit", "_QuantizeBitGroomNumberOfSignificantDigits")):
                    raise ValueError(f"{name}: packed/quantized cycling fields are unsupported")
            elif name in CORE:
                raise ValueError(f"required target variable {name} is missing")
            for dim, size in zip(_dims(src[name]), data.shape):
                if dim not in dst.dimensions or len(dst.dimensions[dim]) != size:
                    raise ValueError(f"{name}: target layer/grid dimension {dim} differs")
            records[name] = {"shape": list(data.shape), "dtype": str(data.dtype), "cells": int(mask.sum()),
                             "sha256": _digest(data, mask), "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                             "mask": "vegetation" if name in VEGETATION else
                             "non-glacier land" if name in ("WA", "WT") else "land"}
        validate_columns(src, index, land, noahmp=int(dst.SF_SURFACE_PHYSICS) == 4)
        if int(dst.SF_SURFACE_PHYSICS) == 4:
            # FVEG is persisted as auxiliary activity metadata, never a forced
            # seasonal-vegetation setting. Refuse an incompatible existing leaf.
            veg, _ = _read(src, "FVEG", index)
            if _dims(src["FVEG"]) != SURFACE_DIMS or veg.shape != land.shape:
                raise ValueError("FVEG: incompatible mask dimensions")
            if "FVEG" in dst.variables and (dst["FVEG"].dimensions != ("Time",) + SURFACE_DIMS or dst["FVEG"].dtype != src["FVEG"].dtype):
                raise ValueError("FVEG: target mask dimensions/dtype differ")
        report = {"schema": 1, "valid_time": time, "source_frame": index,
                  "source": str(source), "source_sha256": source_sha,
                  "wrfinput": str(wrfinput), "wrfinput_sha256": input_sha,
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "land_cells": int(land.sum()), "fields": records,
                  "optional_absent": [n for n in MP_OPTIONAL if n not in names],
                  "policy": "exact-time/same-grid; land-only; no interpolation, clipping or atmosphere cycling"}
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".gpuwrf-cycle-", dir=output.parent)
        os.close(fd)
        try:
            shutil.copyfile(wrfinput, temporary)
            with Dataset(temporary, "r+") as result:
                # FVEG is only needed to reproduce the explicit canopy transfer mask.
                write_names = names + (["FVEG"] if int(dst.SF_SURFACE_PHYSICS) == 4 else [])
                for name in write_names:
                    data, _ = _read(src, name, index)
                    mask = field_mask(src, name, index, land, data.shape)
                    if name not in result.variables:
                        original = src[name]
                        attrs = {k: original.getncattr(k) for k in original.ncattrs() if k != "_FillValue"}
                        var = result.createVariable(name, original.dtype, ("Time",) + _dims(original),
                                                    fill_value=getattr(original, "_FillValue", None))
                        var.setncatts(attrs)
                        var[0] = np.zeros(data.shape, dtype=data.dtype)
                    var = result[name]
                    before = np.asarray(var[0])
                    var[0] = np.where(mask, data, before)
                result.setncattr(MARKER, json.dumps(report, sort_keys=True, separators=(",", ":")))
                result.sync()
            # Refuse concurrent modifications of either input during the transfer.
            if sha256(source) != source_sha or sha256(wrfinput) != input_sha:
                raise ValueError("input changed during cycling")
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, output)  # atomic no-clobber publication
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {**report, "output": str(output), "output_sha256": sha256(output)}


def restore_land_state(ds, land_state):
    """Honor a cycling marker once at Noah-MP initialization (never per step).

    Default inputs return before imports/array work. Zero is a valid spun-up
    value: do not use the ordinary cold-input nonzero heuristic on cycled data.
    """
    if MARKER not in ds.ncattrs():
        return None
    report = json.loads(ds.getncattr(MARKER))
    if report.get("schema") != 1 or report.get("valid_time") != _times(ds)[0]:
        raise ValueError("invalid land-cycle provenance schema/time")
    if not set(CORE + MP_REQUIRED).issubset(report.get("fields", {})):
        raise ValueError("incomplete Noah-MP land-cycle provenance")
    land, _ = _read(ds, "LANDMASK", 0)
    land = land == 1
    if "LAKEMASK" in ds.variables:
        lakes, _ = _read(ds, "LAKEMASK", 0)
        land &= lakes == 0
    validate_columns(ds, 0, land, noahmp=True)
    import jax.numpy as jnp
    for name, record in report["fields"].items():
        data, mask = _field(ds, name, 0, land)
        if (_digest(data, mask) != record["sha256"] or int(mask.sum()) != record["cells"]
                or hashlib.sha256(mask.tobytes()).hexdigest() != record["mask_sha256"]
                or list(data.shape) != record["shape"] or str(data.dtype) != record["dtype"]):
            raise ValueError(f"{name}: cycled state no longer matches provenance")
        if name in LAND_LEAVES:
            attr = LAND_LEAVES[name]
            previous = getattr(land_state, attr)
            # Keep fallback buffers on their device: np.asarray(previous) would
            # add a blocking device-to-host copy for every restored land leaf.
            setattr(land_state, attr, jnp.where(jnp.asarray(mask),
                    jnp.asarray(data, dtype=previous.dtype), previous))
    return report
