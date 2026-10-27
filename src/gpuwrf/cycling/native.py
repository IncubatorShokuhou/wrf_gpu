"""Native WRF cycling inputs: resolved source grid, fresh issue/forcing clock.

SAVE_TOPO_FROM_REAL=0 in WRF history/restart output prevents a second terrain
blend (output_wrf.F:1004-1021; mediation_integrate.F:739). Fresh real.exe inputs
supply missing input-only fields, never replace the source's resolved bases.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrf_init_math import _wrf_start_domain_t_init_alb32
from .core import _grid, _exact, _dims, _read, _times, sha256
from .boundary import finite_field, _sha


def _topography_flag(ds, index):
    value, mask = _read(ds, "SAVE_TOPO_FROM_REAL", index)
    if value.size != 1 or value.dtype.kind not in "iu" or mask.any() or int(value.reshape(())) not in (0, 1):
        raise ValueError("native cycling requires integer SAVE_TOPO_FROM_REAL=0/1")
    return int(value.reshape(()))


def _admit_pair(src, fresh, index):
    from .atmosphere import BASE_FIELDS
    _grid(src, fresh, index, allow_terrain_change=True)
    for attr in ("MP_PHYSICS", "BL_PBL_PHYSICS", "HYBRID_OPT", "HYPSOMETRIC_OPT", "USE_THETA_M",
                 "PARENT_ID", "I_PARENT_START", "J_PARENT_START", "PARENT_GRID_RATIO"):
        if attr not in src.ncattrs() or attr not in fresh.ncattrs() or src.getncattr(attr) != fresh.getncattr(attr):
            raise ValueError(f"native cycling grid/scheme metadata differs: {attr}")
    for name in BASE_FIELDS + ("RDN",):
        if name not in ("PB", "PHB", "MUB"):
            _exact(src, fresh, name, index)
    flag = _topography_flag(src, index)
    fresh_flag = _topography_flag(fresh, 0)
    if fresh_flag != 1:
        raise ValueError("native cycling analysis template must be original real.exe input (SAVE_TOPO_FROM_REAL=1)")
    if flag == 1:
        # Cold-input identity is useful, but cannot bypass material terrain
        # admission by merely selecting the native API.
        for name in ("HGT", "PB", "PHB", "MUB"):
            _exact(src, fresh, name, index)
    return flag


# WRFDA da_update_bc update_low_bdy (default .true. when cycling, nests always;
# da_update_bc.f90:96-110,131-150,213-222): lower-boundary/static fields come
# from the fresh real.exe input (:409-436). LANDMASK/XLAND/IVGTYP/ISLTYP are
# already admitted byte-equal to the source (core._grid), so copying them is exact.
LOW_BDY_FROM_FRESH = ("TMN", "SST", "VEGFRA", "ALBBCK", "SEAICE", "LANDMASK", "XLAND", "IVGTYP", "ISLTYP")


def _update_low_bdy(out, fresh):
    """Apply WRFDA update_low_bdy with its defaults keep_tsk_wrf=keep_snow_wrf=.true.,
    update_lsm=.false. (da_update_bc.f90:101-106): soil, CANWAT and land snow stay
    from the cycled state; TSK comes from the fresh input where its IVGTYP is water
    (:365-405); snow is removed over water (:476-517). ISWATER is the fresh input's
    category attribute (WRFDA's namelist iswater must match the land-use dataset)."""
    def field(ds, name):
        data, missing = _read(ds, name, 0)
        if missing.any() or (data.dtype.kind == "f" and not np.isfinite(data).all()):
            raise ValueError(f"low-boundary update {name}: missing or non-finite values")
        return np.asarray(data)

    iswater = int(fresh.getncattr("ISWATER"))
    ivgtyp = field(fresh, "IVGTYP")
    water = ivgtyp == iswater
    record = {"iswater": iswater, "water_cells": int(water.sum()), "fields": {}}

    def put(name, data):
        target = out[name]
        if tuple(target.shape[1:]) != data.shape or target.dimensions[:1] != ("Time",) or target.dtype != data.dtype:
            raise ValueError(f"low-boundary update {name}: incompatible template/fresh field")
        target[0] = data
        return _sha(data)

    for name in LOW_BDY_FROM_FRESH:
        if name not in out.variables:  # WRFDA skips variables absent from the analysis
            continue
        if name not in fresh.variables:
            raise ValueError(f"low-boundary update {name}: missing from the fresh real.exe input")
        data = field(fresh, name)
        record["fields"][name] = {"sha256": put(name, data), "rule": "fresh real.exe (da_update_bc.f90:409-436)"}
    if "TSK" in out.variables:
        warm, new = field(out, "TSK"), field(fresh, "TSK")
        data = np.where(water, new, warm).astype(warm.dtype)
        record["fields"]["TSK"] = {"sha256": put("TSK", data), "water_cells_from_fresh": int(water.sum()),
                                   "rule": "keep_tsk_wrf: cycled where fresh IVGTYP/=ISWATER, else fresh (:365-405)"}
    if "SNOW" in out.variables:
        snow = field(out, "SNOW")
        remove = water & (snow > 0)
        record["snow_over_water_removed_cells"] = int(remove.sum())
        for name in ("SNOW", "SNOWC", "SNOWH"):
            if name in out.variables:
                data = field(out, name)
                data = np.where(remove, np.zeros_like(data), data)
                record["fields"][name] = {"sha256": put(name, data),
                                          "rule": "keep_snow_wrf; zero where IVGTYP==ISWATER and SNOW>0 (:476-517)"}
    return record


def _copy_var(source, target, name, index):
    original = source[name]
    if "Time" in original.dimensions and original.dimensions[0] != "Time":
        raise ValueError(f"native cycling {name}: Time must be first")
    original.set_auto_maskandscale(False)
    original.set_auto_chartostring(False)
    attrs = {k: original.getncattr(k) for k in original.ncattrs() if k != "_FillValue"}
    var = target.createVariable(name, original.datatype, original.dimensions,
                                fill_value=getattr(original, "_FillValue", None))
    var.setncatts(attrs)
    var.set_auto_maskandscale(False)
    var.set_auto_chartostring(False)
    data = np.asarray(original[index] if original.dimensions[:1] == ("Time",) else original[:])
    if data.dtype.kind in "fiu" and not np.isfinite(data).all():
        raise ValueError(f"native cycling {name}: non-finite source/template values")
    if original.dimensions[:1] == ("Time",):
        var[0] = data
    else:
        var[:] = data
    return {"shape": list(data.shape), "dtype": str(data.dtype), "sha256": _sha(data)}


def _make_template(source, fresh_path, path):
    with Dataset(source) as src, Dataset(fresh_path) as fresh:
        stamps = _times(fresh)
        if len(stamps) != 1:
            raise ValueError("native analysis template must have one time")
        matches = [i for i, t in enumerate(_times(src)) if t == stamps[0]]
        if len(matches) != 1:
            raise ValueError("native source must contain exactly one frame at the new issue time")
        index = matches[0]
        flag = _admit_pair(src, fresh, index)
        with Dataset(path, "w", format=fresh.data_model) as out:
            attrs = {k: src.getncattr(k) for k in src.ncattrs()}
            for name in fresh.ncattrs():
                if name not in attrs or name in ("START_DATE", "SIMULATION_START_DATE", "GMT", "JULYR", "JULDAY"):
                    attrs[name] = fresh.getncattr(name)
            out.setncatts(attrs)
            for ds in (src, fresh):
                for name, dim in ds.dimensions.items():
                    size = 1 if name == "Time" else len(dim)
                    if name in out.dimensions:
                        if len(out.dimensions[name]) != size and name != "Time":
                            raise ValueError(f"native source/template dimension differs: {name}")
                    else:
                        out.createDimension(name, None if name == "Time" else size)
            copied, filled = {}, {}
            for name in src.variables:
                copied[name] = _copy_var(src, out, name, index)
            for name in fresh.variables:
                if name not in out.variables:
                    filled[name] = _copy_var(fresh, out, name, 0)
            low_bdy = _update_low_bdy(out, fresh)
            for name in low_bdy["fields"]:
                copied.pop(name, None)
                filled.pop(name, None)
            clock = {}
            for name in ("XTIME", "STEP_NUMBER"):
                if name in out.variables and name in fresh.variables:
                    original, target = fresh[name], out[name]
                    original.set_auto_maskandscale(False)
                    data = np.asarray(original[0] if original.dimensions[:1] == ("Time",) else original[:])
                    if target.dimensions != original.dimensions or target.dtype != original.dtype:
                        raise ValueError(f"native cycling clock {name}: incompatible template")
                    if target.dimensions[:1] == ("Time",):
                        target[0] = data
                    else:
                        target[:] = data
                    for attr in ("units", "description"):
                        if attr in original.ncattrs():
                            target.setncattr(attr, original.getncattr(attr))
                    clock[name] = {"sha256": _sha(data), "source": "fresh real.exe issue clock"}
                    copied.pop(name, None)
            derived = {}
            if "ALB" not in src.variables or "T_INIT" not in src.variables:
                scalar_names = {"p_top": "P_TOP", "p00": "P00", "t00": "T00", "a": "TLP",
                                "tiso": "TISO", "a_strat": "TLP_STRAT", "p_strat": "P_STRAT"}
                scalars = {k: np.float32(finite_field(out, v).reshape(())) for k,v in scalar_names.items()}
                t_init, alb = _wrf_start_domain_t_init_alb32(finite_field(out, "PB"), scalars)
                for name, data in (("T_INIT", t_init), ("ALB", alb)):
                    if name not in src.variables:
                        out[name][0] = data
                        derived[name] = {"sha256": _sha(data), "rule": "WRF reference profile from SOURCE PB/base parameters"}
                        filled.pop(name, None)
            if "AL" not in src.variables:
                from .atmosphere import _inverse_density
                data = (_inverse_density(out) - finite_field(out, "ALB")).astype(out["AL"].dtype)
                out["AL"][0] = data
                derived["AL"] = {"sha256": _sha(data), "rule": "start_em inverse density from source PH/MU minus source ALB"}
                filled.pop("AL", None)
            out.sync()
    return {"source_topography_flag": flag, "source_frame": index, "copied": copied,
            "input_only_from_fresh": filled, "derived": derived, "clock": clock,
            "low_boundary_update": low_bdy}


def transfer_native(source, fresh, staging):
    """Compose a source-based native input, then reuse land/field admission."""
    from .atmosphere import _transfer, MARKER
    from .core import MARKER as LAND_MARKER
    source, fresh = Path(source), Path(fresh)
    fresh_sha = sha256(fresh)
    if sha256(source) == fresh_sha:
        with Dataset(fresh) as ds:
            if _topography_flag(ds, 0) != 1:
                raise ValueError("native identity template must be a real.exe wrfinput")
        report = _transfer(source, fresh, staging)
        report["native"] = {"identity": True, "source_topography_flag": 1}
        with Dataset(staging, "r+") as out:
            out.setncattr(MARKER, json.dumps(report, sort_keys=True, separators=(",", ":")))
        return report
    fd, name = tempfile.mkstemp(prefix=".gpuwrf-native-template-", dir=Path(staging).parent)
    os.close(fd)
    temporary = Path(name)
    try:
        inventory = _make_template(source, fresh, temporary)
        template_sha = sha256(temporary)
        report = _transfer(source, temporary, staging, preserve_source_derived=True)
        for field, record in report["fields"].items():
            if field in inventory["input_only_from_fresh"]:
                inventory["input_only_from_fresh"].pop(field)
                inventory["derived"][field] = {"sha256": record["sha256"],
                    "rule": "carried/initialized from source variable " + record["source_variable"]}
        report.update(wrfinput=str(fresh), wrfinput_sha256=fresh_sha, native=inventory,
                      native_template_sha256=template_sha,
                      static_policy="native WRF cycling: SOURCE resolved HGT/PB/PHB/MUB retained; fresh forcing outside wrfinput",
                      policy="source-frame native cycling; no second terrain blend when SAVE_TOPO_FROM_REAL=0")
        with Dataset(staging, "r+") as out:
            if LAND_MARKER in out.ncattrs():
                land = json.loads(out.getncattr(LAND_MARKER))
                land.update(wrfinput=str(fresh), wrfinput_sha256=fresh_sha, native_template_sha256=template_sha)
                out.setncattr(LAND_MARKER, json.dumps(land, sort_keys=True, separators=(",", ":")))
            out.setncattr(MARKER, json.dumps(report, sort_keys=True, separators=(",", ":")))
            out.sync()
        return report
    finally:
        temporary.unlink(missing_ok=True)
