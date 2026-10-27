"""Opt-in, exact-grid atmospheric cycling into fresh real.exe inputs.

All operations run offline on host NumPy arrays. Default forecasts and land
cycling do not import this module. This is a warm initialization, not a WRF
restart: time integrator scratch and unrecorded physics state are not invented.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from .boundary import finite_field, update_first_interval, _sha
from .core import _dims, _exact, _grid, _times, _cycle_land, sha256

MARKER = "GPUWRF_ATMOSPHERE_CYCLE"
MASS_DIMS = ("bottom_top", "south_north", "west_east")
BASE_FIELDS = ("PB", "PHB", "MUB", "P_TOP", "ZNU", "ZNW", "DNW", "RDNW",
               "C1H", "C2H", "C3H", "C4H", "C1F", "C2F", "C3F", "C4F",
               "MAPFAC_M", "MAPFAC_U", "MAPFAC_V")
CURRENT_SLOTS = {name: name + "_2" for name in ("U", "V", "W", "PH", "MU", "THM")}
MYNN_FIELDS = ("qke_adv", "QKE_ADV", "TSQ", "QSQ", "COV", "TKE_PBL", "EL_PBL", "SH3D", "SM3D")


def _source_name(ds, name, index=0):
    # Registry qke_adv is lowercase in original real.exe input/boundary files,
    # and is not in ordinary history. WRF's MYNN first-call driver initializes
    # this advection slot from QKE (:729-734); never fabricate a zero field.
    if name in ("qke_adv", "QKE_ADV") and name not in ds.variables:
        alias = "QKE_ADV" if name == "qke_adv" else "qke_adv"
        return alias if alias in ds.variables else "QKE"
    slot = CURRENT_SLOTS.get(name)
    if slot in ds.variables:
        if name in ds.variables and not np.array_equal(finite_field(ds, name, index), finite_field(ds, slot, index)):
            raise ValueError(f"{name}: ambiguous current restart slot {slot}")
        return slot
    return name


def _field_names(src, dst):
    from gpuwrf.contracts.physics_registry import (MOIST_WRFOUT_NAME, NUMBER_WRFOUT_NAME,
                                                 moist_species_for_mp, number_species_for_mp)
    mp, bl = int(dst.MP_PHYSICS), int(dst.BL_PBL_PHYSICS)
    if mp not in (0, 1, 2, 3, 4, 6, 8, 10, 14, 16, 24, 26, 28, 38, 40, 97) or bl != 5:
        raise ValueError("atmospheric cycling currently supports bulk microphysics with MYNN (bl=5); unsupported scheme")
    species = {MOIST_WRFOUT_NAME[n] for n in moist_species_for_mp(mp)}
    species |= {NUMBER_WRFOUT_NAME[n] for n in number_species_for_mp(mp)}
    if mp == 38:
        species.add("QVGRAUPEL")
    known = set(MOIST_WRFOUT_NAME.values()) | set(NUMBER_WRFOUT_NAME.values()) | {"QVGRAUPEL", "QVHAIL"}
    for ds in (src, dst):
        for name, var in ds.variables.items():
            if _dims(var) == MASS_DIMS and name.startswith("Q") and name not in ("QKE", "QSQ", "QKE_ADV", "QC_BL", "QI_BL"):
                if name not in known:
                    raise ValueError(f"{name}: unregistered atmospheric species; refusing a partial transfer")
                species.add(name)
    fields = ["U", "V", "W", "T", "THM", "PH", "P", "MU", *sorted(species), "QKE"]
    fields += [n for n in MYNN_FIELDS if n in src.variables or n in dst.variables]
    return fields


def _validate_grid(src, dst, index):
    _grid(src, dst, index, allow_terrain_change=True)
    for attr in ("MP_PHYSICS", "BL_PBL_PHYSICS", "HYBRID_OPT", "HYPSOMETRIC_OPT", "USE_THETA_M",
                 "PARENT_ID", "I_PARENT_START", "J_PARENT_START", "PARENT_GRID_RATIO"):
        if attr not in src.ncattrs() or attr not in dst.ncattrs() or src.getncattr(attr) != dst.getncattr(attr):
            raise ValueError(f"atmospheric grid/scheme metadata differs: {attr}")
    if int(dst.HYPSOMETRIC_OPT) != 2 or int(dst.USE_THETA_M) not in (0, 1):
        raise ValueError("atmospheric cycling requires HYPSOMETRIC_OPT=2 and USE_THETA_M=0/1")
    for dim, base in (("bottom_top_stag", "bottom_top"), ("west_east_stag", "west_east"),
                      ("south_north_stag", "south_north")):
        if dim not in dst.dimensions or base not in dst.dimensions or len(dst.dimensions[dim]) != len(dst.dimensions[base]) + 1:
            raise ValueError(f"invalid atmospheric grid dimension {dim}")
        if dim not in src.dimensions or len(src.dimensions[dim]) != len(dst.dimensions[dim]):
            raise ValueError(f"atmospheric grid dimension differs: {dim}")
    for name in BASE_FIELDS:
        if name in ("PB", "PHB", "MUB"):
            a, b = finite_field(src, name, index), finite_field(dst, name)
            if a.shape != b.shape or _dims(src[name]) != _dims(dst[name]):
                raise ValueError(f"atmospheric base dimensions differ: {name}")
            continue
        _exact(src, dst, name, index)


def _inverse_density(ds):
    """WRF start_em :803-812, HYPSOMETRIC_OPT=2; host-only derivation of AL."""
    def get(name):
        return finite_field(ds, name).astype(np.float64)
    def col(name):
        return get(name)[:, None, None]
    mu = get("MU") + get("MUB")
    phi = get("PH") + get("PHB")
    ptop = float(get("P_TOP").reshape(()))
    pfaces = col("C3F") * mu + col("C4F") + ptop
    pmid = col("C3H") * mu + col("C4H") + ptop
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        alpha = np.diff(phi, axis=0) / (pmid * np.log(pfaces[:-1] / pfaces[1:]))
    if not np.isfinite(alpha).all() or np.any(alpha <= 0):
        raise ValueError("MU/PH: invalid inverse density in start_em reconstruction")
    return alpha


def _transfer(source, template, staging, *, preserve_source_derived=False):
    source_sha, template_sha = sha256(source), sha256(template)
    with Dataset(source) as src, Dataset(template) as dst:
        stamps = _times(dst)
        if len(stamps) != 1:
            raise ValueError("new wrfinput must have exactly one initialization time")
        matches = [i for i, stamp in enumerate(_times(src)) if stamp == stamps[0]]
        if len(matches) != 1:
            raise ValueError("atmospheric source must contain exactly one frame at new init time")
        index = matches[0]
        _validate_grid(src, dst, index)
        names = _field_names(src, dst)
        records = {}
        identical = True
        for name in names:
            origin = _source_name(src, name, index)
            data = finite_field(src, origin, index)
            dims = MASS_DIMS
            if name == "U":
                dims = ("bottom_top", "south_north", "west_east_stag")
            elif name == "V":
                dims = ("bottom_top", "south_north_stag", "west_east")
            elif name in ("W", "PH"):
                dims = ("bottom_top_stag", "south_north", "west_east")
            elif name == "MU":
                dims = MASS_DIMS[1:]
            # TKE_PBL/EL_PBL are WRF Z-staggered in some MYNN histories.
            elif name in ("TKE_PBL", "EL_PBL"):
                dims = _dims(src[origin])
                if dims not in (MASS_DIMS, ("bottom_top_stag",) + MASS_DIMS[1:]):
                    raise ValueError(f"{name}: invalid MYNN state dimensions")
            if _dims(src[origin]) != dims or data.shape != tuple(len(dst.dimensions[d]) for d in dims):
                raise ValueError(f"{name}: incompatible atmospheric field dimensions")
            if name in dst.variables:
                var = dst[name]
                before = finite_field(dst, name)
                if var.dimensions != ("Time",) + dims or var.dtype != src[origin].dtype:
                    raise ValueError(f"{name}: target dimensions/dtype differ")
                from .core import _units
                if _units(var) != _units(src[origin]):
                    raise ValueError(f"{name}: target units differ")
                identical &= np.array_equal(data, before)
            else:
                identical = False
            records[name] = {"source_variable": origin, "source_frame": index,
                             "shape": list(data.shape), "dtype": str(data.dtype), "sha256": _sha(data),
                             "inherited_extrema": {"min": float(data.min()), "max": float(data.max()),
                                                   "negative_count": int(np.count_nonzero(data < 0))}}
        # Registry T/THM convention: a bad moist-theta slot must never pass as
        # a valid dry-theta field. The eight-operation REAL bound is generous
        # enough for the forward/inverse history conversion, without repair.
        theta = finite_field(src, "T", index).astype(np.float64) + 300.
        expected = theta * (1 + (461.6 / 287.) * finite_field(src, "QVAPOR", index)) if int(dst.USE_THETA_M) else theta
        thm = finite_field(src, _source_name(src, "THM", index), index).astype(np.float64) + 300.
        if np.any(np.abs(expected - thm) > 8 * np.finfo(src["T"].dtype).eps * np.abs(expected)):
            raise ValueError("THM: inconsistent dry/moist potential temperature convention")
        # Existing land admission/copy is reused without altering its default path.
        if source_sha == template_sha:
            shutil.copyfile(template, staging)
            land_report = {"fields": {}, "land_cells": 0}
        else:
            land_report = _cycle_land(source, template, staging, allow_terrain_change=True)
        with Dataset(staging, "r+") as out:
            for name in names:
                original = src[records[name]["source_variable"]]
                if name not in out.variables:
                    var = out.createVariable(name, original.dtype, ("Time",) + _dims(original),
                                             fill_value=getattr(original, "_FillValue", None))
                    var.setncatts({k: original.getncattr(k) for k in original.ncattrs() if k != "_FillValue"})
                data = finite_field(src, original.name, index)
                if name in ("P", "PH", "MU"):
                    base_name = {"P": "PB", "PH": "PHB", "MU": "MUB"}[name]
                    old_base, new_base = finite_field(src, base_name, index), finite_field(dst, base_name)
                    delta = old_base.astype(np.float64) - new_base.astype(np.float64)
                    if np.any(delta != 0):
                        identical = False
                        data = (data.astype(np.float64) + delta).astype(data.dtype)
                        records[name]["rebased"] = True
                        records[name]["source_sha256"] = records[name]["sha256"]
                        records[name]["sha256"] = _sha(data)
                out[name][0] = data
            derived = {}
            if not identical and not preserve_source_derived:
                if "AL" in out.variables:
                    data = (_inverse_density(out) - finite_field(out, "ALB")).astype(out["AL"].dtype)
                    out["AL"][0] = data
                    derived["AL"] = {"rule": "start_em PH/MU inverse density minus new ALB", "sha256": _sha(data)}
                if "P_HYD" in out.variables:
                    data = finite_field(out, "P") + finite_field(out, "PB")
                    out["P_HYD"][0] = data
                    derived["P_HYD"] = {"rule": "wrfinput total P+PB (start_em)", "sha256": _sha(data)}
            report = {"schema": 1, "valid_time": stamps[0], "source": str(source), "source_sha256": source_sha,
                      "required_namelist": {"time_control": {"cycling": True}},
                      "wrfinput": str(template), "wrfinput_sha256": template_sha,
                      "grid_id": int(dst.GRID_ID), "parent_id": int(dst.PARENT_ID),
                      "grid": {k: float(dst.getncattr(k)) for k in ("DX", "DY", "PARENT_GRID_RATIO", "I_PARENT_START", "J_PARENT_START")},
                      "created_utc": datetime.now(timezone.utc).isoformat(), "fields": records,
                      "derived": derived,
                      "land": {"fields": len(land_report["fields"]), "land_cells": land_report["land_cells"]},
                      "static_policy": "fresh template; roundoff-only bases; preserve total P/MU/PH by perturbation rebasing",
                      "policy": "atmospheric warm initialization; no interpolation/blending/clipping or restart claim"}
            out.sync()
        from gpuwrf.io.wrfinput_consistency import validate_wrfinput
        report["consistency"] = validate_wrfinput(staging, surface_physics=int(dst.SF_SURFACE_PHYSICS),
                                transport_reference=source, transport_frame=index)
        with Dataset(staging, "r+") as out:
            out.setncattr(MARKER, json.dumps(report, sort_keys=True, separators=(",", ":")))
            out.sync()
    if sha256(source) != source_sha or sha256(template) != template_sha:
        raise ValueError("atmospheric input changed during cycling")
    return report


def cycle_atmosphere(source, wrfinput, output, *, wrfbdy, boundary_output, nests=(), native=False):
    """Preflight/stage all domains and publish new files without clobbering inputs.

    ``nests`` contains (source, fresh_wrfinput, output) triples ordered d02..dN.
    The root boundary is updated; nest boundaries come from the cycled parent.
    A failed publication rolls back only links created by this invocation.
    """
    if wrfbdy is None or boundary_output is None:
        raise ValueError("atmospheric cycling requires wrfbdy and boundary_output")
    if nests and not native:
        raise ValueError("fresh-base overlay cycling is d01-only; nested cycling requires native_atmosphere=True")
    domains = [tuple(Path(p).resolve() for p in triple) for triple in [(source, wrfinput, output), *nests]]
    wrfbdy, boundary_output = Path(wrfbdy).resolve(), Path(boundary_output).resolve()
    inputs = {wrfbdy, *(p for s, t, _ in domains for p in (s, t))}
    outputs = [*(o for _, _, o in domains), boundary_output]
    if len(set(outputs)) != len(outputs) or any(o in inputs or o.exists() for o in outputs):
        raise ValueError("outputs must be new, distinct files separate from all inputs")
    temporary, published, reports = [], [], []
    input_sha = {p: sha256(p) for p in inputs}
    try:
        for i, (s, t, o) in enumerate(domains):
            o.parent.mkdir(parents=True, exist_ok=True)
            fd, path = tempfile.mkstemp(prefix=".gpuwrf-atmosphere-", dir=o.parent)
            os.close(fd)
            Path(path).unlink()  # the reused land publisher requires a new name
            temporary.append(Path(path))
            if native:
                from .native import transfer_native
                report = transfer_native(s, t, path)
            else:
                report = _transfer(s, t, path)
            if report["grid_id"] != i + 1 or (i and report["parent_id"] not in range(1, i + 1)):
                raise ValueError("domains must form an ordered d01..dN nest tree")
            grid = report["grid"]
            if i:
                parent = reports[report["parent_id"] - 1]["grid"]
                ratio = grid["PARENT_GRID_RATIO"]
                if ratio < 2 or ratio != int(ratio) or min(grid["I_PARENT_START"], grid["J_PARENT_START"]) < 1:
                    raise ValueError("invalid nested domain ratio/start")
                if any(not np.isclose(grid[k] * ratio, parent[k], rtol=2*np.finfo(np.float32).eps, atol=0) for k in ("DX", "DY")):
                    raise ValueError("nested domain spacing disagrees with parent/ratio")
            elif grid["PARENT_GRID_RATIO"] != 1 or report["parent_id"] != 1:
                raise ValueError("root domain parent/ratio must equal one")
            if reports and report["valid_time"] != reports[0]["valid_time"]:
                raise ValueError("nested domain initialization times differ")
            reports.append(report)
        boundary_output.parent.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix=".gpuwrf-atmosphere-", dir=boundary_output.parent)
        os.close(fd)
        temporary.append(Path(path))
        shutil.copyfile(wrfbdy, path)
        with Dataset(temporary[0]) as initial, Dataset(path, "r+") as bdy:
            bdy_report = update_first_interval(initial, bdy, allowed_fields=set(reports[0]["fields"]))
            bdy.setncattr(MARKER, json.dumps({"schema": 1, "valid_time": reports[0]["valid_time"],
                           "wrfbdy_sha256": input_sha[wrfbdy], **bdy_report}, sort_keys=True, separators=(",", ":")))
            bdy.sync()
        if any(sha256(p) != digest for p, digest in input_sha.items()):
            raise ValueError("input changed during cycling")
        for path, target in zip(temporary, outputs):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(path, target)
            published.append(target)
        return {"schema": 1, "valid_time": reports[0]["valid_time"],
                "domains": [{**r, "output": str(o), "output_sha256": sha256(o)} for r, o in zip(reports, outputs)],
                "boundary": {**bdy_report, "source": str(wrfbdy), "source_sha256": input_sha[wrfbdy],
                             "output": str(boundary_output), "output_sha256": sha256(boundary_output)}}
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
