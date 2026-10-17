"""Uniform fractional wind perturbations in WRF's stored IC/BC variables.

The same positive factor multiplies U and V in every nest and all boundary
values/rates for one member. Mass and map factors stay fixed, so multiplication
commutes with WRF's momentum coupling. This is an unbalanced ensemble recipe,
not SKEBS/SPPT, a spatial noise model, or a WRF RNG implementation.
"""
from __future__ import annotations

import json
import math
from numbers import Integral
from pathlib import Path
import shutil

import numpy as np

from gpuwrf.ens.members import identity
from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset

_IC_DIMS = {
    "U": ("Time", "bottom_top", "south_north", "west_east_stag"),
    "V": ("Time", "bottom_top", "south_north_stag", "west_east"),
}
_SIDES = ("XS", "XE", "YS", "YE")
_BC_NAMES = tuple(f"{field}_{kind}{side}" for field in ("U", "V")
                  for kind in ("B", "BT") for side in _SIDES)


def factors(members: int, seed: int, wind_std: float) -> np.ndarray:
    """PCG64 normals, centered and rescaled to sample std (ddof=1).

    Normalization induces dependence between members. The ensemble is finite
    and centered, rather than independent normal samples with prescribed sigma.
    """
    if isinstance(members, bool) or not isinstance(members, Integral) or not 2 <= members <= 10000:
        raise ValueError("members must be an integer in [2,10000]")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0,2**64)")
    if isinstance(wind_std, bool) or not math.isfinite(wind_std) or not 0 < wind_std < 1:
        raise ValueError("wind_std must be finite and in (0,1)")
    noise = np.random.Generator(np.random.PCG64(seed)).standard_normal(members)
    noise -= noise.mean()
    noise *= wind_std / noise.std(ddof=1)
    result = 1.0 + noise
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("recipe would reverse wind direction; reduce wind_std")
    if np.unique(result.astype(np.float32)).size < 2:
        raise ValueError("wind_std is too small to produce distinct REAL32 multipliers")
    return result


def _values(variable):
    if np.dtype(variable.dtype) != np.dtype("float32"):
        raise ValueError(f"{variable.name}: expected WRF REAL float32")
    if any(name in variable.ncattrs() for name in ("scale_factor", "add_offset")):
        raise ValueError(f"{variable.name}: packed wind fields are unsupported")
    value = variable[:]
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError(f"{variable.name}: missing wind data")
    value = np.asarray(value)
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"{variable.name}: empty or NaN/Inf wind data")
    return value


def _check(path, *, boundary):
    names = _BC_NAMES if boundary else tuple(_IC_DIMS)
    with Dataset(path) as dataset:
        if dataset.groups:
            raise ValueError(f"{path.name}: grouped files are unsupported")
        for name in names:
            if name not in dataset.variables:
                raise ValueError(f"{path.name}: missing {name}")
            variable = dataset[name]
            if boundary:
                along = ("south_north" + ("_stag" if name[0] == "V" else "") if name[-2] == "X"
                         else "west_east" + ("_stag" if name[0] == "U" else ""))
                expected = ("Time", "bdy_width", "bottom_top", along)
                if variable.dimensions != expected:
                    raise ValueError(f"{name}: expected WRF boundary layout {expected}")
            elif variable.dimensions != _IC_DIMS[name] or variable.shape[0] != 1:
                raise ValueError(f"{name}: expected one-time WRF staggered IC layout {_IC_DIMS[name]}")
            _values(variable)
    return names


def _scale(path, names, factor):
    rows = {}
    with Dataset(path, "r+") as dataset:
        for name in names:
            variable = dataset[name]
            # Match the REAL storage and expose the actual rounded multiplier.
            value = _values(variable)
            with np.errstate(over="ignore", invalid="ignore"):
                changed = value * np.float32(factor)
            if not np.isfinite(changed).all():
                raise ValueError(f"{name}: perturbation produced NaN/Inf")
            variable[:] = changed
            rows[name] = {"cells": int(value.size), "changed": int(np.count_nonzero(changed != value))}
    return rows


def prepare(case, output, *, members, seed, wind_std, ic_only=False):
    """Copy only modified inputs; link fixed inputs; publish hashed provenance.

    Outputs are new directories, never modified source files or existing cases.
    The manifest records the exact factors, original and prepared input hashes,
    and all changed variable counts. The cases can feed ``ens prepare batch``.
    """
    multipliers = factors(members, seed, wind_std)
    if not isinstance(ic_only, bool):
        raise ValueError("ic_only must be boolean")
    case = Path(case).resolve(strict=True)
    output = Path(output).absolute()
    if not case.is_dir():
        raise ValueError("case must be a directory")
    if output.exists() or output.is_symlink():
        raise ValueError("output exists; refusing overwrite")
    config = parse_namelist(case / "namelist.input")
    count = config.get("domains", {}).get("max_dom", 1)
    if isinstance(count, list):
        count = count[0]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 99:
        raise ValueError("namelist max_dom must be an integer in [1,99]")
    inputs = [case / f"wrfinput_d{i:02d}" for i in range(1, count + 1)]
    boundary = case / "wrfbdy_d01"
    if not ic_only and not boundary.is_file():
        raise ValueError("matched IC/BC perturbations require wrfbdy_d01; use --ic-only explicitly")
    boundaries = sorted(case.glob("wrfbdy_d[0-9][0-9]"))
    modified = {path: _check(path, boundary=False) for path in inputs}
    if not ic_only:
        modified.update({path: _check(path, boundary=True) for path in boundaries})
    linked = [case / "namelist.input", *sorted(case.glob("wrflowinp_d[0-9][0-9]"))]
    if ic_only:
        linked.extend(boundaries)
    # One source hash pass regardless of member count. Resolve links in receipt.
    source_ids = {path.name: identity(path) for path in (*modified, *linked)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        rows = []
        for index, factor in enumerate(multipliers):
            name = f"member-{index:03d}"
            dest = output / name
            dest.mkdir()
            changed = {}
            for path, names in modified.items():
                target = dest / path.name
                shutil.copyfile(path, target)
                if identity(target)["sha256"] != source_ids[path.name]["sha256"]:
                    raise ValueError(f"source changed while copying {path.name}")
                changed[path.name] = _scale(target, names, factor)
            for path in linked:
                (dest / path.name).symlink_to(path.resolve(strict=True))
            prepared = {path.name: identity(dest / path.name) for path in modified}
            rows.append({"id": name, "case": str(dest.resolve()), "factor": float(factor),
                         "factor_real32": float(np.float32(factor)), "changes": changed, "prepared": prepared})
        # Detect a source edit during preparation, including linked fixed inputs.
        if any(identity(case / name)["sha256"] != item["sha256"] for name, item in source_ids.items()):
            raise ValueError("source case changed during preparation")
        result = {"schema": "gpuwrf.perturb.wind.v1", "recipe": "uniform fractional U/V; centered PCG64; sample ddof=1",
                  "seed": int(seed), "wind_std": float(wind_std), "ic_only": ic_only,
                  "source": source_ids, "members": rows,
                  "qualification": "Offline input arithmetic; unbalanced winds; forecast skill/GPU qualification pending"}
        (output / "perturb_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException:
        # This directory was exclusively created above; never remove an input.
        shutil.rmtree(output)
        raise
