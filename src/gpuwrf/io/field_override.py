"""Declarative, offline edits to private copies of WRF initialization files."""
from __future__ import annotations

from collections.abc import Mapping
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import warnings

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfinput_consistency import (
    EDITABLE_FIELDS, NOAHMP_NO_EFFECT, WrfinputConsistencyError, checked_waivers,
    validate_wrfinput,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _keys(obj, allowed, label):
    if not isinstance(obj, Mapping) or set(obj) - set(allowed):
        raise ValueError(f"{label}: expected mapping with keys {sorted(allowed)}")


def load_spec(path):
    """Load JSON/YAML safely; file references resolve beside the spec."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        spec = json.loads(path.read_text())
    else:
        import yaml
        try:
            spec = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML spec: {exc}") from exc
    return spec, path.parent


def _values(description, base, default_variable):
    """A scalar/array or {file,variable} (npy, npz, NetCDF), with provenance."""
    if not isinstance(description, Mapping):
        return np.asarray(description), {"kind": "literal", "values": description}
    _keys(description, {"file", "variable"}, "file values")
    if not isinstance(description.get("file"), str):
        raise ValueError("file values require a file path")
    path = (base / description["file"]).resolve()
    before = sha256(path)
    variable = description.get("variable", default_variable)
    if path.suffix.lower() == ".npy":
        if "variable" in description:
            raise ValueError("npy has no variable selector")
        a = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            a = archive[variable].copy()
    else:
        with Dataset(path) as ds:
            if variable not in ds.variables:
                raise ValueError(f"{path}: missing {variable}")
            a = ds[variable][:]
            if np.ma.isMaskedArray(a) and np.any(a.mask):
                raise ValueError(f"{path}:{variable}: masked/fill values")
            a = np.asarray(a)
    if sha256(path) != before:
        raise ValueError(f"values file changed during read: {path}")
    return a, {"kind": "file", "path": str(path), "sha256": before, "variable": variable}


def _numeric(a, label):
    a = np.asarray(a)
    if a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)):
        raise ValueError(f"{label}: numeric finite values required (NaN/Inf refused)")
    return a


def _field_values(a, shape):
    if a.ndim == 0:
        return np.broadcast_to(a, shape)
    if a.shape == shape or (shape[0] == 1 and a.shape == shape[1:]):
        return a.reshape(shape)
    raise ValueError(f"values shape {a.shape}: expected scalar or exact field shape {shape} (Time=1 may be omitted)")


def _mask(description, ds, field, base, *, init_path):
    shape = ds[field].shape
    if isinstance(description, str) and description in {"land", "water"}:
        if "LANDMASK" not in ds.variables:
            with Dataset(init_path) as initial:
                land = _numeric(initial["LANDMASK"][:], "LANDMASK")
        else:
            land = _numeric(ds["LANDMASK"][:], "LANDMASK")
        a = land == (1 if description == "land" else 0)
        provenance = {"kind": "named", "mask": description}
    else:
        a, provenance = _values(description, base, "mask")
        a = np.asarray(a)
        if a.dtype.kind not in "biuf" or not np.all(np.isfinite(a)) or not np.all(np.isin(a, [0, 1])):
            raise ValueError("mask must contain only finite boolean/0/1 values")
    if a.shape not in (shape, shape[1:], shape[-2:], (1, *shape[-2:])) and a.ndim != 0:
        raise ValueError(f"mask shape {a.shape} incompatible with {shape}")
    return np.broadcast_to(a.astype(bool), shape), provenance


def _edit(path, item, base, *, surface_physics, force_no_effect):
    field = item["field"]
    note = None
    if surface_physics == 4 and field in NOAHMP_NO_EFFECT:
        note = f"{field}: NO EFFECT under Noah-MP; use --surface-controls"
        if not force_no_effect:
            raise ValueError(note + " (explicit --force-no-effect required to store it anyway)")
        warnings.warn(note, UserWarning, stacklevel=2)
    with Dataset(path, "r+") as ds:
        ds.set_auto_maskandscale(False)
        if field not in ds.variables:
            raise ValueError(f"{path.name}: missing field {field}; this tool never creates variables")
        var = ds[field]
        if any(k in var.ncattrs() for k in ("scale_factor", "add_offset", "_Unsigned")):
            raise ValueError(f"{field}: packed variables are unsupported")
        before = _numeric(var[:], field)
        values, provenance = _values(item.get("values", {"file": item.get("file"), **({"variable": item["variable"]} if "variable" in item else {})}), base, field)
        values = _field_values(_numeric(values, field), before.shape)
        mask, mask_provenance = (_mask(item["mask"], ds, field, base, init_path=path.with_name(path.name.replace("wrflowinp", "wrfinput"))) if "mask" in item
                                  else (np.ones(before.shape, bool), {"kind": "all"}))
        op = item.get("op", "set")
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = values if op == "set" else before.astype(np.float64) * values if op == "scale" else before.astype(np.float64) + values
        _numeric(candidate, field)
        if var.dtype.kind in "iu" and (np.any(candidate[mask] != np.floor(candidate[mask])) or
            np.any(candidate[mask] < np.iinfo(var.dtype).min) or np.any(candidate[mask] > np.iinfo(var.dtype).max)):
            raise ValueError(f"{field}: fractional/out-of-range integer edit")
        after = before.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            after[mask] = candidate[mask]
        _numeric(after, field)
        # Record storage changes (including signed zero), not float comparisons.
        width = before.dtype.itemsize
        changed = np.any(before.view(np.uint8).reshape(*before.shape, width) != after.view(np.uint8).reshape(*after.shape, width), axis=-1)
        if np.any(changed):
            var[:] = after
            ds.sync()
            if np.ascontiguousarray(var[:]).tobytes() != after.tobytes():
                raise ValueError(f"{field}: stored values differ from requested edits (quantized variable?)")
        return {"field": field, "op": op, "provenance": provenance, "mask": mask_provenance,
                "sha256_before": _array_sha(before), "sha256_after": _array_sha(after),
                "dtype": str(before.dtype), "shape": list(before.shape),
                "cells_changed": int(changed.sum()), "warning": note}, changed


def _validate_aux(path, wrfinput, changes, waivers):
    """Aux4 has many Times and fewer fields; validate each edited forcing record.

    Borrow static masks/schema from the matching init through a disposable
    in-memory-shaped check file would add IO. Here check the forcing arrays
    directly and retain the static-mask check with the init mask.
    """
    checks = []
    with Dataset(wrfinput) as initial, Dataset(path) as ds:
        land = np.asarray(initial["LANDMASK"][:]) == 1
        for name, changed in changes.items():
            var = ds[name]
            if var.dimensions != ("Time", "south_north", "west_east") or var.shape[1:] != land.shape[1:]:
                raise ValueError(f"wrflowinp {name}: expected Time,y,x on matching init grid")
            a = _numeric(var[:], name)
            ok = True if name == "ALBBCK" else not np.any(changed & np.broadcast_to(land if name == "SST" else ~land, changed.shape))
            status = "PASS" if ok else "WAIVED" if "land_water_mask" in waivers else "FAIL"
            checks.append({"name": "land_water_mask", "status": status, "detail": f"aux4 {name} edit footprint"})
            if name == "SST":
                ok = np.all(a[changed] > 0)
            elif name == "VEGFRA":
                ok = np.all((a[changed] >= 0) & (a[changed] <= 100))
            else:
                ok = np.all((a[changed] >= 0) & (a[changed] <= 1))
            status = "PASS" if ok else "WAIVED" if "field_ranges" in waivers else "FAIL"
            checks.append({"name": "field_ranges", "status": status, "detail": f"aux4 {name} valid range"})
    report = {"passed": not any(c["status"] == "FAIL" for c in checks), "checks": checks, "waivers": waivers}
    if not report["passed"]:
        raise WrfinputConsistencyError(report)
    return report


def _publish(staging, output):
    # Same Linux atomic, no-clobber publication as the existing cycle-case seam.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(output))


def override_inputs(input_dir, output_dir, spec, *, spec_base=None, surface_physics=4,
                    force_no_effect=False):
    """Publish a new directory of edited wrfinput/optional wrflowinp copies.

    Output must not exist. No original is opened writable. The sibling staging
    directory is removed on refusal; publication uses an exclusive output claim.
    Empty specs copy every selected file without opening it writable.
    """
    _keys(spec, {"overrides", "waivers", "max_relative_humidity"}, "spec")
    entries = spec.get("overrides", [])
    if not isinstance(entries, list):
        raise ValueError("overrides must be a list")
    waivers = checked_waivers(spec.get("waivers"))
    root, output = Path(input_dir).resolve(), Path(output_dir).absolute()
    base = Path(spec_base or ".").resolve()
    if not root.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("input directory must exist and output must be new")
    files = {p.name: p for p in root.iterdir() if re.fullmatch(r"wrfinput_d[0-9]{2}|wrflowinp_d[0-9]{2}", p.name)}
    if not any(name.startswith("wrfinput_") for name in files):
        raise ValueError("input directory has no wrfinput_dNN")
    plans = {}
    for item in entries:
        _keys(item, {"nest", "field", "op", "values", "file", "variable", "mask", "target"}, "override")
        nest, field = item.get("nest"), item.get("field")
        if isinstance(nest, bool) or not isinstance(nest, int) or not 1 <= nest <= 99:
            raise ValueError("nest must be an integer 1..99")
        if surface_physics == 4 and field in {"ALBEDO", "SH2O"}:
            guidance = " or edit SMOIS + TSLB (SH2O is rebuilt from them)" if field == "SH2O" else ""
            raise ValueError(f"{field}: recomputed by Noah-MP; NO EFFECT under Noah-MP; use --surface-controls{guidance}")
        if field not in EDITABLE_FIELDS:
            raise ValueError(f"unsupported field {field} (HGT/base-state edits are refused)")
        if item.get("op", "set") not in {"set", "scale", "add"}:
            raise ValueError("op must be set, scale or add")
        if ("values" in item) == ("file" in item) or ("variable" in item and "file" not in item):
            raise ValueError("provide exactly one of values or file (variable selects a file field)")
        target = item.get("target", "wrfinput")
        if target not in {"wrfinput", "wrflowinp"}:
            raise ValueError("target must be wrfinput or wrflowinp")
        if target == "wrflowinp" and field not in {"SST", "VEGFRA", "ALBBCK"}:
            raise ValueError("wrflowinp edits support SST/VEGFRA/ALBBCK only")
        name = f"{target}_d{nest:02d}"
        if name not in files or f"wrfinput_d{nest:02d}" not in files:
            raise ValueError(f"missing input file for {name}")
        plans.setdefault(name, []).append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.override-", dir=output.parent))
    report = {"schema_version": 1, "tool": "gpuwrf override", "surface_physics": surface_physics,
              "force_no_effect": bool(force_no_effect), "spec": spec, "files": {},
              "scope": "input files only; carry namelist.input/wrfbdy from the source case separately"}
    try:
        for name, source in sorted(files.items()):
            original_sha = sha256(source)
            target = staging / name
            shutil.copyfile(source, target)
            if sha256(target) != original_sha:
                raise ValueError(f"source changed while copying: {source}")
            changes, edits = {}, []
            for item in plans.get(name, []):
                edit, changed = _edit(target, item, base, surface_physics=surface_physics, force_no_effect=force_no_effect)
                edits.append(edit)
                changes[item["field"]] = changes.get(item["field"], False) | changed
            if name.startswith("wrfinput_"):
                validation = validate_wrfinput(target, changed_fields=changes, waivers=waivers,
                    surface_physics=surface_physics, max_relative_humidity=spec.get("max_relative_humidity", 1.0))
            elif changes:
                validation = _validate_aux(target, staging / name.replace("wrflowinp", "wrfinput"), changes, waivers)
            else:
                validation = {"passed": True, "checks": [{"name": "unchanged_copy", "status": "PASS", "detail": "byte-exact aux4 copy"}], "waivers": waivers}
            for edit in edits:
                edit["checks"] = validation["checks"]
            report["files"][name] = {"source": str(source), "output": str(output / name),
                "sha256_before": original_sha, "sha256_after": sha256(target),
                "edits": edits, "validation": validation}
        # Refuse a moving input set as well as moving value files.
        if any(sha256(files[name]) != record["sha256_before"] for name, record in report["files"].items()):
            raise ValueError("source inputs changed during override")
        (staging / "OVERRIDE_PROTOCOL.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        _publish(staging, output)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)
