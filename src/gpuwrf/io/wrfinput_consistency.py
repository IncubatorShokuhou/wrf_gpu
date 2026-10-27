"""Host-only admission checks for edited real.exe inputs; never repairs data.

``changed_fields`` may be names (conservatively, every cell changed), or a
mapping of names to boolean arrays indicating actual edits. A successful report
may contain explicitly WAIVED checks; it is not a numerical WRF fidelity gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import hashlib

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset

MOIST_FIELDS = frozenset("QVAPOR QCLOUD QRAIN QICE QSNOW QGRAUP QHAIL QNCLOUD "
                         "QNDROP QNRAIN QNICE QNSNOW QNGRAUPEL QNHAIL QNWFA QNIFA QVGRAUPEL QVHAIL".split())
SURFACE_FIELDS = frozenset("SST TSK SMOIS TSLB LAI VEGFRA LU_INDEX IVGTYP ISLTYP EMISS ALBBCK ZNT".split())
EDITABLE_FIELDS = MOIST_FIELDS | SURFACE_FIELDS | {"T", "P"}
NOAHMP_NO_EFFECT = frozenset({"EMISS", "ALBBCK", "ZNT"})
WAIVABLE_CHECKS = frozenset({"moisture_positivity", "supersaturation_bound",
    "hydrostatic_consistency", "soil_porosity", "land_water_mask", "modis_classes",
    "field_ranges"})
XY = ("Time", "south_north", "west_east")
XYZ = ("Time", "bottom_top", "south_north", "west_east")
SOIL = ("Time", "soil_layers_stag", "south_north", "west_east")
CHECKPOINT_LAND = frozenset("SH2O TSNO SNICE SNLIQ SNOWH SNOW CANLIQ CANICE TV TG TAH SAI".split())
CHECKPOINT_FIELDS = CHECKPOINT_LAND | {"U", "V", "W", "PH", "QKE"}


class WrfinputConsistencyError(ValueError):
    def __init__(self, report):
        self.report = report
        failed = [c["name"] + ": " + c["detail"] for c in report["checks"] if c["status"] == "FAIL"]
        super().__init__("; ".join(failed))


def checked_waivers(waivers):
    waivers = {} if waivers is None else waivers
    if not isinstance(waivers, Mapping):
        raise ValueError("waivers must map check names to nonempty reasons")
    for name, reason in waivers.items():
        if name not in WAIVABLE_CHECKS or not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"invalid/nonwaivable check or empty waiver reason: {name}")
    return dict(waivers)


def _porosity(table_path):
    """WRF SOILPARM STAS read order: ITMP,BEXP,SMCDRY,F1,SMCMAX,... ."""
    from gpuwrf.config.paths import wrf_run_dir
    path = Path(table_path) if table_path is not None else wrf_run_dir() / "SOILPARM.TBL"
    raw = path.read_bytes()
    lines = raw.decode().splitlines()
    if lines[1].strip() != "STAS":
        raise ValueError("soil_porosity requires the SOILPARM STAS table")
    n = int(lines[2].split(",")[0])
    result = np.full(n + 1, np.nan)
    for line in lines[3:3 + n]:
        parts = line.split(",")
        index = int(parts[0])
        if index < 1 or index > n or np.isfinite(result[index]):
            raise ValueError("invalid/duplicate SOILPARM category")
        result[index] = np.float32(parts[4])
    if not np.all(np.isfinite(result[1:])) or np.any(result[1:] <= 0):
        raise ValueError("invalid SOILPARM porosity")
    return result, {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "block": "STAS"}


def validate_soil_porosity(smois, smcmax, landmask, lu_index, seaice=None,
                           surface_physics=4, *, sh2o=None, isltyp=None):
    """Return whether land water stores fit explicit effective SMCMAX values.

    Scalars/cell/layer arrays broadcast to SMOIS. With ``isltyp``, SMCMAX is a
    1-D WRF class table (1-based, unused slot 0); gather before broadcasting.
    Callers supply effective caps, including urban policy and consumer precision.
    Noah-MP glacier15 without sea ice retains the pristine SMOIS=1 exception.
    Optional SH2O must also satisfy 0 <= SH2O <= SMOIS <= cap on land. Invalid
    shapes, selected classes, NaN/Inf or masked data return False. No data writes,
    table reads, clipping or tolerance adjustment are performed.
    """
    def numeric(value):
        if np.ma.isMaskedArray(value) and np.any(value.mask):
            return None
        a = np.asarray(value)
        return a if a.dtype.kind in "biuf" and np.all(np.isfinite(a)) else None

    moisture, mask, vegetation = (numeric(v) for v in (smois, landmask, lu_index))
    if moisture is None or mask is None or vegetation is None or not moisture.size:
        return False
    cap = np.asanyarray(smcmax)
    if isltyp is not None:
        soil = numeric(isltyp)
        if soil is None or cap.ndim != 1 or not np.all((soil == np.floor(soil)) & (soil >= 1) & (soil < cap.size)):
            return False
        cap = cap[soil.astype(np.intp)]
    cap = numeric(cap)
    ice = numeric(0 if seaice is None else seaice)
    liquid = numeric(sh2o) if sh2o is not None else None
    if cap is None or ice is None or (sh2o is not None and (liquid is None or liquid.shape != moisture.shape)):
        return False
    try:
        cap = np.broadcast_to(cap, moisture.shape)
        land = np.broadcast_to(mask == 1, moisture.shape)
        glacier = np.broadcast_to(vegetation == 15, moisture.shape)
        ice = np.broadcast_to(ice, moisture.shape)
    except ValueError:
        return False
    if not np.all((cap >= 0) & (cap <= 1)):
        return False
    if surface_physics == 4:
        cap = np.where(glacier & (ice <= 0), 1., cap)
    ok = np.all(((moisture >= 0) & (moisture <= cap)) | ~land)
    if liquid is not None:
        ok = ok and np.all(((liquid >= 0) & (liquid <= moisture) & (liquid <= cap)) | ~land)
    return bool(ok)


def _retained_state(fields, meta, reference, reference_meta):
    """Verify representation-only transport against an array source view."""
    details = {}
    def pair(name):
        origin = name + "_2" if name in {"U", "V", "W", "PH", "MU", "THM"} and name + "_2" in reference else name
        if origin not in reference or name not in fields or tuple(reference_meta["variable_dimensions"][origin]) != tuple(meta["variable_dimensions"][name]):
            raise ValueError(f"transport reference: missing/mismatched {name}")
        a, b = np.asarray(reference[origin][0]), np.asarray(fields[name][0])
        if origin != name and name in reference and not np.array_equal(a, np.asarray(reference[name][0])):
            raise ValueError(f"transport reference: ambiguous current slot {origin}")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"transport reference: shape/dtype mismatch {name}")
        if a.dtype.kind != "S" and (not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
            raise ValueError(f"transport reference: nonfinite {name}")
        return a, b

    for name in ["Times", "T", *sorted((set(reference) | set(fields)) & (MOIST_FIELDS | {"U", "V", "W", "THM"}))]:
        a, b = pair(name)
        if a.tobytes() != b.tobytes():
            raise ValueError(f"transport reference: {name} must retain source storage bytes")
    for name in ("HGT", "PB", "PHB", "MUB"):
        a, b = pair(name)
        delta = np.abs(a.astype(float) - b.astype(float))
        tolerance = 4 * np.maximum(np.spacing(np.abs(a)).astype(float), np.spacing(np.abs(b)).astype(float))
        details[name + "_max_delta"] = float(np.max(delta))
        if np.any(delta > tolerance):
            raise ValueError(f"transport reference: material {name} change; max delta={np.max(delta):.9g}")
    for field, base in (("P", "PB"), ("PH", "PHB"), ("MU", "MUB")):
        a, b = pair(field)
        aa, bb = pair(base)
        delta = np.abs((a.astype(float) + aa) - (b.astype(float) + bb))
        tolerance = sum(np.spacing(np.abs(v)).astype(float) for v in (a, b, aa, bb))
        details[field + "_total_max_delta"] = float(np.max(delta))
        if np.any(delta > tolerance):
            raise ValueError(f"transport reference: {field}+{base} total not retained; max delta={np.max(delta):.9g}")
    return details


def _extrema(values, flagged):
    """Small, JSON-safe diagnostics; count the horizontal outer ring once."""
    finite = np.isfinite(values)
    count = int(np.count_nonzero(flagged))
    finite_values = values[finite] if not np.all(finite) else values
    return {"min": float(np.min(finite_values)) if finite_values.size else None,
            "max": float(np.max(finite_values)) if finite_values.size else None,
            "cells": int(values.size), "nonfinite_cells": int(values.size - np.count_nonzero(finite)),
            "flagged_cells": count,
            "outer_ring_flagged_cells": count - int(np.count_nonzero(flagged[..., 1:-1, 1:-1]))}


def validate_fields(fields, meta, *, changed_fields=(), waivers=None, surface_physics=4,
                    checkpoint=False, max_relative_humidity=1.0, soil_table=None,
                    transport_reference=None, effective_smcmax=None):
    """Check host WRF arrays without copying, repairing or writing input data.

    ``meta`` contains integer ``dimensions``, ``variable_dimensions`` tuples and
    global ``attributes``; optional ``variable_attributes`` holds fill markers.
    All arrays use WRF dimension order including Time=1. Checkpoint mode admits
    evolving state/Noah-MP leaves; unchanged checkpoints are not new analyses.
    ``transport_reference=(fields, meta)`` is a verified model-state source view.
    """
    if not isinstance(fields, Mapping) or not isinstance(meta, Mapping):
        raise ValueError("fields and meta must be mappings")
    dimensions = meta.get("dimensions", {})
    variable_dimensions = meta.get("variable_dimensions", {})
    attributes = meta.get("attributes", {})
    variable_attributes = meta.get("variable_attributes", {})
    if any(not isinstance(v, Mapping) for v in (dimensions, variable_dimensions, attributes, variable_attributes)):
        raise ValueError("meta requires dimensions, variable_dimensions and attributes mappings")
    if type(checkpoint) is not bool:
        raise ValueError("checkpoint must be True or False")
    waivers = checked_waivers(waivers)
    if not np.isfinite(max_relative_humidity) or max_relative_humidity <= 0:
        raise ValueError("max_relative_humidity must be positive and finite")
    report = {"passed": False, "checks": [], "waivers": waivers,
              "scope": "edited fields and their consistency dependencies; schema is not a CPU forecast",
              "max_relative_humidity": float(max_relative_humidity)}

    def check(name, ok, detail, **extra):
        status = "PASS" if ok else ("WAIVED" if name in waivers else "FAIL")
        report["checks"].append({"name": name, "status": status, "detail": detail, **extra})

    edits = dict(changed_fields) if isinstance(changed_fields, Mapping) else dict.fromkeys(changed_fields)
    schema_errors = []
    for name, size in dimensions.items():
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size <= 0:
            schema_errors.append(f"{name}: dimension length must be a positive integer")
    if schema_errors:
        check("schema", False, "; ".join(schema_errors))
        raise WrfinputConsistencyError(report)
    for name in ("Time", "DateStrLen", "south_north", "west_east", "bottom_top", "bottom_top_stag"):
        if name not in dimensions or dimensions[name] <= 0:
            schema_errors.append(f"missing/empty dimension {name}")
    if "Time" in dimensions and dimensions["Time"] != 1:
        schema_errors.append("wrfinput must have exactly one Time record")
    if "DateStrLen" in dimensions and dimensions["DateStrLen"] != 19:
        schema_errors.append("DateStrLen must be 19")
    if {"bottom_top", "bottom_top_stag"} <= dimensions.keys():
        if dimensions["bottom_top_stag"] != dimensions["bottom_top"] + 1:
            schema_errors.append("bottom_top_stag must equal bottom_top + 1")
    dims = {**dict.fromkeys(MOIST_FIELDS | {"T", "P", "PB"}, XYZ),
            **dict.fromkeys(SURFACE_FIELDS | {"MU", "MUB", "LANDMASK", "XLAND"}, XY),
            "SMOIS": SOIL, "TSLB": SOIL, "SH2O": SOIL, "QKE": XYZ,
            "PH": ("Time", "bottom_top_stag", "south_north", "west_east"),
            "PHB": ("Time", "bottom_top_stag", "south_north", "west_east"),
            "Times": ("Time", "DateStrLen"),
            "U": ("Time", "bottom_top", "south_north", "west_east_stag"),
            "V": ("Time", "bottom_top", "south_north_stag", "west_east"),
            "W": ("Time", "bottom_top_stag", "south_north", "west_east")}
    required = {"T", "P", "PB", "PH", "PHB", "MU", "MUB", "QVAPOR", "Times",
                "LANDMASK", "XLAND", "LU_INDEX", "ISLTYP"} | set(edits)
    if set(edits) & {"LU_INDEX", "IVGTYP", "ISLTYP", "SMOIS", "SH2O"}:
        required.add("SMOIS")
    for name in required:
        if name not in fields:
            schema_errors.append(f"missing field {name}")
        elif name in dims and tuple(variable_dimensions.get(name, ())) != dims[name]:
            schema_errors.append(f"{name}: expected dimensions {dims[name]}")
    for name in (set(fields) if checkpoint else required):
        if name not in fields:
            continue
        declared = variable_dimensions.get(name)
        if declared is None or any(d not in dimensions for d in declared):
            schema_errors.append(f"{name}: missing/unknown variable dimensions")
        elif (fields.array_shape(name) if isinstance(fields, _NetCDFFields) else np.shape(fields[name])) != tuple(dimensions[d] for d in declared):
            schema_errors.append(f"{name}: array shape disagrees with declared dimensions")
        if name in CHECKPOINT_LAND and name != "SH2O" and declared is not None:
            expected_rank = 4 if name in {"TSNO", "SNICE", "SNLIQ"} else 3
            if len(declared) != expected_rank or tuple(declared[-2:]) != XY[-2:] or declared[0] != "Time":
                schema_errors.append(f"{name}: expected WRF Time,[snow layer],y,x dimensions")
    for name in edits:
        allowed = EDITABLE_FIELDS | ({"PH", "MU", "U", "V", "W", "THM"} if transport_reference is not None else set())
        if checkpoint:
            allowed = (allowed | CHECKPOINT_FIELDS) - {"MU"}
        if name not in allowed:
            schema_errors.append(f"unsupported field {name} (HGT/base-state edits are refused)")
    if "Times" in fields:
        try:
            datetime.strptime(np.asarray(fields["Times"], dtype="S1").tobytes().decode(), "%Y-%m-%d_%H:%M:%S")
        except (ValueError, UnicodeError):
            schema_errors.append("invalid WRF Times")
    check("schema", not schema_errors, "; ".join(schema_errors) or "WRF dimensions, Times and required variables present")
    if schema_errors:
        raise WrfinputConsistencyError(report)

    def read(name):
        value = fields[name]
        a = np.asarray(value)
        masked = np.ma.isMaskedArray(value) and np.any(value.mask)
        if masked or a.dtype.kind not in "fiu" or not np.all(np.isfinite(a)) or (not checkpoint and np.any(np.abs(a) >= 1.e30)):
            check("finite", False, f"{name}: NaN/Inf, fill/sentinel or nonnumeric data")
            raise WrfinputConsistencyError(report)
        if "_FillValue" in variable_attributes.get(name, {}) and np.any(a == variable_attributes[name]["_FillValue"]):
            check("finite", False, f"{name}: fill values")
            raise WrfinputConsistencyError(report)
        return a

    # NaN/Inf in retained prognostic or edited-capable fields also refuses an
    # empty spec. This read-only pass never repairs real.exe storage values.
    finite_names = (set(fields) - {"Times"}) if checkpoint else (set(fields) & EDITABLE_FIELDS) | (required - {"Times"})
    for name in finite_names:
        read(name)

    masks = {}
    for name, cells in edits.items():
        a = read(name)
        if cells is None:
            masks[name] = np.ones(a.shape, dtype=bool)
        else:
            mask = np.asarray(cells)
            if mask.dtype != bool or mask.shape != a.shape:
                check("schema", False, f"{name}: changed mask must be boolean with exact field shape")
                raise WrfinputConsistencyError(report)
            masks[name] = mask
        if np.any(np.abs(a[masks[name]]) >= 1.e30):
            check("finite", False, f"{name}: edited fill/sentinel values")
            raise WrfinputConsistencyError(report)
    active = {name for name, mask in masks.items() if np.any(mask)}
    if checkpoint and not active and transport_reference is None:
        check("finite", True, "unchanged checkpoint arrays finite; no thermodynamic readmission")
        report["passed"] = True
        return report

    # This exemption is earned by source identity/totals, never by merely
    # passing transport_reference. A failed proof still runs strict edit checks.
    inherited_transport = False
    if transport_reference is not None:
        try:
            details = _retained_state(fields, meta, *transport_reference)
            inherited_transport = True
            check("hydrostatic_consistency", True, "source model-state totals retained (not a hydrostatic equilibrium assertion)",
                  reference=transport_reference[1].get("source_path", "array reference"),
                  frame=transport_reference[1].get("source_frame", 0), **details)
            report["inherited_source"] = {
                "admission": "verified source storage/totals; finite/schema/geometry required; no repair or waiver",
                "computed_on": "verified transported state (totals within source storage roundoff)",
                "moisture": {},
            }
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            check("source_state_transport", False, str(exc))

    mass = read("MU").astype(float) + read("MUB")
    thickness = np.diff(read("PH").astype(float) + read("PHB"), axis=1)
    check("mass_geometry", bool(np.all(mass > 0) and np.all(thickness > 0)),
          "positive total dry column mass and PH+PHB layer thickness",
          min_dry_mass_pa=float(np.min(mass)), min_layer_geopotential=float(np.min(thickness)))
    if {"C3F", "C4F", "C3H", "C4H", "P_TOP"} <= fields.keys():
        # Pristine WRF start_em HYPSOMETRIC_OPT=2 (:803-812). A source model
        # state need not be hydrostatic: disclose this EOS residual, do not
        # turn a diagnostic into an invented scientific tolerance.
        def coefficient(name):
            return read(name).reshape(1, -1, 1, 1)
        ptop = read("P_TOP").reshape(-1)
        if ptop.size != 1:
            check("schema", False, "P_TOP must be scalar")
            raise WrfinputConsistencyError(report)
        faces = coefficient("C3F") * mass[:, None] + coefficient("C4F") + ptop[0]
        mid = coefficient("C3H") * mass[:, None] + coefficient("C4H") + ptop[0]
        valid = bool(np.all(faces > 0) and np.all(mid > 0) and np.all(np.diff(faces, axis=1) < 0))
        check("mass_geometry", valid, "positive, vertically decreasing hybrid dry pressure")
        if valid:
            alpha = thickness / (mid * np.log(faces[:, :-1] / faces[:, 1:]))
            p = read("P").astype(float) + read("PB")
            theta = read("T").astype(float) + 300
            q = read("QVAPOR").astype(float)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                eos = 287. * theta * (p / 100000.) ** (2. / 7.) * (1 + (461.6 / 287.) * q) / p
                residual = np.abs(eos / alpha - 1)
            report["eos_diagnostic"] = {"gate": False, "geometry": "WRF start_em hypsometric_opt=2",
                "max_relative_residual": float(np.max(residual)) if np.all(np.isfinite(residual)) else None}

    for name in active & MOIST_FIELDS:
        if not inherited_transport:
            check("moisture_positivity", bool(np.all(read(name) >= 0)), f"{name} must be nonnegative")

    thermo = bool(active & (MOIST_FIELDS - {n for n in MOIST_FIELDS if n.startswith("QN")} | {"T", "P"}))
    balance = thermo or (checkpoint and "PH" in active)
    if transport_reference is not None:
        thermo = True
    if thermo:
        for name in set(fields) & MOIST_FIELDS:
            values = read(name)
            if inherited_transport:
                stats = _extrema(values, values < 0)
                stats["negative_cells"] = stats.pop("flagged_cells")
                stats["outer_ring_negative_cells"] = stats.pop("outer_ring_flagged_cells")
                report["inherited_source"]["moisture"][name] = stats
            else:
                check("moisture_positivity", bool(np.all(values >= 0)), f"{name} must be nonnegative")
    if thermo:
        # T in real.exe wrfinput is always dry theta perturbation, also when USE_THETA_M=1.
        q, theta, p = read("QVAPOR").astype(float), read("T").astype(float) + 300, read("P").astype(float) + read("PB")
        with np.errstate(invalid="ignore", over="ignore"):
            temperature = theta * (p / 100000.) ** (2. / 7.)
        valid_cells = (p > 0) & (theta > 0) & (temperature > 29.65)
        valid = np.all(valid_cells)
        if not inherited_transport:
            check("thermodynamic_state", bool(valid), "positive pressure/dry theta and valid saturation temperature")
        if valid or inherited_transport:
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                e_sat = 611.2 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
                e = p * q / (287. / 461.6 + q)
                rh = e / e_sat
            finite = bool(np.all(np.isfinite(rh)))
            if inherited_transport:
                stats = _extrema(rh, np.isfinite(rh) & (rh > 1.))
                stats["supersaturated_cells"] = stats.pop("flagged_cells")
                stats["outer_ring_supersaturated_cells"] = stats.pop("outer_ring_flagged_cells")
                stats["over_bound_cells"] = int(np.count_nonzero(np.isfinite(rh) & (rh > max_relative_humidity)))
                stats["invalid_temperature_or_pressure_cells"] = int(np.count_nonzero(~valid_cells))
                report["inherited_source"]["relative_humidity_over_liquid"] = stats
            else:
                check("finite", finite, "computed RH must be finite")
                check("supersaturation_bound", finite and bool(np.all(rh <= max_relative_humidity)),
                      "RH over liquid water within bound", max_rh=float(np.max(rh)) if finite else None)
    if balance or transport_reference is not None:
        if not inherited_transport and balance:
            check("hydrostatic_consistency", False,
                  "T/P/moist-mass edits require real.exe-consistent MU/PH rebalance; unavailable (explicit waiver required)")
        # These are not repaired or overwritten by this tool.
        for name in ("PH", "PHB", "MU", "MUB"):
            read(name)

    if active & (SURFACE_FIELDS | CHECKPOINT_LAND | {"LANDMASK", "XLAND"}):
        landmask, xland = read("LANDMASK"), read("XLAND")
        valid = np.all(np.isin(landmask, [0, 1])) and np.all(xland == np.where(landmask == 1, 1, 2))
        check("land_water_mask", bool(valid), "LANDMASK/XLAND must agree (land=1)")
        land = landmask == 1
        for name in active & ({"SST", "SMOIS", "TSLB", "LAI", "VEGFRA"} | CHECKPOINT_LAND):
            allowed = ~land if name == "SST" else land
            check("land_water_mask", not bool(np.any(masks[name] & ~np.broadcast_to(allowed, masks[name].shape))),
                  f"{name}: edits only on {'water' if name == 'SST' else 'land'}")

    if active & {"LU_INDEX", "IVGTYP", "ISLTYP", "SMOIS", "SH2O"} or effective_smcmax is not None:
        dataset = str(attributes.get("MMINLU", ""))
        lu, soil = read("LU_INDEX"), read("ISLTYP")
        ok = dataset == "MODIFIED_IGBP_MODIS_NOAH" and int(attributes.get("NUM_LAND_CAT", 0)) == 21
        ok = ok and bool(np.all((lu == np.floor(lu)) & (lu >= 1) & (lu <= 21)))
        ok = ok and bool(np.all((soil == np.floor(soil)) & (soil >= 1) & (soil <= 19)))
        if "IVGTYP" in fields:
            ok = ok and bool(np.array_equal(read("IVGTYP"), lu))
        check("modis_classes", ok, "MODIS LU/IVGTYP 1..21 (aliases equal); STAS ISLTYP 1..19")
        water = np.isin(lu, [17, 21])
        check("land_water_mask", bool(np.array_equal(water, read("LANDMASK") == 0)),
              "MODIS water/lake LU classes must agree with LANDMASK")
        if active & {"SMOIS", "ISLTYP", "SH2O", "LU_INDEX", "IVGTYP"} or effective_smcmax is not None:
            porosity, provenance = _porosity(soil_table)
            report["soil_table"] = provenance
            indices = np.clip(soil.astype(int), 1, len(porosity) - 1)
            moisture = read("SMOIS")
            seaice = read("SEAICE") if "SEAICE" in fields else None
            liquid = read("SH2O") if checkpoint and "SH2O" in fields else None
            cap = porosity[indices] if effective_smcmax is None else effective_smcmax
            if effective_smcmax is not None:
                report["soil_cap_source"] = "explicit effective SMCMAX"
            ok = validate_soil_porosity(moisture, cap, read("LANDMASK"), lu,
                seaice, surface_physics, sh2o=liquid)
            detail = ("land SMOIS in [0,STAS porosity], Noah-MP glacier cap=1" if effective_smcmax is None
                      else "land SMOIS in [0,effective SMCMAX], Noah-MP glacier cap=1")
            if liquid is not None:
                detail += "; land liquid soil water SH2O in [0,SMOIS]"
            check("soil_porosity", ok, detail)

    for name in active & {"SST", "TSK", "TSLB", "LAI", "VEGFRA", "EMISS", "ALBBCK", "ZNT"}:
        values = read(name)[masks[name]]
        lo, hi = ((0., 100.) if name == "VEGFRA" else (0., 1.) if name in {"EMISS", "ALBBCK"}
                  else (0., np.inf))
        ok = np.all((values >= lo) & (values <= hi))
        if name in {"SST", "TSK", "TSLB"}:
            ok = ok and np.all(values > 0)
        check("field_ranges", bool(ok), f"{name}: valid physical range; VEGFRA is percent")

    if checkpoint:
        for name in active & (CHECKPOINT_LAND - {"SH2O"} | {"QKE"}):
            values = read(name)[masks[name]]
            if name in {"TV", "TG", "TAH"}:
                ok = np.all(values > 0)
            elif name == "TSNO":
                snow = read("SNICE") + read("SNLIQ") if {"SNICE", "SNLIQ"} <= fields.keys() else np.ones(read(name).shape)
                ok = np.all(values >= 0) and np.all(read(name)[masks[name] & (snow > 0)] > 0)
            else:
                ok = np.all(values >= 0)
            check("field_ranges", bool(ok), f"{name}: nonnegative checkpoint store/variance or positive active temperature")

    check("finite", True, "edited fields and checked dependencies contain finite storage values")
    report["passed"] = not any(c["status"] == "FAIL" for c in report["checks"])
    if not report["passed"]:
        raise WrfinputConsistencyError(report)
    return report


class _NetCDFFields(Mapping):
    """Lazy host array view; retain one variable at a time, optionally one frame."""
    def __init__(self, dataset, frame=None):
        self.dataset, self.frame = dataset, frame

    def __iter__(self):
        return iter(self.dataset.variables)

    def __len__(self):
        return len(self.dataset.variables)

    def __contains__(self, name):
        return name in self.dataset.variables

    def array_shape(self, name):
        var = self.dataset[name]
        return tuple(1 if self.frame is not None and dim == "Time" else size
                     for dim, size in zip(var.dimensions, var.shape))

    def __getitem__(self, name):
        if name not in self:
            raise KeyError(name)
        var = self.dataset[name]
        key = [slice(None)] * var.ndim
        if self.frame is not None and "Time" in var.dimensions:
            key[var.dimensions.index("Time")] = slice(self.frame, self.frame + 1)
        return np.asarray(var[tuple(key)] if key else var[...])


def _file_view(ds, frame=None):
    ds.set_auto_maskandscale(False)
    ds.set_auto_chartostring(False)
    dimensions = {name: len(dim) for name, dim in ds.dimensions.items()}
    if frame is not None:
        count = dimensions.get("Time", 0)
        if type(frame) is not int or not -count <= frame < count:
            raise ValueError("transport_frame outside source Time records")
        frame %= count
        dimensions["Time"] = 1
    return _NetCDFFields(ds, frame), {
        "dimensions": dimensions,
        "variable_dimensions": {name: var.dimensions for name, var in ds.variables.items()},
        "attributes": {name: ds.getncattr(name) for name in ds.ncattrs()},
        "variable_attributes": {name: {"_FillValue": var.getncattr("_FillValue")}
                                for name, var in ds.variables.items() if "_FillValue" in var.ncattrs()},
        "source_path": str(Path(ds.filepath()).resolve()), "source_frame": frame,
    }


def validate_wrfinput(path, *, changed_fields=(), waivers=None, surface_physics=4,
                      max_relative_humidity=1.0, soil_table=None,
                      transport_reference=None, transport_frame=-1, effective_smcmax=None):
    """Read-only, lazy file wrapper around :func:`validate_fields`."""
    from contextlib import ExitStack
    with ExitStack() as stack:
        fields, meta = _file_view(stack.enter_context(Dataset(path)))
        reference = None
        if transport_reference is not None:
            reference = _file_view(stack.enter_context(Dataset(transport_reference)), transport_frame)
        return validate_fields(fields, meta, changed_fields=changed_fields, waivers=waivers,
            surface_physics=surface_physics, max_relative_humidity=max_relative_humidity,
            soil_table=soil_table, transport_reference=reference,
            **({"effective_smcmax": effective_smcmax} if effective_smcmax is not None else {}))
