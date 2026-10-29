"""Read-only, column-sized history diagnostics for optimizer wave reports.

This consumer never enables a namelist switch, imports a model kernel, or
reconstructs unrecorded physics calls. Missing histories remain UNAVAILABLE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from gpuwrf.io.mynn_budget_history import METADATA
from gpuwrf.io.netcdf_lock import Dataset

SCHEMA = "gpuwrf.optimize.profiles.v1"
RAD_FIELDS = ("RTHRATLW", "RTHRATSW", "RTHRATLWC", "RTHRATSWC",
              "QC_RAD", "QI_RAD", "QS_RAD", "QG_RAD")
LATER_TAPS = {
    "radiation_call_time": ("History stores held rates, not their producer call time.", "1–2 days"),
    "mynn_call_time": ("History stores held rates, not their producer call time.", "1–2 days"),
    "water_budget_residual": ("History lacks interval-integrated transport, boundary, microphysics, "
                              "surface and nesting sources on the same operands.", "3–5 days"),
    "energy_budget_residual": ("History lacks interval-integrated transport, boundary, phase-change, "
                               "surface and nesting sources on the same operands.", "3–5 days"),
}


def unavailable(reason, units=None):
    return {"status": "UNAVAILABLE", "reason": reason, "units": units, "value": None}


def available(value, units):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite derived diagnostic")
    return {"status": "AVAILABLE", "units": units, "value": value.tolist()}


def _points(points):
    points = [dict(p) for p in points]
    if not points or len(points) > 32:
        raise ValueError("select 1–32 explicit profile points")
    names = set()
    for point in points:
        if set(point) != {"id", "region", "domain", "j", "i"}:
            raise ValueError("points require exactly id, region, domain, j, i")
        if not isinstance(point["id"], str) or not point["id"].strip() or point["id"] in names:
            raise ValueError("point ids must be unique nonempty strings")
        names.add(point["id"])
        if point["region"] not in ("north_windward", "sea"):
            raise ValueError("point region must be north_windward or sea")
        domain = point["domain"]
        if not isinstance(domain, str) or len(domain) != 3 or domain[0] != "d" or not domain[1:].isdigit() or int(domain[1:]) < 1:
            raise ValueError("point domain must be d01 … d99")
        if any(type(point[k]) is not int or point[k] < 0 for k in ("j", "i")):
            raise ValueError("point j/i must be nonnegative integer mass-grid indices")
    return points


def _stat(path):
    s = path.stat()
    return {"path": str(path), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "inode": s.st_ino}


def _units(name):
    if name in METADATA:
        return METADATA[name][1]
    if name.startswith("RTHRAT"):
        return "K s-1"
    if name == "QFX":
        return "kg m-2 s-1"
    if name.startswith("Q"):
        return "kg kg-1"
    return {"T": "K", "P": "Pa", "PB": "Pa", "PH": "m2 s-2", "PHB": "m2 s-2",
            "HGT": "m", "XLAT": "degree_north", "XLONG": "degree_east", "LANDMASK": "1",
            "HFX": "W m-2", "LH": "W m-2", "GRDFLX": "W m-2", "QFX": "kg m-2 s-1"}[name]


def _normal_unit(unit):
    return unit.strip().replace("**", "").replace("^", "").replace(" ", "").lower()


class _Column:
    def __init__(self, ds, frame, point):
        self.ds, self.frame, self.point = ds, frame, point
        self.nz = len(ds.dimensions["bottom_top"])
        self.cache = {}
        self.digest = hashlib.sha256()
        self.bytes_read = 0
        for key, dim in (("j", "south_north"), ("i", "west_east")):
            if point[key] >= len(ds.dimensions[dim]):
                raise ValueError(f"{point['id']}: {key} outside {dim}")

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        if name not in self.ds.variables:
            self.cache[name] = None
            return None
        var = self.ds.variables[name]
        vertical = "bottom_top_stag" if name in ("PH", "PHB") or METADATA.get(name, (None, None, None))[2] == "Z" else "bottom_top"
        surface = name in ("HGT", "XLAT", "XLONG", "LANDMASK", "HFX", "LH", "GRDFLX", "QFX")
        expected = {"south_north", "west_east"} | (set() if surface else {vertical})
        dims = set(var.dimensions)
        if dims - {"Time"} != expected or len(dims) != len(var.dimensions):
            raise ValueError(f"{name}: wrong history dimensions {var.dimensions}")
        keys = {"Time": self.frame, "south_north": self.point["j"], "west_east": self.point["i"], vertical: slice(None)}
        a = np.asanyarray(var[tuple(keys[d] for d in var.dimensions)])
        if np.ma.isMaskedArray(a) and np.ma.getmaskarray(a).any():
            raise ValueError(f"{name}: masked/fill history")
        a = np.asarray(a)
        if not np.isfinite(a).all():
            raise ValueError(f"{name}: nonfinite history")
        expected_units = {_normal_unit(_units(name))}
        if name == "XLAT":
            expected_units.update(("degrees_north", "degreenorth"))
        if name == "XLONG":
            expected_units.update(("degrees_east", "degreeeast"))
        if name in METADATA and METADATA[name][1] == "-":
            expected_units.add("1")
        if name == "LANDMASK":
            expected_units.update(("", "-"))
        if _normal_unit(str(getattr(var, "units", ""))) not in expected_units:
            raise ValueError(f"{name}: missing/unsupported history units")
        self.bytes_read += a.nbytes
        self.digest.update(name.encode() + a.dtype.str.encode() + str(a.shape).encode() + a.tobytes())
        if not surface and a.shape != (self.nz + (vertical == "bottom_top_stag"),):
            raise ValueError(f"{name}: wrong profile length")
        # Registry Z budget outputs already contain the destaggered mass prefix.
        # Averaging them again would corrupt the original terms.
        if name in METADATA and vertical == "bottom_top_stag":
            if a[-1] != 0:
                raise ValueError(f"{name}: nonzero unused interface top")
            a = a[:-1]
        if name in ("qSHEAR", "qBUOY", "qDISS", "qWT", "Dtke") and a[-1] != 0:
            raise ValueError(f"{name}: nonzero unused mass top")
        self.cache[name] = a.astype(np.float64)
        return self.cache[name]

    def derive(self, names, units, function):
        values = [self.get(name) for name in names]
        missing = [name for name, value in zip(names, values) if value is None]
        if missing:
            return unavailable("history missing " + ", ".join(missing), units)
        return available(function(*values), units)


def _geometry(t, p, pb, ph, phb, hgt, qv):
    theta, pressure = t + 300., p + pb
    zi = (ph + phb) / 9.81 - hgt
    z = .5 * (zi[1:] + zi[:-1])
    if np.any(theta <= 0) or np.any(pressure <= 0) or np.any(qv < 0) or np.any(z <= 0) or np.any(np.diff(zi) <= 0):
        raise ValueError("nonphysical thermodynamic/height history")
    exner = (pressure / 100000.) ** (287. / 1004.5)
    # WRF q is per kg dry air. Use dry-air density, not moist density.
    rho_d = pressure / (287. * theta * exner * (1. + qv / (287. / 461.6)))
    return theta, z, np.diff(zi), rho_d, exner


def _sample(column):
    c = column
    result = {}
    for name in ("T", "QVAPOR", "QCLOUD", "QICE", "QC_BL", "QI_BL", *RAD_FIELDS, *METADATA,
                 "HFX", "LH", "QFX", "GRDFLX", "XLAT", "XLONG", "LANDMASK"):
        result[name] = c.derive((name,), _units(name), lambda a: a)
    result["theta"] = c.derive(("T",), "K", lambda a: a + 300.)
    geometry_names = ("T", "P", "PB", "PH", "PHB", "HGT", "QVAPOR")
    geometry_values = [c.get(n) for n in geometry_names]
    if any(v is None for v in geometry_values):
        why = "geometry missing " + ", ".join(n for n, v in zip(geometry_names, geometry_values) if v is None)
        for name, units in (("height_agl", "m"), ("inversion_height_agl", "m"),
                            ("CWP", "g m-2"), ("LWP", "g m-2"), ("cloud_top_agl", "m"),
                            ("cloud_top_lw_cooling", "K s-1"), ("pbl_water_tendency", "kg m-2 s-1"),
                            ("pbl_radiation_sensible_energy_tendency", "W m-2")):
            result[name] = unavailable(why, units)
    else:
        theta, z, dz, rho_d, exner = _geometry(*geometry_values)
        mass = rho_d * dz
        result["height_agl"] = available(z, "m")
        middle = .5 * (z[1:] + z[:-1])
        gradient = np.diff(theta) / np.diff(z)
        candidates = np.flatnonzero((middle >= 100.) & (middle <= 4000.) & (gradient > 0))
        result["inversion_height_agl"] = (available(middle[candidates[np.argmax(gradient[candidates])]], "m")
                                           if candidates.size else unavailable("no positive theta gradient in 100–4000 m AGL", "m"))

        def nonnegative(*values):
            if any(np.any(v < 0) for v in values):
                raise ValueError("negative condensate in cloud diagnostic")
            return sum(values)

        result["LWP"] = c.derive(("QCLOUD",), "g m-2", lambda q: 1000. * np.sum(mass * nonnegative(q)))
        result["CWP"] = c.derive(("QCLOUD", "QICE"), "g m-2", lambda q, i: 1000. * np.sum(mass * nonnegative(q, i)))
        qc, qi = c.get("QCLOUD"), c.get("QICE")
        cloudy = np.flatnonzero(nonnegative(qc, qi) >= 1.e-6) if qc is not None and qi is not None else []
        why = "history missing QCLOUD/QICE" if qc is None or qi is None else "no prognostic cloud >= 1e-6 kg kg-1"
        result["cloud_top_agl"] = available(z[cloudy[-1]], "m") if len(cloudy) else unavailable(why, "m")
        result["cloud_top_lw_cooling"] = (c.derive(("RTHRATLW",), "K s-1", lambda r: -r[cloudy[-1]] * exner[cloudy[-1]])
                                            if len(cloudy) else unavailable(why, "K s-1"))
        result["pbl_water_tendency"] = c.derive(("RQVBLTEN", "RQCBLTEN", "RQIBLTEN"), "kg m-2 s-1",
                                                lambda *r: np.sum(mass * sum(r)))
        result["pbl_radiation_sensible_energy_tendency"] = c.derive(("RTHBLTEN", "RTHRATLW", "RTHRATSW"), "W m-2",
                                                                    lambda *r: np.sum(mass * 1004.5 * exner * sum(r)))
    result["tke_budget_residual"] = c.derive(("Dtke", "qSHEAR", "qBUOY", "qWT", "qDISS"), "m2 s-3",
                                            lambda d, s, b, w, loss: (d - (s + b + w - loss))[:-1])
    residual = result["tke_budget_residual"]
    if residual["status"] == "AVAILABLE":
        a = np.asarray(residual["value"])
        if not a.size:
            raise ValueError("TKE residual needs at least two mass levels")
        residual.update(max_abs=float(np.max(np.abs(a))), rms=float(np.sqrt(np.mean(a*a))), levels=int(a.size))
    for name, (reason, _) in LATER_TAPS.items():
        result[name] = unavailable(reason)
    return result


def write_profile_report(files, points, output):
    """Write a new JSON report from explicitly selected, completed history files.

    Only point columns are read. Source stat identities and sampled-byte digests
    bind this extraction; these are not full-file SHA256 identities or physics
    qualification. ``output`` is created exclusively after every input passes QC.
    """
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs
    from gpuwrf.verify.wrfout import _times

    started = time.monotonic()
    points = _points(points)
    paths = [Path(p).resolve() for p in files]
    output = Path(output).resolve()
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("provide nonempty distinct history files")
    refuse_protected_inputs("optimizer profiles", paths, outputs=[output])
    if output.exists() or output in paths:
        raise ValueError("profile output must be a new destination")
    sources = [_stat(p) for p in paths]
    rows, seen, issues, grids = [], set(), set(), {}
    for path in paths:
        with Dataset(path, "r") as ds:
            domain_id = getattr(ds, "GRID_ID", None)
            issue = getattr(ds, "SIMULATION_START_DATE", getattr(ds, "START_DATE", None))
            if domain_id is None or issue is None:
                raise ValueError("history requires GRID_ID and SIMULATION_START_DATE/START_DATE")
            domain = f"d{int(domain_id):02d}"
            issues.add(issue)
            if len(issues) != 1:
                raise ValueError("profile report requires one forecast issue")
            for frame, valid in enumerate(_times(ds)):
                key = (domain, valid.isoformat())
                if key in seen:
                    raise ValueError("duplicate domain/history valid time")
                seen.add(key)
                for point in points:
                    if point["domain"] != domain:
                        continue
                    c = _Column(ds, frame, point)
                    metrics = _sample(c)
                    grid = [metrics[n]["value"] for n in ("XLAT", "XLONG", "LANDMASK")]
                    if point["id"] in grids and grids[point["id"]] != grid:
                        raise ValueError("profile point grid changed across history")
                    grids[point["id"]] = grid
                    if grid[2] is not None and grid[2] != (0 if point["region"] == "sea" else 1):
                        raise ValueError("profile point LANDMASK contradicts its selected region")
                    rows.append({"point": point["id"], "domain": domain, "valid_time": valid.isoformat(),
                                 "issue_time": issue, "metrics": metrics, "sample_sha256": c.digest.hexdigest(),
                                 "sample_bytes": c.bytes_read})
    if [_stat(p) for p in paths] != sources:
        raise ValueError("history changed during profile extraction")
    rows.sort(key=lambda r: (r["valid_time"], r["domain"], r["point"]))
    missing = [p["id"] for p in points if p["id"] not in grids]
    missing_terms = sorted({k for r in rows for k, v in r["metrics"].items() if v["status"] == "UNAVAILABLE"})
    report = {"schema": SCHEMA, "mode": "READ_ONLY_HISTORY", "status": "UNAVAILABLE" if missing else "COMPLETE_WITH_UNAVAILABLE_TERMS",
              "sources": sources, "points": points, "missing_points": missing, "unavailable_terms": missing_terms,
              "rows": rows, "extract_wall_s": time.monotonic() - started,
              "sample_bytes": sum(r["sample_bytes"] for r in rows),
              "qualification": {"model_program": "UNCHANGED: consumer runs after history publication; no model imports or switches",
                                "rad_diag": "Existing CPU-WRF column oracles + 4536 shared GPU history byte comparisons; hidden carry identity OPEN",
                                "mynn_budget": "Existing CPU component oracle only; in-loop coupled GPU ON/OFF identity FAIL (E242)",
                                "residual": "local MYNN closure diagnostic, not zero-closure gate or total forecast conservation"}}
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return report


def _finite_report_number(token):
    """Reject nonstandard JSON constants and exponent overflow before reduction."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("nonfinite number in optimizer profile report")
    return value


def wave_report_entry(profile_report):
    """Small, source-bound entry for an optimizer wave's diagnostic section."""
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs

    path = Path(profile_report).resolve()
    refuse_protected_inputs("optimizer profile summary", [path])
    raw = path.read_bytes()
    report = json.loads(raw, parse_float=_finite_report_number, parse_constant=_finite_report_number)
    if report.get("schema") != SCHEMA or report.get("mode") != "READ_ONLY_HISTORY":
        raise ValueError("invalid optimizer profile report")
    residuals = [r["metrics"]["tke_budget_residual"]["max_abs"] for r in report["rows"]
                 if r["metrics"]["tke_budget_residual"]["status"] == "AVAILABLE"]
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "status": report["status"],
            "profile_rows": len(report["rows"]), "unavailable_terms": report["unavailable_terms"],
            "missing_points": report["missing_points"], "extract_wall_s": report["extract_wall_s"],
            "sample_bytes": report["sample_bytes"], "max_abs_tke_residual_m2_s3": max(residuals) if residuals else None,
            "prognostic_effect": "NONE: read-only consumer", "qualification": report["qualification"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--points", type=Path, required=True, help="JSON list of id/region/domain/j/i")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from gpuwrf.assimilation.letkf.member_driver import refuse_protected_inputs

    refuse_protected_inputs("optimizer profile points", [args.points])
    write_profile_report(args.files, json.loads(args.points.read_text()), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
