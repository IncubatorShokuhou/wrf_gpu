"""Publish a CPU-admitted d01 cycling case without changing its source inputs.

Whole forcing records retain their original storage values, including boundary
tendencies and time metadata. A source with nests may supply its root domain;
the published case always has max_dom=1. Nested analyses are refused.
"""
from __future__ import annotations

import contextlib
import ctypes
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np

from gpuwrf.io.namelist_parser import parse_namelist
from gpuwrf.io.netcdf_lock import Dataset

_FORMAT = "%Y-%m-%d_%H:%M:%S"
_TOKEN = re.compile(r"![^\n]*|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|[&/=,*():]|[^\s&/=,*():!'\"]+")


def _first(group, key, default=None):
    value = group.get(key, default)
    return value[0] if isinstance(value, (list, tuple)) else value


def _date(value):
    if isinstance(value, str):
        value = datetime.strptime(value, _FORMAT)
    if not isinstance(value, datetime):
        raise ValueError("t_a must be a datetime or a WRF Times string")
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    if value.microsecond:
        raise ValueError("t_a must be an exact second")
    return value


def _clock(tc, prefix):
    return datetime(*[int(_first(tc, f"{prefix}_{part}", default)) for part, default in
                      (("year", None), ("month", None), ("day", None),
                       ("hour", 0), ("minute", 0), ("second", 0))])


def _identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def _edit_namelist(text, updates):
    """Replace only selected value spans; retain comments and unrelated bytes."""
    tokens = [m for m in _TOKEN.finditer(text) if not m.group().startswith("!")]
    edits, found, endings = [], set(), {}
    group = None
    i = 0
    while i < len(tokens):
        token = tokens[i].group().lower()
        if token == "&":
            name = tokens[i + 1].group().lower()
            if name == "end":
                endings[group] = tokens[i].start()
                group = None
            else:
                group = name
            i += 2
        elif token == "/":
            endings[group] = tokens[i].start()
            group = None
            i += 1
        elif i + 1 < len(tokens) and tokens[i + 1].group() == "=":
            j = i + 2
            while j < len(tokens) and tokens[j].group() not in ("/", "&"):
                if j + 1 < len(tokens) and tokens[j + 1].group() == "=":
                    break
                j += 1
            key = (group, token)
            if key in updates:
                # Individual token edits preserve even comments inside a list.
                edits.append((tokens[i + 1].end(), tokens[i + 1].end(), f" {updates[key]},"))
                edits.extend((t.start(), t.end(), "") for t in tokens[i + 2:j])
                found.add(key)
            i = j
        else:
            i += 1
    for group in {g for g, _ in updates}:
        missing = [(key, value) for (g, key), value in updates.items() if g == group and (g, key) not in found]
        if missing:
            if group not in endings:
                raise ValueError(f"namelist missing &{group}")
            edits.append((endings[group], endings[group], "".join(f" {k} = {v},\n" for k, v in missing)))
    for start, end, value in sorted(edits, reverse=True):
        text = text[:start] + value + text[end:]
    return text


def _times(ds, name="Times"):
    if name not in ds.variables:
        raise ValueError(f"{ds.filepath()}: missing {name}")
    var = ds[name]
    if var.dimensions != ("Time", "DateStrLen"):
        raise ValueError(f"{name}: expected (Time, DateStrLen)")
    return [_date(b"".join(np.asarray(row, dtype="S1").tolist()).decode("ascii").strip("\0 "))
            for row in var[:]]


def _forcing_plan(path, start, end, interval, *, boundary):
    with Dataset(path) as ds:
        times = _times(ds)
        if not times or any(b - a != timedelta(seconds=interval) for a, b in zip(times, times[1:])):
            raise ValueError(f"{path.name}: Times must increase at the declared interval")
        if start not in times:
            raise ValueError(f"{path.name}: no record at cycle start")
        # Keep an endpoint/bracketing record, as required by ens.workflows._case.
        after = next((i for i, stamp in enumerate(times) if stamp >= end), None)
        if after is None:
            raise ValueError(f"{path.name}: coverage too short for requested forecast")
        first = times.index(start)
        metadata = []
        if boundary:
            for name in ds.variables:
                lowered = name.lower()
                if "thisbdytime" in lowered or "nextbdytime" in lowered:
                    expected = times if "thisbdytime" in lowered else [t + timedelta(seconds=interval) for t in times]
                    if _times(ds, name) != expected:
                        raise ValueError(f"{path.name}: {name} disagrees with Times/interval")
                    metadata.append(name)
        attrs = {name: str(ds.getncattr(name)) for name in ("START_DATE", "SIMULATION_START_DATE") if name in ds.ncattrs()}
    return {"records": list(range(first, after + 1)), "source_records": len(times),
            "first_time": start.strftime(_FORMAT), "last_time": times[after].strftime(_FORMAT),
            "interval_seconds": interval, "checked_time_variables": ["Times", *metadata],
            "source_clock_attributes": attrs}


def _slice_netcdf(source, output, records, start):
    """Copy raw records one at a time: no packing, rounding, or whole-file array."""
    with Dataset(source) as src, Dataset(output, "w", format=src.data_model) as dst:
        src.set_auto_maskandscale(False)
        src.set_auto_chartostring(False)
        attrs = {name: src.getncattr(name) for name in src.ncattrs()}
        changed = {}
        for name in ("START_DATE", "SIMULATION_START_DATE"):
            if name in attrs:
                changed[name] = {"before": str(attrs[name]), "after": start.strftime(_FORMAT)}
                attrs[name] = start.strftime(_FORMAT)
        # Other global attrs and every variable value stay exactly as supplied.
        dst.setncatts(attrs)
        for name, dim in src.dimensions.items():
            size = len(records) if name == "Time" else len(dim)
            dst.createDimension(name, None if dim.isunlimited() else size)
        digests = {}
        for name, var in src.variables.items():
            options = {}
            if "_FillValue" in var.ncattrs():
                options["fill_value"] = var.getncattr("_FillValue")
            filters = var.filters() or {}
            if filters.get("zlib"):
                options.update(zlib=True, complevel=filters["complevel"], shuffle=filters["shuffle"])
            target = dst.createVariable(name, var.datatype, var.dimensions, **options)
            target.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"})
            target.set_auto_maskandscale(False)
            target.set_auto_chartostring(False)
            digest = hashlib.sha256()
            if "Time" in var.dimensions:
                axis = var.dimensions.index("Time")
                for out_index, in_index in enumerate(records):
                    key = [slice(None)] * var.ndim
                    key[axis] = in_index
                    data = var[tuple(key)]
                    digest.update(np.ascontiguousarray(data).tobytes())
                    key[axis] = out_index
                    target[tuple(key)] = data
                    if np.ascontiguousarray(target[tuple(key)]).tobytes() != np.ascontiguousarray(data).tobytes():
                        raise ValueError(f"{name}: storage values changed while slicing")
            else:
                data = var[...]
                target[...] = data
                digest.update(np.ascontiguousarray(data).tobytes())
                if np.ascontiguousarray(target[...]).tobytes() != np.ascontiguousarray(data).tobytes():
                    raise ValueError(f"{name}: storage values changed while copying")
            digests[name] = digest.hexdigest()
    return {"changed_global_attributes": changed, "raw_kept_values_sha256": digests}


_COLD_EXTRA = ("FVEG", "VEGFRA", "LAI", "TMN")


def _cold_land(analysis_ds, source_wrfinput, start):
    """land='cold' (a1-io): run-start IC member. No GPUWRF_LAND_CYCLE marker (the port's cold Noah-MP init
    must build the canopy/snow state real.exe leaves at 0) and every land/Noah-MP variable bitwise equal to
    the source case wrfinput_d01, which must itself be valid at t_a."""
    from gpuwrf.cycling.core import CORE, MARKER, MP_OPTIONAL, MP_REQUIRED

    if MARKER in analysis_ds.ncattrs():
        raise ValueError("land='cold' analysis wrfinput must not carry GPUWRF_LAND_CYCLE provenance")
    with Dataset(source_wrfinput) as src:
        if _times(src) != [start]:
            raise ValueError("land='cold' requires the source wrfinput_d01 to be valid at t_a (run start)")
        names = [n for n in dict.fromkeys(CORE + MP_REQUIRED + MP_OPTIONAL + _COLD_EXTRA)
                 if n in src.variables or n in analysis_ds.variables]
        for name in names:
            if name not in src.variables or name not in analysis_ds.variables:
                raise ValueError(f"land='cold': {name} present in only one of source/analysis wrfinput")
            a, b = analysis_ds[name], src[name]
            a.set_auto_maskandscale(False)
            b.set_auto_maskandscale(False)
            va, vb = np.ascontiguousarray(a[:]), np.ascontiguousarray(b[:])
            if a.dimensions != b.dimensions or va.dtype != vb.dtype or va.tobytes() != vb.tobytes():
                raise ValueError(f"land='cold': {name} differs from the source wrfinput_d01 (land must be unchanged)")
    return {"status": "PASS", "fields": len(names), "names": names, "marker": "absent"}


def _admit(case, hours, start, land="cycled", source_wrfinput=None):
    """Call the real port preflights; never load a model or a GPU backend."""
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("cycle case admission requires JAX_PLATFORMS=cpu")
    from gpuwrf.cli import main
    from gpuwrf.cycling.core import MARKER, LAND_LEAVES, restore_land_state
    from gpuwrf.ens.workflows import _case
    from gpuwrf.io.lower_boundary import load_lower_boundary
    from gpuwrf.io.wrfbdy_coverage import require_wrfbdy_coverage

    config = parse_namelist(case / "namelist.input")
    coverage = require_wrfbdy_coverage(case, hours)
    if coverage is None:
        raise ValueError("wrfbdy_d01 is required")
    spec = _case(case, case / "namelist.input", hours)
    with Dataset(case / "wrfinput_d01") as ds:
        if _times(ds) != [start]:
            raise ValueError("analysis wrfinput Times must equal t_a")
        shape = (len(ds.dimensions["south_north"]), len(ds.dimensions["west_east"]))
        if land == "cold":
            land_check = _cold_land(ds, source_wrfinput, start)
        else:
            if MARKER not in ds.ncattrs():
                raise ValueError("analysis wrfinput requires GPUWRF_LAND_CYCLE provenance")
            sink = SimpleNamespace(**{attr: np.zeros(ds[name].shape[1:], dtype=ds[name].dtype)
                                      for name, attr in LAND_LEAVES.items() if name in ds.variables})
            marker = restore_land_state(ds, sink)
            land_check = {"status": "PASS", "valid_time": marker["valid_time"], "fields": len(marker["fields"])}
    dt = float(Fraction(str(_first(config["domains"], "time_step"))) +
               Fraction(str(_first(config["domains"], "time_step_fract_num", 0))) /
               Fraction(str(_first(config["domains"], "time_step_fract_den", 1))))
    lower = load_lower_boundary(case, config, "d01", run_start=start, dt_s=dt, shape=shape)
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        rc = main(["run", "--input-dir", str(case), "--output-dir", str(case / "forecast"),
                   "--scratch-dir", str(case / "scratch"), "--domains-from-namelist",
                   "--hours", str(hours), "--dry-run"])
    if rc != 0:
        raise ValueError(f"CLI --dry-run refused case: {stderr.getvalue()}")
    return {"wrfbdy_coverage": coverage, "ensemble_case": {"status": "PASS", "issue": spec["issue"].isoformat()},
            "lower_boundary": {"status": "PASS", "records": 0 if lower is None else len(lower.activation_steps)},
            ("land_cold" if land == "cold" else "land_cycle_marker"): land_check,
            "cli_dry_run": {"status": "PASS", "returncode": rc, "plan": json.loads(stdout.getvalue()),
                            "stderr": stderr.getvalue()}}


def _publish(temporary, output):
    # Linux renameat2 is atomic and refuses even a concurrently created empty dir.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(temporary), -100, os.fsencode(output), 1):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(output))


def write_cycle_case(source_case_dir, analysis_wrfinputs, t_a, hours, out_dir, *, land="cycled"):
    """Write a new d01 forecast case and return its SHA256/admission manifest.

    ``analysis_wrfinputs`` must be ``{'d01': path}``; the input must already be
    retimed and carry validated ``gpuwrf cycle`` land provenance. The exact same
    API writes the zero-update control from an identity member-frame wrfinput.
    ``land='cold'`` (a1-io, perturbed-IC members at the source run start): no land
    marker, land/Noah-MP variables bitwise equal to the source wrfinput_d01.
    Run with ``JAX_PLATFORMS=cpu``. No model integration occurs.
    """
    if land not in ("cycled", "cold"):
        raise ValueError("land must be 'cycled' or 'cold'")
    source = Path(source_case_dir).resolve(strict=True)
    requested_output = Path(out_dir).absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise ValueError("output must be a new directory; refusing overwrite")
    output = requested_output.parent.resolve() / requested_output.name
    start = _date(t_a)
    if isinstance(hours, bool) or not isinstance(hours, int) or hours <= 0 or hours % 3:
        raise ValueError("hours must be a positive integer multiple of 3")
    if set(analysis_wrfinputs) != {"d01"}:
        raise ValueError("only d01/max_dom=1 cycle cases are supported")
    if source == output or source in output.parents:
        raise ValueError("output must be outside the read-only source case")
    end = start + timedelta(hours=hours)
    namelist = source / "namelist.input"
    config = parse_namelist(namelist)
    tc = config["time_control"]
    origin = _clock(tc, "start")
    interval = float(_first(tc, "interval_seconds", 21600))
    aux_interval = sum(float(_first(tc, f"auxinput4_interval_{unit}", 0)) * factor
                       for unit, factor in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)))
    if aux_interval == 0:
        aux_interval = float(_first(tc, "auxinput4_interval", 0)) * 60
    enabled = int(_first(config.get("physics", {}), "sst_update", 0))
    elapsed = (start - origin).total_seconds()
    if not np.isfinite(interval) or interval <= 0 or elapsed < 0 or elapsed % interval:
        raise ValueError("t_a must align with the source wrfbdy interval")
    if enabled and (not np.isfinite(aux_interval) or aux_interval <= 0 or elapsed % aux_interval):
        raise ValueError("t_a must align with the source auxinput4 interval")
    updates = {("domains", "max_dom"): 1}
    for prefix, stamp in (("start", start), ("end", end)):
        for part in ("year", "month", "day", "hour", "minute", "second"):
            updates[("time_control", f"{prefix}_{part}")] = getattr(stamp, part)
    updates.update({("time_control", f"run_{part}"): value for part, value in
                    (("days", 0), ("hours", hours), ("minutes", 0), ("seconds", 0))})
    analysis = Path(analysis_wrfinputs["d01"]).resolve(strict=True)
    with Dataset(analysis) as ds:
        if _times(ds) != [start]:
            raise ValueError("analysis wrfinput Times must equal t_a")
        for name in ("START_DATE", "SIMULATION_START_DATE"):
            if name in ds.ncattrs() and _date(str(ds.getncattr(name))) != start:
                raise ValueError(f"analysis wrfinput {name} must equal t_a")
    plans = {"wrfbdy_d01": _forcing_plan(source / "wrfbdy_d01", start, end, interval, boundary=True)}
    if enabled:
        template = str(_first(tc, "auxinput4_inname", "wrflowinp_d<domain>"))
        auxname = template.replace("<domain>", "01")
        if Path(auxname).name != auxname or "<" in auxname:
            raise ValueError("auxinput4 must name a file inside the source case")
        plans[auxname] = _forcing_plan(source / auxname, start, end, aux_interval, boundary=False)
    # WRF runtime tables include extensionless RRTM_DATA and CLWRF data files.
    # Only static table/data inputs are shared: histories/restarts are never linked.
    static = [p for p in sorted(source.iterdir()) if p.is_file() and
              (p.suffix.lower() in (".tbl", ".dat", ".formatted", ".asc") or
               p.name.startswith(("RRTM_DATA", "RRTMG_", "CLWRF", "ETAMPNEW", "CAM_ABS_DATA", "CAM_AEROPT_DATA")))]
    sources = [namelist, analysis, *(source / name for name in plans), *static]
    if land == "cold":  # the land-equality reference is an admission input too
        sources.append(source / "wrfinput_d01")
    identities = {str(p): _identity(p) for p in sources}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        (temporary / "namelist.input").write_text(_edit_namelist(namelist.read_text(), updates))
        written = parse_namelist(temporary / "namelist.input")
        expected = json.loads(json.dumps(config))
        for (group, key), value in updates.items():
            expected[group][key] = value
        if written != expected:
            raise ValueError("namelist round trip changed an unrelated setting")
        shutil.copyfile(analysis, temporary / "wrfinput_d01")
        for name, plan in plans.items():
            plan.update(_slice_netcdf(source / name, temporary / name, plan["records"], start))
        for path in static:
            (temporary / path.name).symlink_to(path.resolve(strict=True))
        admission = _admit(temporary, hours, start, land=land, source_wrfinput=source / "wrfinput_d01")
        outputs = {p.name: {**_identity(p), "path": str(output / p.name),
                            "storage": "symlink" if p.is_symlink() else "file"} for p in sorted(temporary.iterdir()) if p.is_file()}
        if outputs["wrfinput_d01"]["sha256"] != identities[str(analysis)]["sha256"]:
            raise ValueError("analysis input changed while copying")
        # A second streaming hash also detects concurrent source modifications.
        if any(_identity(p)["sha256"] != identities[str(p)]["sha256"] for p in sources):
            raise ValueError("source input changed while constructing cycle case")
        # Admission ran in the staging dir; publish paths in its recorded plan.
        admission = json.loads(json.dumps(admission).replace(str(temporary), str(output)))
        manifest = {"schema": 1, "source_case": str(source), "output_case": str(output), "land": land,
                    "source_max_dom": _first(config.get("domains", {}), "max_dom", 1), "max_dom": 1,
                    "start": start.strftime(_FORMAT), "end": end.strftime(_FORMAT), "hours": hours,
                    "phase_origin": origin.strftime(_FORMAT), "inputs": identities, "outputs": outputs,
                    "slicing": plans, "admission": admission,
                    "identity_control": "pass the zero-update member-frame wrfinput through this same writer"}
        (temporary / "cycle_case_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _publish(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
