"""Bounded public GFS retrieval and GRIB metadata checks before WPS.

ECMWF forcing is inventoried using the same WPS-required field codes. The
presence of a Vtable alone never makes missing soil/surface fields available.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MAX_DOWNLOAD_BYTES = 2_000_000_000


def ecmwf_report(plan: dict, source: str) -> dict:
    """Check the live open-data index, without downloading incomplete forcing.

    The 50r1 naming convention uses oper at all four cycles. Raw indexes and
    digests identify this check; availability is not inferred from a Vtable.
    """
    model = {"ifs": "ifs", "aifs": "aifs-single"}[source]
    start = datetime.strptime(plan["start"], "%Y-%m-%d_%H:%M:%S")
    url = (f"https://data.ecmwf.int/forecasts/{start:%Y%m%d}/{start.hour:02d}z/"
           f"{model}/0p25/oper/{start:%Y%m%d%H}0000-0h-oper-fc.index")
    with urlopen(Request(url, headers={"User-Agent": "gpuwrf-onboarding/1"}), timeout=30) as response:
        blob = response.read(2_000_001)
    if len(blob) > 2_000_000:
        raise ValueError("ECMWF index exceeds 2 MB metadata cap")
    records = [json.loads(line) for line in blob.splitlines()]
    params = {str(r.get("param")) for r in records}
    missing = []
    for name in ("2t", "2d", "10u", "10v", "sp", "msl", "skt", "lsm", "sd", "z"):
        if name not in params:
            missing.append(name)
    for kind, aliases in (("soil_temperature", ("stl", "st", "sot")), ("soil_moisture", ("swvl", "vsw"))):
        for layer in range(1, 5):
            if f"{aliases[0]}{layer}" not in params and not any(
                    str(r.get("param")) in aliases[1:] and r.get("levtype") == "sol"
                    and str(r.get("levelist")) == str(layer)
                    for r in records):
                missing.append(f"{kind}_layer_{layer}")
    return dict(source=source, model=model, status="MISSING_FIELDS" if missing else "UNQUALIFIED_VTABLE",
                missing=missing, available_params=sorted(params), url=url, bytes=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(), gpu_qualified=False,
                note="Initial-time open-data inventory only; WPS Vtable/caller qualification and full temporal audit pending")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, budget: int, *, grib: bool = True) -> int:
    """Stream with a cumulative byte cap, atomic publication and no partial reuse."""
    if budget <= 0:
        raise ValueError("download budget exhausted")
    if target.exists():
        raise ValueError(f"download target already exists: {target}")
    temporary = target.with_name(target.name + ".part")
    count = 0
    try:
        with urlopen(Request(url, headers={"User-Agent": "gpuwrf-onboarding/1"}), timeout=90) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > budget:
                raise ValueError("download exceeds remaining byte budget")
            with temporary.open("xb") as out:
                while block := response.read(min(1024 * 1024, budget - count + 1)):
                    count += len(block)
                    if count > budget:
                        raise ValueError("download exceeds remaining byte budget")
                    out.write(block)
        if grib:
            with temporary.open("rb") as stream:
                magic = stream.read(4)
                stream.seek(-4, 2)
                end = stream.read(4)
            if magic != b"GRIB" or end != b"7777":
                raise ValueError("response is not a complete GRIB file (check cycle availability)")
        temporary.replace(target)
        return count
    finally:
        temporary.unlink(missing_ok=True)


def gfs_url(plan: dict, lead: int) -> str:
    start = datetime.strptime(plan["start"], "%Y-%m-%d_%H:%M:%S")
    west, south, east, north = plan["domain_bbox"]
    # Two degrees beyond the PADDED domain for metgrid interpolation stencils.
    if west - 2 < -180 or east + 2 > 180:
        raise ValueError("GFS subset needs 2° interpolation margin without dateline crossing")
    query = dict(file=f"gfs.t{start.hour:02d}z.pgrb2.0p25.f{lead:03d}",
                 dir=f"/gfs.{start:%Y%m%d}/{start.hour:02d}/atmos",
                 all_lev="on", all_var="on", subregion="",
                 leftlon=west - 2, rightlon=east + 2,
                 bottomlat=south - 2, toplat=north + 2)
    return "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urlencode(query)


def inventory_grib(path: Path) -> list[dict]:
    try:
        import eccodes
    except ImportError as exc:
        raise ValueError("GRIB audit requires the optional extra: pip install 'gpuwrf[onboard]'") from exc
    fields = []
    with path.open("rb") as stream:
        while (handle := eccodes.codes_grib_new_from_file(stream)) is not None:
            try:
                def get(key, default=None):
                    try:
                        if key in ("typeOfLevel", "shortName"):
                            return eccodes.codes_get(handle, key)
                        # ecCodes returns display strings ('sfc', 'pl', …) for
                        # coded enums unless the numeric accessor is requested.
                        return eccodes.codes_get_long(handle, key)
                    except eccodes.CodesInternalError:
                        return default
                entry = {key: get(key) for key in ("discipline", "parameterCategory", "parameterNumber",
                         "typeOfFirstFixedSurface", "typeOfLevel", "level", "shortName",
                         "validityDate", "validityTime")}
                if entry["typeOfFirstFixedSurface"] == 106:
                    for end in ("First", "Second"):
                        value = get(f"scaledValueOf{end}FixedSurface")
                        scale = get(f"scaleFactorOf{end}FixedSurface")
                        if value is not None and scale is not None and abs(scale) < 10:
                            entry[end.lower() + "_depth_m"] = value * 10.0 ** (-scale)
                fields.append(entry)
            finally:
                eccodes.codes_release(handle)
    if not fields:
        raise ValueError(f"no GRIB messages found in {path}")
    return fields


def field_report(fields: list[dict], valid: datetime) -> dict:
    """Check actual GRIB2 codes, levels and validity for 4-layer Noah-MP forcing.

    WPS Vtable.GFS and Vtable.ECMWF define these codes; humidity can be either
    relative or specific. This is a metadata gate; real.exe also checks data.
    """
    def matches(code, surface, level=None):
        return [f for f in fields if tuple(f.get(k) for k in
                ("discipline", "parameterCategory", "parameterNumber")) == code
                and f.get("typeOfFirstFixedSurface") == surface
                and (level is None or f.get("level") == level)
                and f.get("validityDate") == int(valid.strftime("%Y%m%d"))
                and f.get("validityTime") == valid.hour * 100 + valid.minute]
    missing = []
    pressure = {}
    for name, code in (("temperature", (0, 0, 0)), ("u", (0, 2, 2)),
                       ("v", (0, 2, 3)), ("height", (0, 3, 5))):
        pressure[name] = {f["level"] for f in matches(code, 100)}
    pressure["humidity"] = {f["level"] for f in matches((0, 1, 1), 100) + matches((0, 1, 0), 100)}
    common = set.intersection(*pressure.values())
    if len(common) < 8 or not common or min(common) > 50 or max(common) < 950:
        missing.append("common pressure-level T/U/V/HGT/humidity: ≥8 levels spanning 50–950 hPa")
    for name, alternatives in (
        ("temperature_2m", [((0, 0, 0), 103, 2)]),
        ("humidity_2m", [((0, 1, 1), 103, 2), ((0, 0, 6), 103, 2), ((0, 1, 0), 103, 2)]),
        ("u_10m", [((0, 2, 2), 103, 10)]), ("v_10m", [((0, 2, 3), 103, 10)]),
        ("surface_pressure", [((0, 3, 0), 1, None)]),
        ("land_sea_mask", [((2, 0, 0), 1, None), ((2, 0, 218), 1, None)]),
        ("skin_temperature", [((0, 0, 0), 1, None), ((0, 0, 17), 1, None)]),
        ("surface_height", [((0, 3, 5), 1, None)]),
        ("snow_water_equivalent", [((0, 1, 13), 1, None)]),
    ):
        if not any(matches(*alt) for alt in alternatives):
            missing.append(name)
    depths = ((0, .1), (.1, .4), (.4, 1), (1, 2))
    soil = {}
    for name, codes in (("soil_temperature", ((2, 0, 2), (0, 0, 0))),
                        ("soil_moisture", ((2, 0, 192), (2, 0, 9)))):
        candidates = [f for code in codes for f in matches(code, 106)]
        soil[name] = [(f.get("first_depth_m"), f.get("second_depth_m")) for f in candidates]
        for top, bottom in depths:
            if not any(a is not None and b is not None and math.isclose(a, top, abs_tol=1e-6)
                       and math.isclose(b, bottom, abs_tol=1e-6) for a, b in soil[name]):
                missing.append(f"{name}_{top:g}_{bottom:g}m")
    return dict(status="PASS" if not missing else "MISSING_FIELDS", missing=missing,
                common_pressure_levels_hpa=sorted(common), soil_depths_m=soil,
                valid_time=valid.isoformat(), messages=len(fields),
                note="Metadata completeness only; no soil replacement or fabricated fields")


def retrieve_gfs(plan: dict, directory: Path, budget: int = MAX_DOWNLOAD_BYTES) -> list[dict]:
    records = []
    start = datetime.strptime(plan["start"], "%Y-%m-%d_%H:%M:%S")
    directory.mkdir()
    for lead in plan["boundary_hours"]:
        path = directory / f"gfs.f{lead:03d}.grib2"
        url = gfs_url(plan, lead)
        count = download(url, path, budget)
        budget -= count
        report = field_report(inventory_grib(path), start + timedelta(hours=lead))
        records.append(dict(lead_hours=lead, path=str(path), url=url, bytes=count,
                            sha256=sha256(path), field_report=report))
        (directory / "forcing.json").write_text(json.dumps(records, indent=2) + "\n")
        if report["status"] != "PASS":
            raise ValueError(f"GFS f{lead:03d} missing required fields: {report['missing']}")
    return records
