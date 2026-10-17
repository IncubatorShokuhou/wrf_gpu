"""Strict versioned JSON stream configuration and mass-grid region selection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

FILENAME = "gpuwrf.streams.json"
# (carry group, member, units). No output-time physics or diagnostic re-solve.
FIELDS = {
    "T2": ("history_diagnostics", "t2", "K"),
    "Q2": ("history_diagnostics", "q2", "kg kg-1"),
    "U10": ("history_diagnostics", "u10", "m s-1"),
    "V10": ("history_diagnostics", "v10", "m s-1"),
    "PSFC": ("history_diagnostics", "psfc", "Pa"),
    "LH": ("history_diagnostics", "lh", "W m-2"),
    "HFX": ("state", "hfx", "W m-2"),
    "PBLH": ("state", "pblh", "m"),
    "TSK": ("state", "t_skin", "K"),
    "RAINC": ("state", "rainc_acc", "mm"),
    "RAINNC": ("state", "rain_acc", "mm"),
    "SWDOWN": ("radiation_diagnostics", "swdown", "W m-2"),
    "GLW": ("radiation_diagnostics", "glw", "W m-2"),
}
STATISTICS = ("instant", "min", "max", "mean", "accumulation")


def positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return float(value)


def only_keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{label}: expected object with keys {sorted(allowed)}")


def _validate_region(region):
    """Check the region payload without loading a domain or selecting grid points."""
    only_keys(region, {"bbox", "polygon", "stations"}, "region")
    if len(region) != 1:
        raise ValueError("region requires exactly one of bbox, polygon, stations")
    if "bbox" in region:
        box = region["bbox"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("bbox is [west, south, east, north] in degrees")
        west, south, east, north = (finite(v, "bbox") for v in box)
        if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            raise ValueError("bbox must be ordered within longitude/latitude bounds; dateline wrapping unsupported")
    elif "polygon" in region:
        vertices = region["polygon"]
        if not isinstance(vertices, list) or len(vertices) < 3:
            raise ValueError("polygon needs at least three [longitude, latitude] vertices")
        for vertex in vertices:
            if not isinstance(vertex, list) or len(vertex) != 2:
                raise ValueError("polygon vertices must be [longitude, latitude]")
            if not (-180 <= finite(vertex[0], "longitude") <= 180 and -90 <= finite(vertex[1], "latitude") <= 90):
                raise ValueError("polygon coordinates out of bounds")
        longitudes = [float(vertex[0]) for vertex in vertices]
        if max(longitudes) - min(longitudes) > 180:
            raise ValueError("dateline wrapping polygon unsupported")
    else:
        stations = region["stations"]
        if not isinstance(stations, list) or not stations:
            raise ValueError("stations must be a nonempty list")
        ids = set()
        for station in stations:
            only_keys(station, {"id", "i", "j", "lat", "lon"}, "station")
            sid = station.get("id")
            if not isinstance(sid, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sid) is None or sid in ids:
                raise ValueError("station id must be unique and use 1..64 letters/digits/_/-")
            ids.add(sid)
            if set(station) == {"id", "i", "j"}:
                positive_int(station["i"], "station i")
                positive_int(station["j"], "station j")
            elif set(station) == {"id", "lat", "lon"}:
                finite(station["lat"], "station latitude")
                finite(station["lon"], "station longitude")
            else:
                raise ValueError("station needs id and either i/j (1-based) or lat/lon")


@dataclass(frozen=True)
class StreamSpec:
    name: str
    domain: str
    channels: tuple[tuple[str, str], ...]
    region_json: str
    interval_steps: int | None
    interval_seconds: float | None
    buffer_frames: int
    signature: str

    def steps(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("domain timestep must be finite and positive")
        if self.interval_steps is not None:
            return self.interval_steps
        raw = self.interval_seconds / dt
        count = round(raw)
        if count < 1 or not math.isclose(raw, count, rel_tol=0, abs_tol=1.e-9):
            raise ValueError(f"{self.name}: interval_seconds must align exactly with domain dt={dt}")
        return count


def read_streams(path):
    def pairs(items):
        result = {}
        for key, val in items:
            if key in result:
                raise ValueError(f"duplicate stream JSON key {key}")
            result[key] = val
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    only_keys(value, {"version", "streams"}, "stream file")
    if type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("stream file requires version=1")
    if not isinstance(value.get("streams"), list):
        raise ValueError("streams must be a list")
    result, names = [], set()
    for stream in value["streams"]:
        only_keys(stream, {"name", "domain", "variables", "region", "interval_steps",
                           "interval_seconds", "buffer_frames"}, "stream")
        name, domain = stream.get("name"), stream.get("domain")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", name) is None:
            raise ValueError("stream name must be a safe identifier, at most 48 characters")
        if name in names:
            raise ValueError(f"duplicate stream name {name}")
        names.add(name)
        if not isinstance(domain, str) or re.fullmatch(r"d[0-9]{2}", domain) is None:
            raise ValueError("stream domain must be d01, d02, ...")
        if ("interval_steps" in stream) == ("interval_seconds" in stream):
            raise ValueError("declare exactly one of interval_steps or interval_seconds")
        steps = positive_int(stream["interval_steps"], "interval_steps") if "interval_steps" in stream else None
        seconds = finite(stream["interval_seconds"], "interval_seconds") if steps is None else None
        if seconds is not None and seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        capacity = positive_int(stream.get("buffer_frames", 32), "buffer_frames")
        variables = stream.get("variables")
        if isinstance(variables, list):
            if not all(isinstance(v, str) for v in variables):
                raise ValueError("stream variables must be names")
            if len(set(variables)) != len(variables):
                raise ValueError("duplicate stream variable")
            variables = {v: ["instant"] for v in variables}
        if not isinstance(variables, dict) or not variables:
            raise ValueError("variables must be a nonempty list or mapping to statistics")
        channels = []
        for var, stats in variables.items():
            if var not in FIELDS:
                raise ValueError(f"unsupported stream variable {var}; available: {', '.join(FIELDS)}")
            if not isinstance(stats, list) or not stats or any(s not in STATISTICS for s in stats):
                raise ValueError(f"{var}: statistics must be a nonempty list from {STATISTICS}")
            if len(set(stats)) != len(stats):
                raise ValueError(f"{var}: duplicate statistic")
            if "accumulation" in stats and var not in {"RAINC", "RAINNC"}:
                raise ValueError("accumulation is an endpoint difference for RAINC/RAINNC only")
            channels.extend((var, stat) for stat in stats)
        region = stream.get("region")
        _validate_region(region)
        canonical = json.dumps(stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        result.append(StreamSpec(name, domain, tuple(channels), json.dumps(region), steps, seconds,
                                 capacity, hashlib.sha256(canonical.encode()).hexdigest()))
    return tuple(result)


def region_points(spec, lat, lon):
    """Return (y,x,point IDs) at mass centers. Polygons include their boundary.

    Bboxes and polygons use lon/lat without dateline wrapping; station coordinates
    select the nearest mass center *within* the domain coordinate envelope.
    """
    lat, lon = np.asarray(lat), np.asarray(lon)
    if lat.ndim != 2 or lat.shape != lon.shape or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("stream latitude/longitude must be finite matching 2-D arrays")
    region = json.loads(spec.region_json)
    _validate_region(region)
    ids = None
    if "bbox" in region:
        west, south, east, north = map(float, region["bbox"])
        y, x = np.nonzero((lon >= west) & (lon <= east) & (lat >= south) & (lat <= north))
    elif "polygon" in region:
        vertices = np.asarray(region["polygon"], float)
        inside, edge = np.zeros(lat.shape, bool), np.zeros(lat.shape, bool)
        for (ax, ay), (bx, by) in zip(vertices, np.roll(vertices, -1, axis=0)):
            if ax == bx and ay == by:
                continue
            cross = (lon-ax)*(by-ay) - (lat-ay)*(bx-ax)
            edge |= (np.abs(cross) < 1.e-10) & (lon >= min(ax,bx)) & (lon <= max(ax,bx)) & (lat >= min(ay,by)) & (lat <= max(ay,by))
            if by != ay:
                inside ^= ((ay > lat) != (by > lat)) & (lon < ax+(lat-ay)*(bx-ax)/(by-ay))
        y, x = np.nonzero(inside | edge)
    else:
        stations = region["stations"]
        ys, xs, ids = [], [], []
        for station in stations:
            sid = station["id"]
            if set(station) == {"id", "i", "j"}:
                xi, yi = station["i"]-1, station["j"]-1
                if xi >= lat.shape[1] or yi >= lat.shape[0]:
                    raise ValueError(f"station {sid} outside domain")
            elif set(station) == {"id", "lat", "lon"}:
                a, o = float(station["lat"]), float(station["lon"])
                if not (lat.min() <= a <= lat.max() and lon.min() <= o <= lon.max()):
                    raise ValueError(f"station {sid} outside domain coordinate envelope")
                distance = (lat-a)**2 + ((lon-o)*np.cos(np.deg2rad(a)))**2
                yi, xi = np.unravel_index(np.argmin(distance), lat.shape)
            ys.append(yi); xs.append(xi); ids.append(sid)
        y, x = np.asarray(ys), np.asarray(xs)
    if len(x) == 0:
        raise ValueError(f"stream {spec.name}: region contains no mass-grid centers")
    if ids is None:
        ids = [f"j{j+1}_i{i+1}" for j,i in zip(y,x)]
    return np.asarray(y, np.int32), np.asarray(x, np.int32), tuple(ids)
