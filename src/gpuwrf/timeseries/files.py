"""Fixed-width ARW tslist and *.TS/profile formats from share/wrf_timeseries.F.

Surface columns follow WRITE_TS process_time_series=1. File indices and station
IDs are one based. Profile PH is mass-level height MSL, not geopotential/AGL.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np

SURFACE_FIELDS = ("T2", "Q2", "U10", "V10", "PSFC", "GLW", "GSW", "HFX", "LH",
                  "TSK", "TSLB", "RAINC", "RAINNC", "CLW")
PROFILE_FIELDS = ("UU", "VV", "WW", "PH", "TH", "QV", "PR")


@dataclass(frozen=True)
class Station:
    description: str
    prefix: str
    first: float
    second: float
    station_id: int
    ij: bool = False


@dataclass(frozen=True)
class StationLocation:
    station: Station
    domain_id: int
    i: int
    j: int
    latitude: float
    longitude: float
    elevation: float


def read_tslist(path: str | Path, max_ts_locs: int = 5) -> tuple[Station, ...]:
    """Read WRF's three header lines and A25,1X,A5,1X,F7.3,1X,F8.3 records.

    The I/J alternative uses I7/I8 in the same columns. Extra stations are
    ignored with a warning, just as wrf_tsin.F does. Unsafe/duplicate prefixes
    are rejected so a run cannot overwrite unrelated files.
    """
    if max_ts_locs <= 0:
        return ()
    lines = Path(path).read_text(encoding="ascii").splitlines()
    if len(lines) < 3:
        raise ValueError("tslist requires three header lines")
    first, second = lines[1][32:39].upper(), lines[1][41:48].upper()
    if "LAT" in first and "LON" in second:
        ij = False
    elif "I" in first and "J" in second:
        ij = True
    else:
        raise ValueError("tslist header requires LAT LON or I J in WRF columns")
    stations = []
    for number, line in enumerate(lines[3:3 + max_ts_locs], 1):
        try:
            prefix = line[26:31].rstrip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,5}", prefix):
                raise ValueError("prefix must be 1-5 letters, digits, '_' or '-'")
            if prefix in {s.prefix for s in stations}:
                raise ValueError("duplicate station prefix")
            convert = int if ij else float
            a, b = convert(line[32:39]), convert(line[40:48])
            if not np.isfinite([a, b]).all():
                raise ValueError("non-finite coordinates")
            if not ij and (abs(a) > 90 or abs(b) > 180):
                raise ValueError("latitude/longitude out of range")
        except ValueError as exc:
            raise ValueError(f"tslist line {number + 3}: {exc}") from exc
        stations.append(Station(line[:25].ljust(25), prefix, a, b, number, ij))
    if len(lines) > max_ts_locs + 3:
        warnings.warn(f"Ignoring time series locations beyond #{max_ts_locs}", stacklevel=2)
    return tuple(stations)


def _integer(value: int, width: int) -> str:
    text = str(value)
    return text.rjust(width) if len(text) <= width else "*" * width


def _real(value: float, width: int, decimals: int) -> str:
    # Fortran F retains the sign of negative zero and drops the leading zero
    # when that makes a value fit. Reject non-finites before formatting.
    if not np.isfinite(value):
        raise ValueError("time series contains NaN/Inf")
    text = f"{value:.{decimals}f}"
    if len(text) > width:
        text = re.sub(r"^(-?)0\.", r"\1.", text)
    return text.rjust(width) if len(text) <= width else "*" * width


def header(location: StationLocation, start_date: str) -> str:
    s = location.station
    a, b = (location.latitude, location.longitude) if s.ij else (s.first, s.second)
    # A26 receives CHARACTER(25)//' '; A6 receives ' '//TRIM(prefix).
    return (s.description.ljust(25)[:25] + " " + _integer(location.domain_id, 2)
            + _integer(s.station_id, 3) + " " + s.prefix.ljust(5)
            + " (" + _real(a, 7, 3) + "," + _real(b, 8, 3) + ") ("
            + _integer(location.i, 4) + "," + _integer(location.j, 4) + ") ("
            + _real(location.latitude, 7, 3) + "," + _real(location.longitude, 8, 3)
            + ") " + _real(location.elevation, 6, 1) + " meters  " + start_date + "\n")


class TimeSeriesWriter:
    """Write bounded host batches after the compiled step loop has completed.

    All targets must be absent. WRF replaces files at startup, but gpuwrf
    deliberately refuses an occupied stream. No restart append is implied.
    """

    def __init__(self, directory: str | Path, locations: tuple[StationLocation, ...],
                 start_date: str, levels: int):
        self.directory = Path(directory)
        self.locations = locations
        self.levels = int(levels)
        if not 1 <= self.levels <= 999:
            raise ValueError("time series levels must be between 1 and 999")
        self.directory.mkdir(parents=True, exist_ok=True)
        targets = [self.path(loc, ending) for loc in locations
                   for ending in ("TS", *PROFILE_FIELDS)]
        if len(set(targets)) != len(targets):
            raise ValueError("duplicate time series target")
        for target in targets:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"time series target exists: {target}")
        for loc in locations:
            for ending in ("TS", *PROFILE_FIELDS):
                with self.path(loc, ending).open("x", encoding="ascii") as file:
                    file.write(header(loc, start_date))
        self.rows = 0

    def path(self, loc: StationLocation, ending: str) -> Path:
        return self.directory / f"{loc.station.prefix}.d{loc.domain_id:02d}.{ending}"

    def write(self, hours, surface, profiles) -> None:
        hours, surface, profiles = map(np.asarray, (hours, surface, profiles))
        count = len(hours)
        expected = (count, len(self.locations))
        if (hours.shape != (count,) or surface.shape != (*expected, 14)
                or profiles.shape != (*expected, 7, self.levels)):
            raise ValueError("time series batch shape mismatch")
        if not all(np.isfinite(x).all() for x in (hours, surface, profiles)):
            raise ValueError("time series contains NaN/Inf")
        if np.any(np.diff(hours) <= 0) or (count and hours[0] <= getattr(self, "last_hour", -np.inf)):
            raise ValueError("time series hours must increase strictly")
        for index, loc in enumerate(self.locations):
            with self.path(loc, "TS").open("a", encoding="ascii") as file:
                for n, hour in enumerate(hours):
                    file.write(_integer(loc.domain_id, 2) + _real(hour, 13, 6)
                               + _integer(loc.station.station_id, 5) + _integer(loc.i, 5)
                               + _integer(loc.j, 5) + " "
                               + "".join(_real(v, 13, 5) + " " for v in surface[n, index]).rstrip() + "\n")
            for field, ending in enumerate(PROFILE_FIELDS):
                with self.path(loc, ending).open("a", encoding="ascii") as file:
                    for n, hour in enumerate(hours):
                        file.write(_real(hour, 13, 6) + " "
                                   + "".join(_real(v, 13, 5) for v in profiles[n, index, field]) + "\n")
        self.rows += count
        if count:
            self.last_hour = float(hours[-1])
