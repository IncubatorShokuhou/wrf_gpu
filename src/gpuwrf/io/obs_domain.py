"""Strict raw decoder for WRF's ``OBS_DOMAIN101`` observation-nudging file.

The fixed-width descriptors come from ``share/wrf_fddaobs_in.F:101-105``.
This is NOT the general little_r format. Raw values and QC codes are retained:
no humidity conversion, pressure synthesis, QC filtering, duplicate replacement,
projection, wind rotation or observation-window selection is performed here.
It is a staged ingestion component; operational ``obs_nudge_opt=1`` stays refused.
This module imports neither NumPy nor JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import TextIO


SOUNDING_FIELDS = ("pressure", "height", "temperature", "u", "v", "relative_humidity")
SURFACE_FIELDS = ("sea_level_pressure", "reference_pressure", "height", "temperature",
                  "u", "v", "relative_humidity", "surface_pressure", "precipitation")


@dataclass(frozen=True)
class ObservationValue:
    value: float
    qc: float


@dataclass(frozen=True)
class ObservationRecord:
    """One raw station report; sounding levels remain grouped and in file order.

    Units in WRF input: pressures Pa; heights/elevation m; temperature K;
    winds m/s; relative humidity percent. QC and missing sentinels are opaque.
    ``measurements`` tuples follow SOUNDING_FIELDS or SURFACE_FIELDS.
    Surface reports have exactly one measurement row regardless of meas_count.
    """

    time: datetime
    latitude: float
    longitude: float
    station_id: str
    name: str
    platform: str
    source: str
    elevation: float
    is_sounding: bool
    bogus: bool
    measurement_count: int
    measurements: tuple[tuple[ObservationValue, ...], ...]


def read_obs_domain(path: str | Path) -> tuple[ObservationRecord, ...]:
    """Read chronological OBS_DOMAIN records; reject malformed/non-finite input.

    Deliberately stricter than Fortran blank padding: required fields must be
    present. Missing scientific data must use a numeric WRF sentinel, never a
    blank, NaN or infinity. No records are discarded, deduplicated or reordered.
    """
    with Path(path).open(encoding="ascii") as f:
        return _read(f, str(path))


def _read(f: TextIO, label: str) -> tuple[ObservationRecord, ...]:
    records = []
    lineno = 0

    def error(message):
        return ValueError(f"{label}:{lineno}: {message}")

    def line(width):
        nonlocal lineno
        value = f.readline()
        lineno += 1
        if value == "":
            raise error("truncated OBS_DOMAIN record")
        value = value.rstrip("\r\n")
        if len(value) < width:
            raise error(f"truncated fixed-width field (need {width} columns, got {len(value)})")
        return value

    def real(value):
        try:
            result = float(value.replace("D", "E").replace("d", "e"))
        except ValueError as exc:
            raise error(f"invalid numeric field {value!r}") from exc
        if not math.isfinite(result):
            raise error("non-finite observation field")
        return result

    def logical(value):
        value = value.strip().upper().lstrip(".")
        if value.startswith("T"):
            return True
        if value.startswith("F"):
            return False
        raise error(f"invalid Fortran logical field {value!r}")

    while True:
        first = f.readline()
        if first == "":
            return tuple(records)
        lineno += 1
        if len(first.rstrip("\r\n")) < 15 or not first[1:15].isascii() or not first[1:15].isdigit():
            raise error("invalid observation date (1x,a14 YYYYMMDDHHMMSS)")
        try:
            time = datetime.strptime(first[1:15], "%Y%m%d%H%M%S")
        except ValueError as exc:
            raise error("invalid observation date (1x,a14 YYYYMMDDHHMMSS)") from exc
        if records and time < records[-1].time:
            raise error("observations are not in chronological order")
        coords = line(21)  # 2x,2(f9.4,1x); final trailing blank optional
        latitude, longitude = real(coords[2:11]), real(coords[12:21])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise error("latitude/longitude out of range")
        identity = line(85)  # 2x,2(a40,3x)
        station_id, name = identity[2:42].rstrip(), identity[45:85].rstrip()
        header = line(65)  # 2x,2(a16,2x),f8.0,2x,2(l4,2x),i5
        platform, source = header[2:18].rstrip(), header[20:36].rstrip()
        elevation = real(header[38:46])
        sounding, bogus = logical(header[48:52]), logical(header[54:58])
        try:
            count = int(header[60:65])
        except ValueError as exc:
            raise error("invalid measurement count") from exc
        if count < 1:
            raise error("measurement count must be positive")
        nfields = 6 if sounding else 9
        measurements = []
        for _ in range(count if sounding else 1):
            fields = line(1 + 24*nfields - 1)  # 1x,N(f11.3,1x,f11.3,1x)
            measurements.append(tuple(ObservationValue(real(fields[1+24*n:12+24*n]),
                                                        real(fields[13+24*n:24+24*n]))
                                      for n in range(nfields)))
        records.append(ObservationRecord(time, latitude, longitude, station_id, name,
                       platform, source, elevation, sounding, bogus, count, tuple(measurements)))
