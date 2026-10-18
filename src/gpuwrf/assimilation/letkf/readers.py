"""Observation source readers for ``gpuwrf da`` (config ``obs.sources``) -> one ``obs.ObsBatch``.

Generic formats (no site-specific assumptions):

``igra2``        NOAA IGRA2 v2 sounding file of any station (``obs.read_igra2``; mandatory + significant levels).
``isd_lite``     NOAA ISD-Lite hourly surface file (12 integer columns, -9999 missing). ISD-Lite carries no
                 station coordinates: give ``station`` {id, lat, lon, elevation_m} or ``isd_history`` (NOAA
                 isd-history.csv) + ``station_id`` "USAF-WBAN". Uses T, dew point and wind; sea-level pressure is
                 never used as surface pressure.
``csv_surface``  one row per station and time: ``station_id, lat, lon, elevation_m, time`` and any of
                 ``t2_k|t2_c``, ``q2_kgkg|td2_k|td2_c|rh2_pct``, ``u10_ms+v10_ms|wspd10_ms+wdir10_deg``,
                 ``psfc_pa|psfc_hpa`` (station-level pressure). Empty cell = missing.
``csv_profile``  upper-air levels: ``station_id, lat, lon, time`` + ``p_pa|p_hpa`` and any of ``height_m``,
                 ``t_k|t_c``, ``qv_kgkg|td_k|td_c|rh_pct``, ``u_ms+v_ms|wspd_ms+wdir_deg``; one launch per
                 (station_id, time).
``obsbatch_json`` ``ObsBatch.to_dict`` JSON (this package's own table; round trip).

Site-specific adapters kept from the A1 probe: ``obsdb_parquet`` (ObsDB extract parquet written by guarded
extraction, ``obs.read_station_obs``) and ``msg_clm_npz`` (MSG cloud mask, experimental, withheld only).

Conventions are those of :mod:`obs`: UTC, T in K, humidity as water-vapour MIXING RATIO [kg/kg] (from dew
point or RH with the station pressure, else the ICAO standard atmosphere at the station height — counted),
EARTH-relative winds from meteorological direction (blowing FROM), errors from the obs error table. Times
must carry an explicit UTC offset; a naive time is refused. For each station the record nearest to the
analysis time within ``window_minutes`` is used (ties: the earlier record). Nothing is dropped silently:
skips are counted in ``meta['counters']``. Host NumPy only.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np

from . import obs as ob

_MISSING = ("", "na", "nan", "null", "none", "-9999", "-999", "-888888")


# ------------------------------------------------------------------------------------------- helpers

def _utc_text(text, where):
    try:
        when = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{where}: not an ISO 8601 time: {text!r}") from exc
    if when.tzinfo is None:
        raise ValueError(f"{where}: time {text!r} has no UTC offset")
    return when.astimezone(timezone.utc).replace(microsecond=0)


def _num(row, key, where):
    if key not in row:
        return None
    text = (row[key] or "").strip()
    if text.lower() in _MISSING:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{where}: column {key!r} is not a number: {text!r}") from exc
    if not math.isfinite(value):
        return None
    return value


def _dt64(when: datetime):
    return np.datetime64(when.replace(tzinfo=None), "s")


def _in_bbox(lat, lon, bbox):
    if bbox is None:
        return True
    lon = (lon + 180.0) % 360.0 - 180.0
    return bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]


def _nearest(records, t_a, window):
    """records: list of (time, payload); nearest to t_a within window (ties -> earlier)."""
    best = None
    for when, payload in records:
        off = abs((when - t_a).total_seconds())
        if off > window.total_seconds():
            continue
        key = (off, when)
        if best is None or key < best[0]:
            best = (key, when, payload)
    return None if best is None else (best[1], best[2])


def _humidity(t_k, p_pa, *, q=None, td_k=None, rh_pct=None, counters=None):
    """Mixing ratio from one of q / dew point / RH (RH outside (0, 105] % refused and counted)."""
    if q is not None:
        return q if q >= 0 else None
    if td_k is not None:
        return float(ob.qv_from_dewpoint(td_k, p_pa))
    if rh_pct is not None and t_k is not None:
        if 0.0 < rh_pct <= 105.0:
            return float(ob.qv_from_rh(min(rh_pct, 100.0) / 100.0, t_k, p_pa))
        if counters is not None:
            counters["rh_out_of_range"] = counters.get("rh_out_of_range", 0) + 1
    return None


def _wind(u, v, speed, direction, counters):
    if u is not None and v is not None:
        return u, v
    if speed is None or direction is None:
        if (speed is None) != (direction is None):
            counters["wind_pair_missing"] = counters.get("wind_pair_missing", 0) + 1
        return None
    if speed < 0 or not (0.0 <= direction <= 360.0) or (direction == 0.0 and speed > 0.0):
        counters["wind_invalid"] = counters.get("wind_invalid", 0) + 1   # ISD/SYNOP: 0 deg only for calm
        return None
    ue, ve = ob.wind_components(speed, direction)
    return float(ue), float(ve)


class _Rows:
    """Column accumulator for ObsBatch.from_columns."""

    def __init__(self):
        self.cols = {k: [] for k in ("var", "value", "error", "lat", "lon", "p_pa", "height_m", "dt_s", "station",
                                     "time", "launch")}

    def add(self, **kw):
        for key in self.cols:
            self.cols[key].append(kw.get(key, np.nan if key in ("p_pa", "height_m") else ""))

    def batch(self, meta):
        c = self.cols
        if not c["var"]:
            return ob.ObsBatch.empty(meta)
        return ob.ObsBatch.from_columns(
            var=np.asarray(c["var"], np.int16), value=np.asarray(c["value"], float),
            error=np.asarray(c["error"], float), lat=np.asarray(c["lat"], float), lon=np.asarray(c["lon"], float),
            p_pa=np.asarray(c["p_pa"], float), height_m=np.asarray(c["height_m"], float),
            dt_s=np.asarray(c["dt_s"], float), station=np.asarray(c["station"], dtype=object),
            launch=np.asarray(c["launch"], dtype=object), time=np.asarray(c["time"], dtype="datetime64[s]"),
            meta=meta)


def _surface_rows(rows, station, lat, lon, elev, when, t_a, values, table, counters, include_pressure):
    """values: dict with optional t_k, q, td_k, rh_pct, u, v, speed, direction, p_pa."""
    dt = (when - t_a).total_seconds()
    common = dict(lat=lat, lon=lon, height_m=elev, dt_s=dt, station=station, time=_dt64(when))
    p_stn = values.get("p_pa")
    if p_stn is not None and include_pressure:
        rows.add(var=ob.PSFC, value=p_stn, error=ob.error_for(ob.SURFACE, ob.PSFC, table), **common)
    t_k = values.get("t_k")
    if t_k is not None:
        rows.add(var=ob.T2, value=t_k, error=ob.error_for(ob.SURFACE, ob.T2, table), **common)
    p_use = p_stn if p_stn is not None else float(ob.std_atmosphere_pressure(elev))
    q = _humidity(t_k, p_use, q=values.get("q"), td_k=values.get("td_k"), rh_pct=values.get("rh_pct"),
                  counters=counters)
    if q is not None:
        if t_k is None:
            counters["q2_without_t2"] = counters.get("q2_without_t2", 0) + 1
        else:
            if p_stn is None and values.get("q") is None:
                counters["q2_std_atmosphere_pressure"] = counters.get("q2_std_atmosphere_pressure", 0) + 1
            rows.add(var=ob.Q2, value=q, error=float(ob.error_for(ob.SURFACE, ob.Q2, table, t_k=t_k, p_pa=p_use)),
                     **common)
    wind = _wind(values.get("u"), values.get("v"), values.get("speed"), values.get("direction"), counters)
    if wind is not None:
        rows.add(var=ob.U10, value=wind[0], error=ob.error_for(ob.SURFACE, ob.U10, table), **common)
        rows.add(var=ob.V10, value=wind[1], error=ob.error_for(ob.SURFACE, ob.V10, table), **common)


# ------------------------------------------------------------------------------------------- CSV formats

def _pick(row, *pairs):
    """First available of (column, converter) pairs."""
    for key, conv in pairs:
        if key in row:
            value = row[key]
            if value is not None:
                return conv(value)
    return None


def _csv_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        header = [h.strip() for h in reader.fieldnames]
        if len(set(header)) != len(header):
            raise ValueError(f"{path}: duplicate column names")
        for number, row in enumerate(reader, start=2):
            yield number, header, {k.strip(): v for k, v in row.items() if k is not None}


_SURFACE_COLUMNS = {"station_id", "lat", "lon", "elevation_m", "time", "t2_k", "t2_c", "q2_kgkg", "td2_k", "td2_c",
                    "rh2_pct", "u10_ms", "v10_ms", "wspd10_ms", "wdir10_deg", "psfc_pa", "psfc_hpa"}
_PROFILE_COLUMNS = {"station_id", "lat", "lon", "time", "p_pa", "p_hpa", "height_m", "t_k", "t_c", "qv_kgkg",
                    "td_k", "td_c", "rh_pct", "u_ms", "v_ms", "wspd_ms", "wdir_deg"}


def _check_header(path, header, allowed, required):
    unknown = sorted(set(header) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown column(s) {unknown}; allowed {sorted(allowed)}")
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")


def read_csv_surface(path, t_a, *, window_minutes=30.0, include_pressure=False, bbox=None, error_table=None):
    path = Path(path)
    table = error_table or ob.DEFAULT_OBS_ERRORS
    window = timedelta(minutes=float(window_minutes))
    counters = {"rows": 0, "outside_window": 0, "outside_bbox": 0}
    by_station, header_checked = {}, False
    for number, header, raw in _csv_rows(path):
        if not header_checked:
            _check_header(path, header, _SURFACE_COLUMNS, ("station_id", "lat", "lon", "elevation_m", "time"))
            header_checked = True
        where = f"{path}:{number}"
        counters["rows"] += 1
        sid = (raw.get("station_id") or "").strip()
        if not sid:
            raise ValueError(f"{where}: empty station_id")
        lat, lon, elev = (_num(raw, k, where) for k in ("lat", "lon", "elevation_m"))
        if lat is None or lon is None or elev is None:
            raise ValueError(f"{where}: lat, lon and elevation_m are required")
        when = _utc_text(raw.get("time"), where)
        if abs((when - t_a).total_seconds()) > window.total_seconds():
            counters["outside_window"] += 1
            continue
        vals = {k: _num(raw, k, where) for k in _SURFACE_COLUMNS - {"station_id", "time"}}
        meta = by_station.setdefault(sid, {"lat": lat, "lon": lon, "elev": elev, "records": []})
        if (meta["lat"], meta["lon"], meta["elev"]) != (lat, lon, elev):
            raise ValueError(f"{where}: station {sid!r} changes its coordinates within one file")
        meta["records"].append((when, vals))
    rows = _Rows()
    for sid, meta in sorted(by_station.items()):
        if not _in_bbox(meta["lat"], meta["lon"], bbox):
            counters["outside_bbox"] += 1
            continue
        when, v = _nearest(meta["records"], t_a, window)
        values = {
            "t_k": _pick(v, ("t2_k", float), ("t2_c", lambda x: x + 273.15)),
            "q": v.get("q2_kgkg"),
            "td_k": _pick(v, ("td2_k", float), ("td2_c", lambda x: x + 273.15)),
            "rh_pct": v.get("rh2_pct"),
            "u": v.get("u10_ms"), "v": v.get("v10_ms"),
            "speed": v.get("wspd10_ms"), "direction": v.get("wdir10_deg"),
            "p_pa": _pick(v, ("psfc_pa", float), ("psfc_hpa", lambda x: 100.0 * x)),
        }
        if (values["u"] is None) != (values["v"] is None):
            counters["wind_pair_missing"] = counters.get("wind_pair_missing", 0) + 1
            values["u"] = values["v"] = None
        _surface_rows(rows, f"csv:{sid}", meta["lat"], meta["lon"], meta["elev"], when, t_a, values, table,
                      counters, include_pressure)
    return rows.batch({"source": "csv_surface", "path": str(path), "t_a": t_a.isoformat(),
                       "window_minutes": float(window_minutes), "include_pressure": include_pressure,
                       "counters": counters, "stations": len(by_station)})


def read_csv_profile(path, t_a, *, window_minutes=90.0, bbox=None, error_table=None):
    path = Path(path)
    table = error_table or ob.DEFAULT_OBS_ERRORS
    window = timedelta(minutes=float(window_minutes))
    counters = {"rows": 0, "outside_window": 0, "outside_bbox": 0, "no_pressure": 0}
    rows = _Rows()
    header_checked = False
    for number, header, raw in _csv_rows(path):
        if not header_checked:
            _check_header(path, header, _PROFILE_COLUMNS, ("station_id", "lat", "lon", "time"))
            if not ({"p_pa", "p_hpa"} & set(header)):
                raise ValueError(f"{path}: needs a p_pa or p_hpa column")
            header_checked = True
        where = f"{path}:{number}"
        counters["rows"] += 1
        sid = (raw.get("station_id") or "").strip()
        lat, lon = _num(raw, "lat", where), _num(raw, "lon", where)
        if not sid or lat is None or lon is None:
            raise ValueError(f"{where}: station_id, lat and lon are required")
        when = _utc_text(raw.get("time"), where)
        if abs((when - t_a).total_seconds()) > window.total_seconds():
            counters["outside_window"] += 1
            continue
        if not _in_bbox(lat, lon, bbox):
            counters["outside_bbox"] += 1
            continue
        v = {k: _num(raw, k, where) for k in _PROFILE_COLUMNS - {"station_id", "time"}}
        p = _pick(v, ("p_pa", float), ("p_hpa", lambda x: 100.0 * x))
        if p is None or p <= 0:
            counters["no_pressure"] += 1
            continue
        t_k = _pick(v, ("t_k", float), ("t_c", lambda x: x + 273.15))
        td_k = _pick(v, ("td_k", float), ("td_c", lambda x: x + 273.15))
        launch = f"{sid}@{when.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        common = dict(lat=lat, lon=lon, p_pa=p, height_m=np.nan if v.get("height_m") is None else v["height_m"],
                      dt_s=(when - t_a).total_seconds(), station=f"csvprof:{sid}", time=_dt64(when), launch=launch)
        if t_k is not None:
            rows.add(var=ob.T, value=t_k, error=ob.error_for(ob.SONDE, ob.T, table), **common)
        q = _humidity(t_k, p, q=v.get("qv_kgkg"), td_k=td_k, rh_pct=v.get("rh_pct"), counters=counters)
        if q is not None and t_k is not None:
            rows.add(var=ob.QV, value=q, error=float(ob.error_for(ob.SONDE, ob.QV, table, t_k=t_k, p_pa=p)), **common)
        uv = (v.get("u_ms"), v.get("v_ms"))
        if (uv[0] is None) != (uv[1] is None):
            counters["wind_pair_missing"] = counters.get("wind_pair_missing", 0) + 1
            uv = (None, None)
        wind = _wind(uv[0], uv[1], v.get("wspd_ms"), v.get("wdir_deg"), counters)
        if wind is not None:
            rows.add(var=ob.U, value=wind[0], error=ob.error_for(ob.SONDE, ob.U, table), **common)
            rows.add(var=ob.V, value=wind[1], error=ob.error_for(ob.SONDE, ob.V, table), **common)
    return rows.batch({"source": "csv_profile", "path": str(path), "t_a": t_a.isoformat(),
                       "window_minutes": float(window_minutes), "counters": counters})


# ------------------------------------------------------------------------------------------- ISD-Lite

def _isd_history_station(path, station_id):
    usaf, wban = station_id.split("-")
    with open(path, newline="", encoding="utf-8", errors="replace") as stream:
        for row in csv.DictReader(stream):
            if row.get("USAF") == usaf and row.get("WBAN") == wban:
                lat, lon, elev = (_num(row, k, str(path)) for k in ("LAT", "LON", "ELEV(M)"))
                if lat is None or lon is None or elev is None:
                    raise ValueError(f"{path}: station {station_id} has no LAT/LON/ELEV(M)")
                return {"id": station_id, "lat": lat, "lon": lon, "elevation_m": elev}
    raise LookupError(f"{path}: station {station_id} not in the ISD history file")


def read_isd_lite(path, t_a, *, station=None, isd_history=None, station_id=None, window_minutes=30.0, bbox=None,
                  error_table=None):
    """ISD-Lite fields: year month day hour T(0.1 C) Td(0.1 C) SLP(0.1 hPa) dir(deg) speed(0.1 m/s) sky p1h p6h."""
    path = Path(path)
    if station is None:
        station = _isd_history_station(isd_history, station_id)
    table = error_table or ob.DEFAULT_OBS_ERRORS
    window = timedelta(minutes=float(window_minutes))
    counters = {"rows": 0, "outside_window": 0}
    records = []
    opener = __import__("gzip").open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii", errors="replace") as stream:
        for number, line in enumerate(stream, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 12:
                raise ValueError(f"{path}:{number}: ISD-Lite rows have 12 integer fields, got {len(parts)}")
            try:
                year, month, day, hour, t10, td10, _slp, wd, ws10, *_ = (int(x) for x in parts)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: non-integer ISD-Lite field") from exc
            counters["rows"] += 1
            when = datetime(year, month, day, hour, tzinfo=timezone.utc)
            if abs((when - t_a).total_seconds()) > window.total_seconds():
                counters["outside_window"] += 1
                continue
            records.append((when, {"t_k": None if t10 == -9999 else t10 / 10.0 + 273.15,
                                   "td_k": None if td10 == -9999 else td10 / 10.0 + 273.15,
                                   "speed": None if ws10 == -9999 else ws10 / 10.0,
                                   "direction": None if wd in (-9999, 999) else float(wd)}))
    rows = _Rows()
    meta = {"source": "isd_lite", "path": str(path), "station": station, "t_a": t_a.isoformat(),
            "window_minutes": float(window_minutes), "counters": counters,
            "pressure": "not used (ISD-Lite carries sea-level pressure only); Q2 uses the ICAO standard "
                        "atmosphere at the station height"}
    if not _in_bbox(station["lat"], station["lon"], bbox):
        counters["outside_bbox"] = 1
        return rows.batch(meta)
    picked = _nearest(records, t_a, window)
    if picked is None:
        counters["no_record_in_window"] = 1
        return rows.batch(meta)
    when, values = picked
    _surface_rows(rows, f"isd:{station['id']}", station["lat"], station["lon"], station["elevation_m"], when, t_a,
                  values, table, counters, False)
    return rows.batch(meta)


# ------------------------------------------------------------------------------------------- dispatch

def _domain_bbox(geo, margin_deg=0.0):
    lat, lon = np.asarray(geo["XLAT"], float), np.asarray(geo["XLONG"], float)
    return (float(lat.min()) - margin_deg, float(lat.max()) + margin_deg,
            float(lon.min()) - margin_deg, float(lon.max()) + margin_deg)


def read_source(source: dict, t_a: datetime, geo, *, error_table=None):
    """One validated ``obs.sources`` entry -> ObsBatch (dt_s relative to ``t_a``)."""
    fmt = source["format"]
    path = source["path"]
    bbox = _domain_bbox(geo)
    t64 = ob.to_datetime64(t_a)
    if fmt == "igra2":
        when = source.get("valid_time", t_a)
        return ob.read_igra2(path, source["station"], when, analysis_time=t64, error_table=error_table,
                             include_surface=source.get("include_surface", True), drift=source.get("drift", False),
                             ascent_mps=source.get("ascent_mps", 5.0))
    if fmt == "isd_lite":
        return read_isd_lite(path, t_a, station=source.get("station"), isd_history=source.get("isd_history"),
                             station_id=source.get("station_id"), window_minutes=source.get("window_minutes", 30.0),
                             bbox=bbox, error_table=error_table)
    if fmt == "csv_surface":
        return read_csv_surface(path, t_a, window_minutes=source.get("window_minutes", 30.0),
                                include_pressure=source.get("include_pressure", False), bbox=bbox,
                                error_table=error_table)
    if fmt == "csv_profile":
        return read_csv_profile(path, t_a, window_minutes=source.get("window_minutes", 90.0), bbox=bbox,
                                error_table=error_table)
    if fmt == "obsbatch_json":
        batch = ob.ObsBatch.from_dict(json.loads(Path(path).read_text()))
        return batch.replace(meta={**batch.meta, "source_file": str(path)})
    if fmt == "obsdb_parquet":
        options = dict(source.get("options", {}))
        if source.get("window_hours"):
            from .fourd import window_station_obs
            return window_station_obs(path, t64, float(source["window_hours"]), bbox=bbox, error_table=error_table,
                                      **options)
        window = float(source.get("window_minutes", 0.0)) * 60.0
        return ob.read_station_obs(path, t_a.timestamp() - window, t_a.timestamp() + window, bbox,
                                   analysis_time=t64, error_table=error_table, **options)
    if fmt == "msg_clm_npz":
        return ob.read_msg_clm(path, t64, geo, analysis_time=t64, stride=source.get("stride", 1),
                               min_pixels=source.get("min_pixels", 3), error_table=error_table, role="withhold")
    raise ValueError(f"unknown observation format {fmt!r}")


def read_sources(obs_cfg: dict, t_a: datetime, geo, frames=None):
    """All configured sources -> one ObsBatch (+ sonde layer super-obbing when ``sonde_superob``)."""
    table = ob.obs_error_table(obs_cfg.get("error_overrides"))
    batches = [read_source(s, t_a, geo, error_table=table) for s in obs_cfg["sources"]]
    batch = batches[0] if len(batches) == 1 else ob.ObsBatch.concat(batches)
    info = {"sources": [{"format": s["format"], "path": s["path"], "n": int(len(b))}
                        for s, b in zip(obs_cfg["sources"], batches)]}
    if obs_cfg.get("sonde_superob", True) and frames is not None and (np.asarray(batch.kind) == ob.SONDE).any():
        from .operators import sonde_layer_superob
        n_before = len(batch)
        batch = sonde_layer_superob(batch, frames, geo)
        info["sonde_superob"] = {"n_before": int(n_before), "n_after": int(len(batch))}
    return batch, info
