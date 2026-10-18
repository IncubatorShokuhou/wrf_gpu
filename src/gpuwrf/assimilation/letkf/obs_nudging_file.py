"""ObsBatch -> WRF ``OBS_DOMAIN<d>01`` observation-nudging file (fddaobs), for the A3 equal-data baseline.

Format = ``share/wrf_fddaobs_in.F`` (pristine WRF) formats 101/102/1021/103/104/105, read back by the port's strict
decoder ``gpuwrf.io.obs_domain.read_obs_domain``. Semantics follow WRF ``in4dob`` (same file, lines 512-940):

- records in chronological order; one record per sonde launch (``is_sound`` T, one row per pressure level:
  p [Pa], height [m], T [K], u, v [m/s], RH [%]) and one per surface station and time (``is_sound`` F, 9 fields:
  SLP, ref p, height, T, u, v, RH, PSFC, precip);
- winds are EARTH-relative and carry QC = 129 on both components, which makes WRF rotate them to the grid
  (``rotate_vector``); any other good QC would be taken as grid-relative;
- humidity: WRF converts RH to mixing ratio with ``rh2r`` (es = 6.1078 hPa exp(17.2693882 (T - 273.16) / (T - 35.86)),
  eps 0.62197, r = RH/100 * r_sat) at the record pressure (sonde level p, surface PSFC). The writer therefore writes
  RH = 100 q / r_sat,WRF(T, p) and the SAME pressure, so WRF recovers the ObsBatch mixing ratio q (to the f11.3
  rounding of RH). RH may exceed 100 % when q > r_sat,WRF (no clipping);
- surface PSFC = the station's PSFC observation at that time if present, else the ICAO standard-atmosphere pressure
  at the station height (the pressure ``obs.read_station_obs`` used for its RH -> q conversion). WRF needs a valid
  PSFC (70-105 kPa) to use T and RH; obs nudging never nudges pressure;
- missing value = -888888. with QC -888888.; good QC = 0.

Only usable (qc == 0) observations with role 'assimilate' are written by default; withheld obs never enter the file.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import obs as ob

MISSING = -888888.0
QC_GOOD = 0.0
QC_EARTH_WIND = 129.0          # wrf_fddaobs_in.F: u_met_qc == v_met_qc == 129 -> rotate earth -> grid
WRF_EPS, WRF_E0, WRF_A, WRF_B, WRF_T0 = 0.62197, 6.1078, 17.2693882, 35.86, 273.16
PLATFORM = {"sonde": "FM-35 TEMP", "metar": "FM-15 METAR", "surface": "FM-12 SYNOP"}

# WRF namelist &fdda for obs nudging on d01: values = pristine WRF test/em_real/examples.namelist lines 201-219 (d01
# column; Registry.EM_COMMON:2753-2808 defaults are 0/off for most of them). The window is the DA window t0 .. t_f.
NAMELIST_FDDA_D01 = OrderedDict([
    ("obs_nudge_opt", 1), ("max_obs", 150000), ("fdda_start", 0.0), ("fdda_end", 720.0),
    ("obs_nudge_wind", 1), ("obs_coef_wind", 6.0e-4), ("obs_nudge_temp", 1), ("obs_coef_temp", 6.0e-4),
    ("obs_nudge_mois", 1), ("obs_coef_mois", 6.0e-4), ("obs_rinxy", 240.0), ("obs_rinsig", 0.1),
    ("obs_twindo", 0.6666667), ("obs_npfi", 10), ("obs_ionf", 2), ("obs_idynin", 0), ("obs_dtramp", 40.0),
    ("obs_prt_max", 10), ("obs_prt_freq", 10), ("obs_ipf_in4dob", ".true."), ("obs_ipf_errob", ".true."),
    ("obs_ipf_nudob", ".true."), ("obs_ipf_init", ".true."),
])


def wrf_rsat(t_k, p_hpa):
    """WRF rh2r saturation mixing ratio over water (iice = 0), p in hPa."""
    t_k, p_hpa = np.asarray(t_k, float), np.asarray(p_hpa, float)
    esat = WRF_E0 * np.exp(WRF_A * (t_k - WRF_T0) / (t_k - WRF_B))
    return WRF_EPS * esat / (p_hpa - esat)


def wrf_rh2r(rh_pct, t_k, p_hpa):
    """WRF rh2r: r = RH/100 * r_sat(T, p)."""
    return np.asarray(rh_pct, float) * 0.01 * wrf_rsat(t_k, p_hpa)


def _f11_3(x):
    s = f"{x:11.3f}"
    if len(s) != 11:
        raise ValueError(f"value {x} does not fit Fortran f11.3")
    return s


def _pair(value, qc):
    return f"{_f11_3(value)} {_f11_3(qc)} "


def _logical(flag):
    return "   T" if flag else "   F"


def _header(when, lat, lon, station_id, name, platform, source, elevation, is_sound, count):
    if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(elevation)):
        raise ValueError("record location/elevation must be finite")
    lines = [" " + when.strftime("%Y%m%d%H%M%S"),
             f"  {lat:9.4f} {lon:9.4f} ",
             f"  {station_id[:40]:<40}   {name[:40]:<40}   ",
             f"  {platform[:16]:<16}  {source[:16]:<16}  {elevation:8.0f}  {_logical(is_sound)}  {_logical(False)}  "
             f"{count:5d}"]
    if len(lines[1]) != 22 or len(f"{elevation:8.0f}") != 8:
        raise ValueError("lat/lon/elevation do not fit the fixed-width format")
    return lines


def _as_datetime(t64):
    return datetime.fromtimestamp(int(np.datetime64(t64, "s").astype("int64")), tz=timezone.utc).replace(tzinfo=None)


def _good(x):
    return x is not None and math.isfinite(x)


def obsbatch_to_records(obs, *, include_withheld=False):
    """Group usable ObsBatch rows into OBS_DOMAIN records: list of dicts (time, lat, lon, ..., rows)."""
    sel = obs.usable & np.isin(obs.kind, [ob.SONDE, ob.SURFACE])
    if not include_withheld:
        sel &= obs.role == "assimilate"
    sub = obs.subset(sel)
    if np.isnat(sub.time).any():
        raise ValueError("every written observation needs an absolute time")
    records = []
    # --- soundings: one record per launch (station + launch id + time)
    sonde = sub.subset(sub.kind == ob.SONDE)
    for key in sorted(set(zip(sonde.station, sonde.launch, sonde.time.astype("int64")))):
        part = sonde.subset((sonde.station == key[0]) & (sonde.launch == key[1]) &
                            (sonde.time.astype("int64") == key[2]))
        levels = OrderedDict()
        for p in sorted(set(part.p_pa.tolist()), reverse=True):
            at = part.subset(part.p_pa == p)
            vals = {int(v): (float(x), float(h)) for v, x, h in zip(at.var, at.value, at.height_m)}
            levels[p] = vals
        rows = []
        for p, vals in levels.items():
            t = vals.get(ob.T, (None,))[0]
            q = vals.get(ob.QV, (None,))[0]
            u = vals.get(ob.U, (None,))[0]
            v = vals.get(ob.V, (None,))[0]
            h = next(iter(vals.values()))[1]
            rh = 100.0 * q / float(wrf_rsat(t, p / 100.0)) if (_good(q) and _good(t)) else None
            wind = _good(u) and _good(v)
            rows.append([(p, QC_GOOD), (h, QC_GOOD) if _good(h) else (MISSING, MISSING),
                         (t, QC_GOOD) if _good(t) else (MISSING, MISSING),
                         (u, QC_EARTH_WIND) if wind else (MISSING, MISSING),
                         (v, QC_EARTH_WIND) if wind else (MISSING, MISSING),
                         (rh, QC_GOOD) if _good(rh) else (MISSING, MISSING)])
        lat, lon = float(np.mean(part.lat)), float(np.mean(part.lon))
        elev = float(np.nanmin(part.height_m)) if np.isfinite(part.height_m).any() else MISSING
        records.append({"time": _as_datetime(key[2].astype("datetime64[s]") if isinstance(key[2], np.datetime64)
                                             else np.datetime64(int(key[2]), "s")),
                        "lat": lat, "lon": lon, "station_id": str(key[0]), "name": str(key[1]),
                        "platform": PLATFORM["sonde"], "source": "IGRA2", "elevation": elev, "is_sound": True,
                        "rows": rows, "drifted": bool(np.ptp(part.lat) > 0 or np.ptp(part.lon) > 0)})
    # --- surface: one record per station and time
    surf = sub.subset(sub.kind == ob.SURFACE)
    for key in sorted(set(zip(surf.station, surf.time.astype("int64")))):
        part = surf.subset((surf.station == key[0]) & (surf.time.astype("int64") == key[1]))
        vals = {int(v): float(x) for v, x in zip(part.var, part.value)}
        elev = float(part.height_m[0])
        t, q = vals.get(ob.T2), vals.get(ob.Q2)
        u, v = vals.get(ob.U10), vals.get(ob.V10)
        psfc = vals.get(ob.PSFC)
        p_src = "PSFC observation"
        if not _good(psfc):
            psfc, p_src = float(ob.std_atmosphere_pressure(elev)), "ICAO standard atmosphere at station height"
        rh = 100.0 * q / float(wrf_rsat(t, psfc / 100.0)) if (_good(q) and _good(t)) else None
        wind = _good(u) and _good(v)
        network = str(key[0]).split(":", 1)[0]
        rows = [[(MISSING, MISSING), (MISSING, MISSING), (elev, QC_GOOD),
                 (t, QC_GOOD) if _good(t) else (MISSING, MISSING),
                 (u, QC_EARTH_WIND) if wind else (MISSING, MISSING),
                 (v, QC_EARTH_WIND) if wind else (MISSING, MISSING),
                 (rh, QC_GOOD) if _good(rh) else (MISSING, MISSING),
                 (psfc, QC_GOOD), (MISSING, MISSING)]]
        records.append({"time": _as_datetime(np.datetime64(int(key[1]), "s")), "lat": float(part.lat[0]),
                        "lon": float(part.lon[0]), "station_id": str(key[0]), "name": str(part.site[0]),
                        "platform": PLATFORM["metar"] if network == "metar" else PLATFORM["surface"],
                        "source": network.upper(), "elevation": elev, "is_sound": False, "rows": rows,
                        "psfc_source": p_src})
    records.sort(key=lambda r: r["time"])          # stable: WRF requires chronological order
    return records


def write_obs_domain(obs, path, *, include_withheld=False):
    """Write ``path`` (e.g. OBS_DOMAIN101) and return a JSON-able receipt with counts."""
    records = obsbatch_to_records(obs, include_withheld=include_withheld)
    lines = []
    for rec in records:
        lines += _header(rec["time"], rec["lat"], rec["lon"], rec["station_id"], rec["name"], rec["platform"],
                         rec["source"], rec["elevation"], rec["is_sound"], len(rec["rows"]))
        for row in rec["rows"]:
            lines.append(" " + "".join(_pair(x, qc) for x, qc in row))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    return {"path": str(path), "records": len(records), "soundings": sum(r["is_sound"] for r in records),
            "surface": sum(not r["is_sound"] for r in records),
            "sounding_levels": sum(len(r["rows"]) for r in records if r["is_sound"]),
            "drifted_soundings_use_mean_position": sum(r.get("drifted", False) for r in records),
            "psfc_from_std_atmosphere": sum(r.get("psfc_source", "").startswith("ICAO") for r in records),
            "namelist_fdda_d01": dict(NAMELIST_FDDA_D01)}


def namelist_fdda_text(fdda_end_min=720.0, **overrides):
    """&fdda block for d01 obs nudging over the DA window (append to namelist.input)."""
    values = OrderedDict(NAMELIST_FDDA_D01)
    values["fdda_end"] = float(fdda_end_min)
    values.update(overrides)
    body = "\n".join(f" {k:<20} = {v}," for k, v in values.items())
    return f"&fdda\n{body}\n/\n"


def records_to_values(records):
    """Decoded ``io.obs_domain`` records -> flat dict list (time, lat, lon, station, kind, p, T, u, v, q) with q
    recovered exactly as WRF does (rh2r at the record pressure). Used by the round-trip gate."""
    out = []
    for rec in records:
        for meas in rec.measurements:
            v = [m.value for m in meas]
            if rec.is_sounding:
                p, h, t, u, w, rh = v
                q = float(wrf_rh2r(rh, t, p / 100.0)) if rh != MISSING and t != MISSING else None
            else:
                _, _, h, t, u, w, rh, p, _ = v
                q = float(wrf_rh2r(rh, t, p / 100.0)) if rh != MISSING and t != MISSING else None
            out.append({"time": rec.time, "lat": rec.latitude, "lon": rec.longitude, "station": rec.station_id,
                        "sounding": rec.is_sounding, "p": p, "height": h, "t": None if t == MISSING else t,
                        "u": None if u == MISSING else u, "v": None if w == MISSING else w, "q": q,
                        "wind_qc": [m.qc for m in meas][3 if rec.is_sounding else 4]})
    return out
