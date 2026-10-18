"""Observation ingest, QC and super-obbing for the LETKF probe (lane a1-obs).

Host-side NumPy only (fp64). Pure data handling: no JAX, no model imports. The
batch layout is ``proofs/a1_da/DESIGN.md`` section 13 plus per-observation
localisation scales ``lh_scale``/``lv_scale`` (multipliers of the domain L_h/L_v).

Conventions
- times: UTC, ``numpy.datetime64[s]``; ``dt_s`` = observation time - analysis time.
- ``value`` units: T/T2 [K], QV/Q2 water-vapour MIXING RATIO [kg/kg] (WRF QVAPOR/Q2),
  U/V/U10/V10 EARTH-relative [m/s], PSFC station-level pressure [Pa], CLDMASK cloud
  fraction in [0, 1]. ``p_pa`` [Pa] (NaN for surface/cloud obs), ``height_m`` [m MSL].
- ``qc`` is a bit field, 0 = usable. Readers never drop rejected data silently: they
  either flag it or count the skip in ``meta`` (masks -> counters).
- DESIGN_v2 section 4 columns: ``obs_id`` (unique string), ``site`` (alias group / site id),
  ``launch`` (sonde launch id), ``time`` (absolute UTC), ``averaging`` (instant/interval),
  ``role`` (assimilate/withhold); ``sigma`` is the observation-error std (= ``error``).

Moisture: WRF ``module_model_constants`` saturation vapour pressure (Bolton 1980 form,
svp1=0.6112 kPa, svp2=17.67, svp3=29.65 K, svpt0=273.15 K, over liquid water at all
temperatures, as WMO RH and WRF real.exe) and ``ep_2 = R_d/R_v = 287/461.6``:
    e = es(Td)            (dewpoint)   or   e = RH * es(T)   (relative humidity)
    qv = ep_2 * e / (p - e)            (mixing ratio, kg/kg)
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- codes
SONDE, SURFACE, CLOUD = 1, 2, 3
KIND_NAMES = {SONDE: "SONDE", SURFACE: "SURFACE", CLOUD: "CLOUD"}

T, QV, U, V = 1, 2, 3, 4
T2, Q2, U10, V10, PSFC = 11, 12, 13, 14, 15
CLDMASK = 21
VAR_NAMES = {T: "T", QV: "QV", U: "U", V: "V", T2: "T2", Q2: "Q2", U10: "U10", V10: "V10",
             PSFC: "PSFC", CLDMASK: "CLDMASK"}
VAR_KIND = {T: SONDE, QV: SONDE, U: SONDE, V: SONDE, T2: SURFACE, Q2: SURFACE, U10: SURFACE,
            V10: SURFACE, PSFC: SURFACE, CLDMASK: CLOUD}

QC_OK = 0
QC_GROSS = 1 << 0          # background (gross) check |y - Hxbar| > f * sqrt(so^2 + var Hx)
QC_OUTSIDE = 1 << 1        # outside domain, outside the model column, no matching land/sea node
QC_DZ = 1 << 2             # |z_station - z_model| above threshold
QC_SOURCE = 1 << 3         # rejected by the source (IGRA -8888, invalid pixel, missing partner)
QC_DUPLICATE = 1 << 4      # co-located alias of a higher-priority station
QC_THINNED = 1 << 5        # consumed by a super-observation
QC_BLACKLIST = 1 << 6      # protected / blacklisted station
QC_NONFINITE_HX = 1 << 7   # model equivalent not finite in at least one member
QC_SPREAD = 1 << 8         # ensemble spread in H(x) below the use threshold (cloud mask)
QC_NAMES = {QC_GROSS: "gross", QC_OUTSIDE: "outside", QC_DZ: "dz", QC_SOURCE: "source",
            QC_DUPLICATE: "duplicate", QC_THINNED: "thinned", QC_BLACKLIST: "blacklist",
            QC_NONFINITE_HX: "nonfinite_hx", QC_SPREAD: "spread"}

# ------------------------------------------------------------------------ constants
R_D, R_V, G = 287.0, 461.6, 9.81            # WRF module_model_constants
EP_2 = R_D / R_V
SVP1, SVP2, SVP3, SVPT0 = 0.6112, 17.67, 29.65, 273.15
LAPSE_RATE = 0.0065                          # K/m (standard atmosphere)


def sat_vapor_pressure(t_k):
    """Saturation vapour pressure over water [Pa] (WRF svp1..svpt0, Bolton 1980)."""
    t_k = np.asarray(t_k, dtype=float)
    return 1000.0 * SVP1 * np.exp(SVP2 * (t_k - SVPT0) / (t_k - SVP3))


def mixing_ratio(e_pa, p_pa):
    e_pa, p_pa = np.asarray(e_pa, dtype=float), np.asarray(p_pa, dtype=float)
    return EP_2 * e_pa / (p_pa - e_pa)


def qv_from_dewpoint(td_k, p_pa):
    return mixing_ratio(sat_vapor_pressure(td_k), p_pa)


def qv_from_rh(rh_frac, t_k, p_pa):
    return mixing_ratio(np.asarray(rh_frac, dtype=float) * sat_vapor_pressure(t_k), p_pa)


def qv_saturation(t_k, p_pa):
    return mixing_ratio(sat_vapor_pressure(t_k), p_pa)


def rh_from_qv(qv, t_k, p_pa):
    """Inverse of :func:`qv_from_rh` (fraction)."""
    qv, p_pa = np.asarray(qv, dtype=float), np.asarray(p_pa, dtype=float)
    e = qv * p_pa / (EP_2 + qv)
    return e / sat_vapor_pressure(t_k)


def wind_components(speed, direction_deg):
    """Meteorological direction (blowing FROM, degrees clockwise from north) -> earth u, v."""
    speed = np.asarray(speed, dtype=float)
    rad = np.deg2rad(np.asarray(direction_deg, dtype=float))
    return -speed * np.sin(rad), -speed * np.cos(rad)


def std_atmosphere_pressure(z_m):
    """ICAO standard-atmosphere pressure [Pa] at height z [m] (troposphere)."""
    return 101325.0 * (1.0 - 2.25577e-5 * np.asarray(z_m, dtype=float)) ** 5.25588


def to_datetime64(value):
    """UTC ``datetime64[s]`` from str/datetime/np.datetime64/pandas Timestamp (tz-aware -> UTC)."""
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[s]")
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return np.datetime64(int(round(float(value))), "s")  # POSIX seconds (UTC)
    import pandas as pd
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.to_datetime64(), "s")


# ----------------------------------------------------------------------- ObsBatch
_COLUMNS = {  # name -> kind of column
    "kind": "i1", "var": "i2", "value": "f", "error": "f", "lat": "f", "lon": "f", "p_pa": "f", "height_m": "f",
    "dt_s": "f", "station": "s", "qc": "i4", "lh_scale": "f", "lv_scale": "f",
    "obs_id": "s", "site": "s", "launch": "s", "time": "t", "averaging": "s", "role": "s"}
_FIELDS = tuple(_COLUMNS)
_FLOAT = tuple(k for k, v in _COLUMNS.items() if v == "f")
_DEFAULTS = {"error": np.nan, "p_pa": np.nan, "height_m": np.nan, "dt_s": 0.0, "station": "", "qc": QC_OK,
             "lh_scale": 1.0, "lv_scale": 1.0, "obs_id": "", "site": "", "launch": "",
             "time": np.datetime64("NaT", "s"), "averaging": "instant", "role": "assimilate"}
ROLES = ("assimilate", "withhold")
AVERAGING = ("instant", "interval")


def _coerce(name, raw, n):
    code = _COLUMNS[name]
    if code == "s":
        arr = np.atleast_1d(np.asarray(raw, dtype=object))
        return np.array([str(x) for x in arr], dtype=object) if len(arr) else np.zeros(0, object)
    if code == "t":
        return np.atleast_1d(np.asarray(raw)).astype("datetime64[s]")
    if code.startswith("i"):
        return np.atleast_1d(np.asarray(raw)).astype({"i1": np.int8, "i2": np.int16, "i4": np.int32}[code])
    return np.atleast_1d(np.asarray(raw, dtype=float)).astype(np.float64)


@dataclasses.dataclass(frozen=True, eq=False)
class ObsBatch:
    """Structure-of-arrays observation table (one row per scalar observation)."""

    kind: np.ndarray
    var: np.ndarray
    value: np.ndarray
    error: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    p_pa: np.ndarray
    height_m: np.ndarray
    dt_s: np.ndarray
    station: np.ndarray
    qc: np.ndarray
    lh_scale: np.ndarray
    lv_scale: np.ndarray
    obs_id: np.ndarray
    site: np.ndarray
    launch: np.ndarray
    time: np.ndarray
    averaging: np.ndarray
    role: np.ndarray
    meta: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        n = len(np.atleast_1d(self.value))
        for name in _FIELDS:
            arr = _coerce(name, getattr(self, name), n)
            if arr.ndim != 1 or len(arr) != n:
                raise ValueError(f"ObsBatch.{name}: expected shape ({n},), got {arr.shape}")
            object.__setattr__(self, name, arr)
        bad_var = ~np.isin(self.var, list(VAR_NAMES))
        if bad_var.any():
            raise ValueError(f"unknown var codes {np.unique(self.var[bad_var]).tolist()}")
        if np.any(self.kind != np.array([VAR_KIND[int(v)] for v in self.var], dtype=np.int8)):
            raise ValueError("kind does not match var")
        if not (np.isfinite(self.lat).all() and np.isfinite(self.lon).all()):
            raise ValueError("observation lat/lon must be finite")
        if (np.abs(self.lat) > 90).any():
            raise ValueError("latitude out of range")
        usable = self.qc == QC_OK
        if not np.isfinite(self.value[usable]).all():
            raise ValueError("usable observations need finite values")
        if not (self.error[usable] > 0).all():
            raise ValueError("usable observations need a positive finite error (sigma)")
        if (self.lh_scale <= 0).any() or (self.lv_scale <= 0).any():
            raise ValueError("localisation scales must be positive")
        if not np.isin(self.role, ROLES).all() or not np.isin(self.averaging, AVERAGING).all():
            raise ValueError(f"role must be in {ROLES}, averaging in {AVERAGING}")
        object.__setattr__(self, "lon", (self.lon + 180.0) % 360.0 - 180.0)
        missing = self.obs_id == ""
        if missing.any():
            ids = self.obs_id.copy()
            for k in np.flatnonzero(missing):
                stamp = str(self.time[k]) if not np.isnat(self.time[k]) else f"dt{self.dt_s[k]:+.0f}s"
                lev = f"/{self.p_pa[k]:.0f}Pa" if np.isfinite(self.p_pa[k]) else ""
                ids[k] = f"{self.station[k] or 'obs'}/{VAR_NAMES[int(self.var[k])]}/{stamp}{lev}"
            object.__setattr__(self, "obs_id", ids)
        empty_site = self.site == ""
        if empty_site.any():
            object.__setattr__(self, "site", np.where(empty_site, self.station, self.site).astype(object))

    # constructors -----------------------------------------------------------------
    @classmethod
    def from_columns(cls, *, var, value, lat, lon, meta=None, **cols):
        var = np.atleast_1d(np.asarray(var, dtype=np.int16))
        n = len(var)
        out = {"var": var, "value": value, "lat": np.broadcast_to(np.asarray(lat, float), (n,)),
               "lon": np.broadcast_to(np.asarray(lon, float), (n,))}
        out["kind"] = cols.pop("kind", np.array([VAR_KIND[int(v)] for v in var], dtype=np.int8))
        for name, default in _DEFAULTS.items():
            given = cols.pop(name, default)
            if _COLUMNS[name] == "s" and isinstance(given, str):
                given = np.array([given] * n, dtype=object)
            elif _COLUMNS[name] != "s":
                given = np.broadcast_to(np.asarray(given), (n,)).copy()
            out[name] = given
        if cols:
            raise TypeError(f"unknown ObsBatch columns {sorted(cols)}")
        return cls(**out, meta=dict(meta or {}))

    @classmethod
    def empty(cls, meta=None):
        return cls.from_columns(var=np.zeros(0, np.int16), value=np.zeros(0), lat=np.zeros(0),
                                lon=np.zeros(0), meta=meta)

    @classmethod
    def concat(cls, batches, meta=None):
        return concat(batches, meta=meta)

    # helpers ----------------------------------------------------------------------
    def __len__(self):
        return len(self.value)

    @property
    def sigma(self):
        """Observation-error standard deviation (DESIGN_v2 name)."""
        return self.error

    @property
    def usable(self):
        return self.qc == QC_OK

    @property
    def assimilate(self):
        return self.usable & (self.role == "assimilate")

    def columns(self):
        return {name: getattr(self, name) for name in _FIELDS}

    def replace(self, **cols):
        meta = cols.pop("meta", self.meta)
        merged = self.columns()
        merged.update(cols)
        return ObsBatch(**merged, meta=dict(meta))

    def subset(self, sel):
        sel = np.asarray(sel)
        if sel.dtype == bool and sel.shape != (len(self),):
            raise ValueError("boolean selector must match the batch length")
        return ObsBatch(**{k: v[sel] for k, v in self.columns().items()}, meta=dict(self.meta))

    def with_qc(self, mask, flag):
        mask = np.asarray(mask, dtype=bool)
        return self.replace(qc=np.where(mask, self.qc | np.int32(flag), self.qc))

    def with_role(self, mask, role):
        if role not in ROLES:
            raise ValueError(f"role must be in {ROLES}")
        return self.replace(role=np.where(np.asarray(mask, bool), role, self.role).astype(object))

    def to_dict(self):
        out = {}
        for k, v in self.columns().items():
            code = _COLUMNS[k]
            if code == "s":
                out[k] = [str(x) for x in v]
            elif code == "t":
                out[k] = [None if np.isnat(x) else str(x) + "Z" for x in v]
            else:
                out[k] = [None if isinstance(x, float) and not math.isfinite(x) else x for x in v.tolist()]
        out["meta"] = self.meta
        return out

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        meta = data.pop("meta", {})
        cols = {}
        for k, v in data.items():
            if _COLUMNS[k] == "f":
                cols[k] = np.array([np.nan if x is None else x for x in v], dtype=float)
            elif _COLUMNS[k] == "t":
                cols[k] = np.array(["NaT" if x is None else x.rstrip("Z") for x in v], dtype="datetime64[s]")
            else:
                cols[k] = v
        return cls(**cols, meta=meta)

    def counts(self):
        """Per kind/variable totals, usable, withheld and per-flag counts (JSON-able)."""
        out = {}
        for key in sorted(set(zip(self.kind.tolist(), self.var.tolist()))):
            sel = (self.kind == key[0]) & (self.var == key[1])
            row = {"total": int(sel.sum()), "usable": int((sel & self.usable).sum()),
                   "stations": int(len(set(self.station[sel])))}
            withheld = int((sel & self.usable & (self.role == "withhold")).sum())
            if withheld:
                row["withheld_usable"] = withheld
            for flag, name in QC_NAMES.items():
                hits = int((sel & ((self.qc & flag) != 0)).sum())
                if hits:
                    row[name] = hits
            out[f"{KIND_NAMES[key[0]]}/{VAR_NAMES[key[1]]}"] = row
        return out


def concat(batches, meta=None):
    batches = [b for b in batches if b is not None]
    if not batches:
        return ObsBatch.empty(meta)
    cols = {name: np.concatenate([getattr(b, name) for b in batches]) for name in _FIELDS}
    merged = dict(meta or {})
    merged.setdefault("sources", [b.meta for b in batches])
    return ObsBatch(**cols, meta=merged)


# -------------------------------------------------------------- observation errors
DEFAULT_OBS_ERRORS = {
    "schema": "a1_da.obs_error_table.v1",
    "provenance": "proofs/a1_da/DESIGN.md section 4 starting values (conservative, NOT tuned)",
    "SONDE": {"T": {"sigma": 1.0, "units": "K"}, "QV": {"sigma_rh": 0.10, "units": "RH fraction -> kg/kg"},
              "U": {"sigma": 2.0, "units": "m/s"}, "V": {"sigma": 2.0, "units": "m/s"}},
    "SURFACE": {"T2": {"sigma": 1.75, "units": "K"}, "Q2": {"sigma_rh": 0.15, "units": "RH fraction -> kg/kg"},
                "U10": {"sigma": 2.5, "units": "m/s"}, "V10": {"sigma": 2.5, "units": "m/s"},
                "PSFC": {"sigma": 100.0, "units": "Pa"}},
    "CLOUD": {"CLDMASK": {"sigma": 0.3, "units": "cloud fraction"}},
    "humidity_note": "sigma_q = sigma_rh * qsat(T, p) at the observation (T, p) [WRF svp, mixing ratio]",
    "surface_representativeness_per_100m": {"T2": 0.5, "Q2": 3.0e-4, "U10": 0.5, "V10": 0.5, "PSFC": 30.0,
                                            "note": "added in quadrature per 100 m |z_station - z_model| "
                                                    "(inflate_surface_errors; opt-in, not tuned)"},
    "q_floor_kg_kg": 2.0e-4,
}


def obs_error_table(overrides=None):
    """Deep copy of the default table, with ``{"SONDE": {"T": {"sigma": 1.2}}}``-style overrides."""
    table = copy.deepcopy(DEFAULT_OBS_ERRORS)
    for kind, entries in (overrides or {}).items():
        if not isinstance(entries, dict):
            table[kind] = entries
            continue
        for var, spec in entries.items():
            table.setdefault(kind, {}).setdefault(var, {}).update(spec)
    json.dumps(table)  # must stay JSON-able
    return table


def error_for(kind, var, table=None, *, t_k=None, p_pa=None):
    """Observation error(s) for one (kind, var); humidity needs T [K] and p [Pa]."""
    table = table or DEFAULT_OBS_ERRORS
    spec = table[KIND_NAMES[kind]][VAR_NAMES[var]]
    if "sigma_rh" in spec:
        if t_k is None or p_pa is None:
            raise ValueError(f"{VAR_NAMES[var]} error needs T and p")
        return np.maximum(spec["sigma_rh"] * qv_saturation(t_k, p_pa), table.get("q_floor_kg_kg", 0.0))
    return float(spec["sigma"])


def inflate_surface_errors(obs, dz_m, table=None):
    """sigma^2 += (c_var * |dz| / 100 m)^2 for SURFACE obs (terrain representativeness; opt-in).

    ``dz_m`` (n_obs,) = z_station - z_model, e.g. ``operators.apply_operators(...).dz_m``; NaN dz leaves
    the error unchanged (such obs are rejected by the operator anyway).
    """
    table = table or DEFAULT_OBS_ERRORS
    coef = table["surface_representativeness_per_100m"]
    dz = np.abs(np.asarray(dz_m, dtype=float))
    if dz.shape != (len(obs),):
        raise ValueError("dz_m must be (n_obs,)")
    c = np.array([coef.get(VAR_NAMES[int(v)], 0.0) if k == SURFACE else 0.0 for k, v in zip(obs.kind, obs.var)])
    extra = np.where(np.isfinite(dz), c * dz / 100.0, 0.0)
    return obs.replace(error=np.sqrt(obs.error ** 2 + extra ** 2))


# ------------------------------------------------------------------------ QC
def gross_check(obs, hx, *, factor=4.0):
    """Background check on usable obs. ``hx`` (N, n_obs). Returns (batch, counters).

    Rejects |y - mean Hx| > factor * sqrt(error^2 + var(Hx, ddof=1)) with QC_GROSS and
    obs whose H(x) is non-finite in any member with QC_NONFINITE_HX.
    """
    hx = np.asarray(hx, dtype=float)
    if hx.ndim != 2 or hx.shape[1] != len(obs):
        raise ValueError(f"hx must be (N, {len(obs)})")
    usable = obs.usable.copy()
    finite = np.isfinite(hx).all(axis=0)
    nonfinite = usable & ~finite
    mean = np.where(finite, np.nanmean(np.where(finite, hx, 0.0), axis=0), np.nan)
    var = np.var(np.where(finite, hx, 0.0), axis=0, ddof=1) if hx.shape[0] > 1 else np.zeros(len(obs))
    if not np.isfinite(obs.error[usable & finite]).all():
        raise ValueError("gross_check needs finite observation errors")
    limit = factor * np.sqrt(obs.error ** 2 + var)
    with np.errstate(invalid="ignore"):
        gross = usable & finite & (np.abs(obs.value - mean) > limit)
    out = obs.with_qc(nonfinite, QC_NONFINITE_HX).with_qc(gross, QC_GROSS)
    counters = {}
    for key in sorted(set(zip(obs.kind[usable].tolist(), obs.var[usable].tolist()))):
        sel = usable & (obs.kind == key[0]) & (obs.var == key[1])
        counters[f"{KIND_NAMES[key[0]]}/{VAR_NAMES[key[1]]}"] = {
            "checked": int(sel.sum()), "gross": int((sel & gross).sum()),
            "nonfinite_hx": int((sel & nonfinite).sum())}
    return out, counters


def spread_check(obs, hx, *, kind=CLOUD, min_spread=0.1):
    """Flag usable obs of ``kind`` whose ensemble std of H(x) is below ``min_spread``."""
    hx = np.asarray(hx, dtype=float)
    spread = np.std(hx, axis=0, ddof=1) if hx.shape[0] > 1 else np.zeros(len(obs))
    low = obs.usable & (obs.kind == kind) & ~(spread >= min_spread)
    return obs.with_qc(low, QC_SPREAD), int(low.sum())


def superob(obs, bins, *, error_mode="keep"):
    """Average usable obs sharing (kind, var, role, bin); ``bins < 0`` rows pass through untouched.

    Super-ob value/lat/lon/height/dt are means, pressure is exp(mean ln p). Error:
    ``keep`` = mean input error (conservative: representativeness errors are correlated)
    or ``sqrt_n`` = mean error / sqrt(n). Inputs are retained in the output flagged
    QC_THINNED so counts stay auditable.
    """
    if error_mode not in ("keep", "sqrt_n"):
        raise ValueError("error_mode must be keep or sqrt_n")
    bins = np.asarray(bins, dtype=np.int64)
    if bins.shape != (len(obs),):
        raise ValueError("bins must be (n_obs,)")
    active = obs.usable & (bins >= 0)
    if not active.any():
        return obs
    role_code = (obs.role == "withhold").astype(np.int64)  # withheld obs never merge with assimilated ones
    keys = np.stack([obs.kind[active].astype(np.int64), obs.var[active].astype(np.int64), role_code[active],
                     bins[active]], axis=1)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    idx = np.flatnonzero(active)

    def mean(x):
        return np.bincount(inverse, weights=x, minlength=len(uniq)) / counts

    lon0 = obs.lon[idx]
    ref = np.zeros(len(uniq))
    ref[inverse] = lon0  # any member of the group as unwrap reference
    lon_rel = (lon0 - ref[inverse] + 180.0) % 360.0 - 180.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lnp = np.log(obs.p_pa[idx])
    stations, sites, launches, ids, times = [], [], [], [], []
    for g in range(len(uniq)):
        members = idx[inverse == g]
        names = set(obs.station[members])
        stations.append(names.pop() if len(names) == 1 else f"superob:{counts[g]}")
        site_set = set(obs.site[members])
        sites.append(site_set.pop() if len(site_set) == 1 else stations[-1])
        launch_set = set(obs.launch[members])
        launches.append(launch_set.pop() if len(launch_set) == 1 else "")
        ids.append(f"superob[{counts[g]}]:{obs.obs_id[members[0]]}")
        tt = obs.time[members]
        times.append(tt[0] if (tt == tt[0]).all() else np.datetime64("NaT", "s"))
    error = mean(obs.error[idx])
    if error_mode == "sqrt_n":
        error = error / np.sqrt(counts)
    new = ObsBatch.from_columns(
        kind=uniq[:, 0].astype(np.int8), var=uniq[:, 1].astype(np.int16), value=mean(obs.value[idx]), error=error,
        lat=mean(obs.lat[idx]), lon=ref + mean(lon_rel), p_pa=np.exp(mean(lnp)), height_m=mean(obs.height_m[idx]),
        dt_s=mean(obs.dt_s[idx]), station=np.array(stations, dtype=object),
        lh_scale=mean(obs.lh_scale[idx]), lv_scale=mean(obs.lv_scale[idx]), obs_id=np.array(ids, dtype=object),
        site=np.array(sites, dtype=object), launch=np.array(launches, dtype=object),
        time=np.array(times, dtype="datetime64[s]"),
        role=np.where(uniq[:, 2] == 1, "withhold", "assimilate").astype(object))
    meta = dict(obs.meta)
    meta["superob"] = {"groups": int(len(uniq)), "inputs": int(active.sum()), "error_mode": error_mode,
                       "max_inputs_per_group": int(counts.max())}
    return concat([obs.with_qc(active, QC_THINNED), new], meta=meta)


def sonde_layer_bins(obs, p_mass_pa):
    """Bin SONDE obs by model layer (ln p midpoints of ``p_mass_pa``, descending) per station.

    Returns bins for :func:`superob`: ≤ 1 super-ob per (station, launch, variable, model layer);
    non-sonde rows get -1. Obs above/below the column get their own end bins.
    """
    p_mass_pa = np.asarray(p_mass_pa, dtype=float)
    if p_mass_pa.ndim != 1 or np.any(np.diff(p_mass_pa) >= 0):
        raise ValueError("p_mass_pa must be a 1-D strictly decreasing column")
    lnp = np.log(p_mass_pa)
    edges = 0.5 * (lnp[1:] + lnp[:-1])  # decreasing
    bins = np.full(len(obs), -1, dtype=np.int64)
    sonde = obs.kind == SONDE
    if sonde.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            layer = np.searchsorted(-edges, -np.log(obs.p_pa[sonde]), side="right")
        launch_key = np.array([f"{a}|{b}" for a, b in zip(obs.station[sonde], obs.launch[sonde])], dtype=object)
        _, station_code = np.unique(launch_key, return_inverse=True)
        bins[sonde] = station_code.ravel() * 1000 + layer
    return bins


# ------------------------------------------------------------- IGRA2 radiosondes
IGRA_STATIONS = {"SPM00060018": {"name": "Guimar (Tenerife)", "wmo": "60018", "elevation_m": 105.0}}


def _igra_int(text):
    text = text.strip()
    return int(text) if text else -9999


def _hypsometric_heights(p_pa, z_m, t_k, qv):
    """Fill missing heights by hypsometric integration (virtual temperature) from known heights.

    Levels are processed in order of decreasing pressure; every reported GPH re-anchors the
    integration; levels below the first known height are integrated downward from it.
    """
    order = np.argsort(-p_pa, kind="stable")
    p, z = p_pa[order], z_m[order].copy()
    lnp = np.log(p)
    good_t = np.isfinite(t_k[order])
    if good_t.sum() < 2:
        return z_m
    tt = np.interp(-lnp, -lnp[good_t], t_k[order][good_t])
    q = qv[order]
    good_q = np.isfinite(q)
    qq = np.interp(-lnp, -lnp[good_q], q[good_q]) if good_q.sum() >= 2 else np.zeros_like(tt)
    tv = tt * (1.0 + 0.608 * qq)
    known = np.flatnonzero(np.isfinite(z))
    if not len(known):
        return z_m
    for k in range(known[0] + 1, len(p)):
        if not np.isfinite(z[k]):
            z[k] = z[k - 1] + R_D * 0.5 * (tv[k] + tv[k - 1]) / G * (lnp[k - 1] - lnp[k])
    for k in range(known[0] - 1, -1, -1):
        z[k] = z[k + 1] - R_D * 0.5 * (tv[k] + tv[k + 1]) / G * (lnp[k] - lnp[k + 1])
    out = np.empty_like(z)
    out[order] = z
    return out


R_EARTH_M = 6370000.0  # WRF sphere


def _drift_positions(lat0, lon0, p_pa, z_m, u, v, etime_s, ascent_mps):
    """Balloon positions per level: x(t) = integral u dt along the ascent (trapezoid), t from IGRA ETIME or
    (z - z_launch) / ascent_mps; winds interpolated in ln p to levels without wind (held at the ends)."""
    order = np.argsort(-p_pa, kind="stable")
    lnp = np.log(p_pa[order])
    good = np.isfinite(u[order]) & np.isfinite(v[order])
    if good.sum() < 1:
        return np.full(len(p_pa), lat0), np.full(len(p_pa), lon0), np.zeros(len(p_pa))
    uu = np.interp(-lnp, -lnp[good], u[order][good])
    vv = np.interp(-lnp, -lnp[good], v[order][good])
    zz = z_m[order]
    tt = etime_s[order].copy()
    est = ~np.isfinite(tt)
    tt[est] = np.maximum(zz[est] - np.nanmin(zz), 0.0) / ascent_mps
    tt = np.maximum.accumulate(np.nan_to_num(tt))
    dt = np.diff(tt, prepend=tt[0])
    x = np.cumsum(0.5 * (uu + np.roll(uu, 1)) * dt * (np.arange(len(tt)) > 0))
    y = np.cumsum(0.5 * (vv + np.roll(vv, 1)) * dt * (np.arange(len(tt)) > 0))
    lat = lat0 + np.rad2deg(y / R_EARTH_M)
    lon = lon0 + np.rad2deg(x / (R_EARTH_M * np.cos(np.deg2rad(lat0))))
    out_lat, out_lon, out_t = np.empty_like(lat), np.empty_like(lon), np.empty_like(tt)
    out_lat[order], out_lon[order], out_t[order] = lat, lon, tt
    return out_lat, out_lon, out_t


def read_igra2(path, station="SPM00060018", valid_time=None, *, analysis_time=None, error_table=None,
               include_surface=True, drift=False, ascent_mps=5.0, level_time=False):
    """One IGRA2 v2 sounding (nominal ``valid_time``) -> ObsBatch of T, QV, U, V per level.

    Keeps every pressure level with data (mandatory + significant; the significant levels
    resolve the trade-wind inversion). IGRA2 data columns (1-based): LVLTYP1 1, LVLTYP2 2,
    ETIME 4-8, PRESS 10-15 [Pa], GPH 17-21 [m], TEMP 23-27 [0.1 C], RH 29-33 [0.1 %],
    DPDP 35-39 [0.1 C], WDIR 41-45 [deg], WSPD 47-51 [0.1 m/s]; -9999 missing,
    -8888 removed by IGRA QC (counted as source rejects). Non-pressure levels are skipped
    (counted). T = TEMP + 273.15. QV from Td = T - DPDP (or reported RH) and p (module
    docstring formula). U/V earth-relative from WDIR/WSPD. Heights = GPH where reported,
    otherwise hypsometric (virtual temperature) from the reported heights. Positions are
    the launch site by default (DESIGN section 4); ``drift=True`` moves every level downwind by
    integrating the earth winds over the ascent time (IGRA ETIME when reported, else
    (z - z_launch) / ``ascent_mps``), and ``meta['drift']`` records the maximum displacement below
    500 hPa and overall. ``dt_s`` uses the nominal time; the release time is recorded in ``meta``.
    ``level_time=True`` (for 4D windows) sets ``time``/``dt_s`` per level = release time + ascent time
    (ETIME or (z - z_launch) / ``ascent_mps``); it requires a reported release time.
    """
    if valid_time is None:
        raise ValueError("valid_time (nominal sounding time) is required")
    when = to_datetime64(valid_time)
    t_an = to_datetime64(analysis_time) if analysis_time is not None else when
    target = str(when.astype("datetime64[h]"))  # YYYY-MM-DDTHH
    y, mo, d, h = int(target[:4]), int(target[5:7]), int(target[8:10]), int(target[11:13])
    header, rows = None, []
    with open(path, encoding="ascii", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                continue
            if (line[1:12].strip() == station and int(line[13:17]) == y and int(line[18:20]) == mo
                    and int(line[21:23]) == d and int(line[24:26]) == h):
                header = line
                nlev = int(line[32:36])
                rows = [next(handle) for _ in range(nlev)]
                break
    if header is None:
        raise LookupError(f"no IGRA2 sounding {station} {target}Z in {path}")
    lat, lon = int(header[55:62]) / 1e4, int(header[63:71]) / 1e4
    reltime = header[27:31].strip()
    release = None
    if reltime and reltime[:2] != "99":
        # RELTIME is the actual launch clock time: a 00Z sounding released at 2315 left the PREVIOUS day.
        rel = np.datetime64(f"{target[:10]}T{reltime[:2]}:{reltime[2:] if reltime[2:] != '99' else '00'}:00", "s")
        if rel - when > np.timedelta64(12, "h"):
            rel -= np.timedelta64(1, "D")
        elif when - rel > np.timedelta64(12, "h"):
            rel += np.timedelta64(1, "D")
        release = f"{rel}Z"
    lvl = {k: [] for k in ("t1", "t2", "p", "z", "t", "rh", "dpd", "wd", "ws", "et")}
    skipped = {"non_pressure_level": 0, "igra_qc_removed": 0}
    for row in rows:
        p = _igra_int(row[9:15])
        if p <= 0:
            skipped["non_pressure_level"] += 1
            continue
        vals = [_igra_int(row[16:21]), _igra_int(row[22:27]), _igra_int(row[28:33]), _igra_int(row[34:39]),
                _igra_int(row[40:45]), _igra_int(row[46:51])]
        skipped["igra_qc_removed"] += sum(v == -8888 for v in vals)
        z, t, rh, dpd, wd, ws = (np.nan if v in (-9999, -8888) else float(v) for v in vals)
        et = _igra_int(row[3:8])
        et_s = np.nan if et in (-9999, -8888) else float(60 * (et // 100) + et % 100)  # mmmss
        for key, value in zip(lvl, (int(row[0]), int(row[1]), float(p), z, t / 10 + 273.15, rh / 1000,
                                    dpd / 10, wd, ws / 10, et_s)):
            lvl[key].append(value)
    arr = {k: np.asarray(v, dtype=float) for k, v in lvl.items()}
    surface = arr["t2"] == 1
    keep = np.ones(len(arr["p"]), bool) if include_surface else ~surface
    arr = {k: v[keep] for k, v in arr.items()}
    p, tk = arr["p"], arr["t"]
    qv = np.full_like(p, np.nan)
    from_dpd = np.isfinite(tk) & np.isfinite(arr["dpd"])
    qv[from_dpd] = qv_from_dewpoint(tk[from_dpd] - arr["dpd"][from_dpd], p[from_dpd])
    from_rh = ~from_dpd & np.isfinite(tk) & np.isfinite(arr["rh"])
    qv[from_rh] = qv_from_rh(arr["rh"][from_rh], tk[from_rh], p[from_rh])
    z = _hypsometric_heights(p, arr["z"], tk, qv)
    u, v = wind_components(arr["ws"], arr["wd"])
    lat_lv, lon_lv = np.full(len(p), lat), np.full(len(p), lon)
    drift_meta = "ignored (launch site)"
    t_lv = None
    if drift or level_time:
        d_lat, d_lon, t_lv = _drift_positions(lat, lon, p, z, u, v, arr["et"], ascent_mps)
    if drift:
        lat_lv, lon_lv = d_lat, d_lon
        disp = _haversine_km(lat, lon, lat_lv, lon_lv)
        drift_meta = {"method": "trapezoid wind integration", "ascent_mps": ascent_mps,
                      "etime_levels": int(np.isfinite(arr["et"]).sum()),
                      "max_km_below_500hPa": float(disp[p >= 5.0e4].max(initial=0.0)),
                      "max_km": float(disp.max(initial=0.0)), "max_ascent_s": float(np.nanmax(t_lv))}
    table = error_table or DEFAULT_OBS_ERRORS
    dt = float((when - t_an) / np.timedelta64(1, "s"))
    time_lv = np.full(len(p), when)
    if level_time:
        if release is None:
            raise ValueError("level_time needs a reported release time (RELTIME)")
        rel = np.datetime64(release.rstrip("Z"), "s")
        time_lv = rel + np.round(t_lv).astype("timedelta64[s]")
    dt_lv = (time_lv - t_an) / np.timedelta64(1, "s")
    launch_id = f"{station}@{target}Z"
    parts = []
    for code, values, errors in (
            (T, tk, error_for(SONDE, T, table)),
            (QV, qv, None),
            (U, u, error_for(SONDE, U, table)),
            (V, v, error_for(SONDE, V, table))):
        sel = np.isfinite(values)
        if not sel.any():
            continue
        if code == QV:
            errors = error_for(SONDE, QV, table, t_k=tk[sel], p_pa=p[sel])
        parts.append(ObsBatch.from_columns(
            var=np.full(sel.sum(), code, np.int16), value=values[sel], error=errors, lat=lat_lv[sel],
            lon=lon_lv[sel], p_pa=p[sel],
            height_m=z[sel], dt_s=dt_lv[sel], station=f"igra:{station}", site=f"igra:{station}", launch=launch_id,
            time=time_lv[sel], averaging="instant",
            obs_id=np.array([f"{launch_id}/{VAR_NAMES[code]}/{n}" for n in np.flatnonzero(sel)], dtype=object)))
    meta = {"source": "IGRA2 v2", "path": str(path), "station": station, "nominal_time": target + ":00:00Z",
            "release_time": release, "launch_lat": lat, "launch_lon": lon, "levels": int(len(p)),
            "surface_level_included": bool(include_surface and surface.any()), "skipped": skipped,
            "qv_formula": "e=es(T-DPD) WRF svp; qv=ep_2*e/(p-e)", "height": "GPH else hypsometric Tv",
            "drift": drift_meta,
            "time_convention": ("per level: release + ascent time" if level_time else "dt_s from nominal time")}
    return concat(parts, meta=meta)


# ------------------------------------------------------------- surface stations
DEFAULT_NETWORKS = ("aemet", "grafcan", "metar", "iac")
NO_WIND_NETWORKS = ("iac",)                    # sheltered telescope masts (DATA_FINDINGS #4)
NETWORK_PRIORITY = ("aemet", "metar", "grafcan", "iac", "cabildo", "puertos")
# A-E prospective-contract protected stations (never read their values) + their aliases.
PROTECTED_STATIONS = frozenset({
    "cabezo:CABEZO_SURF", "windguru:14924", "aemet:C429I", "grafcan:0047", "grafcan:grafcan_0047",
    "elmedano:ELMEDANO_T1_DESHIELDED", "elmedano:ELMEDANO_T2_ONWATER", "elmedano:T1_DESHIELDED",
    "elmedano:T2_ONWATER", "wu:elmedano_cabezo_station_wmr928_min5",
    "metar:60025"})                             # TFS METAR = alias of aemet:C429I (conservative)
PROTECTED_SOURCES = frozenset({"cabezo", "elmedano"})
# Known coordinate errors (shared_truth/DATA_FINDINGS_opus_v1.md #1): La Rambleta.
COORDINATE_FIXES = {"volcano:VOLCANO_TEIDE_CABLE_CAR": (28.2708, -16.6394)}
# Alias / co-location groups (DATA_FINDINGS #2); generic 2 km / 50 m rule catches the rest.
ALIAS_GROUPS = (
    ("aemet:C430E", "metar:60010", "iac:gong", "iac:stella", "iac:speculoos", "iac:song"),
    ("aemet:C447A", "metar:60015"),
    ("aemet:C449C", "metar:60020", "grafcan:0056"),
)


def is_protected(station_id):
    station_id = str(station_id)
    return station_id in PROTECTED_STATIONS or station_id.split(":", 1)[0] in PROTECTED_SOURCES


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi, dlmb = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def site_of(station_id, alias_groups=ALIAS_GROUPS):
    """Alias-group site id (``site:<first member>``) or the station id itself."""
    for members in alias_groups:
        if station_id in members:
            return f"site:{members[0]}"
    return str(station_id)


def dedup_colocated(obs, *, radius_km=2.0, dz_m=50.0, priority=NETWORK_PRIORITY):
    """Flag QC_DUPLICATE on SURFACE obs of a co-located lower-priority station (same var, same time).

    Two stations are one site when they share ``site`` (alias groups) or lie within ``radius_km``
    and ``dz_m``. Priority: network order, then station id (deterministic).
    """
    rank = {net: r for r, net in enumerate(priority)}
    dup = np.zeros(len(obs), bool)
    sel = np.flatnonzero(obs.usable & (obs.kind == SURFACE))
    keys = {}
    for i in sel:
        keys.setdefault((int(obs.var[i]), float(obs.dt_s[i])), []).append(i)
    for members in keys.values():
        members.sort(key=lambda i: (rank.get(str(obs.station[i]).split(":", 1)[0], 99), str(obs.station[i])))
        kept = []
        for i in members:
            same = False
            for k in kept:
                if obs.station[i] == obs.station[k] or obs.site[i] == obs.site[k]:
                    same = True
                elif (_haversine_km(obs.lat[i], obs.lon[i], obs.lat[k], obs.lon[k]) <= radius_km
                      and (not np.isfinite(obs.height_m[i]) or not np.isfinite(obs.height_m[k])
                           or abs(obs.height_m[i] - obs.height_m[k]) <= dz_m)):
                    same = True
                if same:
                    break
            if same:
                dup[i] = True
            else:
                kept.append(i)
    return obs.with_qc(dup, QC_DUPLICATE), int(dup.sum())


# Native ObsDB variable names -> role (see the extract receipt for the actual inventory).
STATION_VARIABLES = {
    "t2_c": "t_c", "temperature_c": "t_c", "air_temperature_c": "t_c", "temp_c": "t_c",
    "rh_pct": "rh_pct", "relative_humidity_pct": "rh_pct", "humidity_pct": "rh_pct",
    "dewpoint_c": "td_c",
    "wind_speed_ms": "ws", "wind_dir_deg": "wd",
    "pressure_hpa": "p_hpa", "station_pressure_hpa": "p_hpa", "pres_hpa": "p_hpa",
}


def _load_station_source(source):
    """Directory (stations.parquet + observations.parquet) or ONE joined parquet with lat/lon/elevation_m."""
    import pandas as pd

    source = Path(source)
    if source.is_dir():
        stations = pd.read_parquet(source / "stations.parquet")
        obs = pd.read_parquet(source / "observations.parquet")
        receipt = source / "EXTRACT_RECEIPT.json"
    else:
        obs = pd.read_parquet(source)
        cols = [c for c in ("station_id", "lat", "lon", "elevation_m", "elevation_source") if c in obs]
        stations = obs[cols].drop_duplicates("station_id")
        receipt = source.with_suffix(".receipt.json")
    meta = json.loads(receipt.read_text()).get("snapshot", {}) if receipt.exists() else None
    return stations, obs, meta


def read_station_obs(source, t0, t1, bbox=None, *, analysis_time=None, networks=DEFAULT_NETWORKS,
                     no_wind_networks=NO_WIND_NETWORKS, hourly_tolerance_s=600, error_table=None,
                     variables=None, dedup=True, include_pressure=False,
                     reject_altitude_flags=("check_gt30m", "offshore")):
    """Hourly station obs in [t0, t1] -> ObsBatch of T2, Q2, U10, V10 (+ PSFC only if ``include_pressure``).

    ``source`` = directory with ``stations.parquet`` (station_id, lat, lon, elevation_m,
    elevation_source) + ``observations.parquet`` (station_id, ts_utc, variable, value, qc_flag), or
    ONE parquet with both column sets, written by ``proofs/a1_da/obs/extract_station_obs.py`` from
    the ALISIOS ObsDB snapshot. For every full hour in [t0, t1] (str/datetime/POSIX seconds) and
    station the record nearest to the hour within ``hourly_tolerance_s`` is used (instantaneous
    sampling, like history frames). ObsDB qc_flag != 0 rows are dropped and counted. Stations
    without a finite elevation, or whose catalog ``altitude_flag`` is in ``reject_altitude_flags``
    (declared height disagrees with the DEM, offshore), are rejected (QC_SOURCE, counted). Q2 from T + RH (or dewpoint) with the
    station pressure if reported, else the ICAO standard atmosphere at the station height
    (disclosed). U10/V10 earth-relative from speed/direction at the same timestamp; sheltered
    networks (IAC) give no wind. Pressure is NOT used by default (DESIGN_v2: MSLP != PSFC).
    Protected/holdout stations are refused even if present in the source.
    """
    import pandas as pd

    t0, t1 = to_datetime64(t0), to_datetime64(t1)
    t_an = to_datetime64(analysis_time) if analysis_time is not None else t0
    variables = dict(variables or STATION_VARIABLES)
    stations, obs, snapshot = _load_station_source(source)
    meta = {"source": str(source), "t0": str(t0) + "Z", "t1": str(t1) + "Z", "networks": list(networks),
            "no_wind_networks": list(no_wind_networks), "hourly_tolerance_s": hourly_tolerance_s,
            "include_pressure": include_pressure, "extract_snapshot": snapshot, "counters": {}}
    counters = meta["counters"]
    protected = obs.station_id.map(is_protected)
    if protected.any():
        raise ValueError(f"source contains protected stations {sorted(set(obs.station_id[protected]))}")
    obs = obs[obs.station_id.str.split(":").str[0].isin(networks)]
    counters["rows_in_networks"] = int(len(obs))
    ts = pd.to_datetime(obs.ts_utc, utc=True) if len(obs) else pd.to_datetime(pd.Series([], dtype=str), utc=True)
    obs = obs.assign(ts=ts.dt.tz_localize(None))
    lo = pd.Timestamp(t0) - pd.Timedelta(seconds=hourly_tolerance_s)
    hi = pd.Timestamp(t1) + pd.Timedelta(seconds=hourly_tolerance_s)
    obs = obs[(obs.ts >= lo) & (obs.ts <= hi)]
    if (obs.ts >= pd.Timestamp("2026-07-01")).any():
        raise ValueError("ALISIOS data rule: observations valid >= 2026-07-01 are locked")
    flagged = obs.qc_flag.fillna(0).astype(float) != 0
    counters["obsdb_qc_flag_dropped"] = int(flagged.sum())
    obs = obs[~flagged & obs.variable.isin(variables)]
    obs = obs.assign(role=obs.variable.map(variables))
    nonfinite = ~np.isfinite(obs.value.astype(float))
    counters["nonfinite_dropped"] = int(nonfinite.sum())
    obs = obs[~nonfinite]
    st = stations.drop_duplicates("station_id").set_index("station_id")
    hours = np.arange(t0.astype("datetime64[h]"), t1.astype("datetime64[h]") + np.timedelta64(1, "h"),
                      np.timedelta64(1, "h"))
    hours = [pd.Timestamp(h) for h in hours if t0 <= h.astype("datetime64[s]") <= t1]
    table = error_table or DEFAULT_OBS_ERRORS
    rows = {k: [] for k in ("var", "value", "error", "lat", "lon", "height_m", "dt_s", "station", "time", "qc")}
    skipped = {"no_station_meta": 0, "outside_bbox": 0, "no_record_near_hour": 0, "wind_pair_missing": 0,
               "rh_out_of_range": 0, "sheltered_wind_excluded": 0, "q2_std_atmosphere_pressure": 0,
               "pressure_not_used": 0, "unknown_elevation_rejected": 0}
    flagged_alt = set()
    if "altitude_flag" in st.columns:
        flagged_alt = set(st.index[st.altitude_flag.astype(str).isin(reject_altitude_flags)])
    meta["altitude_flag_rejected_stations"] = sorted(flagged_alt)
    for sid, part in obs.groupby("station_id"):
        if sid not in st.index:
            skipped["no_station_meta"] += 1
            continue
        lat, lon = float(st.loc[sid, "lat"]), float(st.loc[sid, "lon"])
        elev = float(st.loc[sid, "elevation_m"]) if "elevation_m" in st.columns else np.nan
        lat, lon = COORDINATE_FIXES.get(sid, (lat, lon))
        if bbox is not None and not (bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]):
            skipped["outside_bbox"] += 1
            continue
        network = sid.split(":", 1)[0]
        flag = QC_OK if (np.isfinite(elev) and sid not in flagged_alt) else QC_SOURCE
        for hour in hours:
            near = part[(part.ts - hour).abs() <= pd.Timedelta(seconds=hourly_tolerance_s)]
            if near.empty:
                skipped["no_record_near_hour"] += 1
                continue
            when = np.datetime64(hour.to_datetime64(), "s")
            dt = float((when - t_an) / np.timedelta64(1, "s"))
            pick = {}
            for role, grp in near.groupby("role"):
                best = grp.iloc[np.argmin((grp.ts - hour).abs().to_numpy())]
                pick[role] = (best.ts, float(best.value))

            def add(code, value, error):
                for key, val in (("var", code), ("value", value), ("error", error), ("lat", lat), ("lon", lon),
                                 ("height_m", elev), ("dt_s", dt), ("station", sid), ("time", when), ("qc", flag)):
                    rows[key].append(val)
                skipped["unknown_elevation_rejected"] += int(flag != QC_OK)

            p_stn = pick["p_hpa"][1] * 100.0 if "p_hpa" in pick else None
            if p_stn is not None:
                if include_pressure:
                    add(PSFC, p_stn, error_for(SURFACE, PSFC, table))
                else:
                    skipped["pressure_not_used"] += 1
            if "t_c" in pick:
                tk = pick["t_c"][1] + 273.15
                add(T2, tk, error_for(SURFACE, T2, table))
                p_use = p_stn if p_stn is not None else float(std_atmosphere_pressure(elev if np.isfinite(elev) else 0.0))
                q = None
                if "td_c" in pick:
                    q = float(qv_from_dewpoint(pick["td_c"][1] + 273.15, p_use))
                elif "rh_pct" in pick:
                    rh = pick["rh_pct"][1]
                    if 0.0 < rh <= 105.0:
                        q = float(qv_from_rh(min(rh, 100.0) / 100.0, tk, p_use))
                    else:
                        skipped["rh_out_of_range"] += 1
                if q is not None:
                    skipped["q2_std_atmosphere_pressure"] += int(p_stn is None)
                    add(Q2, q, float(error_for(SURFACE, Q2, table, t_k=tk, p_pa=p_use)))
            if "ws" in pick or "wd" in pick:
                if network in no_wind_networks:
                    skipped["sheltered_wind_excluded"] += 1
                elif "ws" in pick and "wd" in pick and pick["ws"][0] == pick["wd"][0]:
                    ue, ve = wind_components(pick["ws"][1], pick["wd"][1])
                    add(U10, float(ue), error_for(SURFACE, U10, table))
                    add(V10, float(ve), error_for(SURFACE, V10, table))
                else:
                    skipped["wind_pair_missing"] += 1
    counters.update(skipped)
    station_ids = np.asarray(rows["station"], dtype=object)
    batch = ObsBatch.from_columns(
        var=np.asarray(rows["var"], np.int16), value=np.asarray(rows["value"]), error=np.asarray(rows["error"]),
        lat=np.asarray(rows["lat"]), lon=np.asarray(rows["lon"]), height_m=np.asarray(rows["height_m"]),
        dt_s=np.asarray(rows["dt_s"]), station=station_ids, qc=np.asarray(rows["qc"], np.int32),
        time=np.asarray(rows["time"], dtype="datetime64[s]"),
        site=np.array([site_of(x) for x in station_ids], dtype=object), meta=meta)
    meta["station_location_source"] = ("ObsDB station catalog lat/lon; elevation = altitude_verified_m "
                                       "else altitude_m (see extract receipt); COORDINATE_FIXES applied")
    meta["q2_formula"] = "qv_from_rh(RH, T2, p_station or ICAO std atmosphere(z)); dewpoint if present"
    meta["averaging"] = "instant (record nearest to the full hour; ObsDB native sampling not re-averaged)"
    if dedup and len(batch):
        batch, n_dup = dedup_colocated(batch)
        counters["colocated_duplicates"] = n_dup
    return batch


# ------------------------------------------------------------- MSG cloud mask
def read_msg_clm(npz, valid_time, grid, *, analysis_time=None, stride=1, min_pixels=3, max_dt_s=450,
                 error_table=None, role="withhold"):
    """MSG CLM (corrected npz) -> binary cloud obs super-obbed on ``stride`` boxes of the mass grid.

    ``grid`` = dict with XLAT, XLONG (2-D mass grid) or a ``operators.GridLocator``. The slot
    nearest to ``valid_time`` (within ``max_dt_s`` of the slot's nominal start; SEVIRI scans
    the Canaries ~10 min after it) is used. Pixels with code 255 (invalid) are counted and
    dropped; pixels outside the grid are dropped. Each box with >= ``min_pixels`` valid
    pixels gives one CLDMASK obs = cloudy fraction in [0, 1] at the pixel centroid.
    EXPERIMENTAL: the source receipt says training_authority False / not truth-admitted. Default
    ``role="withhold"``: cloud is a validation diagnostic only (DESIGN_v2 decision 11).
    """
    from .operators import GridLocator

    data = np.load(npz, allow_pickle=False)
    when = to_datetime64(valid_time)
    t_an = to_datetime64(analysis_time) if analysis_time is not None else when
    times = data["time"].astype("datetime64[ns]").astype("datetime64[s]")
    off = np.abs((times - when) / np.timedelta64(1, "s"))
    slot = int(np.argmin(off))
    if off[slot] > max_dt_s:
        raise LookupError(f"no MSG slot within {max_dt_s} s of {when}")
    codes = data["cloud_binary"][slot].astype(np.int16)
    lat = data["latitude"].astype(float)
    lon = data["longitude"].astype(float)
    locator = grid if isinstance(grid, GridLocator) else GridLocator(grid["XLAT"], grid["XLONG"])
    valid = codes != 255
    ii, jj, inside = locator.locate(lat, lon)
    use = valid & inside
    ny, nx = locator.shape
    bi = np.clip(np.floor(ii[use] + 0.5).astype(int), 0, nx - 1) // stride
    bj = np.clip(np.floor(jj[use] + 0.5).astype(int), 0, ny - 1) // stride
    nbx = (nx - 1) // stride + 1
    box = bj * nbx + bi
    uniq, inverse, counts = np.unique(box, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    frac = np.bincount(inverse, weights=(codes[use] == 1).astype(float), minlength=len(uniq)) / counts
    blat = np.bincount(inverse, weights=lat[use], minlength=len(uniq)) / counts
    blon = np.bincount(inverse, weights=lon[use], minlength=len(uniq)) / counts
    keep = counts >= min_pixels
    table = error_table or DEFAULT_OBS_ERRORS
    meta = {"source": "MSG CLM corrected npz (EXPERIMENTAL, not truth-admitted)", "path": str(npz),
            "slot_time": str(times[slot]) + "Z", "slot_offset_s": float(off[slot]),
            "sensor": str(data["sensor_id"][slot]) if "sensor_id" in data.files else None,
            "pixels_total": int(len(codes)), "pixels_invalid": int((~valid).sum()),
            "pixels_outside_grid": int((valid & ~inside).sum()), "pixels_used": int(use.sum()),
            "stride": stride, "min_pixels": min_pixels, "boxes": int(len(uniq)),
            "boxes_below_min_pixels": int((~keep).sum()),
            "value": "cloudy fraction of valid pixels in the box (code 1 = cloud)"}
    dt = float((times[slot] - t_an) / np.timedelta64(1, "s"))
    meta["role"] = role
    return ObsBatch.from_columns(var=np.full(keep.sum(), CLDMASK, np.int16), value=frac[keep],
                                 error=error_for(CLOUD, CLDMASK, table), lat=blat[keep], lon=blon[keep],
                                 dt_s=dt, station=np.array([f"msg:box{b}" for b in uniq[keep]], dtype=object),
                                 time=times[slot], averaging="instant", role=role, meta=meta)
