"""Trade-wind inversion (TWI) and moist-layer diagnostics for soundings and model columns (lane a1-obs).

The rules are EXACTLY those of the shared ALISIOS truth script
``<DATA_ROOT>/alisios/state/manager/runs/core_interp_research_20260930/shared_truth/radiosonde_60018/code/derive_60018.py``
(sha256 1fd7fd21ad20f05b669d6e9056ad29621ab9a88e70070a8c4a8595b8b633d26f):

- parse (lines 22-25): PRESS > 0 else NaN; GPH, TEMP, RH, DPDP, WDIR, WSPD > -8888 else NaN; tenths scaled.
- RH (lines 30-34): reported RH, else Magnus over water from T and DPD: a = 17.625, b = 243.04 degC.
- heights (lines 38-46): levels with pressure, sorted by pressure descending; IGRA GPH where reported, else
  hypsometric from the previous level only (Rd = 287.05, g = 9.80665, virtual temperature neglected, both
  temperatures finite); keep finite height and temperature; sort by height.
- twi (lines 49-76): levels z <= 5000 m (first of duplicate heights); < 5 levels -> no result; warming steps
  dT >= 0 with dz > 0; consecutive warm steps join when the step starts < 100 m above the current segment end;
  TWI = first segment with 300 <= base <= 4000 m and depth >= 50 m: base, top, strength = T_top - T_base, depth,
  RH and T at base/top; moist layer = levels below base + 1 m with z >= 0 and RH >= 90 %: lowest/highest height;
  max RH below the base.

Heights are MSL (IGRA GPH; the thresholds 300/4000/5000 m are MSL in the shared truth). Model columns use the
mass-level geopotential height 0.5 (PH + PHB)(k) + (k+1) / g (WRF g = 9.81) at the operator location (bilinear,
``operators.GridLocator``); AGL heights are reported alongside but the rules run on MSL so sonde and model are
judged identically. Model RH uses the SAME Magnus constants (es = 610.94 Pa exp(a T / (b + T)), Alduchov &
Eskridge prefactor) with e = qv p / (ep_2 + qv), ep_2 = 287/461.6, so a model column at the observed T/Td gives
the observed RH.
"""
from __future__ import annotations

import numpy as np

DERIVE_60018 = {
    "path": "<DATA_ROOT>/alisios/state/manager/runs/core_interp_research_20260930/shared_truth/radiosonde_60018/"
            "code/derive_60018.py",
    "sha256": "1fd7fd21ad20f05b669d6e9056ad29621ab9a88e70070a8c4a8595b8b633d26f",
    "lines": {"parse": "22-25", "rh": "30-34", "heights": "38-46", "twi": "49-76"},
}
RD_DERIVE, G_DERIVE = 287.05, 9.80665          # derive_60018.py line 9
MAGNUS_A, MAGNUS_B = 17.625, 243.04             # derive_60018.py line 31
MAGNUS_E0_PA = 610.94                           # Alduchov & Eskridge 1996 (cancels for RH from DPD)
EP_2 = 287.0 / 461.6                            # WRF r_d / r_v
G_WRF = 9.81
TWI_FIELDS = ("n_levels_0_5km", "twi_found", "twi_base_m", "twi_top_m", "twi_strength_k", "twi_depth_m",
              "rh_base_pct", "rh_top_pct", "t_base_c", "t_top_c", "moist_base_m", "moist_top_m", "max_rh_below_twi")


# ----------------------------------------------------------------------------- humidity
def rh_from_dewpoint_depression(t_c, dpd_c):
    """derive_60018.py lines 31-32 [%]."""
    t_c, td = np.asarray(t_c, float), np.asarray(t_c, float) - np.asarray(dpd_c, float)
    return 100 * np.exp(MAGNUS_A * td / (MAGNUS_B + td) - MAGNUS_A * t_c / (MAGNUS_B + t_c))


def rh_from_mixing_ratio(qv, t_c, p_pa):
    """Model RH [%] with the derive_60018 Magnus constants (over water)."""
    qv, p_pa, t_c = np.asarray(qv, float), np.asarray(p_pa, float), np.asarray(t_c, float)
    e = qv * p_pa / (EP_2 + qv)
    return 100 * e / (MAGNUS_E0_PA * np.exp(MAGNUS_A * t_c / (MAGNUS_B + t_c)))


# ----------------------------------------------------------------------------- soundings
def _igra_levels(lines):
    """derive_60018.py parse() lines 22-25 + RH lines 30-34 for one sounding's data lines."""
    import pandas as pd

    rows = []
    for line in lines:
        p = int(line[9:15]); z = int(line[16:21]); t = int(line[22:27]); rh = int(line[28:33]); dp = int(line[34:39])
        wd = int(line[40:45]); ws = int(line[46:51])
        rows.append((int(line[0]), int(line[1]), p if p > 0 else np.nan, z if z > -8888 else np.nan,
                     t / 10 if t > -8888 else np.nan, rh / 10 if rh > -8888 else np.nan,
                     dp / 10 if dp > -8888 else np.nan, wd if wd > -8888 else np.nan, ws / 10 if ws > -8888 else np.nan))
    lv = pd.DataFrame(rows, columns=["lvltyp1", "lvltyp2", "p_pa", "gph_m", "t_c", "rh_pct", "dpd_c", "wdir", "wspd"])
    num = ["p_pa", "gph_m", "t_c", "rh_pct", "dpd_c", "wdir", "wspd"]
    lv[num] = lv[num].astype(float)   # the whole-file frame in derive_60018 is float (NaNs somewhere)
    td = lv.t_c - lv.dpd_c
    a, b = MAGNUS_A, MAGNUS_B
    rh_dpd = 100 * np.exp(a * td / (b + td) - a * lv.t_c / (b + lv.t_c))
    lv["rh_source"] = np.where(lv.rh_pct.notna(), "reported", np.where(rh_dpd.notna(), "from_dpd", "none"))
    lv["rh_pct"] = lv.rh_pct.fillna(rh_dpd)
    return lv


def iter_igra2_soundings(path, station=None, start=None, end=None):
    """Yield (header dict, levels DataFrame) for each sounding (optionally station / [start, end] nominal time)."""
    import pandas as pd

    lo = pd.Timestamp(start) if start is not None else None
    hi = pd.Timestamp(end) if end is not None else None
    sid = -1  # derive_60018 numbers every header of the file (sid)
    with open(path, encoding="ascii", errors="replace") as handle:
        line = handle.readline()
        while line:
            if not line.startswith("#"):
                line = handle.readline()
                continue
            sid += 1
            head = {"sid": sid, "station": line[1:12].strip(), "year": int(line[13:17]), "month": int(line[18:20]),
                    "day": int(line[21:23]), "hour": int(line[24:26]), "nlev": int(line[32:36]),
                    "lat": int(line[55:62]) / 1e4, "lon": int(line[63:71]) / 1e4}
            head["time"] = pd.Timestamp(year=head["year"], month=head["month"], day=head["day"],
                                        hour=head["hour"] if head["hour"] <= 23 else 0)
            body = [handle.readline() for _ in range(head["nlev"])]
            line = handle.readline()
            if station is not None and head["station"] != station:
                continue
            if (lo is not None and head["time"] < lo) or (hi is not None and head["time"] > hi):
                continue
            yield head, _igra_levels(body)


def sounding_profile(levels):
    """derive_60018.py heights() lines 38-46: DataFrame with z_m, t_c, rh_pct (sorted by z)."""
    g_ = levels.dropna(subset=["p_pa"]).sort_values("p_pa", ascending=False)
    z = g_.gph_m.values.astype(float)
    p = g_.p_pa.values
    t = g_.t_c.values + 273.15
    for i in range(1, len(z)):
        if np.isnan(z[i]) and np.isfinite(z[i - 1]) and np.isfinite(t[i]) and np.isfinite(t[i - 1]):
            z[i] = z[i - 1] + RD_DERIVE * 0.5 * (t[i] + t[i - 1]) / G_DERIVE * np.log(p[i - 1] / p[i])
    g_ = g_.assign(z_m=z)
    return g_[np.isfinite(g_.z_m) & np.isfinite(g_.t_c)].sort_values("z_m")


def _twi(profile):
    """derive_60018.py twi() lines 49-76 on a DataFrame with z_m, t_c, rh_pct (sorted by z)."""
    pr = profile[(profile.z_m <= 5000)].drop_duplicates("z_m")
    if len(pr) < 5:
        return {}
    z, t, rh = pr.z_m.values, pr.t_c.values, pr.rh_pct.values
    dz = np.diff(z)
    dt_ = np.diff(t)
    warm = (dt_ >= 0) & (dz > 0)
    res = {"n_levels_0_5km": len(pr)}
    seg = []
    for k in range(len(warm)):
        if warm[k]:
            if seg and (z[k] - z[seg[-1][1]] < 100):
                seg[-1][1] = k + 1
            else:
                seg.append([k, k + 1])
    for a, b in seg:
        if 300 <= z[a] <= 4000 and z[b] - z[a] >= 50:
            res.update(twi_base_m=z[a], twi_top_m=z[b], twi_strength_k=t[b] - t[a], twi_depth_m=z[b] - z[a],
                       rh_base_pct=rh[a], rh_top_pct=rh[b], t_base_c=t[a], t_top_c=t[b])
            below = (z < z[a] + 1) & (z >= 0) & np.isfinite(rh)
            moist = below & (rh >= 90)
            if moist.any():
                res.update(moist_base_m=z[moist].min(), moist_top_m=z[moist].max())
            res["max_rh_below_twi"] = float(np.nanmax(rh[below])) if below.any() else np.nan
            break
    res["twi_found"] = "twi_base_m" in res
    return {k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer,)) else v))
            for k, v in res.items()}


def detect_inversion(z_m, t_c, rh_pct):
    """TWI + moist layer on any column (heights MSL [m], T [degC], RH [%]); {} if < 5 levels <= 5 km."""
    import pandas as pd

    prof = pd.DataFrame({"z_m": np.asarray(z_m, float), "t_c": np.asarray(t_c, float),
                         "rh_pct": np.asarray(rh_pct, float)})
    prof = prof[np.isfinite(prof.z_m) & np.isfinite(prof.t_c)].sort_values("z_m", kind="stable")
    return _twi(prof)


def sounding_inversion(path, station, valid_time):
    """TWI of one IGRA2 sounding (nominal time), identical to the shared derived parquet row."""
    import pandas as pd

    when = pd.Timestamp(valid_time)
    if when.tzinfo is not None:
        when = when.tz_convert("UTC").tz_localize(None)
    for head, levels in iter_igra2_soundings(path, station, when, when):
        out = _twi(sounding_profile(levels))
        out["time"] = str(head["time"])
        out["sid"] = head["sid"]
        return out
    raise LookupError(f"no sounding {station} {when}")


# ----------------------------------------------------------------------------- model columns
def _field(frames, geo, name):
    if name in frames:
        arr = np.asarray(frames[name])
    elif name in geo:
        arr = np.asarray(geo[name])
    else:
        raise KeyError(f"model column needs {name} in frames or geo")
    return arr


def model_columns(frames, geo, lat, lon, *, locator=None):
    """Member columns at (lat, lon) with the operator geometry: dict of (N, nz) arrays + HGT.

    Keys: z_msl, z_agl (m), t_c (degC), qv (kg/kg), p_pa, rh_pct. Needs T (dry theta - 300), QVAPOR, P, PH and
    PB/PHB (frames or geo).
    """
    from . import operators as ops

    locator = locator or ops.GridLocator(geo["XLAT"], geo["XLONG"])
    fi, fj, inside = locator.locate([lat], [lon])
    if not inside[0]:
        raise ValueError("column outside the domain")
    jj, ii, w, _ = ops._stencil(fi, fj, *locator.shape)

    def corners(name):
        arr = _field(frames, geo, name)
        if arr.ndim == 3:
            arr = arr[None]
        return arr[..., jj, ii].astype(np.float64)                                   # (N|1, nz, 4, 1)

    def col(c):
        return np.sum(c * w, axis=-2)[..., 0]                                          # (N, nz)

    p_c = corners("P") + corners("PB")
    lnp = col(np.log(p_c))
    temp = col((corners("T") + 300.0) * (p_c / ops.P0) ** ops.RCP)
    qv = col(corners("QVAPOR"))
    zw_col = col((corners("PH") + corners("PHB")) / G_WRF)
    z = 0.5 * (zw_col[:, 1:] + zw_col[:, :-1])
    hgt = float(np.sum(np.asarray(geo["HGT"], float)[jj, ii] * w, axis=-2)[0])
    p = np.exp(lnp)
    t_c = temp - 273.15
    return {"z_msl": z, "z_agl": z - hgt, "t_c": t_c, "qv": qv, "p_pa": p,
            "rh_pct": rh_from_mixing_ratio(qv, t_c, p), "hgt_m": hgt}


def model_inversion(frames, geo, lat, lon, *, locator=None):
    """TWI per member and on the member-mean column at (lat, lon) + summary (heights MSL, like the sonde)."""
    cols = model_columns(frames, geo, lat, lon, locator=locator)
    members = [detect_inversion(cols["z_msl"][m], cols["t_c"][m], cols["rh_pct"][m]) for m in range(len(cols["t_c"]))]
    mean = detect_inversion(cols["z_msl"].mean(0), cols["t_c"].mean(0), cols["rh_pct"].mean(0))
    found = [m for m in members if m.get("twi_found")]

    def stat(key):
        vals = np.array([m[key] for m in found if key in m], float)
        return {"n": int(len(vals)), "median": float(np.median(vals)) if len(vals) else None,
                "min": float(vals.min()) if len(vals) else None, "max": float(vals.max()) if len(vals) else None}

    return {"members": members, "member_mean_column": mean, "model_hgt_m": cols["hgt_m"],
            "levels_below_5km_msl": int((cols["z_msl"].mean(0) <= 5000).sum()),
            "summary": {"n_members": len(members), "found_fraction": len(found) / max(len(members), 1),
                        "twi_base_m": stat("twi_base_m"), "twi_strength_k": stat("twi_strength_k"),
                        "moist_found_fraction": sum("moist_base_m" in m for m in members) / max(len(members), 1)},
            "rules": DERIVE_60018}
