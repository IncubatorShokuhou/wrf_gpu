"""V5 pre-registered real-case scoring for the A1 DA arms (lane a1-obs). Protocol: proofs/a1_da/V5_PREREG.md.

Host NumPy/pandas only. Inputs are wrfout frames (one deterministic forecast per arm, case and valid time) and
``letkf.obs.ObsBatch`` observations; outputs are per-pair losses and pre-registered verdicts.

Verdicts: IMPROVE / NONINFERIOR / FAIL / INCOMPLETE. INCOMPLETE (never PASS) whenever support is below the
pre-registered minimum, a matched forecast value is non-finite, a pair exists in one arm but not the other, or a
statistic is non-finite (E200: NaN must never pass a gate). Leakage refusals raise ``ValueError``.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from . import obs as ob

VERDICTS = ("IMPROVE", "NONINFERIOR", "FAIL", "INCOMPLETE")
SURFACE_VARS = ("T2", "Q2", "WIND10")       # WIND10 = vector error of earth U10/V10 at the same site and time
SONDE_VARS = ("SONDE_T", "SONDE_QV")
INVERSION_VAR = "TWI_BASE"
CONFIRMATORY_CASES = ("20260407", "20260408", "20260412", "20260418", "20260423", "20260514", "20260515",
                      "20260531", "20260601", "20260602", "20260608", "20260612", "20260613", "20260614",
                      "20260615", "20260617")
DEVELOPMENT_CASES = ("20260227", "20260502", "20260503")   # used in V3 / tuning: never confirmatory


@dataclasses.dataclass(frozen=True)
class V5Protocol:
    """Frozen V5 protocol constants (any change after the first scored arm = dated amendment, claims void)."""

    holdout_seed: str = "a1-v5-holdout-v1"
    holdout_fraction: float = 1.0 / 3.0
    leads_h: tuple = (3, 6, 12, 24)
    sonde_leads_h: tuple = (12, 24)
    sonde_layer_m: tuple = (0.0, 3000.0)
    inversion_miss_penalty_m: float = 1000.0
    n_boot: int = 10000
    seed: int = 20261017
    alpha: float = 0.05
    margin_surface: float = 0.05
    margin_upper_air: float = 0.10
    min_blocks_surface: int = 8
    min_pairs_surface: int = 100
    min_launches_sonde: int = 8
    min_levels_per_launch: int = 3
    min_launches_inversion: int = 6
    cutoff: str = "2026-07-01T00:00:00"
    primary: tuple = (("T2", 6), ("Q2", 6), ("SONDE_QV", 12), ("TWI_BASE", 12))


PROTOCOL = V5Protocol()
LEAD_WINDOWS_H = {3: (1, 3), 6: (4, 6), 12: (7, 12), 24: (13, 24)}   # surface: label -> hourly leads (inclusive)


def lead_window(lead_hours):
    """Pre-registered surface lead window label for an hourly lead after the final analysis (None if outside)."""
    for label, (lo, hi) in LEAD_WINDOWS_H.items():
        if lo <= lead_hours <= hi:
            return label
    return None


# ------------------------------------------------------------------------------ holdouts
def holdout_score(site, seed=PROTOCOL.holdout_seed):
    """Deterministic uniform [0, 1) from sha256(seed|site); independent of list order and of other sites."""
    digest = hashlib.sha256(f"{seed}|{site}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def is_holdout(site, protocol=PROTOCOL):
    return holdout_score(site, protocol.holdout_seed) < protocol.holdout_fraction


def holdout_table(stations, protocol=PROTOCOL):
    """stations: DataFrame station_id, lat, lon, elevation_m (+ island). One row per alias-group SITE."""
    rows = {}
    for rec in stations.itertuples():
        site = ob.site_of(rec.station_id)
        row = rows.setdefault(site, {"site": site, "members": [], "holdout": is_holdout(site, protocol),
                                     "score": round(holdout_score(site, protocol.holdout_seed), 6),
                                     "island": getattr(rec, "island", None),
                                     "elevation_m": float(rec.elevation_m)})
        row["members"].append(rec.station_id)
    return sorted(rows.values(), key=lambda r: r["site"])


def check_no_leakage(scored_obs, assimilated_by_arm, holdout_sites):
    """Refuse when a scored observation, or ANY observation of a holdout site, was assimilated by any arm.

    ``assimilated_by_arm``: arm -> ObsBatch of the observations that arm actually assimilated (its receipt).
    """
    scored = set(map(str, scored_obs.obs_id))
    holdout_sites = set(holdout_sites)
    for arm, batch in assimilated_by_arm.items():
        used = batch.subset(batch.usable & (batch.role == "assimilate"))
        both = scored & set(map(str, used.obs_id))
        if both:
            raise ValueError(f"leakage: arm {arm} assimilated {len(both)} scored observations, e.g. {sorted(both)[:3]}")
        leaked_sites = set(map(str, used.site[used.kind == ob.SURFACE])) & holdout_sites
        if leaked_sites:
            raise ValueError(f"leakage: arm {arm} assimilated holdout sites {sorted(leaked_sites)[:5]}")
    return True


def _check_cutoff(obs, protocol):
    t = obs.time[~np.isnat(obs.time)]
    if len(t) and (t >= np.datetime64(protocol.cutoff)).any():
        raise ValueError("ALISIOS data rule: observations valid >= 2026-07-01 are locked")


# ------------------------------------------------------------------------------ pairs
_COLUMNS = ("arm", "case", "block", "unit", "var", "lead_h", "key", "obs", "fc", "loss")


def _frame_mean(frames):
    """Deterministic forecast: member-axis fields with N > 1 are averaged (ensemble-mean = secondary product);
    N = 1 fields and member-less static fields (e.g. PB/PHB without a member axis) pass through unchanged."""
    from .operators import _FIELD_NDIM

    out = {}
    for k, v in frames.items():
        arr = np.asarray(v)
        nd = _FIELD_NDIM.get(k)
        out[k] = arr.mean(axis=0, keepdims=True) if (nd is not None and arr.ndim == nd + 1 and arr.shape[0] > 1) else arr
    return out


def surface_pairs(obs, frames, geo, *, arm, case, block, lead_h, holdout_sites, protocol=PROTOCOL):
    """Per holdout site-time losses for T2 [K^2], Q2 [(g/kg)^2] and WIND10 [(m/s)^2 vector].

    Operator geometry rejects (outside / |dz| > 150 m / no land node) are identical for every arm on the same
    domain and are dropped and counted; a NaN forecast at a geometry-valid observation stays in the table
    (-> INCOMPLETE downstream).
    """
    import pandas as pd

    from .operators import apply_operators

    _check_cutoff(obs, protocol)
    sel = obs.usable & (obs.kind == ob.SURFACE) & np.isin(obs.site, list(holdout_sites)) & \
        np.isin(obs.var, [ob.T2, ob.Q2, ob.U10, ob.V10])
    sub = obs.subset(sel)
    res = apply_operators(sub, _frame_mean(frames), geo)
    geom = res.reason == 0
    hx = res.hx[0]
    rows, dropped = [], int((~geom).sum())
    stamp = [str(t) if not np.isnat(t) else f"dt{d:+.0f}" for t, d in zip(sub.time, sub.dt_s)]
    for n in np.flatnonzero(geom & np.isin(sub.var, [ob.T2, ob.Q2])):
        scale = 1000.0 if sub.var[n] == ob.Q2 else 1.0
        d = (hx[n] - sub.value[n]) * scale
        rows.append((arm, case, block, sub.site[n], ob.VAR_NAMES[int(sub.var[n])], lead_h,
                     f"{sub.site[n]}|{stamp[n]}", sub.value[n] * scale, hx[n] * scale, d * d))
    wind = {}
    for n in np.flatnonzero(geom & np.isin(sub.var, [ob.U10, ob.V10])):
        wind.setdefault((sub.site[n], stamp[n]), {})[int(sub.var[n])] = n
    for (site, st), comp in wind.items():
        if len(comp) != 2:
            dropped += 1
            continue
        iu, iv = comp[ob.U10], comp[ob.V10]
        du, dv = hx[iu] - sub.value[iu], hx[iv] - sub.value[iv]
        speed_o = float(np.hypot(sub.value[iu], sub.value[iv]))
        speed_f = float(np.hypot(hx[iu], hx[iv]))
        rows.append((arm, case, block, site, "WIND10", lead_h, f"{site}|{st}", speed_o, speed_f, du * du + dv * dv))
    out = pd.DataFrame(rows, columns=_COLUMNS)
    out.attrs["geometry_dropped"] = dropped
    return out


def sonde_pairs(sonde, frames, geo, *, arm, case, block, lead_h, protocol=PROTOCOL):
    """Whole-launch holdout: T [K^2] and QV [(g/kg)^2] losses at sonde levels within the pre-registered MSL layer."""
    import pandas as pd

    from .operators import apply_operators

    _check_cutoff(sonde, protocol)
    lo, hi = protocol.sonde_layer_m
    sel = sonde.usable & (sonde.kind == ob.SONDE) & np.isin(sonde.var, [ob.T, ob.QV]) & \
        (sonde.height_m >= lo) & (sonde.height_m <= hi)
    sub = sonde.subset(sel)
    res = apply_operators(sub, _frame_mean(frames), geo)
    rows = []
    for n in np.flatnonzero(res.reason == 0):
        scale = 1000.0 if sub.var[n] == ob.QV else 1.0
        d = (res.hx[0, n] - sub.value[n]) * scale
        rows.append((arm, case, block, sub.launch[n], "SONDE_T" if sub.var[n] == ob.T else "SONDE_QV", lead_h,
                     str(sub.obs_id[n]), sub.value[n] * scale, res.hx[0, n] * scale, d * d))
    out = pd.DataFrame(rows, columns=_COLUMNS)
    out.attrs["geometry_dropped"] = int((res.reason != 0).sum())
    return out


def inversion_pair(sonde_twi, model_twi, *, arm, case, block, lead_h, launch, protocol=PROTOCOL):
    """|TWI base error| [m]; forecast miss when the sonde has a TWI = pre-registered penalty. Sonde without TWI
    -> no pair (false alarms are reported separately). Inputs are ``letkf.diagnostics`` dicts."""
    import pandas as pd

    if not sonde_twi.get("twi_found"):
        return pd.DataFrame([], columns=_COLUMNS)
    obs_base = float(sonde_twi["twi_base_m"])
    if model_twi.get("twi_found"):
        fc = float(model_twi["twi_base_m"])
        loss = abs(fc - obs_base)
    elif model_twi == {} or model_twi.get("twi_found") is False:
        fc, loss = np.nan, protocol.inversion_miss_penalty_m
    else:
        fc, loss = np.nan, np.nan
    return pd.DataFrame([(arm, case, block, launch, INVERSION_VAR, lead_h, launch, obs_base, fc, loss)],
                        columns=_COLUMNS)


# ------------------------------------------------------------------------------ statistics
def cluster_bootstrap_rel(loss_a, loss_r, blocks, units, *, n_boot, seed):
    """Two-stage cluster bootstrap of rel = (sum loss_a - sum loss_r) / sum loss_r.

    Stage 1 resamples blocks (runs of consecutive case days) with replacement; stage 2 resamples units (sites or
    launches) with replacement inside each drawn block. Paired: both arms use the same draws.
    """
    loss_a, loss_r = np.asarray(loss_a, float), np.asarray(loss_r, float)
    blocks, units = np.asarray(blocks), np.asarray(units)
    rng = np.random.default_rng(seed)
    block_ids = np.unique(blocks)
    groups = {}
    for b in block_ids:
        in_b = np.flatnonzero(blocks == b)
        unit_ids = np.unique(units[in_b])
        groups[b] = [(in_b[units[in_b] == u]) for u in unit_ids]
    # per (block, unit) sums for speed
    sums = {b: (np.array([loss_a[g].sum() for g in gs]), np.array([loss_r[g].sum() for g in gs]))
            for b, gs in groups.items()}
    out = np.empty(n_boot)
    for k in range(n_boot):
        sa = sr = 0.0
        for b in rng.choice(block_ids, size=len(block_ids), replace=True):
            ua, ur = sums[b]
            pick = rng.integers(0, len(ua), size=len(ua))
            sa += ua[pick].sum()
            sr += ur[pick].sum()
        out[k] = (sa - sr) / sr if sr > 0 else np.nan
    return out


def holm(pvalues):
    """Holm step-down adjusted p-values (dict name -> p)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def _minimums(var, protocol):
    if var in SURFACE_VARS:
        return {"blocks": protocol.min_blocks_surface, "pairs": protocol.min_pairs_surface, "units": 1,
                "margin": protocol.margin_surface}
    if var in SONDE_VARS:
        return {"blocks": 1, "pairs": protocol.min_launches_sonde * protocol.min_levels_per_launch,
                "units": protocol.min_launches_sonde, "margin": protocol.margin_upper_air}
    return {"blocks": 1, "pairs": protocol.min_launches_inversion, "units": protocol.min_launches_inversion,
            "margin": protocol.margin_upper_air}


def compare_arms(pairs, arm, ref, *, var, lead_h, kind="superiority", protocol=PROTOCOL):
    """Pre-registered comparison of ``arm`` against ``ref`` for one variable and lead. Returns a JSON-able dict."""
    import pandas as pd

    if kind not in ("superiority", "noninferiority"):
        raise ValueError("kind must be superiority or noninferiority")
    need = _minimums(var, protocol)
    sel = pairs[(pairs["var"] == var) & (pairs["lead_h"] == lead_h)]
    a = sel[sel.arm == arm].set_index(["case", "key"])
    r = sel[sel.arm == ref].set_index(["case", "key"])
    out = {"arm": arm, "ref": ref, "var": var, "lead_h": lead_h, "kind": kind, "verdict": "INCOMPLETE",
           "reasons": [], "margin_rel": need["margin"]}
    if a.index.duplicated().any() or r.index.duplicated().any():
        raise ValueError("duplicate (case, key) pairs inside one arm")
    unmatched = a.index.symmetric_difference(r.index)
    m = a.join(r, how="inner", lsuffix="_a", rsuffix="_r")
    out.update(n_pairs=int(len(m)), n_unmatched=int(len(unmatched)),
               n_blocks=int(m.block_a.nunique()) if len(m) else 0,
               n_units=int(m.unit_a.nunique()) if len(m) else 0)
    if len(unmatched):
        out["reasons"].append(f"{len(unmatched)} pairs present in only one arm")
    if len(m) and (m.block_a != m.block_r).any():
        out["reasons"].append("block labels differ between arms")
    if len(m):
        nonfinite = int((~np.isfinite(m.loss_a.to_numpy(float)) | ~np.isfinite(m.loss_r.to_numpy(float))).sum())
        if nonfinite:
            out["reasons"].append(f"{nonfinite} non-finite matched losses (NaN forecast/obs)")
    if out["n_pairs"] < need["pairs"]:
        out["reasons"].append(f"support {out['n_pairs']} pairs < {need['pairs']}")
    if out["n_blocks"] < need["blocks"]:
        out["reasons"].append(f"support {out['n_blocks']} blocks < {need['blocks']}")
    if out["n_units"] < need["units"]:
        out["reasons"].append(f"support {out['n_units']} units < {need['units']}")
    if out["reasons"]:
        return out
    la, lr = m.loss_a.to_numpy(float), m.loss_r.to_numpy(float)
    if lr.sum() <= 0:
        out["reasons"].append("reference loss is zero (degenerate)")
        return out
    boot = cluster_bootstrap_rel(la, lr, m.block_a.to_numpy(), m.unit_a.to_numpy(), n_boot=protocol.n_boot,
                                 seed=protocol.seed)
    rel = float((la.sum() - lr.sum()) / lr.sum())
    if not (np.isfinite(rel) and np.isfinite(boot).all()):
        out["reasons"].append("non-finite statistic")
        return out
    upper = float(np.quantile(boot, 1 - protocol.alpha))
    lower = float(np.quantile(boot, protocol.alpha))
    p_improve = float((np.sum(boot >= 0) + 1) / (len(boot) + 1))
    out.update(loss_arm=float(la.mean()), loss_ref=float(lr.mean()), rel_delta=rel, ci_lower=lower, ci_upper=upper,
               p_improve=p_improve, n_boot=protocol.n_boot)
    if kind == "superiority" and upper < 0:
        out["verdict"] = "IMPROVE"
    elif upper < need["margin"]:
        out["verdict"] = "NONINFERIOR"
    else:
        out["verdict"] = "FAIL"
    return out


COMPARISONS = (("A2_letkf", "A1_zero_update", "superiority"),     # C1 primary: DA effect beyond the seam
               ("A1_zero_update", "A0_continuous", "noninferiority"),  # C2 safety: cycling seam cost
               ("A2_letkf", "A3_nudging", "superiority"),         # C3 LETKF vs equal-data simple baseline
               ("A2_letkf", "A0_continuous", "superiority"))      # C4 overall


def evaluate(pairs, protocol=PROTOCOL, comparisons=COMPARISONS):
    """All pre-registered comparisons; Holm across the C1 primaries; overall status INCOMPLETE if any primary is."""
    results = []
    variables = list(SURFACE_VARS) + list(SONDE_VARS) + [INVERSION_VAR]
    for arm, ref, kind in comparisons:
        for var in variables:
            leads = protocol.leads_h if var in SURFACE_VARS else protocol.sonde_leads_h
            for lead in leads:
                res = compare_arms(pairs, arm, ref, var=var, lead_h=lead, kind=kind, protocol=protocol)
                res["primary"] = (arm, ref) == (comparisons[0][0], comparisons[0][1]) and (var, lead) in protocol.primary
                results.append(res)
    prim = [r for r in results if r["primary"]]
    complete = [r for r in prim if r["verdict"] != "INCOMPLETE"]
    adjusted = holm({f"{r['var']}@{r['lead_h']}": r["p_improve"] for r in complete}) if complete else {}
    for r in prim:
        key = f"{r['var']}@{r['lead_h']}"
        if key in adjusted:
            r["p_holm"] = adjusted[key]
            if r["verdict"] == "IMPROVE" and adjusted[key] >= protocol.alpha:
                r["verdict"] = "NONINFERIOR" if r["ci_upper"] < r["margin_rel"] else "FAIL"
                r["reasons"].append("IMPROVE not significant after Holm")
    status = "INCOMPLETE" if any(r["verdict"] == "INCOMPLETE" for r in prim) or len(prim) != len(protocol.primary) \
        else ("IMPROVE" if all(r["verdict"] == "IMPROVE" for r in prim) else "MIXED")
    return {"schema": "a1_da.v5_scoring.v1", "protocol": dataclasses.asdict(protocol), "primary_status": status,
            "results": results}


# ------------------------------------------------------------------------------ IO
SURFACE_FIELDS = ("T2", "Q2", "U10", "V10", "PSFC")
UPPER_FIELDS = ("T", "QVAPOR", "U", "V", "P", "PB", "PH", "PHB")
GEO_FIELDS = ("XLAT", "XLONG", "HGT", "COSALPHA", "SINALPHA", "LANDMASK")


def read_wrfout(path, names=SURFACE_FIELDS + UPPER_FIELDS, time_index=0):
    """One wrfout frame -> (frames with a member axis of 1, geo). Refuses masked/fill values (no silent repair)."""
    from gpuwrf.io.netcdf_lock import Dataset

    frames, geo = {}, {}
    with Dataset(path) as ds:
        ds.set_auto_mask(True)
        for n in names:
            if n not in ds.variables:
                continue
            v = ds.variables[n][time_index]
            if np.ma.isMaskedArray(v) and np.ma.getmaskarray(v).any():
                raise ValueError(f"{path}: {n} has masked/fill values")
            frames[n] = np.asarray(v)[None]
        for n in GEO_FIELDS:
            geo[n] = np.asarray(ds.variables[n][time_index], dtype=np.float64)
        geo["DX"] = float(ds.getncattr("DX"))
        times = ds.variables["Times"][time_index]
        geo["Times"] = b"".join(times).decode() if times.dtype.kind == "S" else str(times)
    return frames, geo


def wrfout_inventory(paths, *, tolerance_s=60):
    """Map nominal hour (datetime64[s]) -> path from the files' Times. d01 history sits at hh:00:18/hh:00:36
    (dt = 54 s); times within ``tolerance_s`` of the full hour are assigned to it (disclosed), others refused."""
    from gpuwrf.io.netcdf_lock import Dataset

    out = {}
    for path in paths:
        with Dataset(path) as ds:
            raw = ds.variables["Times"][0]
            stamp = b"".join(raw).decode() if raw.dtype.kind == "S" else str(raw)
        t = np.datetime64(stamp.replace("_", "T"), "s")
        hour = (t + np.timedelta64(30, "m")).astype("datetime64[h]").astype("datetime64[s]")
        if abs((t - hour) / np.timedelta64(1, "s")) > tolerance_s:
            raise ValueError(f"{path}: valid time {t} not within {tolerance_s} s of a full hour")
        if hour in out:
            raise ValueError(f"two history files for {hour}")
        out[hour] = path
    return out


def case_pairs(arm, case, block, t_final, files_by_hour, surface_obs, holdout_sites, *, sondes=None,
               protocol=PROTOCOL, read=read_wrfout):
    """All pre-registered pairs of one arm and case.

    files_by_hour: nominal valid hour -> wrfout (``wrfout_inventory``); surface_obs: holdout-site obs with absolute
    ``time``; sondes: {lead_h: (sonde ObsBatch, sonde TWI dict from diagnostics.sounding_inversion)} for the
    +12/+24 h launches. Missing files are recorded in ``attrs['missing_valid_times']`` and yield no pairs, so a
    file present in one arm only makes the comparison INCOMPLETE (unmatched pairs).
    """
    import pandas as pd

    from . import diagnostics

    t_final = np.datetime64(t_final, "s")
    parts, missing, dropped = [], [], 0
    for lead in range(1, 25):
        window = lead_window(lead)
        vt = t_final + np.timedelta64(lead, "h")
        at = surface_obs.subset(surface_obs.time == vt)
        if not len(at):
            continue
        if vt not in files_by_hour:
            missing.append(str(vt))
            continue
        frames, geo = read(files_by_hour[vt])
        part = surface_pairs(at, frames, geo, arm=arm, case=case, block=block, lead_h=window,
                             holdout_sites=holdout_sites, protocol=protocol)
        dropped += part.attrs.get("geometry_dropped", 0)
        parts.append(part)
    for lead, (sonde, sonde_twi) in (sondes or {}).items():
        if lead not in protocol.sonde_leads_h:
            raise ValueError(f"sonde lead {lead} h is not pre-registered")
        vt = t_final + np.timedelta64(lead, "h")
        if vt not in files_by_hour:
            missing.append(str(vt))
            continue
        frames, geo = read(files_by_hour[vt])
        part = sonde_pairs(sonde, frames, geo, arm=arm, case=case, block=block, lead_h=lead, protocol=protocol)
        dropped += part.attrs.get("geometry_dropped", 0)
        parts.append(part)
        launch = str(sonde.launch[0]) if len(sonde) else f"{case}+{lead}"
        model_twi = diagnostics.model_inversion(frames, geo, float(sonde.lat[0]), float(sonde.lon[0]))
        parts.append(inversion_pair(sonde_twi, model_twi["member_mean_column"], arm=arm, case=case, block=block,
                                    lead_h=lead, launch=launch, protocol=protocol))
    parts = [p for p in parts if len(p)]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame([], columns=_COLUMNS)
    out.attrs["missing_valid_times"] = missing
    out.attrs["geometry_dropped"] = dropped
    return out
