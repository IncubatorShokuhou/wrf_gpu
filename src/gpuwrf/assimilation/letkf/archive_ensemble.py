"""Causal same-phase archive ensemble (lead a1-da; DESIGN_v2 §6 alternative after the V3 smoke).

Members are deviations of archived CPU-WRF states valid at the SAME UTC hour as the analysis (no diurnal-phase
contamination, unlike time-expanded sampling), from runs issued BEFORE the analysis time and valid no later than
it (causal), on the identical grid, re-centred on the control frame:

    x_m = x_ctrl(t_a) + s * (x_i - mean_i x_i)

Only the requested (analysis + H(x)) fields are perturbed; every passive field comes from the control. This is
an EnOI-like, flow-independent climatological covariance (seasonally mixed), NOT a cycling ensemble; the
scale s is calibrated from innovation statistics of the assimilated observations (``innovation_scale``).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import re

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset

GRID_IDENTITY = ("XLAT", "XLONG", "HGT", "ZNU", "ZNW", "PB", "PHB", "MUB")
# Moisture is perturbed in log space and re-centred multiplicatively: positive by construction, ensemble mean
# exactly the control (V3 0228: additive archive deviations made 305 k background qv cells negative).
LOG_FIELDS = ("QVAPOR", "Q2")
_Q_FLOOR = 1.0e-12


def _log_members(centre, stack, scale):
    logs = np.log(np.maximum(stack, _Q_FLOOR))
    factors = np.exp(scale * (logs - logs.mean(axis=0)[None]))
    return np.maximum(centre, 0.0)[None] * factors / factors.mean(axis=0)[None]
_RUN_RE = re.compile(r"wg_(\d{8})_(\d{2})z_a\d+")


def _times(ds):
    raw = ds["Times"][:]
    return [b"".join(np.asarray(r, dtype="S1").tolist()).decode().strip("\0 ") for r in raw]


def _sha(path, limit=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        digest.update(stream.read(limit))
    return digest.hexdigest()


def candidate_frames(run_dirs, domain, t_a, *, hour=None, exclude=(), max_age_days=None):
    """(path, valid_time, run) of archived history frames at the analysis UTC hour, causal w.r.t. t_a.

    ``max_age_days`` keeps only states valid within that many days before t_a (same season)."""
    hour = t_a.hour if hour is None else hour
    out = []
    for run in run_dirs:
        run = Path(run)
        match = _RUN_RE.search(str(run))
        if match:
            issued = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
            if issued >= t_a:
                continue
        for path in sorted(run.glob(f"wrfout_{domain}_*_{hour:02d}:00:*")):
            stamp = datetime.strptime(path.name.split("_", 2)[2][:13], "%Y-%m-%d_%H").replace(tzinfo=timezone.utc)
            too_old = max_age_days is not None and stamp < t_a - timedelta(days=float(max_age_days))
            if stamp <= t_a and not too_old and str(path) not in exclude:
                out.append((path, stamp, str(run)))
    return out


def read_fields(path, names):
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        return {n: np.asarray(ds[n][0]) for n in names}, _times(ds)[0]


def same_phase_members(control_file, run_dirs, domain, t_a, names, *, scale=1.0, max_members=None, max_age_days=45,
                       picks=None):
    """Return (frames {name: (N, …) float32}, meta). ``picks`` = guard-filtered candidate_frames() (preferred)."""
    control, control_time = read_fields(control_file, tuple(names) + GRID_IDENTITY)
    if picks is None:
        picks = candidate_frames(run_dirs, domain, t_a, exclude=(str(control_file),), max_age_days=max_age_days)
    if max_members is not None:
        picks = picks[-int(max_members):]           # most recent first-issued states
    states, used = [], []
    for path, stamp, run in picks:
        fields, valid = read_fields(path, tuple(names) + GRID_IDENTITY)
        for g in GRID_IDENTITY:
            if not np.array_equal(fields[g], control[g]):
                raise ValueError(f"{path}: grid identity differs in {g}")
        states.append({n: fields[n].astype(np.float64) for n in names})
        used.append({"path": str(path), "valid": valid, "run": run, "sha256_head": _sha(path)})
    if len(states) < 2:
        raise ValueError("need at least two archived same-phase states")
    frames = {}
    for n in names:
        stack = np.stack([st[n] for st in states])
        if n in LOG_FIELDS:
            frames[n] = _log_members(control[n].astype(np.float64), stack, scale).astype(np.float32)
        else:
            frames[n] = (control[n][None] + scale * (stack - stack.mean(axis=0)[None])).astype(np.float32)
    meta = {"kind": "same-phase causal archive ensemble (EnOI-like, flow-independent)", "control_file": str(control_file),
            "control_time": control_time, "scale": float(scale), "max_age_days": max_age_days,
            "n_members": len(states), "members": used}
    return frames, meta


def rescale(frames, factor):
    """Scale member deviations about the ensemble mean (the mean = control is kept).

    LOG_FIELDS are rescaled in log space and re-centred multiplicatively (stay positive)."""
    out = {}
    for name, value in frames.items():
        value = np.asarray(value, dtype=np.float64)
        mean = value.mean(axis=0)
        if name in LOG_FIELDS:
            out[name] = _log_members(mean, value, factor).astype(np.float32)
        else:
            out[name] = (mean[None] + factor * (value - mean[None])).astype(np.float32)
    return out


def rescale_levels(frames, scale_k, *, surface_scale=None):
    """Vertically varying deviation scale: 3-D fields use scale_k[k] per mass level (W/PH-staggered fields are
    not analysed), 2-D fields use ``surface_scale`` (default scale_k[0]). Log fields as in :func:`rescale`."""
    scale_k = np.asarray(scale_k, dtype=np.float64)
    surface_scale = float(scale_k[0] if surface_scale is None else surface_scale)
    out = {}
    for name, value in frames.items():
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 4 and value.shape[1] == len(scale_k):
            out[name] = np.stack([rescale({name: value[:, k]}, scale_k[k])[name] for k in range(len(scale_k))], axis=1)
        elif value.ndim == 3:
            out[name] = rescale({name: value}, surface_scale)[name]
        else:
            out[name] = value.astype(np.float32)
    return out


def innovation_scale(d, sigma_o, hx_var):
    """Spread scale s with E[d^2] = sigma_o^2 + s^2 var(Hx) (Desroziers-type consistency on assimilated obs).

    Returns (s, diagnostics); s = 0 when the innovations do not exceed the observation error.
    """
    d = np.asarray(d, float)
    so2 = np.asarray(sigma_o, float) ** 2
    hv = np.asarray(hx_var, float)
    excess = float(np.mean(d * d) - np.mean(so2))
    s2 = max(excess, 0.0) / float(np.mean(hv))
    return float(np.sqrt(s2)), {"mean_d2": float(np.mean(d * d)), "mean_sigma_o2": float(np.mean(so2)),
                                "mean_var_hx": float(np.mean(hv)), "n": int(d.size)}
