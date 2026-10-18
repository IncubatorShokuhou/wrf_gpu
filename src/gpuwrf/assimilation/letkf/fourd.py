"""4D-LETKF observation equivalents over the cycle window from each member's own hourly frames (lane a1-obs).

``hx_4d(obs, member_run_dirs, t_a, window_h, geo)`` evaluates every observation valid in the window
(t_a - window_h, t_a] on each member's OWN history frame nearest to the observation time (within
``half_window_s``), with the single-time operators of ``letkf.operators`` (``model_equivalents_4d`` binning).
Observations outside the window or without a member frame within ``half_window_s`` get NaN (and QC_OUTSIDE when
no member has a frame). Frames are found from the wrfout FILE NAMES (nominal hour, d01 history sits at
hh:00:18/hh:00:36 because dt = 54 s; offsets up to ``name_tolerance_s`` are assigned to the full hour and
reported); a frame is opened only if an observation needs it. Any member file below the ALISIOS data roots is
opened only inside ``letkf.alisios_guard.guarded_session`` (an active session, or one opened here when
``guard_receipt_dir`` is given); otherwise the call is refused before anything is opened.
"""
from __future__ import annotations

import dataclasses
import glob
import re
from pathlib import Path

import numpy as np

from . import obs as ob
from .operators import GridLocator, model_equivalents_4d

FIELDS_BY_KIND = {ob.SONDE: ("T", "QVAPOR", "U", "V", "P", "PB"), ob.SURFACE: ("T2", "Q2", "U10", "V10", "PSFC"),
                  ob.CLOUD: ("CLDFRA",)}
_NAME = re.compile(r"wrfout_(d0\d)_(\d{4}-\d{2}-\d{2})_(\d{2}):(\d{2}):(\d{2})$")


@dataclasses.dataclass(frozen=True)
class Hx4D:
    """hx (N, n_obs); frame_time (N, n_obs) nominal hour used (NaT if none); qc (n_obs,) operator/window bits;
    in_window (n_obs,); frame_offset_s {member: {hour: file offset s}}; files_opened {member: [paths]}."""

    hx: np.ndarray
    frame_time: np.ndarray
    qc: np.ndarray
    in_window: np.ndarray
    frame_offset_s: dict
    files_opened: dict


def wrfout_frames(run_dir, *, domain="d01", name_tolerance_s=60):
    """{nominal hour (datetime64[s]): (path, offset_s)} from the history file NAMES (nothing is opened)."""
    out = {}
    for path in sorted(glob.glob(str(Path(run_dir) / f"wrfout_{domain}_*"))):
        m = _NAME.search(Path(path).name)
        if not m:
            continue
        t = np.datetime64(f"{m.group(2)}T{m.group(3)}:{m.group(4)}:{m.group(5)}", "s")
        hour = (t + np.timedelta64(30, "m")).astype("datetime64[h]").astype("datetime64[s]")
        off = float((t - hour) / np.timedelta64(1, "s"))
        if abs(off) > name_tolerance_s:
            continue
        if hour in out:
            raise ValueError(f"{run_dir}: two history files for {hour}")
        out[hour] = (path, off)
    return out


def _needed_fields(obs):
    names = []
    for kind in sorted(set(obs.kind.tolist())):
        names += [n for n in FIELDS_BY_KIND[int(kind)] if n not in names]
    return tuple(names)


def _obs_seconds(obs, t_a):
    """Observation time relative to t_a [s]: absolute ``time`` when set, else ``dt_s``."""
    rel = obs.dt_s.astype(float).copy()
    have = ~np.isnat(obs.time)
    rel[have] = (obs.time[have] - t_a) / np.timedelta64(1, "s")
    return rel


def hx_4d(obs, member_run_dirs, t_a, window_h, geo, *, half_window_s=1800.0, domain="d01", read=None,
          guard_receipt_dir=None, **operator_kwargs):
    """4D model equivalents. ``member_run_dirs``: one directory per member (member order = output row order)."""
    from . import alisios_guard
    from .scoring import read_wrfout

    t_a = ob.to_datetime64(t_a)
    w = np.timedelta64(int(round(window_h * 3600)), "s")
    rel = _obs_seconds(obs, t_a)
    in_window = (rel > -window_h * 3600.0) & (rel <= 0.0)
    timed = obs.replace(dt_s=rel)
    names = _needed_fields(obs)
    inventories = [wrfout_frames(d, domain=domain) for d in member_run_dirs]
    alis = sorted({p for inv in inventories for h, (p, _) in inv.items()
                   if t_a - w < h <= t_a and alisios_guard.is_alisios_data(p)})
    if alis and alisios_guard._ACTIVE is None:
        if guard_receipt_dir is None:
            raise PermissionError(f"{len(alis)} member files are ALISIOS data: call inside alisios_guard."
                                  "guarded_session or pass guard_receipt_dir")
        days = {p: re.search(r"(\d{4})-(\d{2})-(\d{2})_", Path(p).name).group(0)[:10].replace("-", "") for p in alis}
        issues = sorted({re.search(r"/wg_(\d{8})_", p).group(1) for p in alis})
        with alisios_guard.guarded_session(days, issues, guard_receipt_dir):
            return hx_4d(obs, member_run_dirs, t_a, window_h, geo, half_window_s=half_window_s, domain=domain,
                         read=read, **operator_kwargs)
    read = read or (lambda path: read_wrfout(path, names=names))
    locator = operator_kwargs.pop("locator", None) or GridLocator(geo["XLAT"], geo["XLONG"])
    n = len(obs)
    hx = np.full((len(member_run_dirs), n), np.nan)
    ftime = np.full((len(member_run_dirs), n), np.datetime64("NaT", "s"))
    qc = np.zeros(n, np.int32)
    any_frame = np.zeros(n, bool)
    offsets, opened = {}, {}
    sel = np.flatnonzero(in_window & obs.usable)
    sub = timed.subset(sel)
    for m, inv in enumerate(inventories):
        hours = sorted(h for h in inv if t_a - w < h <= t_a)
        offsets[m] = {str(h): inv[h][1] for h in hours}
        opened[m] = []

        def loader(path, m=m):
            def load():
                opened[m].append(path)
                frames, _ = read(path)
                return frames
            return load

        frame_dt = [float((h - t_a) / np.timedelta64(1, "s")) for h in hours]
        lazy = [loader(inv[h][0]) for h in hours]
        part, flags, which = model_equivalents_4d(sub, lazy, frame_dt, geo, half_window_s=half_window_s,
                                                  locator=locator, **operator_kwargs)
        got = which >= 0
        hx[m, sel] = part[0]
        ftime[m, sel[got]] = np.array([hours[k] for k in which[got]], dtype="datetime64[s]")
        any_frame[sel[got]] = True
        qc[sel] |= np.where(got, flags, 0).astype(np.int32)
    qc[~in_window] |= ob.QC_OUTSIDE
    qc[in_window & obs.usable & ~any_frame] |= ob.QC_OUTSIDE
    return Hx4D(hx=hx, frame_time=ftime, qc=qc, in_window=in_window, frame_offset_s=offsets, files_opened=opened)


def window_station_obs(source, t_a, window_h, **kwargs):
    """Hourly station obs at the full hours in (t_a - window_h, t_a], dt_s relative to t_a (instantaneous)."""
    t_a = ob.to_datetime64(t_a)
    first = t_a - np.timedelta64(int(round(window_h * 3600)), "s") + np.timedelta64(1, "h")
    return ob.read_station_obs(source, first, t_a, analysis_time=t_a, **kwargs)
