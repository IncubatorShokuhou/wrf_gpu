"""Backend-free admission of forecast endpoints on whole root steps."""
from __future__ import annotations

from datetime import datetime, timezone
import math


def aligned_steps(hours, dt_s):
    seconds, dt = float(hours) * 3600, float(dt_s)
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("forecast duration and root dt must be positive and finite")
    raw = seconds / dt
    steps = round(raw)
    if abs(raw - steps) > 1.e-9:
        raise ValueError(f"forecast duration {seconds:g}s must be a whole multiple of root dt={dt:g}s (would need {raw:g} steps)")
    return steps


def explicit_window(namelist, *, minutes=None, end_time=None):
    """Return hours since input issue time; no model or backend imports."""
    def first(value):
        return value[0] if isinstance(value, (list, tuple)) else value

    if minutes is not None:
        hours = float(minutes) / 60
    else:
        tc = namelist.get("time_control", {})
        try:
            start = datetime(*(int(first(tc["start_" + key])) for key in
                               ("year", "month", "day", "hour")),
                             int(first(tc.get("start_minute", 0))),
                             int(first(tc.get("start_second", 0))), tzinfo=timezone.utc)
            end = datetime.fromisoformat(str(end_time).replace("_", "T").replace("Z", "+00:00"))
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            hours = (end - start).total_seconds() / 3600
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("--end-time needs an ISO UTC endpoint and namelist start date") from exc
    dt = first(namelist.get("domains", {}).get("time_step",
               namelist.get("time_control", {}).get("time_step")))
    if dt is None:
        raise ValueError("namelist must declare root time_step for a short window")
    aligned_steps(hours, dt)
    return hours
