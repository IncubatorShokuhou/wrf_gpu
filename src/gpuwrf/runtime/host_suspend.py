"""Host suspend/hibernation detection for run receipts (host-only, no JAX).

CLOCK_BOOTTIME counts time spent in suspend/hibernation, CLOCK_MONOTONIC does not; both are slewed alike,
so their difference grows only while the host sleeps. A run that spans a suspend keeps going if the GPU
context survives (its wall timings are then invalid, FINDINGS E60) or fails loudly; either way the receipt
should say so. This module never blocks or inhibits a suspend.
"""
from __future__ import annotations

import time

# Offset growth below this is clock noise, not a suspend.
SUSPEND_THRESHOLD_S = 1.0


def boottime_offset_s() -> float | None:
    """CLOCK_BOOTTIME − CLOCK_MONOTONIC in seconds, or None where the clocks do not exist (non-Linux)."""
    try:
        return time.clock_gettime(time.CLOCK_BOOTTIME) - time.clock_gettime(time.CLOCK_MONOTONIC)
    except (AttributeError, OSError):
        return None


class SuspendWatch:
    """Measures how long the host was suspended since construction."""

    def __init__(self) -> None:
        self._start = boottime_offset_s()

    def suspended_s(self) -> float | None:
        now = boottime_offset_s()
        if self._start is None or now is None:
            return None
        return max(0.0, now - self._start)

    def summary(self) -> dict:
        """Receipt entry: suspended seconds and whether wall timings of this run are valid."""
        suspended = self.suspended_s()
        if suspended is None:
            return {"suspended_s": None, "timing_valid": None, "clock": "unavailable"}
        return {"suspended_s": round(suspended, 3), "timing_valid": suspended < SUSPEND_THRESHOLD_S,
                "clock": "CLOCK_BOOTTIME-CLOCK_MONOTONIC"}

    def failure_note(self, *, checkpoint_dir=None) -> str:
        """Suffix for a failure message when the host slept during the run, else ''."""
        suspended = self.suspended_s()
        if suspended is None or suspended < SUSPEND_THRESHOLD_S:
            return ""
        hint = (f"; resume from the last verified generation in {checkpoint_dir} with --resume-checkpoint"
                if checkpoint_dir else "; rerun, or use --checkpoint-dir to make long runs resumable")
        return f" [host was suspended for {suspended:.0f} s during this run; the GPU context may not have survived{hint}]"
