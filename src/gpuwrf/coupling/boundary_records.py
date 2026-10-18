"""Root lateral-boundary record windows: the in-step side.

The standalone root's ``State.*_bdy`` leaves used to hold EVERY wrfbdy record
(+ the synthesized end level), so the record count was a program shape and a
48 h input set could never reuse a program primed on 72 h inputs (v0.4.0
bundle incident). With record windows the leaves hold a fixed number of
consecutive records and the traced clock base carries ``bdy_window`` =
``int32[(base, last)]``: ``base`` is the global record in slot 0 and ``last``
the full stack's last valid lower bracket (records - 2).

Every consumer keeps the released GLOBAL clock arithmetic (fraction, bracket
clip, alpha/dtbc) and only maps the record index to ``global - base``, so a
windowed run reads exactly the records a full stack would. A bracket outside
the resident window is poisoned to NaN (loud) instead of reading another
record; the host (``integration/record_windows.py``) refreshes the window
before every step call.

The window is scoped per traced step (``_physics_boundary_step``): nested
children step with their own two-level force-down packages and no window.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

import jax.numpy as jnp

_SCOPE = threading.local()


@contextmanager
def lateral_window_scope(window):
    """Make ``window`` (traced ``int32[2]`` or ``None``) the active root record window."""
    previous = getattr(_SCOPE, "window", None)
    _SCOPE.window = window
    try:
        yield
    finally:
        _SCOPE.window = previous


def active_window():
    """The traced ``(base, last)`` window of the step being traced, else ``None``."""
    return getattr(_SCOPE, "window", None)


def window_bracket(lower, records: int, window, *, upper_clip: bool = False):
    """Map an UNCLIPPED global bracket index to ``(lower, lower_slot, upper_slot)``.

    Without a window this is the released full-stack clip: ``lower`` in
    ``[0, records-2]`` (``upper_clip``: ``[0, records-1]`` with the upper index
    clipped too, the legacy value interpolator). With a window the clip bound
    is the traced global ``last`` (``last+1`` for ``upper_clip``) and the slots
    are relative to ``base``. A bracket outside the window gets the slot
    ``records`` (one past the window): ``jnp.take``'s default fill mode then
    reads NaN (loud), and the float graph stays the released one -- only the
    integer index math differs (a select on the values would change XLA's
    fusion and FMA contraction of ``value + dtbc*rate``).
    """
    if window is None:
        if upper_clip:
            lower = jnp.clip(lower, 0, records - 1)
            return lower, lower, jnp.clip(lower + 1, 0, records - 1)
        lower = jnp.clip(lower, 0, records - 2)
        return lower, lower, lower + 1
    window = jnp.asarray(window, jnp.int32)
    base, last = window[0], window[1]
    if upper_clip:
        lower = jnp.clip(lower, 0, last + 1)
        upper = jnp.clip(lower + 1, 0, last + 1)
    else:
        lower = jnp.clip(lower, 0, last)
        upper = lower + 1
    lower_slot, upper_slot = lower - base, upper - base
    bad = (lower_slot < 0) | (upper_slot > records - 1)
    return lower, jnp.where(bad, records, lower_slot), jnp.where(bad, records, upper_slot)


__all__ = ["active_window", "lateral_window_scope", "window_bracket"]
