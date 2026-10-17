"""WRF ``sign(1,time_step)`` of the odd-order advection upwinding (trace-time).

Every WRF advection statement function carries the time-step sign in its upwind correction,
``flux5 = flux6 - sign(1,time_step)*sign(1.,ua)*(...)/60`` and
``flux3 = flux4 + sign(1,time_step)*sign(1.,ua)*(...)/12`` (dyn_em/module_advect_em.F:202-212,
:1612-1620, :3110-3118, :4443-4451, :6173-6181, ...), so a backward integration (the TDFI
backward stage, ``time_step < 0``) upwinds along the backward-in-time transport and stays
dissipative.  The port's flux stencils use ``sign(vel)`` (forward time); ``upwind_sign`` applies
the WRF time-step sign at TRACE time: outside a backward context it returns its argument
unchanged, so every forward program is byte-identical.

``operational_mode`` enters :func:`wrf_time_step_sign` for a step whose ``dt_s < 0``.
"""

from __future__ import annotations

import contextlib
import contextvars

__all__ = ["time_step_sign", "upwind_sign", "wrf_time_step_sign"]

_SIGN: contextvars.ContextVar[int] = contextvars.ContextVar("gpuwrf_wrf_time_step_sign", default=1)


def time_step_sign() -> int:
    """``sign(1, time_step)`` of the step being traced (+1 unless inside a backward step)."""

    return _SIGN.get()


@contextlib.contextmanager
def wrf_time_step_sign(dt_s: float):
    """Trace the enclosed step with WRF's ``sign(1, time_step)`` for ``dt_s``."""

    token = _SIGN.set(-1 if float(dt_s) < 0.0 else 1)
    try:
        yield
    finally:
        _SIGN.reset(token)


def upwind_sign(velocity_sign):
    """``sign(1,time_step) * sign(1.,ua)`` given the stencil's ``sign(1.,ua)`` term."""

    return velocity_sign if _SIGN.get() > 0 else -velocity_sign
