"""Record windows: step programs independent of the wrfbdy/wrflowinp record count.

v0.4.0 passed the root's WHOLE ``wrfbdy_d01`` (records + 1 synthesized end
level) as ``State.*_bdy`` and EVERY ``wrflowinp`` record as the aux4
``LowerBoundary``, so the record count was a program shape: 48 h input files
could never reuse programs primed on 72 h files (r-pkg040 bundle incident,
2026-10-17). Here the host owns the full records and the programs see fixed
windows:

* lateral (root only): ``WINDOW_RECORDS`` consecutive wrfbdy levels in
  ``State.*_bdy`` plus ``clock_base.bdy_window`` = ``int32[(base, last)]``.
  The in-step consumers keep the full-stack clock arithmetic and read slot
  ``global - base`` (coupling/boundary_records.py), so values are identical;
* aux4 (every ``sst_update`` domain): ``clock_base.lower_boundary`` holds
  ``WINDOW_RECORDS - 1`` resident records in order (activation padded with
  INT32 max past the file) and one sentinel slot: NaN data activated at the
  first non-resident record, so a call that outruns its window fails loudly.

Before every step call the owner (:class:`RecordWindows`) checks the call's
step range, refreshes the window if needed (one host->device copy per few
boundary intervals) and splits chunks that would cross more records than a
window holds. Host only: no jit, no device reads.

No device op may see a record count either (each would be another cached
program): the windows are chosen BEFORE the initial carry is built
(:func:`prepare_record_windows`), the loader's dtype casts of the full stacks
run on the host (:func:`host_cast_to_state`, the same round-to-nearest cast),
and the aux4 history frames select from the window.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

WINDOW_RECORDS = 4
_INT32_MAX = np.iinfo(np.int32).max
# Clock fractions are formed in fp32 (relative error ~6e-8) or f64; a lead
# within this many record intervals of a record time may round to either side.
_FRACTION_TOLERANCE = 1.0e-4


def _record_leaf_names(state: Any, records: int) -> tuple[tuple[str, ...], str | None]:
    """The State ``*_bdy`` leaves that carry the record axis, or a refusal reason."""
    from gpuwrf.contracts.state import State

    names = []
    for name in State.__slots__:
        if not name.endswith("_bdy"):
            continue
        value = getattr(state, name, None)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if name == "tracer_bdy" or not shape or int(shape[0]) != records:
            return (), f"{name} leading dimension {shape[:1] if shape else None} != {records} records"
        names.append(name)
    if "u_bdy" not in names:
        return (), "no u_bdy record leaf"
    return tuple(names), None


class RecordWindows:
    """Host owner of one domain's full boundary records and its active windows."""

    def __init__(self, *, dt_s: float, lateral: dict[str, np.ndarray] | None = None,
                 interval_s: float | None = None, sharding: Any = None, lower_boundary: Any = None) -> None:
        self.dt_s = float(dt_s)
        self.window = WINDOW_RECORDS
        self._in_call = False
        self._lateral = None
        self._lateral_base: int | None = None
        self._lateral_leaf: Any = None
        self._window_arrays: dict[int, jax.Array] = {}
        if lateral:
            self.adopt_lateral(lateral, interval_s=interval_s, sharding=sharding)
        # Aux4 records: full host copy (activation steps + 4 fields).
        self._aux = None
        if lower_boundary is not None:
            self._aux = tuple(np.asarray(jax.device_get(value)) for value in lower_boundary)
            self._aux_type = type(lower_boundary)
            self._aux_records = int(self._aux[0].shape[0])
        self._aux_base = 0
        self._aux_windows: dict[int, Any] = {}

    def adopt_lateral(self, lateral: dict[str, np.ndarray], *, interval_s: float | None, sharding: Any = None,
                      installed: Any = None) -> None:
        """Own the root's full lateral records (host, in the carry's dtype).

        ``installed``: the carry's current ``u_bdy`` when it already holds the
        base-0 window (load time), so the first call needs no refresh.
        """
        self._lateral = dict(lateral)
        self._sharding = sharding
        sizes = {int(value.shape[0]) for value in self._lateral.values()}
        if len(sizes) != 1:
            raise ValueError("record windows need one record count across *_bdy leaves")
        self.records = sizes.pop()
        if self.records < 2:
            raise ValueError("record windows need at least two wrfbdy levels")
        self.last = self.records - 2
        self.interval_s = float(interval_s)
        if not self.interval_s > 0:
            raise ValueError("record windows need a positive boundary interval")
        if installed is not None:
            self._lateral_base, self._lateral_leaf = 0, installed

    # ------------------------------------------------------------------ ranges
    @property
    def lateral(self) -> bool:
        return self._lateral is not None

    @property
    def aux(self) -> bool:
        return self._aux is not None

    def lateral_range(self, start_step: int, n_steps: int) -> tuple[int, int]:
        """Global records a call of steps ``start_step..+n_steps-1`` may read (inclusive).

        Boundary leads of step s lie in [(s-1)*dt, (s+1)*dt] (end-of-step and
        RK stage leads); each lead reads the bracket ``ceil(f)-1`` or ``floor(f)``
        and the next record, all clipped to ``[0, last]`` like the full stack.
        """
        first, end = int(start_step) - 1, int(start_step) + int(n_steps)
        leads = (first * self.dt_s, end * self.dt_s)
        g_a = min(leads) / self.interval_s
        g_b = max(leads) / self.interval_s
        lo = math.floor(g_a - _FRACTION_TOLERANCE)
        hi = math.floor(g_b + _FRACTION_TOLERANCE) + 1
        lo = min(max(lo, 0), self.last)
        hi = min(max(hi, lo + 1), self.last + 1)
        return lo, hi

    def _aux_index(self, entry_step: int) -> int:
        return int(np.sum(int(entry_step) >= self._aux[0])) - 1

    def aux_range(self, start_step: int, n_steps: int) -> tuple[int, int]:
        """Aux4 records selected at the step entries of the call (inclusive)."""
        first = self._aux_index(int(start_step) - 1)
        last = self._aux_index(int(start_step) + int(n_steps) - 2)
        return max(first, 0), max(last, 0)

    def _fits(self, start_step: int, n_steps: int) -> bool:
        if self.lateral:
            lo, hi = self.lateral_range(start_step, n_steps)
            if hi - lo + 1 > self.window:
                return False
        if self.aux:
            first, last = self.aux_range(start_step, n_steps)
            if last - first + 1 > self.window - 1:
                return False
        return True

    def split(self, start_step: int, n_steps: int) -> list[tuple[int, int]]:
        """Chunks of one call that each fit a window (the same program runs them all)."""
        chunks = []
        start, remaining = int(start_step), int(n_steps)
        while remaining > 0:
            if self._fits(start, remaining):
                size = remaining
            else:
                low, high = 1, remaining  # largest fitting size by bisection
                if not self._fits(start, 1):
                    raise ValueError(
                        f"one step at {start} needs more than {self.window} boundary records; "
                        "the boundary interval is too short for record windows")
                while low < high:
                    middle = (low + high + 1) // 2
                    low, high = (middle, high) if self._fits(start, middle) else (low, middle - 1)
                size = low
            chunks.append((start, size))
            start += size
            remaining -= size
        return chunks

    # ----------------------------------------------------------------- windows
    def _window_array(self, base: int) -> jax.Array:
        array = self._window_arrays.get(base)
        if array is None:
            array = jnp.asarray(np.array([base, self.last], dtype=np.int32))
            self._window_arrays[base] = array
        return array

    def _lateral_rows(self, base: int) -> np.ndarray:
        return np.minimum(np.arange(base, base + self.window), self.records - 1)

    def lateral_window(self, name: str, base: int, *, source: np.ndarray | None = None) -> np.ndarray:
        """Host window of one record leaf (rows past the stack repeat its last level)."""
        full = self._lateral[name] if source is None else source
        return np.ascontiguousarray(full[self._lateral_rows(base)])

    def install_lateral(self, state: Any, base: int) -> Any:
        rows = self._lateral_rows(base)
        updates = {}
        for name, full in self._lateral.items():
            window = np.ascontiguousarray(full[rows])
            updates[name] = (jax.device_put(window, self._sharding) if self._sharding is not None
                             else jnp.asarray(window))
        self._lateral_base = int(base)
        state = state.replace(**updates)
        self._lateral_leaf = state.u_bdy
        return state

    def _aux_window(self, base: int) -> Any:
        window = self._aux_windows.get(base)
        if window is not None:
            return window
        act, *fields = self._aux
        count = self._aux_records
        resident = [min(base + i, count - 1) for i in range(self.window - 1)]
        steps = [int(act[base + i]) if base + i < count else _INT32_MAX for i in range(self.window - 1)]
        sentinel = base + self.window - 1
        steps.append(int(act[sentinel]) if sentinel < count else _INT32_MAX)
        arrays = []
        for field in fields:
            rows = field[resident]
            tail = (np.full_like(field[:1], np.nan) if sentinel < count else field[count - 1:count])
            arrays.append(jnp.asarray(np.concatenate([rows, tail], axis=0)))
        window = self._aux_type(jnp.asarray(np.asarray(steps, dtype=act.dtype)), *arrays)
        if len(self._aux_windows) > 8:
            self._aux_windows.clear()
        self._aux_windows[base] = window
        return window

    def initial_lower_boundary(self) -> Any:
        """The namelist's fixed placeholder: the run-start aux window."""
        return self._aux_window(0)

    def clock_base(self, clock: Any) -> Any:
        """``clock`` with the CURRENT windows (build_clock_base)."""
        updates = {}
        if self.lateral:
            updates["bdy_window"] = self._window_array(self._lateral_base or 0)
        if self.aux:
            updates["lower_boundary"] = self._aux_window(self._aux_base)
        return clock._replace(**updates)

    # -------------------------------------------------------------------- calls
    def prepare(self, carry: Any, clock_base: Any, start_step: int, n_steps: int) -> tuple[Any, Any]:
        """Make ``carry``/``clock_base`` hold every record the call may read."""
        if self.lateral:
            lo, hi = self.lateral_range(start_step, n_steps)
            state = carry.state
            known = state.u_bdy is self._lateral_leaf and self._lateral_base is not None
            if not (known and self._lateral_base <= lo and hi <= self._lateral_base + self.window - 1):
                base = min(lo, max(self.records - self.window, 0))
                carry = carry.replace(state=self.install_lateral(state, base))
            for name in self._lateral:
                if int(getattr(carry.state, name).shape[0]) != self.window:
                    raise ValueError(f"record window: {name} is not a {self.window}-record window")
        if self.aux:
            first, last = self.aux_range(start_step, n_steps)
            if not (self._aux_base <= first and last <= self._aux_base + self.window - 2):
                self._aux_base = min(first, max(self._aux_records - (self.window - 1), 0))
        return carry, self.clock_base(clock_base)

    def note(self, carry: Any) -> None:
        """Remember the window leaf of a call's output (same records as its input)."""
        if self.lateral:
            self._lateral_leaf = carry.state.u_bdy

    @property
    def in_call(self) -> bool:
        return self._in_call

    def advance(self, call: Callable[[Any, int, int, Any], Any], carry: Any, start_step: int, n_steps: int,
                clock_base: Any) -> Any:
        """Run ``call(carry, start, n, clock_base)`` over window-sized chunks."""
        for start, size in self.split(start_step, n_steps):
            carry, prepared = self.prepare(carry, clock_base, start, size)
            self._in_call = True
            try:
                carry = call(carry, start, size, prepared)
            finally:
                self._in_call = False
            self.note(carry)
        return carry

    def lower_boundary_history(self, own_step: int) -> dict[str, Any]:
        """Held aux4 fields of the last completed step, selected on device from the current window.

        The window prepared for the last call holds the record of its last
        step entry, so io.lower_boundary.lower_boundary_history on the window
        selects the same values with the same device ops (no host->device
        field copies per history frame).
        """
        from gpuwrf.io.lower_boundary import lower_boundary_history

        index = self._aux_index(max(int(own_step) - 1, 0))
        if not self._aux_base <= index <= self._aux_base + self.window - 2:
            raise ValueError(f"aux4 history record {index} is not in the window at {self._aux_base}")
        return lower_boundary_history(self._aux_window(self._aux_base), own_step)


def _host_records(state: Any, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: np.asarray(jax.device_get(getattr(state, name))) for name in names}


def host_cast_to_state(state: Any, leaves: dict[str, Any]) -> dict[str, Any]:
    """Cast record leaves to the dtypes ``state.replace`` would give them, on the host.

    Same IEEE round-to-nearest cast as the device's, without a device op shaped
    by the record count. A leaf holding non-finite or REAL-subnormal values keeps
    the device cast (FTZ and NaN payloads may differ between the two).
    """
    from gpuwrf.contracts.precision import DEFAULT_DTYPES

    tiny = np.finfo(np.float32).tiny
    out = {}
    for name, value in leaves.items():
        current = getattr(state, name, None)
        target = np.dtype(current.dtype if current is not None and hasattr(current, "dtype")
                          else DEFAULT_DTYPES.dtype_for(name))
        if not hasattr(value, "dtype") or np.dtype(value.dtype) == target or not (
                np.issubdtype(value.dtype, np.floating) and np.issubdtype(target, np.floating)):
            out[name] = value
            continue
        host = np.asarray(jax.device_get(value))
        if not np.isfinite(host).all() or (target.itemsize < host.dtype.itemsize
                                           and np.any((host != 0) & (np.abs(host) < tiny))):
            out[name] = value
            continue
        out[name] = jax.device_put(jnp.asarray(host.astype(target)))
    return out


def _cast_hazard(raw: dict[str, np.ndarray]) -> str | None:
    """A host REAL cast equals the device cast except where FTZ matters: refuse such stacks."""
    tiny = np.finfo(np.float32).tiny
    for name, value in raw.items():
        if not np.issubdtype(value.dtype, np.floating):
            continue
        if not np.isfinite(value).all():
            return f"{name} holds non-finite values"
        if value.dtype == np.float64 and np.any((value != 0) & (np.abs(value) < tiny)):
            return f"{name} holds values below the REAL normal range"
    return None


class PendingRecordWindows:
    """Windows chosen before the initial carry is built (_load_domains).

    The state's record leaves already hold the base-0 window in their RAW dtype,
    so the carry initialisation (precision casts, device commit) only ever sees
    window shapes; :meth:`finish` adopts the full records in the carry's dtype.
    """

    def __init__(self, owner: RecordWindows, raw: dict[str, np.ndarray] | None, report: dict) -> None:
        self.owner, self._raw, self.report = owner, raw, report

    def finish(self, carry: Any, namelist: Any) -> tuple[Any, dict]:
        """Return ``(namelist with the owner, report)`` for the committed initial ``carry``."""
        from dataclasses import replace

        if self._raw is not None:
            converted = {}
            for name, raw in self._raw.items():
                leaf = getattr(carry.state, name)
                full = raw.astype(np.dtype(leaf.dtype))
                window = np.asarray(jax.device_get(leaf))
                rows = np.minimum(np.arange(WINDOW_RECORDS), full.shape[0] - 1)
                if window.shape != full[rows].shape or window.tobytes() != full[rows].tobytes():
                    raise RuntimeError(f"record windows: the carry's {name} is not a cast of its wrfbdy records")
                converted[name] = full
            self.owner.adopt_lateral(converted, interval_s=float(namelist.boundary_config.update_cadence_s),
                                     sharding=carry.state.u_bdy.sharding, installed=carry.state.u_bdy)
            self._raw = None
            full_bytes = sum(value.nbytes for value in converted.values())
            self.report["lateral"].update(device_bytes_full=int(full_bytes), device_bytes_window=int(
                full_bytes * WINDOW_RECORDS // self.owner.records))
        if self.owner.aux:
            self.report["aux4"].update(device_bytes_full=int(sum(v.nbytes for v in self.owner._aux)),
                                       device_bytes_window=int(sum(v.nbytes for v in self.owner._aux_window(0))))
        return replace(namelist, record_windows=self.owner), self.report


def prepare_record_windows(*, state: Any, namelist: Any, dt_s: float, root: bool,
                           boundary_meta: dict | None) -> tuple[Any, Any, PendingRecordWindows | None, dict]:
    """Window one freshly loaded domain BEFORE its initial carry is built.

    Returns ``(state, namelist, pending, report)``: the root's native wrfbdy
    leaves hold the base-0 window (raw dtype), the namelist's aux4
    ``LowerBoundary`` the run-start window. Any layout this cannot window
    keeps the full stacks (correct, record-count specific) and says why.
    """
    report: dict[str, Any] = {"window_records": WINDOW_RECORDS}
    raw = None
    if root and (boundary_meta or {}).get("schema") == "wrfbdy-decoupled-leaf-v1":
        records = int(boundary_meta["times"])
        names, reason = _record_leaf_names(state, records)
        if reason is None:
            raw = _host_records(state, names)
            reason = _cast_hazard(raw)
        if reason is None:
            report["lateral"] = {"records": records, "leaves": len(names)}
        else:
            raw = None
            report["lateral"] = f"full stacks: {reason}"
    lower_boundary = getattr(namelist, "lower_boundary", None)
    if raw is None and lower_boundary is None:
        return state, namelist, None, report
    owner = RecordWindows(dt_s=dt_s, lower_boundary=lower_boundary)
    if raw is not None:
        rows = np.minimum(np.arange(WINDOW_RECORDS), records - 1)
        state = state.replace(_cast=False, **{
            name: jax.device_put(jnp.asarray(np.ascontiguousarray(value[rows]))) for name, value in raw.items()})
    if lower_boundary is not None:
        from dataclasses import replace

        namelist = replace(namelist, lower_boundary=owner.initial_lower_boundary())
        report["aux4"] = {"records": owner._aux_records}
    return state, namelist, PendingRecordWindows(owner, raw, report), report


def record_windows_of(namelist: Any) -> RecordWindows | None:
    return getattr(namelist, "record_windows", None)


__all__ = ["PendingRecordWindows", "RecordWindows", "WINDOW_RECORDS", "host_cast_to_state",
           "prepare_record_windows", "record_windows_of"]
