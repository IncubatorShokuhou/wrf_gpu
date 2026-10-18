"""Per-program AOT cheap keys of a nested case without a GPU or a compile (``gpuwrf aot-key``).

The step programs of a run are addressed by :mod:`gpuwrf.runtime.aot_cheap_key`. To report
the keys a case WOULD use, :func:`capture_step_keys` lets the real runtime build each call
(record windows, clock bases, fused cascade arguments) and stops it right after the product
key function returned, before any blob load, lowering or compile. Nothing here computes a
key on its own: the keys are the product's.

Keys computed in a CPU process carry the CPU target in their executable part, so they are
compared between CASES (two input directories on one machine), not against a GPU cache
listing; :func:`program_components` names the HLO-identity component that differs.

The probe never runs model work: inside :func:`capture_step_keys` the model-step entry of
the domain tree refuses, so a call that would step without computing a key (AOT off, a
non-fori loop mode) yields an ``unavailable`` record instead of a forecast step.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

__all__ = ["capture_call", "capture_step_keys", "program_components", "case_program_keys"]


class _Captured(BaseException):
    """Stops a step call after its key was computed (BaseException: product code catches Exception)."""

    def __init__(self, record: dict[str, Any]) -> None:
        super().__init__("aot key captured")
        self.record = record


class _StepRefused(BaseException):
    """A step call reached the model without computing a key; nothing was dispatched."""


def program_components(fn: Any, args: tuple, kwargs: dict, namelist: Any) -> dict[str, str]:
    """The cheap-key components of one call, by name (aot_cheap_key's own functions)."""
    from gpuwrf.runtime import aot_cheap_key as ck
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash

    return {
        "program_config": ck.program_config_hash(),
        "fn_identity": ck.fn_identity_hash(fn),
        "source_fingerprint": ck.source_fingerprint_hash(),
        "static_config": ck.static_config_hash(namelist),
        "carry_aval": ck.carry_aval_hash(args, kwargs),
        "global_trace_env": ck.global_trace_env_hash(),
        "module_const_env": ck.module_const_env_hash(),
        "baked_tables": baked_table_hash(fn, namelist),
        "program_key": ck.program_key(fn, args, kwargs, namelist),
    }


@contextmanager
def capture_step_keys(*, components: bool = False, lower: bool = False) -> Iterator[None]:
    """Inside the block a domain-tree step call raises ``_Captured`` with its key record.

    ``lower`` also lowers the fused cascade with the key's arguments (tests); the
    advance program is never lowered here.
    """
    from gpuwrf.runtime import domain_tree as dt
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori

    memo_key, fused_key, model_step = dt._CheapKeyMemo.key, dt._fused_cascade_cheap_key, dt._advance_chunk
    lowering = {"active": False}

    def refuse_model_step(*args, **kwargs):
        if lowering["active"]:  # the explicit fused lowering below traces the step, never runs it
            return model_step(*args, **kwargs)
        raise _StepRefused("a model step was reached without an AOT cheap key (AOT off or non-fori loop mode)")

    def advance_key(self, carry, namelist, start, clock_base, *, n_steps, cadence):
        key = memo_key(self, carry, namelist, start, clock_base, n_steps=n_steps, cadence=cadence)
        record: dict[str, Any] = {"program": "advance", "cheap_key": key}
        if components:
            args, kw = (carry, namelist, start, clock_base), {"n_steps": n_steps, "cadence": cadence}
            record["components"] = program_components(_advance_chunk_fori, args, kw, namelist)
        raise _Captured(record)

    def cascade_key(fused_jit, fused_aux, parent_carry, child_carries, parent_start, child_starts,
                    parent_clock_base, child_clock_bases, namelists=()):
        key = fused_key(fused_jit, fused_aux, parent_carry, child_carries, parent_start, child_starts,
                        parent_clock_base, child_clock_bases, *((namelists,) if namelists else ()))
        record: dict[str, Any] = {"program": "fused", "cheap_key": key}
        args = (parent_carry, tuple(child_carries), int(parent_start), tuple(int(s) for s in child_starts),
                parent_clock_base, child_clock_bases, *namelists)
        if components:
            record["components"] = program_components(fused_jit, args, {}, fused_aux)
        if lower:
            lowering["active"] = True
            try:
                record["lowered"] = fused_jit.lower(*args)
            finally:
                lowering["active"] = False
        raise _Captured(record)

    dt._CheapKeyMemo.key, dt._fused_cascade_cheap_key = advance_key, cascade_key
    dt._advance_chunk = refuse_model_step
    try:
        yield
    finally:
        dt._CheapKeyMemo.key, dt._fused_cascade_cheap_key = memo_key, fused_key
        dt._advance_chunk = model_step


def capture_call(call, *args, **capture_kwargs) -> dict[str, Any] | None:
    """Run one step call under :func:`capture_step_keys` and return its key record.

    ``cheap_key`` is ``None`` (with ``unavailable``) when the call computed no key; the
    return value is ``None`` only if the call returned without reaching a step program.
    """
    with capture_step_keys(**capture_kwargs):
        try:
            call(*args)
        except _Captured as captured:
            return captured.record
        except _StepRefused as refused:
            return {"program": None, "cheap_key": None, "unavailable": str(refused)}
    return None


@contextmanager
def _key_capture_mode() -> Iterator[None]:
    """Runtimes built inside compute AOT keys (nested AOT on) whatever the caller's env says."""
    previous = os.environ.get("GPUWRF_NESTED_AOT")
    os.environ["GPUWRF_NESTED_AOT"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GPUWRF_NESTED_AOT", None)
        else:
            os.environ["GPUWRF_NESTED_AOT"] = previous


def case_program_keys(tree: Any, carries: dict[str, Any], *, start_step: int = 1,
                      components: bool = False) -> dict[str, dict[str, Any]]:
    """Cheap keys of every step program a run of ``tree`` dispatches first.

    Per-domain advance programs (the root, and any domain the scheduler cannot fuse)
    and one fused cascade per parent whose children are all leaves. ``carries`` are
    the initial carries (nested_pipeline._load_domains).
    """
    from gpuwrf.runtime import domain_tree as dt

    if os.environ.get("GPUWRF_DECOMPOSE"):
        raise ValueError("aot-key does not cover multi-device decomposition (GPUWRF_DECOMPOSE)")
    keys: dict[str, dict[str, Any]] = {}
    with _key_capture_mode():
        runtime = dt._prepare_operational_domain_tree_runtime(tree, feedback_enabled=bool(tree.feedback_enabled))
        for name in tree.domains:
            record = capture_call(runtime.advance, name, carries[name], start_step, 1, components=components)
            keys[name] = record or {"program": None, "cheap_key": None, "unavailable": "no step program reached"}
        if runtime.fused_cascade is not None:
            for parent in tree.domains:
                fused = runtime.fused_cascade(parent)
                if fused is None:
                    continue
                children = tuple(edge.child for edge in tree.edges.get(parent, ()))
                record = capture_call(fused, carries[parent], tuple(carries[c] for c in children), start_step,
                                      tuple(start_step for _ in children), components=components)
                keys[f"fused/{parent}"] = record or {"program": None, "cheap_key": None,
                                                     "unavailable": "no step program reached"}
    return keys
