"""Diagnostic-only host init attribution. No tensor values or device operations.

Durations can overlap/nest. These observed walls must never replace plain S0
timing. Array signatures use compiler aval shape/dtype metadata only.
"""
import functools
import sys


def stack_metadata():
    callers, signature = [], None
    frame = sys._getframe(1)
    try:
        for _ in range(48):
            if frame is None:
                break
            module = frame.f_globals.get("__name__", "")
            if module.startswith("gpuwrf") and len(callers) < 12:
                callers.append({"module": module, "function": frame.f_code.co_name, "line": frame.f_lineno})
            if signature is None:
                for key in ("in_avals", "arg_avals", "avals"):
                    vals = frame.f_locals.get(key)
                    if isinstance(vals, (tuple, list)) and vals:
                        fields = []
                        for v in vals[:32]:
                            shape, dtype = getattr(v, "shape", None), getattr(v, "dtype", None)
                            if shape is not None and dtype is not None:
                                fields.append({"shape": [str(dim) for dim in shape], "dtype": str(dtype),
                                               "weak_type": bool(getattr(v, "weak_type", False))})
                        if fields:
                            signature = {"local_name": key, "fields": fields, "truncated": len(vals) > 32}
                            break
            frame = frame.f_back
    finally:
        del frame
    return {"gpuwrf_callers": callers, "compiler_aval_signature": signature}


def install(h, jax, *, pipeline=None, init_kernels=None, aotp=None):
    for key in ("w44_init_lowering", "w44_init_cache", "w44_init_load", "w44_init_kernel"):
        h.REC.setdefault(key, [])
    def duration(event, seconds, **kw):
        if event.startswith("/jax/core/compile/"):
            end = h._t()
            h._add("w44_init_lowering", {"event": event, "start": end - seconds, "end": end,
                                         "s": seconds, "fun": kw.get("fun_name"), **stack_metadata()})

    def cache(event, **kw):
        if event.startswith("/jax/compilation_cache/"):
            h._add("w44_init_cache", {"event": event, "t": h._t(), **stack_metadata()})

    jax.monitoring.register_event_duration_secs_listener(duration)
    jax.monitoring.register_event_listener(cache)
    # Register first: importing pipeline itself can lower eager table operations.
    if pipeline is None:
        import gpuwrf.integration.nested_pipeline as pipeline
    if init_kernels is None:
        import gpuwrf.integration.init_kernels as init_kernels
    if aotp is None:
        import gpuwrf.runtime.aot_precompile as aotp

    def observe_load(original, name):
        @functools.wraps(original)
        def call(*args, **kwargs):
            start = h._t()
            try:
                result = original(*args, **kwargs)
                status = result[1] if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict) else {}
                h._add("w44_init_load", {"name": name, "start": start, "end": h._t(),
                                         "program": args[0] if args and isinstance(args[0], str) else None,
                                         "cheap_key": kwargs.get("cheap_key"), "hlo_sha256": kwargs.get("hlo_sha256"),
                                         "status": {k: status[k] for k in ("source", "loaded", "real_load", "load_wall_s",
                                                                             "prefetch_load_s", "hlo_sha256", "reason") if k in status}})
                return result
            except BaseException as exc:
                h._add("w44_init_load", {"name": name, "start": start, "end": h._t(), "exception_type": type(exc).__name__})
                raise
        return call

    pipeline._load_domains = observe_load(pipeline._load_domains, "_load_domains")
    aotp._load_domain_blob = observe_load(aotp._load_domain_blob, "_load_domain_blob")
    original_init = init_kernels.InitKernelCache.__call__

    @functools.wraps(original_init)
    def init_call(self, fn, *args, **kwargs):
        start = h._t()
        before = len(self.events)
        try:
            return original_init(self, fn, *args, **kwargs)
        finally:
            events = [{k: row[k] for k in ("kernel", "hlo_sha256", "cheap_key", "hit", "lower_s", "load_s", "setup_dispatch_s") if k in row}
                      for row in self.events[before:]]
            h._add("w44_init_kernel", {"name": getattr(fn, "__name__", type(fn).__name__),
                                       "start": start, "end": h._t(), "events": events})

    init_kernels.InitKernelCache.__call__ = init_call
