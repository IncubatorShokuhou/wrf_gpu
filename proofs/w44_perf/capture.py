"""Common s0 harness, optionally stop profiling after three complete PROD roots.

Only host call boundaries are wrapped. No array observations, JIT functions,
physics switches or model return values change. S0's GPU assertion is retained.
"""
import importlib.util
import os
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "scripts/bench/s0_nested_harness.py"
spec = importlib.util.spec_from_file_location("w44_s0", path)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
original_install = h._install_wrappers


def install(jax):
    if os.environ.get("W44_INIT_OBSERVER") == "1":
        import init_observer
        init_observer.install(h, jax)
    original_install(jax)
    if os.environ.get("W44_HOST_PROFILE") == "1":
        import host_observer
        host_observer.install(h)
    import gpuwrf.runtime.domain_tree as dt
    original_callbacks = dt.run_domain_tree_callbacks
    count = 0
    capture_start = int(os.environ.get("W44_CAPTURE_START", "0"))
    started = not capture_start
    original_profiler = h._cuda_profiler
    if capture_start:
        def deferred_profiler(event):
            if event != "cudaProfilerStart":
                original_profiler(event)
        h._cuda_profiler = deferred_profiler

    def before_root(name, start_step):
        nonlocal count, started
        if os.environ.get("W44_CAPTURE_ROOTS") == "3" and h.PROFILE_SEGMENT["active"] and name == "d01":
            if not started:
                if int(start_step) < capture_start:
                    return
                assert int(start_step) == capture_start, "capture skipped the intended root"
                original_profiler("cudaProfilerStart")
                started = True
            if count == 3:
                h._cuda_profiler("cudaProfilerStop")
                h.PROFILE_SEGMENT["active"] = False
                h.REC["w44_capture"] = {"complete_roots": 3, "stopped_before_root_step": int(start_step)}
                if capture_start:
                    h.REC["w44_capture"]["first_root_step"] = capture_start
            count += 1

    def callbacks(hierarchy, carries, *, advance, force=None, **kwargs):
        def adv(name, carry, start_step, n_steps):
            before_root(name, start_step)
            return advance(name, carry, start_step, n_steps)
        lookup = kwargs.get("fused_cascade")
        if lookup is not None:
            def fused_lookup(name):
                fn = lookup(name)
                if fn is None:
                    return None

                def fused(*args, **kw):
                    before_root(name, args[2])
                    t = h._t()
                    with jax.profiler.TraceAnnotation(f"W44:fused:{name}:{int(args[2])}"):
                        result = fn(*args, **kw)
                    h._add("advance", {"d": "fused_" + name, "s0": int(args[2]), "n": 1,
                                       "t": round(t, 4), "host_s": round(h._t() - t, 6)})
                    return result

                return fused

            kwargs["fused_cascade"] = fused_lookup
        return original_callbacks(hierarchy, carries, advance=adv, force=force, **kwargs)

    dt.run_domain_tree_callbacks = callbacks


h._install_wrappers = install

if __name__ == "__main__":
    if os.environ.get("W44_LAUNCH_SMOKE") == "1":
        import launch_smoke_cpu
        raise SystemExit(launch_smoke_cpu.main(h))
    rc = h.main()
    if rc == 0 and os.environ.get("W44_CAPTURE_ROOTS") == "3":
        assert h.REC.get("w44_capture", {}).get("complete_roots") == 3, "three-root capture did not stop"
    raise SystemExit(rc)
