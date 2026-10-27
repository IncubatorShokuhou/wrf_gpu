"""CPU stdlib-only checks of observation forwarding and exact capture length.

This tests harness wiring, not weather numerics or GPU execution.
"""
import contextlib
import importlib.util
import sys
import types
from pathlib import Path

spec = importlib.util.spec_from_file_location("capture_under_test", Path(__file__).with_name("capture.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
assert "jax" not in sys.modules
gpuwrf = types.ModuleType("gpuwrf")
runtime = types.ModuleType("gpuwrf.runtime")
dt = types.ModuleType("gpuwrf.runtime.domain_tree")
gpuwrf.runtime = runtime
runtime.domain_tree = dt
sys.modules.update({"gpuwrf": gpuwrf, "gpuwrf.runtime": runtime, "gpuwrf.runtime.domain_tree": dt})
fake_jax = types.SimpleNamespace(profiler=types.SimpleNamespace(TraceAnnotation=lambda _: contextlib.nullcontext()))
c.original_install = lambda _: None

for use_fused, late_start in ((False, False), (True, False), (False, True), (True, True)):
    stopped = []
    seen = []
    marker = object()
    child = object()

    def forward(name, carry, start_step, n_steps):
        assert name == "d01" and carry is marker and n_steps == 1
        seen.append(start_step)
        return marker

    def fused(parent, children, start_step, child_starts):
        assert parent is marker and children == (child,) and child_starts == (start_step * 3,)
        seen.append(start_step)
        return marker, children

    def callbacks(hierarchy, carries, *, advance, force=None, **kwargs):
        assert hierarchy is marker and carries is child
        for step in (range(98, 105) if late_start else range(68, 73)):
            if use_fused:
                result = kwargs["fused_cascade"]("d01")(marker, (child,), step, (step * 3,))
                assert result == (marker, (child,))
            else:
                assert advance("d01", marker, step, 1) is marker
        return marker

    dt.run_domain_tree_callbacks = callbacks
    c.h.PROFILE_SEGMENT.update(k=2, active=True)
    c.h.REC.pop("w44_capture", None)
    c.h._cuda_profiler = lambda event: stopped.append(event)
    c.os.environ["W44_CAPTURE_ROOTS"] = "3"
    if late_start:
        c.os.environ["W44_CAPTURE_START"] = "100"
    else:
        c.os.environ.pop("W44_CAPTURE_START", None)
    c.install(fake_jax)
    if late_start:
        c.h._cuda_profiler("cudaProfilerStart")  # S0 segment-2 entry: must defer
        assert stopped == []
    kwargs = {"fused_cascade": lambda _: fused} if use_fused else {}
    result = dt.run_domain_tree_callbacks(marker, child, advance=forward, **kwargs)
    assert result is marker and seen == list(range(98, 105) if late_start else range(68, 73))
    assert stopped == (["cudaProfilerStart", "cudaProfilerStop"] if late_start else ["cudaProfilerStop"])
    expected = {"complete_roots": 3, "stopped_before_root_step": 103 if late_start else 71}
    if late_start:
        expected["first_root_step"] = 100
    assert c.h.REC["w44_capture"] == expected
    assert c.h.PROFILE_SEGMENT["active"] is False
    print(f"PASS forwarding, object identity, exact capture; fused={use_fused}, radiation_window={late_start}")

assert "jax" not in sys.modules
