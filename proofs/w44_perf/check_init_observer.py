"""Stdlib-only observer forwarding, failure and metadata controls."""
from collections import defaultdict
import importlib.util
from pathlib import Path
import types

spec = importlib.util.spec_from_file_location("observer", Path(__file__).with_name("init_observer.py"))
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
rows = defaultdict(list)
clock = [0.0]


def t():
    clock[0] += 0.01
    return clock[0]


records = {}


def add(key, row):
    assert key in records, "observer did not register its receipt collection"
    rows[key].append(row)


h = types.SimpleNamespace(_t=t, _add=add, REC=records)
listeners = {}
jax = types.SimpleNamespace(monitoring=types.SimpleNamespace(
    register_event_duration_secs_listener=lambda fn: listeners.update(duration=fn),
    register_event_listener=lambda fn: listeners.update(cache=fn)))
marker = object()
status = {"loaded": True, "hlo_sha256": "abc"}
result = (marker, status)


def load(*args, **kw):
    assert args == ("d01",) and kw == {"cheap_key": "key", "return_status": True}
    return result


pipeline = types.SimpleNamespace(_load_domains=load)
aotp = types.SimpleNamespace(_load_domain_blob=load)


class Cache:
    def __init__(self):
        self.events = []

    def __call__(self, fn, *args, **kw):
        assert fn is load and args == (marker,) and kw == {"debug": True}
        self.events.append({"kernel": "init_test", "hit": True, "lower_s": 0.0})
        return marker


o.install(h, jax, pipeline=pipeline, init_kernels=types.SimpleNamespace(InitKernelCache=Cache), aotp=aotp)
assert pipeline._load_domains("d01", cheap_key="key", return_status=True) is result
assert aotp._load_domain_blob("d01", cheap_key="key", return_status=True) is result
assert Cache()(load, marker, debug=True) is marker
assert len(rows["w44_init_load"]) == 2 and len(rows["w44_init_kernel"]) == 1
assert rows["w44_init_kernel"][0]["events"][0]["hit"] is True
try:
    pipeline._load_domains("WRONG")
except AssertionError:
    pass
else:
    raise AssertionError("original failure swallowed")
assert rows["w44_init_load"][-1]["exception_type"] == "AssertionError"


def emit():
    in_avals = (types.SimpleNamespace(shape=(2, 3), dtype="float32", weak_type=False),)
    listeners["duration"]("/jax/core/compile/jaxpr_to_mlir_module_duration", 0.1, fun_name="jit(add)")
    listeners["cache"]("/jax/compilation_cache/cache_hits")
    assert in_avals[0].shape == (2, 3)


emit()
assert rows["w44_init_lowering"][0]["compiler_aval_signature"]["fields"] == [
    {"shape": ["2", "3"], "dtype": "float32", "weak_type": False}]
assert len(rows["w44_init_cache"]) == 1
print("PASS identity/argument forwarding, original failure propagation, init hit, aval-only signature and cache event")
