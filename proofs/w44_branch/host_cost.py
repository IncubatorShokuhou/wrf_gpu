"""One-time host branch cost on original WN3 input views; not a forecast bench."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
import tracemalloc
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("tree", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"
sys.path[:0] = [str(args.tree / "scripts/v025"), str(args.tree / "src")]
import cpu_guard  # noqa: F401
from gpuwrf.runtime.checkpoint_branch import apply_overrides
from gpuwrf.runtime.checkpoint_consistency import checkpoint_validator
import jax
assert jax.devices()[0].platform == "cpu"

source = args.tree / "tests/v025/restart/test_branch_consistency.py"
loader = importlib.util.spec_from_file_location("branch_real_view", source)
module = importlib.util.module_from_spec(loader)
loader.loader.exec_module(module)
snapshot = {"carries": {}, "own_steps": {}, "driver_state": {"outputs": {}, "forecast_hours": 3}}
bundles = {}
for name in ("d01", "d02", "d03"):
    _, item, _ = module.real_input.__wrapped__(SimpleNamespace(param=name))
    snapshot["carries"].update(item["carries"])
    snapshot["own_steps"].update(item["own_steps"])
    snapshot["driver_state"]["outputs"][name] = []
    bundles[name] = SimpleNamespace(namelist=SimpleNamespace(sf_surface_physics=4))
validator = checkpoint_validator(module.CASE, bundles)
result = {"backend": "cpu", "scope": "one-time host branch on original 0227 real.exe arrays; shared CPU load; not full checkpoint/GPU/forecast timing",
          "cache": "NONE", "modes": {}}
tracemalloc.start()
for mode in ("null", "qv"):
    changes = {"schema_version": 1}
    if mode == "qv":
        changes["domains"] = {name: {"state": {"qv": {"multiply": .99}}} for name in bundles}
        changes["waivers"] = {name: {"hydrostatic_consistency": "Research fixed-MU host-cost probe",
                                     "supersaturation_bound": "Source model-state research admission"} for name in bundles}
    trials, peaks = [], []
    for _ in range(3):
        tracemalloc.reset_peak()
        base_bytes = tracemalloc.get_traced_memory()[0]
        started = time.perf_counter()
        updated, report = apply_overrides(snapshot, changes, validate=validator)
        trials.append(time.perf_counter() - started)
        peaks.append(tracemalloc.get_traced_memory()[1] - base_bytes)
        assert all(v["passed"] for v in report["consistency"].values())
        del updated, report
    result["modes"][mode] = {"seconds": trials, "median_s": statistics.median(trials),
                             "traced_allocation_peak_bytes": max(peaks)}
result["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
