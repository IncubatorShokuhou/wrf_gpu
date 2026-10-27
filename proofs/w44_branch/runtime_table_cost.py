"""CPU startup metadata cost/inventory, never a forecast speed measurement."""
import argparse
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("tree", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"
sys.path[:0] = [str(args.tree / "scripts/v025"), str(args.tree / "src")]
import cpu_guard  # noqa: F401
from gpuwrf.runtime.runtime_data_identity import runtime_data_identity, check_reader_coverage

bundle = {"d01": SimpleNamespace(namelist=SimpleNamespace(mp_physics=8))}
trials = []
for _ in range(3):
    start = time.perf_counter()
    identity = runtime_data_identity(bundle)
    trials.append(time.perf_counter() - start)
result = {"scope": "CPU shared-load one-time runtime-data identity, mp8/common tables; not model/GPU timing",
          "times_s": trials, "first_s": trials[0], "median_s": statistics.median(trials),
          "logical_bound_files": len(identity["files"]),
          "logical_bound_bytes": sum(value.get("bytes", 0) for value in identity["files"].values()),
          "coverage": check_reader_coverage(), "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "identity": identity,
          "notes": "Logical bytes include source/data aliases; one call deduplicates resolved files. No across-call mtime cache; hash buffers are at most 2 MiB live. Inactive large hail/Jensen assets skipped using actual loader selection."}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "identity"}))
