"""Halo bytes per column of the RELEASE carry (fast defaults) for the cost model (lane o2-multigpu M4).

Loads the case under the release defaults on CPU (Pallas interpret patches; nothing is executed beyond the
product loader), classifies every carry leaf, traces ONE release step body for the live-leaf set
(tile_driver.release_step_live_leaves) and reports bytes per horizontal column: all ``h`` leaves vs live ones.

  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_FAST_DEFAULTS=1 GPUWRF_WRF_ROOT=... PYTHONPATH=src \
    python release_bytes_probe.py --input CASE --out OUT.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"
t0 = time.perf_counter()
import gpuwrf  # noqa: F401,E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

spec_ = importlib.util.spec_from_file_location("gpu_smoke_v034", Path(__file__).resolve().parents[2] / "scripts/gpu_smoke_v034.py")
smoke = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(smoke)
smoke._cpu_patches()
from gpuwrf.parallel import decomposition as dc  # noqa: E402
from gpuwrf.parallel.harness import load_case  # noqa: E402
from gpuwrf.parallel.tile_driver import release_step_live_leaves  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

carry, nl, meta = load_case(args.input, args.out.parent / "scratch_bytes", device=jax.devices()[0])
spec = dc.DecompositionSpec.build(nl.grid.nx, nl.grid.ny, 1, 1, 0)
leaves = jax.tree_util.tree_leaves(carry)
kinds = [dc.classify(np.shape(x), spec) for x in leaves]
t1 = time.perf_counter()
live = release_step_live_leaves(carry, nl, om.build_clock_base(nl), int(nl.radiation_cadence_steps))
trace_s = time.perf_counter() - t1


def per_col(sel):
    return int(sum(int(np.prod(np.shape(x)[:-2], dtype=np.int64)) * np.dtype(x.dtype).itemsize
                   for x, k, s in zip(leaves, kinds, sel) if k[0] == "h" and s))


dtypes = {}
for x, k in zip(leaves, kinds):
    if k[0] == "h":
        dtypes[str(x.dtype)] = dtypes.get(str(x.dtype), 0) + 1
rec = {"case": meta, "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"), "leaves": len(leaves),
       "h_leaves": sum(k[0] == "h" for k in kinds), "h_leaf_dtypes": dtypes,
       "live_h_leaves": sum(1 for k, s in zip(kinds, live) if k[0] == "h" and s),
       "bytes_per_column_all_h": per_col([True] * len(leaves)), "bytes_per_column_live_h": per_col(live),
       "carry_bytes_total": int(sum(np.asarray(x).nbytes for x in leaves)), "live_trace_s": round(trace_s, 1),
       "process_s": round(time.perf_counter() - t0, 1)}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec))
