"""Inventory the real single-domain carry + namelist leaves (path, shape, dtype) on CPU (lane o2-multigpu, M2 design).

Usage: JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 GPUWRF_WRF_ROOT=... python inventory_probe.py --input CASE --out DIR
"""
from __future__ import annotations

import argparse
import dataclasses
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
import gpuwrf  # noqa: F401
import jax
import numpy as np
from gpuwrf.contracts import state as state_contract

state_contract._gpu_device = lambda: jax.devices("cpu")[0]
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains

config = NestedPipelineConfig(args.input, args.out / "stream", args.out / "proof", hours=1, max_dom=1)
hierarchy, bundles, meta, run_start, dts, carries = _load_domains(config, ("d01",))
load_s = time.perf_counter() - t0
nl, carry = bundles["d01"].namelist, carries["d01"]
rows = []
for path, leaf in jax.tree_util.tree_flatten_with_path(carry)[0]:
    rows.append({"tree": "carry", "path": jax.tree_util.keystr(path), "shape": list(np.shape(leaf)), "dtype": str(np.asarray(leaf).dtype)})
for path, leaf in jax.tree_util.tree_flatten_with_path(nl)[0]:
    rows.append({"tree": "namelist_children", "path": jax.tree_util.keystr(path), "shape": list(np.shape(leaf)), "dtype": str(np.asarray(leaf).dtype)})


def walk(obj, prefix, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, (np.ndarray, jax.Array)):
        rows.append({"tree": "namelist_all", "path": prefix, "shape": list(obj.shape), "dtype": str(obj.dtype), "type": type(obj).__name__})
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            walk(getattr(obj, f.name), f"{prefix}.{f.name}", seen)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        for k, v in zip(obj._fields, obj):
            walk(v, f"{prefix}.{k}", seen)
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            walk(v, f"{prefix}[{i}]", seen)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{prefix}[{k!r}]", seen)
    elif obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return
    else:
        rows.append({"tree": "namelist_all", "path": prefix, "type": type(obj).__module__ + "." + type(obj).__qualname__, "opaque": True})


walk(nl, "nl", set())
g = nl.grid
summary = {"load_s": round(load_s, 1), "nx": g.nx, "ny": g.ny, "nz": g.nz, "dt": dts, "acoustic_substeps": nl.acoustic_substeps,
           "radiation_cadence_steps": nl.radiation_cadence_steps, "n_rows": len(rows), "run_start": str(run_start)}
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "inventory.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
print(json.dumps(summary))
