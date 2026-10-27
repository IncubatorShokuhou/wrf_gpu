"""Compiled W2 static-holder/clock-operand check on retained real land columns.

This is regression/value-transport evidence. The pristine Fortran oracle is a
separate gate, and CPU interpreted Pallas is not a GPU performance measurement.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--names", required=True)
p.add_argument("--cold", action="store_true")
p.add_argument("--factor", type=float, default=1.1)
p.add_argument("--roundoff", action="store_true", help="report manager-authorized ON-default roundoff; no automatic acceptance")
p.add_argument("--dump-optimized", action="store_true", help="retain optimized OFF/ON HLO for measured association diagnosis")
a = p.parse_args()
sys.path[:0] = [str(a.repo / "src"), str(a.repo / "proofs/w44_lsmparams")]
import jax
import numpy as np
from columns import build
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.io.noahmp_options import NoahMPOptions
from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime.operational_mode import OperationalNamelist, build_clock_base
from gpuwrf.surface_params import controls as sc

platform = jax.devices()[0].platform
assert platform == "cpu"
names = a.names.split(",")
options = NoahMPOptions(opt_crs=2) if "rsmin" in names else None
land, forcing, static, ep, rp, fixture_nl, rows = build(cold=a.cold, options=options)
dt = float(rows[0]["dt"])
off = OperationalNamelist(grid=None, tendencies=None, metrics=None, dt_s=dt,
                          use_noahmp=True, noahmp_static=static, noahmp_land=land,
                          noahmp_energy_params=ep, noahmp_rad_params=rp, noahmp_options=options)
defaults = sc.prior_controls(fixture_nl, names)
changed = sc.expand_controls(fixture_nl, {name: a.factor for name in names})
arms = [off, dataclasses.replace(off, surface_controls=defaults),
        dataclasses.replace(off, surface_controls=changed)]
traces = []


def program(land, forcing, nl, cb):
    traces.append(1)
    s, ep, rp = nl.noahmp_static, nl.noahmp_energy_params, nl.noahmp_rad_params
    values = sc.traced_controls(nl, cb)
    if values is not None:
        s, ep, rp = sc.apply_surface_controls(s, ep, rp, values)
    return noah_mp_step(land, forcing, s, dt, energy_params=ep, rad_params=rp,
                        options=nl.noahmp_options, return_diag=True)


fn = jax.jit(program)
results, keys, cache_sizes = [], [], []
paths = None
for arm_index, nl in enumerate(arms):
    cb = build_clock_base(nl)
    key = ck.cheap_key(fn, (land, forcing, nl, cb), {}, nl)
    assert key is not None, "cheap key unavailable; no SAME-key claim"
    keys.append(key)
    output = fn(land, forcing, nl, cb)
    if paths is None:
        paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(output)[0]]
    if a.dump_optimized and arm_index < 2:
        compiled = fn.lower(land, forcing, nl, cb).compile()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        (a.out.parent / (a.out.stem + ("_off" if arm_index == 0 else "_on") + ".optimized.hlo")).write_text(compiled.as_text())
    results.append([np.asarray(x).copy() for x in jax.tree.leaves(output)])
    cache_sizes.append(fn._cache_size())
assert all(np.isfinite(x).all() for values in results for x in values), "non-finite output"
identical = [x.tobytes() == y.tobytes() for x, y in zip(results[0], results[1], strict=True)]
changed_leaves = sum(x.tobytes() != y.tobytes() for x, y in zip(results[1], results[2], strict=True))
differences = []
for path, x, y in zip(paths, results[0], results[1], strict=True):
    if x.tobytes() == y.tobytes():
        continue
    delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
    if x.dtype == np.float32:
        def ordered(v):
            bits = v.view(np.int32).astype(np.int64)
            return np.where(bits < 0, -(bits & 0x7fffffff), bits)
        ulps = np.abs(ordered(x) - ordered(y))
        max_ulps = int(ulps.max())
    else:
        max_ulps = None
    differences.append({"path": path, "dtype": str(x.dtype), "shape": list(x.shape),
                        "changed_cells": int(np.count_nonzero(x != y)), "max_abs": float(delta.max()),
                        "max_rel": float((delta / np.maximum(np.abs(x.astype(np.float64)), np.finfo(np.float64).tiny)).max()),
                        "max_ulps": max_ulps})
receipt = {
    "source": str(a.repo.resolve()), "platform": platform, "gpu_executed": False,
    "scope": "compiled CPU native-physics/interpreted Pallas; regression, not a WRF oracle or GPU timing",
    "names": names, "factor": a.factor, "cold": a.cold,
    "columns": [row["name"] for row in rows], "dt": dt,
    "keys": keys, "cache_sizes": cache_sizes, "traces": len(traces),
    "output_leaves": len(identical), "default_equal_leaves": sum(identical),
    "default_byte_exact": all(identical), "changed_leaves": changed_leaves,
    "differences": differences, "output_paths": paths,
    "default_deltas": [float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                       for x, y in zip(results[0], results[1], strict=True)],
    "output_sha256": [[hashlib.sha256(x.tobytes()).hexdigest() for x in values] for values in results],
}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k not in ("output_sha256", "default_deltas")}), flush=True)
assert keys[1] == keys[2], "values changed the actual cheap key"
assert cache_sizes == [1, 2, 2] and len(traces) == 2, "changed values retraced/recompiled"
if not a.roundoff:
    assert all(identical), "ON defaults differ from OFF: measure cause/ULPs/oracle under manager rule"
assert changed_leaves > 0, "control has no observed consumer on this cohort"
