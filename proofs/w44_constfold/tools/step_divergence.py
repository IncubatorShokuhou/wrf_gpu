"""Execute N production steps of the RELEASE program on CPU and dump the carry (w44-constfold magnitude probe).

Loads the case through the real CLI loader (nested_pipeline._load_domains, max_dom=1, release defaults) like
proofs/o1_nlbind/probe.py, keeps every pallas_call in interpret mode for load AND execution (CPU cannot run the
Triton kernels), runs N steps of operational_mode._advance_chunk_fori and saves all floating State leaves.
Run once with GPUWRF_WRF_REAL_CONSTFOLD=1 and once with =0 (separate processes: import-time flag), then compare
with --compare.  CPU interpret execution is NOT the GPU program bit-for-bit; ON vs OFF on the same backend
isolates the constant-fold change (same inputs, same process recipe).

usage: JAX_PLATFORMS=cpu python step_divergence.py SRC_ROOT CASE_DIR N OUT_NPZ
       python step_divergence.py --compare A.npz B.npz OUT_JSON
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if sys.argv[1] == "--compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    out = {}
    for k in sorted(set(a.files) & set(b.files)):
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        if x.shape != y.shape:
            out[k] = {"shape_mismatch": [list(x.shape), list(y.shape)]}
            continue
        d = np.abs(x - y)
        out[k] = {"finite": bool(np.isfinite(x).all() and np.isfinite(y).all()), "max_abs": float(d.max()),
                  "rms": float(np.sqrt((d ** 2).mean())), "cells_diff": int((x != y).sum()), "cells": int(x.size),
                  "field_max_abs": float(np.abs(x).max())}
    Path(sys.argv[4]).write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out.items():
        if v.get("cells_diff"):
            print(f"{k:14s} diff cells {v['cells_diff']:7d}/{v['cells']:7d} max {v['max_abs']:.3e} rms {v['rms']:.3e}")
    print("identical leaves:", sum(1 for v in out.values() if v.get("cells_diff") == 0), "of", len(out))
    sys.exit(0)

assert os.environ.get("JAX_PLATFORMS") == "cpu"
root, case, nsteps, out_npz = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind")]
import jax  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.backends import pallas as facade  # noqa: E402
from gpuwrf.contracts import real_constfold as rc  # noqa: E402
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu" and om.__file__.startswith(str(root))
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
_real = pl.pallas_call


def _interp(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real(*a, **k)


pl.pallas_call = _interp
facade.cuda_native = lambda: False
with tempfile.TemporaryDirectory(dir="<DATA_ROOT>/wrf_gpu2/scratch/w44-constfold") as tmp:
    c = Path(tmp) / "case"
    c.mkdir()
    for f in case.iterdir():
        (c / f.name).symlink_to(f)
    cfg = pipeline.NestedPipelineConfig(c, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
facade.cuda_native = lambda: True  # release dispatch at trace (as release_identity.py), Pallas interpreted
ns, carry = bundles["d01"].namelist, carries["d01"]
om._resolve_operational_suite(ns)
cb = om.build_clock_base(ns)
for i in range(nsteps):
    carry = om._advance_chunk_fori(carry, ns, i + 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
    jax.block_until_ready(carry)
    print("step", i + 1, flush=True)
leaves = {}
st = carry.state
for name in st.__slots__ if hasattr(st, "__slots__") else vars(st):
    v = getattr(st, name, None)
    if v is not None and hasattr(v, "dtype") and np.issubdtype(np.asarray(v).dtype, np.floating):
        leaves[name] = np.asarray(v)
land = getattr(carry, "noahmp_land", None)
if land is not None:
    for name, v in (land._asdict().items() if hasattr(land, "_asdict") else vars(land).items()):
        if v is not None and hasattr(v, "dtype") and np.issubdtype(np.asarray(v).dtype, np.floating):
            leaves["land_" + name] = np.asarray(v)
np.savez(out_npz, **leaves)
print("flag", rc.ENABLED, "leaves", len(leaves), "steps", nsteps)
