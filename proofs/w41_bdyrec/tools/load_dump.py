"""Load one case through the CLI loader and dump everything the step programs receive.

usage: JAX_PLATFORMS=cpu python load_dump.py SRC_ROOT CASE_DIR MAX_DOM OUT_NPZ
Dumps every initial-carry leaf, every bundle-state leaf, the namelist aux4 LowerBoundary and, when a
record-window owner exists, its full host records (lateral + aux4); OUT_NPZ.json lists every jit
compile of the load (jax_log_compiles) and the record-window reports. No stepping, no step compile.
"""
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

src, case, max_dom, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0, str(src / "src"))
import jax  # noqa: E402
import numpy as np  # noqa: E402
import gpuwrf  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert Path(gpuwrf.__file__).resolve().is_relative_to(src), gpuwrf.__file__
compiles = []


class _Grab(logging.Handler):
    def emit(self, record):
        message = record.getMessage()
        if message.startswith("Compiling "):
            compiles.append(message[:400])


handler = _Grab()
for name in ("jax._src.interpreters.pxla", "jax._src.dispatch"):
    logging.getLogger(name).addHandler(handler)
    logging.getLogger(name).setLevel(logging.WARNING)
jax.config.update("jax_log_compiles", True)
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402

names = pipeline.domain_names_for(max_dom)
cfg = pipeline.NestedPipelineConfig(case, out.parent / "ld_out", out.parent / "ld_proof", hours=3, max_dom=max_dom)
t0 = time.time()
_hierarchy, bundles, meta, _start, _dt, carries = pipeline._load_domains(cfg, names)
load_s = time.time() - t0
arrays = {}


def put(prefix, tree):
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if isinstance(leaf, (jax.Array, np.ndarray)):
            arrays[f"{prefix}{jax.tree_util.keystr(path)}"] = np.asarray(leaf)


for name in names:
    put(f"carry/{name}", carries[name])
    put(f"bundle_state/{name}", bundles[name].state)
    namelist = bundles[name].namelist
    if getattr(namelist, "lower_boundary", None) is not None:
        put(f"lower_boundary/{name}", tuple(namelist.lower_boundary))
    owner = getattr(namelist, "record_windows", None)
    if owner is not None:
        if owner.lateral:
            for key, value in owner._lateral.items():
                arrays[f"owner_lateral/{name}/{key}"] = value
        if owner.aux:
            for i, value in enumerate(owner._aux):
                arrays[f"owner_aux/{name}/{i}"] = value
np.savez(out, **arrays)
summary = {"src": str(src), "case": str(case), "load_s": round(load_s, 1), "leaves": len(arrays),
           "digest": hashlib.sha256(b"".join(k.encode() + v.tobytes() for k, v in sorted(arrays.items()))).hexdigest(),
           "record_windows": {n: (meta["domains"].get(n) or {}).get("record_windows") for n in names},
           "load_compiles": compiles}
out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: summary[k] for k in ("load_s", "leaves", "digest", "record_windows")}, default=str))
