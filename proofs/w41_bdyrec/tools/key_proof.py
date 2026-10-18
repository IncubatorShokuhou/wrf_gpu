"""CPU lowering proof: step-program identities for one case directory.

usage: JAX_PLATFORMS=cpu python key_proof.py SRC_ROOT CASE_DIR MAX_DOM OUT_JSON
Loads the case through nested_pipeline._load_domains (the CLI loader), drives the
real domain-tree runner for one root step with nested AOT on, and intercepts the
cheap-key computation of every step program (d01 advance, fused cascade): records
the cheap key, the lowered StableHLO digest and the argument avals, then aborts
before compiling. Also records every jit compile of the load (jax_log_compiles).
"""
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

src, case, max_dom, out = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0, str(Path(src) / "src"))
os.environ.setdefault("GPUWRF_NESTED_AOT", "1")
import jax  # noqa: E402
import gpuwrf  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert Path(gpuwrf.__file__).resolve().is_relative_to(Path(src).resolve()), gpuwrf.__file__

compiles = []


class _Grab(logging.Handler):
    def emit(self, record):
        message = record.getMessage()
        if "Compiling" in message or "compiling" in message:
            compiles.append(message[:400])


handler = _Grab()
for name in ("jax._src.interpreters.pxla", "jax._src.dispatch", "jax"):
    logging.getLogger(name).addHandler(handler)
    logging.getLogger(name).setLevel(logging.WARNING)
jax.config.update("jax_log_compiles", True)

from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import aot_cheap_key as ck  # noqa: E402
from gpuwrf.runtime import domain_tree as dt  # noqa: E402

names = pipeline.domain_names_for(max_dom)
t0 = time.time()
cfg = pipeline.NestedPipelineConfig(case, out.parent / "kp_out", out.parent / "kp_proof", hours=3, max_dom=max_dom)
hierarchy, bundles, meta, run_start, dt_by_domain, carries = pipeline._load_domains(cfg, names)
load_compiles = list(compiles)
compiles.clear()
load_s = time.time() - t0

records = {}


class _Stop(Exception):
    pass


def _avals(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return hashlib.sha256(repr([(tuple(getattr(x, "shape", ())), str(getattr(x, "dtype", type(x).__name__)))
                                for x in leaves]).encode()).hexdigest()


def _record(label, key, fn, args, kwargs):
    t1 = time.time()
    text = fn.lower(*args, **kwargs).as_text()
    records[label] = {
        "cheap_key": key, "lowered_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "lowered_chars": len(text), "arg_avals_sha256": _avals(args), "lower_s": round(time.time() - t1, 1),
        "rank5_leading_dims": sorted({int(x.shape[0]) for x in jax.tree_util.tree_leaves(args)
                                      if getattr(x, "ndim", 0) == 5}),
    }


original_memo_key = dt._CheapKeyMemo.key


def memo_key(self, carry, namelist, start, clock_base, *, n_steps, cadence):
    key = original_memo_key(self, carry, namelist, start, clock_base, n_steps=n_steps, cadence=cadence)
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori
    _record("advance", key, _advance_chunk_fori, (carry, namelist, start, clock_base),
            {"n_steps": n_steps, "cadence": cadence})
    raise _Stop


original_fused_key = dt._fused_cascade_cheap_key


def fused_key(fused_jit, fused_aux, parent_carry, child_carries, parent_start, child_starts, parent_cb, child_cbs):
    key = original_fused_key(fused_jit, fused_aux, parent_carry, child_carries, parent_start, child_starts,
                             parent_cb, child_cbs)
    raise _Stop((key, (parent_carry, tuple(child_carries), int(parent_start), tuple(child_starts))))


dt._CheapKeyMemo.key = memo_key
dt._fused_cascade_cheap_key = fused_key
tree = dt.DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
runtime = dt._prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
def _closure(fn, name):
    return next(c.cell_contents for c in fn.__closure__ if getattr(c.cell_contents, "__name__", "") == name)


try:
    runtime.advance(names[0], carries[names[0]], 1, 1)
except _Stop:
    pass
for parent in names:
    fused = runtime.fused_cascade(parent)
    if fused is None:
        continue
    children = tuple(edge.child for edge in tree.edges.get(parent, ()))
    try:
        fused(carries[parent], tuple(carries[c] for c in children), 1, tuple(1 for _ in children))
    except _Stop as stop:
        key, _ = stop.args[0]
        starts = tuple(1 for _ in children)
        try:  # lane tree: windows prepared exactly as fused() does
            call = _closure(fused, "_fused_call")
            pc, cc, clocks = _closure(fused, "_prepare_windows")(
                carries[parent], tuple(carries[c] for c in children), 1, starts)
            args = _closure(call, "_args")(pc, cc, 1, starts, clocks)
        except StopIteration:  # released tree (no record windows)
            call = fused
            args = _closure(call, "_args")(carries[parent], tuple(carries[c] for c in children), 1, starts)
        _record(f"fused/{parent}", key, _closure(call, "fused_jit"), args, {})
result = {"src": str(Path(src).resolve()), "case": str(case), "max_dom": max_dom, "load_s": round(load_s, 1),
          "programs": records,
          "record_windows": {name: (meta["domains"].get(name) or {}).get("record_windows") for name in names},
          "load_compiles": load_compiles, "step_compiles": list(compiles)}
out.write_text(json.dumps(result, indent=1, default=str))
print(json.dumps({"programs": records, "record_windows": result["record_windows"],
                  "n_load_compiles": len(load_compiles)}, default=str))
