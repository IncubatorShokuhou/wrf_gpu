"""Location-stripped jaxpr identity of the actual WN3 step programs for ONE tree (CPU, no compile, no execution).

usage: JAX_PLATFORMS=cpu python program_identity.py SRC_ROOT CASE_DIR OUT_JSON
Loads CASE through the CLI loader (nested_pipeline._load_domains, the case's own namelist, here icloud = 0), drives the
real domain-tree runner for one root step with nested AOT on (technique of w41-bundlekey key_diff.py) and intercepts
the d01 advance program and every fused child cascade (d01 -> d02 -> d03) at their cheap-key call, before any compile.
For each program: jax.make_jaxpr of the exact call, canonicalised by dropping source locations (name_and_src_info,
'/path.py:line[:col]', E109/E151), sha256 of that text, sha256 of every closed-over constant's bytes, equation count.
Two trees whose records are equal trace the same model program (radiation default path included: the radiation call is
a branch of the production step loop). The cheap key is recorded too (it hashes source/env and may differ).
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

src, case, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(Path(src) / "src"))
os.environ.setdefault("GPUWRF_NESTED_AOT", "1")
import jax  # noqa: E402
import numpy as np  # noqa: E402

import gpuwrf  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert Path(gpuwrf.__file__).resolve().is_relative_to(Path(src).resolve()), gpuwrf.__file__
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
import gpuwrf.contracts.state as _state_mod  # noqa: E402

try:
    _state_mod._gpu_device()
except RuntimeError:
    _state_mod._gpu_device = lambda: jax.devices("cpu")[0]
from gpuwrf.runtime import aot_cheap_key as ck  # noqa: E402
from gpuwrf.runtime import domain_tree as dt  # noqa: E402

LOC = re.compile(r"/[^ \n:\"']*\.py:\d+(:\d+)?")


def canon(text: str) -> str:
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", text)
    return LOC.sub("", text)


def record(fn, args, kwargs, namelist, traced, call):
    """``traced``: the carry operands (abstract inputs); everything else is closed over as in production."""
    t1 = time.time()
    closed = jax.make_jaxpr(call)(*traced)
    text = canon(str(closed))
    h = hashlib.sha256()
    for c in closed.consts:
        a = np.asarray(c)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    try:
        key = ck.cheap_key(fn, args, kwargs, namelist)
    except Exception as exc:  # noqa: BLE001
        key = f"ERROR {type(exc).__name__}"
    return {"jaxpr_stripped_sha256": hashlib.sha256(text.encode()).hexdigest(), "jaxpr_chars": len(text),
            "n_eqns_top": len(closed.jaxpr.eqns), "n_consts": len(closed.consts), "consts_sha256": h.hexdigest(),
            "cheap_key": key, "trace_s": round(time.time() - t1, 1)}


names = pipeline.domain_names_for(3)
t0 = time.time()
cfg = pipeline.NestedPipelineConfig(case, out.parent / "pi_out", out.parent / "pi_proof", hours=3, max_dom=3)
hierarchy, bundles, meta, run_start, dt_by_domain, carries = pipeline._load_domains(cfg, names)
load_s = time.time() - t0
icloud = {d: int(getattr(bundles[d].namelist, "icloud", 1)) for d in names}
records = {}


class _Stop(Exception):
    pass


def memo_key(self, carry, namelist, start, clock_base, *, n_steps, cadence):
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori
    kw = {"n_steps": n_steps, "cadence": cadence}
    records["advance/d01"] = record(_advance_chunk_fori, (carry, namelist, start, clock_base), kw, namelist, (carry,),
                                    lambda c: _advance_chunk_fori(c, namelist, start, clock_base, **kw))
    raise _Stop


def fused_key(fused_jit, fused_aux, parent_carry, child_carries, parent_start, child_starts, parent_cb, child_cbs,
              namelists=()):
    args = (parent_carry, child_carries, int(parent_start), tuple(int(s) for s in child_starts), parent_cb, child_cbs,
            *namelists)
    rest = args[2:]
    raise _Stop(record(fused_jit, args, {}, fused_aux, (parent_carry, child_carries),
                       lambda pc, cc: fused_jit(pc, cc, *rest)))


dt._CheapKeyMemo.key = memo_key
dt._fused_cascade_cheap_key = fused_key
tree = dt.DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
runtime = dt._prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
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
        records[f"fused/{parent}"] = stop.args[0]
result = {"src": str(Path(src).resolve()), "head": os.popen(f"git -C {src} rev-parse HEAD").read().strip(),
          "case": str(case), "namelist_icloud": icloud, "load_s": round(load_s, 1), "programs": records}
out.write_text(json.dumps(result, indent=1, default=str))
print(json.dumps({k: (v["jaxpr_stripped_sha256"][:16], v["consts_sha256"][:16], v["jaxpr_chars"]) for k, v in records.items()}))
