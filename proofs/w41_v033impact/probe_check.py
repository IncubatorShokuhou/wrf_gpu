"""Which case constants does the v0.3.x fused d02/d03 program bake, and which reach its outputs? (CPU, no compile)

usage: JAX_PLATFORMS=cpu python probe_baked.py SRC OUT_DIR CASE_DIR [CASE_DIR ...] [--trace=N]

For every case: load through the product nested loader (nested_pipeline._load_domains), build the product
domain-tree runtime and capture the kwargs of domain_tree._build_fused_cascade_program (the namelists the
fused_jit CLOSES OVER -> baked as HLO constants under v0.3.x). Dump every array leaf of the closed parent/child
OperationalNamelists (d02/d03) to leaves_<case>.npz + a digest table.
For the first N cases (--trace, default 1) also trace the SAME fused_jit body with those namelist leaves lifted to
jaxpr inputs (closure cells swapped, code object unchanged) and run a structured liveness pass (jit/scan/while/
cond/custom aware, fixed point over loop carries) from (a) all outputs and (b) the prognostic State outputs only
-> per baked leaf: consumed by the program output or dead (e.g. superseded by a traced clock_base copy).
A structural jaxpr hash (+ hash of the remaining non-namelist constants) per traced case shows whether two dates
differ ONLY in the namelist leaf values.
"""
import hashlib
import json
import os
import re
import sys
import time
import types
from pathlib import Path

src = sys.argv[1]
out_dir = Path(sys.argv[2])
cases = [Path(a) for a in sys.argv[3:] if not a.startswith("--")]
n_trace = int(next((a.split("=", 1)[1] for a in sys.argv[3:] if a.startswith("--trace=")), "1"))
out_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(src) / "src"))
os.environ.setdefault("GPUWRF_NESTED_AOT", "1")
import gpuwrf  # noqa: E402  (FIRST: applies the release env defaults incl. JAX_ENABLE_X64 before jax is imported)
import jax  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import pallas as _pl  # noqa: E402

assert jax.config.jax_enable_x64, "x64 must be on like the product process"

_orig_pallas_call = _pl.pallas_call


def _interpret_pallas_call(*a, **k):  # probe-only: CPU process, every kernel in interpret mode (E156)
    k["interpret"] = True
    return _orig_pallas_call(*a, **k)


_pl.pallas_call = _interpret_pallas_call

assert jax.devices()[0].platform == "cpu"
assert Path(gpuwrf.__file__).resolve().is_relative_to(Path(src).resolve()), gpuwrf.__file__
import gpuwrf.contracts.state as _st  # noqa: E402

_CPU = jax.devices("cpu")[0]
_st._gpu_device = lambda: _CPU  # probe-only (E126/E156): v0.3.3 State.zeros insists on a GPU device

from jax._src import core as jcore  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import domain_tree as dt  # noqa: E402
from gpuwrf.runtime import operational_mode as op  # noqa: E402

CHILD_NAMES = ("tendencies", "metrics", "radiation_static", "gwdo_statics", "data_assimilation", "noahmp_case",
               "lower_boundary")
NOAH_CASE = ("tbot", "shdfac")
captured = {}
_orig_build = dt._build_fused_cascade_program


def _capture_build(**kw):
    prog = _orig_build(**kw)
    captured[kw["parent_name"]] = (kw, prog)
    return prog


dt._build_fused_cascade_program = _capture_build


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}", flush=True)


def leaf_name(path):
    first = path[0]
    idx = getattr(first, "idx", None)
    if idx is None:
        idx = getattr(first, "key", first)
    name = CHILD_NAMES[idx] if isinstance(idx, int) and idx < len(CHILD_NAMES) else str(first)
    rest = jax.tree_util.keystr(tuple(path[1:]))
    if name == "noahmp_case":
        m = re.match(r"\[(\d+)\]", rest)
        if m:
            rest = "." + NOAH_CASE[int(m.group(1))] + rest[m.end():]
    return name + rest


def namelist_rows(nl):
    flat, treedef = jax.tree_util.tree_flatten_with_path(nl)
    return [(leaf_name(p), leaf) for p, leaf in flat], treedef


def sha(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()


def find_fused_jit(prog, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if id(prog) in seen or depth > 4:
        return None
    seen.add(id(prog))
    wrapped = getattr(prog, "__wrapped__", None)
    if wrapped is not None and getattr(wrapped, "__name__", "") == "fused_jit" and hasattr(prog, "lower"):
        return prog
    for cell in getattr(prog, "__closure__", None) or ():
        try:
            v = cell.cell_contents
        except ValueError:
            continue
        if callable(v):
            hit = find_fused_jit(v, depth + 1, seen)
            if hit is not None:
                return hit
    return None


# ---------------------------------------------------------------- liveness
_memo = {}


def _closed(j):
    return j.jaxpr if isinstance(j, jcore.ClosedJaxpr) else j


def live(jaxpr, used_out):
    """used flags for jaxpr.constvars + jaxpr.invars given used outputs (structured, conservative only for
    primitives without a sub-jaxpr rule: any used output -> all inputs used)."""
    key = (id(jaxpr), tuple(used_out))
    hit = _memo.get(key)
    if hit is not None:
        return hit
    alive = set()
    for v, u in zip(jaxpr.outvars, used_out):
        if u and isinstance(v, jcore.Var):
            alive.add(v)
    for eqn in reversed(jaxpr.eqns):
        uo = [isinstance(v, jcore.Var) and not isinstance(v, jcore.DropVar) and v in alive for v in eqn.outvars]
        if not any(uo):
            continue
        ui = eqn_rule(eqn, uo)
        for v, u in zip(eqn.invars, ui):
            if u and isinstance(v, jcore.Var):
                alive.add(v)
    res = [v in alive for v in list(jaxpr.constvars) + list(jaxpr.invars)]
    _memo[key] = res
    return res


def _inner(jaxpr, used_out):
    j = _closed(jaxpr)
    r = live(j, used_out)
    return r[len(j.constvars):]


def eqn_rule(eqn, uo):
    name = eqn.primitive.name
    p = eqn.params
    n_in = len(eqn.invars)
    if name == "scan":
        body = _closed(p["jaxpr"])
        nc, ncar = p["num_consts"], p["num_carry"]
        used_carry = list(uo[:ncar])
        ys = list(uo[ncar:])
        while True:
            r = live(body, used_carry + ys)[len(body.constvars):]
            new = [a or b for a, b in zip(used_carry, r[nc:nc + ncar])]
            if new == used_carry:
                break
            used_carry = new
        return r[:nc] + used_carry + r[nc + ncar:]
    if name == "while":
        cj, bj = _closed(p["cond_jaxpr"]), _closed(p["body_jaxpr"])
        cn, bn = p["cond_nconsts"], p["body_nconsts"]
        cr = live(cj, [True])[len(cj.constvars):]
        used_carry = [a or b for a, b in zip(uo, cr[cn:])]
        while True:
            br = live(bj, used_carry)[len(bj.constvars):]
            new = [a or b for a, b in zip(used_carry, br[bn:])]
            if new == used_carry:
                break
            used_carry = new
        return cr[:cn] + br[:bn] + used_carry
    if name == "cond":
        acc = [False] * (n_in - 1)
        for br in p["branches"]:
            r = _inner(br, uo)
            acc = [a or b for a, b in zip(acc, r)]
        return [True] + acc
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        sub = p.get(key)
        if isinstance(sub, (jcore.ClosedJaxpr, jcore.Jaxpr)):
            j = _closed(sub)
            if len(j.invars) == n_in and len(j.outvars) == len(eqn.outvars):
                return _inner(sub, uo)
    return [True] * n_in


# ---------------------------------------------------------------- structural hash
_ADDR = re.compile(r" at 0x[0-9a-f]+|0x[0-9a-f]{6,}")
_hmemo = {}


def jhash(jaxpr):
    j = _closed(jaxpr)
    hit = _hmemo.get(id(j))
    if hit is not None:
        return hit
    h = hashlib.sha256()
    ids = {}

    def vid(v):
        if isinstance(v, jcore.Literal):
            val = np.asarray(v.val)
            return f"L{val.dtype}{val.shape}:{sha(val)[:16]}"
        if v not in ids:
            ids[v] = len(ids)
        return f"v{ids[v]}:{v.aval.str_short()}"

    h.update(" ".join(vid(v) for v in list(j.constvars) + list(j.invars)).encode())
    for eqn in j.eqns:
        parts = [eqn.primitive.name]
        for k in sorted(eqn.params):
            val = eqn.params[k]
            if isinstance(val, (jcore.ClosedJaxpr, jcore.Jaxpr)):
                parts.append(f"{k}=J{jhash(val)[:16]}")
            elif isinstance(val, (tuple, list)) and val and all(isinstance(x, (jcore.ClosedJaxpr, jcore.Jaxpr)) for x in val):
                parts.append(f"{k}=[{','.join(jhash(x)[:16] for x in val)}]")
            elif k in ("name_and_src_info", "debug_info", "metadata"):
                continue
            else:
                parts.append(f"{k}={_ADDR.sub('@', repr(val))[:4000]}")
        parts.append("<-" + " ".join(vid(v) for v in eqn.invars))
        parts.append("->" + " ".join(vid(v) for v in eqn.outvars))
        h.update("|".join(parts).encode())
    h.update(("OUT " + " ".join(vid(v) for v in j.outvars)).encode())
    d = h.hexdigest()
    _hmemo[id(j)] = d
    return d


def count_eqns(jaxpr, seen=None):
    seen = seen if seen is not None else set()
    j = _closed(jaxpr)
    if id(j) in seen:
        return 0
    seen.add(id(j))
    n = len(j.eqns)
    for eqn in j.eqns:
        for val in eqn.params.values():
            subs = val if isinstance(val, (tuple, list)) else (val,)
            for s in subs:
                if isinstance(s, (jcore.ClosedJaxpr, jcore.Jaxpr)):
                    n += count_eqns(s, seen)
    return n


# ---------------------------------------------------------------- main (cross-check)
from jax._src.interpreters import partial_eval as pe  # noqa: E402
import dataclasses  # noqa: E402

names = pipeline.domain_names_for(3)
case = cases[0]
tag = case.name
work = out_dir / f"work_{tag}"
cfg = pipeline.NestedPipelineConfig(case, work / "out", work / "proof", hours=3, max_dom=3)
hierarchy, bundles, meta, run_start, dt_by_domain, carries = pipeline._load_domains(cfg, names)
log(f"{tag}: loaded")
tree = dt.DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
runtime = dt._prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
assert runtime.fused_cascade("d02") is not None
kw, prog = captured["d02"]
pnl, cnls = kw["parent_namelist"], tuple(kw["child_namelists"])
fused_jit = find_fused_jit(prog)
target = fused_jit.__wrapped__
code, free = target.__code__, target.__code__.co_freevars
p_rows, p_def = namelist_rows(pnl)
p_leaves = [leaf for _, leaf in p_rows]
c_rows = [namelist_rows(nl) for nl in cnls]
c_leaves, c_defs = [[leaf for _, leaf in r[0]] for r in c_rows], [r[1] for r in c_rows]
leaf_names = [f"d02|{n}" for n, _ in p_rows] + [f"{c}|{n}" for c, r in zip(kw["child_names"], c_rows) for n, _ in r[0]]


def lifted(p_l, c_l, args):
    pnl_t = jax.tree_util.tree_unflatten(p_def, p_l)
    cnls_t = tuple(jax.tree_util.tree_unflatten(d, l) for d, l in zip(c_defs, c_l))
    cells = tuple(types.CellType(pnl_t) if n == "parent_namelist" else
                  types.CellType(cnls_t) if n == "child_namelists" else c
                  for n, c in zip(free, target.__closure__))
    fn = types.FunctionType(code, target.__globals__, target.__name__, target.__defaults__, cells)
    return fn(*args)


children = tuple(kw["child_names"])
args = (carries["d02"], tuple(carries[c] for c in children), 1, tuple(1 for _ in children),
        op.build_clock_base(pnl), tuple(op.build_clock_base(nl) for nl in cnls))
closed, out_shape = jax.make_jaxpr(lifted, return_shape=True)(p_leaves, c_leaves, args)
n_nl = len(leaf_names)
flat_out, out_def = jax.tree_util.tree_flatten(out_shape)
idx_tree = jax.tree_util.tree_unflatten(out_def, list(range(len(flat_out))))
groups = {}
for dom, carry_idx in (("d02", idx_tree[0]), ("d03", idx_tree[1][0])):
    st = carry_idx.state
    fields = [f.name for f in dataclasses.fields(st)] if dataclasses.is_dataclass(st) else list(getattr(st, "__slots__", ()))
    for fname in fields:
        for i in jax.tree_util.tree_leaves(getattr(st, fname)):
            groups[i] = (dom, "state." + fname)
    for fname in ("noahmp_land",):
        sub = getattr(carry_idx, fname, None)
        for i in jax.tree_util.tree_leaves(sub):
            groups[i] = (dom, fname)
state_mask = [i in groups and groups[i][1].startswith("state.") and "_bdy" not in groups[i][1] for i in range(len(flat_out))]
land_mask = [i in groups and groups[i][1] == "noahmp_land" for i in range(len(flat_out))]
rest_mask = [not l for l in land_mask]
log(f"outputs {len(flat_out)}: state(non-bdy) {sum(state_mask)}, noahmp_land {sum(land_mask)}")
# (1) JAX's own conservative DCE (while_loop: all-inputs-used when any output used)
jx = pe.convert_constvars_jaxpr(closed.jaxpr)
t = time.time()
_, used_jax = pe.dce_jaxpr(jx, [True] * len(jx.outvars))
log(f"jax dce {time.time() - t:.1f}s")
ncv = len(closed.jaxpr.constvars)
used_jax = used_jax[ncv:ncv + n_nl]
# (2) structured liveness, three output sets
t = time.time()
res = {k: live(closed.jaxpr, m)[ncv:ncv + n_nl] for k, m in
       (("all", [True] * len(flat_out)), ("state_nonbdy", state_mask), ("all_but_noahmp_land", rest_mask))}
log(f"structured liveness {time.time() - t:.1f}s")
rows = [{"leaf": n, "jax_dce_used": bool(a), **{f"live_{k}": bool(v[i]) for k, v in res.items()}}
        for i, (n, a) in enumerate(zip(leaf_names, used_jax))]
(out_dir / "crosscheck.json").write_text(json.dumps({"case": tag, "n_eqns_total": count_eqns(closed.jaxpr),
    "n_outputs": len(flat_out), "n_state_nonbdy": sum(state_mask), "n_noahmp_land": sum(land_mask), "rows": rows}, indent=1))
for r in rows:
    print(r)
print("DONE")
