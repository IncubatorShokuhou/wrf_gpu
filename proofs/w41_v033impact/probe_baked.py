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


# ---------------------------------------------------------------- main
names = pipeline.domain_names_for(3)
summary = {"src": str(Path(src).resolve()), "jax": jax.__version__, "cases": {}}
for ci, case in enumerate(cases):
    tag = case.name
    captured.clear()
    t0 = time.time()
    work = out_dir / f"work_{tag}"
    cfg = pipeline.NestedPipelineConfig(case, work / "out", work / "proof", hours=3, max_dom=3)
    hierarchy, bundles, meta, run_start, dt_by_domain, carries = pipeline._load_domains(cfg, names)
    load_s = time.time() - t0
    log(f"{tag}: loaded in {load_s:.0f}s run_start={run_start}")
    tree = dt.DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    runtime = dt._prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
    fused_parents = [p for p in names if runtime.fused_cascade(p) is not None]
    assert fused_parents == ["d02"], fused_parents
    kw, prog = captured["d02"]
    pnl, cnls = kw["parent_namelist"], tuple(kw["child_namelists"])
    doms = [("d02", pnl)] + [(c, nl) for c, nl in zip(kw["child_names"], cnls)]
    rows, arrays = [], {}
    treedefs = {}
    for dom, nl in doms:
        lr, treedefs[dom] = namelist_rows(nl)
        for lname, leaf in lr:
            a = np.asarray(leaf)
            arrays[f"{dom}|{lname}"] = a
            rows.append({"dom": dom, "leaf": lname, "shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha(a)})
    np.savez(out_dir / f"leaves_{tag}.npz", **arrays)
    other = {k: kw[k] for k in ("child_bdy_widths", "child_ratios", "child_cadences", "parent_cadence", "child_names")}
    wrows = []
    for c, w in zip(kw["child_names"], kw["child_weights"]):
        for p_, leaf in jax.tree_util.tree_flatten_with_path(w)[0]:
            wrows.append({"child": c, "leaf": jax.tree_util.keystr(p_), "sha256": sha(leaf)})
    rec = {"load_s": round(load_s, 1), "run_start": str(run_start), "dt_by_domain": {k: float(v) for k, v in dt_by_domain.items()},
           "closure_scalars": {k: (list(v) if isinstance(v, tuple) else v) for k, v in other.items()},
           "child_weights": wrows, "leaves": rows,
           "clock_base_lower_boundary_present": {dom: op.build_clock_base(nl).lower_boundary is not None for dom, nl in doms}}
    if ci < n_trace:
        fused_jit = find_fused_jit(prog)
        assert fused_jit is not None, "fused_jit not found in program closure"
        target = fused_jit.__wrapped__
        code, free = target.__code__, target.__code__.co_freevars
        assert "parent_namelist" in free and "child_namelists" in free, free
        p_leaves, p_def = jax.tree_util.tree_flatten(pnl)
        c_flat = [jax.tree_util.tree_flatten(nl) for nl in cnls]
        c_leaves, c_defs = [f[0] for f in c_flat], [f[1] for f in c_flat]

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
        t1 = time.time()
        closed, out_shape = jax.make_jaxpr(lifted, return_shape=True)(p_leaves, c_leaves, args)
        trace_s = time.time() - t1
        log(f"{tag}: traced lifted fused_jit in {trace_s:.0f}s, top eqns {len(closed.jaxpr.eqns)}")
        n_p = len(p_leaves)
        n_c = [len(x) for x in c_leaves]
        n_nl = n_p + sum(n_c)
        out_paths = [jax.tree_util.keystr(p_) for p_, _ in jax.tree_util.tree_flatten_with_path(out_shape)[0]]
        assert len(out_paths) == len(closed.jaxpr.outvars)
        prog_mask = [(".state." in s and "_bdy" not in s and "census" not in s) for s in out_paths]
        t2 = time.time()
        res_all = live(closed.jaxpr, [True] * len(out_paths))
        res_state = live(closed.jaxpr, prog_mask)
        live_s = time.time() - t2
        ncv = len(closed.jaxpr.constvars)
        used_all = res_all[ncv:ncv + n_nl]
        used_state = res_state[ncv:ncv + n_nl]
        assert len(rows) == n_nl, (len(rows), n_nl)
        for r, ua, us in zip(rows, used_all, used_state):
            r["consumed_any_output"] = bool(ua)
            r["consumed_prognostic_state"] = bool(us)
        t3 = time.time()
        rec["trace"] = {"trace_s": round(trace_s, 1), "liveness_s": round(live_s, 1), "hash_s": None,
                        "n_eqns_total": count_eqns(closed.jaxpr), "n_outputs": len(out_paths),
                        "n_prognostic_outputs": int(sum(prog_mask)), "n_namelist_inputs": n_nl,
                        "n_other_consts": ncv,
                        "other_consts_sha": hashlib.sha256("".join(sha(c) for c in closed.consts).encode()).hexdigest(),
                        "jaxpr_struct_sha": jhash(closed.jaxpr)}
        rec["trace"]["hash_s"] = round(time.time() - t3, 1)
        log(f"{tag}: liveness {live_s:.0f}s; struct {rec['trace']['jaxpr_struct_sha'][:16]}")
        del closed
        _memo.clear()
        _hmemo.clear()
    summary["cases"][tag] = rec
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"{tag}: done ({time.time() - t0:.0f}s)")
    del hierarchy, bundles, carries, tree, runtime, prog, kw
print("DONE")
