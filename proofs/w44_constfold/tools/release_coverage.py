"""Which constants does the RELEASE program contain, and where do they come from? (w44-constfold filter)

Runs the release identity probe (proofs/s1_port/release_identity.py -> proofs/o1_nlbind/probe.py: real CLI
loader, release defaults, one production step `_advance_chunk_fori` traced with make_jaxpr) and records
  * every src/gpuwrf line executed while tracing the step (line tracer; "step" coverage),
  * every floating LITERAL / small constant of the traced step jaxpr, recursively through every sub-jaxpr
    (pjit, cond branches, while/scan bodies, custom_jvp/vjp, pallas_call kernel bodies), with dtype,
    bit pattern and the innermost gpuwrf source frame (file:line) of the consuming equation.
The identity sha the probe writes is unchanged (tracing only observes).

usage: JAX_PLATFORMS=cpu python release_coverage.py SOURCE_ROOT OUT_PREFIX [CASE_DIR]
writes OUT_PREFIX.identity.json (probe output), OUT_PREFIX.coverage.json, OUT_PREFIX.literals.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

assert os.environ.get("JAX_PLATFORMS") == "cpu"
root, prefix = Path(sys.argv[1]).resolve(), sys.argv[2]
case = sys.argv[3] if len(sys.argv) > 3 else None
SRC = str(root / "src" / "gpuwrf")
lines = set()
_on = [False]


def _local(frame, event, arg):
    if event == "line" and _on[0]:
        lines.add((frame.f_code.co_filename, frame.f_lineno))
    return _local


def _global(frame, event, arg):
    if _on[0] and frame.f_code.co_filename.startswith(SRC):
        lines.add((frame.f_code.co_filename, frame.f_lineno))
        return _local
    return None


sys.path[:0] = [str(root / "src")]
import jax  # noqa: E402
from jax._src import core as jcore  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

_orig_step = om._advance_chunk_fori


def _traced_step(*a, **k):
    _on[0] = True
    sys.settrace(_global)
    try:
        return _orig_step(*a, **k)
    finally:
        sys.settrace(None)
        _on[0] = False


om._advance_chunk_fori = _traced_step
captured = []
_orig_make_jaxpr = jax.make_jaxpr


def _make_jaxpr(fn, *a, **k):
    inner = _orig_make_jaxpr(fn, *a, **k)

    def call(*aa, **kk):
        out = inner(*aa, **kk)
        captured.append(out)
        return out
    return call


jax.make_jaxpr = _make_jaxpr


def _frame(eqn):
    tb = eqn.source_info.traceback
    if tb is None:
        return None
    for fr in tb.frames:  # innermost first
        if fr.file_name.startswith(SRC):
            return f"{Path(fr.file_name).relative_to(root)}:{fr.line_num}"
    return None


def _subjaxprs(params):
    for v in params.values():
        for x in (v if isinstance(v, (tuple, list)) else (v,)):
            if isinstance(x, jcore.ClosedJaxpr):
                yield x.jaxpr, x.consts
            elif isinstance(x, jcore.Jaxpr):
                yield x, ()


def _dce_all_outputs(jaxpr, consts):
    """Drop equations that feed no output (while bodies have no JAX DCE rule; all carries count as live)."""
    try:
        from jax._src.interpreters import partial_eval as pe
        nconst = len(jaxpr.constvars)
        flat = pe.convert_constvars_jaxpr(jaxpr) if nconst else jaxpr
        live, used_in = pe.dce_jaxpr(flat, [True] * len(flat.outvars), instantiate=True)
        return live, [c for c, u in zip(consts, used_in[:nconst]) if u] if nconst else ()
    except Exception:  # noqa: BLE001 - keep the original body
        return jaxpr, consts


def walk(jaxpr, consts, depth, acc):
    for c in consts:
        arr = np.asarray(c)
        if np.issubdtype(arr.dtype, np.floating) and arr.size <= 64:
            for val in np.unique(arr.ravel()):
                acc.append({"where": "constvar", "dtype": str(arr.dtype), "value": float(val),
                            "bits": int(np.asarray(val, arr.dtype).view(np.uint32 if arr.dtype == np.float32 else np.uint64)),
                            "frame": None, "prim": None, "depth": depth})
    for eqn in jaxpr.eqns:
        fr = None
        for iv in eqn.invars:
            if isinstance(iv, jcore.Literal):
                arr = np.asarray(iv.val)
                if np.issubdtype(arr.dtype, np.floating):
                    fr = fr or _frame(eqn)
                    for val in np.unique(arr.ravel())[:64]:
                        acc.append({"where": "literal", "dtype": str(arr.dtype), "value": float(val),
                                    "bits": int(np.asarray(val, arr.dtype).view(np.uint32 if arr.dtype == np.float32 else np.uint64)),
                                    "frame": fr, "prim": eqn.primitive.name, "depth": depth})
        for sub, sc in _subjaxprs(eqn.params):
            if eqn.primitive.name != "pallas_call":
                sub, sc = _dce_all_outputs(sub, sc)
            walk(sub, sc, depth + 1, acc)


argv = [str(root / "proofs/s1_port/release_identity.py"), str(root), prefix + ".identity.json"]
if case:
    argv.append(case)
sys.argv = argv
script = root / "proofs/s1_port/release_identity.py"
exec(compile(script.read_text(), str(script), "exec"), {"__file__": str(script), "__name__": "__main__"})
cov = sorted([str(Path(f).relative_to(root)), n] for f, n in lines if f.startswith(SRC))
Path(prefix + ".coverage.json").write_text(json.dumps(cov) + "\n")
from jax._src.interpreters import partial_eval as pe  # noqa: E402

acc = []
dce_ok = []
for cj in captured:
    try:  # live program only: make_jaxpr keeps dead equations (E106); DCE recurses into pjit/cond/while/scan
        nconst = len(cj.jaxpr.constvars)
        flat = pe.convert_constvars_jaxpr(cj.jaxpr)
        live, used_in = pe.dce_jaxpr(flat, [True] * len(flat.outvars), instantiate=True)
        used_consts = [c for c, u in zip(cj.consts, used_in[:nconst]) if u]
        for c in used_consts:  # record live constvars like walk() does
            arr = np.asarray(c)
            if np.issubdtype(arr.dtype, np.floating) and arr.size <= 64:
                for val in np.unique(arr.ravel()):
                    acc.append({"where": "constvar", "dtype": str(arr.dtype), "value": float(val),
                                "bits": int(np.asarray(val, arr.dtype).view(np.uint32 if arr.dtype == np.float32 else np.uint64)),
                                "frame": None, "prim": None, "depth": 0})
        walk(live, [], 0, acc)
        dce_ok.append([len(cj.jaxpr.eqns), len(live.eqns)])
    except Exception as exc:  # noqa: BLE001
        print("DCE failed:", type(exc).__name__, str(exc)[:300])
        walk(cj.jaxpr, cj.consts, 0, acc)
        dce_ok.append(False)
uniq = {}
for r in acc:
    key = (r["where"], r["dtype"], r["bits"], r["frame"], r["prim"])
    if key not in uniq:
        uniq[key] = dict(r, count=0)
    uniq[key]["count"] += 1
Path(prefix + ".literals.json").write_text(json.dumps(list(uniq.values())) + "\n")
# Program STRUCTURE identity: the probe's canonical text (locations stripped, E151) with every floating literal
# value masked -> equal hashes mean the same equations/primitives/shapes/dtypes, i.e. only constants differ.
import hashlib  # noqa: E402
import re  # noqa: E402

_canon = re.sub(r"name_and_src_info=[^\]\n]*", "", str(captured[0]))
_canon = re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", _canon)
_masked = re.sub(r"-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?:f(?:16|32|64)\[\]", "LIT:f[]", _canon)
_masked = re.sub(r"-?(?:inf|nan):f(?:16|32|64)\[\]", "LIT:f[]", _masked)
Path(prefix + ".structure.json").write_text(json.dumps({
    "canon_sha256": hashlib.sha256(_canon.encode()).hexdigest(),
    "masked_float_literal_sha256": hashlib.sha256(_masked.encode()).hexdigest(),
    "chars": len(_canon), "eqn_lines": _canon.count("\n")}) + "\n")
print({"lines": len(cov), "literal_records": len(acc), "unique": len(uniq), "captured": len(captured), "dce": dce_ok})
