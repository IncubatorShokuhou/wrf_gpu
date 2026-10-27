"""Reverse sweep: release-jaxpr f32 literals that are NOT a single source literal/constant of their line.

For every float32 literal (from release_coverage.py) consumed at gpuwrf file:line, collect the 'explaining'
values of the enclosing statement: f32 of every numeric literal, +/-, and the f32 (port and REAL4) values of every
name/attribute the w44-constfold scanner resolves to a module/function constant.  A literal whose bits are not in
that set was COMPUTED (a multi-operation fold, a helper-returned constant, a namelist scalar combination, a table
value ...) and is listed for manual classification.  Literals 0, +-1, +-2, 0.5, inf, nan are ignored.

usage: python unexplained_literals.py SRC_ROOT OUT_JSON PREFIX [PREFIX ...]
"""
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constfold_scan as cs  # noqa: E402

root = Path(sys.argv[1]).resolve()
out_path = sys.argv[2]
scanner = cs.Scanner(root)
TRIVIAL = {0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5}


def b32(x):
    return int(np.asarray(np.float32(x)).view(np.uint32))


stmt_cache = {}
stmt_index = {}


def _index(mod):
    """line -> innermost simple statement covering it (built once per module)."""
    if mod.name in stmt_index:
        return stmt_index[mod.name]
    idx = {}
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                                                 ast.If, ast.For, ast.While, ast.With, ast.Try)):
            span = getattr(node, "end_lineno", node.lineno) - node.lineno
            for ln in range(node.lineno, node.lineno + span + 1):
                if ln not in idx or span < idx[ln][0]:
                    idx[ln] = (span, node)
    stmt_index[mod.name] = idx
    return idx


module_cache = {}


def module_values(mod):
    """f32 bits of every numeric literal (+-) and resolvable module constant (port/REAL4) of the module:
    single-leaf constants that reach a line through locals, tuples, loops or helpers."""
    if mod.name in module_cache:
        return module_cache[mod.name]
    vals = set()
    for sub in ast.walk(mod.tree):
        if isinstance(sub, ast.Constant) and cs._num(sub.value):
            for v in (sub.value, -sub.value):
                vals.add(b32(v))
    for name in list(mod.defs) + list(mod.imports):
        cv = mod.lookup(name)
        if cv is not None and cv.nops == 0:
            for v in (cv.port, cv.r4, -cv.port, -cv.r4):
                vals.add(b32(v))
    module_cache[mod.name] = vals
    return vals


def statement_values(rel, line):
    key = (rel, line)
    if key in stmt_cache:
        return stmt_cache[key]
    modname = ".".join(Path(rel).relative_to("src").with_suffix("").parts)
    if modname.endswith(".__init__"):
        modname = modname[: -len(".__init__")]
    mod = scanner.module(modname)
    vals, text = set(), ""
    if mod is not None:
        best = _index(mod).get(line, (None, None))[1]
        vals |= module_values(mod)
        if best is not None:
            text = (ast.get_source_segment(mod.src, best) or "")[:300]
            for sub in ast.walk(best):
                if isinstance(sub, ast.Constant) and cs._num(sub.value):
                    for v in (sub.value, -sub.value):
                        vals.add(b32(v))
                elif isinstance(sub, (ast.Name, ast.Attribute)):
                    cv = scanner.evaluate(mod, sub, {})
                    if cv is not None:
                        for v in (cv.port, cv.r4, -cv.port, -cv.r4):
                            vals.add(b32(v))
    stmt_cache[key] = (vals, text)
    return vals, text


rows = defaultdict(lambda: {"count": 0, "prims": set(), "prefixes": set()})
for pre in sys.argv[3:]:
    for rec in json.load(open(pre + ".literals.json")):
        if rec["dtype"] != "float32" or not rec["frame"] or rec["where"] != "literal":
            continue
        v = rec["value"]
        if v in TRIVIAL or not np.isfinite(v):
            continue
        rel, line = rec["frame"].rsplit(":", 1)
        vals, text = statement_values(rel, int(line))
        if rec["bits"] in vals:
            continue
        row = rows[(rec["frame"], rec["bits"])]
        row.update(frame=rec["frame"], value=v, bits=rec["bits"], text=text)
        row["count"] += rec["count"]
        row["prims"].add(rec["prim"])
        row["prefixes"].add(Path(pre).name)
out = []
for row in rows.values():
    row["prims"], row["prefixes"] = sorted(row["prims"]), sorted(row["prefixes"])
    out.append(row)
out.sort(key=lambda r: r["frame"])
Path(out_path).write_text(json.dumps(out, indent=1) + "\n")
print(len(out), "unexplained literal (frame,bits) pairs;", len({r['frame'] for r in out}), "lines")
