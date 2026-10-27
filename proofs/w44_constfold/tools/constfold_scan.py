"""w44-constfold AST scanner: Python constant EXPRESSIONS whose f64 fold differs from a REAL4 fold.

WRF (RWORDSIZE=4) folds REAL PARAMETER expressions per operation in REAL4 (gfortran: every leaf a
REAL4 literal/PARAMETER, every operation rounded to REAL4).  A Python constant subtree such as
``(T0C - 6.0) - TICE`` is evaluated in float64 and only the final value is rounded to float32 when it
meets an f32 array (weak typing) -> the release program can hold a constant 1+ ulps away from WRF's.

For every module of ``src/gpuwrf`` the scanner
  * resolves module-level / function-local numeric constants (incl. imports, aliases, np.float32 casts),
  * evaluates every maximal constant expression twice: as the port evaluates it (Python float64 or
    numpy-typed scalars, NEP 50) and as a REAL4 per-operation fold,
  * reports every site where float32(port value) != REAL4 fold ("MISMATCH"), plus every use of a
    module constant whose own definition mismatches ("TAINTED-NAME").
It does not decide WRF relevance: the WRF source decides whether REAL4 folding applies (manual step).

usage: python constfold_scan.py SRC_ROOT OUT_JSON
"""
from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np

F32, F64 = np.float32, np.float64
CAST32 = {"float32"}
CAST64 = {"float64", "float"}
MATH = {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10, "sin": math.sin,
        "cos": math.cos, "atan": math.atan, "tanh": math.tanh}


class CV:
    """Dual constant: ``port`` = value as the Python source evaluates it, ``r4`` = REAL4 fold."""

    __slots__ = ("port", "r4", "nops", "names", "exact")

    def __init__(self, port, r4, nops=0, names=(), exact=True):
        self.port, self.r4, self.nops, self.names, self.exact = port, r4, nops, tuple(names), exact

    def port32(self):
        return F32(self.port)


def _num(v):
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)


def _binop(op, a, b):
    if isinstance(op, ast.Add):
        return a + b
    if isinstance(op, ast.Sub):
        return a - b
    if isinstance(op, ast.Mult):
        return a * b
    if isinstance(op, ast.Div):
        return a / b
    if isinstance(op, ast.Pow):
        return a ** b
    if isinstance(op, ast.FloorDiv):
        return a // b
    if isinstance(op, ast.Mod):
        return a % b
    raise TypeError(op)


def _r4_binop(op, a, b):
    a32 = F32(a)
    if isinstance(op, ast.Pow):
        bb = float(b)
        if bb.is_integer() and 0 < bb <= 8:  # Fortran x**n -> repeated REAL4 products at run time
            out = a32
            for _ in range(int(bb) - 1):
                out = F32(out * a32)
            return out
        return F32(float(a32) ** float(F32(b)))
    with np.errstate(all="ignore"):
        return F32(_binop(op, a32, F32(b)))


class Module:
    def __init__(self, scanner, name, path):
        self.scanner, self.name, self.path = scanner, name, path
        self.src = path.read_text()
        self.tree = ast.parse(self.src)
        self.imports = {}      # local name -> ("mod", modname) | ("sym", modname, symbol)
        self.defs = {}         # name -> ast expr (module-level single assignment)
        self.multi = set()
        self.star = []
        self.env = {}          # name -> CV | None
        self.resolving = set()
        self._collect()

    def _abs(self, level, mod):
        if not level:
            return mod
        parts = self.name.split(".")
        pkg = parts[: len(parts) - level] if not self.path.name == "__init__.py" else parts[: len(parts) - level + 1]
        return ".".join(pkg + ([mod] if mod else []))

    def _collect(self):
        for node in self.tree.body:
            self._collect_stmt(node, top=True)

    def _collect_stmt(self, node, top):
        if isinstance(node, ast.Import):
            for a in node.names:
                self.imports[a.asname or a.name.split(".")[0]] = ("mod", a.name if a.asname else a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            base = self._abs(node.level, node.module)
            for a in node.names:
                if a.name == "*":
                    self.star.append(base)
                else:
                    self.imports[a.asname or a.name] = ("sym", base, a.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and top:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                return
            for t in targets:
                if isinstance(t, ast.Name):
                    if t.id in self.defs:
                        self.multi.add(t.id)
                    self.defs[t.id] = value
                elif isinstance(t, ast.Tuple) and isinstance(value, ast.Tuple) and len(t.elts) == len(value.elts):
                    for tt, vv in zip(t.elts, value.elts):
                        if isinstance(tt, ast.Name):
                            if tt.id in self.defs:
                                self.multi.add(tt.id)
                            self.defs[tt.id] = vv
        elif isinstance(node, ast.If) and top:
            # conditional module constants: names become ambiguous
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            self.multi.add(t.id)
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    self._collect_stmt(sub, top=False)
        elif isinstance(node, ast.Try) and top:
            for sub in node.body:
                self._collect_stmt(sub, top=True)

    def lookup(self, name):
        if name in self.env:
            return self.env[name]
        if name in self.resolving:
            return None
        self.resolving.add(name)
        try:
            val = None
            if name in self.defs and name not in self.multi:
                val = self.scanner.evaluate(self, self.defs[name], {})
            elif name in self.imports and self.imports[name][0] == "sym":
                _, mod, sym = self.imports[name]
                m = self.scanner.module(mod)
                if m is not None:
                    val = m.lookup(sym)
                else:  # "from pkg import submodule"
                    val = None
            else:
                for mod in self.star:
                    m = self.scanner.module(mod)
                    if m is not None and (name in m.defs or name in m.imports):
                        val = m.lookup(name)
                        break
            self.env[name] = val
            return val
        finally:
            self.resolving.discard(name)

    def alias_kind(self, node):
        """'f32' / 'f64' if node names a float32/float64 constructor (np.float32, jnp.float32, alias)."""
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr in CAST32:
                return "f32"
            if node.attr in CAST64:
                return "f64"
        if isinstance(node, ast.Name):
            if node.id in CAST64 and node.id not in self.defs:
                return "f64"
            target = self.defs.get(node.id)
            if target is not None and node.id not in self.multi:
                return self.alias_kind(target)
            imp = self.imports.get(node.id)
            if imp and imp[0] == "sym":
                if imp[2] in CAST32:
                    return "f32"
                m = self.scanner.module(imp[1])
                if m is not None and imp[2] in m.defs:
                    return m.alias_kind(ast.Name(imp[2]))
        return None


class Scanner:
    def __init__(self, root: Path):
        self.root = root
        self.files = {}
        pkg_root = root / "src"
        for path in sorted((pkg_root / "gpuwrf").rglob("*.py")):
            rel = path.relative_to(pkg_root).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            self.files[".".join(parts)] = path
        self.modules = {}

    def module(self, name):
        if name in self.modules:
            return self.modules[name]
        path = self.files.get(name)
        mod = None
        if path is not None:
            try:
                mod = Module(self, name, path)
            except SyntaxError:
                mod = None
        self.modules[name] = mod
        return mod

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, mod, node, local):
        try:
            return self._eval(mod, node, local)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
            return None

    def _eval(self, mod, node, local):
        if isinstance(node, ast.Constant):
            if _num(node.value):
                v = node.value
                return CV(v, F32(v), 0, (), float(F32(v)) == float(v))
            return None
        if isinstance(node, ast.Name):
            if node.id in local:
                return local[node.id]
            cv = mod.lookup(node.id)
            if cv is None:
                return None
            return CV(cv.port, cv.r4, cv.nops, (node.id,), cv.exact)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            imp = mod.imports.get(node.value.id)
            target = None
            if imp and imp[0] == "mod":
                target = self.module(imp[1])
            elif imp and imp[0] == "sym":
                target = self.module(imp[1] + "." + imp[2])
            if target is not None:
                cv = target.lookup(node.attr)
                if cv is not None:
                    return CV(cv.port, cv.r4, cv.nops, (node.value.id + "." + node.attr,), cv.exact)
            if node.value.id in ("math", "np", "numpy", "jnp") and node.attr in ("pi", "e"):
                v = getattr(math, node.attr)
                return CV(v, F32(v), 0, (node.attr,), False)
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            a = self._eval(mod, node.operand, local)
            if a is None:
                return None
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return CV(-a.port if sign < 0 else a.port, F32(-a.r4 if sign < 0 else a.r4), a.nops, a.names, a.exact)
        if isinstance(node, ast.BinOp):
            a = self._eval(mod, node.left, local)
            if a is None:
                return None
            b = self._eval(mod, node.right, local)
            if b is None:
                return None
            with np.errstate(all="ignore"):
                port = _binop(node.op, a.port, b.port)
            if not _num(port):
                return None
            r4 = _r4_binop(node.op, a.r4, b.r4)
            return CV(port, r4, a.nops + b.nops + 1, a.names + b.names, a.exact and b.exact)
        if isinstance(node, ast.Call):
            kind = mod.alias_kind(node.func)
            if kind and len(node.args) == 1 and not node.keywords:
                a = self._eval(mod, node.args[0], local)
                if a is None:
                    return None
                port = F32(a.port) if kind == "f32" else F64(a.port)
                return CV(port, a.r4, a.nops, a.names, a.exact)
            fn = None
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and \
                    node.func.value.id in ("math", "np", "numpy") and node.func.attr in MATH:
                fn = MATH[node.func.attr]
            if fn is not None and len(node.args) == 1:
                a = self._eval(mod, node.args[0], local)
                if a is None:
                    return None
                if isinstance(a.port, np.float32) and node.func.value.id != "math":
                    port = getattr(np, node.func.attr)(a.port)
                else:
                    port = fn(float(a.port))
                return CV(port, F32(fn(float(a.r4))), a.nops + 1, a.names, a.exact)
            return None
        return None


# ---------------------------------------------------------------------- site walk
CONST_PARENTS = (ast.BinOp, ast.UnaryOp)


def scan_module(scanner, mod):
    sites = []
    parents = {}
    for node in ast.walk(mod.tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    lines = mod.src.splitlines()

    def scope_locals(fn):
        """Function-local names bound exactly once to a constant expression (plus parameter defaults)."""
        local = {}
        counts = {}
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    for nm in ast.walk(t):
                        if isinstance(nm, ast.Name):
                            counts[nm.id] = counts.get(nm.id, 0) + 1
            elif isinstance(sub, (ast.AugAssign, ast.AnnAssign)) and isinstance(sub.target, ast.Name):
                counts[sub.target.id] = counts.get(sub.target.id, 0) + 2
            elif isinstance(sub, (ast.For, ast.comprehension)):
                for nm in ast.walk(sub.target):
                    if isinstance(nm, ast.Name):
                        counts[nm.id] = counts.get(nm.id, 0) + 2
        args = fn.args
        for a in args.args + args.kwonlyargs + args.posonlyargs:
            counts[a.arg] = counts.get(a.arg, 0) + 2  # parameters are not constants (defaults vary)
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Name):
                nm = sub.targets[0].id
                if counts.get(nm) == 1:
                    cv = scanner.evaluate(mod, sub.value, local)
                    if cv is not None:
                        local[nm] = cv
            elif (isinstance(sub, ast.Assign) and len(sub.targets) == 1 and isinstance(sub.targets[0], ast.Tuple)
                  and isinstance(sub.value, ast.Tuple) and len(sub.targets[0].elts) == len(sub.value.elts)):
                for tt, vv in zip(sub.targets[0].elts, sub.value.elts):  # a, b = (X, Y)
                    if isinstance(tt, ast.Name) and counts.get(tt.id) == 1:
                        cv = scanner.evaluate(mod, vv, local)
                        if cv is not None:
                            local[tt.id] = cv
        shadow = {k for k, v in counts.items() if k not in local}
        return local, shadow

    def record(node, cv, kind, scope):
        p32, r4 = cv.port32(), F32(cv.r4)
        if not (np.isfinite(p32) and np.isfinite(r4)):
            return
        if p32 == r4:
            return
        seg = ast.get_source_segment(mod.src, node) or ""
        ulps = int(abs(int(np.asarray(p32).view(np.int32)) - int(np.asarray(r4).view(np.int32))))
        sites.append({"module": mod.name, "file": str(mod.path.relative_to(scanner.root)), "line": node.lineno,
                      "end_line": getattr(node, "end_lineno", node.lineno), "kind": kind, "scope": scope,
                      "expr": seg[:200], "port_f32": float(p32), "real4_fold": float(r4), "ulps": ulps,
                      "port_is_typed": type(cv.port).__name__, "names": sorted(set(cv.names)),
                      "nops": cv.nops, "line_text": lines[node.lineno - 1].strip()[:200]})

    def pruned(top):
        stack = [top]
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue  # own scope
            yield node
            stack.extend(ast.iter_child_nodes(node))

    def visit_scope(body_nodes, local, shadow, scope):
        for top in body_nodes:
            for node in pruned(top):
                if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call)):
                    par = parents.get(node)
                    if isinstance(par, CONST_PARENTS) and scanner.evaluate(mod, par, local) is not None:
                        continue  # not maximal
                    if isinstance(node, ast.Call) and mod.alias_kind(node.func) is None:
                        continue
                    cv = scanner.evaluate(mod, node, local)
                    if cv is not None and (cv.nops >= 1 or not cv.exact):
                        if any(n in shadow for n in cv.names):
                            continue
                        record(node, cv, "MISMATCH", scope)
                elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(getattr(node, "ctx", None), ast.Load):
                    par = parents.get(node)
                    if isinstance(par, (ast.BinOp, ast.UnaryOp, ast.Call)) and scanner.evaluate(mod, par, local) is not None:
                        continue  # part of a larger constant expression (reported there)
                    if isinstance(par, ast.Attribute):
                        continue
                    if isinstance(node, ast.Name) and node.id in shadow:
                        continue
                    cv = scanner.evaluate(mod, node, local)
                    if cv is not None and cv.nops >= 1:
                        record(node, cv, "TAINTED-NAME", scope)

    # module level definitions
    for name, value in mod.defs.items():
        if name in mod.multi:
            continue
        cv = mod.lookup(name)
        if cv is not None and cv.nops >= 1:
            p32, r4 = cv.port32(), F32(cv.r4)
            if np.isfinite(p32) and np.isfinite(r4) and p32 != r4:
                record(value, cv, "DEFINITION", "<module>")
    for node in ast.walk(mod.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local, shadow = scope_locals(node)
            body = node.body
            visit_scope(body, local, shadow, node.name)
    return sites


def main():
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    scanner = Scanner(root)
    sites = []
    errors = []
    for name in scanner.files:
        mod = scanner.module(name)
        if mod is None:
            errors.append(name)
            continue
        sites.extend(scan_module(scanner, mod))
    # de-duplicate (nested function scopes visit inner bodies too)
    seen, uniq = set(), []
    for s in sites:
        key = (s["file"], s["line"], s["expr"], s["kind"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    out.write_text(json.dumps({"root": str(root), "n_modules": len(scanner.files), "parse_errors": errors,
                               "n_sites": len(uniq), "sites": uniq}, indent=1) + "\n")
    by_kind = {}
    for s in uniq:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
    print(len(uniq), by_kind, "parse_errors", len(errors))


if __name__ == "__main__":
    main()
