"""Offline mechanical translation of P3's level-local process block.

This is intentionally scoped to ONE frozen WRF routine, not a runtime Fortran
interpreter. Output is ordinary, reviewable JAX Python committed with the port.
Every emitted assignment carries the original source line. Level-independent
work broadcasts over columns/levels; adaptive sedimentation is separate.
"""
import argparse
import ast
from pathlib import Path
import re


def constants(source, out):
    import numpy as np
    c = {}
    scope = {"v": c, "_r": np.float32, "jnp": np}
    begun = False
    for n, line in logical_lines(source):
        if re.match(r"pi\s*=\s*3\.14159265", line):
            begun = True
        if not begun or n > 315:
            continue
        if "=" not in line:
            continue
        name, rhs = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"\w+", name):
            continue
        if name == "e0":
            c[name] = np.float32(6.11239921) * np.float32(100)
        else:
            c[name] = np.float32(eval(expr(rhs), scope))
    derf = source[source.index(" real function DERF(X)"):source.index(" end function DERF")]
    arrays = []
    for match in re.finditer(r"data\s+(?:a|\(b\(i\),[^/]+)\s*/(.*?)/", derf, re.I | re.S):
        arrays.append(tuple(float(x) for x in re.findall(r"[+-]?\d+\.\d+(?:E[+-]?\d+)", match[1], re.I)))
    assert len(arrays) == 6 and all(len(a) == 13 for a in arrays[1:]) and len(arrays[0]) == 65
    a, b = arrays[0], sum(arrays[1:], ())
    out.write_text('"""WRF P3 REAL constants and original DERF coefficients; generated offline."""\nimport numpy as np\n\n'
                   + "CONSTANTS = {\n" + "".join(f"    {k!r}: np.float32({float(v)!r}),\n" for k,v in c.items()) + "}\n\n"
                   + f"DERF_A = {a!r}\nDERF_B = {b!r}\n")

RATES = set("qccol qwgrth qidep qrcol qinuc nccol nrcol ninuc qisub qimlt nimlt nisub nislf qchetc qcheti qrhetc qrheti nchetc ncheti nrhetc nrheti nrshdr qcshd qrmul nimul ncshdc rhorime_c eii_fact epsi dumm0 dumm3 log_wetgrowth".split())
CATS = set("qitot qirim nitot birim zitot diam_ice rimefraction rimevolume diag_effi diag_vmi diag_di diag_rhoi".split())
MATRICES = {"qicol", "nicol"}
TABLES = {"revap_table", "vn_table", "vm_table", "mu_r_table", "dnu"}
INTRINSICS = {"max": "_max", "min": "_min", "sum": "_sum", "exp": "jnp.exp", "dexp": "jnp.exp",
              "log": "jnp.log", "alog10": "jnp.log10", "log10": "jnp.log10", "sqrt": "jnp.sqrt",
              "abs": "jnp.abs", "gamma": "gamma", "real": "_r", "float": "_r", "sngl": "_r",
              "dble": "_d", "int": "_int", "qv_sat": "qv_sat", "derf": "derf",
              "compute_mu_3moment": "compute_mu_3moment", "g_of_mu": "g_of_mu"}
LOOPS = {"iice", "catcoll", "imu"}
STATIC = {"ncat", "log_predictnc", "log_3momentice", "iparam", "scpf_on", "log_predictssat", "log_hmossopon", "niter_mui"}


def logical_lines(text):
    pending = ""
    start = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        # The selected block contains no literal ! or ; in character constants.
        line = raw.split("!", 1)[0].strip().lower()
        if not line:
            continue
        if not pending:
            start = lineno
        pending += " " + line.lstrip("&").rstrip("&").strip()
        if line.endswith("&"):
            continue
        for statement in pending.strip().split(";"):
            yield start, statement.strip()
        pending = ""


def parse_expr(text):
    text = text.lower().replace(":", "allidx")
    for a, b in ((".and.", " and "), (".or.", " or "), (".not.", " not "),
                 (".ge.", ">="), (".gt.", ">"), (".le.", "<="), (".lt.", "<"),
                 (".eq.", "=="), (".ne.", "!="), ("/=", "!="), (".true.", "True"), (".false.", "False")):
        text = text.replace(a, b)
    return ast.parse(text.strip(), mode="eval").body


class Expression(ast.NodeTransformer):
    def __init__(self, strict=False):
        self.strict = strict

    @staticmethod
    def rank(node):
        if isinstance(node, ast.Name):
            return 2 if node.id in MATRICES else 1 if node.id in RATES | CATS else 0
        if isinstance(node, ast.Call):
            if node.func.id == 'sum':
                return max(0, Expression.rank(node.args[0]) - 1)
            if node.func.id in RATES | CATS | MATRICES:
                return 1 if any(isinstance(a, ast.Name) and a.id == 'allidx' for a in node.args) else 0
            return max((Expression.rank(a) for a in node.args), default=0)
        if isinstance(node, ast.BinOp):
            return max(Expression.rank(node.left), Expression.rank(node.right))
        if isinstance(node, ast.UnaryOp):
            return Expression.rank(node.operand)
        return 0

    def visit_BinOp(self, node):
        if (isinstance(node.op, ast.Pow) and isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, float) and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, int)):
            # WRF folds REAL literal integer powers. Triton's integer_pow
            # lowering expects an IR operand, not a typed scalar literal.
            import numpy as np
            value=float(np.float32(np.float32(node.left.value) ** node.right.value))
            return ast.Call(func=ast.Name(id='_r',ctx=ast.Load()),args=[ast.Constant(value)],keywords=[])
        lr, rr = self.rank(node.left), self.rank(node.right)
        left, right = self.visit(node.left), self.visit(node.right)
        if lr != rr:
            which = right if lr > rr else left
            expanded = ast.Call(func=ast.Name(id='_expand', ctx=ast.Load()),
                                args=[which, ast.Constant(abs(lr-rr))], keywords=[])
            if lr > rr:
                right = expanded
            else:
                left = expanded
        operation = {ast.Add:'add', ast.Sub:'sub', ast.Mult:'mul', ast.Div:'div'}.get(type(node.op))
        if self.strict and operation is not None:
            return ast.Call(func=ast.Name(id='_real_binary', ctx=ast.Load()),
                            args=[ast.Constant(operation), left, right], keywords=[])
        return ast.BinOp(left=left, op=node.op, right=right)

    def visit_Name(self, node):
        if node.id in LOOPS or node.id in STATIC or node.id == "allidx":
            return node
        return ast.Subscript(value=ast.Name(id="v", ctx=ast.Load()), slice=ast.Constant(node.id), ctx=ast.Load())

    def visit_Constant(self, node):
        if type(node.value) is float:
            return ast.Call(func=ast.Name(id="_r", ctx=ast.Load()), args=[node], keywords=[])
        return node

    def visit_BoolOp(self, node):
        func = "logical_and" if isinstance(node.op, ast.And) else "logical_or"
        values = [self.visit(x) for x in node.values]
        result = values[0]
        for value in values[1:]:
            result = ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr=func, ctx=ast.Load()), args=[result, value], keywords=[])
        return result

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return self.visit(ast.Constant(-node.operand.value))
        if isinstance(node.op, ast.Not):
            if isinstance(node.operand, ast.Name) and node.operand.id in STATIC:
                return ast.UnaryOp(op=ast.Not(), operand=self.visit(node.operand))
            return ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="logical_not", ctx=ast.Load()), args=[self.visit(node.operand)], keywords=[])
        return self.generic_visit(node)

    def visit_Call(self, node):
        assert isinstance(node.func, ast.Name)
        name = node.func.id
        if name in INTRINSICS:
            function = ast.parse(INTRINSICS[name], mode="eval").body
            return ast.Call(func=function, args=[self.visit(x) for x in node.args], keywords=[])
        return self.array(name, node.args)

    def array(self, name, args):
        value = self.visit_Name(ast.Name(id=name, ctx=ast.Load()))
        if len(args) >= 2 and isinstance(args[0], ast.Name) and args[0].id == "i":
            assert isinstance(args[1], ast.Name) and args[1].id == "k", (name, ast.unparse(args[1]))
            args = args[2:]
            if not args or (len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == "allidx"):
                return value
        elif len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == "k":
            return value
        elif name not in RATES | MATRICES | TABLES:
            raise ValueError(f"Unrecognized P3 array {name}({','.join(ast.unparse(a) for a in args)})")
        if any(isinstance(a, ast.Name) and a.id == "allidx" for a in args):
            assert len(args) == 1
            return value
        indices = [ast.BinOp(left=self.visit(a), op=ast.Sub(), right=ast.Constant(1)) for a in args]
        if name in TABLES:
            return ast.Subscript(value=value, slice=ast.Tuple(elts=[ast.Call(func=ast.Name(id="_int", ctx=ast.Load()), args=[x], keywords=[]) for x in indices], ctx=ast.Load()), ctx=ast.Load())
        # Category indices may be a different destination at each point.
        return ast.Call(func=ast.Name(id="_get", ctx=ast.Load()), args=[value, ast.Tuple(elts=indices, ctx=ast.Load())], keywords=[])


def expr(text, *, strict=False):
    return ast.unparse(ast.fix_missing_locations(Expression(strict).visit(parse_expr(text))))


def target(text):
    node = parse_expr(text)
    if isinstance(node, ast.Name):
        return node.id, "None"
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    name = node.func.id
    args = node.args
    if len(args) >= 2 and isinstance(args[0], ast.Name) and args[0].id == "i":
        args = args[2:]
    elif len(args) == 1 and isinstance(args[0], ast.Name) and args[0].id == "k":
        args = []
    if not args or any(isinstance(a, ast.Name) and a.id == "allidx" for a in args):
        return name, "None"
    return name, "(" + ", ".join(f"({expr(ast.unparse(a))}) - 1" for a in args) + ",)"


CALL_OUTPUTS = {
    "get_cloud_dsd2": (1, 2, 4, 6, 7, 8, 9, 10),
    "get_rain_dsd2": (1, 2, 3, 4, 5),
    "calc_bulkrhorime": (1, 2, 3),
    "impose_max_total_ni": (0,),
    "find_lookuptable_indices_1a": tuple(range(6)),
    "find_lookuptable_indices_1b": (0, 1),
    "find_lookuptable_indices_1c": (0, 1),
    "find_lookuptable_indices_2": tuple(range(12)),
    "find_lookuptable_indices_3": tuple(range(6)),
    "access_lookup_table": (7,),
    "access_lookup_table_coll": (9,),
    "access_lookup_table_3mom": (9,),
    "access_lookup_table_coll_3mom": (11,),
    "access_lookup_table_colli": (13,),
    "icecat_destination": (4,),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("initial", "processes", "final"), default="processes")
    parser.add_argument("--constants-out", type=Path)
    args = parser.parse_args()
    if args.constants_out:
        constants(args.source.read_text(), args.constants_out)
    bounds = {"initial": (2321, 2410), "processes": (2450, 3926), "final": (4527, 4895)}[args.phase]
    all_statements = list(logical_lines(args.source.read_text()))
    statements = [(n, s) for n, s in all_statements if bounds[0] <= n <= bounds[1]]
    observations = {n: tag for tag, pattern in ((1,'sinks=(qcaut+qcacc'),(2,'iice_loop3:do'))
                    for n,s in all_statements if s.replace(' ','').startswith(pattern)}
    assert len(observations)==2
    output = []
    stack = []
    mask = "alive"
    indent = 1
    counter = 0
    assigned = set()
    optional_depth = 0
    def emit(s):
        output.append("    " * indent + s)
    def assignment(lhs, rhs, line):
        name, index = target(lhs)
        assigned.add(name)
        emit(f"_put(v, {name!r}, {index}, {rhs}, {mask})  # WRF:{line}")
    for line, statement in statements:
        if args.phase == 'processes' and line in observations:
            emit(f"if observer is not None: observer('r{observations[line]}', v)")
        statement = re.sub(r"^\w+\s*:\s*", "", statement)
        if optional_depth:
            if re.match(r"if\s*\(.*\)\s*then$", statement):
                optional_depth += 1
            if statement.startswith("endif") or statement.startswith("end if"):
                optional_depth -= 1
            continue
        if "present(diag_" in statement:
            if statement.endswith("then"):
                optional_depth = 1
            continue  # not passed by the WRF wrapper
        if not statement:
            continue
        if re.match(r"^(444|555)\s+continue$", statement):
            if statement.startswith("444"):
                emit("alive = level_alive  # WRF: restore nucleation path at label444")
            continue
        if statement.startswith("if (global_status"):
            continue  # no GEM error path; destination helper validated for WRF nCat1/2
        if statement.startswith("enddo") or statement.startswith("end do"):
            if stack and stack[-1][0] == "level":
                stack.pop()
                continue
            kind, parent, saved = stack.pop()
            assert kind == "for", (line, kind)
            indent -= 1
            mask = parent
            continue
        if statement.startswith("endif") or statement.startswith("end if"):
            kind, parent, saved = stack.pop()
            if kind == "static":
                indent -= 1
            else:
                assert kind == "if", (line, kind)
            mask = parent
            continue
        if statement.startswith("else"):
            kind, parent, taken = stack[-1]
            rest = re.sub(r"^else\s*if|^elseif", "if", statement)
            if kind == "static":
                indent -= 1
                if statement == "else":
                    emit("else:")
                else:
                    match = re.match(r"(?:else\s*if|elseif)\s*\((.*)\)\s*then$", statement)
                    assert match, statement
                    emit("elif " + ast.unparse(parse_expr(match[1])) + ":")
                indent += 1
                continue
            assert kind == "if"
            if statement == "else":
                counter += 1
                mask = f"mask_{counter}"
                emit(f"{mask} = jnp.logical_and({parent}, jnp.logical_not({taken}))")
            else:
                match = re.match(r"(?:else\s*if|elseif)\s*\((.*)\)\s*then$", statement)
                assert match, statement
                counter += 1
                mask = f"mask_{counter}"
                emit(f"{mask} = jnp.logical_and(jnp.logical_and({parent}, jnp.logical_not({taken})), {expr(match[1])})")
                emit(f"{taken} = jnp.logical_or({taken}, {mask})")
            continue
        if statement.startswith("do ") or statement.startswith("do "):
            match = re.match(r"do\s+(\w+)\s*=\s*(.*)", statement)
            assert match, statement
            name, ranges = match.groups()
            if name == "k":
                stack.append(("level", mask, None))
                continue  # simultaneous independent levels
            assert name in LOOPS
            bounds = ranges.split(",")
            start, end = bounds[:2]
            step = bounds[2] if len(bounds) == 3 else "1"
            emit(f"for {name} in range({expr(start)}, ({expr(end)}) + (1 if ({expr(step)}) > 0 else -1), {expr(step)}):")
            stack.append(("for", mask, None))
            indent += 1
            continue
        if statement.startswith("if"):
            match = re.match(r"if\s*\((.*)\)\s*then$", statement)
            if match:
                condition = match[1]
                identifiers = {n.id for n in ast.walk(parse_expr(condition)) if isinstance(n, ast.Name)}
                if identifiers <= STATIC | LOOPS:
                    emit(f"if {ast.unparse(parse_expr(condition))}:")
                    stack.append(("static", mask, None)); indent += 1
                else:
                    counter += 1
                    child = f"mask_{counter}"; taken = f"taken_{counter}"
                    emit(f"{child} = jnp.logical_and({mask}, {expr(condition)})")
                    emit(f"{taken} = {child}")
                    stack.append(("if", mask, taken)); mask = child
                continue
            # Single-line IF; last balanced ')' separates condition from action.
            depth = 0; end = None
            for pos, char in enumerate(statement[statement.index("("):], statement.index("(")):
                depth += (char == "(") - (char == ")")
                if depth == 0:
                    end = pos; break
            assert end is not None
            condition, action = statement[statement.index("(")+1:end], statement[end+1:].strip()
            counter += 1
            child = f"mask_{counter}"
            emit(f"{child} = jnp.logical_and({mask}, {expr(condition)})")
            if action.startswith("goto"):
                dest = action.split()[-1]
                if dest == "555":
                    emit(f"alive = jnp.logical_and(alive, jnp.logical_not({child}))")
                    emit("level_alive = alive")
                elif dest == "444":
                    emit(f"alive = jnp.logical_and(alive, jnp.logical_not({child}))")
                else:
                    raise ValueError((line, statement))
            else:
                lhs, rhs = action.split("=", 1)
                parent = mask; mask = child; assignment(lhs.strip(), expr(rhs), line); mask = parent
            continue
        if statement.startswith("call "):
            node = parse_expr(statement[5:])
            name = node.func.id
            outs = CALL_OUTPUTS[name]
            ins = [expr(ast.unparse(arg)) for k, arg in enumerate(node.args) if k not in outs]
            # Inputs that are also updated by the callee remain real arguments.
            for k in outs:
                if name in {"get_cloud_dsd2", "get_rain_dsd2", "calc_bulkrhorime", "impose_max_total_ni"} and k in (0, 1, 2):
                    pass
            # All arguments are passed; helpers ignore undefined INTENT(OUT) placeholders.
            emit(f"call_out = _call_{name}(" + ", ".join(expr(ast.unparse(arg)) for arg in node.args) + ", tables=tables, three=log_3momentice)")
            for k, arg in enumerate(outs):
                assignment(ast.unparse(node.args[arg]).replace("allidx", ":"), f"call_out[{k}]", line)
            continue
        if "=" in statement:
            lhs, rhs = statement.split("=", 1)
            assignment(lhs.strip(), expr(rhs, strict=3770 <= line <= 3884), line)
        else:
            raise ValueError((line, statement))
    assert not stack, stack
    header = ['"""Generated literal P3 level-local processes; see proofs/s2_p3/port_processes.py."""',
              "from .process_support import *  # noqa: F403", "", "",
              "@_with_double", f"def {args.phase}(v, tables, *, ncat, log_predictnc, log_3momentice, observer=None):",
              "    v = dict(v)", "    iparam = 3", "    scpf_on = log_predictssat = False",
              "    niter_mui = 5", "    log_hmossopon = ncat > 1", "    alive = v['level_active']", "    level_alive = alive"]
    header += [f"    v.setdefault({name!r}, _initial(v['qv'], {name!r}, ncat))" for name in sorted(assigned)]
    # Read-before-write scalar scratch is defined once; the WRF branch masks
    # guarantee it cannot influence an executed path before its first write.
    all_names = set(re.findall(r"v\['(\w+)'\]", "\n".join(output)))
    header += [f"    assert {name!r} in v, 'P3 required WRF input/constant missing: {name}'" for name in sorted(all_names - assigned)]
    footer = ["    return v", ""]
    text = "\n".join(header + output + footer)
    ast.parse(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"{len(statements)} original statements -> {len(text.splitlines())} Python lines; {len(assigned)} assigned names")


if __name__ == "__main__":
    main()
