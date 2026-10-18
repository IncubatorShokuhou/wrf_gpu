"""Bounded syntax-only date bindings. No eval, imports of readers, or data IO.

Values are sets of literal alternatives. Unresolved names stay placeholders.
Function parameters receive only syntactically present call/decorator values;
Python functions, methods, decorators and comprehensions are never executed.
"""
import ast
from datetime import datetime, timedelta
import itertools
import json
import re

ROOT = re.compile(r'^/mnt/(?:data/alisios|server/work/src/alisios)(?:/|$)')
DAY = re.compile(r'(?<!\d)(20\d{2})-?(\d{2})-?(\d{2})(?!\d)')
LIMIT = 128


def days(value):
    out = set()
    for m in DAY.finditer(value):
        try:
            out.add(datetime(*map(int, m.groups())).date().isoformat())
        except ValueError:
            pass
    return sorted(out)


def extract(text):
    facts = []
    lines = text.splitlines()

    def fact(kind, paths, ds, node=None, evidence=None, bounded=False, bindings=None):
        paths = sorted({p for p in paths if isinstance(p, str) and ROOT.match(p)})
        if not paths:
            return
        # ast.get_source_segment splits the entire input on every fact. A
        # bounded source-line excerpt preserves location/hash provenance with
        # much less CPU work for the 133 immutable source files.
        excerpt = '' if node is None else '\n'.join(lines[node.lineno - 1:min(node.end_lineno, node.lineno + 11)])[:1600]
        facts.append({'kind': kind, 'paths': paths, 'valid_days': sorted(set(ds)),
                      'bounded_valid_extent': bounded,
                      'line': getattr(node, 'lineno', None),
                      'evidence': evidence or excerpt,
                      'bindings': bindings or {}})

    try:
        obj = json.loads(text)
    except ValueError:
        obj = None
    if obj is not None:
        def visit(v, key='$'):
            if isinstance(v, dict):
                paths = [x for x in v.values() if isinstance(x, str) and ROOT.match(x)]
                # Forecast windows only. start_utc/end_utc describe the reader's
                # execution, so must never become forecast/valid dates here.
                candidates = [(v, key)]
                if isinstance(v.get('window'), dict):
                    candidates.append((v['window'], key + '.window'))
                for window, location in candidates:
                    for sk, ek in (('start_time', 'end_time'), ('start_date', 'end_date'),
                                   ('valid_start', 'valid_end')):
                        if paths and isinstance(window.get(sk), str) and isinstance(window.get(ek), str):
                            try:
                                start = datetime.fromisoformat(window[sk].replace('_', 'T').replace('Z', '+00:00'))
                                end = datetime.fromisoformat(window[ek].replace('_', 'T').replace('Z', '+00:00'))
                                n = (end.date() - start.date()).days
                                if 0 <= n <= 366:
                                    ds = [(start + timedelta(days=i)).date().isoformat() for i in range(n + 1)]
                                    fact('SAME_OBJECT_SOURCE_FORECAST_WINDOW', paths, ds, bounded=True,
                                         evidence=f'{key} source sibling; {location}.{sk}={window[sk]}; {location}.{ek}={window[ek]}')
                            except ValueError:
                                pass
                for k, x in v.items():
                    visit(x, key + '.' + k)
            elif isinstance(v, list):
                for i, x in enumerate(v):
                    visit(x, f'{key}[{i}]')
        visit(obj)
        return facts

    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Shell caller defaults: recover the known default without certifying
        # that an override or an unknown history extent has that same day.
        defaults = dict(re.findall(r'(\w+)=\$\{\d+:-([^}]+)\}', text))
        for i, line in enumerate(lines, 1):
            for m in re.finditer(r'/mnt/(?:data/alisios|server/work/src/alisios)/[^\s"\';]+', line):
                path = m.group()
                for name, value in defaults.items():
                    path = path.replace('${' + name + '%%_icp*}', value.split('_icp', 1)[0])
                    path = path.replace('${' + name + '}', value).replace('$' + name, value)
                if path != m.group():
                    fact('SHELL_DEFAULT_CASE_OVERRIDE_UNKNOWN', [path], [],
                         evidence=f'line {i}: {line}; defaults={defaults}')
        return facts

    def cap(v):
        return set(sorted(v)[:LIMIT])

    def val(n, env):
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, int, float)):
            return {str(n.value)}
        if isinstance(n, ast.Name):
            return env.get(n.id, {'{' + n.id + '}'})
        if isinstance(n, ast.Attribute):
            return env.get(ast.unparse(n), {'{' + ast.unparse(n) + '}'})
        if isinstance(n, (ast.List, ast.Tuple, ast.Set)):
            return cap(set().union(*(val(x, env) for x in n.elts))) if n.elts else set()
        if isinstance(n, ast.Dict):
            return cap(set().union(*(val(x, env) for x in n.values if x is not None)))
        if isinstance(n, ast.Subscript):
            # For an unknown map index, retain all explicit map values. These
            # are alternatives, not an assertion that every arm executed.
            if isinstance(n.value, ast.Dict):
                return val(n.value, env)
            if isinstance(n.value, ast.Name):
                return env.get(n.value.id, {'{' + ast.unparse(n) + '}'})
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.Add)):
            out = set()
            for a, b in itertools.islice(itertools.product(sorted(val(n.left, env)), sorted(val(n.right, env))), LIMIT):
                out.add(a.rstrip('/') + '/' + b if isinstance(n.op, ast.Div) else a + b)
            return out
        if isinstance(n, ast.JoinedStr):
            out = {''}
            for x in n.values:
                if isinstance(x, ast.Constant):
                    choices = {x.value}
                else:
                    choices = val(x.value, env)
                    if x.format_spec:
                        spec = next(iter(val(x.format_spec, env)), '')
                        formatted = set()
                        for s in choices:
                            try:
                                formatted.add(format(int(s), spec) if spec.endswith('d') else format(s, spec))
                            except (ValueError, TypeError):
                                formatted.add('{' + ast.unparse(x.value) + ':' + spec + '}')
                        choices = formatted
                out = cap({a + b for a, b in itertools.islice(itertools.product(sorted(out), sorted(choices)), LIMIT)})
            return out
        if isinstance(n, ast.Call):
            name = ast.unparse(n.func)
            if name in ('Path', 'str', 'os.fspath') and n.args:
                return val(n.args[0], env)
            if isinstance(n.func, ast.Attribute) and n.func.attr in ('resolve', 'absolute'):
                return val(n.func.value, env)  # lexical identity only, NO filesystem operation
            if isinstance(n.func, ast.Attribute) and n.func.attr in ('items', 'values'):
                return val(n.func.value, env)
        return {'{' + ast.unparse(n)[:120] + '}'}

    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    args = {name: {} for name in funcs}
    definitions = {}

    def target(n, value, env):
        if isinstance(n, (ast.Name, ast.Attribute)):
            name = ast.unparse(n)
            env[name] = value
            definitions[name] = f'line {n.lineno}: {lines[n.lineno - 1]}'
        elif isinstance(n, (ast.Tuple, ast.List)):
            for x in n.elts:
                target(x, value, env)

    def assignment(n, value, env):
        if isinstance(n, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) and len(n.elts) == len(value.elts):
            for left, right in zip(n.elts, value.elts):
                target(left, val(right, env), env)
        else:
            target(n, val(value, env), env)

    def inspect(n, env):
        for x in ast.walk(n):
            if isinstance(x, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
                local = dict(env)
                for generator in x.generators:
                    target(generator.target, val(generator.iter, local), local)
                inspect(x.elt, local)
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(x, (ast.BinOp, ast.JoinedStr, ast.Call)):
                paths = {p for p in val(x, env) if ROOT.match(p)}
                if paths:
                    names = {y.id for y in ast.walk(x) if isinstance(y, ast.Name)}
                    bindings = {name: {'values': sorted(env[name]), 'definition': definitions.get(name, 'call/decorator/loop binding')}
                                for name in sorted(names) if name in env and len(env[name]) <= LIMIT}
                    fact('PYTHON_PATH_BINDING_ALTERNATIVES', paths, [], x, bindings=bindings)
            if isinstance(x, ast.Dict):
                for k, v in zip(x.keys, x.values):
                    if k is not None:
                        paths = val(k, env)
                        ds = [d for s in val(v, env) if re.fullmatch(r'20\d{6}|20\d{2}-\d{2}-\d{2}', s) for d in days(s)]
                        if ds:
                            fact('EXPLICIT_GUARD_PATH_DAY_MAPPING_EXTENT_UNKNOWN', paths, ds, x)
            if isinstance(x, ast.Assign):
                for t in x.targets:
                    if isinstance(t, ast.Subscript):
                        ds = [d for s in val(x.value, env) if re.fullmatch(r'20\d{6}|20\d{2}-\d{2}-\d{2}', s) for d in days(s)]
                        if ds:
                            fact('EXPLICIT_GUARD_PATH_DAY_MAPPING_EXTENT_UNKNOWN', val(t.slice, env), ds, x)
            if isinstance(x, ast.Call):
                name = ast.unparse(x.func)
                if name in funcs:
                    params = funcs[name].args.args
                    for p, actual in zip(params, x.args):
                        args[name][p.arg] = cap(args[name].get(p.arg, set()) | val(actual, env))
                    for kw in x.keywords:
                        if kw.arg:
                            args[name][kw.arg] = cap(args[name].get(kw.arg, set()) | val(kw.value, env))
                # This reader explicitly supplies the basename separately from
                # its case directory (rather than using a Path / expression).
                if name.split('.')[-1] == 'load_real_snapshot' and x.args:
                    for kw in x.keywords:
                        if kw.arg == 'wrfout_name':
                            fact('READER_CASE_PLUS_WRFOUT_NAME',
                                 {a.rstrip('/') + '/' + b for a in val(x.args[0], env) for b in val(kw.value, env)}, [], x)

    def scan(stmts, env):
        for n in stmts:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    assignment(t, n.value, env)
            elif isinstance(n, ast.AnnAssign) and n.value:
                assignment(n.target, n.value, env)
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                local = dict(env)
                target(n.target, val(n.iter, env), local)
                inspect(n.iter, env)
                scan(n.body, local)
                scan(n.orelse, dict(env))
                continue
            elif isinstance(n, (ast.If, ast.While, ast.With, ast.Try)):
                # Inspect individual expressions, then branches in separate
                # environments, so one synthetic branch cannot overwrite another.
                for field in ('test',):
                    if hasattr(n, field):
                        inspect(getattr(n, field), env)
                if isinstance(n, ast.With):
                    for item in n.items:
                        inspect(item.context_expr, env)
                for field in ('body', 'orelse', 'finalbody'):
                    scan(getattr(n, field, []), dict(env))
                for h in getattr(n, 'handlers', []):
                    scan(h.body, dict(env))
                continue
            inspect(n, env)
        return env

    global_env = scan(tree.body, {})
    # Bind decorator alternatives only to that function's named parameter.
    for name, fn in funcs.items():
        for deco in fn.decorator_list:
            if isinstance(deco, ast.Call) and ast.unparse(deco.func).endswith('parametrize') and len(deco.args) >= 2:
                if isinstance(deco.args[0], ast.Constant) and isinstance(deco.args[0].value, str) and ',' not in deco.args[0].value:
                    args[name][deco.args[0].value] = val(deco.args[1], global_env)
    for _ in range(4):
        for name, fn in funcs.items():
            local = dict(global_env)
            for p in fn.args.args:
                local[p.arg] = args[name].get(p.arg, {'{' + p.arg + '}'})
            scan(fn.body, local)
    unique = {}
    for f in facts:
        key = (f['kind'], tuple(f['paths']), tuple(f['valid_days']), f['line'], f['evidence'])
        unique[key] = f
    return list(unique.values())
