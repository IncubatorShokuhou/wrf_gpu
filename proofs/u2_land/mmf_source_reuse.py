"""Check the pre-MMF option science without repeating unchanged numerical arms.

This is source equivalence, not WRF fidelity. Specialize only the new host-level
MMF routing to an absent MMF bundle; keep every remaining physics AST node.
The opt_run=5 missing-carry refusal is outside the reused run1/2/3/8 scope.
"""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

tree, out = map(Path, sys.argv[1:3])
base = '1732180bf'


def show(rev, path):
    return subprocess.check_output(['git', '-C', str(tree), 'show', f'{rev}:{path}'])


class AbsentMMF(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        positional = node.args.args
        first_default = len(positional) - len(node.args.defaults)
        node.args.defaults = [default for arg, default in zip(positional[first_default:], node.args.defaults)
                              if arg.arg not in ('mmf', 'itimestep')]
        node.args.args = [arg for arg in positional if arg.arg not in ('mmf', 'itimestep')]
        pairs = [(arg, default) for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                 if arg.arg not in ('mmf', 'itimestep')]
        node.args.kwonlyargs = [p[0] for p in pairs]
        node.args.kw_defaults = [p[1] for p in pairs]
        return self.generic_visit(node)

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.left, ast.Name) and node.left.id == 'mmf'
                and len(node.ops) == 1 and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value is None):
            return ast.Constant(isinstance(node.ops[0], ast.Is))
        return node

    def visit_If(self, node):
        # New admission-only guard, impossible for the four reused smoke arms.
        if 'options.opt_run == 5' in ast.unparse(node.test):
            assert 'MMF runtime carry' in ast.unparse(node)
            return []
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        if isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And):
            if any(isinstance(v, ast.Constant) and v.value is False for v in node.test.values):
                return []
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        return node

    def visit_Assign(self, node):
        if any(isinstance(t, ast.Name) and t.id == 'mmf_aux' for t in node.targets):
            return []
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        node.keywords = [k for k in node.keywords
                         if not (k.arg is None and isinstance(k.value, ast.Dict) and not k.value.keys)]
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Add) and isinstance(node.right, ast.Name) and node.right.id == 'mmf_aux':
            return node.left
        return node


targets = {
    'src/gpuwrf/physics/noahmp/noahmp_driver.py': 'noah_mp_step',
    'src/gpuwrf/physics/noahmp/water_hydro.py': 'noahmp_water_hydro',
    'src/gpuwrf/physics/noahmp_coupler.py': 'noahmp_surface_adapter',
    'src/gpuwrf/coupling/noahmp_surface_hook.py': 'noahmp_surface_step',
}
report = {'base': base, 'candidate': subprocess.check_output(
    ['git', '-C', str(tree), 'rev-parse', 'HEAD'], text=True).strip(),
    'scope': 'source equivalence for absent MMF; no new fidelity measurement', 'specialized': {}, 'byte_files': {}}
for path, name in targets.items():
    before = next(n for n in ast.parse(show(base, path)).body if isinstance(n, ast.FunctionDef) and n.name == name)
    after = next(n for n in ast.parse((tree / path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    after = ast.fix_missing_locations(AbsentMMF().visit(after))
    dumps = [ast.dump(n, include_attributes=False) for n in (before, after)]
    report['specialized'][path] = dict(exact=dumps[0] == dumps[1],
        before=hashlib.sha256(dumps[0].encode()).hexdigest(), after=hashlib.sha256(dumps[1].encode()).hexdigest())
    if dumps[0] != dumps[1]:
        out.with_suffix('.before.py').write_text(ast.unparse(before))
        out.with_suffix('.after.py').write_text(ast.unparse(after))
        out.write_text(json.dumps(report, indent=2) + '\n')
        raise AssertionError(path)
paths = subprocess.check_output(['git', '-C', str(tree), 'ls-tree', '-r', '--name-only', base,
    'src/gpuwrf/physics/noahmp', 'src/gpuwrf/physics/sst_skin.py',
    'src/gpuwrf/coupling/noahmp_option_tiles.py'], text=True).splitlines()
for path in paths:
    if path in targets:
        continue
    previous = show(base, path); current = (tree / path).read_bytes()
    assert previous == current, path
    report['byte_files'][path] = hashlib.sha256(current).hexdigest()
for path in ('src/gpuwrf/physics/noahmp/groundwater.py', 'src/gpuwrf/io/noahmp_mmf.py'):
    assert show('ca3168dd3', path) == (tree / path).read_bytes(), path
report['main_mmf_model_and_binder_byte'] = True
report['status'] = 'PASS'
out.write_text(json.dumps(report, indent=2) + '\n')
print('PASS', len(report['specialized']), 'routing ASTs;', len(report['byte_files']), 'physics files BYTE')
