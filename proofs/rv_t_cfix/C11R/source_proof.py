import ast
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-cfix/C11R')
tree = root / 'snapshot'
def git(*args):
    return subprocess.check_output(['git', '-C', str(tree), *args])
base, fix, head = '0e645e8c1c', 'adc3fe84d9', 'aab981f480'
path = 'src/gpuwrf/coupling/scan_adapters.py'
assert git('diff', '--name-only', base, head, '--', 'src').decode().splitlines() == [path]
assert not git('diff', fix, head, '--', 'src')
assert not git('diff', '5c15881ede', base, '--', 'src')
assert not git('status', '--porcelain')
a, b = [ast.parse(git('show', rev + ':' + path)) for rev in [base, head]]
changed_functions = []
old_functions = {n.name: ast.dump(n) for n in a.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
for node in b.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.dump(node) != old_functions[node.name]:
        changed_functions.append(node.name)
assert changed_functions == ['wdm6_adapter']
class Normalize(ast.NodeTransformer):
    count = 0
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == '_small_grid_firewall':
            assert ast.unparse(node.args[0]) == 'mp(state.Nn)'
            assert ast.unparse(node.args[1]) == '(ny, nx)'
            self.count += 1
            return node.args[0]
        return node
normal = Normalize()
b = normal.visit(b)
b.body = [n for n in b.body if not (isinstance(n, ast.ImportFrom) and n.module == 'gpuwrf.kernels.layout_pin')]
assert normal.count == 1 and ast.dump(a) == ast.dump(b)
paths = ['src/gpuwrf/kernels/layout_pin.py', 'src/gpuwrf/physics/microphysics_wdm6.py']
unchanged = {}
for name in paths:
    blobs = [git('show', rev + ':' + name) for rev in [base, head]]
    assert blobs[0] == blobs[1]
    unchanged[name] = hashlib.sha256(blobs[0]).hexdigest()
ids = {}
for name in ['swiss', 'prod', 'c11', 'c11_main']:
    receipt = json.loads(Path('<USER_HOME>/wrf_gpu2_lanes/t-cfix-cu/identity6/' + name + '.json').read_text())
    assert receipt['port_probe']['platform'] == 'cpu' and not receipt['port_probe']['gpu_executed']
    ids[name] = {k: receipt['arms']['release'][k] for k in ['sha', 'chars', 'bound']}
out = dict(base=base, fixed_model=fix, packet=head, immutable_clean=True,
           source_5c15_equals_base=True, source_fix_equals_packet=True,
           changed_functions=changed_functions,
           normalized_entire_module_AST_exact=True, unchanged_bytes=unchanged,
           identities=ids, identity_scope='Producer native-facade CPU trace receipts carried with independent source proof; no fresh critic trace or GPU execution')
(root / 'source_proof.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
