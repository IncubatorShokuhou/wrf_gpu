"""Prove the final staggered writer fix preserves the measured model operands."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import numpy as np

root, case, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(root / 'src'))
from gpuwrf.nesting import initial_feedback as current
from gpuwrf.coupling.boundary_feedback import build_state_feedback_weights

assert jax.devices()[0].platform == 'cpu'
baseline = '4605ba5e2'
path = 'src/gpuwrf/nesting/initial_feedback.py'
old_source = subprocess.check_output(['git', 'show', f'{baseline}:{path}'], cwd=root, text=True)
old_ast = ast.parse(old_source)
new_ast = ast.parse((root / path).read_text())
find = lambda tree, name: next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
assert ast.dump(find(old_ast, 'initialize_two_way_domains')) == ast.dump(find(new_ast, 'initialize_two_way_domains'))
namespace = dict(vars(current))
exec(compile(ast.Module(body=[find(old_ast, '_input_point_overrides')], type_ignores=[]), '<measured-helper>', 'exec'), namespace)
weights = build_state_feedback_weights(parent_grid_ratio=3, i_parent_start=46, j_parent_start=26,
    parent_grid=SimpleNamespace(nx=120, ny=70), child_grid=SimpleNamespace(nx=36, ny=30),
    spec_zone=1, smooth_option=2)
run = SimpleNamespace(path=case)
old = namespace['_input_point_overrides'](run, 'd01', 'd02', weights)
new = current._input_point_overrides(run, 'd01', 'd02', weights)
assert set(old) <= set(new)
common = {}
for name in sorted(old):
    a, b = np.asarray(old[name]), np.asarray(new[name])
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), name
    common[name] = hashlib.sha256(a.tobytes()).hexdigest()
added = sorted(set(new) - set(old))
assert added
assert all(current.FEEDBACK_FIELDS[name][2] in ('X', 'Y') for name in added)
readers = ['land_state.py', 'noahmp_land_init.py', 'land_history.py', 'gwdo_static.py']
for reader in readers:
    tree = ast.parse((root / 'src/gpuwrf/io' / reader).read_text())
    names = {x.value for x in ast.walk(tree) if isinstance(x, ast.Constant) and isinstance(x.value, str)}
    assert not set(added) & names, (reader, added)
source_delta = subprocess.check_output(['git', 'diff', '--name-only', baseline, 'HEAD', '--', 'src'], cwd=root, text=True).splitlines()
assert source_delta == ['src/gpuwrf/io/nesting_options.py', path], source_delta
result = dict(baseline=baseline, candidate=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    backend='cpu', cache='NONE', common_overrides_byte_identical=common, added_writer_fields=added,
    parent_initialization_body_AST_identical=True, physics_readers_exclude_added_names=readers,
    source_delta=source_delta, admission_proof='TB40-prejax-ratio-proof.json',
    scientific_reuse='TB37 original-WRF trajectories retained: added staggered output coordinates/maps are not read through field_overrides by physics; live metrics already use unchanged separate feedback',
    verdict='PASS')
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
