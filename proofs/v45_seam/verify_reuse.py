"""Bind existing physics-prefix receipts to the boundary-only experiment.

This source-equivalence evidence carries the two existing prefix results only;
it does not establish whole-root or actual GPU identity after a fusion change.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

before, after, output = sys.argv[1:]
file = 'src/gpuwrf/runtime/operational_mode.py'
git = lambda *args: subprocess.check_output(['git', *args], text=True)
assert git('diff', '--name-only', before, after, '--', 'src').splitlines() == [file]
trees = [ast.parse(git('show', pin + ':' + file)) for pin in (before, after)]
functions = [{n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
             for tree in trees]
assert functions[0].keys() == functions[1].keys()
changed = [name for name in functions[0] if ast.dump(functions[0][name]) != ast.dump(functions[1][name])]
assert changed == ['_acoustic_core_state_from_prep'], changed
visited = set()

def visit(name):
    if name in visited or name not in functions[1]:
        return
    visited.add(name)
    for node in ast.walk(functions[1][name]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            visit(node.func.id)

visit('_physics_step_forcing')
assert not set(changed) & visited
rows = []
for label, expected in [('BISECT_FIXED_PHYSICS01', 208), ('BISECT_FIXED_PRE_PBL01', 199)]:
    path = Path('<USER_HOME>/wrf_gpu2_lanes/v45-lsmparams') / label / 'receipt.json'
    raw = path.read_bytes()
    rec = json.loads(raw)
    assert git('rev-parse', rec['source_commit'] + ':src').strip() == git('rev-parse', before + ':src').strip()
    assert rec['phase'] == 'COMPLETE'
    assert set(rec['leaves'].values()) == {expected}
    assert rec['comparisons']['on0'] == {'default_byte_exact': True, 'differences': []}
    rows.append(dict(receipt=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                     commit=rec['source_commit'], byte_identical_leaves=expected))
record = dict(scope=__doc__, before=before, after=after, changed_functions=changed,
              unchanged_local_physics_call_closure=sorted(visited), reused=rows,
              whole_root_identity='UNMEASURED', actual_gpu_identity='UNMEASURED')
Path(output).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(changed_functions=changed, reused_leaves=[r['byte_identical_leaves'] for r in rows])))
