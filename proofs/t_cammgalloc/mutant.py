"""Restore one faulty behavior in memory; require a genuine regression failure."""
import ast
import inspect
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root, baseline, mode, out = sys.argv[1:]
sys.path.insert(0, str(Path(root) / 'src'))
sys.path.insert(0, str(Path(root) / 'scripts/v025'))
import cpu_guard  # Must precede JAX, including pytest's conftest (E50).
from gpuwrf.contracts import state as S
from gpuwrf.coupling import physics_couplers as C
import pytest

if mode == 'allocation':
    source = (Path(baseline) / 'src/gpuwrf/contracts/state.py').read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == 'conditional_state_leaves_for_mp')
    ns = dict(vars(S))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<allocation-mutant>', 'exec'), ns)
    S.conditional_state_leaves_for_mp = ns[node.name]
    target = 'test_scan_does_not_allocate_inactive_cammg[8]'
else:
    source = inspect.getsource(C._rrtmg_column_inputs_impl)
    before, after = (('if int(mp_physics) == 11:', 'if state.lradius is not None:') if mode == 'radii'
                     else ('if int(mp_physics) == 11 and state.cldfra_mp_all is not None:',
                           'if state.cldfra_mp_all is not None:'))
    assert source.count(before) == 1
    exec(compile(source.replace(before, after), '<radiation-mutant>', 'exec'), vars(C))
    target = 'test_radiation_ignores_other_scheme_cammg_fields[8]'
out = Path(out)
xml = out.with_suffix('.xml')
test = str(Path(root) / 'tests/v040/test_cammg_activation.py')
rc = pytest.main(['-q', '-p', 'no:cacheprovider', '--tb=short',
    '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/t-cammgalloc/pytest',
    '--junitxml', str(xml), test + '::' + target,
    test + '::test_active_cammg_scan_preserves_every_held_leaf'])
cases = ET.parse(xml).findall('.//testcase')
failures = [c for c in cases if c.find('failure') is not None]
errors = [c for c in cases if c.find('error') is not None]
assert rc == 1 and len(cases) == 2 and len(failures) == 1 and not errors
failure = failures[0].find('failure')
# Pytest assertion rewriting renders a bare assertion as "E assert False",
# while NumPy's numeric assertions retain the exception class name.
assert (failure.get('message', '').startswith('assert ')
        or 'AssertionError' in failure.text)
out.write_text(json.dumps(dict(mode=mode, rc=rc, failures=1, healthy_controls=1,
                              setup_errors=0, status='KILLED_ASSERTION'), indent=2) + '\n')
