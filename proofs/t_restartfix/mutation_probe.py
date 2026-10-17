"""Delete only the final JSON canonicalization, in process on a frozen tree."""
import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/v025'))
import cpu_guard  # noqa: F401 -- must precede any JAX/gpuwrf import
from gpuwrf.runtime import restart_store
import pytest

tree = ast.parse(Path(restart_store.__file__).read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'nested_identity')
assert isinstance(function.body[-1], ast.Return)
function.body[-1].value = ast.Name(id='identity', ctx=ast.Load())
namespace = dict(restart_store.__dict__)
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
             restart_store.__file__, 'exec'), namespace)
restart_store.nested_identity = namespace['nested_identity']
sys.exit(pytest.main(['tests/test_restart_identity.py', '-k', 'json_canonical',
                     '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/t-restartfix/pytest',
                     '--junitxml', '<USER_HOME>/wrf_gpu2_lanes/t-restartfix/RF06_mutant.xml',
                     '-q', '--tb=short']))
