"""Replay a reviewed, faulty function against the new genuine controls.

Only function source is replaced in memory; tracked sources and reference
fixtures stay untouched. Exit zero requires one test failure and no setup
errors, with the failure attributable to the reverted guard.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

REVIEWED = '0feac83ce84816904e983b172aacb062d7049ce4'


def restore_function(module, path, name):
    source = subprocess.check_output(['git', 'show', REVIEWED + ':' + path], text=True)
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), module.__dict__)
    return hashlib.sha256(ast.unparse(node).encode()).hexdigest()


def delete_read_guards(module, name):
    tree = ast.parse(inspect.getsource(getattr(module, name)))
    original = ast.unparse(tree)
    removed = []

    class Drop(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id in ('validate_urban_bep', 'validate_urban_radiation')):
                removed.append(call.func.id)
                return None
            return self.generic_visit(node)

    mutant = Drop().visit(tree)
    assert sorted(removed) == ['validate_urban_bep', 'validate_urban_radiation']
    exec(compile(ast.fix_missing_locations(mutant), module.__file__, 'exec'), module.__dict__)
    return dict(original=hashlib.sha256(original.encode()).hexdigest(),
                mutant=hashlib.sha256(ast.unparse(mutant).encode()).hexdigest())


def run(kind, out):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    root = Path.cwd()
    assert (root / 'tests/test_v040_urban_boundaries.py').is_file()
    # Match tests/conftest.py's required order before importing model modules.
    sys.path.insert(0, str(root / 'scripts/v025'))
    import cpu_guard  # noqa: F401
    if kind == 'restart':
        from gpuwrf.io import restart_schema as module
        source_hashes = {}
        for name in ('validate_urban_bep', 'validate_urban_radiation'):
            source_hashes[name] = restore_function(module, 'src/gpuwrf/io/restart_schema.py', name)
            old = getattr(module, name)
            # The reviewed validator predates explicit NetCDF domain kwargs.
            setattr(module, name, lambda group, namelist, _old=old, **kwargs: _old(group, namelist))
        node = 'test_checkpoint_rejects_every_missing_field_shape_and_nonfinite_on_write_and_read[pickle-2]'
        expected = 'DID NOT RAISE'
    elif kind.startswith('read-'):
        fmt = kind.removeprefix('read-')
        if fmt == 'pickle':
            from gpuwrf.io import restart as module
            function = '_read_payload'
        else:
            from gpuwrf.io import wrfrst_netcdf as module
            function = '_read_optional_carry_groups'
        source_hashes = {function: delete_read_guards(module, function)}
        node = 'test_checkpoint_rejects_every_missing_field_shape_and_nonfinite_on_write_and_read[' + fmt + '-2]'
        expected = 'DID NOT RAISE'
    else:
        import pbl_parity as module
        source_hashes = {'run': restore_function(module, 'proofs/v040/urban_bep_bem/pbl_parity.py', 'run')}
        node = 'test_one_nonfinite_original_output_is_rejected_before_physics[pbl-nan]'
        expected = 'corrupted reference reached physics/JIT'
    reports = []

    class Reports:
        def pytest_runtest_logreport(self, report):
            reports.append(report)

    code = pytest.main(['-q', 'tests/test_v040_urban_boundaries.py::' + node,
                        '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/s1-bepbem/pytest',
                        '-p', 'no:cacheprovider'], plugins=[Reports()])
    failures = [r for r in reports if r.when == 'call' and r.failed]
    errors = [r for r in reports if r.when != 'call' and r.failed]
    killed = code == 1 and len(failures) == 1 and not errors and expected in str(failures[0].longrepr)
    receipt = dict(kind=kind, reviewed=REVIEWED, source_function_hashes=source_hashes,
                   pytest_exit=int(code), call_failures=len(failures), setup_errors=len(errors),
                   expected_failure=expected, killed=killed)
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    assert killed, receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('restart', 'read-pickle', 'read-netcdf', 'pbl'))
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.kind, args.out)
