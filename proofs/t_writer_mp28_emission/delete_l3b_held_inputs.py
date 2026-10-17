"""Delete the actual runner's held-emission seed; expect consumption ASSERT."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

mutated = False


def pytest_runtest_setup(item):
    global mutated
    if mutated:
        return
    from test_frozen_emission_threading import RUNNER
    source = inspect.getsource(RUNNER.run_port)
    tree = ast.parse(source)
    class Delete(ast.NodeTransformer):
        count = 0
        def visit_Assign(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'replace'
                    and {'nwfa2d','nifa2d'} <= {k.arg for k in call.keywords}):
                self.count += 1
                return ast.copy_location(ast.Pass(), node)
            return self.generic_visit(node)
    edit = Delete();tree = edit.visit(tree)
    assert edit.count == 1
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(RUNNER.__file__)+'::<held-seed-deleted>', 'exec'), vars(RUNNER))
    mutated = True
    Path(os.environ['T_WRITER_MUTANT_RECEIPT']).write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(), deletions=edit.count,
        mutant_ast=ast.unparse(tree)), indent=2)+'\n')
