"""Pytest plugin: genuinely delete the regional post-solve branch in memory.

Run only test_post_solve_observer_hook_and_default_no_work; a genuine assertion
failure is required, with no setup errors. Original production files stay intact.
"""
import ast
import hashlib
import inspect


def pytest_runtest_setup(item):
    from gpuwrf.runtime import operational_mode as om
    source = inspect.getsource(om._physics_boundary_step)
    tree = ast.parse(source)
    function = tree.body[0]
    candidates = [stmt for stmt in function.body if isinstance(stmt,ast.If)
                  and 'regional_streams' in ast.unparse(stmt.test)]
    assert len(candidates)==1
    function.body.remove(candidates[0])
    mutated = ast.unparse(tree)
    print('SOURCE_HOOK_DELETION',hashlib.sha256(source.encode()).hexdigest(),hashlib.sha256(mutated.encode()).hexdigest())
    exec(compile(tree,inspect.getsourcefile(om),'exec'),om.__dict__)
