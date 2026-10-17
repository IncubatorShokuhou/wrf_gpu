"""Private pytest plugin: delete rain bucket restoration in memory only.

Run pytest -p proofs.u1_post.bucket_mutant -k bucket_restore_is_deletion_sensitive.
Expected: genuine numeric AssertionError, 0 setup errors. Never edit source files.
"""
import inspect


def pytest_sessionstart(session):
    from gpuwrf.post import products
    source = inspect.getsource(products.accumulator)
    expression = "a = a + bucket*count"
    assert source.count(expression) == 1
    exec(compile(source.replace(expression, "a = a  # MUTANT: bucket restoration deleted"),
                 "<bucket-restoration-deletion-mutant>", "exec"), vars(products))
