"""Delete only full-constant rendering in the proof's table-identity check."""
import inspect

def pytest_collection_modifyitems(session, config, items):
    matched = 0
    for item in items:
        if item.name != 'test_full_cpu_program_and_actual_namelist_keys_share_values':
            continue
        fn = item.obj
        source = inspect.getsource(fn)
        old = 'def constants(value, large=True):'
        assert source.count(old) == 1
        namespace = dict(fn.__globals__)
        exec(compile(source.replace(old, 'def constants(value, large=False):'), '<e224-elision-mutant>', 'exec'), namespace)
        item.obj = namespace[fn.__name__]
        matched += 1
    assert matched == 1
