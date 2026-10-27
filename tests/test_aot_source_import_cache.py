"""Source-scope cache proofs against full ast.walk and real package files."""
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from gpuwrf.runtime import aot_cheap_key as ck


def reference(data, filename, cache_dir=None):
    # Historical full-AST contribution, independent of the optimized traversal.
    rows=[]
    for node in ast.walk(ast.parse(data.decode('utf-8'), filename=filename)):
        if isinstance(node, ast.Import):
            rows.append(['import',0,None,[a.name for a in node.names]])
        elif isinstance(node, ast.ImportFrom):
            rows.append(['from',node.level,node.module,[a.name for a in node.names]])
    return rows


def unordered(rows):
    return Counter(json.dumps(row) for row in rows)


def test_real_package_contributions_equal_historical_full_ast():
    root=ck._gpuwrf_package_root()
    files=sorted(root.rglob('*.py'))
    assert root/'runtime/operational_mode.py' in files
    for path in files:
        data=path.read_bytes()
        assert unordered(ck._import_cache_payload(data,str(path))) == unordered(reference(data,str(path))), path


def test_all_nested_statement_containers():
    data=b'''import gpuwrf.top
def outer():
    import gpuwrf.function
    class Nested:
        from . import class_body
        def method(self):
            if True:
                from ..deep import x
            else:
                import gpuwrf.other
    for i in ():
        import gpuwrf.for_body
    else:
        import gpuwrf.for_else
    while False:
        import gpuwrf.while_body
    else:
        import gpuwrf.while_else
    with context():
        import gpuwrf.with_body
    try:
        import gpuwrf.try_body
    except Exception:
        import gpuwrf.except_body
    else:
        import gpuwrf.try_else
    finally:
        import gpuwrf.finally_body
    try:
        pass
    except* Exception:
        import gpuwrf.except_star
    match value:
        case [x]:
            import gpuwrf.match_body
async def async_outer():
    async for i in items:
        import gpuwrf.async_for
    async with context():
        import gpuwrf.async_with
'''
    assert unordered(ck._import_cache_payload(data,'grammar.py')) == unordered(reference(data,'grammar.py'))


@pytest.fixture
def package_copy(tmp_path,monkeypatch):
    original=ck._gpuwrf_package_root();root=tmp_path/'src/gpuwrf'
    # Fingerprint inputs are source bytes only; no model assets/caches copied.
    for path in original.rglob('*.py'):
        dest=root/path.relative_to(original);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    monkeypatch.setattr(ck,'_gpuwrf_package_root',lambda:root)
    cache=tmp_path/'imports';monkeypatch.setattr(ck,'_source_import_cache_dir',lambda:cache)
    ck.source_fingerprint_hash.cache_clear()
    yield root,cache
    ck.source_fingerprint_hash.cache_clear()


def fingerprint():
    ck.source_fingerprint_hash.cache_clear()
    return ck.source_fingerprint_hash()


def test_full_tree_fingerprint_and_aot_key_cold_cached_legacy(package_copy,monkeypatch):
    root,cache=package_copy
    class Static:
        def tree_flatten(self): return (),()
    def fn(x): return x+1
    args=(np.zeros((2,),np.float32),)
    def key():
        result=ck.cheap_key(fn,args,{},Static());assert result is not None;return result
    with monkeypatch.context() as m:
        m.setattr(ck,'_source_import_specs',reference)
        legacy=fingerprint();legacy_key=key()
    cold=fingerprint();cold_key=key()
    assert list(cache.glob('*.json'))
    def no_parse(*a): raise AssertionError('a cached source contribution was reparsed')
    with monkeypatch.context() as m:
        m.setattr(ck,'_import_cache_payload',no_parse)
        cached=fingerprint();cached_key=key()
    assert legacy==cold==cached
    assert legacy_key==cold_key==cached_key


def test_one_byte_same_stat_edit_recomputes_only_that_file(package_copy,monkeypatch):
    root,cache=package_copy;base=fingerprint()
    path=root/'kernels/layout_pin.py';data=path.read_bytes();stat=path.stat()
    # One docstring byte changes the source key, not its import edges.
    at=data.index(b'"""')+3
    replacement=b'X' if data[at:at+1]!=b'X' else b'Y'
    path.write_bytes(data[:at]+replacement+data[at+1:]);os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    assert path.stat().st_size==stat.st_size and path.stat().st_mtime_ns==stat.st_mtime_ns
    calls=[];original=ck._import_cache_payload
    def counted(data,name): calls.append(name);return original(data,name)
    with monkeypatch.context() as m:
        m.setattr(ck,'_import_cache_payload',counted);changed=fingerprint()
    assert calls==[str(path)]
    with monkeypatch.context() as m:
        m.setattr(ck,'_source_import_specs',reference);expected=fingerprint()
    assert changed==expected and changed!=base


def test_new_import_edge_discovers_new_file(package_copy,monkeypatch):
    root,cache=package_copy;base=fingerprint()
    extra=root/'source_cache_probe.py';extra.write_text('sentinel = 1\n')
    assert fingerprint()==base
    entry=root/'runtime/operational_mode.py';entry.write_bytes(entry.read_bytes()+b'\nimport gpuwrf.source_cache_probe\n')
    changed=fingerprint();assert changed!=base
    assert extra in ck._trace_reachable_source_files(root)
    with monkeypatch.context() as m:
        m.setattr(ck,'_source_import_specs',reference);expected=fingerprint()
    assert changed==expected


@pytest.mark.parametrize('corruption',['truncated','payload','schema','source_digest'])
def test_corruption_recomputes(corruption,tmp_path,monkeypatch):
    data=b'from gpuwrf.kernels import layout_pin\n';cache=tmp_path/'cache'
    expected=ck._source_import_specs(data,'file.py',cache);path,=cache.glob('*.json')
    record=json.loads(path.read_text())
    if corruption=='truncated': path.write_bytes(path.read_bytes()[:25])
    else:
        key={'payload':'imports','schema':'schema','source_digest':'source_sha256'}[corruption]
        record['payload'][key]=[] if corruption=='payload' else 'wrong'
        path.write_text(json.dumps(record))
    calls=[];original=ck._import_cache_payload
    def counted(*args): calls.append(1);return original(*args)
    monkeypatch.setattr(ck,'_import_cache_payload',counted)
    assert ck._source_import_specs(data,'file.py',cache)==expected
    assert calls==[1]
    monkeypatch.setattr(ck,'_import_cache_payload',lambda *a:pytest.fail('repair was not cached'))
    assert ck._source_import_specs(data,'file.py',cache)==expected


def test_failed_atomic_replace_recomputes_and_cleans_temp(tmp_path,monkeypatch):
    data=b'import gpuwrf.kernels.layout_pin\n';cache=tmp_path/'cache'
    def failed(*a): raise OSError('interrupted replace')
    with monkeypatch.context() as m:
        m.setattr(ck.os,'replace',failed)
        assert ck._source_import_specs(data,'file.py',cache)==reference(data,'file.py')
    assert list(cache.iterdir())==[]
    assert ck._source_import_specs(data,'file.py',cache)==reference(data,'file.py')
    assert len(list(cache.iterdir()))==1


def test_unwritable_cache_falls_back(tmp_path):
    blocked=tmp_path/'not_a_directory';blocked.write_text('blocked')
    data=b'import gpuwrf.kernels.layout_pin\n'
    assert ck._source_import_specs(data,'file.py',blocked)==reference(data,'file.py')


def test_relative_edges_are_resolved_per_module(tmp_path,monkeypatch):
    path=tmp_path/'module.py';path.write_text('from . import neighbor\n')
    cache=tmp_path/'cache';monkeypatch.setattr(ck,'_source_import_cache_dir',lambda:cache)
    assert ck._gpuwrf_imports_in_file(path,'gpuwrf.one.module')=={'gpuwrf.one','gpuwrf.one.neighbor'}
    assert ck._gpuwrf_imports_in_file(path,'gpuwrf.two.module')=={'gpuwrf.two','gpuwrf.two.neighbor'}
    assert len(list(cache.glob('*.json')))==1


def test_algorithm_version_invalidates_contribution_cache(tmp_path,monkeypatch):
    data=b'import gpuwrf\n';cache=tmp_path/'cache';expected=ck._source_import_specs(data,'file.py',cache)
    monkeypatch.setattr(ck,'_SOURCE_IMPORT_CACHE_SCHEMA','test-next-version')
    assert ck._source_import_specs(data,'file.py',cache)==expected
    assert len(list(cache.glob('*.json')))==2


def test_concurrent_writers_publish_complete_entry(tmp_path):
    cache=tmp_path/'cache';data=b'from gpuwrf.kernels import layout_pin\n'
    with ThreadPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(lambda _:ck._source_import_specs(data,'file.py',cache),range(8)))
    assert all(x==reference(data,'file.py') for x in values)
    assert len(list(cache.iterdir()))==1
    record=json.loads(next(cache.iterdir()).read_text());assert record['payload']['imports']==values[0]


def test_cache_disable_respected(monkeypatch):
    monkeypatch.setenv('GPUWRF_JAX_CACHE','0')
    assert ck._source_import_cache_dir() is None

