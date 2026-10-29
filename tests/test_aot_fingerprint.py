"""Host admission controls: missing fingerprint fields never imply a match."""
import dataclasses
import pickle

import jax
import pytest

from gpuwrf.runtime import aot_executable as ax, aot_precompile as ap, aot_cheap_key as ck
from tests.test_aot_executable import _compile_dropping_graph
from tests._jax_cache_isolation import private_jax_cache


@pytest.fixture
def live():
    return ax.target_fingerprint(jax.devices()[0])


def test_empty_fingerprint_refused(live):
    assert not ax.fingerprint_matches({}, live)
    assert ax.fingerprint_matches({}, live, allow_legacy_fingerprint=True)


@pytest.mark.parametrize('key', ['jaxlib_version', 'jax_version', 'device_kind',
                                  'platform', 'compute_capability', 'cuda_driver',
                                  'x64', 'xla_flags'])
def test_each_required_field_missing_or_mismatched(live, key):
    assert ax.fingerprint_matches(live, live)
    partial = dict(live)
    del partial[key]
    assert not ax.fingerprint_matches(partial, live)
    assert ax.fingerprint_matches(partial, live, allow_legacy_fingerprint=True)
    bad = dict(live, **{key: 'MISMATCH'})
    assert not ax.fingerprint_matches(bad, live)
    assert not ax.fingerprint_matches(bad, live, allow_legacy_fingerprint=True)


def test_unknown_target_refused_except_cpu_capability(live):
    assert live['platform'] == 'cpu'
    assert live['compute_capability'] is None
    assert ax.fingerprint_matches(live, live)
    for key in live:
        if key == 'compute_capability':
            continue
        unknown = dict(live, **{key: None})
        assert not ax.fingerprint_matches(unknown, unknown), key
    gpu_unknown = dict(live, platform='gpu', compute_capability=None)
    assert not ax.fingerprint_matches(gpu_unknown, gpu_unknown)


@pytest.mark.parametrize('missing', [True, False])
def test_internal_loaders_require_named_legacy_opt_in(tmp_path, monkeypatch, missing):
    with private_jax_cache(monkeypatch, tmp_path / 'jit') as cache:
        compiled, args, expected = _compile_dropping_graph(0)
        blob, meta = ax.serialize(compiled)
        if missing:
            meta = dataclasses.replace(meta, fingerprint={})
        opts = {} if missing else {'check_fingerprint': False}
        error = 'fingerprint mismatch' if missing else 'requires allow_legacy_fingerprint=True'
        with pytest.raises(RuntimeError, match=error):
            ax.load(blob, meta, **opts)
        call = ax.load(blob, meta, allow_legacy_fingerprint=True, **opts)
        assert (call(*args)['y'] == expected['y']).all()
        bp, mp = ap._aot_blob_paths('d01', cache)
        ap._atomic_write_bytes(bp, blob)
        ap._atomic_write_bytes(mp, pickle.dumps(meta))
        for loader in (ap.load_domain_blob, ap._load_domain_blob):
            call, status = loader('d01', cache, return_status=True, **opts)
            assert call is None and not status['loaded']
            assert error in status['error']
            call, status = loader('d01', cache, return_status=True,
                                  allow_legacy_fingerprint=True, **opts)
            assert status['loaded'] and call is not None
            assert (call(*args)['y'] == expected['y']).all()


def test_unnamed_bypass_cannot_consume_prefetch(tmp_path, monkeypatch):
    with private_jax_cache(monkeypatch, tmp_path / 'jit') as cache:
        compiled, _, _ = _compile_dropping_graph(0)
        status = ap._serialize_domain_blob('d01', compiled, str(cache), cheap_key='a' * 64,
                                          key_schema=ck.KEY_SCHEMA)
        assert status['aot_written']
        with ap.prefetch_domain_blobs({'d01': 'a' * 64}, cache, dev=jax.devices()[0]) as handle:
            call, status = handle.futures['d01'].result(timeout=10)
            assert call is not None, status
            call, status = ap.load_domain_blob('d01', cache, cheap_key='a' * 64,
                                               check_fingerprint=False, return_status=True)
            assert call is None and not status['loaded']
            assert 'requires allow_legacy_fingerprint=True' in status['error']
        assert not ap._prefetched_loads and not ap._active_prefetches
