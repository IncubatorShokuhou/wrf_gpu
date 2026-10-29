"""CPU-only admission controls using the real paid donor and its AOT metadata."""
import dataclasses
import pickle
import pytest
import runtime_meta as r
from test_partition_only import DONOR


@pytest.fixture(scope='module')
def donor():
    return pickle.loads(DONOR.with_suffix('.meta').read_bytes()), DONOR.read_bytes()


def test_only_runtime_fingerprint_changes(donor):
    meta, blob = donor
    flags = r.min1_flags(meta.fingerprint['xla_flags'])
    derived, receipt = r.derive_runtime_meta(meta, blob, flags)
    assert derived.fingerprint != meta.fingerprint, 'runtime derivation was deleted'
    assert derived.fingerprint['xla_flags'] == flags
    for field in dataclasses.fields(meta):
        if field.name != 'fingerprint':
            assert getattr(derived, field.name) is getattr(meta, field.name)
    live = dict(meta.fingerprint, xla_flags=flags)
    assert r.p.ax.fingerprint_matches(derived.fingerprint, live)
    assert not r.p.ax.fingerprint_matches(meta.fingerprint, live)
    assert not r.p.ax.fingerprint_matches(derived.fingerprint, dict(live, x64=not live['x64']))
    assert receipt['entire_blob_byte_identical']
    assert not receipt['runtime_capture_verified']


def test_other_flag_delta_refused(donor):
    meta, blob = donor
    flags = r.min1_flags(meta.fingerprint['xla_flags']) + ' --xla_gpu_autotune_level=0'
    with pytest.raises(ValueError, match='only min5'):
        r.derive_runtime_meta(meta, blob, flags)


def test_damaged_blob_refused(donor):
    meta, blob = donor
    with pytest.raises(ValueError, match='SHA mismatch'):
        r.derive_runtime_meta(meta, blob[:-1], r.min1_flags(meta.fingerprint['xla_flags']))


def test_changed_stored_option_refused(donor):
    meta, blob = donor
    modified = r.p.derive(blob)
    meta = dataclasses.replace(meta, blob_sha256=r.p.sha(modified))
    with pytest.raises(ValueError, match='varint 5'):
        r.derive_runtime_meta(meta, modified, r.min1_flags(meta.fingerprint['xla_flags']))


@pytest.mark.parametrize('flags', [
    '--xla_gpu_graph_min_graph_size=1',
    '--xla_gpu_graph_min_graph_size=5 --xla_gpu_graph_min_graph_size=5',
    '--some_flag_without_value',
    None,
])
def test_ambiguous_donor_flags_refused(flags):
    with pytest.raises(ValueError):
        r.min1_flags(flags)
