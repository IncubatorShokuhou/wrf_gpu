"""Deletion-sensitive checks against a real compiled-thunk GPU donor; CPU only."""
from pathlib import Path
import pytest
import partition_only as p

DONOR = Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/jax/aot/'
             '0.4.3-jax0.10.0-jaxlib0.10.0-cuda_sm120/d02/'
             '4e8d00fa090ded2ca40ebc7c785a5ddd264b1c96917a7ef656867960d221d554.xlaexec')


@pytest.fixture(scope='module')
def donor():
    return DONOR.read_bytes()


def rebuild(blob, rest):
    head, env, native, _ = p.unpack(blob)
    return head + p.asm._repack([(1, 2, p.asm._append_simple_chunk(native, [rest]))] + env[1:])


def test_real_native_identity_and_partition_delta(donor):
    changed = p.derive(donor)
    partition_changed = changed != donor
    assert partition_changed, 'partition derivation was deleted'
    bh, be, native, old_rest = p.unpack(donor)
    ah, ae, new_native, new_rest = p.unpack(changed)
    assert p.rest_min(new_rest, 1, 5) == old_rest
    assert new_native == native
    assert p.verify_delta(donor, changed)['all_other_rest_bytes_identical']


def test_noop_is_refused(donor):
    with pytest.raises(ValueError, match='varint 1'):
        p.verify_delta(donor, donor)


def test_repatch_is_refused(donor):
    with pytest.raises(ValueError, match='varint 5'):
        p.derive(p.derive(donor))


def test_other_rest_change_is_refused(donor):
    changed = p.derive(donor)
    _, _, _, rest = p.unpack(changed)
    # Unknown field is deliberately preserved by protobuf, but forbidden here.
    fs = p.fields(rest) + [(19000, 0, 7)]
    altered = rebuild(changed, p.asm._repack(fs))
    with pytest.raises(ValueError, match='beyond DebugOptions208'):
        p.verify_delta(donor, altered)


def test_native_chunk_change_is_refused(donor):
    changed = bytearray(p.derive(donor))
    head, env, native, _ = p.unpack(bytes(changed))
    # Corrupt data inside native prefix, away from chunk framing. The verifier
    # must refuse even if all rest/config fields still match.
    _, _, data_pos = p.asm._chunks(native)[-1]
    at = bytes(changed).index(native) + data_pos + 10
    changed[at] ^= 1
    with pytest.raises(ValueError, match='native record chunks'):
        p.verify_delta(donor, bytes(changed))


@pytest.mark.parametrize('payload', [b'', b'bad', b'\x80'])
def test_invalid_format_is_refused(payload):
    with pytest.raises(ValueError):
        p.derive(payload)


def test_missing_and_duplicate_min_refused():
    def rest(dbg):
        return p.asm._repack([(1, 2, p.asm._repack([(2, 2, p.asm._repack([(14, 2, dbg)]))]))])
    for dbg in (b'', p.asm._repack([(208, 0, 5), (208, 0, 5)])):
        with pytest.raises(ValueError, match='exactly one'):
            p.rest_min(rest(dbg), 5, 1)


def test_wrong_wire_type_refused():
    dbg = p.asm._repack([(208, 2, b'5')])
    rest = p.asm._repack([(1, 2, p.asm._repack([(2, 2, p.asm._repack([(14, 2, dbg)]))]))])
    with pytest.raises(ValueError, match='varint 5'):
        p.rest_min(rest, 5, 1)
