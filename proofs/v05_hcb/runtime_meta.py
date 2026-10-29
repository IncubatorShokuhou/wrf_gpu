"""Proof-only metadata derivation for an IDENTICAL blob under min5/min1 reload.

This is not a generic cache re-key. Call the ordinary explicit AOT loader; never
install these metadata in the automatic cache. The receipt retains the original
compile fingerprint and describes the derived runtime fingerprint truthfully.
No load, device access, compilation, or product default changes occur here.
"""
from dataclasses import replace
import hashlib
import shlex
import partition_only as p

FLAG = '--xla_gpu_graph_min_graph_size'


def parse_flags(value):
    if not isinstance(value, str):
        raise ValueError('missing exact XLA_FLAGS fingerprint')
    tokens = shlex.split(value)
    found = {}
    for token in tokens:
        if not token.startswith('--') or '=' not in token:
            raise ValueError('requires explicit flag=value tokens')
        key, val = token.split('=', 1)
        if key in found:
            raise ValueError('duplicate XLA flag')
        found[key] = val
    return tokens, found


def min1_flags(donor_flags):
    tokens, values = parse_flags(donor_flags)
    if values.get(FLAG, '5') != '5':
        raise ValueError('donor runtime min must be 5')
    if FLAG in values:
        return ' '.join(FLAG+'=1' if t.startswith(FLAG+'=') else t for t in tokens)
    return (donor_flags + ' ' + FLAG+'=1').strip()


def derive_runtime_meta(meta, blob, live_flags):
    digest = hashlib.sha256(blob).hexdigest()
    if meta.blob_sha256 != digest:
        raise ValueError('donor blob SHA mismatch or missing')
    _, _, _, rest = p.unpack(blob)
    # Demand a real explicit stored min5 as an additional provenance check.
    p.rest_min(rest, 5, 5)
    donor_flags = meta.fingerprint.get('xla_flags')
    expected = min1_flags(donor_flags)
    if live_flags != expected:
        raise ValueError('runtime flag delta must be only min5 -> min1')
    fingerprint = dict(meta.fingerprint, xla_flags=live_flags)
    derived = replace(meta, fingerprint=fingerprint)
    receipt = dict(
        scope='proposed runtime-partition derivation; GPU admission pending',
        blob_sha256=digest, entire_blob_byte_identical=True,
        donor_compile_fingerprint=meta.fingerprint,
        derived_runtime_fingerprint=fingerprint,
        allowed_delta='process XLA graph_min_graph_size: 5 -> 1 only',
        loader='ordinary explicit aot_executable.load, all guards enabled',
        automatic_cache_aliasing='FORBIDDEN; original cheap_key is provenance only',
        runtime_capture_verified=False)
    return derived, receipt
