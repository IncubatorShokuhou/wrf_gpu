"""Proof-only byte-preservation experiment for stored AOT DebugOptions208.

Stored208 alone does NOT select the reload partition; see runtime_meta.py.
No executable is compiled or loaded. The only allowed semantic delta is stored
DebugOptions208 (5 -> 1). Native Riegeli chunks and every other rest field are
checked byte for byte. Use private variant stores; never rewrite a donor cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fields(value):
    fs = asm._exact_fields(value, 'partition-only')
    if asm._repack(fs) != value:
        raise ValueError('noncanonical protobuf; would rewrite unrelated bytes')
    return fs


def replace_child(value, number, transform):
    fs = fields(value)
    indexes = [i for i, (f, w, _) in enumerate(fs) if f == number and w == 2]
    if len(indexes) != 1:
        raise ValueError(f'expected one length-delimited field {number}')
    i = indexes[0]
    f, w, child = fs[i]
    fs[i] = f, w, transform(child)
    return asm._repack(fs)


def rest_min(rest, expected, new):
    def debug(value):
        fs = fields(value)
        indexes = [i for i, (f, _, _) in enumerate(fs) if f == 208]
        if len(indexes) != 1:
            raise ValueError('expected exactly one explicit DebugOptions208')
        i = indexes[0]
        if fs[i] != (208, 0, expected):
            raise ValueError(f'DebugOptions208 is not varint {expected}')
        fs[i] = 208, 0, new
        return asm._repack(fs)

    return replace_child(rest, 1, lambda wc: replace_child(
        wc, 2, lambda cfg: replace_child(cfg, 14, debug)))


def unpack(blob):
    if ax._gpu_blob_format(blob) != ax.GPU_THUNK_FORMAT:
        raise ValueError('requires compiled-thunk Riegeli AOT')
    size, pos = asm._read_varint(blob, 0)
    head = blob[:pos + size]
    env = fields(blob[len(head):])
    if not env or env[0][:2] != (1, 2) or sum(f == 1 for f, _, _ in env) != 1:
        raise ValueError('unsupported IFRT envelope')
    riegeli = env[0][2]
    begin, info, data_pos = asm._chunks(riegeli)[-1]
    if info['type'] != ord('r') or info['num_records'] != 1:
        raise ValueError('expected final simple one-record rest chunk')
    data, _ = asm._read_logical(riegeli, data_pos, info['data_size'])
    (rest,) = asm._decode_simple_chunk(data, 1)
    return head, env, riegeli[:begin], rest


def verify_delta(before, after):
    bh, be, bn, br = unpack(before)
    ah, ae, an, ar = unpack(after)
    if bh != ah or be[1:] != ae[1:] or bn != an:
        raise ValueError('IFRT/native record chunks changed')
    if rest_min(ar, 1, 5) != br:
        raise ValueError('rest differs beyond DebugOptions208')
    if br == ar:
        raise ValueError('no partition change')
    return dict(donor_sha256=sha(before), derived_sha256=sha(after),
                native_chunks_sha256=sha(bn), native_chunks_bytes=len(bn),
                rest_sha256_before=sha(br), rest_sha256_after=sha(ar),
                only_delta='HloModuleConfig.DebugOptions208: 5 -> 1',
                all_native_chunks_byte_identical=True,
                all_other_rest_bytes_identical=True)


def derive(blob):
    head, env, native, rest = unpack(blob)
    changed = rest_min(rest, 5, 1)
    out = head + asm._repack([(1, 2, asm._append_simple_chunk(native, [changed]))] + env[1:])
    verify_delta(blob, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('donor', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    blob = a.donor.read_bytes()
    changed = derive(blob)
    receipt = verify_delta(blob, changed)
    receipt.update(scope='CPU bytes proof only; GPU capture/execution unmeasured',
                   donor=str(a.donor.resolve()), derived=str(a.out.resolve()))
    # Exclusive creation prevents mutation of either a donor or an existing arm.
    with a.out.open('xb') as f:
        f.write(changed)
    with a.receipt.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
