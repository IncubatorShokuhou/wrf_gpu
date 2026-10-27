"""Rewrite ONE DebugOptions field in a compiled-thunk GPU AOT blob's stored module config (CPU only).

Path (as aot_slim_module.slim_blob): IFRT envelope -> Riegeli -> last simple chunk 'rest' (GpuExecutableProto) -> f1
HloModuleProtoWithConfig -> f2 HloModuleConfigProto -> f14 DebugOptions -> field 208 (xla_gpu_graph_min_graph_size).
Every other byte of the rest record and every other Riegeli chunk (native records 0/1/2) is kept; the tool asserts that.
The matching .meta gets its blob_sha256 updated (the loader re-verifies it). Output mirrors P's relative paths so the
files can be placed into a bundle-consumer view. usage: patch_min_graph.py OUT_ROOT VALUE BLOB...
"""
import hashlib, json, os, pickle, sys, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / 'src'))
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/wave_cubin_gate.py')
g = importlib.util.module_from_spec(spec); sys.modules['gate'] = g; spec.loader.exec_module(g)
P = Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3')
FIELD = 208


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n: return bytes(out)


def patch(blob: bytes, value: int) -> bytes:
    size, pos = asm._read_varint(blob, 0)
    head = blob[:pos + size]
    env = asm._exact_fields(blob[len(head):], 'IFRT envelope')
    riegeli = env[0][2]
    begin, info, data_pos = asm._chunks(riegeli)[-1]
    assert info['type'] == ord('r') and info['num_records'] == 1
    data, _ = asm._read_logical(riegeli, data_pos, info['data_size'])
    (rest,) = asm._decode_simple_chunk(data, 1)
    rest_f = asm._exact_fields(rest, 'rest')
    wc = asm._exact_fields(asm._only(rest_f, 1, 'with_config'), 'with_config')
    cfg = asm._exact_fields(asm._only(wc, 2, 'config'), 'config')
    dbg = asm._exact_fields(asm._only(cfg, 14, 'debug'), 'debug')
    old = [v for f, w, v in dbg if f == FIELD]
    new_dbg = asm._repack([x for x in dbg if x[0] != FIELD] + [(FIELD, 0, value)])
    new_cfg = asm._repack([(f, w, new_dbg if f == 14 else v) for f, w, v in cfg])
    new_wc = asm._repack([(f, w, new_cfg if f == 2 else v) for f, w, v in wc])
    new_rest = asm._repack([(f, w, new_wc if f == 1 else v) for f, w, v in rest_f])
    out = head + asm._repack([(1, 2, asm._append_simple_chunk(riegeli[:begin], [new_rest]))] + env[1:])
    return out, old


def main():
    out_root, value = Path(sys.argv[1]), int(sys.argv[2])
    receipts = []
    for b in map(Path, sys.argv[3:]):
        blob = b.read_bytes()
        new, old = patch(blob, value)
        for i in (0, 1, 2):
            assert g._record(blob, i, ax, asm) == g._record(new, i, ax, asm), f'native record {i} changed'
        rel = b.relative_to(P)
        dst = out_root / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(new)
        meta_src = b.with_suffix('.meta'); meta = pickle.loads(meta_src.read_bytes())
        assert meta.blob_sha256 in (None, hashlib.sha256(blob).hexdigest()), 'meta does not describe this blob'
        import dataclasses
        meta = dataclasses.replace(meta, blob_sha256=hashlib.sha256(new).hexdigest())
        (dst.with_suffix('.meta')).write_bytes(pickle.dumps(meta))
        receipts.append({'blob': str(rel), 'orig_sha256': hashlib.sha256(blob).hexdigest(), 'new_sha256': meta.blob_sha256,
                         'field': FIELD, 'old_values': old, 'new_value': value, 'native_records_0_1_2_identical': True,
                         'bytes': [len(blob), len(new)]})
        print(json.dumps(receipts[-1]), flush=True)
    (out_root / 'PATCH_RECEIPT.json').write_text(json.dumps({'scope': __doc__, 'patched': receipts}, indent=1) + '\n')


if __name__ == '__main__':
    main()
