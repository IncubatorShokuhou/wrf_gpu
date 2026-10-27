"""Replay of the pinned XLA b6f37ab CommandBufferConversionPass on a serialized compiled-thunk executable (CPU only).

Mirrors command_buffer_conversion_pass.cc @b6f37ab (copy in receipts/): Run() appends a CONVERTIBLE thunk to the pending
sequence without descending into it; a NON-convertible While/Conditional is descended into (body / every branch);
FlushCommandBuffer converts a pending sequence only if len >= max(1, min_graph_size). IsConvertible: kernel and D2D copy
-> FUSION; While -> WHILE enabled and every body+condition thunk convertible; Conditional -> CONDITIONAL enabled and every
branch thunk convertible. Thunk proto layout (decoded from the bundle blobs): ThunkProto f1=ThunkInfo, f8=kernel, f13=copy,
f7=while{f2=condition seq, f3=body seq, f4=trip count}, f6=conditional{f2=branch seqs}; sequences = repeated f1 ThunkProto.
Output: static thunk counts captured in command buffers vs left for host launch, per variant. Not a GPU measurement.
"""
import argparse, json, os, sys, importlib.util, zstandard
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / 'src'))
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/wave_cubin_gate.py')
g = importlib.util.module_from_spec(spec); sys.modules['gate'] = g; spec.loader.exec_module(g)
F = lambda d: list(ax._proto_fields(d))
KERNEL, COPY, WHILE, COND = 8, 13, 7, 6


class T:
    __slots__ = ('kind', 'name', 'body', 'cond', 'branches')

    def __init__(self, proto):
        fs = F(proto)
        info = next((v for f, w, v in fs if f == 1 and w == 2), b'')
        leaf = next((v for f, w, v in F(info) if f == 1 and w == 2), b'') if info else b''
        self.name = leaf.decode(errors='replace')
        kinds = [f for f, w, v in fs if f != 1 and w == 2]
        self.kind = kinds[0] if kinds else None
        self.body = self.cond = None; self.branches = []
        impl = next((v for f, w, v in fs if f == self.kind), b'')
        if self.kind == WHILE:
            sub = F(impl)
            self.cond = seq(next(v for f, w, v in sub if f == 2))
            self.body = seq(next(v for f, w, v in sub if f == 3))
        elif self.kind == COND:
            self.branches = [seq(v) for f, w, v in F(impl) if f == 2]


def seq(data):
    return [T(v) for f, w, v in F(data) if f == 1 and w == 2]


def load(path):
    raw = Path(path).read_bytes()
    blob = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 31)[4:] if path.endswith('-cache') else raw
    gpu = g._gpu_payload(blob, ax, asm)
    _, info, pos = asm._chunks(gpu)[-1]
    data, _ = asm._read_logical(gpu, pos, info['data_size'])
    (rest,) = asm._decode_simple_chunk(data, 1)
    return [T(v) for f, w, v in F(rest) if f == 13 and w == 2]


CMD_TYPES = {1: 'FUSION', 2: 'CUBLAS', 3: 'CUDNN', 4: 'COLLECTIVES', 5: 'CONDITIONAL', 6: 'WHILE', 7: 'CUSTOM_CALL',
             8: 'CUBLASLT', 9: 'DYNAMIC_SLICE_FUSION'}


def stored_cfg(path):
    """Command-buffer types (DebugOptions 258) and min graph size (208) stored in the executable's HloModuleConfig."""
    raw = Path(path).read_bytes()
    blob = zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 31)[4:] if path.endswith('-cache') else raw
    gpu = g._gpu_payload(blob, ax, asm)
    if ax._gpu_blob_format(blob) == ax.GPU_THUNK_FORMAT:
        _, info, pos = asm._chunks(gpu)[-1]
        data, _ = asm._read_logical(gpu, pos, info['data_size'])
        (gpu,) = asm._decode_simple_chunk(data, info['num_records'])
    hwc = next(v for f, w, v in F(gpu) if f == 1)
    cfg = next(v for f, w, v in F(hwc) if f == 2)
    dbg = next(v for f, w, v in F(cfg) if f == 14)
    types, size = [], 0  # proto3: absent field reads 0; the pass then uses max(1, 0)
    for f, w, v in F(dbg):
        if f == 258:
            types += [CMD_TYPES[x] for x in (v if isinstance(v, bytes) else bytes([v]))]
        elif f == 208:
            size = v
    return {'types': types, 'min': size}


def convertible(t, cfg):
    if t.kind in (KERNEL, COPY):
        return True
    if t.kind == WHILE:
        return 'WHILE' in cfg['types'] and all(convertible(x, cfg) for x in t.body + t.cond)
    if t.kind == COND:
        return 'CONDITIONAL' in cfg['types'] and all(convertible(x, cfg) for b in t.branches for x in b)
    return False


def kind_census(ts, acc=None):
    """Every sequence entry (named or not) by impl field id; unknown kinds are non-convertible in convertible()."""
    from collections import Counter
    acc = Counter() if acc is None else acc
    for t in ts:
        acc[(t.kind, bool(t.name))] += 1
        if t.kind == WHILE:
            kind_census(t.body, acc); kind_census(t.cond, acc)
        if t.kind == COND:
            for b in t.branches:
                kind_census(b, acc)
    return acc


def count(ts):
    n = 0
    for t in ts:
        n += 1 if t.kind in (KERNEL, COPY) else 0
        if t.kind == WHILE: n += count(t.body) + count(t.cond)
        if t.kind == COND: n += sum(count(b) for b in t.branches)
    return n


def run(ts, cfg, stats, depth=0):
    cur = []
    def flush():
        if not cur:
            return
        if len(cur) < max(1, cfg['min']):
            stats['host_seq_items'] += len(cur); stats['host_leaf_thunks'] += count(cur)
            stats['rejected_short'].append((depth, [t.name for t in cur][:6], count(cur)))
        else:
            stats['command_buffers'] += 1; stats['captured_leaf_thunks'] += count(cur)
            stats['buffers'].append((depth, len(cur), count(cur), [t.name for t in cur][:3]))
        cur.clear()
    for t in ts:
        if convertible(t, cfg):
            cur.append(t); continue
        if t.kind == WHILE:
            run(t.body, cfg, stats, depth + 1)
            stats['host_while'] += 1
        elif t.kind == COND:
            for b in t.branches:
                run(b, cfg, stats, depth + 1)
            stats['host_cond'] += 1
        flush()
    flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('blobs', nargs='+'); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    base = ['FUSION', 'CUBLAS', 'CUBLASLT', 'CUSTOM_CALL', 'CUDNN', 'DYNAMIC_SLICE_FUSION', 'CONDITIONAL']
    variants = {'+WHILE, min 5 (v0.3.3+)': {'types': base + ['WHILE'], 'min': 5},
                'V1 (+WHILE, min 1)': {'types': base + ['WHILE'], 'min': 1},
                'V2 (no WHILE, min 5)': {'types': base, 'min': 5}}
    out = {}
    for b in a.blobs:
        top = load(b)
        stored = stored_cfg(b)
        vs = {f"STORED ({','.join(stored['types'])}; min {stored['min']})": stored, **variants}
        name = Path(b).name[:60]
        census = kind_census(top)
        out[name] = {'top_level': [t.name for t in top], 'leaf_thunks_total': count(top),
                     'all_entries_by_kind_and_named': {f'f{k}|named={n}': c for (k, n), c in sorted(census.items(), key=str)}}
        print(name[:40], 'all entries', out[name]['all_entries_by_kind_and_named'], flush=True)
        for vn, cfg in vs.items():
            st = dict(command_buffers=0, captured_leaf_thunks=0, host_leaf_thunks=0, host_seq_items=0, host_while=0, host_cond=0,
                      rejected_short=[], buffers=[])
            run(top, cfg, st)
            st['rejected_short'] = st['rejected_short'][:12]; st['buffers'] = sorted(st['buffers'], key=lambda x: -x[2])[:8]
            out[name][vn] = st
            print(f"{name[:40]} {vn:28.28s} CBs={st['command_buffers']:4d} captured={st['captured_leaf_thunks']:5d} "
                  f"host_leaf={st['host_leaf_thunks']:5d} host_while={st['host_while']} host_cond={st['host_cond']}", flush=True)
    a.out.write_text(json.dumps({'scope': __doc__, 'programs': out}, indent=1) + '\n')


if __name__ == '__main__':
    main()
