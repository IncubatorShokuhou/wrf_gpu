"""Census of serialized thunk kinds (oneof field id) per program; names from ThunkInfo; no IR dumped."""
import os, sys, importlib.util, re, zstandard
from collections import Counter, defaultdict
from pathlib import Path
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
ROOT = Path('<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf'); sys.path.insert(0, str(ROOT / 'src'))
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/wave_cubin_gate.py')
g = importlib.util.module_from_spec(spec); sys.modules['gate'] = g; spec.loader.exec_module(g)
def fields(d):
    try: return list(ax._proto_fields(d))
    except Exception: return []
for p, hlo in zip(sys.argv[1::2], sys.argv[2::2]):
    names = set(re.findall(r'^\s*(?:ROOT )?%([\w.\-]+) = ', Path(hlo).read_text(), re.M))
    ops = {}
    for line in Path(hlo).read_text().splitlines():
        m = re.match(r'^\s*(?:ROOT )?%([\w.\-]+) = (?:\([^=]*?\)|\S+) ([\w\-]+)\(', line)
        if m: ops[m.group(1)] = m.group(2)
    blob = zstandard.ZstdDecompressor().decompress(Path(p).read_bytes(), max_output_size=1 << 31)[4:]
    payload = g._gpu_payload(blob, ax, asm)
    _, info, pos = asm._chunks(payload)[-1]
    data, _ = asm._read_logical(payload, pos, info['data_size'])
    rest, = asm._decode_simple_chunk(data, 1)
    kinds = Counter(); examples = defaultdict(list)
    def walk(d, depth=0):
        if depth > 60: return
        fs = fields(d)
        if not fs: return
        first = next((v for f, w, v in fs if f == 1 and w == 2), None)
        name = None
        if first is not None:
            leaf = next((v for f, w, v in fields(first) if f == 1 and w == 2), None)
            try: name = leaf.decode() if leaf is not None else None
            except Exception: name = None
        if name in names:
            impl = tuple(sorted({f for f, w, v in fs if f != 1}))
            sub = []
            for f, w, v in fs:
                if f != 1 and w == 2:
                    sub.append((f, tuple(sorted({ff for ff, ww, vv in fields(v)}))))
            impl = impl + tuple(sub) if impl and impl[0] in (13, 6, 7) else impl
            kinds[impl] += 1
            if len(examples[impl]) < 4: examples[impl].append((name, ops.get(name)))
        for f, w, v in fs:
            if w != 2 or f == 1 and name in names: continue
            try:
                if v.decode().isprintable(): continue
            except Exception: pass
            walk(v, depth + 1)
    for f, w, v in fields(rest):
        if f == 13 and w == 2: walk(v)
    print('==', Path(p).name[:50])
    for k, c in kinds.most_common(): print('   impl', k, c, examples[k])
    del blob, payload, data, rest
