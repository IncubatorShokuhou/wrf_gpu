"""Structural dump of the serialized runtime thunk tree (field-id skeleton only, no IR)."""
import os, sys, importlib.util, json, zstandard
from collections import Counter
from pathlib import Path
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
ROOT = Path('<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf'); sys.path.insert(0, str(ROOT / 'src'))
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/wave_cubin_gate.py')
g = importlib.util.module_from_spec(spec); sys.modules['gate'] = g; spec.loader.exec_module(g)
def fields(d):
    try: return list(ax._proto_fields(d))
    except Exception: return []
def name_of(d):
    fs = fields(d)
    info = next((v for f, w, v in fs if f == 1 and w == 2), None)
    if info is None: return None
    leaf = next((v for f, w, v in fields(info) if f == 1 and w == 2), None)
    try: return leaf.decode() if leaf else None
    except Exception: return None
def skel(d, depth, maxd, out, path):
    fs = fields(d)
    ids = [f for f, w, v in fs if f != 1]
    nm = name_of(d)
    key = tuple(sorted(set(ids)))
    out.append((depth, path, nm, key, len(d)))
    if depth >= maxd: return
    for i, (f, w, v) in enumerate(fs):
        if f == 1 or w != 2: continue
        # descend only into messages (not strings)
        try:
            s = v.decode()
            if s.isprintable(): continue
        except Exception: pass
        skel(v, depth + 1, maxd, out, path + (f,))
for p in sys.argv[1:]:
    blob = zstandard.ZstdDecompressor().decompress(Path(p).read_bytes(), max_output_size=1 << 31)[4:]
    payload = g._gpu_payload(blob, ax, asm)
    _, info, pos = asm._chunks(payload)[-1]
    data, _ = asm._read_logical(payload, pos, info['data_size'])
    rest, = asm._decode_simple_chunk(data, 1)
    top = Counter(f for f, w, v in fields(rest))
    print('==', Path(p).name[:60], 'rest fields', dict(top))
    runtime = [v for f, w, v in fields(rest) if f == 13 and w == 2]
    for i, r in enumerate(runtime):
        out = []
        skel(r, 0, 3, out, ())
        d0 = out[0]
        print(f' top[{i}] name={d0[2]} ids={d0[3]} bytes={d0[4]}')
        cnt = Counter((o[0], o[1], o[3]) for o in out[1:])
        for (dep, path, key), c in sorted(cnt.items())[:25]:
            print('    ', dep, path, key, c)
    del blob, payload, data, rest
