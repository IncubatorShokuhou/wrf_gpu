"""Diff the stored DebugOptions (HloModuleConfig field 14) of two executables; field ids + values only."""
import os, sys, importlib.util, zstandard
from pathlib import Path
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
ROOT = Path('<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf'); sys.path.insert(0, str(ROOT / 'src'))
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/wave_cubin_gate.py')
g = importlib.util.module_from_spec(spec); sys.modules['gate'] = g; spec.loader.exec_module(g)
F = lambda d: list(ax._proto_fields(d))
def debug_options(path):
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
    out = {}
    for f, w, v in F(dbg):
        out.setdefault(f, []).append(v if not isinstance(v, bytes) else (v[:80] if v.isascii() else v.hex()[:80]))
    cfg_fields = {f: (v if not isinstance(v, bytes) else len(v)) for f, w, v in F(cfg) if f != 14}
    return out, cfg_fields
a, ca = debug_options(sys.argv[1]); b, cb = debug_options(sys.argv[2])
print('debug fields', len(a), len(b))
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k): print('  DIFF field', k, 'A=', str(a.get(k))[:200], 'B=', str(b.get(k))[:200])
for k in sorted(set(ca) | set(cb)):
    if ca.get(k) != cb.get(k): print('  CFG DIFF field', k, 'A=', str(ca.get(k))[:120], 'B=', str(cb.get(k))[:120])
# print command-buffer-like repeated enum fields (small ints repeated)
for k in sorted(a):
    if len(a[k]) > 1 and all(isinstance(x, int) and x < 20 for x in a[k]): print('  repeated small-int field', k, a[k], b.get(k))
for k in sorted(a):
    if isinstance(a[k][0], bytes) and len(a[k]) == 1 and k in b and isinstance(b[k][0], bytes): pass
print('packed candidates', {k: a[k] for k in a if isinstance(a[k][0], (bytes, str)) and len(str(a[k][0])) < 40})
