import os, sys, importlib.util, zstandard
from pathlib import Path
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
ROOT = Path('<DATA_ROOT>/wrf_gpu2/worktrees/o5-perf'); sys.path.insert(0, str(ROOT / 'src'))
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/wave_cubin_gate.py')
g = importlib.util.module_from_spec(spec); sys.modules['gate'] = g; spec.loader.exec_module(g)
F = lambda d: list(ax._proto_fields(d))
blob = zstandard.ZstdDecompressor().decompress(Path(sys.argv[1]).read_bytes(), max_output_size=1 << 31)[4:]
gpu = g._gpu_payload(blob, ax, asm)
_, info, pos = asm._chunks(gpu)[-1]; data, _ = asm._read_logical(gpu, pos, info['data_size']); (gpu,) = asm._decode_simple_chunk(data, 1)
hwc = next(v for f, w, v in F(gpu) if f == 1); cfg = next(v for f, w, v in F(hwc) if f == 2); dbg = next(v for f, w, v in F(cfg) if f == 14)
vals = {}
for f, w, v in F(dbg): vals.setdefault(f, []).append(v)
for k in map(int, sys.argv[2:]): print(k, vals.get(k))
