"""Extract optimized HLO text (with metadata) from a pinned JAX-cache / AOT blob.
Usage: extract_hlo.py BLOB OUT_TXT [jax_cache|aot]"""
import sys, importlib.util, hashlib
from pathlib import Path
import zstandard
spec = importlib.util.spec_from_file_location("wcg", sys.argv[4] if len(sys.argv) > 4 else "scripts/wave_cubin_gate.py")
wcg = importlib.util.module_from_spec(spec); sys.modules["wcg"] = wcg; spec.loader.exec_module(wcg)
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
import jaxlib._jax as J
blob = Path(sys.argv[1]).read_bytes()
kind = sys.argv[3] if len(sys.argv) > 3 else "jax_cache"
if kind == "jax_cache":
    blob = zstandard.ZstdDecompressor().decompress(blob, max_output_size=1 << 33)[4:]
gpu = wcg._gpu_payload(blob, ax, asm)
if ax._gpu_blob_format(blob) == ax.GPU_THUNK_FORMAT:
    begin, info, data_pos = asm._chunks(gpu)[-1]
    data, _ = asm._read_logical(gpu, data_pos, info["data_size"])
    (gpu,) = asm._decode_simple_chunk(data, info["num_records"])
hwc = next(v for f, w, v in ax._proto_fields(gpu) if f == 1)
proto = next(v for f, w, v in ax._proto_fields(hwc) if f == 1)
Path(sys.argv[2] + ".pb").write_bytes(proto)
mod = J.HloModule.from_serialized_hlo_module_proto(proto)
txt = mod.to_string()
Path(sys.argv[2]).write_text(txt)
print("proto bytes", len(proto), "text bytes", len(txt), "sha", hashlib.sha256(proto).hexdigest()[:16])
