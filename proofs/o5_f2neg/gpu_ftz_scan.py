"""E117 GPU-side check: do the SHIPPED GPU kernels flush f32 denormals (SASS .FTZ)?

Reads one cached stepping executable of the sealed v0.4.3 release bundle (read-only), extracts its CUDA
cubins with the repo's E41 parser helpers (scripts/wave_cubin_gate.py), disassembles with cuobjdump -sass and
counts f32 arithmetic/compare instructions with and without the .FTZ modifier, overall and for kernels whose
name matches a pattern (default: the sixth-order diffusion Pallas kernel).  CPU only, no GPU.
usage: gpu_ftz_scan.py <jax_cache_entry> <scratch_dir> [name_regex]
"""
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import zstandard  # noqa: E402

import wave_cubin_gate as W  # noqa: E402
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm  # noqa: E402

entry, scratch = Path(sys.argv[1]), Path(sys.argv[2])
pattern = re.compile(sys.argv[3] if len(sys.argv) > 3 else r"sixth", re.I)
scratch.mkdir(parents=True, exist_ok=True)
blob = zstandard.ZstdDecompressor().decompress(entry.read_bytes(), max_output_size=1 << 31)[4:]
fmt = ax._gpu_blob_format(blob)
native = W._record(blob, 2, ax, asm) if fmt == ax.GPU_THUNK_FORMAT else blob
del blob
OPS = re.compile(r"\b(FMUL|FFMA|FADD|FSETP|FMNMX|FCHK|MUFU|FSEL)(\.[A-Z0-9_.]+)?\b")
total, matched, kernels = Counter(), Counter(), []
ftz_kernels = Counter()
for i, cubin in enumerate(W._elfs(native)):
    path = scratch / f"e{i}.cubin"
    path.write_bytes(cubin)
    try:
        sass = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", str(path)], check=True,
                              capture_output=True, text=True).stdout
    finally:
        path.unlink(missing_ok=True)
    current = None
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            current = m.group(1)
            if pattern.search(current):
                kernels.append(current)
            continue
        for op, mods in OPS.findall(line):
            key = f"{op}{'.FTZ' if mods and '.FTZ' in mods else ''}"
            total[key] += 1
            if key.endswith(".FTZ"):
                ftz_kernels[current] += 1
            if current and pattern.search(current):
                matched[key] += 1
out = {"entry": entry.name, "entry_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(), "format": fmt,
       "pattern": pattern.pattern, "matched_kernels": kernels, "matched_f32_ops": dict(matched),
       "all_kernels_f32_ops": dict(total), "kernels_with_ftz": dict(ftz_kernels.most_common()),
       "ftz_fraction_all": sum(v for k, v in total.items() if k.endswith(".FTZ")) / max(1, sum(total.values()))}
print(json.dumps(out, indent=1))
