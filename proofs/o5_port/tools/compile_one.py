"""Compile ONE captured TTIR for ONE target with stock Triton 3.6 on the CPU (A2-route proxy). One JSON line.

usage: compile_one.py TTIR {cuda|hip} ARCH NUM_WARPS NUM_STAGES
cuda: ARCH = 70|75|80|86|89|90|120; Triton-bundled ptxas (12.8) for every arch, plus the jax-cuda13
      ptxas 13.2 (XLA's assembler, B48) for arch >= 75 as a second opinion on the same PTX.
hip:  ARCH = gfx942|gfx90a|gfx950|gfx1100|gfx1201 (wave64 for CDNA, wave32 for RDNA).
No device is touched; this is code-generation evidence, not hardware qualification.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time

path, backend, arch, nw, ns = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
_cache = tempfile.mkdtemp(prefix="o5tc_", dir=os.environ.get("O5_TRITON_TMP", "<DATA_ROOT>/wrf_gpu2/scratch/o5-port"))
os.environ["TRITON_CACHE_DIR"] = _cache  # private per compile, deleted at exit (never ~/.triton on /)
import shutil  # noqa: E402
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402

CU13 = "<USER_HOME>/miniconda3/lib/python3.13/site-packages/nvidia/cu13/bin/ptxas"
out = {"ttir": os.path.basename(path), "backend": backend, "arch": arch, "num_warps": nw, "num_stages": ns}
t0 = time.time()


def ptxas_stats(exe, ptx, sm):
    with tempfile.NamedTemporaryFile(suffix=".ptx", mode="w") as f:
        f.write(ptx)
        f.flush()
        r = subprocess.run([exe, "-v", f"-arch={sm}", f.name, "-o", os.devnull],
                           capture_output=True, text=True, timeout=1800)
    txt = r.stdout + r.stderr
    res = {"rc": r.returncode}
    m = re.search(r"Used (\d+) registers", txt)
    res["regs"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", txt)
    if m:
        res["stack"], res["spill_st"], res["spill_ld"] = map(int, m.groups())
    m = re.search(r"(\d+) bytes smem", txt)
    res["smem_static"] = int(m.group(1)) if m else 0
    if r.returncode:
        res["error"] = txt[-400:]
    return res


try:
    if backend == "hip":
        wave = 32 if arch.startswith("gfx1") else 64
        tgt = GPUTarget("hip", arch, wave)
    else:
        tgt = GPUTarget("cuda", int(arch), 32)
    k = triton.compile(path, target=tgt, options={"num_warps": nw, "num_stages": ns})
    out["compile_s"] = round(time.time() - t0, 2)
    out["status"] = "PASS"
    out["shared_bytes"] = getattr(k.metadata, "shared", None)
    if backend == "hip":
        a = k.asm["amdgcn"]
        out["wavefront"] = wave

        def grab(key):
            m = re.search(r"\.%s:\s+(\d+)" % key, a)
            return int(m.group(1)) if m else None
        for key in ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count", "agpr_count",
                    "private_segment_fixed_size", "group_segment_fixed_size"):
            out[key] = grab(key)
        m = re.search(r";\s*Occupancy:\s*(\d+)", a)
        out["occupancy_waves_per_simd"] = int(m.group(1)) if m else None
        ins = [ln.strip().split()[0] for ln in a.splitlines() if re.match(r"\s+[vsgbd][a-z_0-9]+", ln)]
        out["n_instr"] = len(ins)
        out["n_f64_instr"] = sum(1 for i in ins if i.endswith("_f64"))
        out["n_fma_f32"] = sum(1 for i in ins if re.match(r"v_(pk_)?fma[ckma]*_f32|v_dual_fmac", i))
        out["n_scratch_ops"] = sum(1 for i in ins if i.startswith("scratch_"))
        m = re.search(r"\.amdhsa_float_denorm_mode_32\s+(\d+)", a)
        out["denorm_mode_32"] = int(m.group(1)) if m else None
        out["hsaco_bytes"] = len(k.asm.get("hsaco", b""))
    else:
        ptx = k.asm["ptx"]
        out["n_f64_ptx"] = len(re.findall(r"\.f64\b", ptx))
        out["n_rn_asm_ptx"] = len(re.findall(r"\b(?:add|sub|mul|div)\.rn\.f32\b", ptx))
        sm = f"sm_{arch}a" if int(arch) >= 90 else f"sm_{arch}"
        bundled = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas")
        if int(arch) >= 100:
            bundled += "-blackwell"
        out["ptxas_bundled"] = ptxas_stats(bundled, ptx, sm)
        if int(arch) >= 75 and os.path.exists(CU13):
            out["ptxas_cu13"] = ptxas_stats(CU13, ptx, sm)
        b = out["ptxas_bundled"]
        out.update(regs=b.get("regs"), stack=b.get("stack"), spill_st=b.get("spill_st"), spill_ld=b.get("spill_ld"))
        if b["rc"]:
            out["status"] = "PTXAS_FAIL"
except Exception as e:  # noqa: BLE001 - a compile failure is a result
    out["status"] = "FAIL"
    out["compile_s"] = round(time.time() - t0, 2)
    out["error"] = f"{type(e).__name__}: {str(e)[-600:]}"
print(json.dumps(out), flush=True)
shutil.rmtree(_cache, ignore_errors=True)
os._exit(0)
