"""Compile one captured TTIR for one target with stock Triton 3.6 on CPU (w5-portplan W5P01). Prints one JSON line."""
import json, os, re, subprocess, sys, tempfile, time
path, backend, arch, nw, ns = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
import triton
from triton.backends.compiler import GPUTarget
out = {"ttir": os.path.basename(path), "backend": backend, "arch": arch, "num_warps": nw, "num_stages": ns}
t0 = time.time()
try:
    if backend == "hip":
        tgt = GPUTarget("hip", arch, 32 if int(arch[3:-2]) >= 10 else 64)
    else:
        tgt = GPUTarget("cuda", int(arch), 32)
    k = triton.compile(path, target=tgt, options={"num_warps": nw, "num_stages": ns})
    out["compile_s"] = round(time.time() - t0, 2)
    out["status"] = "PASS"
    if backend == "hip":
        a = k.asm["amdgcn"]
        def grab(key):
            m = re.search(r"\.%s:\s+(\d+)" % key, a); return int(m.group(1)) if m else None
        for key in ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count", "private_segment_fixed_size",
                    "group_segment_fixed_size", "agpr_count", "wavefront_size"):
            out[key] = grab(key)
        m = re.search(r";\s*Occupancy:\s*(\d+)", a); out["occupancy_waves_per_simd"] = int(m.group(1)) if m else None
        ins = [l.strip().split()[0] for l in a.splitlines() if re.match(r"\s+[vsgbd][a-z_0-9]+", l)]
        out["n_instr"] = len(ins)
        out["n_f64_instr"] = sum(1 for i in ins if i.endswith("_f64"))
        out["n_fma_f32"] = sum(1 for i in ins if re.match(r"v_(fma|fmac|pk_fma)_f32|v_fmamk_f32|v_fmaak_f32|v_dual_fmac", i))
        out["n_mul_f32"] = sum(1 for i in ins if re.match(r"v_(pk_)?mul_f32|v_dual_mul_f32", i))
        out["n_scratch_ops"] = sum(1 for i in ins if i.startswith("scratch_") or i.startswith("buffer_store") and "off" in i)
        out["n_global_load"] = sum(1 for i in ins if i.startswith(("global_load", "buffer_load")))
        out["n_div_fixup_f32"] = sum(1 for i in ins if i.startswith("v_div_fixup_f32"))
        out["n_rcp_f32"] = sum(1 for i in ins if i.startswith("v_rcp_f32"))
        m = re.search(r"\.amdhsa_float_denorm_mode_32\s+(\d+)", a); out["denorm_mode_32"] = int(m.group(1)) if m else None
        m = re.search(r"\.amdhsa_next_free_vgpr\s+(\d+)", a); out["next_free_vgpr"] = int(m.group(1)) if m else None
        m = re.search(r"\.amdhsa_private_segment_fixed_size\s+(\d+)", a); out["amdhsa_scratch"] = int(m.group(1)) if m else None
        out["hsaco_bytes"] = len(k.asm.get("hsaco", b""))
    else:
        ptx = k.asm["ptx"]
        out["n_f64_ptx"] = len(re.findall(r"\.f64\b", ptx))
        out["n_div_full_f32_ptx"] = len(re.findall(r"div\.full\.f32", ptx)); out["n_div_rn_f32_ptx"] = len(re.findall(r"div\.rn\.f32", ptx))
        out["n_fma_rn_f32_ptx"] = len(re.findall(r"fma\.rn\.f32", ptx))
        ptxas = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas")
        with tempfile.NamedTemporaryFile(suffix=".ptx", mode="w") as f:
            f.write(ptx); f.flush()
            sm = f"sm_{arch}a" if int(arch) >= 90 else f"sm_{arch}"
            if int(arch) >= 100: ptxas = ptxas + "-blackwell"
            r = subprocess.run([ptxas, "-v", f"-arch={sm}", f.name, "-o", os.devnull], capture_output=True, text=True, timeout=600)
            txt = r.stdout + r.stderr
            m = re.search(r"Used (\d+) registers", txt); out["regs"] = int(m.group(1)) if m else None
            m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", txt)
            if m: out["stack"], out["spill_st"], out["spill_ld"] = map(int, m.groups())
            out["ptxas_rc"] = r.returncode
            if r.returncode: out["status"] = "PTXAS_FAIL"; out["error"] = txt[-400:]
except Exception as e:
    out["status"] = "FAIL"; out["compile_s"] = round(time.time() - t0, 2)
    out["error"] = f"{type(e).__name__}: {str(e)[-600:]}"
print(json.dumps(out), flush=True)
os._exit(0)
