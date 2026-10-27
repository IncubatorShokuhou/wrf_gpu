"""Build PORT_INVENTORY.json from the W5P01 static scan, capture index and compile census (w5-portplan)."""
import json, os, re, subprocess, sys
from collections import defaultdict
LANE = "<USER_HOME>/wrf_gpu2_lanes/w5-portplan"; TREE = "<USER_HOME>/src/wrf_gpu2_wt/w5-portplan"
static = {r["file"].replace("src/gpuwrf/", ""): r for r in json.load(open(sys.argv[1]))}
fam_inv = json.load(open(f"{TREE}/proofs/s1_port/FAMILY_INVENTORY.json"))
release = {f["module"].replace("src/gpuwrf/", "") for f in fam_inv["families"]}
FAMILY = {"dyn_acoustic_fp32": "acoustic", "fused_vertical_implicit": "acoustic (legacy advance_w)", "dyn_flux_fp32": "advection",
          "dyn_momflux_fp32": "advection", "dyn_pd_fp32": "PD transport", "dyn_ring_relax_fp32": "lateral boundary",
          "dyn_boundary_fp32": "lateral boundary", "boundary_apply": "lateral boundary", "dyn_rk_fp32": "RK/EOS", "dyn_uvtend_fp32": "RK/EOS",
          "dyn_rhsph_fp32": "RK/EOS", "dyn_omega_fp32": "RK/EOS", "dyn_diff_fp32": "diffusion", "phys_mynn": "MYNN", "fused_edmf_plume": "MYNN",
          "phys_thompson": "Thompson", "phys_noahmp": "Noah-MP", "phys_gwdo": "GWDO", "phys_hydrostatic": "physics coupling",
          "phys_kf_column": "KF", "rad_": "RRTMG", "sint_kernel": "nest force-down"}
def family(mod):
    base = os.path.basename(mod)[:-3]
    for k, v in FAMILY.items():
        if base.startswith(k) or base == k: return v
    return "option scheme / diagnostics"
cap = [json.loads(l) for l in open(f"{LANE}/W5P01/rep/index.jsonl")]
if os.path.exists(f"{LANE}/W5P01/rep_opt/index.jsonl"):
    cap += [dict(json.loads(l), rep_dir="rep_opt") for l in open(f"{LANE}/W5P01/rep_opt/index.jsonl")]
comp = defaultdict(dict)
import glob
for path in sorted(glob.glob(f"{LANE}/W5P01/compile_*.jsonl")):
    if "smoke" in path: continue
    for l in open(path):
        r = json.loads(l)
        if r["arch"] not in comp[r["ttir"]] or r["status"] == "PASS": comp[r["ttir"]][r["arch"]] = r
fails = json.load(open(f"{LANE}/W5P01/all/failures.json"))
KEEP = ("status", "compile_s", "vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count", "amdhsa_scratch", "group_segment_fixed_size",
        "occupancy_waves_per_simd", "n_instr", "n_f64_instr", "n_fma_f32", "n_div_fixup_f32", "regs", "stack", "spill_st", "spill_ld",
        "n_f64_ptx", "n_div_full_f32_ptx", "error")
mods = defaultdict(lambda: {"kernels": {}})
test_local = []
for m in cap:
    if ":" not in m["site"]:
        test_local.append({"kernel": m["kernel"], "test": m["test"], "platform": m["platform"], "census": {a: {x: r[x] for x in KEEP if x in r} for a, r in comp.get(f'{m["platform"]}_{m["ttir_sha16"]}.ttir', {}).items()}})
        continue
    mod = m["site"].split(":")[0]
    k = mods[mod]["kernels"].setdefault(f'{m["kernel"]}@{m["site"].split(":")[1]}', {"kernel": m["kernel"], "site": m["site"], "grid_test_shape": m["grid"],
        "num_warps": m["num_warps"], "n_in": m["n_in"], "n_out": m["n_out"], "census": {}})
    fn = f'{m["platform"]}_{m["ttir_sha16"]}.ttir'
    k[f'ttir_{m["platform"]}'] = {"file": f"W5P01/{m.get('rep_dir', 'rep')}/{fn}", "chars": m["ttir_chars"]}
    for arch, r in comp.get(fn, {}).items():
        k["census"][arch] = {x: r[x] for x in KEEP if x in r}
for mod, s in static.items():
    if s["pallas_call"] == 0 and s["inline_asm_ptx"] == 0 and s["stock_triton_route"] == 0: continue
    d = mods[mod]
    d.update({"lines": s["lines"], "pallas_call_sites": s["pallas_call"], "release_default_path": mod in release, "family": family(mod),
              "vendor_constructs": {k: s[k] for k in ("inline_asm_ptx", "ptx_rn", "stock_triton_route", "custom_lowering", "f64", "i64", "atomic", "backend_check", "tridiag") if s[k]}})
for f in fails:
    mod = f.get("site", "?").split(":")[0]
    mods[mod].setdefault("lowering_failures", []).append({"platform": f.get("platform"), "status": f["status"], "error": f.get("error", "")[:240]})
def _neutrality():
    import re as _re
    rows = [json.loads(l) for l in open(f"{LANE}/W5P01/all/index.jsonl")]
    by = {}
    for m in rows: by.setdefault((m["site"], m["kernel"], str(m["in_avals"])), {})[m["platform"]] = m["ttir_sha16"]
    norm = lambda t: _re.sub(r'symbol = "[^"]*"', "symbol = X", _re.sub(r"loc\([^)]*\)", "", t))
    pairs = [v for v in by.values() if len(v) == 2]; same = sum(v["rocm"] == v["cuda"] for v in pairs)
    ext = sum(1 for v in pairs if v["rocm"] != v["cuda"] and norm(open(f"{LANE}/W5P01/all/rocm_{v['rocm']}.ttir").read()) == norm(open(f"{LANE}/W5P01/all/cuda_{v['cuda']}.ttir").read()))
    return {"release_invocations_with_both_platforms": len(pairs), "byte_identical_ttir": same, "differ_only_in_extern_symbols": ext, "other_differences": len(pairs) - same - ext}
base = subprocess.run(["git", "-C", TREE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
out = {"schema": "w5-portplan PORT_INVENTORY v1", "date": "2026-10-17", "base_commit": base,
       "method": "static source scan + W5P01 capture (44 S1-port family nodes, JAX 0.10 lower_jaxpr_to_triton_module rocm/cuda) + stock Triton 3.6 offline compile (hip gfx942/gfx1100, cuda sm_75/sm_61 via bundled ptxas 12.8); CPU only, no GPU",
       "labels": "census = [M CPU compile], not execution",
       "modules": dict(sorted(mods.items())),
       "test_local_kernels": test_local,
       "ttir_vendor_neutrality": _neutrality(),
       "runtime_assumptions": json.load(open(f"{LANE}/tools/runtime_assumptions.json")),
       "device_time_families_PROD_BP83_RC3": json.load(open(f"{LANE}/tools/families_bp83.json")),
       "hardware": json.load(open(f"{LANE}/tools/hardware.json")),
       "plan": json.load(open(f"{LANE}/tools/plan_meta.json"))}
json.dump(out, open(f"{LANE}/PORT_INVENTORY.json", "w"), indent=1)
print("modules", len(mods), "release", sum(1 for m in mods.values() if m.get("release_default_path")))
