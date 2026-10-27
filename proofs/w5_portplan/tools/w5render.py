"""Render PORT_PLAN.md from PORT_PLAN.template.md + PORT_INVENTORY.json census (w5-portplan)."""
import json, subprocess, sys
LANE = "<USER_HOME>/wrf_gpu2_lanes/w5-portplan"
inv = json.load(open(f"{LANE}/PORT_INVENTORY.json"))
rel = {m: d for m, d in inv["modules"].items() if d.get("release_default_path")}
def recs(arch, mods=rel):
    out = []
    for m, d in mods.items():
        for k in d["kernels"].values():
            r = k["census"].get(arch)
            out.append((m, k["kernel"], r))
    return out
def tot(arch):
    rs = recs(arch); done = [r for _, _, r in rs if r]
    return sum(r["status"] == "PASS" for r in done), len(done), len(rs)
def cell(mod_prefixes, arch="gfx942"):
    rs = [r for m, k, r in recs(arch) if any(m.startswith(p) or ("/" + p) in m for p in mod_prefixes)]
    done = [r for r in rs if r]
    if not done: return "pending"
    p = sum(r["status"] == "PASS" for r in done)
    vg = max(r.get("vgpr_count") or 0 for r in done); oc = min(r.get("occupancy_waves_per_simd") or 99 for r in done)
    ss = sum(r.get("sgpr_spill_count") or 0 for r in done); vs = sum(r.get("vgpr_spill_count") or 0 for r in done)
    pend = len(rs) - len(done)
    return f"{p}/{len(done)} PASS{' (+'+str(pend)+' pending)' if pend else ''}, ≤ {vg} VGPR, occ ≥ {oc}, VGPR spills {vs}" + (f", SGPR spills {ss}" if ss else "")
summary = subprocess.run([sys.executable, f"{LANE}/tools/w5summary.py"], capture_output=True, text=True).stdout.strip()
g942, g1100, s75, s61 = tot("gfx942"), tot("gfx1100"), tot("75"), tot("61")
vsp = sum((r.get("vgpr_spill_count") or 0) for a in ("gfx942", "gfx1100") for _, _, r in recs(a) if r)
scr = sum((r.get("amdhsa_scratch") or 0) for a in ("gfx942", "gfx1100") for _, _, r in recs(a) if r)
pending = sorted({k for a in ("gfx942", "gfx1100", "75", "61") for _, k, r in recs(a) if not r})
C = {}
def pend(a): return sorted({k for _, k, r in recs(a) if not r})
def nonpass(a): return sorted({f"{k} {r['status']}" + (" > 900 s wall at nice 19" if r["status"] == "TIMEOUT" else "") for _, k, r in recs(a) if r and r["status"] != "PASS"})
parts = []
for a, lab, t in (("gfx942", "gfx942", g942), ("gfx1100", "gfx1100", g1100), ("75", "sm_75", s75), ("61", "sm_61", s61)):
    x = f"{lab} {t[0]}/{t[1]} PASS"
    if nonpass(a): x += " [" + "; ".join(nonpass(a)) + "]"
    if pend(a): x += f" ({len(pend(a))} large kernels not compiled: drivers stopped at FINAL)"
    parts.append(x)
def _sp(a):
    rs = [(m, k, r) for m, k, r in recs(a) if r and r["status"] == "PASS" and ((r.get("vgpr_spill_count") or 0) or (r.get("amdhsa_scratch") or 0))]
    return "none" if not rs else "; ".join(f"`{m.split('/')[-1][:-3]}` {k} {r.get('vgpr_spill_count')} VGPR spills / {r.get('amdhsa_scratch')} B scratch" for m, k, r in rs)
C["C_SUMMARY"] = ", ".join(parts) + f" of {g942[2]} sites. VGPR spills/scratch: gfx942 {_sp('gfx942')}; gfx1100 {_sp('gfx1100')}"
open(f"{LANE}/PORT_CENSUS.md", "w").write("# W5P01 compile census, per module (appendix to PORT_PLAN)\n\n" + summary + "\n")
# family-aggregated census for the plan body
famrows = {}
for m, d in rel.items():
    famrows.setdefault(d["family"], []).append(m)
order = ["acoustic", "acoustic (legacy advance_w)", "MYNN", "advection", "PD transport", "lateral boundary", "RK/EOS", "Thompson", "diffusion",
         "Noah-MP", "GWDO", "physics coupling", "nest force-down", "RRTMG", "KF"]
def fcell(mods, arch):
    rs = [k["census"].get(arch) for m in mods for k in rel[m]["kernels"].values()]
    done = [r for r in rs if r]; n = len(rs)
    if not done: return "pending"
    p = sum(r["status"] == "PASS" for r in done)
    bad = [r["status"] for r in done if r["status"] != "PASS"]
    txt = f"{p}/{n}"
    if arch in ("gfx942", "gfx1100"):
        txt += f" · ≤{max(r.get('vgpr_count') or 0 for r in done if r['status']=='PASS') if p else '-'} · {min(r.get('occupancy_waves_per_simd') or 99 for r in done if r['status']=='PASS') if p else '-'}"
    elif arch == "75":
        txt += f" · ≤{max(r.get('regs') or 0 for r in done if r['status']=='PASS') if p else '-'}"
    if bad: txt += " (" + ",".join(bad) + ")"
    if len(done) < n: txt += f" +{n-len(done)}"
    return txt
lines = ["| Family | Kernel sites | gfx942 PASS · max VGPR · min occ | gfx1100 PASS · max VGPR · min occ | gfx90a / gfx950 / gfx1201 PASS | sm_75 PASS · max regs | sm_61 PASS |", "|---|---:|---|---|---|---|---|"]
for f in order:
    mods = famrows.get(f, [])
    if not mods: continue
    n = sum(len(rel[m]["kernels"]) for m in mods)
    extra = " / ".join(fcell(mods, a).split(" ·")[0] for a in ("gfx90a", "gfx950", "gfx1201"))
    lines.append(f"| {f} | {n} | {fcell(mods,'gfx942')} | {fcell(mods,'gfx1100')} | {extra} | {fcell(mods,'75')} | {fcell(mods,'61')} |")
lines.append("")
lines.append("`+k` = k sites without a result yet: either not compiled because the drivers were stopped at FINAL, or (gfx90a/gfx950/gfx1201) TTIR larger than 400k characters, which was not attempted. Lowering failures (no TTIR, so not counted above): `boundary_apply` scalar couple and `dyn_ring_relax` (`reduce_precision`). Per-module detail with VGPR/SGPR spills and scratch: `PORT_CENSUS.md`.")
C["C_TABLE"] = "\n".join(lines)
th = cell(["kernels/phys_thompson_sedimentation", "kernels/phys_kf_column"])
C["C_GATHER"] = f"[M] the `tt.gather` kernels (Thompson sedimentation, KF) compile for gfx942: {th}"
C["C_OLDNV"] = f"sm_75 {s75[0]}/{s75[1]}, sm_61 {s61[0]}/{s61[1]} of {s75[2]} sites compiled so far, ≤ {max((r.get('regs') or 0) for _,_,r in recs('75') if r)} registers on sm_75"
import glob as _g
gi = []
for path in _g.glob(f"{LANE}/W5P01/compile_giant.jsonl"):
    for l in open(path):
        r = json.loads(l); gi.append(r)
tm = {}
for path in _g.glob(f"{LANE}/W5P01/giant_*.time"):
    try: tm[path] = json.load(open(path))
    except Exception: pass
if gi:
    r = gi[-1]; t = next(iter(tm.values()), {})
    C["C_GIANT"] = f"{r['status']}" + (f" ({r.get('compile_s')} s wall, {r.get('vgpr_count')} VGPR, occ {r.get('occupancy_waves_per_simd')})" if r['status']=='PASS' else "") + (f", CPU {t.get('cpu_user_s')} s user" if t else "")
else:
    C["C_GIANT"] = "still running at v2 time (result appended to PORT_INVENTORY.json when done)"
orc = [(m, k["kernel"], k["census"].get("gfx942")) for m, d in inv["modules"].items() if not d.get("release_default_path") for k in d["kernels"].values()]
od = [r for _, _, r in orc if r]
from collections import Counter as _C
C["C_OPT"] = (f"gfx942 {sum(r['status']=='PASS' for r in od)}/{len(orc)} PASS, " + ", ".join(f"{n} {s}" for s, n in _C(r['status'] for r in od if r['status']!='PASS').items())
              + f", {len(orc)-len(od)} not attempted/pending; PTX-asm kernels (Ferrier, QNSE, typed memory) CRASH on the PTX constraint; NTU erf LOWER_FAIL (cuda-only table); `nwp_helicity_column` AMD PassManager FAIL")
C["C_SM61"] = f"that ptxas accepts all {s61[1]} completed kernels for sm_61; the Triton runtime launch on Pascal remains [U] (one vast.ai 1080 Ti smoke in M5 decides)"
mc = cell(["kernels/rad_mcica"]); mc2 = cell(["kernels/rad_mcica"], "gfx1100")
C["C_I64"] = f"the KISS lane kernel compiles on gfx942 ({mc}) and gfx1100 ({mc2}); the stochastic RNG (option) compiles on gfx942; bitwise stream equality is a hardware gate in M4"
fam = open(f"{LANE}/tools/famtable.template.md").read()
F = {"F_AC": ["kernels/dyn_acoustic"], "F_MYNN": ["kernels/phys_mynn", "kernels/fused_edmf"], "F_ADV": ["kernels/dyn_flux", "kernels/dyn_momflux"],
     "F_PD": ["kernels/dyn_pd"], "F_BDY": ["kernels/dyn_boundary", "kernels/dyn_ring_relax", "coupling/boundary_apply"],
     "F_RK": ["kernels/dyn_rk", "kernels/dyn_uvtend", "kernels/dyn_rhsph", "kernels/dyn_omega"], "F_TH": ["kernels/phys_thompson"],
     "F_DIFF": ["kernels/dyn_diff"], "F_NOAH": ["kernels/phys_noahmp", "kernels/phys_gwdo", "kernels/phys_hydrostatic"], "F_SINT": ["nesting/sint"],
     "F_RAD": ["kernels/rad_"], "F_KF": ["kernels/phys_kf_column"]}
for key, pre in F.items():
    v = cell(pre)
    if key == "F_BDY": v += "; `boundary_apply` scalar couple + `dyn_ring_relax`: LOWER_FAIL (reduce_precision)"
    fam = fam.replace(f"<<{key}>>", v)
opt = {m: d for m, d in inv["modules"].items() if not d.get("release_default_path")}
orecs = [(m, k["kernel"], k["census"].get("gfx942")) for m, d in opt.items() for k in d["kernels"].values()]
od = [r for _, _, r in orecs if r]
fam = fam.replace("<<F_OPT>>", f"captured {len({m for m,_,_ in orecs})} of 19 modules; gfx942 {sum(r['status']=='PASS' for r in od)}/{len(od)} PASS; PTX-asm kernels (Ferrier, QNSE, typed memory) CRASH 'constraint f'; NTU erf extern LOWER_FAIL (cuda-only table); `nwp_helicity_column` AMD PassManager FAIL")
C["C_FAMILY_TABLE"] = fam
t = open(f"{LANE}/" + (sys.argv[2] if len(sys.argv) > 2 else "PORT_PLAN_v2.template.md")).read()
for k, v in C.items(): t = t.replace(f"<<{k}>>", v)
assert "<<" not in t, [l for l in t.splitlines() if "<<" in l][:3]
open(f"{LANE}/" + (sys.argv[1] if len(sys.argv) > 1 else "PORT_PLAN_v2.md"), "w").write(t)
print("rendered", len(t.split()), "words")
