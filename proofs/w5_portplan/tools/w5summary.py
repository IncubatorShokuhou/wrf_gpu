"""Summarize W5P01 census per module/family and per target (markdown + json)."""
import json, os, sys
from collections import defaultdict
LANE = "<USER_HOME>/wrf_gpu2_lanes/w5-portplan"
inv = json.load(open(f"{LANE}/PORT_INVENTORY.json"))
targets = ["gfx942", "gfx1100", "75", "61", "gfx90a", "gfx950", "gfx1201"]
fam = defaultdict(lambda: defaultdict(list))
lines = ["| Family | Module | Kernels | gfx942 PASS · max VGPR · VGPR/SGPR spills (scratch) · min occ | gfx1100 same | sm_75 PASS · max regs · spill B | sm_61 PASS | gfx90a / gfx950 / gfx1201 PASS |", "|---|---|---:|---|---|---|---|---|"]
tot = defaultdict(lambda: [0, 0])
for mod, d in inv["modules"].items():
    if not d.get("release_default_path") or not d["kernels"]: continue
    ks = list(d["kernels"].values())
    cells = []
    for t in targets:
        rs = [k["census"].get(t) for k in ks]
        done = [r for r in rs if r]
        npass = sum(r["status"] == "PASS" for r in done)
        tot[t][0] += npass; tot[t][1] += len(done)
        if t in ("gfx942", "gfx1100"):
            vg = max((r.get("vgpr_count") or 0) for r in done) if done else None
            vs = sum((r.get("vgpr_spill_count") or 0) for r in done); ss = sum((r.get("sgpr_spill_count") or 0) for r in done)
            sc = max((r.get("amdhsa_scratch") or 0) for r in done) if done else 0
            oc = min((r.get("occupancy_waves_per_simd") or 99) for r in done) if done else None
            cells.append(f"{npass}/{len(done)} · {vg} · {vs}/{ss}{' ('+str(sc)+' B)' if sc else ''} · {oc}" if done else "pending")
        elif t == "75":
            rg = max((r.get("regs") or 0) for r in done) if done else None
            sb = max((r.get("spill_st") or 0) for r in done) if done else 0
            cells.append(f"{npass}/{len(done)} · {rg} · {sb}" if done else "pending")
        elif t == "61":
            cells.append(f"{npass}/{len(done)}" if done else "pending")
        else:
            cells.append(f"{npass}/{len(done)}" if done else "–")
    cells = cells[:4] + [" / ".join(cells[4:])]
    lines.append(f"| {d['family']} | `{mod.replace('kernels/','')}` | {len(ks)} | " + " | ".join(cells) + " |")
lines.append("")
lines.append("Totals (kernel sites PASS/compiled): " + ", ".join(f"{t}: {a}/{b}" for t, (a, b) in tot.items()))
fails = [(mod, k["kernel"], t, r.get("status"), (r.get("error") or "")[:160]) for mod, d in inv["modules"].items() for k in d["kernels"].values()
         for t, r in k["census"].items() if r.get("status") != "PASS"]
if fails:
    lines.append(""); lines.append("Non-PASS: " + "; ".join(f"`{m}` {k} {t} {s}: {e}" for m, k, t, s, e in fails))
print("\n".join(lines))
