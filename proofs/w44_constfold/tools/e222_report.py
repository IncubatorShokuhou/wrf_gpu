"""Summarise the E222 production-convention receipts (o5-port off_identity_probe) for w44-constfold.

For each case: verdicts, canon/raw shas for base / cand OFF / cand ON, and which Pallas kernels (by name) change
TTIR between base and cand ON (location-free body sha).  usage: python e222_report.py E222_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
out = {}
for case in ("swiss", "prod", "wn3"):
    rec = {}
    probes = {}
    for arm in ("base", "off", "on"):
        p = d / f"{arm}_{case}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        probes[arm] = r
        rec[arm] = {k: r.get(k) for k in ("verdict", "canon_sha256", "raw_hlo_sha256", "triton_custom_calls",
                                          "n_kernel_bodies", "n_pallas_lowerings", "operator_env")}
        rec[arm]["interpreted"] = r.get("native_pallas", {}).get("interpreted")
    for other in ("off", "on"):
        c = d / f"cmp_base_{other}_{case}.json"
        if c.exists():
            cmp = json.loads(c.read_text())
            rec[f"cmp_base_{other}"] = {"verdict": cmp.get("verdict"),
                                        "checks": {k: v.get("equal") for k, v in cmp.get("checks", {}).items()}}
    if "base" in probes and "on" in probes:
        def by_name(r):
            m = {}
            for k in r["kernels"]:
                m.setdefault(k["name"], set()).add((k["ttir_sha16"], k["num_warps"], k["num_stages"], k["calls"]))
            return m
        b, o = by_name(probes["base"]), by_name(probes["on"])
        rec["kernels_changed_on"] = sorted(n for n in set(b) | set(o) if b.get(n) != o.get(n))
        rec["kernels_total"] = len(set(b) | set(o))
    out[case] = rec
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
for case, rec in out.items():
    print(case, {a: (rec[a]["verdict"], (rec[a]["canon_sha256"] or "")[:12], (rec[a]["raw_hlo_sha256"] or "")[:12],
                     rec[a]["n_kernel_bodies"], rec[a]["n_pallas_lowerings"]) for a in ("base", "off", "on") if a in rec})
    for k in ("cmp_base_off", "cmp_base_on"):
        if k in rec:
            print("  ", k, rec[k]["verdict"], {c: e for c, e in rec[k]["checks"].items() if not e})
    print("   kernels changed ON:", rec.get("kernels_changed_on"), "of", rec.get("kernels_total"))
