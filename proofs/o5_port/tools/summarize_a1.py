"""Markdown summary of a1_deviceless.py outputs (lane o5-port).

usage: summarize_a1.py OUT.md CENSUS_RECEIPT.json A1_DIR...
CENSUS_RECEIPT = capture_real.py receipt of the same program (kernel names -> source module -> family).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from families import family  # noqa: E402


out, census = sys.argv[1], json.load(open(sys.argv[2]))
names = {}
for k in census["kernels"]:
    if "ttir_sha16" in k:
        names[k["name"]] = family(k["src"])
lines = ["| target | status | compile s | kernels (Pallas) | max regs/VGPR (Pallas) | STACK>0 / spill (Pallas) | max STACK | temp MB | while / cond / cmd-buf in HLO |",
         "|---|---|---:|---:|---:|---|---:|---:|---|"]
details = []
for d in sys.argv[3:]:
    summ = json.load(open(Path(d) / "summary.json"))
    vendor = summ.get("vendor", "cuda")
    for label, rec in summ["targets"].items():
        if rec.get("status") != "PASS":
            lines.append(f"| {label} | **{rec.get('status')}** | – | – | – | – | – | – | {str(rec.get('error', ''))[:120]} |")
            continue
        rows = json.load(open(Path(d) / f"{label}_kernels.json"))

        def is_pallas(r):
            base = re.sub(r"(_\d+)+$", "", r["kernel"])
            return r["kernel"] in names or base in names
        pal = [r for r in rows if is_pallas(r)]
        if vendor == "rocm":
            regs = max((r.get("vgpr") or 0 for r in pal), default=0)
            bad = [r for r in pal if r.get("vgpr_spill") or r.get("scratch")]
            badtxt = f"{len(bad)} (VGPR spill/scratch)"
            mstack = max((r.get("scratch") or 0 for r in rows), default=0)
        else:
            regs = max((r.get("regs") or 0 for r in pal), default=0)
            bad = [r for r in pal if r.get("stack") or r.get("local")]
            badtxt = f"{len(bad)}"
            mstack = max((r.get("stack") or 0 for r in rows), default=0)
        mem = rec.get("memory", {}).get("temp_size_in_bytes")
        cf = f"{rec.get('optimized_hlo_while', '?')} / {rec.get('optimized_hlo_conditional', '?')} / {rec.get('optimized_hlo_command_buffers', '?')}"
        lines.append(f"| {label} | PASS | {rec.get('compile_s')} | {len(rows)} ({len(pal)}) | {regs} | {badtxt} | {mstack} | "
                     f"{mem / 1e6 if mem else '–'} | {cf} |")
        for r in sorted(bad, key=lambda r: -(r.get("stack") or r.get("scratch") or 0))[:8]:
            details.append(f"- {label}: `{r['kernel'][:60]}` ({names.get(r['kernel'], names.get(re.sub(r'(_[0-9]+)+$', '', r['kernel']), '?'))}) "
                           + ", ".join(f"{k}={r.get(k)}" for k in ("regs", "stack", "local", "vgpr", "vgpr_spill", "sgpr_spill", "scratch")
                                       if r.get(k) is not None))
open(out, "w").write("\n".join(lines + ([""] + details if details else [])) + "\n")
print("\n".join(lines + details))
