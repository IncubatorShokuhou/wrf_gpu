"""Markdown summary of compile_matrix.py JSONL results (A2 stock-Triton route). usage: summarize_a2.py OUT.md IN.jsonl..."""
import json
import sys
from collections import defaultdict

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from families import family  # noqa: E402

out = sys.argv[1]
rows = [json.loads(line) for f in sys.argv[2:] for line in open(f) if line.strip()]
targets = sorted({(r["backend"], r["arch"]) for r in rows}, key=lambda t: (t[0], str(t[1]).zfill(8)))
by_kernel = defaultdict(dict)
for r in rows:
    by_kernel[(r.get("src", "?"), r.get("name", "?"), r["ttir"])][(r["backend"], r["arch"])] = r


def cell(r):
    if r is None:
        return "–"
    if r["status"] != "PASS":
        return r["status"]
    if r["backend"] == "hip":
        s = f"{r.get('vgpr_count')}v"
        if r.get("vgpr_spill_count"):
            s += f" **{r['vgpr_spill_count']}vsp**"
        if r.get("private_segment_fixed_size"):
            s += f" **{r['private_segment_fixed_size']}B scr**"
        if r.get("sgpr_spill_count"):
            s += f" {r['sgpr_spill_count']}ssp"
        if r.get("occupancy_waves_per_simd") is not None:
            s += f" o{r['occupancy_waves_per_simd']}"
        return s
    s = f"{r.get('regs')}r"
    if r.get("spill_st") or r.get("spill_ld"):
        s += f" **{r.get('spill_st')}/{r.get('spill_ld')}B sp**"
    if r.get("stack"):
        s += f" st{r['stack']}"
    return s


lines = ["| target | kernels | PASS | FAIL | TIMEOUT | other | spill/scratch kernels | max regs (VGPR) | max compile s |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for t in targets:
    rs = [r for r in rows if (r["backend"], r["arch"]) == t]
    n = {s: sum(r["status"] == s for r in rs) for s in ("PASS", "FAIL", "TIMEOUT")}
    other = len(rs) - sum(n.values())
    if t[0] == "hip":
        sp = sum(1 for r in rs if r["status"] == "PASS" and (r.get("vgpr_spill_count") or r.get("private_segment_fixed_size")))
        mx = max((r.get("vgpr_count") or 0 for r in rs if r["status"] == "PASS"), default=0)
    else:
        sp = sum(1 for r in rs if r["status"] == "PASS" and (r.get("spill_st") or r.get("spill_ld") or r.get("stack")))
        mx = max((r.get("regs") or 0 for r in rs if r["status"] == "PASS"), default=0)
    cs = max((r.get("compile_s") or 0 for r in rs), default=0)
    lines.append(f"| {t[0]}:{t[1]} | {len(rs)} | {n['PASS']} | {n['FAIL']} | {n['TIMEOUT']} | {other} | {sp} | {mx} | {cs:.0f} |")
lines += ["", "| family | kernel (src) | TTIR chars | " + " | ".join(f"{b}:{a}" for b, a in targets) + " |",
          "|---|---|---:|" + "---|" * len(targets)]
for (src, name, ttir), cells in sorted(by_kernel.items(), key=lambda kv: (family(kv[0][0]), kv[0][0])):
    chars = next(iter(cells.values())).get("ttir_chars", "")
    lines.append(f"| {family(src)} | `{src}` {name[:28]} | {chars} | " + " | ".join(cell(cells.get(t)) for t in targets) + " |")
fails = [r for r in rows if r["status"] != "PASS"]
if fails:
    lines += ["", "Non-PASS detail:", ""]
    lines += [f"- {r['backend']}:{r['arch']} `{r.get('src')}` {r.get('name')}: {r['status']} {str(r.get('error', ''))[:200]}"
              for r in fails]
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:2 + len(targets)]))
