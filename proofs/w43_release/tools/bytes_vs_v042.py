"""C3/C4 of RELEASE_PLAN.md: wrfout whole-file SHA of every v0.4.3 arm vs the v0.4.2 bundle receipt of the same arm.

usage: python3 bytes_vs_v042.py V043_RUNS_DIR V042_RECEIPTS_DIR OUT_JSON [RW3T_RUNS_DIR]
Arm pairs: each phase-1 arm <name> vs P42/receipts/<name>/wrfout_sha.txt; RW3T N72 (v0.4.3 warm 72 h) vs P42 A3 and vs
v0.4.3 A3 (RW1); O72 (v0.4.2 bundle replay in this session) vs P42 A3 (v0.4.2 determinism control). PASS per pair = same
file-name set (non-empty) and every SHA equal. Missing arm = MISSING (never PASS).
"""
import json, sys
from pathlib import Path

runs, ref = Path(sys.argv[1]), Path(sys.argv[2]); out = Path(sys.argv[3])
rw = Path(sys.argv[4]) if len(sys.argv) > 4 else None


def shas(p):
    try:
        return dict(l.split()[::-1] for l in p.read_text().splitlines() if l.strip())
    except OSError:
        return None


ARMS = ["A1_swiss", "A2_prod", "A4_wn3_0227_icloud0", "A3_wn3_0227_72h", "B2_wn3_0502_3h", "B4_wn3_0227in48h_3h",
        "B5_swiss_3h", "B6_prod_3h", "B8_wn3_0502_icloud0_3h", "B3_wn3_0614_48h"]
pairs = [(f"{n} v0.4.3 vs v0.4.2", runs / n / "wrfout_sha.txt", ref / n / "wrfout_sha.txt") for n in ARMS]
if rw:
    pairs += [("N72 v0.4.3 warm vs v0.4.2 A3", rw / "N72/wrfout_sha.txt", ref / "A3_wn3_0227_72h/wrfout_sha.txt"),
              ("N72 v0.4.3 warm vs v0.4.3 A3 (RW1)", rw / "N72/wrfout_sha.txt", runs / "A3_wn3_0227_72h/wrfout_sha.txt"),
              ("O72 v0.4.2 replay vs v0.4.2 A3 (control)", rw / "O72/wrfout_sha.txt", ref / "A3_wn3_0227_72h/wrfout_sha.txt")]
res, ok = {"pairs": {}}, True
for name, pa, pb in pairs:
    sa, sb = shas(pa), shas(pb)
    if not sa or not sb:
        res["pairs"][name] = {"verdict": "MISSING", "a": str(pa), "b": str(pb)}; ok = False; continue
    diff = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
    v = "PASS" if not diff and set(sa) == set(sb) else "FAIL"
    res["pairs"][name] = {"verdict": v, "files": len(sa), "files_ref": len(sb), "differing": diff[:50],
                          "n_differing": len(diff), "a": str(pa), "b": str(pb)}
    ok = ok and v == "PASS"
res["verdict"] = "PASS" if ok else "FAIL_OR_MISSING"
out.write_text(json.dumps(res, indent=1) + "\n")
print("BYTES_VS_V042", res["verdict"], {k: (v["verdict"], v.get("files")) for k, v in res["pairs"].items()})
