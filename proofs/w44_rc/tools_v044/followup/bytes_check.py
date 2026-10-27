"""w44-rc v0.4.4: wrfout whole-file SHA comparisons from SHA lists (sha256sum format). Stdlib, reads files only.

usage: python3 -I bytes_check.py OUT_JSON LABEL=MODE:LIST_A:LIST_B [...]
  MODE full   = same file-name set (non-empty) and every SHA equal
  MODE prefix = every file of LIST_B (the shorter run) is in LIST_A with the same SHA (run-length independence)
Missing list = MISSING (never PASS).
"""
import json
import sys
from pathlib import Path


def shas(p):
    try:
        return dict(l.split()[::-1] for l in Path(p).read_text().splitlines() if l.strip())
    except OSError:
        return None


res, ok = {"pairs": {}}, True
for spec in sys.argv[2:]:
    label, rest = spec.split("=", 1)
    mode, a, b = rest.split(":", 2)
    sa, sb = shas(a), shas(b)
    if not sa or not sb:
        res["pairs"][label] = dict(verdict="MISSING", a=a, b=b); ok = False; continue
    keys = set(sa) | set(sb) if mode == "full" else set(sb)
    diff = sorted(k for k in keys if sa.get(k) != sb.get(k))
    v = "PASS" if not diff and (mode != "full" or set(sa) == set(sb)) else "FAIL"
    res["pairs"][label] = dict(verdict=v, mode=mode, files_a=len(sa), files_b=len(sb), n_differing=len(diff),
                               first_differing=diff[:20], a=a, b=b)
    ok = ok and v == "PASS"
res["verdict"] = "PASS" if ok else "FAIL_OR_MISSING"
Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
print("BYTES", res["verdict"], {k: (v["verdict"], v.get("files_b")) for k, v in res["pairs"].items()})
