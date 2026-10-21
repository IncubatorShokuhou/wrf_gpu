"""C0b: per-program cheap-key components, v0.4.2 bundle source vs v0.4.3 pin (`gpuwrf aot-key --json` reports of the same
case pairs). PASS = same program set; per program every component equal except fn_identity and source_fingerprint, and
those two are explained by CLOSURE_DIFF.json (closure differs only in __init__.py = the version line; fn_identity equal
after reverting the version string).

Also C0b'': v043pre_* (main 218e27ea28, pre-bump) records must equal v042_* exactly (same cheap key + components).

usage: python3 aotkey_compare.py AOTKEY_DIR CLOSURE_DIFF_JSON OUT_JSON
"""
import json, sys
from pathlib import Path

d, cl, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
EXPLAINED = {"fn_identity", "source_fingerprint", "program_key"}   # program_key = digest over the other components (aot_cheap_key.py program_key)
explained_ok = (cl["differing_files"] == ["__init__.py"] and not cl["only_a"] and not cl["only_b"]
                and cl["fn_identity_equal_after_version_revert"] and not cl["cli_or_host_suspend_in_closure"])
res, ok = {"closure_explains_version_only": explained_ok, "pairs": {}}, explained_ok
for f42 in sorted(d.glob("v042_*.json")):
    f43 = d / f42.name.replace("v042_", "v043_", 1)
    a, b = json.loads(f42.read_text()), json.loads(f43.read_text()) if f43.exists() else None
    if b is None:
        res["pairs"][f42.stem[5:]] = "MISSING"; ok = False; continue
    rows = {}
    for label in ("case", "compare"):
        pa, pb = a[label]["programs"], b[label]["programs"]
        for prog in sorted(set(pa) | set(pb)):
            ca, cb = (pa.get(prog) or {}).get("components"), (pb.get(prog) or {}).get("components")
            if not ca or not cb:
                rows[f"{label}:{prog}"] = {"verdict": "MISSING"}; ok = False; continue
            diff = sorted(c for c in set(ca) | set(cb) if ca.get(c) != cb.get(c))
            v = "PASS" if set(diff) <= EXPLAINED else "FAIL"
            rows[f"{label}:{prog}"] = {"verdict": v, "differing_components": diff,
                                       "cheap_key": (pa[prog].get("cheap_key"), pb[prog].get("cheap_key"))}
            ok = ok and v == "PASS"
    res["pairs"][f42.stem[5:]] = rows
# C0b'': main 218e27ea28 (= v0.4.2 src + cli.py/host_suspend, version still 0.4.2) must give cheap keys IDENTICAL to v0.4.2
strict = {}
for f42 in sorted(d.glob("v042_*.json")):
    fp = d / f42.name.replace("v042_", "v043pre_", 1)
    if not fp.exists():
        strict[f42.stem[5:]] = "MISSING"; continue
    a, b = json.loads(f42.read_text()), json.loads(fp.read_text())
    strict[f42.stem[5:]] = {f"{lab}:{prog}": ("PASS" if a[lab]["programs"][prog].get("cheap_key")
                                              and a[lab]["programs"][prog] == b[lab]["programs"].get(prog) else "FAIL")
                            for lab in ("case", "compare") for prog in sorted(set(a[lab]["programs"]) | set(b[lab]["programs"]))
                            if prog in a[lab]["programs"]}
res["pre_bump_keys_equal_v042"] = strict
pre_ok = bool(strict) and all(isinstance(v, dict) and all(x == "PASS" for x in v.values()) for v in strict.values())
res["pre_bump_verdict"] = "PASS" if pre_ok else "FAIL_OR_MISSING"
ok = ok and pre_ok
res["verdict"] = "PASS" if ok else "FAIL_OR_MISSING"
out.write_text(json.dumps(res, indent=1) + "\n")
print("AOTKEY_COMPONENTS", res["verdict"], "pre_bump", res["pre_bump_verdict"], {p: ({k: (v["verdict"], v.get("differing_components")) for k, v in r.items()}
                                               if isinstance(r, dict) else r) for p, r in res["pairs"].items()})
