"""w43-release (copy of w41-rw3add bytes_check.py; change: explicit reference SHA lists): wrfout byte identity of the
timed RW3T arms (determinism / same-work evidence, GPU-vs-GPU = regression only).

usage: python3 rw3t_bytes_check.py RUNS_DIR N_REF_SHA O_REF_SHA OUT_JSON
RUNS_DIR/<arm>/wrfout_sha.txt (arm.sh). N arms must equal each other, O arms each other; every arm's files must equal the
same-named files of its reference list: N_REF_SHA = phase-1 B1 (v0.4.3 bundle, WN3 0227 72 h) wrfout_sha.txt, O_REF_SHA =
the v0.4.2 release-day B1 (w41-bundle041 runs/B1_wn3_0227_72h/wrfout_sha.txt, archived receipts).
"""
import json
import sys
from pathlib import Path

R, NREF, OREF, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]


def sha(p):
    return dict(reversed(l.split()) for l in p.read_text().splitlines() if l.strip()) if p.exists() else {}


ref = {"N": sha(NREF), "O": sha(OREF)}
arms = {d.name: sha(d / "wrfout_sha.txt") for d in sorted(R.iterdir()) if d.is_dir() and "aborted" not in d.name}
res = {"arms": {}, "groups": {}}
for name, files in arms.items():
    k = name[0]
    r = ref[k]
    res["arms"][name] = dict(files=len(files), equal_ref=sum(files[f] == r.get(f) for f in files),
                             missing_in_ref=sorted(f for f in files if f not in r)[:5],
                             differ_ref=sorted(f for f in files if f in r and files[f] != r[f])[:5])
for k in "NO":
    g = {n: f for n, f in arms.items() if n[0] == k}
    for h in ("24", "72"):
        sel = {n: f for n, f in g.items() if (n.endswith("72")) == (h == "72")}
        if len(sel) > 1:
            first = next(iter(sel.values()))
            res["groups"][f"{k}_{h}h"] = dict(arms=sorted(sel), all_identical=all(f == first for f in sel.values()))
nf = [n for n in arms if n[0] == "N"]
of = [n for n in arms if n[0] == "O"]
if nf and of:   # N vs O differ by design (different releases); count identical files for the record
    a, b = arms[nf[0]], arms[of[0]]
    res["N_vs_O_identical_files"] = f"{sum(a[f] == b.get(f) for f in a)}/{len(a)} ({nf[0]} vs {of[0]})"
res["all_ok"] = all(v["equal_ref"] == v["files"] and v["files"] > 0 for v in res["arms"].values()) and \
    all(g["all_identical"] for g in res["groups"].values())
open(OUT, "w").write(json.dumps(res, indent=1) + "\n")
print(json.dumps({"all_ok": res["all_ok"], "groups": res["groups"], "N_vs_O": res.get("N_vs_O_identical_files"),
                  "arms": {n: f"{v['equal_ref']}/{v['files']}" for n, v in res["arms"].items()}}))
