"""w41-rw3add: wrfout byte identity of the timed arms (determinism / same-work evidence, GPU-vs-GPU = regression only).

usage: python3 bytes_check.py RUNS_DIR REF_DIR OUT_JSON
RUNS_DIR/<arm>/wrfout_sha.txt (arm.sh). N arms must equal each other, O arms each other; every arm's files must equal the
same-named files of the bundle041 72 h runs (REF_DIR/{B1_wn3_0227_72h (v0.4.1), B7_v040_wn3_0227_72h (v0.4.0)}/wrfout_sha.txt).
"""
import json
import sys
from pathlib import Path

R, REF, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def sha(p):
    return dict(reversed(l.split()) for l in p.read_text().splitlines() if l.strip()) if p.exists() else {}


ref = {"N": sha(REF / "B1_wn3_0227_72h/wrfout_sha.txt"), "O": sha(REF / "B7_v040_wn3_0227_72h/wrfout_sha.txt")}
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
