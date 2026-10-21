"""Verify SHA-256 of a sample of read wrfouts against the ADD admission list (one tau24 frame per domain and side)."""
import glob, hashlib, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gread
listed = gread.use_add()
S = "<DATA_ROOT>/alisios/scratch"
paths = []
for k in range(1, 10):
    for pat in (f"{S}/n148-cpuref-20261018_12z/cpu_reference_a2_a3/wrfout_d0{k}_2026-10-19_18:00:*", f"{S}/n148-gpu-v042/runs/warm72/wrfout/wrfout_d0{k}_2026-10-19_18:00:*"):
        m = glob.glob(pat)
        if m:
            paths.append(os.path.realpath(m[0]))
sess, rdir = gread.session(paths, sys.argv[1])
res = {}
with sess:
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        res[p] = {"sha256": h.hexdigest(), "admitted_sha256": listed[p], "match": h.hexdigest() == listed[p]}
gread.close_receipt(sess, rdir)
json.dump(res, open(HERE / "receipts" / "sha_sample.json", "w"), indent=1)
print(len(res), "files, matches:", sum(v["match"] for v in res.values()))
