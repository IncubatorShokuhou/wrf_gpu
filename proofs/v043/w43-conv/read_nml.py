"""R01: guarded read of the CPU-reference namelists (A-148 base admission) -> analysis/data/docs (untracked)."""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gread
A = "<DATA_ROOT>/alisios/"
docs = json.load(open(gread.ADMISSION["path"]))["files"]
want = sorted({os.path.realpath(f["path"]) for f in docs if f["kind"] == "namelist"})
out = HERE / "data" / "docs"; out.mkdir(parents=True, exist_ok=True)
s, rdir = gread.session(want, sys.argv[1])
with s:
    for p in want:
        (out / p[len(A):].replace("/", "__")).write_bytes(open(p, "rb").read())
gread.close_receipt(s, rdir)
print("copied", want)
