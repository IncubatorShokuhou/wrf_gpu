import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gread
A = "<DATA_ROOT>/alisios/"
docs = json.load(open(gread.ADMISSION["path"]))["files"]
import os
lex = [f["path"] for f in docs if f["kind"] in ("namelist", "gpu_run_config", "fidelity_domain_json", "fidelity_rainnc_json", "manifest", "dir_manifest")]
alias = {p: os.path.realpath(p) for p in lex}
want = sorted(set(alias.values()))
(Path(__file__).resolve().parent / "receipts" / "R02_alias.json").write_text(json.dumps(alias, indent=1))
out = Path(__file__).resolve().parent / "data" / "docs"
s, rdir = gread.session(want, "R02_docs")
with s:
    for p in want:
        rel = p[len(A):].replace("/", "__")
        with open(p, "rb") as fi:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / rel).write_bytes(fi.read())
gread.close_receipt(s, rdir)
print("copied", len(want))
