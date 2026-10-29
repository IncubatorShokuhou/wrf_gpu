"""v45-rc: delete exactly the receipted wrfout files of one arm (content-verified), nothing else.

usage: python3 -I delete_listed.py RUN_ROOT ARM_DIR [--no-verify]
Reads ARM_DIR/wrfout_sha.txt (written by arm_receipt.py); for every listed name, ARM_DIR/wrfout/<name> must be a regular
file (no symlink) inside RUN_ROOT whose current SHA-256 equals the listed one (skipped with --no-verify), then it is
unlinked; the then-empty wrfout dir is removed. Unlisted files are never touched (reported).
"""
import hashlib
import json
import sys
from pathlib import Path

root, arm = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
verify = "--no-verify" not in sys.argv[3:]
assert arm.resolve().is_relative_to(root), (arm, root)
w = arm / "wrfout"
listed = dict(l.split()[::-1] for l in (arm / "wrfout_sha.txt").read_text().splitlines() if l.strip())
deleted, mismatched, missing = [], [], []
for name, digest in sorted(listed.items()):
    p = w / name
    if not p.is_file() or p.is_symlink():
        missing.append(name)
        continue
    if verify:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            mismatched.append(name)
            continue
    p.unlink()
    deleted.append(name)
left = sorted(p.name for p in w.iterdir()) if w.is_dir() else []
if w.is_dir() and not left:
    w.rmdir()
res = dict(arm=str(arm), deleted=len(deleted), mismatched=mismatched, missing=missing, unlisted_left=left)
print(json.dumps(res))
sys.exit(1 if mismatched else 0)
