"""Byte comparison of two wn3_score output dirs (w42-scorer identity gate, CPU only).
Every *.json / *.md result file of REF must exist in NEW with identical bytes, except two volatile values that differ
between any two runs of the same scorer: the comparator's "generated_utc" and "elapsed_seconds" (dXX.json) and the
"Generated UTC" line (dXX.md); those are masked textually before comparing bytes. Logs and NEW-only files (resources.json)
are listed, not compared. --alias REF_PATH=NEW_PATH maps a directory that legitimately differs between the two runs (the
scorer tree root that holds the frozen D6 manifest, an input dir reached through hard links) to one token on both sides;
each alias and its hit counts are reported. Prints a JSON verdict; exit 0 iff identical.
usage: python compare_outputs.py REF_DIR NEW_DIR [--alias REF_PATH=NEW_PATH ...] [--out OUT_JSON]"""
import hashlib
import json
import re
import sys
from pathlib import Path

VOLATILE = [(re.compile(rb'"generated_utc": "[^"]*"'), b'"generated_utc": "<masked>"'),
            (re.compile(rb'"elapsed_seconds": [-0-9.eE+]+'), b'"elapsed_seconds": "<masked>"'),
            (re.compile(rb"Generated UTC: `[^`]*`"), b"Generated UTC: `<masked>`")]


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("ref", type=Path)
ap.add_argument("new", type=Path)
ap.add_argument("--alias", action="append", default=[])
ap.add_argument("--out", type=Path)
args = ap.parse_args()
ref, new = args.ref, args.new
aliases = [a.split("=", 1) for a in args.alias]
alias_hits = {a: [0, 0] for a in args.alias}


def masked(data: bytes, side: int) -> tuple[bytes, int]:
    hits = 0
    for pattern, repl in VOLATILE:
        data, n = pattern.subn(repl, data)
        hits += n
    for i, (raw, pair) in enumerate(zip(args.alias, aliases)):
        token = f"<ALIAS{i}>".encode()
        before = data.count(pair[side].encode())
        data = data.replace(pair[side].encode(), token)
        alias_hits[raw][side] += before
    return data, hits


rows, ok = {}, True
for path in sorted(p for p in ref.iterdir() if p.suffix in {".json", ".md"}):
    other = new / path.name
    if not other.exists():
        rows[path.name] = "MISSING_IN_NEW"
        ok = False
        continue
    a, na = masked(path.read_bytes(), 0)
    b, nb = masked(other.read_bytes(), 1)
    same = a == b
    ok &= same and na == nb
    rows[path.name] = {"identical": same, "masked_values": [na, nb], "sha256_masked": hashlib.sha256(a).hexdigest()[:16],
                       "bytes": [path.stat().st_size, other.stat().st_size]}
extra = sorted(p.name for p in new.iterdir() if not (ref / p.name).exists())
result = {"ref": str(ref), "new": str(new), "identical": ok, "aliases": alias_hits, "files": rows, "new_only": extra}
text = json.dumps(result, indent=1)
if args.out:
    args.out.write_text(text + "\n")
print(json.dumps({"identical": ok, "compared": len(rows), "new_only": extra, "aliases": alias_hits,
                  "differ": [k for k, v in rows.items() if v == "MISSING_IN_NEW" or not v["identical"]]}))
sys.exit(0 if ok else 1)
