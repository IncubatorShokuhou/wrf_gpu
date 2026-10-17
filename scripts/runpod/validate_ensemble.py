#!/usr/bin/env python3
"""Check completed ensemble members and identical, nonempty output sets after timing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def validate(root: Path, expected: int = 4, min_frames: int = 6) -> dict:
    errors, members = [], {}
    paths = sorted(p for p in root.glob("member*") if p.is_dir())
    names = {f"member{i:02d}" for i in range(1, expected + 1)}
    if {p.name for p in paths} != names:
        errors.append(f"expected members {sorted(names)}, found {[p.name for p in paths]}")
    for path in paths:
        try:
            receipt = json.loads((path / "receipt.json").read_text())
            if receipt.get("rc") != 0:
                errors.append(f"{path.name}: incomplete/failed receipt rc={receipt.get('rc')}")
            frames = sorted((path / "wrfout").glob("wrfout_d01_*"))
            if len(frames) < min_frames or any(p.stat().st_size == 0 for p in frames):
                errors.append(f"{path.name}: missing/empty history frames ({len(frames)} < {min_frames})")
            hashes = {}
            for frame in frames:
                with frame.open("rb") as stream:
                    hashes[frame.name] = hashlib.file_digest(stream, "sha256").hexdigest()
            members[path.name] = {"gpu": receipt.get("gpu", {}).get("index"), "rc": receipt.get("rc"),
                                  "wall_s": receipt.get("wall_s"), "frames": hashes}
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"{path.name}: {exc}")
    sets = [m["frames"] for m in members.values()]
    if sets and any(frames != sets[0] for frames in sets[1:]):
        errors.append("member history filename sets or literal file hashes differ")
    return {"verdict": "PASS" if members and not errors else "FAIL", "members": members, "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--expected", type=int, default=4)
    ap.add_argument("--min-frames", type=int, default=6)
    a = ap.parse_args()
    report = validate(a.root, a.expected, a.min_frames)
    print(json.dumps(report, indent=1))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
