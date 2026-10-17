#!/usr/bin/env python3
"""t-cfix-pblmix: merge real-column inputs + pristine driver dumps into one JSON fixture.

Usage: make_fixture.py <build_dir> <out.json> <label=relpath> [<label=relpath> ...]
Each dump has 'PBL c pblh kpbl' and 'TEN c k RU RV RTH RQV RQC RQI' lines.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TEN = ("RUBLTEN", "RVBLTEN", "RTHBLTEN", "RQVBLTEN", "RQCBLTEN", "RQIBLTEN")


def parse(path: Path, ncol: int, kx: int) -> dict:
    out = {k: [[0.0] * kx for _ in range(ncol)] for k in TEN}
    out["PBLH"], out["KPBL"] = [0.0] * ncol, [0] * ncol
    for line in path.read_text().splitlines():
        w = line.split()
        if w[0] == "PBL":
            c = int(w[1]) - 1
            out["PBLH"][c], out["KPBL"][c] = float(w[2]), int(w[3])
        elif w[0] == "UST":  # INOUT surface stress rewrite (GFS bl_gfs.F:464)
            out.setdefault("UST", [0.0] * ncol)[int(w[1]) - 1] = float(w[2])
        elif w[0] == "TEN":
            c, k = int(w[1]) - 1, int(w[2]) - 1
            for name, val in zip(TEN, w[3:]):
                out[name][c][k] = float(val)
    return out


def main() -> int:
    build, dest = Path(sys.argv[1]), Path(sys.argv[2])
    inputs = json.loads((build / "inputs.json").read_text())
    ncol, kx = len(inputs["j"]), inputs["kx"]
    fixture = {"inputs": inputs, "outputs": {},
               "sources_sha256": (build / "sources.sha256").read_text().split("\n")}
    for spec in sys.argv[3:]:
        label, rel = spec.split("=", 1)
        dump = build / rel
        fixture["outputs"][label] = parse(dump, ncol, kx)
        fixture["outputs"][label]["dump_sha256"] = hashlib.sha256(dump.read_bytes()).hexdigest()
    dest.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(dest, dest.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
