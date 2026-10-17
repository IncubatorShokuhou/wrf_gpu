#!/usr/bin/env python3
"""t-cfix-pblmix F17: inputs.json + GBMPBL oracle dumps (with RTHRATEN, and the same columns
with RTHRATEN=0) -> JSON fixture.  Usage: make_gbm_fixture.py <build_dir> <out.json>"""
import hashlib
import json
import sys
from pathlib import Path


def parse(path, ncol, kx):
    out = {}
    for line in Path(path).read_text().splitlines():
        w = line.split()
        if w[0] == "pbl":
            c = int(w[1]) - 1
            for key, val in (("pblh", float(w[2])), ("kpbl", int(w[3])), ("ust", float(w[4]))):
                out.setdefault(key, [0] * ncol)[c] = val
        else:
            c, k = int(w[1]) - 1, int(w[2]) - 1
            out.setdefault(w[0], [[0.0] * kx for _ in range(ncol)])[c][k] = float(w[3])
    return out


build, dest = Path(sys.argv[1]), Path(sys.argv[2])
inputs = json.loads((build / "inputs.json").read_text())
n, kx = len(inputs["j"]), inputs["kx"]
fixture = {"inputs": inputs,
           "rad": parse(build / "out_zero.txt", n, kx), "norad": parse(build / "norad" / "out.txt", n, kx),
           "rad_r8": parse(build / "r8" / "out.txt", n, kx),
           "build": json.loads((build / "build_zero.json").read_text()),
           "norad_out_sha256": hashlib.sha256((build / "norad" / "out.txt").read_bytes()).hexdigest()}
dest.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
print(dest, dest.stat().st_size)
