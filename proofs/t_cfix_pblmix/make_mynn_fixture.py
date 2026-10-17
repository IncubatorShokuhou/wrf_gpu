#!/usr/bin/env python3
"""t-cfix-pblmix F8: inputs.json + WRF MYNN number-oracle dump (capture build) -> JSON fixture."""
import hashlib
import json
import sys
from pathlib import Path

MF = ("s_aw", "s_awqc", "s_awqnc", "s_awqni", "s_awqnwfa", "s_awqnifa", "s_awqv")
TEN = ("dfh", "sqc", "sqi", "Dqnc", "Dqni", "Dqnwfa", "Dqnifa")
CAP = ("sqc2", "sqi2", "qnc2", "qni2", "qnwfa2", "qnifa2")

build, dest = Path(sys.argv[1]), Path(sys.argv[2])
inputs = json.loads((build / "inputs.json").read_text())
n, kx = len(inputs["j"]), inputs["kx"]
wrf = {k: [[0.0] * (kx + 1) for _ in range(n)] for k in MF}
wrf.update({k: [[0.0] * kx for _ in range(n)] for k in TEN + CAP})
dump = build / "capture" / "out_capture.txt"
for line in dump.read_text().splitlines():
    w = line.split()
    if w[0] in ("MF", "TEN", "CAP"):
        c, k = int(w[1]) - 1, int(w[2]) - 1
        for name, val in zip({"MF": MF, "TEN": TEN, "CAP": CAP}[w[0]], w[3:]):
            wrf[name][c][k] = float(val)
fixture = {"inputs": inputs, "wrf": wrf, "dump_sha256": hashlib.sha256(dump.read_bytes()).hexdigest(),
           "pristine_dump_sha256": hashlib.sha256((build / "pristine" / "out_pristine.txt").read_bytes()).hexdigest(),
           "sources_sha256": (build / "sources.sha256").read_text().split("\n")}
dest.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
print(dest, dest.stat().st_size)
