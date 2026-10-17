"""Parse the SLUCM oracle text dump into a compressed npz savepoint.

Keys: ``in_<name>`` (nreg,), ``<NAME>`` (nreg, nsteps+1) for state dumped at
step 0..nsteps (outputs only from step 1; step-0 slot NaN), ``regime_names``,
``DZR/DZB/DZG`` (4,), ``precision``.  Values are parsed from ES24.16E3 text,
which round-trips the oracle REAL exactly (17 significant digits).
"""

from __future__ import annotations

import re
import sys

import numpy as np

STEP = re.compile(r"^([A-Za-z0-9_]+)\[(\d+)\]\[(\d+)\]=(.*)$")
INP = re.compile(r"^IN_([A-Za-z0-9_]+)\[(\d+)\]=(.*)$")
GLOB = re.compile(r"^(DZ[RBG])\[(\d+)\]=(.*)$")


def main(src, dst):
    names = {}
    step_vals = {}
    in_vals = {}
    dz = {"DZR": {}, "DZB": {}, "DZG": {}}
    prec = ""
    nreg = 0
    for line in open(src):
        line = line.strip()
        if line.startswith("PRECISION="):
            prec = line.split("=", 1)[1]
            continue
        if line.startswith("NREG="):
            nreg = int(line.split("=", 1)[1])
            continue
        if line.startswith("REGIME_NAME["):
            r = int(line[len("REGIME_NAME["):line.index("]")])
            names[r] = line.split("=", 1)[1]
            continue
        m = GLOB.match(line)
        if m:
            dz[m.group(1)][int(m.group(2))] = float(m.group(3))
            continue
        m = INP.match(line)
        if m:
            in_vals.setdefault(m.group(1), {})[int(m.group(2))] = float(m.group(3))
            continue
        m = STEP.match(line)
        if m:
            step_vals.setdefault(m.group(1), {})[(int(m.group(2)), int(m.group(3)))] = float(m.group(4))
    if not nreg or len(names) != nreg:
        raise SystemExit(f"incomplete dump: nreg={nreg} names={len(names)}")
    nsteps = max(s for d in step_vals.values() for (_, s) in d)
    out = {"precision": np.array(prec), "regime_names": np.array([names[r] for r in range(1, nreg + 1)])}
    for k, d in dz.items():
        out[k] = np.array([d[i] for i in range(1, 5)])
    for k, d in in_vals.items():
        out["in_" + k] = np.array([d[r] for r in range(1, nreg + 1)])
    for k, d in step_vals.items():
        a = np.full((nreg, nsteps + 1), np.nan)
        for (r, s), v in d.items():
            a[r - 1, s] = v
        out[k] = a
    np.savez_compressed(dst, **out)
    print(f"{dst}: nreg={nreg} nsteps={nsteps} fields={len(step_vals)} inputs={len(in_vals)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
