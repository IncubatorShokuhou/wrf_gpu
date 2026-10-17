#!/usr/bin/env python3
"""t-cfix-pblmix: MYJ PBL operands on real cloudy CPU-WRF columns (extends real_columns.py).

Adds to the shared real-column operands: TKE_MYJ = history TKE_PBL (real), HT = HGT,
ELFLX = history LH, SNOW, SICE = SEAICE, interface pressure PINT (PSFC at the
ground, mean of adjacent levels inside, half-step extrapolation at the top), and
deterministic Janjic surface memory (documented; plain inputs to both sides):
AKHS = KARMAN*UST/(GZ1OZ0-PSIH), AKMS = KARMAN*UST/(GZ1OZ0-PSIM), THZ0 = TSK*(1e5/PSFC)^CAPA,
QZ0 = qv1/(1+qv1), QSFC = qv1, UZ0 = VZ0 = 0, CT = 0, CHKLOWQ = 1 (Noah-MP, surface_driver.F:3429).
Usage: myj_columns.py <wrfout[,...]> <ncol per file> <out.txt> <out_inputs.json>
"""
from __future__ import annotations

import json
import sys

import netCDF4
import numpy as np

from real_columns import FIELDS2, FIELDS3, extract

KARMAN, CAPA = 0.4, 287.04 / 1004.6
EXTRA2 = ("HT", "LH", "SNOW", "SICE", "AKHS", "AKMS", "THZ0", "QZ0", "QSFC", "CHKLOWQ", "CT", "UZ0", "VZ0")


def extend(path: str, data: dict) -> dict:
    with netCDF4.Dataset(path) as d:
        v = d.variables
        g = lambda n: np.asarray(v[n][0], np.float64)  # noqa: E731
        tke, hgt, lh, snow, sice = g("TKE_PBL"), g("HGT"), g("LH"), g("SNOW"), g("SEAICE")
    s, c = data["scalars"], data["columns"]
    data["columns"]["TKE"], data["columns"]["PINT"] = [], []
    for k in EXTRA2:
        s[k] = []
    for n, (j, i) in enumerate(zip(data["j"], data["i"])):
        r = lambda x: float(np.float32(x))  # noqa: E731
        p = np.asarray(c["P"][n])
        pint = np.concatenate([[s["PSFC"][n]], 0.5 * (p[:-1] + p[1:]), [max(p[-1] - 0.5 * (p[-2] - p[-1]), 1.0)]])
        data["columns"]["TKE"].append([r(x) for x in tke[: data["kx"], j, i]])  # Registry TKE_PBL is Z-staggered; MYJ reads k=kts..kte
        data["columns"]["PINT"].append([r(x) for x in pint])
        qv1 = c["QV"][n][0]
        vals = dict(HT=hgt[j, i], LH=lh[j, i], SNOW=snow[j, i], SICE=sice[j, i],
                    AKHS=KARMAN * s["UST"][n] / (s["GZ1OZ0"][n] - s["PSIH"][n]),
                    AKMS=KARMAN * s["UST"][n] / (s["GZ1OZ0"][n] - s["PSIM"][n]),
                    THZ0=s["TSK"][n] * (1.0e5 / s["PSFC"][n]) ** CAPA, QZ0=qv1 / (1.0 + qv1), QSFC=qv1,
                    CHKLOWQ=1.0, CT=0.0, UZ0=0.0, VZ0=0.0)
        for k in EXTRA2:
            s[k].append(r(vals[k]))
    return data


def write_text(data: dict, out: str) -> None:
    n, kx = len(data["j"]), data["kx"]
    with open(out, "w") as f:
        f.write(f"{n} {kx}\n")
        for c in range(n):
            f.write(" ".join(f"{data['scalars'][k][c]:.17e}" for k in FIELDS2 + EXTRA2) + "\n")
            for k in range(kx):
                f.write(" ".join(f"{data['columns'][name][c][k]:.17e}" for name in FIELDS3 + ("TKE",)) + "\n")
            f.write(" ".join(f"{x:.17e}" for x in data["columns"]["PINT"][c]) + "\n")


if __name__ == "__main__":
    wrfouts, ncol, out_txt, out_json = sys.argv[1].split(","), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    parts = [extend(w, extract(w, ncol)) for w in wrfouts]
    cols3 = FIELDS3 + ("TKE", "PINT")
    data = dict(source=[p["source"] for p in parts], j=sum((p["j"] for p in parts), []),
                i=sum((p["i"] for p in parts), []), kx=parts[0]["kx"],
                file_index=sum(([n] * len(p["j"]) for n, p in enumerate(parts)), []),
                columns={k: sum((p["columns"][k] for p in parts), []) for k in cols3},
                scalars={k: sum((p["scalars"][k] for p in parts), []) for k in FIELDS2 + EXTRA2})
    write_text(data, out_txt)
    with open(out_json, "w") as f:
        json.dump(data, f)
    print(f"{len(data['j'])} columns, kx={data['kx']}")
