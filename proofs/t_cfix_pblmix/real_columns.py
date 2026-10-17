#!/usr/bin/env python3
"""t-cfix-pblmix: real cloudy PBL operand columns from an ORIGINAL CPU-WRF wrfout.

Picks columns whose lowest ``KSCAN`` levels hold cloud condensate (qc+qi >
1e-5 kg/kg, the in-cloud Ri threshold of MRF/ACM2/GFS) from a CPU-WRF history
file and writes the phy_prep-style PBL operands (bottom-up, WRF mass levels) as
a plain text file the pristine Fortran drivers read (list-directed).  The same
operands go into the JSON fixture the port test consumes, so both sides see
identical inputs (independent oracle: pristine module vs port kernel, E98).

Operands: u/v = mass-point averages of the C-grid winds, th = dry theta (T+300),
p = P+PB, pi = (p/1e5)^(R/cp), t = th*pi, z/dz from (PH+PHB)/9.81 (WRF g),
qv/qc/qi = QVAPOR/QCLOUD/QICE; psfc/tsk/hfx/qfx/ust/xland from the history.
Surface-layer diagnostics that are not history fields (znt, wspd, br, psim,
psih, gz1oz0) are formed deterministically (documented below); they are plain
inputs to both sides.  No JAX import (numpy + netCDF4 only).

Usage: real_columns.py <wrfout[,wrfout...]> <ncol per file> <out.txt> <out_inputs.json>
"""
from __future__ import annotations

import json
import sys

import netCDF4
import numpy as np

G, R_D, CP = 9.81, 287.0, 7.0 * 287.0 / 2.0
ROVCP = R_D / CP
EP1 = 461.6 / 287.0 - 1.0
KSCAN = 15
FIELDS3 = ("U", "V", "T", "TH", "QV", "QC", "QI", "P", "PI", "DZ", "Z")
FIELDS2 = ("PSFC", "ZNT", "UST", "HFX", "QFX", "TSK", "XLAND", "WSPD", "BR", "PSIM", "PSIH", "GZ1OZ0")


def extract(path: str, ncol: int) -> dict:
    with netCDF4.Dataset(path) as d:
        v = d.variables
        g = lambda n: np.asarray(v[n][0], np.float64)  # noqa: E731
        u3, v3 = g("U"), g("V")
        th = g("T") + 300.0
        p = g("P") + g("PB")
        ph = (g("PH") + g("PHB")) / G
        qv, qc = np.maximum(g("QVAPOR"), 0.0), np.maximum(g("QCLOUD"), 0.0)
        qi = np.maximum(g("QICE"), 0.0) if "QICE" in v else np.zeros_like(qc)
        psfc, tsk, hfx, qfx = g("PSFC"), g("TSK"), g("HFX"), g("QFX")
        ust, xland = g("UST"), g("XLAND")
    u = 0.5 * (u3[:, :, :-1] + u3[:, :, 1:])
    vv = 0.5 * (v3[:, :-1, :] + v3[:, 1:, :])
    pi = (p / 1.0e5) ** ROVCP
    z = 0.5 * (ph[:-1] + ph[1:])
    dz = ph[1:] - ph[:-1]
    cond = (qc + qi)[:KSCAN]
    cloudy = (cond > 1.0e-5).any(axis=0)
    # deterministic, regime-diverse pick: rank by condensate amount, then take
    # every stride-th column so weak and strong clouds, land and water, both
    # qc-only and ice-bearing columns are represented.
    score = np.where(cloudy, cond.max(axis=0), -1.0)
    order = np.argsort(-score.ravel(), kind="stable")
    order = order[score.ravel()[order] > 0]
    stride = max(1, len(order) // ncol)
    pick = order[::stride][:ncol]
    jj, ii = np.unravel_index(pick, score.shape)
    cols = {k: [] for k in FIELDS3}
    sc = {k: [] for k in FIELDS2}
    for j, i in zip(jj, ii):
        prof = dict(U=u[:, j, i], V=vv[:, j, i], TH=th[:, j, i], PI=pi[:, j, i],
                    T=th[:, j, i] * pi[:, j, i], QV=qv[:, j, i], QC=qc[:, j, i], QI=qi[:, j, i],
                    P=p[:, j, i], DZ=dz[:, j, i], Z=z[:, j, i])
        for k in FIELDS3:  # REAL operands: exact in both the REAL and DOUBLE builds
            cols[k].append(np.float32(prof[k]).astype(np.float64).tolist())
        land = xland[j, i] < 1.5
        znt = 0.10 if land else 1.0e-4                          # documented constant roughness
        za = z[0, j, i] - ph[0, j, i]                           # lowest mass level AGL
        wspd = max(np.hypot(u[0, j, i], vv[0, j, i]), 0.1)      # sfclayrev WSPD floor
        thv1 = th[0, j, i] * (1.0 + EP1 * qv[0, j, i])
        thvs = tsk[j, i] / pi[0, j, i] * (1.0 + EP1 * qv[0, j, i])
        br = float(np.clip(G * za * (thv1 - thvs) / (thv1 * wspd * wspd), -5.0, 2.0))
        gz = float(np.log(max(za, znt) / znt))
        # neutral-limit psi from the bulk Ri (Businger-Dyer-like closed forms):
        zol = br * gz
        psim = -5.0 * zol if br > 0 else 2.0 * np.log((1 + np.sqrt(1 - 16 * zol)) / 2.0)
        psih = -5.0 * zol if br > 0 else 2.0 * np.log((1 + np.sqrt(1 - 16 * zol)) / 2.0)
        psim, psih = float(np.clip(psim, -0.9 * gz, 0.9 * gz)), float(np.clip(psih, -0.9 * gz, 0.9 * gz))
        vals = dict(PSFC=psfc[j, i], ZNT=znt, UST=max(ust[j, i], 1.0e-3), HFX=hfx[j, i], QFX=qfx[j, i],
                    TSK=tsk[j, i], XLAND=xland[j, i], WSPD=wspd, BR=br, PSIM=psim, PSIH=psih, GZ1OZ0=gz)
        for k in FIELDS2:
            sc[k].append(float(np.float32(vals[k])))
    return dict(source=path, j=jj.tolist(), i=ii.tolist(), kx=int(u.shape[0]),
                columns=cols, scalars=sc)


def write_text(data: dict, out: str) -> None:
    n, kx = len(data["j"]), data["kx"]
    with open(out, "w") as f:
        f.write(f"{n} {kx}\n")
        for c in range(n):
            f.write(" ".join(f"{data['scalars'][k][c]:.17e}" for k in FIELDS2) + "\n")
            for k in range(kx):
                f.write(" ".join(f"{data['columns'][name][c][k]:.17e}" for name in FIELDS3) + "\n")


if __name__ == "__main__":
    wrfouts, ncol, out_txt, out_json = sys.argv[1].split(","), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    parts = [extract(w, ncol) for w in wrfouts]
    data = dict(source=[p["source"] for p in parts], j=sum((p["j"] for p in parts), []),
                i=sum((p["i"] for p in parts), []), kx=parts[0]["kx"],
                file_index=sum(([n] * len(p["j"]) for n, p in enumerate(parts)), []),
                columns={k: sum((p["columns"][k] for p in parts), []) for k in FIELDS3},
                scalars={k: sum((p["scalars"][k] for p in parts), []) for k in FIELDS2})
    write_text(data, out_txt)
    with open(out_json, "w") as f:
        json.dump(data, f)
    print(f"{len(data['j'])} columns, kx={data['kx']}")
