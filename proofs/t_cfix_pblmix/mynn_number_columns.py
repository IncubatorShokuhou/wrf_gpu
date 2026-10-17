#!/usr/bin/env python3
"""t-cfix-pblmix F8: MYNN number-scalar operands on real CPU-WRF mp=28 columns.

Real columns of the ORIGINAL CPU-WRF c19 run (Thompson-aerosol mp=28 + MYNN):
QNCLOUD/QNICE/QNWFA/QNIFA/QKE/QCLOUD/QICE as written by WRF, cloudy columns
picked as in real_columns.py.  Surface forcing from the same history (HFX, QFX,
UST, PBLH, TSK); derived kinematic fluxes (flt=HFX/(rho cp), flq=QFX/rho,
fltv=flt+p608*th*flq, th_sfc=TSK/exner1) are plain inputs to both sides.
REAL-rounded operands.  Usage: mynn_number_columns.py <wrfout[,...]> <ncol/file> <out.txt> <out.json>
"""
from __future__ import annotations

import json
import sys

import netCDF4
import numpy as np

CP, R_D, P608 = 7.0 * 287.0 / 2.0, 287.0, 461.6 / 287.0 - 1.0
PROF = ("u", "v", "w", "th", "tk", "p", "exner", "rho", "dz", "qv", "qc", "qi", "qke",
        "qnc", "qni", "qnwfa", "qnifa")
SCAL = ("ust", "pblh", "psfc", "xland", "flt", "flq", "flqv", "fltv", "th_sfc", "dx", "delt")


def extract(path: str, ncol: int) -> dict:
    with netCDF4.Dataset(path) as d:
        v = d.variables
        g = lambda n: np.asarray(v[n][0], np.float64)  # noqa: E731
        u3, v3, w3 = g("U"), g("V"), g("W")
        th = g("T") + 300.0
        p = g("P") + g("PB")
        ph = (g("PH") + g("PHB")) / 9.81
        qv, qc, qi = (np.maximum(g(n), 0.0) for n in ("QVAPOR", "QCLOUD", "QICE"))
        qke, qnc, qni = g("QKE"), g("QNCLOUD"), g("QNICE")
        qnwfa, qnifa = g("QNWFA"), g("QNIFA")
        ust, pblh, psfc, xland, hfx, qfx, tsk = (g(n) for n in ("UST", "PBLH", "PSFC", "XLAND", "HFX", "QFX", "TSK"))
        dx, delt = float(d.DX), float(d.DT)
    u = 0.5 * (u3[:, :, :-1] + u3[:, :, 1:])
    vv = 0.5 * (v3[:, :-1, :] + v3[:, 1:, :])
    exner = (p / 1.0e5) ** (R_D / CP)
    tk = th * exner
    rho = p / (R_D * tk * (1.0 + P608 * qv))
    dz = ph[1:] - ph[:-1]
    cond = (qc + qi)[:15]
    cloudy = (cond > 1.0e-5).any(axis=0)
    # half the columns: strongest surface heat flux among cloudy columns (active
    # DMP_mf plumes); half: largest condensate (diffusion + cloud floors).
    picks = []
    for score in (np.where(cloudy & (hfx > 0), hfx, -1.0), np.where(cloudy, cond.max(axis=0), -1.0)):
        order = np.argsort(-score.ravel(), kind="stable")
        order = [o for o in order[score.ravel()[order] > 0] if o not in picks]
        want = ncol // 2 if not picks else ncol - len(picks)
        picks += list(order[:: max(1, len(order) // max(want, 1))][:want])
    jj, ii = np.unravel_index(np.asarray(picks, int), cond.shape[1:])
    out = dict(source=path, j=jj.tolist(), i=ii.tolist(), kx=int(u.shape[0]),
               columns={k: [] for k in PROF}, scalars={k: [] for k in SCAL})
    r = lambda a: np.float32(a).astype(np.float64)  # noqa: E731
    for j, i in zip(jj, ii):
        prof = dict(u=u[:, j, i], v=vv[:, j, i], w=w3[:-1, j, i], th=th[:, j, i], tk=tk[:, j, i],
                    p=p[:, j, i], exner=exner[:, j, i], rho=rho[:, j, i], dz=dz[:, j, i], qv=qv[:, j, i],
                    qc=qc[:, j, i], qi=qi[:, j, i], qke=np.maximum(qke[:, j, i], 1.0e-4),
                    qnc=qnc[:, j, i], qni=qni[:, j, i], qnwfa=qnwfa[:, j, i], qnifa=qnifa[:, j, i])
        for k in PROF:
            out["columns"][k].append(r(prof[k]).tolist())
        rho1 = rho[0, j, i]
        flt, flq = hfx[j, i] / (rho1 * CP), qfx[j, i] / rho1
        sc = dict(ust=max(ust[j, i], 1.0e-3), pblh=pblh[j, i], psfc=psfc[j, i], xland=xland[j, i], flt=flt,
                  flq=flq, flqv=flq, fltv=flt + P608 * th[0, j, i] * flq, th_sfc=tsk[j, i] / exner[0, j, i],
                  dx=dx, delt=delt)
        for k in SCAL:
            out["scalars"][k].append(float(r(sc[k])))
    return out


def write_text(data: dict, path: str) -> None:
    n, kx = len(data["j"]), data["kx"]
    with open(path, "w") as f:
        f.write(f"{n} {kx}\n")
        for c in range(n):
            f.write(" ".join(f"{data['scalars'][k][c]:.17e}" for k in SCAL) + "\n")
            for k in range(kx):
                f.write(" ".join(f"{data['columns'][name][c][k]:.17e}" for name in PROF) + "\n")


if __name__ == "__main__":
    files, ncol, out_txt, out_json = sys.argv[1].split(","), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    parts = [extract(w, ncol) for w in files]
    data = dict(source=[p["source"] for p in parts], j=sum((p["j"] for p in parts), []),
                i=sum((p["i"] for p in parts), []), kx=parts[0]["kx"],
                columns={k: sum((p["columns"][k] for p in parts), []) for k in PROF},
                scalars={k: sum((p["scalars"][k] for p in parts), []) for k in SCAL})
    write_text(data, out_txt)
    json.dump(data, open(out_json, "w"))
    print(len(data["j"]), "columns")
