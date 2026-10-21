"""CT01: REAL convective columns of the #148 CPU-reference frames -> Thompson mp_gt_driver inputs (caller-faithful wiring
as proofs/thompson/drizzle_oracle/build_columns.py: th = T + 300 (history dry theta, checked vs THM), p = P + PB,
pii = (p/1e5)**(R_d/cp), w = W[:-1] (bottom face, MPT :1224), dz8w = diff((PH+PHB)/9.81)). Kinds (column-wise):
core (max qr >= 2 g/kg), graupel (max qg >= 1 g/kg), updraft (max W >= 5 m/s), downdraft (min W <= -3 m/s with
qr > 1e-4 in the lowest 5 levels), snow (max qs >= 1 g/kg); plus the max-qr / max-qg / max-W column of each frame.
usage: build_columns.py <frames_dir> <out.npz> <out.json>"""
import glob, json, os, sys
import numpy as np
DT = {"d02": 18.0, "d04": 6.0, "d06": 6.0}
NPICK = 14
rng = np.random.default_rng(20261020)
R_D, CP = np.float32(287.0), np.float32(7.0 * 287.0 / 2.0)
KEYS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz8w")
cols, meta, thm_err = {k: [] for k in KEYS}, [], []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
    z = np.load(f)
    dom, lead = os.path.basename(f)[:3], int(z["lead"])
    g = lambda v: np.asarray(z[v], np.float32)
    qv, qc, qr, qi, qs, qg, ni, nr = (g(v) for v in ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN"))
    T, P, PB, W, PH, PHB = (g(v) for v in ("T", "P", "PB", "W", "PH", "PHB"))
    th = T + np.float32(300.0)
    thm_err.append(float(np.abs((th * (1 + np.float32(461.6 / 287.0) * qv) - 300) - g("THM")).max()))
    p = P + PB
    pii = (p / np.float32(1.0e5)) ** (R_D / CP)
    dz = np.diff((PH + PHB) / np.float32(9.81), axis=0)
    qrm, qgm, qsm, wmax, wmin = qr.max(0), qg.max(0), qs.max(0), W.max(0), W.min(0)
    kinds = {"core": qrm >= 2e-3, "graupel": qgm >= 1e-3, "updraft": wmax >= 5.0,
             "downdraft": (wmin <= -3.0) & (qr[:5].max(0) > 1e-4), "snow": qsm >= 1e-3}
    picks = []
    for kind, m in kinds.items():
        yx = np.argwhere(m)
        if len(yx):
            picks += [(kind, y, x) for y, x in yx[rng.choice(len(yx), min(NPICK, len(yx)), replace=False)]]
    for kind, a in (("maxqr", qrm), ("maxqg", qgm), ("maxw", wmax)):
        y, x = np.unravel_index(np.argmax(a), a.shape); picks.append((kind, y, x))
    for kind, y, x in picks:
        for k, a in zip(KEYS, (qv, qc, qr, qi, qs, qg, ni, nr, th, pii, p, W[:-1], dz)):
            cols[k].append(a[:, y, x])
        meta.append(dict(dom=dom, lead=lead, kind=kind, y=int(y), x=int(x), dt=DT[dom], qr_max=float(qrm[y, x]),
                         qg_max=float(qgm[y, x]), qs_max=float(qsm[y, x]), w_max=float(wmax[y, x]), w_min=float(wmin[y, x]),
                         use_theta_m=int(z["attr_USE_THETA_M"]) if "attr_USE_THETA_M" in z else None))
print("theta check |(T+300)(1+Rv/Rd qv)-300 - THM| max", max(thm_err))
names = np.array(["%s_t%02d_%s_y%d_x%d" % (m["dom"], m["lead"], m["kind"], m["y"], m["x"]) for m in meta])
np.savez_compressed(sys.argv[2], names=names, kind=np.array([m["kind"] for m in meta]),
                    **{f"in_{k}": np.stack(v).astype(np.float32) for k, v in cols.items()},
                    dt=np.array([m["dt"] for m in meta], np.float32))
json.dump(dict(columns=meta), open(sys.argv[3], "w"), indent=0)
from collections import Counter
print(len(meta), Counter((m["dom"], m["kind"]) for m in meta))
