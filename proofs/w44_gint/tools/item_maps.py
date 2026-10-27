"""Per G-INT item cell anatomy: reference (CPU-WRF) vs one or more candidate runs (GPU members / CPU twin), CPU only, values
only as summary numbers (no field dumps).

usage: JAX_PLATFORMS=cpu python3 item_maps.py OUT_JSON REF_DIR NAME=CAND_DIR [NAME=CAND_DIR ...] -- ITEM [ITEM ...]
  ITEM = dXX:FIELD:YYYY-MM-DD_HH  (e.g. d01:QICE:2026-05-03_07)
For each item and each lead offset -3..+3 h: ref/cand field max, cells>0 (3-D: columns with any level > 0), and at the item
hour: flagged columns (presence mismatch, as the annex), their edge-distance histogram, level range, and the overlap of
ref-active columns with each candidate's active columns. For ice-family fields also the max ice supersaturation
ssati = qv/rsif - 1 at the ref-active cells on both sides (WRF Thompson deposition-nucleation threshold ssati >= 0.25,
module_mp_thompson.F:2620) using T = (T+300)*(p/1e5)^(R/cp), p = P+PB, WRF RSIF verbatim.
All operands checked finite (E200).
"""
import glob, json, sys
from datetime import datetime, timedelta

import numpy as np
from netCDF4 import Dataset

ICE = {"QICE", "QNICE", "QSNOW", "QGRAUP"}


def rsif(p, t):
    """WRF Thompson RSIF verbatim (module_mp_thompson.F REAL FUNCTION RSIF), p Pa, t K -> kg/kg (evaluated in float64)."""
    c = (.609868993e03, .499320233e02, .184672631e01, .402737184e-1, .565392987e-3, .521693933e-5, .307839583e-7,
         .105785160e-9, .161444444e-12)
    x = np.maximum(-80.0, t - 273.16)
    esi = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (c[6] + x * (c[7] + x * c[8])))))))
    esi = np.minimum(esi, p * 0.15)
    return 0.622 * esi / np.maximum(1.0e-4, p - esi)


def frame(d, dom, t):
    f = [x for x in glob.glob(f"{d}/wrfout_{dom}_{t}:*") if not x.endswith(".tmp")]
    return f[0] if len(f) == 1 else None


def read(path, names):
    with Dataset(path) as ds:
        out = {n: np.asarray(ds[n][0], dtype=np.float64) for n in names}
    for n, a in out.items():
        assert np.isfinite(a).all(), (path, n)
    return out


def colmax(a):
    return np.abs(a).max(0) if a.ndim == 3 else np.abs(a)


def edge_dist(shape):
    j, i = np.indices(shape)
    return np.minimum.reduce([i, j, shape[1] - 1 - i, shape[0] - 1 - j])


def main():
    out = sys.argv[1]
    k = sys.argv.index("--")
    ref = sys.argv[2]
    cands = dict(a.split("=", 1) for a in sys.argv[3:k])
    items = sys.argv[k + 1:]
    res = {"ref": ref, "cands": cands, "items": {}}
    for it in items:
        dom, fld, t = it.split(":")
        t0 = datetime.strptime(t, "%Y-%m-%d_%H")
        rec = {"series": {}, "at_item": {}}
        for off in range(-3, 4):
            tt = (t0 + timedelta(hours=off)).strftime("%Y-%m-%d_%H")
            row = {}
            for nm, d in [("ref", ref), *cands.items()]:
                f = frame(d, dom, tt)
                if f is None:
                    row[nm] = None
                    continue
                c = colmax(read(f, [fld])[fld])
                row[nm] = {"max": float(c.max()), "cols_gt0": int((c > 0).sum())}
            rec["series"][f"{off:+d}h"] = row
        fr = frame(ref, dom, t)
        need = [fld] + (["T", "P", "PB", "QVAPOR"] if fld in ICE else [])
        r = read(fr, need)
        rc = colmax(r[fld])
        ed = edge_dist(rc.shape)
        lev = np.nonzero(np.abs(r[fld]).max(axis=(1, 2)) > 0)[0] if r[fld].ndim == 3 else []
        base = {"ref_cols_gt0": int((rc > 0).sum()), "ref_max": float(rc.max()),
                "ref_edge_dist_hist": np.bincount(ed[rc > 0].ravel(), minlength=12)[:12].tolist(),
                "ref_edge_dist_ge12": int((ed[rc > 0] >= 12).sum()),
                "ref_levels": [int(lev.min()), int(lev.max())] if len(lev) else None}
        if fld in ICE:
            def ssati(x):
                tk = (x["T"] + 300.0) * ((x["P"] + x["PB"]) / 1.0e5) ** (287.0 / 1004.5)
                return x["QVAPOR"] / rsif(x["P"] + x["PB"], tk) - 1.0
            act = np.abs(r[fld]) > 0
            si = ssati(r)
            base["ref_ssati_at_ref_active"] = {"max": float(si[act].max()) if act.any() else None,
                                               "p90": float(np.percentile(si[act], 90)) if act.any() else None}
        rec["at_item"]["ref"] = base
        for nm, d in cands.items():
            f = frame(d, dom, t)
            if f is None:
                rec["at_item"][nm] = None
                continue
            g = read(f, need)
            gc = colmax(g[fld])
            flag = (rc != 0) ^ (gc != 0)
            row = {"cand_cols_gt0": int((gc > 0).sum()), "cand_max": float(gc.max()), "flagged_cols": int(flag.sum()),
                   "cand_exact0_at_flagged": bool((gc[flag] == 0).all()) if flag.any() else None,
                   "ref_active_also_cand_active": int(((rc > 0) & (gc > 0)).sum())}
            if fld in ICE and (np.abs(r[fld]) > 0).any():
                act = np.abs(r[fld]) > 0
                tk = (g["T"] + 300.0) * ((g["P"] + g["PB"]) / 1.0e5) ** (287.0 / 1004.5)
                si = g["QVAPOR"] / rsif(g["P"] + g["PB"], tk) - 1.0
                row["cand_ssati_at_ref_active"] = {"max": float(si[act].max()), "p90": float(np.percentile(si[act], 90))}
            rec["at_item"][nm] = row
        res["items"][it] = rec
        print(it, json.dumps(rec["at_item"])[:600], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
