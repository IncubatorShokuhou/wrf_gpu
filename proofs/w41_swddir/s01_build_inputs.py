"""S01: pristine RRTMG_SWRAD inputs on real Swiss CPU-WRF day columns (w41-swddir SWDDIR oracle, second region).

Source: the original Swiss CPU-WRF 4.7.1 24 h reference (release-docs RD11/cpu: d01 43x43x44, mp8 + RRTMG 4/4,
2023-01-15), read-only.  Census: day frames 09..14 UTC, per frame up to 3 columns of each regime (clear, sgs_only,
liq_thin, liq_thick, ice, snow, mixed; deterministic seed).  Operands use RE01's extract.column (WRF phy_prep wiring:
t_phy, p_phy, pi_phy, rho, dz8w, p8w/t8w, hydrostatic P/P8W = radiation_driver P/P8W).  Single domain: O3 on the
column itself (p_d01 = p_hyd, lat_d01 = xlat).  QC_BL/CLDFRA_BL are not in the CPU history -> qc_rad = qc, qi_rad = qi
(real-structured operand: the oracle and the port see the same columns).  Arms (julian, aer_opt, scale): the column's
own call and julian 196.6 (a high summer sun on the same columns), both aer_opt 0.

usage: python s01_build_inputs.py OUTDIR    (writes s01_input.bin = AE01 driver layout, s01_operands.npz, s01_arms.json)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
CPU = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
FRAMES = (9, 10, 11, 12, 13, 14)
PER_REGIME = 3
ARMS = [(-1.0, 0, 1.0), (196.6, 0, 1.0)]
G = 9.81

spec = importlib.util.spec_from_file_location("re01_extract", ROOT / "proofs/rrtmg_mp_re/extract.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)  # RE01 phy_prep column + hydrostatic helpers (no file access at import)


def regime(lwp, iwp, swp, cfmax):
    tot = lwp + iwp + swp
    if cfmax < 0.01 and tot < 1e-3:
        return "clear"
    if tot < 1.0:
        return "sgs_only" if cfmax >= 0.01 else None
    if iwp + swp < 0.2 * lwp:
        return "liq_thick" if lwp > 50.0 else "liq_thin"
    if iwp > 0.8 * tot:
        return "ice"
    if swp > 0.8 * tot:
        return "snow"
    return "mixed"


def census():
    rng = np.random.default_rng(20231015)
    picks = []
    for h in FRAMES:
        stamp = f"2023-01-15_{h:02d}:00:00"
        with Dataset(CPU / f"wrfout_d01_{stamp}") as d:
            dp = -np.diff(np.asarray(d["P"][0] + d["PB"][0], np.float64), axis=0)  # mass proxy for paths only
            dp = np.concatenate([dp, dp[-1:]], axis=0)
            path = lambda n: np.sum(np.asarray(d[n][0], np.float64) * dp / G, axis=0) * 1e3  # noqa: E731  [g m-2]
            lwp, iwp, swp = path("QCLOUD"), path("QICE"), path("QSNOW")
            cfmax = np.asarray(d["CLDFRA"][0]).max(axis=0)
            ny, nx = cfmax.shape
            by = {}
            for j in range(1, ny - 1):  # skip the outermost ring (specified boundary)
                for i in range(1, nx - 1):
                    r = regime(lwp[j, i], iwp[j, i], swp[j, i], cfmax[j, i])
                    if r is not None:
                        by.setdefault(r, []).append((j, i))
            for r in sorted(by):
                cells = by[r]
                for k in rng.choice(len(cells), min(PER_REGIME, len(cells)), replace=False):
                    j, i = cells[int(k)]
                    picks.append({"stamp": stamp, "tau": float(h), "j": int(j), "i": int(i), "category": r,
                                  "lwp": float(lwp[j, i]), "iwp": float(iwp[j, i]), "swp": float(swp[j, i])})
    return picks


def f(a):
    return np.asfortranarray(np.asarray(a)).tobytes(order="F")


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    picks = census()
    cols = []
    for pk in picks:
        with Dataset(CPU / f"wrfout_d01_{pk['stamp']}") as c:
            col = X.column(c, pk["j"], pk["i"])
            julday, gmt = int(c.getncattr("JULDAY")), float(c.getncattr("GMT"))
            col.update(p_top=float(np.asarray(c["P_TOP"][0])), julday=julday, gmt=gmt,
                       cpu_swdown=float(np.asarray(c["SWDOWN"][0, pk["j"], pk["i"]])))
        col.update(qc_rad=col["qc"].copy(), qi_rad=col["qi"].copy(), p_d01=col["p_hyd"].copy(), lat_d01=col["xlat"],
                   julian=(col["julday"] - 1) + (col["gmt"] + pk["tau"]) / 24.0, tau=pk["tau"])
        cols.append(col)
    assert len({(c["p_top"], c["gmt"]) for c in cols}) == 1
    ncol, nz = len(cols), cols[0]["p"].size
    stack = lambda k, dt="<f4": np.asarray([c[k] for c in cols], dt)  # noqa: E731
    with (out / "s01_input.bin").open("wb") as h:
        h.write(np.asarray([ncol, nz], "<i4").tobytes())
        h.write(np.asarray([cols[0]["p_top"], cols[0]["gmt"]], "<f4").tobytes())
        h.write(stack("julian").tobytes()); h.write(stack("julday", "<i4").tobytes())
        for k in ("tsk", "emiss", "albedo", "coszen", "xland", "xlat", "xlong", "snow", "xice", "lat_d01"):
            h.write(stack(k).tobytes())
        for k in ("t", "p", "pi", "th", "rho", "qv", "qc", "qi", "ni", "qs", "qr", "qg", "qc_rad", "qi_rad", "cldfra",
                  "p_hyd", "p_d01"):
            h.write(f(stack(k)))
        for k in ("dz8w", "p_hyd_w", "t8w"):
            h.write(f(stack(k)))
        h.write(stack("tau").tobytes())
        h.write(np.asarray([len(ARMS)], "<i4").tobytes())
        h.write(np.asarray([a[0] for a in ARMS], "<f4").tobytes())
        h.write(np.asarray([a[1] for a in ARMS], "<i4").tobytes())
        h.write(np.asarray([a[2] for a in ARMS], "<f4").tobytes())
    ops = {f"in_{k}": stack(k) for k in ("t", "p", "pi", "th", "rho", "qv", "qc", "qi", "ni", "qs", "qr", "qg", "qc_rad",
                                         "qi_rad", "cldfra", "p_hyd", "p_d01", "dz8w", "p_hyd_w", "t8w", "tau", "julian",
                                         "tsk", "emiss", "albedo", "xland", "xlat", "xlong", "snow", "xice", "lat_d01")}
    ops.update(in_julday=stack("julday", "<i4"), in_coszen_hist=stack("coszen"), in_p_top=np.float32(cols[0]["p_top"]),
               in_gmt=np.float32(cols[0]["gmt"]), cpu_swdown=stack("cpu_swdown"),
               category=np.asarray([p["category"] for p in picks]), ji=np.asarray([[p["j"], p["i"]] for p in picks], np.int32),
               stamp=np.asarray([p["stamp"] for p in picks]))
    np.savez(out / "s01_operands.npz", **ops)
    (out / "s01_arms.json").write_text(json.dumps({"arms": ARMS, "source": str(CPU), "picks": picks}, indent=1) + "\n")
    cats = {c: int(sum(p["category"] == c for p in picks)) for c in sorted({p["category"] for p in picks})}
    print("ncol", ncol, "nz", nz, cats)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
