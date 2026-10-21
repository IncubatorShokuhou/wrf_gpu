"""Q-A engine check: GPU vs CPU-WRF near-surface fields on coastal masks.

usage: python qa_pairs.py OUT.json CASE_DIR[:label] ...
CASE_DIR holds gpu/, cpu/ (wrfout, same valid times) and input/wrfinput_d0N (masks).
Masks per domain: sea (LANDMASK 0), land, coast = cells within 2 cells (Chebyshev) of the land/sea edge,
coast_land/coast_sea, medano = 7x7 (d03) / 3x3 (d02) box around the El Medano pixel (28.0465N, 16.5370W).
Per lead: mean CPU, mean GPU, bias (GPU-CPU), RMSE of U10/V10/WS10/UST/CD(=(UST/WS10)^2)/PBLH/HFX/LH/T2/TSK.
day = domain-mean CPU SWDOWN > 50 W/m2. Read-only on inputs; no JAX."""
import json, sys
from pathlib import Path
import numpy as np
import netCDF4 as nc
from scipy.ndimage import binary_dilation

MEDANO = (28.0465, -16.5370)
FIELDS = ("U10", "V10", "WS10", "UST", "CD", "PBLH", "HFX", "LH", "T2", "TSK")


def masks(wrfinput, dom):
    with nc.Dataset(wrfinput) as f:
        lm = np.asarray(f["LANDMASK"][0]) > 0.5
        lat = np.asarray(f["XLAT"][0]); lon = np.asarray(f["XLONG"][0])
    st = np.ones((5, 5), bool)  # +-2 cells
    edge = binary_dilation(lm, st) & binary_dilation(~lm, st)
    m = {"all": np.ones_like(lm), "land": lm, "sea": ~lm, "coast": edge, "coast_land": edge & lm,
         "coast_sea": edge & ~lm, "interior_land": lm & ~edge, "open_sea": ~lm & ~edge}
    d = (lat - MEDANO[0]) ** 2 + ((lon - MEDANO[1]) * np.cos(np.radians(MEDANO[0]))) ** 2
    j, i = np.unravel_index(d.argmin(), d.shape)
    if np.sqrt(d.min()) * 111 < 3 * 3:  # pixel inside the domain (not just nearest edge cell)
        r = 3 if dom == "d03" else 1
        box = np.zeros_like(lm); box[max(j - r, 0):j + r + 1, max(i - r, 0):i + r + 1] = True
        m["medano"] = box
        m["medano_pt"] = np.zeros_like(lm); m["medano_pt"][j, i] = True
    return m, (int(j), int(i))


def fields(path):
    with nc.Dataset(path) as f:
        out = {k: np.asarray(f[k][0], dtype=np.float64) for k in ("U10", "V10", "UST", "PBLH", "HFX", "LH", "T2", "TSK", "SWDOWN")}
    out["WS10"] = np.hypot(out["U10"], out["V10"])
    out["CD"] = (out["UST"] / np.maximum(out["WS10"], 0.5)) ** 2
    return out


def run_case(case, label):
    case = Path(case)
    res = {}
    for dom in ("d01", "d02", "d03"):
        m, pix = masks(case / "input" / f"wrfinput_{dom}", dom)
        cpu = sorted((case / "cpu").glob(f"wrfout_{dom}_*"))
        rows = []
        for lead, cf in enumerate(cpu):
            gf = case / "gpu" / cf.name
            if lead == 0 or not gf.exists():
                continue
            c, g = fields(cf), fields(gf)
            day = bool(c["SWDOWN"].mean() > 50.0)
            row = {"lead_h": lead, "file": cf.name, "day": day}
            for mk, mm in m.items():
                row[mk] = {}
                for fld in FIELDS:
                    a, b = c[fld][mm], g[fld][mm]
                    dlt = b - a
                    row[mk][fld] = [float(a.mean()), float(b.mean()), float(dlt.mean()), float(np.sqrt((dlt ** 2).mean()))]
            rows.append(row)
        res[dom] = {"medano_pixel_jx": pix, "mask_counts": {k: int(v.sum()) for k, v in m.items()}, "rows": rows}
    return res


def main():
    out = Path(sys.argv[1])
    allres = {"doc": __doc__, "stat_order": ["cpu_mean", "gpu_mean", "bias_gpu_minus_cpu", "rmse"], "cases": {}}
    for arg in sys.argv[2:]:
        p, _, label = arg.partition(":")
        allres["cases"][label or Path(p).name] = {"dir": p, **run_case(p, label)}
    out.write_text(json.dumps(allres) + "\n")


if __name__ == "__main__":
    main()
