"""Q-C representativeness: CPU-WRF (and GPU) d03 near-surface wind per pixel in a 7x7 box around El Medano.

usage: python qc_neigh.py OUT.json RUN_DIR:label ... (RUN_DIR has wrfout_d03_* and wrfinput_d03 in RUN_DIR or ../input)
Per pixel: LANDMASK, LU_INDEX, HGT, mean WS10, mean CD=(UST/WS10)^2 (if UST), and per-lead WS10 at the station pixel,
the nearest open-sea pixel and the box max; leads 1..N. No JAX."""
import json, sys
from pathlib import Path
import numpy as np
import netCDF4 as nc

MEDANO = (28.0465, -16.5370)
R = 3


def static(run):
    run = Path(run)
    for cand in (run / "wrfinput_d03", run.parent / "input" / "wrfinput_d03"):
        if cand.exists():
            break
    with nc.Dataset(cand) as f:
        lat = np.asarray(f["XLAT"][0]); lon = np.asarray(f["XLONG"][0])
        st = {k: np.asarray(f[k][0]) for k in ("LANDMASK", "LU_INDEX", "HGT")}
    d = (lat - MEDANO[0]) ** 2 + ((lon - MEDANO[1]) * np.cos(np.radians(MEDANO[0]))) ** 2
    j, i = np.unravel_index(d.argmin(), d.shape)
    return int(j), int(i), st, lat, lon


def main():
    out = {"doc": __doc__, "runs": {}}
    for arg in sys.argv[2:]:
        run, _, label = arg.partition(":")
        j, i, st, lat, lon = static(run)
        sl = (slice(j - R, j + R + 1), slice(i - R, i + R + 1))
        files = sorted(Path(run).glob("wrfout_d03_*"))[1:]
        ws, cd = [], []
        for f in files:
            with nc.Dataset(f) as ds:
                u = np.asarray(ds["U10"][0])[sl]; v = np.asarray(ds["V10"][0])[sl]
                w = np.hypot(u, v); ws.append(w)
                if "UST" in ds.variables:
                    cd.append((np.asarray(ds["UST"][0])[sl] / np.maximum(w, 0.5)) ** 2)
        ws = np.array(ws)
        lm = st["LANDMASK"][sl]
        dist = np.add.outer((np.arange(-R, R + 1)) ** 2, (np.arange(-R, R + 1)) ** 2)
        sea = np.where(lm < 0.5, dist, 999)
        sj, si = np.unravel_index(sea.argmin(), sea.shape)
        out["runs"][label] = {
            "dir": run, "pixel_jx": [j, i], "pixel_latlon": [float(lat[j, i]), float(lon[j, i])], "n_leads": len(files),
            "box_LANDMASK": lm.tolist(), "box_LU_INDEX": st["LU_INDEX"][sl].astype(int).tolist(),
            "box_HGT": np.round(st["HGT"][sl], 1).tolist(),
            "box_mean_WS10": np.round(ws.mean(0), 3).tolist(),
            "box_mean_CD": (np.round(np.array(cd).mean(0), 5).tolist() if cd else None),
            "nearest_sea_offset": [int(sj - R), int(si - R)],
            "series": {"station": ws[:, R, R].round(3).tolist(), "nearest_sea": ws[:, sj, si].round(3).tolist(),
                       "box_max": ws.reshape(len(ws), -1).max(1).round(3).tolist(),
                       "box_sea_mean": np.array([w[lm < 0.5].mean() for w in ws]).round(3).tolist()},
        }
    Path(sys.argv[1]).write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
