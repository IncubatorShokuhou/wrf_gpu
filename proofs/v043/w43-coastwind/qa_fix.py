"""Q-A corrections (critic VERDICT FIX ce84cae260 R1): pooled space-time statistics, extrema with place/time, the 0227
d03 coastal-sea maximum, and a CPU-vs-CPU twin control at the same leads.

usage: python qa_fix.py OUT.json
Pairs: F20 v0.4.0 GPU vs original CPU-WRF (WN3 0227/0502/0614, d02+d03, leads 1-24, identical inputs).
Control: w43-twin0227 CPU-WRF 4x1-decomposition twin (read-only, <DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1/cpu) vs
the same original CPU-WRF reference (2x2), 0227 d03/d02, every lead present.
Masks as qa_pairs.py (coast = within 2 cells of the land/sea edge); points sea (19,55) = ALISIOS T1 point-set pixel,
urban (20,55). Statistics on WS10 = |(U10,V10)|: mean bias, POOLED space-time RMSE (sqrt of the mean over all cells and
leads), mean of hourly spatial RMSE (the statistic reported in the first draft, labelled as such), max |diff| with lead,
cell, day/night. For the largest coastal-sea difference: lat/lon, HGT, LANDMASK, local RAINNC/hourly rain, W max,
spatial extent of |diff| > 1 m/s. No JAX."""
import json, sys
from pathlib import Path
import numpy as np
import netCDF4 as nc
from scipy.ndimage import binary_dilation, label

F20 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016")
TWIN = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1/cpu")


def masks(case, dom):
    with nc.Dataset(F20 / case / "input" / f"wrfinput_{dom}") as f:
        lm = np.asarray(f["LANDMASK"][0]) > 0.5
        hgt = np.asarray(f["HGT"][0]); lat = np.asarray(f["XLAT"][0]); lon = np.asarray(f["XLONG"][0])
    st = np.ones((5, 5), bool)
    edge = binary_dilation(lm, st) & binary_dilation(~lm, st)
    m = {"all": np.ones_like(lm), "open_sea": ~lm & ~edge, "coast_sea": edge & ~lm, "coast_land": edge & lm, "interior_land": lm & ~edge}
    if dom == "d03":
        box = np.zeros_like(lm); box[17:24, 52:59] = True
        m["medano_box"] = box
        for nm, (j, i) in (("sea_19_55", (19, 55)), ("urban_20_55", (20, 55))):
            p = np.zeros_like(lm); p[j, i] = True; m[nm] = p
    return m, dict(hgt=hgt, lat=lat, lon=lon, lm=lm)


def fields(path):
    with nc.Dataset(path) as f:
        u = np.asarray(f["U10"][0], float); v = np.asarray(f["V10"][0], float)
        out = dict(ws=np.hypot(u, v), sw=float(np.asarray(f["SWDOWN"][0]).mean()), rain=np.asarray(f["RAINNC"][0], float),
                   wmax=np.asarray(f["W"][0], float).max(0))
    return out


def pair_stats(pairs, m):
    out = {}
    for mk, mm in m.items():
        d_all, rms_h, mx = [], [], (0.0, None)
        for lead, a, b in pairs:
            d = (b["ws"] - a["ws"])[mm]
            d_all.append(d); rms_h.append(np.sqrt((d ** 2).mean()))
            k = np.abs(d).argmax()
            if abs(d[k]) > mx[0]:
                jj, ii = np.argwhere(mm)[k]
                mx = (float(abs(d[k])), dict(lead=lead, j=int(jj), i=int(ii), day=bool(a["sw"] > 50.0)))
        dd = np.concatenate(d_all)
        out[mk] = dict(n=int(dd.size), bias=float(dd.mean()), rmse_pooled=float(np.sqrt((dd ** 2).mean())),
                       rmse_mean_hourly=float(np.mean(rms_h)), maxabs=mx[0], maxabs_at=mx[1])
    return out


def load_pairs(dir_a, dir_b, dom, leads=None):
    pairs = []
    fa = sorted(Path(dir_a).glob(f"wrfout_{dom}_*"))
    for lead, f in enumerate(fa):
        g = Path(dir_b) / f.name
        if lead == 0 or not g.exists() or (leads and lead not in leads):
            continue
        pairs.append((lead, fields(f), fields(g)))
    return pairs


def event(case, dom, lead, j, i, ref_dir, alt_dir, st):
    name = sorted(Path(ref_dir).glob(f"wrfout_{dom}_*"))[lead].name
    prev = sorted(Path(ref_dir).glob(f"wrfout_{dom}_*"))[lead - 1].name
    a, b = fields(Path(ref_dir) / name), fields(Path(alt_dir) / name)
    ap, bp = fields(Path(ref_dir) / prev), fields(Path(alt_dir) / prev)
    d = b["ws"] - a["ws"]
    lab, n = label(np.abs(d) > 1.0)
    comp = lab == lab[j, i] if lab[j, i] else np.zeros_like(lab, bool)
    r = 5
    sl = (slice(max(j - r, 0), j + r + 1), slice(max(i - r, 0), i + r + 1))
    return dict(file=name, lat=float(st["lat"][j, i]), lon=float(st["lon"][j, i]), hgt=float(st["hgt"][j, i]),
                landmask=int(st["lm"][j, i]), ws_ref=float(a["ws"][j, i]), ws_alt=float(b["ws"][j, i]),
                cells_absdiff_gt1_domain=int((np.abs(d) > 1).sum()), n_components_gt1=int(n), event_component_size=int(comp.sum()),
                largest_component_size=int(np.bincount(lab.ravel())[1:].max()) if n else 0,
                rain_1h_ref_max_11x11=float((a["rain"] - ap["rain"])[sl].max()), rain_1h_alt_max_11x11=float((b["rain"] - bp["rain"])[sl].max()),
                wmax_ref_11x11=float(a["wmax"][sl].max()), wmax_alt_11x11=float(b["wmax"][sl].max()),
                rmse_outside_11x11=float(np.sqrt(np.mean(np.delete(d.ravel(), np.ravel_multi_index(np.indices(d.shape)[:, sl[0], sl[1]].reshape(2, -1), d.shape)) ** 2))))


def main():
    out = {"doc": __doc__, "gpu_vs_cpu": {}, "twin_vs_cpu": {}, "events": {}}
    for case in ("WN3-0227", "WN3-0502", "WN3-0614"):
        for dom in ("d02", "d03"):
            m, st = masks(case, dom)
            out["gpu_vs_cpu"][f"{case}|{dom}"] = s = pair_stats(load_pairs(F20 / case / "cpu", F20 / case / "gpu", dom), m)
            if case == "WN3-0227" and dom == "d03":
                at = s["coast_sea"]["maxabs_at"]
                out["events"]["gpu_vs_cpu|0227|d03|coast_sea_max"] = event(case, dom, at["lead"], at["j"], at["i"], F20 / case / "cpu", F20 / case / "gpu", st)
    for dom in ("d02", "d03"):
        m, st = masks("WN3-0227", dom)
        pairs = load_pairs(F20 / "WN3-0227" / "cpu", TWIN, dom, leads=set(range(1, 25)))
        if pairs:
            s = pair_stats(pairs, m)
            out["twin_vs_cpu"][f"WN3-0227|{dom}"] = dict(leads=[p[0] for p in pairs], stats=s)
            # same-lead GPU-vs-CPU restricted to the twin's leads (like-for-like)
            g = load_pairs(F20 / "WN3-0227" / "cpu", F20 / "WN3-0227" / "gpu", dom, leads={p[0] for p in pairs})
            out["twin_vs_cpu"][f"WN3-0227|{dom}"]["gpu_same_leads"] = pair_stats(g, m)
            if dom == "d03":
                at = s["coast_sea"]["maxabs_at"]
                out["events"]["twin_vs_cpu|0227|d03|coast_sea_max"] = event("WN3-0227", dom, at["lead"], at["j"], at["i"], F20 / "WN3-0227" / "cpu", TWIN, st)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
