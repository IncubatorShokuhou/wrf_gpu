"""A5: post-GPU analysis of the o5-coast arms (CPU only).

usage: python a5_post_gpu.py GPU_OUT_DIR OUT.json
GPU_OUT_DIR = coast_gpu.sh output root with <ARM>/frames/*.npz (stream_extract.py).  Reference = our own CPU-WRF twin
0227 (w43-twin0227, same inputs).  For every arm and domain:
  * lag fit of mean(GPU-CPU) MU and PSFC against the CPU central tendency per boundary ring (0..9, interior)
    -- released clock: d01 rings 1-9 ~ +1 root dt; WRF walk: ~0;
  * U10/V10 (earth-relative) and WS10 RMSE/bias vs the twin by lead block (1-24, 25-47, 48-72) and mask
    (all, open sea, coast-sea, coast-land, interior land);
  * pairwise between arms (e.g. REL vs WALK, and vs te_gpu where frames overlap) with the same statistics;
  * nonfinite counts per arm (stream_extract MANIFEST).
No JAX."""
import glob
import json
import os
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_raw import load_frames  # noqa: E402

TWIN = "<DATA_ROOT>/wrf_gpu2/scratch/w43-twin0227/C2_4x1/cpu"
TE = "<USER_HOME>/wrf_gpu2_lanes/w43-teacher/te_gpu/20260227_gpu/d03_sfc.npz"
BLOCKS = {"t1_24": (1, 24), "t25_47": (25, 47), "t48_72": (48, 72)}


def twin_series(dom, times, fields):
    out = {f: [] for f in fields}
    for t in times:
        p = f"{TWIN}/wrfout_{dom}_{t}"
        with nc.Dataset(p) as f:
            ca = np.asarray(f["COSALPHA"][0], np.float64); sa = np.asarray(f["SINALPHA"][0], np.float64)
            u = np.asarray(f["U10"][0], np.float64); v = np.asarray(f["V10"][0], np.float64)
            vals = dict(MU=np.asarray(f["MU"][0], np.float64), PSFC=np.asarray(f["PSFC"][0], np.float64),
                        u10_east=u * ca - v * sa, v10_north=v * ca + u * sa)
        for k in fields:
            out[k].append(vals[k])
    return {k: np.stack(v) for k, v in out.items()}


def masks(lm):
    lm = lm > 0.5
    k = np.ones((5, 5), bool)
    edge = binary_dilation(lm, k) & binary_dilation(~lm, k)
    return {"all": np.ones_like(lm), "open_sea": ~lm & ~edge, "coast_sea": edge & ~lm, "coast_land": edge & lm,
            "interior_land": lm & ~edge}


def lag_fit(g, c, tend):
    ny, nx = c.shape[1:]
    jj, ii = np.mgrid[0:ny, 0:nx]
    dist = np.minimum(np.minimum(jj, ny - 1 - jj), np.minimum(ii, nx - 1 - ii))
    res = {}
    for name, m in [(f"ring{k}", dist == k) for k in range(10)] + [("interior", dist >= 10)]:
        dm = (g - c)[:, m].mean(1); tm = tend[:, m].mean(1)
        sel = np.isfinite(tm)
        res[name] = dict(lag_s=float((dm[sel] * tm[sel]).sum() / max((tm[sel] ** 2).sum(), 1e-30)),
                         diff_rms=float(np.sqrt((dm[sel] ** 2).mean())))
    return res


def wind_stats(ga, gb, ms, taus):
    out = {}
    for mk, m in ms.items():
        out[mk] = {}
        for bk, (a, b) in BLOCKS.items():
            sel = [i for i, t in enumerate(taus) if a <= t <= b]
            if not sel:
                continue
            row = {}
            for comp in ("u10_east", "v10_north", "ws"):
                d = (ga[comp][sel][:, m] - gb[comp][sel][:, m]).astype(np.float64)
                row[comp] = dict(rmse=float(np.sqrt((d ** 2).mean())), bias=float(d.mean()))
            out[mk][bk] = row
    return out


def main():
    root, outp = Path(sys.argv[1]), Path(sys.argv[2])
    res = {"gpu_out": str(root), "arms": {}}
    arms = sorted(p.name for p in root.iterdir() if (p / "frames").is_dir())
    stacks = {}
    for arm in arms:
        man = root / arm / "frames" / "MANIFEST.jsonl"
        recs = [json.loads(x) for x in man.read_text().splitlines() if x.strip()] if man.exists() else []
        res["arms"][arm] = {"frames": len(recs), "nonfinite_frames": [r["name"] for r in recs if r["nonfinite_fields"]]}
        for dom in ("d01", "d02", "d03"):
            if not glob.glob(str(root / arm / "frames" / f"wrfout_{dom}_*.npz")):
                continue
            a = load_frames(str(root / arm / "frames"), dom)
            a["ws"] = np.hypot(a["u10_east"], a["v10_north"])
            stacks[(arm, dom)] = a
    for (arm, dom), a in stacks.items():
        times = [str(t) for t in a["times"]]
        tall = sorted(os.path.basename(p)[11:] for p in glob.glob(f"{TWIN}/wrfout_{dom}_*"))
        common = [t for t in times if t in tall]
        gi = [times.index(t) for t in common]
        c = twin_series(dom, tall, ("MU", "PSFC", "u10_east", "v10_north"))
        ci = [tall.index(t) for t in common]
        import datetime as dtm
        secs = np.array([dtm.datetime.strptime(t, "%Y-%m-%d_%H:%M:%S").timestamp() for t in tall])
        entry = res["arms"][arm].setdefault(dom, {})
        for fld in ("MU", "PSFC"):
            tend = np.full_like(c[fld], np.nan)
            tend[1:-1] = (c[fld][2:] - c[fld][:-2]) / (secs[2:] - secs[:-2])[:, None, None]
            entry[f"lag_{fld}"] = lag_fit(a[fld][gi].astype(np.float64), c[fld][ci], tend[ci])
        cw = {k: c[k][ci] for k in ("u10_east", "v10_north")}
        cw["ws"] = np.hypot(cw["u10_east"], cw["v10_north"])
        gw = {k: a[k][gi] for k in ("u10_east", "v10_north", "ws")}
        taus = list(range(len(common)))  # frame 0 = t0 (emit_initial_history)
        entry["wind_vs_twin"] = wind_stats(gw, cw, masks(a["s_LANDMASK"]), taus)
    # pairwise between arms on common frames
    pairs = {}
    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            for dom in ("d01", "d02", "d03"):
                if (x, dom) in stacks and (y, dom) in stacks:
                    a, b = stacks[(x, dom)], stacks[(y, dom)]
                    tc = [t for t in a["times"] if t in set(b["times"])]
                    ia = [list(a["times"]).index(t) for t in tc]; ib = [list(b["times"]).index(t) for t in tc]
                    pairs[f"{x}-{y}/{dom}"] = wind_stats({k: a[k][ia] for k in ("u10_east", "v10_north", "ws")},
                                                         {k: b[k][ib] for k in ("u10_east", "v10_north", "ws")},
                                                         masks(a["s_LANDMASK"]), list(range(len(tc))))
    res["pairs"] = pairs
    outp.write_text(json.dumps(res, indent=1))
    for arm in arms:
        for dom in ("d01", "d02", "d03"):
            e = res["arms"][arm].get(dom)
            if e:
                lag = e["lag_MU"]
                print(arm, dom, "MU lag rings1-9", [round(lag[f"ring{k}"]["lag_s"], 1) for k in range(1, 10)],
                      "interior", round(lag["interior"]["lag_s"], 1))


if __name__ == "__main__":
    main()
