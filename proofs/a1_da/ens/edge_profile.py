"""Prior spread and ensemble-mean error vs distance from the lateral boundary (Swiss twin, members at t_a).
Usage: edge_profile.py out.json [A0,A1,A1b]  — reads osse/<arm>/c<k>/m*/wrfout_d01_<t_a> kept by swiss_osse.py."""
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swiss_osse as so  # noqa: E402
from gpuwrf.assimilation.letkf import spread  # noqa: E402

RINGS = ((0, 5), (5, 8), (8, 12), (12, 16), (16, 99))


def profile(arm, c):
    t_a = so.T0 + timedelta(hours=so.CYCLE_H * (c + 1))
    base = so.ROOT / arm / f"c{c}"
    if arm == "A1b" and c == 0:
        base = so.ROOT / "A1" / "c0"
    files = sorted(base.glob(f"m*/wrfout_d01_{so.wt(t_a)}"))
    if len(files) != so.N:
        return None
    fr = so.stack([so.state_io.read_frame(f, so.wt(t_a), ("U", "V", "T", "PH", "PHB")) for f in files])
    tf = so.truth_frame(t_a)
    geo = so.geometry()
    band = spread.band_index(fr, geo)
    ny, nx = band.shape[1:]
    j, i = np.mgrid[0:ny, 0:nx]
    dist = np.minimum(np.minimum(j, ny - 1 - j), np.minimum(i, nx - 1 - i))
    out = {}
    for name in ("T", "U", "V"):
        m = spread._mass(name, fr[name])
        sd = m.std(0, ddof=1)
        err = m.mean(0) - spread._mass(name, np.asarray(tf[name])[None])[0]
        out[name] = {}
        for lo, hi in RINGS:
            sel = (band <= 2) & ((dist >= lo) & (dist < hi))[None]
            out[name][f"{lo}-{hi}"] = {"spread": float(np.sqrt(np.mean(sd[sel] ** 2))),
                                       "rmse": float(np.sqrt(np.mean(err[sel] ** 2)))}
    return out


if __name__ == "__main__":
    res = {}
    for arm in (sys.argv[2].split(",") if len(sys.argv) > 2 else ("A0", "A1", "A1b")):
        for c in range(so.N_ANALYSES):
            p = profile(arm, c)
            if p is not None:
                res[f"{arm}/c{c}"] = p
    res["note"] = "0-5 km AGL (bands 0-2), rings in grid cells from the nearest lateral edge; spec zone 5, relax to 5+5"
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        if k != "note":
            print(k, {n: {r: f"{x['spread']:.2f}/{x['rmse']:.2f}" for r, x in v[n].items()} for n in ("T", "V")})
