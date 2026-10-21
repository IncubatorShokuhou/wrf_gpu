"""Extract El Medano model wind series from staged teacher extracts (alisios.teacher_extract.v2, d03_sfc.npz).

usage: python wt2_points.py OUT.npz EXTRACT_ROOT [label]
Per issue and lead: wind_speed and direction (from u10_east/v10_north) at
  sea   = ALISIOS point-set pixel of ELMEDANO_T1/CABEZO_SURF (grid 19,55; LANDMASK 0),
  urban = pixel 20,55 (LANDMASK 1, MODIS 13; also the windguru:14924 pixel),
  bil   = bilinear interpolation of the 4 surrounding pixels at the T1 point-set coordinate (28.04120 N, 16.53460 W),
  box_sea_mean / box_max over the 7x7 box around (20,55).
Plus actual valid time (unix s), lead_h, tau_h. Values only written to OUT (lane worktree / scratch), never logged."""
import sys
from pathlib import Path
import numpy as np

T1 = (28.04120, -16.53460)


def bilinear_weights(lat, lon):
    # local planar inversion on the 2x2 cell containing T1 (grid ~1 km, Lambert): solve by Newton on (fy, fx)
    d = (lat - T1[0]) ** 2 + ((lon - T1[1]) * np.cos(np.radians(T1[0]))) ** 2
    j, i = np.unravel_index(d.argmin(), d.shape)
    j0 = j - 1 if lat[j, i] > T1[0] else j
    i0 = i - 1 if lon[j, i] > T1[1] else i
    fy, fx = 0.5, 0.5
    for _ in range(20):
        def p(a):
            return ((1 - fy) * (1 - fx) * a[j0, i0] + (1 - fy) * fx * a[j0, i0 + 1] + fy * (1 - fx) * a[j0 + 1, i0] + fy * fx * a[j0 + 1, i0 + 1])
        r = np.array([p(lat) - T1[0], p(lon) - T1[1]])
        dfy = np.array([((1 - fx) * (lat[j0 + 1, i0] - lat[j0, i0]) + fx * (lat[j0 + 1, i0 + 1] - lat[j0, i0 + 1])),
                        ((1 - fx) * (lon[j0 + 1, i0] - lon[j0, i0]) + fx * (lon[j0 + 1, i0 + 1] - lon[j0, i0 + 1]))])
        dfx = np.array([((1 - fy) * (lat[j0, i0 + 1] - lat[j0, i0]) + fy * (lat[j0 + 1, i0 + 1] - lat[j0 + 1, i0])),
                        ((1 - fy) * (lon[j0, i0 + 1] - lon[j0, i0]) + fy * (lon[j0 + 1, i0 + 1] - lon[j0 + 1, i0]))])
        J = np.stack([dfy, dfx], 1)
        step = np.linalg.solve(J, -r)
        fy, fx = fy + step[0], fx + step[1]
    w = {(j0, i0): (1 - fy) * (1 - fx), (j0, i0 + 1): (1 - fy) * fx, (j0 + 1, i0): fy * (1 - fx), (j0 + 1, i0 + 1): fy * fx}
    return w


def main():
    out, root = Path(sys.argv[1]), Path(sys.argv[2])
    res = {}
    for d in sorted(p for p in root.iterdir() if (p / "d03_sfc.npz").exists()):
        z = np.load(d / "d03_sfc.npz")
        lat, lon, lm = z["XLAT"], z["XLONG"], z["LANDMASK"]
        assert lm[19, 55] < 0.5 and lm[20, 55] > 0.5, d.name
        w = bilinear_weights(lat, lon)
        ue, vn = z["u10_east"], z["v10_north"]
        ws = np.hypot(ue, vn)
        box = (slice(17, 24), slice(52, 59))
        bil_u = sum(wt * ue[:, j, i] for (j, i), wt in w.items()); bil_v = sum(wt * vn[:, j, i] for (j, i), wt in w.items())
        sea_box = lm[box] < 0.5
        res[d.name] = dict(
            t=z["actual_time_unix_s"], lead_h=z["lead_h"], tau_h=z["tau_h"],
            ws_sea=ws[:, 19, 55], ws_urban=ws[:, 20, 55], ws_bil=np.hypot(bil_u, bil_v),
            dir_sea=(np.degrees(np.arctan2(-ue[:, 19, 55], -vn[:, 19, 55])) % 360),
            box_sea_mean=np.array([x[box][sea_box].mean() for x in ws]), box_max=ws[:, box[0], box[1]].reshape(len(ws), -1).max(1),
            bil_weights=np.array([[j, i, wt] for (j, i), wt in w.items()]),
            swdown_sea=z["SWDOWN"][:, 19, 55], hfx_urban=z["HFX"][:, 20, 55], pblh_sea=z["PBLH"][:, 19, 55])
    np.savez_compressed(out, **{f"{k}__{kk}": vv for k, v in res.items() for kk, vv in v.items()})
    print(len(res), "issues")


if __name__ == "__main__":
    main()
