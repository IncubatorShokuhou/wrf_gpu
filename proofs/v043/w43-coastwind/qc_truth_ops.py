"""Q-C operator consistency (critic R2): WT2 CPU-WRF vs admitted T1/T2 with three interpolation operators.

usage: python qc_truth_ops.py OUT.json
Operators on the WT2 d03 extracts (guarded stage S03): (a) vector-bilinear at the T1 point-set coordinate (first draft:
bilinear U/V then speed), (b) scalar-bilinear of the native wind_speed field at the same coordinate (the F20 operator),
(c) ALISIOS surf-monitor operator: scalar-bilinear of native wind_speed at their water point 28.04105162321547 N,
-16.53422754236671 E (cross TO_WRF_GPU2_COASTWIND_ANSWER_20261021.md); plus (d) sea pixel (19,55) and (e) urban (20,55).
Obs = T1/T2 from the private matched-row table (qc 0, +-10 min mean). Aggregates only: all hours; local 10-18;
observation-selected surf (T1 >= 15 kt, 10-18 L); forecast-selected surf (operator value >= 15 kt, 10-18 L);
per-issue all-hours bias sign for every operator. Values never logged per sample."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

S = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind")
ROOT = S / "stage/S03_wt2/root2/state/manager/teacher_extract"
PTS = {"T1": (28.04120, -16.53460), "ALISIOS": (28.04105162321547, -16.53422754236671)}
SURF = 7.717


def bil(lat, lon, la, lo):
    d = (lat - la) ** 2 + ((lon - lo) * np.cos(np.radians(la))) ** 2
    j, i = np.unravel_index(d.argmin(), d.shape)
    j0 = j - 1 if lat[j, i] > la else j
    i0 = i - 1 if lon[j, i] > lo else i
    fy, fx = 0.5, 0.5
    for _ in range(30):
        q = lambda a: (1 - fy) * (1 - fx) * a[j0, i0] + (1 - fy) * fx * a[j0, i0 + 1] + fy * (1 - fx) * a[j0 + 1, i0] + fy * fx * a[j0 + 1, i0 + 1]
        r = np.array([q(lat) - la, q(lon) - lo])
        dfy = np.array([(1 - fx) * (lat[j0 + 1, i0] - lat[j0, i0]) + fx * (lat[j0 + 1, i0 + 1] - lat[j0, i0 + 1]),
                        (1 - fx) * (lon[j0 + 1, i0] - lon[j0, i0]) + fx * (lon[j0 + 1, i0 + 1] - lon[j0, i0 + 1])])
        dfx = np.array([(1 - fy) * (lat[j0, i0 + 1] - lat[j0, i0]) + fy * (lat[j0 + 1, i0 + 1] - lat[j0 + 1, i0]),
                        (1 - fy) * (lon[j0, i0 + 1] - lon[j0, i0]) + fy * (lon[j0 + 1, i0 + 1] - lon[j0 + 1, i0])])
        st = np.linalg.solve(np.stack([dfy, dfx], 1), -r); fy, fx = fy + st[0], fx + st[1]
    return {(j0, i0): (1 - fy) * (1 - fx), (j0, i0 + 1): (1 - fy) * fx, (j0 + 1, i0): fy * (1 - fx), (j0 + 1, i0 + 1): fy * fx}


def st(m, o):
    ok = np.isfinite(m) & np.isfinite(o)
    if ok.sum() < 10:
        return None
    d = m[ok] - o[ok]
    return dict(n=int(ok.sum()), model=round(float(m[ok].mean()), 3), obs=round(float(o[ok].mean()), 3), bias=round(float(d.mean()), 3),
                rmse=round(float(np.sqrt((d ** 2).mean())), 3))


def main():
    rows = pd.read_parquet(S / "truth/MATCHED_ROWS.parquet")
    rows = rows[rows.src == "WT2_CPU"].copy()
    ops = {k: np.full(len(rows), np.nan) for k in ("vec_bil_T1", "scal_bil_T1", "scal_bil_ALISIOS", "sea_19_55", "urban_20_55")}
    weights = {}
    for iss, g in rows.groupby("issue"):
        z = np.load(ROOT / iss / "d03_sfc.npz")
        lat, lon, ws, ue, vn = z["XLAT"], z["XLONG"], z["wind_speed"], z["u10_east"], z["v10_north"]
        t = pd.to_datetime(z["actual_time_unix_s"], unit="s")
        idx = {tt: k for k, tt in enumerate(t)}
        wT1, wA = bil(lat, lon, *PTS["T1"]), bil(lat, lon, *PTS["ALISIOS"])
        weights[iss] = {"T1": {f"{a},{b}": round(float(v), 4) for (a, b), v in wT1.items()},
                        "ALISIOS": {f"{a},{b}": round(float(v), 4) for (a, b), v in wA.items()}}
        for r_i, tt in zip(g.index, g.t):
            k = idx.get(pd.Timestamp(tt))
            if k is None:
                continue
            pos = rows.index.get_loc(r_i)
            u = sum(w * ue[k, j, i] for (j, i), w in wT1.items()); v = sum(w * vn[k, j, i] for (j, i), w in wT1.items())
            ops["vec_bil_T1"][pos] = np.hypot(u, v)
            ops["scal_bil_T1"][pos] = sum(w * ws[k, j, i] for (j, i), w in wT1.items())
            ops["scal_bil_ALISIOS"][pos] = sum(w * ws[k, j, i] for (j, i), w in wA.items())
            ops["sea_19_55"][pos] = ws[k, 19, 55]; ops["urban_20_55"][pos] = ws[k, 20, 55]
    for k, v in ops.items():
        rows[k] = v
    surfh = (rows.local_h >= 10) & (rows.local_h < 18)
    out = {"doc": __doc__, "bilinear_weights_first_issue": weights[sorted(weights)[0]], "ops": {}, "per_issue_sign": {},
           "vec_minus_scal_T1": {"mean": float(np.nanmean(ops["vec_bil_T1"] - ops["scal_bil_T1"])),
                                 "maxabs": float(np.nanmax(np.abs(ops["vec_bil_T1"] - ops["scal_bil_T1"])))}}
    for k in ops:
        for truth in ("T1", "T2"):
            out["ops"][f"{k}|{truth}"] = {
                "all_hours": st(rows[k].values, rows[truth].values),
                "local_10_18": st(rows[k][surfh].values, rows[truth][surfh].values),
                "obs_selected_surf": st(rows[k][surfh & (rows.T1 >= SURF)].values, rows[truth][surfh & (rows.T1 >= SURF)].values),
                "fcst_selected_surf": st(rows[k][surfh & (rows[k] >= SURF)].values, rows[truth][surfh & (rows[k] >= SURF)].values)}
        b = rows.groupby("issue").apply(lambda g: np.nanmean(g[k] - g["T1"]))
        out["per_issue_sign"][k] = {"n_issues": int(b.notna().sum()), "n_negative": int((b < 0).sum()),
                                    "negative": {i: round(float(x), 3) for i, x in b[b < 0].items()}}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
