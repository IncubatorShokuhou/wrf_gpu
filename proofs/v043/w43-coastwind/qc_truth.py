"""Q-C truth comparison: WRF 10 m wind at El Medano vs admitted observations (A-V043-COASTWIND-T1-TRUTH-EXPORT).

usage: python qc_truth.py OUT.json
Model series: WT2 CPU-WRF teacher extracts (33 issues, 72 h; scratch WT2_medano_points.npz) and F20 v0.4.0 GPU + original
CPU-WRF (0227/0502/0614, 24 h; wrfout at <DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016).
Points: sea = d03 (19,55) (ALISIOS point-set pixel of T1/Cabezo), urban = (20,55), bil = bilinear at T1 coordinates.
Obs: qc_flag 0, mean of all samples in [t-10 min, t+10 min] around the model valid time t (5-min Cabezo/T1/T2, 10-min WG).
Stats per station x model point: n, mean model, mean obs, bias (model-obs), RMSE; splits by lead block, Cabezo/T1
direction sector (30 deg), local surf hours (10-18 Atlantic/Canary) and a common subset where T1, Cabezo and WG all exist.
Values only go to OUT (aggregates), never logged per sample."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc

S = Path("<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind")
F20 = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016")
STN = {"T1": "elmedano:ELMEDANO_T1_DESHIELDED", "T2": "elmedano:ELMEDANO_T2_ONWATER", "CABEZO": "cabezo:CABEZO_SURF", "WG": "windguru:14924"}
W = {(19, 55): 0.32979518, (19, 56): 0.18298499, (20, 55): 0.31335601, (20, 56): 0.17386381}  # bilinear (WT2 grid == F20 grid)


def obs_table():
    o = pd.read_parquet(S / "truth/COASTWIND_T1_OBS_20261020.parquet")
    o = o[o.qc_flag == 0]
    return {k: o[(o.station_id == sid)].pivot_table(index="ts_utc", columns="variable", values="value", aggfunc="mean").sort_index()
            for k, sid in STN.items()}


def window_mean(df, t, col):
    if col not in df:
        return np.nan
    s = df[col].loc[t - pd.Timedelta(minutes=10): t + pd.Timedelta(minutes=10)].dropna()
    if col == "wind_dir_deg" and len(s):
        a = np.radians(s.values); return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)
    return float(s.mean()) if len(s) else np.nan


def model_rows():
    rows = []
    z = np.load(S / "WT2_medano_points.npz")
    for iss in sorted({k.split("__")[0] for k in z.files}):
        t = z[iss + "__t"]; lead = z[iss + "__tau_h"]
        for k in range(1, len(t)):
            rows.append(dict(src="WT2_CPU", issue=iss, t=pd.Timestamp(int(t[k]), unit="s"), lead=int(lead[k]),
                             sea=float(z[iss + "__ws_sea"][k]), urban=float(z[iss + "__ws_urban"][k]), bil=float(z[iss + "__ws_bil"][k])))
    for case in ("WN3-0227", "WN3-0502", "WN3-0614"):
        for eng in ("cpu", "gpu"):
            files = sorted((F20 / case / eng).glob("wrfout_d03_*"))
            for lead, f in enumerate(files):
                if lead == 0:
                    continue
                with nc.Dataset(f) as ds:
                    ws = np.hypot(np.asarray(ds["U10"][0], float), np.asarray(ds["V10"][0], float))
                    tt = pd.Timestamp(f.name[11:].replace("_", " "))
                rows.append(dict(src=f"F20_{eng.upper()}", issue=case, t=tt, lead=lead, sea=float(ws[19, 55]), urban=float(ws[20, 55]),
                                 bil=float(sum(w * ws[j, i] for (j, i), w in W.items()))))
    return pd.DataFrame(rows)


def stats(m, o):
    ok = np.isfinite(m) & np.isfinite(o)
    if ok.sum() < 5:
        return None
    d = m[ok] - o[ok]
    return dict(n=int(ok.sum()), model=round(float(m[ok].mean()), 3), obs=round(float(o[ok].mean()), 3),
                bias=round(float(d.mean()), 3), rmse=round(float(np.sqrt((d ** 2).mean())), 3), corr=round(float(np.corrcoef(m[ok], o[ok])[0, 1]), 3))


def main():
    obs = obs_table(); df = model_rows()
    for k, tab in obs.items():
        df[k] = [window_mean(tab, t, "wind_speed_ms") for t in df.t]
    df["dir_obs"] = [window_mean(obs["CABEZO"], t, "wind_dir_deg") for t in df.t]
    df["local_h"] = (df.t.dt.tz_localize("UTC").dt.tz_convert("Atlantic/Canary").dt.hour)
    out = {"doc": __doc__, "counts": {s: int((df.src == s).sum()) for s in df.src.unique()}, "overall": {}, "by_lead": {}, "by_sector": {},
           "surf_hours": {}, "common_subset": {}}
    for src, g in df.groupby("src"):
        for st in STN:
            for pt in ("sea", "urban", "bil"):
                key = f"{src}|{st}|{pt}"
                out["overall"][key] = stats(g[pt].values, g[st].values)
                if src == "WT2_CPU" and pt in ("sea", "bil"):
                    for lo, hi in ((1, 24), (25, 48), (49, 72)):
                        m = (g.lead >= lo) & (g.lead <= hi); out["by_lead"][f"{key}|{lo}-{hi}"] = stats(g[pt][m].values, g[st][m].values)
                    for s in range(0, 360, 30):
                        m = ((g.dir_obs - s) % 360) < 30; out["by_sector"][f"{key}|{s}"] = stats(g[pt][m].values, g[st][m].values)
                    m = (g.local_h >= 10) & (g.local_h < 18); out["surf_hours"][key] = stats(g[pt][m].values, g[st][m].values)
        c = g[np.isfinite(g.T1) & np.isfinite(g.CABEZO) & np.isfinite(g.WG)]
        for st in STN:
            out["common_subset"][f"{src}|{st}|sea"] = stats(c["sea"].values, c[st].values)
            out["common_subset"][f"{src}|{st}|bil"] = stats(c["bil"].values, c[st].values)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    df.to_parquet(S / "truth/MATCHED_ROWS.parquet")  # private scratch (truth values), never committed


if __name__ == "__main__":
    main()
