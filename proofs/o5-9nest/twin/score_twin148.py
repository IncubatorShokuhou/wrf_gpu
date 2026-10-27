"""Score a #148 CPU twin against the CPU reference on the 15 frozen D6 rows: C(τ) = RMSE(twin − ref), X(τ) = RMSE(GPU − ref).

Reference/GPU arrays: the w43-148fid derived extracts (d0X.npz, cpu_*/gpu_*; extraction == ALISIOS scorer by_lead, 0.0 diff).
Twin: wrfout files of run_twin148.py (own CPU run), domain numbering mapped back (n18k5: twin d03/d04/d05 = ref d04/d05/d06).
Rule reported (teacher PREREG form of A3, k = 4/3): a row is FLOOR-LIMITED iff EVERY breach lead (X > limit) has a twin frame
and X <= 4/3·C there; missing twin leads -> INCOMPLETE (never PASS). The frozen 'C > limit passes any X' clause is NOT used,
only disclosed per lead. Field definitions like the scorer: 2-D fields all cells; W all levels/cells. Non-finite -> refuse.

usage: JAX_PLATFORMS=cpu python score_twin148.py <npz_dir> <twin_run_dir> <mode n18|n18k5> <out.json>
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

T0 = datetime(2026, 10, 18, 18)
LIMIT = {"RAINNC": 1.0, "W": 0.3, "U10": 1.5, "V10": 1.5}
ROWS = [(24, "d02", "RAINNC"), (24, "d04", "RAINNC"), (24, "d05", "RAINNC"), (24, "d06", "RAINNC"), (24, "d06", "W"),
        (72, "d02", "RAINNC"), (72, "d04", "RAINNC"), (72, "d04", "U10"), (72, "d04", "V10"), (72, "d04", "W"),
        (72, "d05", "RAINNC"), (72, "d06", "RAINNC"), (72, "d06", "U10"), (72, "d06", "V10"), (72, "d06", "W")]
MAP = {"n18": {f"d0{i}": f"d0{i}" for i in range(1, 10)},
       "n18k5": {"d01": "d01", "d02": "d02", "d04": "d03", "d05": "d04", "d06": "d05"}}
K = 4.0 / 3.0


def rmse(d):
    d = np.asarray(d, np.float64)
    if not np.isfinite(d).all():
        raise ValueError("non-finite difference")
    return float(np.sqrt(np.mean(d * d)))


def twin_frames(run_dir, dom):
    """lead -> path; d01 native dt54 alarm times (+18/+36/+54 s) are paired to the nearest hour (|offset| < 60 s)."""
    out = {}
    for p in Path(run_dir).glob(f"wrfout_{dom}_*"):
        t = datetime.strptime(p.name.split("_", 2)[2], "%Y-%m-%d_%H:%M:%S")
        s = (t - T0).total_seconds()
        lead = int(round(s / 3600.0))
        if abs(s - 3600 * lead) < 60:
            out[lead] = p
    return out


def lag_control(cache, run_dir, mode, npz_dir, max_lead=12):
    """R0 control: fit mean_ring(twin - ref)(t) = lag * mean_ring(dPSFC_ref/dt) like lag148.py; a CPU-CPU pair should give ~0 s
    (the GPU shows d01 rings 1-9 ~ +54 s). Uses PSFC of the twin wrfouts and of the reference extract."""
    import netCDF4
    out = {}
    for dom in ("d01", "d02"):
        z = cache.get(dom)
        if z is None:
            p = Path(npz_dir) / f"{dom}.npz"
            if not p.exists():
                continue
            z = cache.setdefault(dom, dict(np.load(p)))
        leads = z["leads"].tolist()
        frames = twin_frames(run_dir, MAP[mode][dom])
        ref = z["cpu_PSFC"].astype(np.float64)
        ny, nx = ref.shape[1:]
        y, x = np.mgrid[0:ny, 0:nx]
        rg = np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x))
        tw = {}
        for i, L in enumerate(leads):
            if 0 <= L <= max_lead + 1 and L in frames:
                with netCDF4.Dataset(frames[L]) as nc:
                    tw[L] = nc.variables["PSFC"][0].astype(np.float64)
        res = {}
        for name, m in [(f"ring{r}", rg == r) for r in range(10)] + [("interior", rg >= 10)]:
            if not m.any():
                continue
            d, t = [], []
            for i, L in enumerate(leads):
                if 1 <= L <= max_lead and L in tw and 0 < i < len(leads) - 1 and leads[i + 1] - leads[i - 1] == 2:
                    d.append((tw[L] - ref[i])[m].mean())
                    t.append(((ref[i + 1] - ref[i - 1]) / 7200.0)[m].mean())
            if len(d) >= 3:
                d, t = np.array(d), np.array(t)
                res[name] = {"lag_s": float((d * t).sum() / (t * t).sum()), "mean_abs_diff_Pa": float(np.abs(d).mean()), "n": len(d)}
        out[dom] = res
    return out


def main(npz_dir, run_dir, mode, out_path):
    import netCDF4
    res, cache = {"rule": __doc__.split("\n\n")[1], "mode": mode, "rows": []}, {}
    for stage, dom, f in ROWS:
        z = cache.setdefault(dom, dict(np.load(Path(npz_dir) / f"{dom}.npz")))
        leads = z["leads"].tolist()
        frames = twin_frames(run_dir, MAP[mode][dom])
        rows = []
        for i, L in enumerate(leads):
            if L > stage:
                continue
            ref, gpu = z[f"cpu_{f}"][i].astype(np.float64), z[f"gpu_{f}"][i].astype(np.float64)
            x = rmse(gpu - ref)
            c, tw = None, None
            if L in frames:
                try:
                    with netCDF4.Dataset(frames[L]) as nc:
                        nc.set_auto_mask(False)
                        tw = nc.variables[f][0].astype(np.float64)
                except OSError:
                    tw = None  # frame still being written
                if tw is not None:
                    if tw.shape != ref.shape:
                        raise ValueError(f"shape {tw.shape} != {ref.shape} for {dom} {f}")
                    c = rmse(tw - ref)
            def corr(a, b):
                a, b = a.ravel(), b.ravel()
                return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
            r_x = corr(gpu, ref)
            r_c = corr(tw, ref) if c is not None else None
            rows.append({"lead": L, "X": x, "C": c, "r_X": r_x, "r_C": r_c, "std_ref": float(ref.std()),
                         "breach": x > LIMIT[f],
                         "x_le_kc": None if c is None else x <= K * c, "c_gt_limit": None if c is None else c > LIMIT[f]})
        br = [r for r in rows if r["breach"]]
        if not br:
            verdict = "NO_BREACH"
        elif any(r["C"] is None for r in br):
            verdict = "INCOMPLETE"
        else:
            verdict = "FLOOR-LIMITED" if all(r["x_le_kc"] for r in br) else "FAIL"
        lm = [r for r in br if r["r_C"] is not None and r["r_X"] is not None]
        res["rows"].append({"stage": stage, "dom": dom, "field": f, "verdict": verdict, "n_breach": len(br),
                            "lead_matched_rX_ge_rC": [sum(r["r_X"] >= r["r_C"] for r in lm), len(lm)],
                            "n_breach_with_twin": sum(r["C"] is not None for r in br),
                            "max_X_over_C": max((r["X"] / r["C"] for r in br if r["C"]), default=None), "by_lead": rows})
    res["lag_control"] = lag_control(cache, run_dir, mode, npz_dir)
    Path(out_path).write_text(json.dumps(res, indent=1))
    for r in res["rows"]:
        print(r["stage"], r["dom"], r["field"], r["verdict"], f"{r['n_breach_with_twin']}/{r['n_breach']}", r["max_X_over_C"])


if __name__ == "__main__":
    main(*sys.argv[1:5])
