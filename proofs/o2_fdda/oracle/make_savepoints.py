"""Generate pristine-WRF grid-FDDA savepoints (lane o2-fdda).

usage: python make_savepoints.py <oracle_build_dir> <out_dir>

Real-structured inputs: crops of original CPU-WRF history frames (Swiss RD11 winter
case, PROD D5 summer case; land + sea + terrain).  Analysis records are CPU-WRF
frames of other hours (``*_NDG_OLD`` = frame(t_k), ``*_NDG_NEW`` = frame(t_k+1)),
model states are frames of in-between hours, so target-minus-state differences
are real.  Per case a CALL SEQUENCE drives the oracle (itimestep, seconds, record
read, frame) through the driver control + coupling + decoupling chain.

numpy + netCDF4 only (no JAX).  Precisions: r4 (WRF REAL) for every case, r8
(-fdefault-real-8) for the cases marked ``r8``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np

SWISS = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
PROD = Path("<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/"
            "alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case")
RVRD = np.float32(np.float32(461.6) / np.float32(287.0))
G = np.float32(9.81)

INT_KEYS = ("if_no_pbl_nudging_uv", "if_no_pbl_nudging_t", "if_no_pbl_nudging_q", "if_zfac_uv", "k_zfac_uv",
            "if_zfac_t", "k_zfac_t", "if_zfac_q", "k_zfac_q", "if_ramping", "interval_m", "end_h", "end_h_sfc",
            "stepfg", "fgdtzero", "specified")
REAL_KEYS = ("guv", "gt", "gq", "dtramp_min", "dt", "dx")


def frame(path: Path, ys: slice, xs: slice) -> dict:
    """One CPU-WRF history frame cropped to mass rows ys / cols xs (port layout (k, j, i))."""

    with netCDF4.Dataset(path) as d:
        g = lambda n: np.asarray(d[n][0], dtype=np.float32)  # noqa: E731
        ys_v = slice(ys.start, ys.stop + 1)
        xs_u = slice(xs.start, xs.stop + 1)
        thm, qv = g("THM")[:, ys, xs], g("QVAPOR")[:, ys, xs]
        th_phy = (thm + np.float32(300.0)) / (np.float32(1.0) + RVRD * qv)  # phy_prep REAL form
        return dict(
            u=g("U")[:, ys, xs_u], v=g("V")[:, ys_v, xs], th_phy=th_phy, qv=qv,
            z_at_w=(g("PHB")[:, ys, xs] + g("PH")[:, ys, xs]) / G,
            pblh=g("PBLH")[ys, xs], mu_2=g("MU")[ys, xs], mub=g("MUB")[ys, xs], ht=g("HGT")[ys, xs],
            c1h=np.asarray(d["C1H"][0], np.float32), c2h=np.asarray(d["C2H"][0], np.float32),
            xland=g("XLAND")[ys, xs],
        )


def build_case(spec: dict) -> dict:
    src, ys, xs = spec["source"], spec["ys"], spec["xs"]
    files = spec["files"]
    frames = [frame(src / files[h], ys, xs) for h in spec["frame_hours"]]
    for f, h_next in zip(frames, spec["muts_hours"]):
        nxt = frame(src / files[h_next], ys, xs)
        f["muts"] = nxt["mu_2"] + nxt["mub"]  # end-of-step mass: a real later column mass
    recs = []
    for h_old, h_new in spec["record_hours"]:
        a, b = frame(src / files[h_old], ys, xs), frame(src / files[h_new], ys, xs)
        recs.append(dict(u_old=a["u"], u_new=b["u"], v_old=a["v"], v_new=b["v"],
                         t_old=a["th_phy"] - np.float32(300.0), t_new=b["th_phy"] - np.float32(300.0),
                         q_old=a["qv"], q_new=b["qv"]))
    if spec.get("pblh_scale"):
        for f in frames:
            f["pblh"] = (f["pblh"] * np.float32(spec["pblh_scale"])).astype(np.float32)
    return dict(frames=frames, records=recs, c1h=frames[0]["c1h"], c2h=frames[0]["c2h"], ht=frames[0]["ht"],
                xland=frames[0]["xland"])


def fort(a: np.ndarray) -> np.ndarray:
    """Port layout (k, j, i) -> Fortran (i, k, j) column-major bytes == numpy (j, k, i) C order."""

    if a.ndim == 3:
        return np.ascontiguousarray(np.transpose(a, (1, 0, 2)))
    return np.ascontiguousarray(a)  # (j, i) C == Fortran (i, j)


def unfort(flat: np.ndarray, shape3) -> np.ndarray:
    nz, ny, nx = shape3
    return np.transpose(flat.reshape(ny, nz, nx), (1, 0, 2))


def run_oracle(exe: Path, case: dict, opts: dict, calls: list, kind: str, work: Path) -> list:
    dt = np.float32 if kind == "r4" else np.float64
    f0 = case["frames"][0]
    nz, ny, nx = f0["th_phy"].shape
    inp, out = work / f"in_{kind}.bin", work / f"out_{kind}.bin"
    with open(inp, "wb") as fh:
        w = lambda a, t=dt: fh.write(np.asarray(a, t).tobytes())  # noqa: E731
        w([nx, ny, nz, len(calls), len(case["records"]), len(case["frames"])], np.int32)
        w([int(opts[k]) for k in INT_KEYS], np.int32)
        w([float(opts[k]) for k in REAL_KEYS])
        w(case["c1h"]); w(case["c2h"]); w(fort(case["ht"]))
        for r in case["records"]:
            for n in ("u_old", "u_new", "v_old", "v_new", "t_old", "t_new", "q_old", "q_new"):
                w(fort(r[n]))
        for f in case["frames"]:
            for n in ("u", "v", "th_phy", "qv", "z_at_w"):
                w(fort(f[n]))
            for n in ("pblh", "mu_2", "mub", "muts"):
                w(fort(f[n]))
        for c in calls:
            w([c["itimestep"], c["secs"], c["rec"], c["frame"]], np.int32)
    subprocess.run([str(exe), str(inp), str(out)], check=True)
    raw = np.fromfile(out, dtype=dt)
    su, sv, sm = (nz, ny, nx + 1), (nz, ny + 1, nx), (nz, ny, nx)
    order = [("u_old", su), ("u_new", su), ("v_old", sv), ("v_new", sv), ("t_old", sm), ("t_new", sm),
             ("q_old", sm), ("q_new", sm)]
    tend = [("run", su), ("rvn", sv), ("rth", sm), ("rqv", sm)]
    blocks = order + [(f"unc_{n}", s) for n, s in tend] + [(f"cpl_{n}", s) for n, s in tend] \
        + [(f"own_{n}", s) for n, s in tend] + [(f"dec_{n}", s) for n, s in tend]
    results, pos = [], 0
    for _ in calls:
        res = {}
        for name, shape in blocks:
            size = int(np.prod(shape))
            res[name] = unfort(raw[pos:pos + size], shape)
            pos += size
        results.append(res)
    assert pos == raw.size, (pos, raw.size)
    return results


SWISS_FILES = {h: f"wrfout_d01_2023-01-15_{h:02d}:00:00" for h in range(24)}
SWISS_FILES[24] = "wrfout_d01_2023-01-16_00:00:00"


def _prod_files():
    out = {}
    for p in sorted(PROD.glob("wrfout_d01_2026-07-2[67]_*")):
        day, hh = p.name[11:21], int(p.name[22:24])
        out[(0 if day.endswith("26") else 24) + hh] = p.name
    return out


def cases() -> list[dict]:
    pf = _prod_files()
    base = dict(guv=3.0e-4, gt=3.0e-4, gq=3.0e-4, if_no_pbl_nudging_uv=0, if_no_pbl_nudging_t=0,
                if_no_pbl_nudging_q=0, if_zfac_uv=0, k_zfac_uv=0, if_zfac_t=0, k_zfac_t=0, if_zfac_q=0,
                k_zfac_q=0, if_ramping=0, dtramp_min=0.0, end_h_sfc=0, stepfg=1, fgdtzero=0)
    return [
        dict(name="swiss_basic", r8=True, source=SWISS, files=SWISS_FILES, ys=slice(14, 22), xs=slice(16, 26),
             frame_hours=[13, 14], muts_hours=[14, 15], record_hours=[(12, 15), (15, 18)],
             opts={**base, "interval_m": 180, "end_h": 6, "specified": 1, "dt": 18.0, "dx": 3000.0},
             calls=[dict(itimestep=1, secs=0, rec=1, frame=1), dict(itimestep=2, secs=18, rec=0, frame=2),
                    dict(itimestep=301, secs=5400, rec=0, frame=1),
                    dict(itimestep=601, secs=10800, rec=2, frame=2),
                    dict(itimestep=1101, secs=19800, rec=0, frame=1)]),
        dict(name="swiss_pbl_zfac_hold", r8=True, source=SWISS, files=SWISS_FILES, ys=slice(14, 22),
             xs=slice(16, 26), frame_hours=[16, 17], muts_hours=[17, 18], record_hours=[(12, 13), (13, 14)],
             pblh_scale=1.0,
             opts={**base, "if_no_pbl_nudging_uv": 1, "if_no_pbl_nudging_t": 1, "if_no_pbl_nudging_q": 1,
                   "if_zfac_uv": 1, "k_zfac_uv": 10, "if_zfac_t": 1, "k_zfac_t": 5, "if_zfac_q": 1,
                   "k_zfac_q": 12, "interval_m": 60, "end_h": 1, "dtramp_min": 120.0, "stepfg": 10,
                   "specified": 0, "dt": 18.0, "dx": 3000.0},
             # it 1 recompute; 2-3 hold (decouple/recouple chain); 10 recompute; 201 record 2 + quirk:
             # if_ramping=0 but end_h_sfc*60+|dtramp| = 120 min keeps nudging past end_h (60 min) at tfac 1;
             # 401 (120 min) still on (xtime == 120 not > 120); 410 past both -> zeros.
             calls=[dict(itimestep=1, secs=0, rec=1, frame=1), dict(itimestep=2, secs=18, rec=0, frame=2),
                    dict(itimestep=3, secs=36, rec=0, frame=1), dict(itimestep=10, secs=162, rec=0, frame=2),
                    dict(itimestep=201, secs=3600, rec=2, frame=1), dict(itimestep=400, secs=7182, rec=0, frame=2),
                    dict(itimestep=410, secs=7362, rec=0, frame=1)]),
        dict(name="prod_day_ramp_pos", r8=False, source=PROD, files=pf, ys=slice(34, 44), xs=slice(106, 118),
             frame_hours=[15, 16], muts_hours=[16, 17], record_hours=[(11, 12), (12, 13), (13, 14)],
             opts={**base, "if_no_pbl_nudging_t": 1, "interval_m": 90, "end_h": 3, "if_ramping": 1,
                   "dtramp_min": 60.0, "specified": 1, "dt": 54.0, "dx": 9000.0},
             # dt 54 s: reads at 0/90/180 min (steps 1/101/201).  45 min (coef .5), 149.4 min (before ramp),
             # 180 (ramp start: record 3, coef forced 0, tfac 1), 209.7 (tfac ~.5), 239.4 (tfac ~.01),
             # 240.3 (past actual_end -> early return zeros)
             calls=[dict(itimestep=1, secs=0, rec=1, frame=1), dict(itimestep=51, secs=2700, rec=0, frame=2),
                    dict(itimestep=101, secs=5400, rec=2, frame=1), dict(itimestep=167, secs=8964, rec=0, frame=2),
                    dict(itimestep=201, secs=10800, rec=3, frame=1), dict(itimestep=234, secs=12582, rec=0, frame=2),
                    dict(itimestep=267, secs=14364, rec=0, frame=1), dict(itimestep=268, secs=14418, rec=0, frame=2)]),
        dict(name="prod_night_ramp_neg_fgdtzero", r8=False, source=PROD, files=pf, ys=slice(34, 44),
             xs=slice(106, 118), frame_hours=[30, 31], muts_hours=[31, 32], record_hours=[(26, 27), (27, 28), (28, 29)],
             opts={**base, "if_no_pbl_nudging_uv": 1, "if_no_pbl_nudging_q": 1, "interval_m": 90, "end_h": 3,
                   "if_ramping": 1, "dtramp_min": -60.0, "stepfg": 5, "fgdtzero": 1, "specified": 0,
                   "dt": 54.0, "dx": 9000.0},
             # it 1 recompute, 2 hold, 5 recompute, 6 fgdtzero zero, 10 recompute, 101 (90 min, record 2, hold),
             # 145 (129.6 min, ramp, recompute), 200 (179.1 min, tfac ~.015), 201 (180 min, record 3, zeroed),
             # 215 (191.4 min, recompute -> early return zeros)
             calls=[dict(itimestep=1, secs=0, rec=1, frame=1), dict(itimestep=2, secs=54, rec=0, frame=2),
                    dict(itimestep=5, secs=216, rec=0, frame=1), dict(itimestep=6, secs=270, rec=0, frame=2),
                    dict(itimestep=10, secs=486, rec=0, frame=1), dict(itimestep=101, secs=5400, rec=2, frame=2),
                    dict(itimestep=145, secs=7776, rec=0, frame=1), dict(itimestep=200, secs=10746, rec=0, frame=2),
                    dict(itimestep=201, secs=10800, rec=3, frame=1), dict(itimestep=215, secs=11556, rec=0, frame=2)]),
    ]


def census(case: dict, opts: dict) -> dict:
    """E154: non-zero/active input census (kpbl spread, land, PBLH range)."""

    out = {}
    for i, f in enumerate(case["frames"]):
        zagl = f["z_at_w"] - f["ht"][None]
        hit = (f["pblh"][None] >= zagl[:-1]) & (f["pblh"][None] < zagl[1:])
        kpbl = np.where(hit.any(0), hit.argmax(0) + 1, 1)
        out[f"frame{i + 1}"] = dict(kpbl_min=int(kpbl.min()), kpbl_max=int(kpbl.max()),
                                    kpbl_distinct=int(np.unique(kpbl).size),
                                    pblh_min=float(f["pblh"].min()), pblh_max=float(f["pblh"].max()))
    out["land_fraction"] = float((case["xland"] < 1.5).mean())
    out["ht_max"] = float(case["ht"].max())
    out["max_abs_u_minus_rec"] = float(np.abs(case["records"][0]["u_new"] - case["frames"][0]["u"]).max())
    out["max_abs_t_minus_rec"] = float(np.abs(case["records"][0]["t_new"] + 300 - case["frames"][0]["th_phy"]).max())
    return out


def main() -> None:
    build, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    work = build / "work"
    work.mkdir(exist_ok=True)
    manifest = {"pristine": (build / "pristine_sha256.txt").read_text(), "cases": {}}
    for spec in cases():
        case = build_case(spec)
        kinds = ("r4", "r8") if spec["r8"] else ("r4",)
        for kind in kinds:
            dt = np.float32 if kind == "r4" else np.float64
            results = run_oracle(build / kind / "fdda_oracle", case, spec["opts"], spec["calls"], kind, work)
            arrays = {"c1h": case["c1h"].astype(dt), "c2h": case["c2h"].astype(dt), "ht": case["ht"].astype(dt)}
            for r, rec in enumerate(case["records"]):
                for n, a in rec.items():
                    arrays[f"rec{r + 1}_{n}"] = a.astype(dt)
            for i, f in enumerate(case["frames"]):
                for n in ("u", "v", "th_phy", "qv", "z_at_w", "pblh", "mu_2", "mub", "muts"):
                    arrays[f"frame{i + 1}_{n}"] = f[n].astype(dt)
            for c, res in enumerate(results):
                for n, a in res.items():
                    if n.startswith(("unc_", "cpl_", "dec_")) or (n.startswith("own_") and c == 0) \
                            or (c == 0 and n in ("u_old", "t_old", "q_old", "v_old")):
                        arrays[f"call{c}_{n}"] = a
            arrays["calls"] = np.asarray([[c["itimestep"], c["secs"], c["rec"], c["frame"]] for c in spec["calls"]],
                                         np.int64)
            path = outdir / f"{spec['name']}_{kind}.npz"
            np.savez_compressed(path, **arrays)
            manifest["cases"][path.name] = dict(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), opts=spec["opts"], calls=spec["calls"],
                frame_hours=spec["frame_hours"], record_hours=spec["record_hours"],
                source=str(spec["source"]), ys=[spec["ys"].start, spec["ys"].stop],
                xs=[spec["xs"].start, spec["xs"].stop], census=census(case, spec["opts"]),
                nonzero_unc=[{k[4:]: int(np.count_nonzero(v)) for k, v in res.items() if k.startswith("unc_")}
                             for res in results])
            print(path.name, path.stat().st_size)
    (outdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))


if __name__ == "__main__":
    main()
