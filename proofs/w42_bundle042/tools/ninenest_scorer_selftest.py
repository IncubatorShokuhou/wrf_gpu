"""w42-bundle042: CPU self-test of the merged scripts/wn3_score.py on a synthetic 9-nest case (d01 .. d09).

usage: python3 ninenest_scorer_selftest.py TREE WORK OUT_JSON [HOURS]
  TREE  = immutable copy of main (git archive of scripts/wn3_score.py, wn3_fast_compare.py, compare_wrfout_grid.py and
          proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json); the scorer is run unchanged as a CLI
  WORK  = empty scratch dir for the synthetic CPU/GPU wrfouts and scorer outputs (delete after)
  HOURS = scored hours (default 24 = 25 frames per domain)
Synthetic files (no ALISIOS data): the W9N/WNN9 horizontal mass shapes from the w41-v033impact namelist geometry
(e_we 121,268,112,58,55,58,73,94,73 / e_sn 71,118,94,73,52,55,79,112,73), 4 levels, the 14 D6-mandatory fields
(T U V W QVAPOR P PH MU U10 V10 T2 PSFC RAINC RAINNC) + HGT LANDMASK XLAT XLONG, the WRF globals wn3_score requires;
GPU = CPU + small noise, statics exact. Cases (real CLI, --domains d01 .. d09, default domain scheduling):
  clean    -> PASS on 9/9 domains, domain_mode sequential, never more than one compare child alive
  t2_d07   -> d07 GPU T2 + 3 K (RMSE 3 > 1.5 K): d07 fails (T2 frame_tolerance), the other 8 pass
  miss_d09 -> one d09 GPU frame missing: refused before the first compare starts (no domain JSON written)
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from netCDF4 import Dataset

TREE, WORK, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
HOURS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
E_WE = (121, 268, 112, 58, 55, 58, 73, 94, 73)
E_SN = (71, 118, 94, 73, 52, 55, 79, 112, 73)
DOMAINS = [f"d{i:02d}" for i in range(1, 10)]
NZ = 4
GLOBALS = {"PARENT_ID": 1, "I_PARENT_START": 1, "J_PARENT_START": 1, "PARENT_GRID_RATIO": 3, "BUCKET_MM": -1.0,
           "BUCKET_J": -1.0, "DT": 6.0, "MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "NUM_LAND_CAT": 21}


def write_frame(path: Path, t: str, d: int, h: int, rng, gpu: bool, t2_offset: float = 0.0) -> None:
    ny, nx = E_SN[d] - 1, E_WE[d] - 1
    yy, xx = np.meshgrid(np.arange(ny, dtype="f4"), np.arange(nx, dtype="f4"), indexing="ij")
    base = np.sin(yy / 7.0 + d) * np.cos(xx / 9.0) + 0.05 * h  # varies in space and from frame to frame

    def noise(shape, scale):
        return rng.standard_normal(shape).astype("f4") * scale if gpu else np.zeros(shape, "f4")

    with Dataset(path, "w", format="NETCDF4") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("bottom_top", NZ), ("bottom_top_stag", NZ + 1),
                           ("south_north", ny), ("south_north_stag", ny + 1), ("west_east", nx),
                           ("west_east_stag", nx + 1)):
            ds.createDimension(name, size)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0, :] = [c.encode() for c in t]
        mass2 = ("Time", "south_north", "west_east")
        statics = {"HGT": yy * 20.0 + xx * 10.0, "LANDMASK": (xx > nx / 2).astype("f4"),
                   "XLAT": 27.5 + yy * 0.01, "XLONG": -17.0 + xx * 0.01}
        for name, values in statics.items():
            ds.createVariable(name, "f4", mass2)[0] = values
        two_d = {"T2": 290.0 + 3.0 * base + t2_offset, "U10": 4.0 * base, "V10": -3.0 * base,
                 "PSFC": 100000.0 + 300.0 * base, "MU": 2000.0 * base, "RAINC": 0.1 * h + 0.01 * (base + 2.0),
                 "RAINNC": 0.2 * h + 0.02 * (base + 2.0)}
        scale = {"T2": 0.05, "U10": 0.05, "V10": 0.05, "PSFC": 2.0, "MU": 2.0, "RAINC": 0.001, "RAINNC": 0.001}
        for name, values in two_d.items():
            ds.createVariable(name, "f4", mass2)[0] = values + noise(values.shape, scale[name])
        lev = np.arange(NZ, dtype="f4")[:, None, None]
        three_d = {  # name: (dims, field, noise scale)
            "T": (("Time", "bottom_top", "south_north", "west_east"), 5.0 * base[None] + lev, 0.02),
            "QVAPOR": (("Time", "bottom_top", "south_north", "west_east"), 1e-3 * (base[None] + 2.0) / (1.0 + lev), 1e-6),
            "P": (("Time", "bottom_top", "south_north", "west_east"), 100.0 * base[None] - 10.0 * lev, 1.0),
            "U": (("Time", "bottom_top", "south_north", "west_east_stag"),
                  np.pad(4.0 * base, ((0, 0), (0, 1)), mode="edge")[None] + lev, 0.05),
            "V": (("Time", "bottom_top", "south_north_stag", "west_east"),
                  np.pad(-3.0 * base, ((0, 1), (0, 0)), mode="edge")[None] - lev, 0.05),
            "W": (("Time", "bottom_top_stag", "south_north", "west_east"),
                  0.1 * base[None] * np.arange(NZ + 1, dtype="f4")[:, None, None], 0.005),
            "PH": (("Time", "bottom_top_stag", "south_north", "west_east"),
                   50.0 * base[None] + 10.0 * np.arange(NZ + 1, dtype="f4")[:, None, None], 0.5),
        }
        for name, (dims, values, sc) in three_d.items():
            values = np.asarray(values, dtype="f4")
            ds.createVariable(name, "f4", dims)[0] = values + noise(values.shape, sc)
        for key, value in GLOBALS.items():
            ds.setncattr(key, value)


def build(case: str) -> tuple[Path, Path]:
    root = WORK / case
    cpu, gpu = root / "cpu", root / "gpu"
    cpu.mkdir(parents=True)
    gpu.mkdir()
    t0 = datetime(2026, 10, 19, 0)
    for d, domain in enumerate(DOMAINS):
        rng = np.random.default_rng(100 + d)
        for h in range(HOURS + 1):
            t = (t0 + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S")
            write_frame(cpu / f"wrfout_{domain}_{t}", t, d, h, rng, gpu=False)
            if case == "miss_d09" and domain == "d09" and h == HOURS // 2:
                continue
            write_frame(gpu / f"wrfout_{domain}_{t}", t, d, h, rng, gpu=True,
                        t2_offset=3.0 if (case == "t2_d07" and domain == "d07") else 0.0)
    return cpu, gpu


def children(pid: int) -> list[int]:
    try:
        return [int(c) for c in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except OSError:
        return []


def is_compare(pid: int) -> bool:
    try:
        return b"wn3_fast_compare.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def run(case: str) -> dict:
    t_build = time.monotonic()
    cpu, gpu = build(case)
    build_s = time.monotonic() - t_build
    out = WORK / case / "score"
    cmd = [sys.executable, str(TREE / "scripts/wn3_score.py"), "--cpu-dir", str(cpu), "--gpu-dir", str(gpu),
           "--out", str(out), "--hours", str(HOURS), "--domains", *DOMAINS]
    log = (WORK / case / "score.log").open("w")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    max_alive, seen = 0, []
    while proc.poll() is None:
        alive = [c for c in children(proc.pid) if is_compare(c)]
        max_alive = max(max_alive, len(alive))
        for c in alive:
            if c not in seen:
                seen.append(c)
        time.sleep(0.05)
    log.close()
    rec = {"case": case, "rc": proc.returncode, "wall_s": round(time.monotonic() - t0, 1), "build_s": round(build_s, 1),
           "compare_children_seen": len(seen), "max_compare_children_alive": max_alive,
           "log_tail": (WORK / case / "score.log").read_text().splitlines()[-12:]}
    summary, resources = out / "summary.json", out / "resources.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        rec["summary_pass"] = s["pass"]
        rec["domain_pass"] = {d: v["pass"] for d, v in s["domains"].items()}
        rec["output_integrity_pass"] = s.get("output_integrity_pass")
        rec["policy_failures"] = {d: sorted({(f.get("field"), f.get("kind")) for f in v["policy_failures"]})
                                  for d, v in s["domains"].items() if v["policy_failures"]}
        rec["paired_files"] = {d: v["pairing"]["paired_file_count"] for d, v in s["domains"].items()}
    if resources.exists():
        r = json.loads(resources.read_text())
        rec["domain_mode"] = r["domain_mode"]
        rec["compares"] = r["domain_compares"]
        rec["peak_rss_bound_mib"] = r["peak_rss_bound_mib"]
        rec["scorer_process_peak_rss_mib"] = r["scorer_process_peak_rss_mib"]
    rec["domain_json_written"] = sorted(p.stem for p in out.glob("d0?.json")) if out.exists() else []
    return rec


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=False)
    results = {c: run(c) for c in ("clean", "t2_d07", "miss_d09")}
    clean, mut, miss = results["clean"], results["t2_d07"], results["miss_d09"]
    checks = {
        "clean_rc0_pass_9of9": clean["rc"] == 0 and clean.get("summary_pass") is True
                               and clean.get("domain_pass") == {d: True for d in DOMAINS},
        "clean_integrity_pass": clean.get("output_integrity_pass") is True,
        "clean_paired_all_frames": clean.get("paired_files") == {d: HOURS + 1 for d in DOMAINS},
        "clean_sequential": clean.get("domain_mode") == "sequential" and clean["max_compare_children_alive"] <= 1
                            and clean["compare_children_seen"] == 9,
        "t2_d07_only_d07_fails": mut.get("summary_pass") is False
                                 and mut.get("domain_pass") == {d: d != "d07" for d in DOMAINS}
                                 and set(mut.get("policy_failures", {})) == {"d07"}
                                 and any(f == "T2" for f, _ in mut["policy_failures"]["d07"]),
        "t2_d07_sequential": mut.get("domain_mode") == "sequential" and mut["max_compare_children_alive"] <= 1,
        "miss_d09_refused_before_compares": miss["rc"] != 0 and miss["compare_children_seen"] == 0
                                            and miss["domain_json_written"] == []
                                            and any("d09" in line and "require all" in line for line in miss["log_tail"]),
    }
    rec = {"schema": "w42-bundle042.ninenest_scorer_selftest.v1", "tree": str(TREE), "hours": HOURS,
           "shapes_mass_cells": {d: [E_WE[i] - 1, E_SN[i] - 1] for i, d in enumerate(DOMAINS)}, "levels": NZ,
           "checks": checks, "pass": all(checks.values()), "cases": results}
    OUT.write_text(json.dumps(rec, indent=1, default=list) + "\n")
    print(json.dumps({"pass": rec["pass"], "checks": checks,
                      "wall_s": {c: r["wall_s"] for c, r in results.items()},
                      "peak_rss_bound_mib": {c: r.get("peak_rss_bound_mib") for c, r in results.items()}}, indent=1))
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
