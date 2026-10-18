"""CPU end-to-end rehearsal of `gpuwrf optimize run` on Swiss d01: 2 released configurations x 2 rungs.

SYNTHETIC DATES, WORKFLOW ONLY: five cases are byte copies of the original Swiss d01 input
(2023-01-15 00 UTC) whose date strings are relabelled to 2023-01-15..19 (wrfinput/wrfbdy Times,
bdy time stamps, START_DATE/SIMULATION_START_DATE/JULDAY, namelist start/end day). Pseudo-
observations are T2 of our own original CPU-WRF Swiss run at 01 UTC on four grid nodes,
relabelled the same way. Nothing here is regional skill; it rehearses preparation, launcher
batches, program reuse, rung receipts, pause/resume and the held-out step on real model runs.

  prepare: python proofs/w41_optimize/cpu_dryrun.py prepare ROOT [HOURS=1]
  then:    gpuwrf optimize plan/run with ROOT/study.json (see run_cpu_dryrun.sh)
  verify:  python proofs/w41_optimize/cpu_dryrun.py verify ROOT OUT_RECEIPT
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"src"))
SWISS = Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss")
DAYS = (15, 16, 17, 18, 19)          # 15-17 train, 18-19 held out
NODES = ((10, 10), (12, 30), (30, 12), (30, 30))  # (south_north, west_east) interior nodes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


DATE = re.compile(rb"\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}")


def shift(text, days):
    """Shift every WRF date string (YYYY-MM-DD_hh:mm:ss) by whole days."""
    def one(match):
        when = datetime.strptime(match.group().decode(), "%Y-%m-%d_%H:%M:%S")+timedelta(days=days)
        return when.strftime("%Y-%m-%d_%H:%M:%S").encode()
    return DATE.sub(one, text)


def relabel(src, dst, day):
    from gpuwrf.io.netcdf_lock import Dataset

    shutil.copyfile(src, dst)
    days = day-15
    with Dataset(dst, "r+") as ds:
        for name, var in ds.variables.items():
            if var.dtype == np.dtype("S1"):
                raw = var[:]
                var[:] = np.frombuffer(shift(raw.tobytes(), days), dtype="S1").reshape(raw.shape)
        for attr in ("START_DATE", "SIMULATION_START_DATE"):
            if attr in ds.ncattrs():
                ds.setncattr(attr, shift(ds.getncattr(attr).encode(), days).decode())
        if "JULDAY" in ds.ncattrs():
            ds.setncattr("JULDAY", np.int32(day))


def reprepare(root):
    """Regenerate the relabelled inputs of an existing root (namelists, stations and observations unchanged)."""
    root = Path(root)
    for day in DAYS:
        case = root/"cases"/f"2023-01-{day:02d}"
        for name in ("wrfinput_d01", "wrfbdy_d01"):
            relabel(SWISS/"input"/name, case/name, day)
    receipt = json.loads((root/"PREPARE_RECEIPT.json").read_text())
    receipt["reprepared"] = {"reason": ("attempt 1: string relabel left the last wrfbdy next-boundary time unshifted "
                                        "(day16+ coverage 0 h; the CLI boundary-coverage check refused member d16); "
                                        "every WRF date string is now shifted by whole days"),
                             "cases": {f"d{day}": {n: sha(root/"cases"/f"2023-01-{day:02d}"/n)
                                                   for n in ("wrfinput_d01", "wrfbdy_d01")} for day in DAYS}}
    (root/"PREPARE_RECEIPT.json").write_text(json.dumps(receipt, indent=1)+"\n")
    print(json.dumps({"reprepared": str(root)}))


def prepare(root, hours="1"):
    from gpuwrf.io.netcdf_lock import Dataset

    hours = int(hours)
    root = Path(root)
    if (root/"study.json").exists() or (root/"cases").exists():
        raise SystemExit(f"{root} already holds a prepared study")
    root.mkdir(parents=True, exist_ok=True)  # importing gpuwrf may already have created ROOT/cache
    namelist = (SWISS/"input"/"namelist.input").read_text()
    cases = []
    for day in DAYS:
        case = root/"cases"/f"2023-01-{day:02d}"
        case.mkdir(parents=True)
        for name in ("wrfinput_d01", "wrfbdy_d01"):
            relabel(SWISS/"input"/name, case/name, day)
        text = namelist.replace("start_day                           = 15", f"start_day                           = {day}")
        text = text.replace("end_day                             = 16", f"end_day                             = {day+1}")
        (case/"namelist.input").write_text(text)
        cases.append({"id": f"d{day}", "date": f"2023-01-{day:02d}", "input_dir": str(case.relative_to(root)),
                      "domain": "d01"})
    ref = SWISS/"cpu"/"wrfout_d01_2023-01-15_01:00:00"
    with Dataset(ref) as ds:
        lat, lon, hgt, t2 = (np.asarray(ds.variables[v][0]) for v in ("XLAT", "XLONG", "HGT", "T2"))
    stations, observations = ["station_uid,lat,lon,elevation_m"], ["station_uid,valid_time,variable,value"]
    for n, (j, i) in enumerate(NODES):
        stations.append(f"s{n},{float(lat[j, i])!r},{float(lon[j, i])!r},{float(hgt[j, i])!r}")
        for day in DAYS:
            observations.append(f"s{n},2023-01-{day:02d}T01:00:00Z,t2,{float(t2[j, i])-273.15!r}")
    (root/"stations.csv").write_text("\n".join(stations)+"\n")
    (root/"observations.csv").write_text("\n".join(observations)+"\n")
    study = {"schema": "gpuwrf.optimize.study.v1", "region": "swiss-d01-SYNTHETIC-DATES", "objective": "T2",
             "hours": hours, "budget_case_hours": 8*hours, "test_start": "2023-01-18",
             "stations": "stations.csv", "observations": "observations.csv",
             "configurations": ["c00", "c36"], "e41": "PASS",
             "rungs": [{"train_dates": 2, "keep": 2}, {"train_dates": 3, "keep": 1}],
             "cases": cases, "sampling": {"method": "nearest", "land_aware": False}}
    (root/"study.json").write_text(json.dumps(study, indent=1)+"\n")
    receipt = {"scope": "SYNTHETIC dates (relabelled Swiss 2023-01-15 copies); workflow rehearsal only",
               "source_inputs": {n: sha(SWISS/"input"/n) for n in ("namelist.input", "wrfinput_d01", "wrfbdy_d01")},
               "pseudo_observation_source": {"path": str(ref), "sha256": sha(ref), "variable": "T2", "nodes": NODES},
               "cases": {c["id"]: {n: sha(root/c["input_dir"]/n) for n in ("namelist.input", "wrfinput_d01", "wrfbdy_d01")}
                         for c in cases}}
    (root/"PREPARE_RECEIPT.json").write_text(json.dumps(receipt, indent=1)+"\n")
    print(json.dumps({"root": str(root), "cases": len(cases)}))


def _member_dirs(run):
    for record in json.loads((run/"ledger.json").read_text())["batches"]:
        plan = json.loads((run/record["batch"]/"plan.json").read_text())
        for member in plan["members"]:
            yield Path(plan["out_root"])/member["id"]


def verify(root, out):
    root = Path(root)
    run = root/"run"
    result = json.loads((run/"result.json").read_text())
    ledger = json.loads((run/"ledger.json").read_text())
    rungs = [json.loads(p.read_text()) for p in sorted((run/"rungs").glob("rung-*.json"))]
    batches = []
    for record in ledger["batches"]:
        plan = json.loads((run/record["batch"]/"plan.json").read_text())
        members = []
        for member in plan["members"]:
            rec = json.loads((Path(plan["out_root"])/member["id"]/"receipt.json").read_text())
            members.append({k: rec.get(k) for k in ("name", "rc", "wall_s", "vmhwm_kb", "aot_cache_warm",
                                                    "aot_cold_executables", "concurrent_at_start", "note")})
        batches.append({"batch": record["batch"], "rc": record.get("rc"), "note": record.get("note"),
                        "wall_s": record.get("wall_s"), "pool_gib": record.get("pool_gib"),
                        "launcher_options": plan["launcher_options"], "members": members})
    summary = {"scope": "CPU (JAX_PLATFORMS=cpu) rehearsal on SYNTHETIC relabelled dates; workflow only, no skill claim",
               "recommended": result["recommended"]["id"], "rounds": [{k: r[k] for k in ("round", "dates", "survivors", "keep")}
                                                                       | {"scores": [(s["id"], s["loss"]) for s in r["scores"]]}
                                                                       for r in result["rounds"]],
               "held_out": {k: result["held_out"][k] for k in ("loss", "metrics")},
               "scheduled_case_hours": result["scheduled_case_hours"], "planned_case_hours": result["planned_case_hours"],
               "rung_receipts": [{"round": r["round"], "members": r["members"]} for r in rungs],
               "batches": batches, "progress": json.loads((run/"progress.json").read_text()),
               "result_sha256": sha(run/"result.json"), "ledger_sha256": sha(run/"ledger.json"),
               "member_sources": sorted({json.dumps(json.loads((Path(plan_out)/"receipt.json").read_text()).get("source"),
                                                    sort_keys=True) for plan_out in _member_dirs(run)})}
    Path(out).write_text(json.dumps(summary, indent=1)+"\n")
    print(json.dumps({k: summary[k] for k in ("recommended", "scheduled_case_hours", "planned_case_hours")}))


if __name__ == "__main__":
    {"prepare": prepare, "reprepare": reprepare, "verify": verify}[sys.argv[1]](*sys.argv[2:])
