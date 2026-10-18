"""Prepare the GPU OSSE smoke of `gpuwrf optimize` continuous mode (lane w41-optcont): Swiss d01, SYNTHETIC dates.

Two cases are the original Swiss d01 input (2023-01-15 00 UTC, local wrf_gpu2 archive, no ALISIOS data) and a byte
copy relabelled to 2023-01-16 (proofs/w41_optimize/cpu_dryrun.relabel: every WRF date string shifted by whole days)
as the held-out date. The held-out case therefore has the SAME initial state as the training case (only the date
differs): a workflow/recovery smoke, never a skill test. Synthetic stations = every 3rd interior grid node (land).

Truth (twin): z0mvt x1.6 for the forest classes (MODIS 1/4/5: 928 cells) and x0.7 for open shrubland (7: 782 cells);
EKI starts from the table prior (multiplier 1) and sees only the synthetic observations (truth model + seeded noise).

  python proofs/w41_optcont/prepare_osse.py ROOT        (CPU, seconds; then `gpuwrf surface-params basis`)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(REPO/"proofs"/"w41_optimize"))
from cpu_dryrun import SWISS, relabel, sha  # noqa: E402

GATES = json.loads((Path(__file__).parent/"REGISTERED_GATES.json").read_text())


def prepare(root):
    from gpuwrf.io.netcdf_lock import Dataset

    root = Path(root)
    if (root/"study.json").exists() or (root/"cases").exists():
        raise SystemExit(f"{root} already holds a prepared study")
    root.mkdir(parents=True, exist_ok=True)
    namelist = (SWISS/"input"/"namelist.input").read_text()
    cases = []
    for day, split in ((15, "train"), (16, "test")):
        case = root/"cases"/f"2023-01-{day:02d}"
        case.mkdir(parents=True)
        for name in ("wrfinput_d01", "wrfbdy_d01"):
            if day == 15:
                shutil.copyfile(SWISS/"input"/name, case/name)
            else:
                relabel(SWISS/"input"/name, case/name, day)
        text = namelist.replace("start_day                           = 15", f"start_day                           = {day}")
        text = text.replace("end_day                             = 16", f"end_day                             = {day+1}")
        (case/"namelist.input").write_text(text)
        cases.append({"id": f"d{day}", "date": f"2023-01-{day:02d}", "input_dir": str(case.relative_to(root)),
                      "domain": "d01"})
    with Dataset(SWISS/"input"/"wrfinput_d01") as ds:
        lat, lon, hgt = (np.asarray(ds.variables[v][0]) for v in ("XLAT", "XLONG", "HGT"))
        veg = np.asarray(ds.variables["IVGTYP"][0])
    lines = ["station_uid,lat,lon,elevation_m"]
    for j in range(2, lat.shape[0]-2, 3):
        for i in range(2, lat.shape[1]-2, 3):
            lines.append(f"n{j:02d}{i:02d},{float(lat[j, i])!r},{float(lon[j, i])!r},{float(hgt[j, i])!r}")
    (root/"stations.csv").write_text("\n".join(lines)+"\n")
    osse = GATES["osse"]
    study = {"schema": "gpuwrf.optimize.continuous.v1", "region": "swiss-d01-OSSE-SYNTHETIC-DATES", "hours": osse["hours"],
             "budget_case_hours": osse["budget_case_hours"], "test_start": "2023-01-16", "configuration": "c00",
             "basis": "basis.npz", "stations": "stations.csv", "cases": cases,
             "parameters": [{"name": "z0_forest", "control": "z0mvt", "classes": [1, 4, 5]},
                            {"name": "z0_shrub", "control": "z0mvt", "classes": [7]}],
             "eki": osse["eki"],
             "scorer": {"kind": "twin", "variables": ["t2", "wind"], "sigma": osse["sigma"], "truth": osse["truth"],
                        "noise_seed": osse["noise_seed"], "tolerance_log": osse["tolerance_log"]},
             "sampling": {"method": "nearest", "land_aware": False}}
    (root/"study.json").write_text(json.dumps(study, indent=1)+"\n")
    counts = {int(k): int(v) for k, v in zip(*np.unique(veg, return_counts=True))}
    receipt = {"scope": "SYNTHETIC dates (2023-01-16 = relabelled copy of the 2023-01-15 Swiss d01 input); OSSE smoke",
               "source_inputs": {n: sha(SWISS/"input"/n) for n in ("namelist.input", "wrfinput_d01", "wrfbdy_d01")},
               "cases": {c["id"]: {n: sha(root/c["input_dir"]/n) for n in ("namelist.input", "wrfinput_d01", "wrfbdy_d01")}
                         for c in cases},
               "ivgtyp_counts": counts, "stations": len(lines)-1, "gates": GATES}
    (root/"PREPARE_RECEIPT.json").write_text(json.dumps(receipt, indent=1)+"\n")
    print(json.dumps({"root": str(root), "stations": len(lines)-1, "ivgtyp_counts": counts}))


if __name__ == "__main__":
    prepare(sys.argv[1])
