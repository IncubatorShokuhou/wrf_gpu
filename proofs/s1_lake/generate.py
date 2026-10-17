"""Build a private pristine Lake/lakeini oracle and freeze small column fixtures.

Run with JAX_PLATFORMS=cpu on the lane core, after the shared QUIET check.
No WRF executable, foreign object/module file, JAX backend, or reference write.
The only compilation directory is --out/build, removed after the last arm.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from variable_schema import variables

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    base = dict(veg=17, soil=8, ht=100.0, snow=0.0, skin=290.0, mask=1.0,
                xice=0.0, xland=2.0, depth=10.0, maskflag=1, depthflag=1,
                use_depth=1, default_depth=50.0, min_elev=5.0, nsteps=3,
                ta=285.0, psfc=95000.0, pbot=94500.0, dzair=40.0, qv=0.006,
                u=4.0, v=1.0, lw=300.0, emiss=0.98, rain=0.0, dt=60.0,
                sw=500.0, alb=0.08, lat=47.0)
    specs = [
        ("open_shallow", dict(depth=1.0)),
        ("open_deep", dict(depth=100.0)),
        ("ice_shallow", dict(depth=1.0, skin=260.0, ta=258.0, sw=0.0)),
        ("freeze", dict(depth=1.0, skin=273.17, ta=250.0, lw=150.0, sw=0.0, nsteps=120)),
        ("snowfall", dict(depth=1.0, skin=265.0, ta=260.0, sw=0.0, rain=0.6)),
        ("rain", dict(rain=1.2)),
        ("nonlake", dict(veg=7, mask=0.0, nsteps=0)),
        ("fractional_mask", dict(mask=0.5, nsteps=0)),
        ("elevation_select", dict(maskflag=0, mask=0.0)),
        ("below_elevation", dict(maskflag=0, ht=4.0, nsteps=0)),
        ("elevation_seaice", dict(maskflag=0, veg=24, xice=0.8, xland=1.0, skin=260.0)),
        ("mask_seaice", dict(veg=24, xice=0.8, xland=1.0, skin=260.0)),
        ("depth_fallback", dict(depth=-1.0)),
        ("depth_disabled", dict(use_depth=0, depth=8.0)),
        ("depth_absent", dict(use_depth=0, depthflag=0)),
        ("standard_depth", dict(use_depth=0, default_depth=-1.0)),
    ]
    # Each source branch, plus boundary-adjacent REAL32 snow depths. Snow mass
    # is independent forcing; depth in lakeini is REAL32 snow*0.005.
    for idx, mass in enumerate((0.0, 1.0, 2.0, 4.0, 6.0, 7.0, 8.0, 11.0, 14.0,
                                20.0, 24.0, 30.0, 36.0, 46.0, 58.0, 70.0,
                                82.0, 100.0, 128.0, 150.0)):
        specs.append((f"snow_{idx:02d}", dict(snow=mass, skin=260.0, ta=258.0,
                                               sw=0.0, depth=1.0)))
    for soil in range(1, 20):
        specs.append((f"soil_{soil:02d}", dict(soil=soil, nsteps=0)))
    return [dict(base, **changes, name=name) for name, changes in specs]


def stdin(case):
    groups = (("veg", "soil", "ht", "snow", "skin", "mask", "xice", "xland", "depth"),
              ("maskflag", "depthflag", "use_depth", "default_depth", "min_elev", "nsteps"),
              ("ta", "psfc", "pbot", "dzair", "qv", "u", "v", "lw", "emiss", "rain",
               "dt", "sw", "alb", "lat"))
    return "\n".join(" ".join(str(case[key]) for key in group) for group in groups) + "\n"


def parse(stdout):
    records, qsat = [], []
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "STEP":
            records.append({"step": int(parts[1])})
        elif parts[0] == "QSAT":
            qsat.append([float(x) for x in parts[1:]])
        else:
            records[-1][parts[0]] = np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return records, np.array(qsat, dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrf", type=Path, default=Path("<USER_HOME>/src/wrf_pristine/WRF"))
    p.add_argument("--compiler", default="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    build = args.out / "build"
    build.mkdir(exist_ok=False)
    module = args.wrf / "phys/module_sf_lake.F"
    (build / module.name).write_bytes(module.read_bytes())
    ff = [args.compiler, "-O0", "-ffree-form", "-ffree-line-length-none", "-cpp",
          "-DEM_CORE=1", "-fcheck=bounds", "-ffp-contract=off"]
    command = ff + [str(HERE / "wrf_stubs.F90"), module.name,
                    str(HERE / "oracle_driver.F90"), "-o", "lake_oracle"]
    report = {"fixture_id": "s1-lake-wrf-real4-r8-columns-v1",
              "source": str(module), "source_commit": "source-sha256:" + sha(module),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha(module),
              "driver_sha256": sha(HERE / "oracle_driver.F90"),
              "stubs_sha256": sha(HERE / "wrf_stubs.F90"),
              "caller_sha256": {name: sha(args.wrf / "phys" / name) for name in
                                ("module_surface_driver.F", "module_physics_init.F")},
              "constants_sha256": sha(args.wrf / "share/module_model_constants.F"),
              "registry_sha256": sha(args.wrf / "Registry/registry.lake"),
              "build_command": command,
              "compiler": subprocess.check_output([args.compiler, "--version"], text=True).splitlines()[0],
              "scenario": "Constructed column forcing; full pristine Lake wrapper after lakeini. NOT a coupled forecast.",
              "precision": "Caller/carry REAL32, explicit WRF r8 DOUBLE internally, no real-kind override.",
              "license_notes": "WRF-derived numerical fixture; unmodified source remains external.",
              "cases": [], "failures": []}
    t0 = time.monotonic()
    try:
        compile_result = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=120)
        (args.out / "build.log").write_text(compile_result.stdout + compile_result.stderr)
        report["compile_rc"] = compile_result.returncode
        if compile_result.returncode:
            raise RuntimeError("Oracle build failed; see build.log")
        report["compile_s"] = time.monotonic() - t0
        payload = {}
        for case in cases():
            result = subprocess.run([str(build / "lake_oracle")], input=stdin(case), cwd=build,
                                    capture_output=True, text=True, timeout=60)
            (args.out / (case["name"] + ".log")).write_text(result.stderr)
            if result.returncode:
                report["failures"].append({"name": case["name"], "rc": result.returncode,
                                           "stderr": result.stderr[-2000:]})
                continue
            records, qsat = parse(result.stdout)
            if len(records) != case["nsteps"] + 1:
                raise ValueError("Missing oracle record")
            finite = all(np.isfinite(v).all() for r in records for k, v in r.items() if k != "step")
            if not finite or not np.isfinite(qsat).all():
                raise ValueError("Nonfinite pristine output: " + case["name"])
            census = {"initial_snl": float(records[0]["snl2d"][0]),
                      "initial_max_ice": float(records[0]["lake_icefrac3d"].max()),
                      "final_max_ice": float(records[-1]["lake_icefrac3d"].max()),
                      "ice_formation": any(np.any(r["lake_icefrac3d"] > records[0]["lake_icefrac3d"])
                                           for r in records[1:]),
                      "initial_snow": float(records[0]["h2osno2d"][0]),
                      "final_snow": float(records[-1]["h2osno2d"][0])}
            report["cases"].append(dict(case, census=census))
            # Keep init, first and last steps; census covers every time step.
            for r in (records[0], *records[1:2], *(records[-1:] if len(records) > 2 else ())):
                for field, value in r.items():
                    if field != "step":
                        payload[f'{case["name"]}/step{r["step"]}/{field}'] = value
            payload["qsat"] = qsat
        np.savez_compressed(args.out / "columns.npz", **payload)
        report["files"] = [{"path": "columns.npz", "sha256": sha(args.out / "columns.npz")}]
        report["variables"] = variables()
        report["tolerances"] = {"init_rtol": 2e-7, "init_atol": 0.0, "qsat_rtol": 2e-14, "qsat_atol": 0.0,
                                "step_rtol": 2e-6, "step_payload_atol": 1e-10, "step_surface_atol": 1e-6,
                                "step_registration": "tests/test_s1_lake_step.py at 692eeae41 before first comparison LK19",
                                "basis": "Frozen before port comparison: <=2 REAL32 ulps for mixed-kind init, DOUBLE polynomial roundoff."}
    finally:
        report["wall_s"] = time.monotonic() - t0
        (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        shutil.rmtree(build)
    print(json.dumps({"cases": len(report["cases"]), "failures": report["failures"],
                      "wall_s": report["wall_s"]}))
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
