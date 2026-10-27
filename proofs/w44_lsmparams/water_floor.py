"""Manager-approved 2026-10-23 C27 amendment: original REAL 1-ulp ensemble.

Fixed 32-member soil-water pattern set, declared before measurement. The floor
for EACH changed-table arm and column is the maximum pristine residual. No
port execution, fitted safety factor, member rejection or adaptive sampling.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

SEED = 20261023


def patterns():
    vertices = list(itertools.product((-1, 1), repeat=4))
    interior = [x for x in itertools.product((-1, 0, 1), repeat=4) if 0 in x and any(x)]
    rng = np.random.default_rng(SEED)
    extra = [interior[index] for index in rng.permutation(len(interior))[:16]]
    values = vertices + extra
    assert len(values) == len(set(values)) == 32
    return values


def perturb(rows, pattern, member):
    output = copy.deepcopy(rows)
    for row in output:
        s = row["state_in"]
        smc, liquid = np.asarray(s["smc"], np.float32), np.asarray(s["sh2o"], np.float32)
        changed, changed_liquid = smc.copy(), liquid.copy()
        for k, direction in enumerate(pattern):
            liquid_direction = pattern[(k + 1) % 4] * (-1 if member % 2 else 1)
            # An existing one-ulp ice fraction also needs coherent directions:
            # independent opposite moves could make liquid exceed total water.
            gap = int(smc[k].view(np.uint32)) - int(liquid[k].view(np.uint32))
            if gap <= 1:
                liquid_direction = direction
            if direction:
                changed[k] = np.nextafter(smc[k], np.float32(np.inf if direction > 0 else -np.inf))
            if liquid_direction:
                changed_liquid[k] = np.nextafter(liquid[k], np.float32(np.inf if liquid_direction > 0 else -np.inf))
        assert np.isfinite(changed).all() and np.isfinite(changed_liquid).all()
        assert np.all((changed_liquid >= 0) & (changed_liquid <= changed))
        s["smc"], s["sh2o"] = changed.astype(np.float64).tolist(), changed_liquid.astype(np.float64).tolist()
    return output


def residual(row, reference):
    s = row["state_in"]
    dz = (-np.diff(np.r_[0., row["zsoil"]])).astype(np.float32).astype(np.float64)
    beginning = np.dot(np.asarray(s["smc"], np.float32).astype(np.float64), dz) * 1000
    beginning += sum(float(np.float32(s[key])) for key in ("sneqv", "canliq", "canice"))
    ending = np.dot(np.asarray(reference["smc_out"], np.float64), dz) * 1000
    ending += reference["snow_out"]["sneqv"] + reference["water_out"]["canliq"] + reference["water_out"]["canice"]
    # Liquid+frozen input rate (mm/s); WRF PRCP = PRCPCONV + PRCPNONC (+ PRCPSHCV = 0 here).
    # Zero for every dry column, so earlier dry receipts are numerically unchanged.
    precipitation = (row["forcing"]["prcpconv"] + row["forcing"]["prcpnonc"]) * row["dt"]
    return float(ending - beginning + reference["driver"]["qfx"] * row["dt"]
                 + reference["water_out"]["runsrf"] + reference["water_out"]["runsub"] - precipitation)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(a.repo / "src"), str(a.repo / "proofs/w44_lsmparams")]
    import oracle as wrapper
    inputs = json.loads(a.oracle.read_text())
    build = json.loads((a.build / "build.json").read_text())
    executable = a.build / "w2_driver.exe"
    assert wrapper.sha(executable) == build["executable_sha256"]
    ptn = patterns()
    members = [perturb(inputs["columns"], pattern, index) for index, pattern in enumerate(ptn)]
    # Fully liquid layers move together. Require 32 distinct input states on
    # every column, rather than counting duplicate members as new coverage.
    for column in range(len(inputs["columns"])):
        signatures = [tuple(row[column]["state_in"][key][k] for key in ("smc", "sh2o") for k in range(4))
                      for row in members]
        assert len(set(signatures)) == 32
    report = {"schema": "w2-c27-pristine-real-ulp-floor-v1", "manager_authority": "MAIN MAILBOX 2026-10-23T22:42:56Z",
        "binding_rule": "native<=max(1.5*pristine_unperturbed,measured_ensemble_max) AND native<0.1mm",
        "seed": SEED, "members": 32, "patterns": ptn, "perturbed_fields": "SMOIS/SH2O, each component at most one REAL ulp",
        "phase_rule": "SMC/SH2O within one ulp move coherently; otherwise independently within one ulp",
        "oracle_sha256": wrapper.sha(a.oracle), "wrf_build": build, "columns": [c["name"] for c in inputs["columns"]],
        "arms": {}}
    a.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH="<USER_HOME>/miniconda3/envs/wrfbuild/lib",
               NOAHMP_WIRE_SOIL="1", NOAHMP_WIRE_FICEOLD="1", NOAHMP_NSTEPS="1")
    parser = wrapper.module("ulp_wrf_parser", a.repo / "proofs/noahmp/build_noahmp_savepoints.py")
    for name, arm in inputs["arms"].items():
        cwd = a.out / name
        cwd.mkdir(parents=True, exist_ok=True)
        for table, expected in arm["tables"].items():
            source = a.oracle.parent / name / table
            assert wrapper.sha(source) == expected
            (cwd / table).write_bytes(source.read_bytes())
        parser.HERE = cwd
        samples = []
        for index, rows in enumerate(members):
            wrapper.write_columns(cwd / "noahmp_columns.in", rows)
            result = subprocess.run([str(executable)], cwd=cwd,
                env=dict(env, W2_OPT_CRS=str(arm["opt_crs"])), capture_output=True, text=True, timeout=60)
            if result.returncode or "NOAHMP_OFFLINE_OK" not in result.stdout:
                (cwd / f"failure_{index}.log").write_text(result.stdout + result.stderr)
                raise RuntimeError(f"original WRF failed on ensemble member {name}/{index}")
            values = parser._parse_savepoints()
            assert len(values) == len(rows)
            signed = [residual(row, value) for row, value in zip(rows, values, strict=True)]
            assert np.isfinite(signed).all()
            samples.append({"member": index, "input_sha256": wrapper.sha(cwd / "noahmp_columns.in"),
                            "output_sha256": wrapper.sha(cwd / "noahmp_savepoints.out"), "signed_residual_mm": signed})
        unperturbed = [abs(residual(row, value)) for row, value in zip(inputs["columns"], arm["oracle"], strict=True)]
        floor = np.max(np.abs(np.asarray([r["signed_residual_mm"] for r in samples])), axis=0)
        report["arms"][name] = {"factor": arm["factor"], "samples": samples,
            "columns": [{"name": row["name"], "pristine_unperturbed_mm": base, "floor_mm": float(bound)}
                        for row, base, bound in zip(inputs["columns"], unperturbed, floor, strict=True)]}
        (a.out / "floor.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"name": name, "members": 32, "floor_mm": floor.tolist()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
