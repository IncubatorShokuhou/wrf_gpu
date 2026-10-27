"""Localize full-step cold water residuals: original WRF / static OFF / ON prior."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--oracle", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--cohort", choices=("warm", "cold"), default="cold")
a = p.parse_args()
sys.path[:0] = [str(a.repo / "src"), str(a.repo / "proofs/w44_lsmparams")]
import jax
from columns import build
import oracle as wrapper
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.runtime.operational_mode import OperationalNamelist, build_clock_base
from gpuwrf.surface_params import controls as sc

assert jax.devices()[0].platform == "cpu"
a.out.mkdir(parents=True, exist_ok=True)
inputs = json.loads(a.oracle.read_text())
build_record = json.loads((a.build / "build.json").read_text())
executable = a.build / "w2_driver.exe"
assert wrapper.sha(executable) == build_record["executable_sha256"]
table_hashes = wrapper.edited_tables(a.repo, Path("<USER_HOME>/src/wrf_pristine/WRF"), a.out, "BASELINE", 1.)
wrapper.write_columns(a.out / "noahmp_columns.in", inputs["columns"])
env = dict(os.environ, LD_LIBRARY_PATH="<USER_HOME>/miniconda3/envs/wrfbuild/lib", NOAHMP_WIRE_SOIL="1",
           NOAHMP_WIRE_FICEOLD="1", NOAHMP_NSTEPS="1", W2_OPT_CRS="1")
completed = subprocess.run([str(executable)], cwd=a.out, env=env, capture_output=True, text=True, timeout=60)
(a.out / "wrf_stdout.log").write_text(completed.stdout)
assert completed.returncode == 0 and "NOAHMP_OFFLINE_OK" in completed.stdout
parser = wrapper.module("static_cold_wrf", a.repo / "proofs/noahmp/build_noahmp_savepoints.py")
parser.HERE = a.out
wrf = parser._parse_savepoints()[:4] if a.cohort == "warm" else parser._parse_savepoints()[4:]
land, forcing, static, ep, rp, fixture, rows = build(cold=a.cohort == "cold")
dt = rows[0]["dt"]
base = OperationalNamelist(grid=None, tendencies=None, metrics=None, dt_s=dt, use_noahmp=True,
                          noahmp_static=static, noahmp_land=land, noahmp_energy_params=ep, noahmp_rad_params=rp)
prior = dataclasses.replace(base, surface_controls=sc.prior_controls(fixture, sorted(set(sc.LSM_CONTROLS) - {"rsmin"})))
traces = []


def program(l, f, nl, cb):
    traces.append(1)
    s, ep, rp = nl.noahmp_static, nl.noahmp_energy_params, nl.noahmp_rad_params
    values = sc.traced_controls(nl, cb)
    if values is not None:
        s, ep, rp = sc.apply_surface_controls(s, ep, rp, values)
    return noah_mp_step(l, f, s, dt, energy_params=ep, rad_params=rp, return_diag=True)


fn = jax.jit(program)
results, payload, report = [], {}, {"platform": "cpu", "gpu_executed": False, "source": str(a.repo),
    "wrf_build": build_record, "tables": table_hashes, "cohort": a.cohort, "columns": [], "arms": []}
for label, nl in (("OFF", base), ("ON_PRIOR", prior)):
    output = jax.device_get(fn(land, forcing, nl, build_clock_base(nl)))
    results.append(output)
    for index, value in enumerate(jax.tree.leaves(output)):
        payload[f"{label}_{index}"] = np.asarray(value)
    new, flux, diagnostics = output
    arm = {"label": label, "columns": []}
    dz = np.asarray(static.dzs, np.float64)
    for j, (row, reference) in enumerate(zip(rows, wrf, strict=True)):
        s = row["state_in"]
        initial = float(np.dot(np.asarray(land.smois)[:, 0, j].astype(np.float64), dz) * 1000
                        + float(land.sneqv[0, j]) + float(land.canliq[0, j]) + float(land.canice[0, j]))
        final = float(np.dot(np.asarray(new.smois)[:, 0, j].astype(np.float64), dz) * 1000
                      + float(new.sneqv[0, j]) + float(new.canliq[0, j]) + float(new.canice[0, j]))
        loss = float(flux.qfx[0, j]) * dt + 1000 * (float(new.sfcrunoff[0, j]) + float(new.udrunoff[0, j]))
        rf = float(np.dot(reference["smc_out"], dz) * 1000 + reference["snow_out"]["sneqv"]
                   + reference["water_out"]["canliq"] + reference["water_out"]["canice"])
        rl = reference["driver"]["qfx"] * dt + reference["water_out"]["runsrf"] + reference["water_out"]["runsub"]
        arm["columns"].append({"name": row["name"], "initial_mass_mm": initial,
            "native_final_mass_mm": final, "native_loss_mm": loss, "native_signed_residual_mm": final - initial + loss,
            "wrf_final_mass_mm": rf, "wrf_loss_mm": rl, "wrf_signed_residual_mm": rf - initial + rl,
            "native_smc": np.asarray(new.smois)[:, 0, j].tolist(), "wrf_smc": reference["smc_out"],
            "native_sh2o": np.asarray(new.sh2o)[:, 0, j].tolist(), "wrf_sh2o": reference["sh2o_out"],
            "native_energy_Wm2": float(diagnostics.erreng[0, j])})
    report["arms"].append(arm)
np.savez(a.out / "native.npz", **payload)
old, new = ([np.asarray(x) for x in jax.tree.leaves(r)] for r in results)
report["default_equal_leaves"] = sum(x.tobytes() == y.tobytes() for x, y in zip(old, new, strict=True))
paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(results[0])[0]]
differences = []
for path, x, y in zip(paths, old, new, strict=True):
    if x.tobytes() == y.tobytes():
        continue
    error = np.abs(x.astype(np.float64) - y.astype(np.float64))
    def ordered(v):
        bits = v.view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7fffffff), bits)
    nonzero = x != 0
    differences.append({"path": path, "changed_cells": int(np.count_nonzero(x != y)),
        "max_abs": float(error.max()), "max_ulp": int(np.abs(ordered(x) - ordered(y)).max()) if x.dtype == np.float32 else None,
        "max_rel_nonzero": float(np.max(error[nonzero] / np.abs(x[nonzero].astype(np.float64)), initial=0.)),
        "zero_reference_changed_cells": int(np.count_nonzero((~nonzero) & (error != 0)))})
report["default_differences"] = differences
report["traces"] = len(traces)
(a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
