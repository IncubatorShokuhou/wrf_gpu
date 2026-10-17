"""Regression-only byte proof for reusing the completed CY07 CPU smoke.

The independent fidelity reference remains original CPU-WRF (test_cycle.py).
Compare the previous NumPy fallback restore to the device-select implementation
for every land/static leaf, in both REAL32 and legacy REAL64 initialization.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import jax
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.cycling import core
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state

ROOT = Path(__file__).resolve().parents[2]
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-cycling")
old_commit = sys.argv[1]
source = subprocess.run(["git", "show", old_commit + ":src/gpuwrf/cycling/core.py"], cwd=ROOT,
                        capture_output=True, text=True, check=True).stdout
old = types.ModuleType("gpuwrf.cycling.previous_restore")
exec(compile(source, old_commit + "/core.py", "exec"), old.__dict__)
assert jax.devices()[0].platform == "cpu"
samples = []
with tempfile.TemporaryDirectory(dir=LANE, prefix="carry-proof-") as temp:
    case = Path(temp) / "case"
    case.mkdir()
    template = Path(temp) / "template"
    shutil.copyfile(ROOT / "examples/switzerland_d01/wrfinput_d01", template)
    with Dataset(template, "r+") as ds:
        ds["Times"][0] = np.frombuffer(b"2023-01-15_01:00:00", dtype="S1")
    core.cycle("<USER_HOME>/wrf_gpu2_lanes/u2-diag/cpu_reference_3h/wrfout_d01_2023-01-15_01:00:00", template, case / "wrfinput_d01")
    shutil.copyfile(ROOT / "examples/switzerland_d01/namelist.input", case / "namelist.input")
    current = core.restore_land_state
    for mode in ("0", "1"):
        os.environ["GPUWRF_NOAHMP_NATIVE_REAL"] = mode
        jax.clear_caches()
        hashes = []
        for function in (old.restore_land_state, current):
            core.restore_land_state = function
            try:
                land, static, _ = build_noahmp_land_state(case, "d01")
            finally:
                core.restore_land_state = current
            values = [np.asarray(value) for value in jax.tree.leaves((land, static))]
            digest = hashlib.sha256()
            for value in values:
                digest.update(str((value.dtype.str, value.shape)).encode())
                digest.update(value.tobytes(order="C"))
            hashes.append(digest.hexdigest())
        assert hashes[0] == hashes[1], (mode, hashes)
        samples.append({"native_real": mode, "leaf_count": len(values), "parent_sha256": hashes[0], "candidate_sha256": hashes[1]})
        jax.clear_caches()
report = {"scope": "regression/carry-byte identity; independent fidelity is CPU-WRF", "parent_commit": old_commit,
          "candidate_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
          "samples": samples, "platform": "cpu", "gpu_executed": False,
          "reuse": "CY07 input land/static bytes unchanged; all timestep model source files unchanged"}
(LANE / "CY09_carry_identity.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
