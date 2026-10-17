"""One real Swiss operational step from cycled Noah-MP state, CPU only.

Reuse the established CPU solver/device guard seam, without altering model
equations. All 42x42 cells and 44 levels are retained; new-init atmospheric
template is explicitly retimed, not a new independent real.exe analysis.
"""
from pathlib import Path
import json
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/u2-cycling")
sys.path.insert(0, str(ROOT / "proofs/o1_nlbind"))
import cpu_safe
cpu_safe.install()
import jax
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.cycling.core import cycle
from gpuwrf.contracts import state as state_contract
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om

assert jax.devices()[0].platform == "cpu"
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
source_case = ROOT / "examples/switzerland_d01"
reference = Path("<USER_HOME>/wrf_gpu2_lanes/u2-diag/cpu_reference_3h/wrfout_d01_2023-01-15_01:00:00")
report = {"platform": "cpu", "gpu_executed": False, "source": str(reference), "steps": 1,
          "shape": [44, 42, 42], "dt_s": 18, "template": "real Swiss atmosphere retimed to 01Z"}
start = time.perf_counter()
with tempfile.TemporaryDirectory(dir=LANE, prefix="coupled-") as temp:
    case = Path(temp) / "case"
    case.mkdir()
    template = Path(temp) / "template"
    shutil.copyfile(source_case / "wrfinput_d01", template)
    stamp = "2023-01-15_01:00:00"
    with Dataset(template, "r+") as ds:
        ds["Times"][0] = np.frombuffer(stamp.encode(), dtype="S1")
        ds.START_DATE = stamp
        ds.SIMULATION_START_DATE = stamp
    transfer = cycle(reference, template, case / "wrfinput_d01")
    (case / "wrfbdy_d01").symlink_to(source_case / "wrfbdy_d01")
    nl = (source_case / "namelist.input").read_text()
    nl = re.sub(r"(?mi)^(\s*start_hour\s*=).*", r"\g<1> 1,", nl)
    (case / "namelist.input").write_text(nl)
    cfg = pipeline.NestedPipelineConfig(case, Path(temp) / "out", Path(temp) / "proof", hours=1, max_dom=1)
    _, bundles, _, _, _, carries = pipeline._load_domains(cfg, ("d01",))
    ns, carry = bundles["d01"].namelist, carries["d01"]
    with Dataset(reference) as src:
        for name, leaf in (("SH2O", "sh2o"), ("SNICE", "snice"), ("SNLIQ", "snliq"), ("ISNOW", "isnow")):
            np.testing.assert_array_equal(np.asarray(getattr(carry.noahmp_land, leaf), dtype=src[name].dtype), src[name][0], err_msg=name)
    report["input_state_byte_exact"] = True
    report["load_s"] = time.perf_counter() - start
    soil_before = np.asarray(carry.noahmp_land.tslb).copy()
    carry = jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, carry)
    cb = om.build_clock_base(ns)
    next_carry = om._advance_chunk_fori(carry, ns, 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
    jax.block_until_ready(next_carry)
    leaves = [np.asarray(x) for x in jax.tree.leaves(next_carry)]
    report["nonfinite_values"] = sum(int((~np.isfinite(x)).sum()) for x in leaves if x.dtype.kind == "f")
    assert next_carry.census is not None, "smoke requires GPUWRF_CENSUS=1"
    report["guard_total"] = int(np.asarray(next_carry.census.guards).sum())
    report["max_soil_temperature_change_k"] = float(np.max(np.abs(np.asarray(next_carry.noahmp_land.tslb) - soil_before)))
    report["source_sha256"] = transfer["source_sha256"]
report["wall_s"] = time.perf_counter() - start
report["fast_defaults"] = os.environ.get("GPUWRF_FAST_DEFAULTS")
(LANE / "CY07_coupled_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
assert report["nonfinite_values"] == 0
assert report["guard_total"] == 0
