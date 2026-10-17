"""Recheck affected mp38 artifacts on an immutable model tree after RV fixes.

Usage: check_review_fixes.py TREE OUTPUT_DIR [--resume]; references remain original WRF.
The prior compiled output comparison is regression evidence only.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

root, output = (Path(value).resolve() for value in sys.argv[1:3])
os.environ["GPUWRF_JAX_CACHE"] = "0"
os.environ["GPUWRF_FAST_DEFAULTS"] = "0"
resume = "--resume" in sys.argv[3:]
output.mkdir(exist_ok=resume)
proof = root / "proofs/v040/mpmisc/thompson38"
sys.path[:0] = [str(root / "scripts/v025"), str(root / "src"), str(proof)]
import cpu_guard
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.runtime.compile_cache import CACHE_STATUS
import check_columns
import check_literal_columns
import check_multistep
import check_default_specialization as specialization
import qualify_compiled
from gpuwrf.physics.thompson_hail.column import ThompsonHailColumnState, step_thompson_hail_column_with_precip as step

assert jax.devices()[0].platform == "cpu"
assert not CACHE_STATUS["enabled"] and CACHE_STATUS["dir"] is None
assert Path(sys.modules[step.__module__].__file__).is_relative_to(root)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
summary = dict(commit=commit, platform="cpu", cache="NONE", gpu_executed=False,
               literal_columns=0, literal_surfaces=0, direct_theta=0, compiled=[])
groups = ((1, 2, 3, 4, 5, 6), (7, 8, 9), (10, 11))
compiled_paths = []
with np.load(proof / "original_theta.npz", allow_pickle=False) as original_theta:
    for cases in groups:
        reference = proof if max(cases) <= 9 else proof.parent / "goddard4ice/koop_gate_witness"
        literal_path = output / f"literal{len(cases)}.json"
        compiled_path = output / f"compiled{len(cases)}.json"
        reuse = resume and literal_path.exists() and compiled_path.exists()
        if reuse:
            assert json.loads(compiled_path.read_text())["candidate_commit"] == commit
        else:
            check_literal_columns.main(reference, literal_path, cases)
        literal = json.loads(literal_path.read_text())
        assert all(row["finite"] and row["byte_equal"] for row in literal["fields"].values())
        assert all(row["finite"] and row["byte_equal"] for row in literal["public_surface"].values())
        summary["literal_columns"] += sum(row["total"] for row in literal["fields"].values())
        summary["literal_surfaces"] += sum(row["total"] for row in literal["public_surface"].values())
        for index, case in enumerate(cases):
            pii = np.asarray(check_columns.parse(reference / f"case{case}.txt")["columns"]["PII"], np.float32)
            theta = np.asarray(literal["inner_temperature"][index], np.float32) / pii
            np.testing.assert_array_equal(theta, original_theta[f"case{case}"])
            summary["direct_theta"] += theta.size
        if not reuse:
            check_columns.main(reference, compiled_path, cases)
        compiled = json.loads(compiled_path.read_text())
        previous = json.loads((Path("<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/thompson38") / f"H38-66-compiled{len(cases)}.json").read_text())
        same = (compiled["inner_temperature"] == previous["inner_temperature"]
                and compiled["precipitation_components"] == previous["precipitation_components"])
        for group in ("fields", "public_surface"):
            for name, row in compiled[group].items():
                assert row["finite"]
                same &= np.array_equal(np.asarray(row["actual"], np.float32), np.asarray(previous[group][name]["actual"], np.float32))
        summary["compiled"].append(dict(cases=cases, cold_and_first_s=compiled["wall_s"], same_as_previous_outputs=bool(same)))
        compiled_paths.append(compiled_path)
check_multistep.main(proof, output / "multistep.json")
multi = json.loads((output / "multistep.json").read_text())
assert multi["byte_equal"] and all(row["public_theta"]["byte_equal"] for row in multi["steps"])
summary["multistep_columns"] = sum(x["total"] for row in multi["steps"] for x in row["fields"].values())
summary["multistep_surfaces"] = sum(len(row["surfaces"]) for row in multi["steps"])
summary["multistep_theta"] = sum(row["public_theta"]["total"] for row in multi["steps"])
paths = {
    "src/gpuwrf/physics/thompson_column.py": ["_wrf_mixed_phase_sources", "_fall_speeds", "_snow_moment", "_snow_moments", "_rain_distribution", "_ice_distribution", "_wrf_warm_rain_rates"],
    "src/gpuwrf/physics/thompson_aero_column.py": ["_entry_cloud_number", "_ice_koop", "_eff_aero", "_saturation_adjustment_aero", "_rain_evaporation_aero", "_sed_cloud_water_aero", "_cloud_water_fall_speeds_aero", "_finish_aero"],
}
summary["default_specializations"] = []
for path, names in paths.items():
    before = subprocess.check_output(["git", "show", f"0eac91890268c6d5e8572a56298fc979440dbd14:{path}"], cwd=root, text=True)
    after = (root / path).read_text()
    for name in names:
        a, b = specialization.function(before, name), specialization.function(after, name)
        assert a == b, name
        summary["default_specializations"].append(dict(function=name, exact=True, sha256=hashlib.sha256(a.encode()).hexdigest()))
qualify_compiled.main(proof / "original_compiled_intervals.npz", compiled_paths, output / "compiled_bound.json")
bound = json.loads((output / "compiled_bound.json").read_text())
summary["compiled_intervals"] = dict(total=bound["total"], outside=bound["outside"], strict_interval_PASS=bound["strict_interval_PASS"])
# Inspect the changed six-column executable, with identical shapes/dtypes as H66.
refs = [check_columns.parse(proof / f"case{case}.txt") for case in groups[0]]
cols = {name: np.asarray([r["columns"][name] for r in refs], np.float32) for name in refs[0]["columns"]}
values = {field: jnp.asarray(cols[code + "_IN"]) for field, code in qualify_compiled.FIELDS.items()}
rho = np.float32(.622) * cols["P"] / (np.float32(287.04) * cols["T_IN"] * (cols["QV_IN"] + np.float32(.622)))
state = ThompsonHailColumnState(**values, p=jnp.asarray(cols["P"]), rho=jnp.asarray(rho), dz=jnp.asarray(cols["DELZ"]), w=jnp.asarray(cols.get("W_IN", np.zeros_like(rho))))
executable = step.lower(state, 90.).compile()
memory = executable.memory_analysis()
summary["cpu_executable_memory"] = {name: getattr(memory, name) for name in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
summary["memory_scope"] = "CPU buffer stats exclude source table constants929643480B and possible legacy duplication; not GPU/VRAM evidence"
times = []
for _ in range(9):
    start = time.monotonic()
    jax.block_until_ready(step(state, 90.))
    times.append(time.monotonic() - start)
summary["cpu_six_column_warm_s"] = dict(median=statistics.median(times), minimum=min(times), maximum=max(times), repeats=len(times))
summary["gate"] = "PASS: original literal and caller BYTE, compiled deviations disclosed, default AST preserved"
(output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
