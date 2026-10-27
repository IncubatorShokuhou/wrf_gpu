"""Deletion mutants for critic F1-F3 on the lane worktree (CPU, sequential, source restored by git).

usage: JAX_PLATFORMS=cpu python proofs/w44_topo/fix_mutants.py <basetemp> <receipt.json>
A mutant counts as KILLED only if pytest reports an AssertionError failure and 0 errors (E202).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTS = ["tests/v025/w44_topo/test_shadow_frozen_oracle.py", "tests/v025/w44_topo/test_shadow_init_seam.py",
         "tests/v025/w44_topo/test_shadow_lifecycle.py",
         "tests/test_parallel_decomposition_fixes.py::test_actual_shadow_operator_ridge_outside_old_halo_is_retained"]
MUTANTS = {
    "M1_shadow_uses_rrtmg_eot_geometry": ("src/gpuwrf/coupling/physics_couplers.py",
        "        shadow_geometry = SolarGeometry(shadow_coszen, geometry.declination_rad, shadow_hrang)\n",
        "        shadow_geometry = geometry\n"),
    "M2b_coupler_drops_nest_held_height": ("src/gpuwrf/coupling/physics_couplers.py",
        "                      if previous_shadow_height_m is None else previous_shadow_height_m) if nested_shadow else None\n",
        "                      if True else previous_shadow_height_m) if nested_shadow else None\n"),
    "M3_kernel_night_height_h": ("src/gpuwrf/kernels/topographic_shadow.py",
        'jnp.where(daylight, height, rn("sub", h0, one * F(.001)))',
        'jnp.where(daylight, height, rn("sub", h, one * F(.001)))'),
    "M4_refresh_passes_zeros": ("src/gpuwrf/coupling/topographic_shadow.py",
        '    return {"previous_shadow_height_m": value, "nested_shadow": nested(namelist)}\n',
        '    return {"previous_shadow_height_m": jnp.zeros_like(value), "nested_shadow": nested(namelist)}\n'),
    "F1a_nested_pipeline_no_reseed": ("src/gpuwrf/integration/nested_pipeline.py",
        "            initial_full_rad = preserve_initial_height(carry.radiation_diagnostics, initial_full_rad)\n", ""),
    "F1b_initial_feedback_no_reseed": ("src/gpuwrf/nesting/initial_feedback.py",
        "            full=preserve_initial_height(new.radiation_diagnostics,full)\n", ""),
    "M5_cpu_reference_night_uses_terrain": ("src/gpuwrf/physics/topographic_shading.py",
        "            jnp.where(daylight, height, local - F(.001)))", "            jnp.where(daylight, height, terrain - F(.001)))"),
    "M6_toposhad_clock_adds_half_radt_twice": ("src/gpuwrf/physics/topographic_shading.py",
        "    xt24 = jnp.fmod(add(F(xtime_minutes), mul(F(radt_minutes), F(.5))), F(1440))",
        "    xt24 = jnp.fmod(add(F(xtime_minutes), mul(F(radt_minutes), F(1.))), F(1440))"),
}


def run(basetemp, tag):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", f"{basetemp}/{tag}", *TESTS]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1200)
    out = p.stdout + p.stderr
    summary = [line for line in out.splitlines() if re.search(r"\d+ (passed|failed|error)", line)][-1:]
    failed = re.findall(r"^FAILED (\S+)", out, re.M)
    errors = len(re.findall(r"^ERROR ", out, re.M))
    return dict(rc=p.returncode, summary=summary, failed=failed, errors=errors,
                assertion=("AssertionError" in out) or ("assert " in out and "E   " in out))


def main():
    basetemp, receipt = sys.argv[1], Path(sys.argv[2])
    status = subprocess.run(["git", "status", "--short", "--", "src", "tests"], cwd=REPO, capture_output=True, text=True).stdout
    assert status.strip() == "", status
    result = {"head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()}
    result["baseline"] = base = run(basetemp, "baseline")
    assert base["rc"] == 0 and base["errors"] == 0, base
    for name, (path, old, new) in MUTANTS.items():
        file = REPO / path
        text = file.read_text()
        assert text.count(old) == 1, (name, text.count(old))
        file.write_text(text.replace(old, new))
        try:
            r = run(basetemp, name)
        finally:
            subprocess.run(["git", "checkout", "--", path], cwd=REPO, check=True)
        r["killed"] = r["rc"] != 0 and r["errors"] == 0 and r["assertion"] and bool(r["failed"])
        result[name] = r
        print(name, r["killed"], r["summary"], r["failed"][:4], flush=True)
    status = subprocess.run(["git", "status", "--short", "--", "src", "tests"], cwd=REPO, capture_output=True, text=True).stdout
    assert status.strip() == "", status
    result["all_killed"] = all(result[n]["killed"] for n in MUTANTS)
    receipt.write_text(json.dumps(result, indent=1) + "\n")
    print("ALL_KILLED", result["all_killed"])


if __name__ == "__main__":
    main()
