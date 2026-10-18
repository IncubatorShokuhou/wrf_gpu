"""Genuine mutants of lane w41-optcont: each patch removes one guarantee; the named tests must FAIL (junit <failure>,
never only <error>: a setup error is not a kill, E202) and the unmutated tree must pass the same selection.

  python proofs/w41_optcont/mutants.py TREE OUT.json BASETEMP

Patches are applied in place to TREE (one at a time, original bytes restored in `finally`, then `git diff --quiet`
is required for the mutated files), so run it only in the lane's own worktree with nothing else using it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

C = "tests/w41_optcont/test_continuous.py"
T = "tests/w41_optcont/test_ctopo_control.py"
MUTANTS = [
    ("M1_program_key_is_candidate", "src/gpuwrf/optimize/study.py",
     '    return candidate.get("program", candidate["id"])', '    return candidate["id"]',
     [f"{C}::test_stations_scorer_runs_from_an_observation_file"]),
    ("M2_members_ignore_theta", "src/gpuwrf/optimize/continuous.py",
     "fields = expand_from_basis(basis, manifest, member_multipliers(spec, theta))",
     "fields = expand_from_basis(basis, manifest, member_multipliers(spec, 0*np.asarray(theta)))",
     [f"{C}::test_twin_recovers_truth_with_one_program_and_cold_first"]),
    ("M3_launcher_drops_per_case_controls", "scripts/parallel_cases.py",
     '                    c.argv += ["--surface-controls", str(c.case_dir / a.case_surface_controls)]',
     "                    pass",
     [f"{C}::test_prepared_members_get_their_own_control_file_through_the_real_scheduler"]),
    ("M4_prepare_does_not_link_controls", "src/gpuwrf/ens/workflows.py",
     '            spec["inputs"].append(path)', "            pass",
     [f"{C}::test_prepared_members_get_their_own_control_file_through_the_real_scheduler"]),
    ("M5_ctopo_plain_multiplier", "src/gpuwrf/surface_params/controls.py",
     "    return 1.0 + np.asarray(factor, dtype=np.float64) * (base - 1.0)",
     "    return np.asarray(factor, dtype=np.float64) * base",
     [f"{T}::test_excess_form_keeps_uncorrected_cells_and_factor_one_is_byte_identical"]),
    ("M6_bind_drops_ctopo", "src/gpuwrf/surface_params/controls.py",
     "    if host:\n        if not allow_host:", "    if False:\n        if not allow_host:",
     [f"{T}::test_bind_replaces_only_the_traced_child_and_lowers_the_same_program"]),
    ("M7_no_static_signature_check", "src/gpuwrf/optimize/continuous.py",
     '        if static_signature(case["input_dir"], domains) != reference:',
     "        if False:",
     [f"{C}::test_basis_must_match_case_statics_and_runtime_tables"]),
    ("M8_no_runtime_table_check", "src/gpuwrf/optimize/continuous.py",
     "    basis_current(basis, manifest, used)\n", "",
     [f"{C}::test_basis_must_match_case_statics_and_runtime_tables"]),
    ("M9_no_protected_input_refusal", "src/gpuwrf/optimize/continuous.py",
     '    refuse_protected_inputs("gpuwrf optimize continuous", [*case_dirs, stations, basis_path, *observations],\n'
     '                            runtime=True)\n', "",
     [f"{C}::test_alisios_inputs_refused_before_any_read"]),
    ("M10_existing_control_file_not_checked", "src/gpuwrf/optimize/continuous.py",
     "    if path.exists():\n        existing, _ = load_controls(path)",
     "    if False:\n        existing, _ = load_controls(path)",
     [f"{C}::test_member_control_files_are_bound_and_refused_when_changed"]),
    ("M11_cli_allows_host_controls_with_checkpoint", "src/gpuwrf/cli.py",
     "        if _sc_host and (getattr(args, \"checkpoint_dir\", None) is not None",
     "        if False and (getattr(args, \"checkpoint_dir\", None) is not None",
     [f"{T}::test_cli_refuses_host_controls_with_checkpoint_or_resume"]),
    ("M12_program_report_always_one", "src/gpuwrf/optimize/continuous.py",
     "    one = bool(keys) and all(len(v) == 1 for v in keys.values()) and not unknown",
     "    one = True",
     [f"{C}::test_program_report_flags_a_second_program_or_missing_proof"]),
    ("M14_no_basis_physics_check", "src/gpuwrf/optimize/continuous.py",
     "        if view != basis_view:", "        if False:",
     [f"{C}::test_configuration_must_run_the_basis_case_physics"]),
    ("M15_no_ctopo_valley_cap", "src/gpuwrf/optimize/continuous.py",
     "                highest = min(highest, float(np.min(1.0/(1.0-below))))", "                pass",
     [f"{C}::test_ctopo_valley_cells_feasible_bounds_and_driver_regression"]),
    ("M16_default_bounds_not_intersected", "src/gpuwrf/optimize/continuous.py",
     "            low, high = max(low, lowest), min(high, highest)", "            pass",
     [f"{C}::test_bounds_must_stay_physically_feasible_and_defaults_are_tightened",
      f"{C}::test_ctopo_valley_cells_feasible_bounds_and_driver_regression"]),
    ("M17_precip_not_refused", "src/gpuwrf/optimize/continuous.py",
     'INTERVAL_VARIABLES = ("precip_rate",)\n', 'INTERVAL_VARIABLES = ()\nVARIABLES = (*VARIABLES, "precip_rate")\n',
     [f"{C}::test_accumulated_precip_is_refused_until_interval_aware"]),
    ("M18_cli_reads_study_before_refusal", "src/gpuwrf/optimize/continuous.py",
     '    refuse_protected_inputs("gpuwrf optimize", [path])\n    path = Path(path).resolve(strict=True)\n',
     "    path = Path(path).resolve(strict=True)\n",
     [f"{C}::test_cli_refuses_a_protected_study_before_the_schema_dispatch_reads_it"]),
    ("M13_resume_ignores_changed_study", "src/gpuwrf/optimize/study.py",
     '        if not saved.is_file() or json.loads(saved.read_text()) != canonical:',
     "        if not saved.is_file():",
     [f"{C}::test_resume_after_pause_and_interrupt_reproduces_the_study"]),
]


def pytest(tree, tests, xml, basetemp):
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    cmd = ["taskset", "-c", "2,18", "nice", "-n", "19", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
           "--basetemp", str(basetemp), f"--junitxml={xml}", *tests]
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=1200).returncode
    root = ET.parse(xml).getroot()
    cases = list(root.iter("testcase"))
    failures = [c.get("name") for c in cases if c.find("failure") is not None]
    errors = [c.get("name") for c in cases if c.find("error") is not None]
    return {"rc": rc, "tests": len(cases), "functions": sorted({c.get("name").split("[")[0] for c in cases}),
            "failures": failures, "errors": errors, "wall_s": round(time.time()-t0, 1)}


def main(tree, out, basetemp):
    tree, basetemp = Path(tree).resolve(), Path(basetemp)
    basetemp.mkdir(parents=True, exist_ok=True)
    selection = sorted({t for m in MUTANTS for t in m[4]})
    control = pytest(tree, selection, basetemp/"control.xml", basetemp/"control")
    results = {"control": control, "mutants": {}}
    # parametrized tests collect several cases per selected node id: compare the test functions that ran
    if control["failures"] or control["errors"] or control["functions"] != sorted({t.split("::")[-1] for t in selection}):
        results["verdict"] = "CONTROL_NOT_GREEN"
    else:
        for name, rel, old, new, tests in MUTANTS:
            path = tree/rel
            original = path.read_bytes()
            text = original.decode()
            if text.count(old) != 1:
                results["mutants"][name] = {"error": "patch anchor not unique/absent"}
                continue
            try:
                path.write_text(text.replace(old, new))
                r = pytest(tree, tests, basetemp/f"{name}.xml", basetemp/name)
            finally:
                path.write_bytes(original)
            r["killed"] = bool(r["failures"]) and not r["errors"]
            results["mutants"][name] = {"file": rel, "tests": tests, **r}
        clean = subprocess.run(["git", "diff", "--quiet", "--", *{m[1] for m in MUTANTS}], cwd=tree).returncode == 0
        results["tree_restored"] = clean
        killed = [n for n, r in results["mutants"].items() if r.get("killed")]
        results["verdict"] = "ALL_KILLED" if clean and len(killed) == len(MUTANTS) else "SURVIVORS"
        results["killed"] = f"{len(killed)}/{len(MUTANTS)}"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True).stdout.strip()
    results["tree_head"] = head
    Path(out).write_text(json.dumps(results, indent=1)+"\n")
    print(json.dumps({k: results.get(k) for k in ("verdict", "killed", "tree_restored")}))
    return 0 if results["verdict"] == "ALL_KILLED" else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
