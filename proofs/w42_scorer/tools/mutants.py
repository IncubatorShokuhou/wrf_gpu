"""Mutant check for the w42-scorer bounds (CPU only). Each mutant edits a COPY of the candidate scripts+tests (never the
worktree, E125) and runs the targeted tests; KILLED only on a genuine assertion failure with zero pytest errors (E202).
usage: python mutants.py SRC_TREE WORK_DIR OUT_JSON"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

src, work, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FILES = ["scripts/compare_wrfout_grid.py", "scripts/wn3_fast_compare.py", "scripts/wn3_score.py",
         "proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json",
         "tests/test_wn3_fast_compare.py", "tests/test_compare_wrfout_grid_bounded.py",
         "tests/test_wn3_score_finite_gate.py"]
REF_FAST = Path("<USER_HOME>/wrf_gpu2_lanes/w42-scorer/ref_main_d96c268a5e/scripts/wn3_fast_compare.py")
T_FAST, T_BOUND = "tests/test_wn3_fast_compare.py", "tests/test_compare_wrfout_grid_bounded.py"
T_FIN = "tests/test_wn3_score_finite_gate.py"

MUTANTS = {
    "M1_iter_frames_holds_every_handle": ("scripts/compare_wrfout_grid.py",
        '''        with Dataset(pair["cpu_file"], "r") as cds, Dataset(pair["gpu_file"], "r") as gds:
            yield pair, cds, gds''',
        '''        yield pair, Dataset(pair["cpu_file"], "r").__enter__(), Dataset(pair["gpu_file"], "r").__enter__()''',
        [f"{T_FAST}::test_handles_are_bounded_to_one_frame_pair", f"{T_FAST}::test_wrapper_does_not_hold_handles"]),
    "M2_wrapper_back_to_held_handles": ("scripts/wn3_fast_compare.py", None, REF_FAST.read_text(),
        [f"{T_FAST}::test_wrapper_does_not_hold_handles"]),
    "M3_planner_ignores_budget": ("scripts/compare_wrfout_grid.py",
        "    if current:\n        batches.append(current)\n    return batches",
        "    return [list(entries)]",
        [f"{T_BOUND}::test_field_batches_respect_the_budget_and_order"]),
    "M4_subset_ignores_unpaired_cells": ("scripts/compare_wrfout_grid.py",
        "parts.append(chunk[mask.ravel() if valid is None else mask[valid]])",
        "parts.append(chunk[mask.ravel()[: chunk.size]])",
        [f"{T_BOUND}::test_split_percentiles_from_overall_chunks_equal_own_copies"]),
    "M5_split_accumulators_keep_copies": ("scripts/compare_wrfout_grid.py",
        "label: StatsAccumulator(element_abs_tolerance=self.element_tol, keep_abs=False) for label in masks",
        "label: StatsAccumulator(element_abs_tolerance=self.element_tol) for label in masks",
        [f"{T_BOUND}::test_a_field_keeps_one_float32_abs_copy"]),
    "M6_sequential_default_off": ("scripts/wn3_score.py",
        "sequential = args.hours > 24 or len(args.domains) > 3",
        "sequential = False",
        [f"{T_BOUND}::test_wn3_score_domain_scheduling"]),
    "M6b_ninenest_rule_dropped": ("scripts/wn3_score.py",
        "sequential = args.hours > 24 or len(args.domains) > 3",
        "sequential = args.hours > 24",
        [f"{T_BOUND}::test_wn3_score_domain_scheduling"]),
    "M7_sequential_does_not_wait": ("scripts/wn3_score.py",
        "        if sequential:\n            rc, usage[domain] = wait_child(jobs[domain][0], domain_timeout)",
        "        if False:\n            rc, usage[domain] = wait_child(jobs[domain][0], domain_timeout)",
        [f"{T_BOUND}::test_wn3_score_domain_scheduling"]),
    "M8_reference_nonfinite_not_checked": ("scripts/wn3_score.py",
        '            if frame["finite_cpu"] != frame["n"] and name not in NONFINITE_REFERENCE_EXEMPT:',
        "            if False:",
        [T_FIN]),
    "M9_exemption_applied_silently": ("scripts/wn3_score.py",
        '    return [{"field": name, "lead_h": frame["lead_h"], "count": frame["n"] - frame["finite_cpu"],',
        '    return [] and [{"field": name, "lead_h": frame["lead_h"], "count": frame["n"] - frame["finite_cpu"],',
        [f"{T_FIN}::test_reference_exemption_is_listed_never_silent"]),
    "M10_exemption_leaks_to_candidate": ("scripts/wn3_score.py",
        '            if frame["finite_gpu"] != frame["n"]:',
        '            if frame["finite_gpu"] != frame["n"] and name not in NONFINITE_REFERENCE_EXEMPT:',
        [f"{T_FIN}::test_reference_exemption_is_listed_never_silent"]),
}

results = {}
for name, (target, old, new, tests) in MUTANTS.items():
    tree = work / name
    if tree.exists():
        shutil.rmtree(tree)
    for f in FILES:
        (tree / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, tree / f)
    text = (tree / target).read_text()
    if old is None:
        text = new
    else:
        assert text.count(old) == 1, (name, "anchor not unique")
        text = text.replace(old, new)
    (tree / target).write_text(text)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp",
                           str(work / f"{name}_pytest"), *tests], cwd=tree, capture_output=True, text=True, timeout=3600)
    tail = proc.stdout[-4000:]
    summary = tail.strip().splitlines()[-1] if tail.strip() else ""
    errors = re.search(r"(\d+) error", summary)
    failed = re.search(r"(\d+) failed", summary)
    genuine = ("AssertionError" in proc.stdout or "assert " in proc.stdout or "\nE   " in proc.stdout)
    killed = bool(failed) and not errors and genuine and proc.returncode == 1
    results[name] = {"killed": killed, "rc": proc.returncode, "summary": summary,
                     "first_E_lines": [l for l in proc.stdout.splitlines() if l.startswith("E ")][:4]}
    print(name, "KILLED" if killed else "SURVIVED", summary, flush=True)
    shutil.rmtree(tree)
    out.write_text(json.dumps({"src": str(src), "complete": len(results) == len(MUTANTS), "mutants": results,
                               "all_killed": all(r["killed"] for r in results.values())}, indent=1) + "\n")
