"""Deletion-sensitivity mutants for lane w41-optimize (E39/E202): each mutant must be killed by a genuine
assertion/expected-exception failure of its target test, with zero setup errors.

usage (CPU, committed clean tree): JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 python proofs/w41_optimize/mutants.py OUT.json
Each mutant edits one file in place, runs its target tests, then restores the committed file (git checkout).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASETEMP = "<USER_HOME>/wrf_gpu2_lanes/w41-optimize/pytest-mutants"
MUTANTS = [
    ("M1 release gate removed", "src/gpuwrf/optimize/released.py",
     "    if entry is None or entry[\"max_dom\"] != max_dom:\n",
     "    if False:\n",
     ["tests/test_optimize.py::test_unreleased_admission_is_refused",
      "tests/test_optimize.py::test_release_list_binds_the_final_checkpoint"]),
    ("M2 one member per launcher batch (no cross-configuration batches)", "src/gpuwrf/optimize/study.py",
     "                take = fit(rest, warm)\n",
     "                take = 1\n",
     ["tests/test_optimize.py::test_run_real_ensemble_preparation_fake_scheduler"]),
    ("M3 resume accepts a changed study", "src/gpuwrf/optimize/study.py",
     "        if not saved.is_file() or json.loads(saved.read_text()) != canonical:\n",
     "        if not saved.is_file():\n",
     ["tests/test_optimize.py::test_chunk_pause_and_resume_reproduce_the_uninterrupted_study"]),
    ("M4 single-domain pool not passed", "src/gpuwrf/optimize/study.py",
     "        kwargs = {\"pool_gib\": pool_gib} if single else {}\n",
     "        kwargs = {}\n",
     ["tests/test_optimize.py::test_run_real_ensemble_preparation_fake_scheduler"]),
    ("M5 driver not pinned to CPU", "src/gpuwrf/optimize/cli.py",
     "    os.environ[\"JAX_PLATFORMS\"] = \"cpu\"\n",
     "    pass\n",
     ["tests/test_optimize.py::test_driver_is_cpu_only_and_members_get_the_caller_platform"]),
    ("M6 no salvage on resume", "src/gpuwrf/optimize/study.py",
     "        _salvage(output, study, ledger)\n",
     "        pass\n",
     ["tests/test_optimize.py::test_interrupted_batch_members_are_salvaged_on_resume"]),
    ("M7 CPU cases still need the GPU lock", "scripts/parallel_cases.py",
     "    if cpu_only:\n        lock_required, lock_policy = False, \"CPU-only cases (JAX_PLATFORMS=cpu)\"\n",
     "",
     ["tests/test_parallel_cases.py::test_refusals"]),
    ("M8 ens.launch ignores the member env", "src/gpuwrf/ens/workflows.py",
     "    environ = os.environ if env is None else env\n",
     "    environ = os.environ\n",
     ["tests/test_ensemble.py::test_launch_lock_policy_follows_the_member_environment"]),
    ("M9 E41 filter ignored", "src/gpuwrf/optimize/released.py",
     "    if e41 == \"PASS\" and entry[\"best_E41\"] != \"PASS\":\n",
     "    if False:\n",
     ["tests/test_optimize.py::test_release_list_binds_the_final_checkpoint",
      "tests/test_optimize.py::test_configurations_by_released_name_without_admissions"]),
    ("M10 failed-batch members dropped", "src/gpuwrf/optimize/study.py",
     "            if entry is not None:\n                ledger[\"members\"][f\"{candidate['id']}|{case['id']}\"] = entry\n",
     "            if entry is not None and rc == 0:\n                ledger[\"members\"][f\"{candidate['id']}|{case['id']}\"] = entry\n",
     ["tests/test_optimize.py::test_failed_member_keeps_the_others_and_resume_reruns_only_it"]),
    ("M11 host need never measured", "src/gpuwrf/optimize/study.py",
     "        return round(1.1*max(peaks)/1e6, 2) if peaks else None\n",
     "        return None\n",
     ["tests/test_optimize.py::test_run_real_ensemble_preparation_fake_scheduler"]),
    ("M12 no cold-first wave", "src/gpuwrf/optimize/study.py",
     "        (rest if key in seen else first).append((candidate, case))\n",
     "        first.append((candidate, case))\n",
     ["tests/test_optimize.py::test_cold_first_compiles_each_configuration_once",
      "tests/test_optimize.py::test_run_real_ensemble_preparation_fake_scheduler"]),
    ("M13 warm waves estimated with cold walls", "src/gpuwrf/optimize/study.py",
     "        walls = warm_walls if warm and warm_walls else [m[\"wall_s\"] for m in members]\n",
     "        walls = [m[\"wall_s\"] for m in members]\n",
     ["tests/test_optimize.py::test_warm_waves_are_estimated_with_warm_member_walls"]),
    # Critic FIX 2026-10-17 (cr-w41-optimize): input binding, charged retries, admitted parallelism.
    ("M14 case inputs bound by path only", "src/gpuwrf/optimize/study.py",
     "                      \"forcing\": [identity(p) for p in forecast_inputs(directory)],\n",
     "                      \"forcing\": [{\"path\": str(p.resolve())} for p in forecast_inputs(directory)],\n",
     ["tests/test_optimize.py::test_resume_refuses_changed_forecast_input_of_completed_members"]),
    ("M15 no input check while the study runs", "src/gpuwrf/optimize/study.py",
     "                if _stat(f[\"path\"]) != stats[f[\"path\"]]:\n",
     "                if False:\n",
     ["tests/test_optimize.py::test_forecast_input_changed_during_the_run_is_refused"]),
    ("M16 budget counts completed members only", "src/gpuwrf/optimize/study.py",
     "        if ledger[\"scheduled_case_hours\"]+len(pairs)*hours > study[\"budget_case_hours\"]:\n",
     "        if (len(ledger[\"members\"])+len(pairs))*hours > study[\"budget_case_hours\"]:\n",
     ["tests/test_optimize.py::test_failed_retries_are_charged_against_the_case_hour_budget"]),
    ("M17 chunk sized with the requested parallelism", "src/gpuwrf/optimize/study.py",
     "        bound = min(count, max_parallel or count)\n",
     "        return min(count, max_parallel or count)\n",
     ["tests/test_optimize.py::test_chunk_size_follows_the_admitted_parallelism"]),
    ("M18 measured admission limit ignored", "src/gpuwrf/optimize/study.py",
     "            bound = min(bound, min(b[\"admitted\"] for b in same))\n",
     "            pass\n",
     ["tests/test_optimize.py::test_chunk_size_follows_the_admitted_parallelism"]),
    ("M19 over-chunk member not run alone", "src/gpuwrf/optimize/study.py",
     "            return 1  # one member alone already exceeds the chunk: it runs alone, recorded as over_chunk\n",
     "            return len(rest)\n",
     ["tests/test_optimize.py::test_warm_waves_are_estimated_with_warm_member_walls"]),
]


def main(out, only=None):
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True,
                      text=True).stdout.strip():
        raise SystemExit("tree must be clean (mutants restore files with git checkout)")
    results = []
    for name, rel, old, new, tests in MUTANTS:
        if only and name.split()[0] not in only:
            continue
        path = REPO/rel
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"{name}: anchor not unique in {rel}")
        path.write_text(text.replace(old, new))
        try:
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", BASETEMP,
                                   *tests], cwd=REPO, capture_output=True, text=True, timeout=2400)
        finally:
            subprocess.run(["git", "checkout", "--", rel], cwd=REPO, check=True)
        tail = proc.stdout[-3000:]
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
        errors = int(m.group(1)) if (m := re.search(r"(\d+) error", tail)) else 0
        reasons = [line.strip() for line in proc.stdout.splitlines() if line.startswith("E ")][:6]
        genuine = (failed > 0 and errors == 0 and "TimeoutExpired" not in proc.stdout
                   and any(re.search(r"AssertionError|DID NOT RAISE|^E\s+assert ", r) for r in reasons))
        results.append({"mutant": name, "file": rel, "tests": tests, "rc": proc.returncode, "failed": failed,
                        "setup_errors": errors, "killed_genuinely": genuine, "failure_lines": reasons,
                        "summary": tail.strip().splitlines()[-1:]})
        print(json.dumps(results[-1]), flush=True)
    Path(out).write_text(json.dumps({"mutants": results, "all_killed": all(r["killed_genuinely"] for r in results)},
                                    indent=1)+"\n")
    return 0 if all(r["killed_genuinely"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], set(sys.argv[2:]) or None))
