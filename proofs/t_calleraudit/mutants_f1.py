"""F1 mutants (E202): each must be killed by a genuine AssertionError, 0 setup errors.

usage: JAX_PLATFORMS=cpu python mutants_f1.py <detached snapshot> <basetemp> <out.json>
Runs ONLY on a detached snapshot (never the live tree); restores each file after its arm.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

TREE, BASE, OUT = map(Path, sys.argv[1:4])
BASE.mkdir(parents=True, exist_ok=True)  # E202: a missing --basetemp parent is a setup ERROR, not a kill
TEST = "tests/v040/test_t_calleraudit_land_precip.py"
MUTANTS = [
    ("M1_old_seed_set", "src/gpuwrf/runtime/operational_mode.py",
     "and int(namelist.mp_physics) != 0:",
     "and int(namelist.mp_physics) in (DEFAULT_MP_PHYSICS, 5, 7, 15, 38):",
     "test_operational_init_seeds_land_precipitation_for_every_precipitating_scheme"),
    ("M2_no_public_slot_filter", "src/gpuwrf/coupling/scan_adapters.py",
     "surface(inc[acc]) if slot in slots and acc in inc else zero",
     "surface(inc[acc]) if acc in inc else zero",
     "test_generic_mp_caller_feeds_only_wrf_public_slots"),
    ("M3_kessler_accumulator_not_public_rainncv", "src/gpuwrf/coupling/scan_adapters.py",
     'if "rainncv" in tendency.diagnostics else',
     'if False else',
     "test_generic_mp_caller_feeds_only_wrf_public_slots"),
    ("M4_nssl_drops_hailncv", "src/gpuwrf/coupling/scan_adapters.py",
     '"ice": zero, "hail": precip["hailncv"], "total": precip["rainncv"]}',
     '"ice": zero, "total": precip["rainncv"]}',
     "test_nssl_and_milbrandt_hail_and_all_phase_total_reach_the_land"),
    ("M5_milbrandt_drops_hailncv", "src/gpuwrf/runtime/operational_mode.py",
     'precip={**precip,"total":precip["rainncv"],"hail":precip["hailncv"]}',
     'precip={**precip,"total":precip["rainncv"]}',
     "test_nssl_and_milbrandt_hail_and_all_phase_total_reach_the_land"),
]
results = []
for name, rel, old, new, selector in MUTANTS:
    path = TREE / rel
    original = path.read_text()
    assert original.count(old) == 1, (name, original.count(old))
    path.write_text(original.replace(old, new))
    try:
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short",
                              "--basetemp", str(BASE / name), f"{TEST}", "-k", selector],
                             cwd=TREE, capture_output=True, text=True, timeout=1200,
                             env={**__import__("os").environ, "PYTHONPATH": str(TREE / "src")})
    finally:
        path.write_text(original)
    text = run.stdout + run.stderr
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    asserted = "AssertionError" in text or re.search(r"\nE\s+assert ", text) is not None  # pytest rewrite
    killed = run.returncode == 1 and " failed" in tail and asserted and " error" not in tail
    results.append(dict(mutant=name, file=rel, rc=run.returncode, killed=killed, summary=tail))
    print(name, killed, tail, flush=True)
OUT.write_text(json.dumps(results, indent=2) + "\n")
assert all(r["killed"] for r in results), results
