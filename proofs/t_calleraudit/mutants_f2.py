"""F2 mutants (E202): each must be killed by a genuine AssertionError, 0 setup errors.

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
    ("M6_postrk_ignores_convective_amount", "src/gpuwrf/runtime/operational_mode.py",
     "elif physics_forcing.convective_amount is not None:", "elif False:",
     "test_stateless_cumulus_raincv_reaches_land_through_the_actual_step"),
    ("M7_cumulus_slot_never_requests_raincv", "src/gpuwrf/runtime/operational_mode.py",
     'want_raincv = getattr(next_carry, "noahmp_precipitation", None) is not None', "want_raincv = False",
     "test_stateless_cumulus_raincv_reaches_land_through_the_actual_step"),
    ("M8_tiedtke_returns_zero_raincv", "src/gpuwrf/coupling/scan_adapters.py",
     "return (next_state, raincv.reshape(ny, nx)) if return_raincv else next_state",
     "return (next_state, 0.0 * raincv.reshape(ny, nx)) if return_raincv else next_state",
     "test_tiedtke_adapter_returns_the_raincv_it_accumulates"),
]
results = []
for name, rel, old, new, selector in MUTANTS:
    path = TREE / rel
    original = path.read_text()
    assert original.count(old) >= 1, (name, original.count(old))
    path.write_text(original.replace(old, new, 1))  # first occurrence (M8: tiedtke_adapter)
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
