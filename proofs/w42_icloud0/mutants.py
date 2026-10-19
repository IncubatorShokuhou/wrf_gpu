"""Deletion/mutation sensitivity of the icloud=0 tests in tests/v041/test_icloud.py (E39/E202).

One source mutation at a time on the working tree (sequential, restored byte-exactly, never concurrent: E125); a mutant
counts as KILLED only on a genuine AssertionError with zero setup errors.
usage: JAX_PLATFORMS=cpu python mutants.py SRC_ROOT OUT.json [MUTANT_ID ...]
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
PC = "src/gpuwrf/coupling/physics_couplers.py"
NB = "src/gpuwrf/io/namelist_binding.py"
OM = "src/gpuwrf/runtime/operational_mode.py"
ZERO = ("        cloud_fraction, qc_columns, qi_columns, qs_columns, qg_columns = map(\n"
        "            jnp.zeros_like, (cloud_fraction, qc_columns, qi_columns, qs_columns, qg_columns))\n")
MUTANTS = [
    ("Z1 condensate kept (CLDFRA 0 only)", PC, ZERO, "        cloud_fraction = jnp.zeros_like(cloud_fraction)\n",
     "icloud0_feeds"),
    # (the early CLDFRA=0 branch alone is an equivalent mutant: the final zeroing covers it; it only avoids tracing
    #  cal_cldfra1 for nothing)
    ("Z2 final zeroing dropped", PC, ZERO, "        pass\n", "icloud0_feeds"),
    ("Z3 MP radii kept", PC, "        pass  # ICLOUD=0 skips", "        radii_kwargs = dict(re_cloud=_to_columns(state.re_cloud), "
     "re_ice=_to_columns(state.re_ice), re_snow=_to_columns(state.re_snow), xland=state.xland)  #", "icloud0_feeds"),
    ("Z4 WSM3 restaging before the icloud=0 guard", PC, "    if int(icloud) == 0:\n        # module_ra_rrtmg_lw.F:12032",
     "    if int(icloud) == 0 and int(mp_physics) != 3:\n        # module_ra_rrtmg_lw.F:12032", "icloud0_feeds"),
    ("Z5 icloud=0 refused again", NB, "if icloud not in (0, 2, 3):", "if icloud not in (2, 3):",
     "honour_report"),
    ("Z6 icloud not forwarded", OM, '"cloud_mp_physics": int(namelist.mp_physics), "icloud": icloud}',
     '"cloud_mp_physics": int(namelist.mp_physics)}', "cloud_policy"),
]
if len(sys.argv) > 3:
    MUTANTS = [m for m in MUTANTS if m[0].split()[0] in sys.argv[3:]]
results = []
for name, rel, old, new, select in MUTANTS:
    path = ROOT / rel
    original = path.read_bytes()
    text = original.decode()
    assert text.count(old) == 1, (name, text.count(old))
    path.write_text(text.replace(old, new))
    try:
        run = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/v041/test_icloud.py", "-q", "-x", "-k", select, "-p", "no:cacheprovider",
             "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/w42-icloud0/pytest-mut"],
            cwd=ROOT, capture_output=True, text=True, timeout=1500)
    finally:
        path.write_bytes(original)
    tail = run.stdout[-4000:]
    summary = tail.strip().splitlines()[-1] if tail.strip() else ""
    results.append({"mutant": name, "select": select, "rc": run.returncode,
                    "killed": run.returncode == 1 and "AssertionError" in tail and " error" not in summary.lower(),
                    "summary": summary})
    print(results[-1], flush=True)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
assert all(r["killed"] for r in results), [r["mutant"] for r in results if not r["killed"]]
