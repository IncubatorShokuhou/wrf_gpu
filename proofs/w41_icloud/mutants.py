"""Deletion/mutation sensitivity of tests/v041/test_icloud.py (E39/E202).

Applies one source mutation at a time to the working tree (sequentially, restored byte-exactly
afterwards, never concurrently, E125), runs the named tests and records whether they fail with a
genuine AssertionError and zero setup errors.
usage: python mutants.py SRC_ROOT OUT.json [MUTANT_ID ...]   (e.g. M5b M9; default: all)
History: the first M5 (gridkm x 0.99985) was too weak to change any test column and survived;
M5b uses a real wrong formula (dx only, 3.0 instead of 4.24 km).
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
CF = "src/gpuwrf/physics/cloud_fraction.py"
PC = "src/gpuwrf/coupling/physics_couplers.py"
OM = "src/gpuwrf/runtime/operational_mode.py"
MUTANTS = [
    ("M1 ocean RH threshold dropped", CF, "jnp.where(ocean, rh00o, rh00l))", "rh00l)", "literal_port"),
    ("M2 adjust_cloudFinal deleted", CF, "    qc = jnp.where(partial & (lwp > f(1.0))", "    qc = jnp.where(partial & (lwp > f(9e9))",
     "literal_port"),
    ("M3 ice scan visits top k_m12C+1", CF, "k_m12c + 1, k_tropo, k_m12c + 2, nz)", "k_m12c + 1, k_tropo, k_m12c + 1, nz)",
     "literal_port"),
    ("M4 dz8w with 9.80665", PC, "jnp.float32(WRF_PHYSICS_G_M_S2)\n        cloud_fraction, qc3",
     "jnp.float32(GRAVITY_M_S2)\n        cloud_fraction, qc3", "real_call_path"),
    ("M5 gridkm dxkm formula", PC, "wrf_gridkm(float(grid.projection.dx_m), float(grid.projection.dy_m))",
     "1.414 * wrf_gridkm(float(grid.projection.dx_m), float(grid.projection.dy_m)) / 1.414213562",
     "real_call_path"),
    ("M5b gridkm from dx only", PC, "wrf_gridkm(float(grid.projection.dx_m), float(grid.projection.dy_m))",
     "float(grid.projection.dx_m) * 0.001", "real_call_path"),
    ("M6 radiation-only qc/qi dropped", PC,
     "        qc_columns, qi_columns = qc3.astype(qc_columns.dtype), qi3.astype(qi_columns.dtype)\n", "", "real_call_path"),
    ("M7 icloud not forwarded", OM, '"cloud_mp_physics": int(namelist.mp_physics), "icloud": icloud}',
     '"cloud_mp_physics": int(namelist.mp_physics)}', "radiation_cloud_policy or operational_driver"),
    ("M8 cal_cldfra2 threshold >=", CF, "else qc) > f(1.0e-6)", "else qc) >= f(1.0e-5)", "cal_cldfra2"),
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
             "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/w41-icloud/pytest-mut"],
            cwd=ROOT, capture_output=True, text=True, timeout=1500)
    finally:
        path.write_bytes(original)
    tail = run.stdout[-3000:]
    results.append({"mutant": name, "select": select, "rc": run.returncode,
                    "assertion": ("AssertionError" in tail or "assert " in tail) and " error" not in tail.lower().split("=")[-1],
                    "summary": tail.strip().splitlines()[-1] if tail.strip() else ""})
    print(results[-1], flush=True)
    OUT.write_text(json.dumps(results, indent=1) + "\n")
