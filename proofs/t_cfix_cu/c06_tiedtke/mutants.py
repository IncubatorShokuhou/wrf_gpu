"""Genuine mutants for the c06 cu6 Tiedtke fixes (ROOT 1/2/3): each must be KILLED by an assertion of the
real-c06 test (no setup/collection errors, E202); the source file is restored afterwards.

usage: JAX_PLATFORMS=cpu python proofs/t_cfix_cu/c06_tiedtke/mutants.py OUT_JSON BASETEMP
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JAXK = "src/gpuwrf/physics/cumulus_tiedtke_jax.py"
NPK = "src/gpuwrf/physics/cumulus_tiedtke.py"
SA = "src/gpuwrf/coupling/scan_adapters.py"
TEST = "tests/test_t_cfix_cu_tiedtke_c06_rows.py"
T_ROWS = TEST + "::test_tile_rows_match_wrf_row_calls_on_real_c06"
T_NP = TEST + "::test_numpy_reference_matches_wrf_single_column_calls"
T_WIRE = TEST + "::test_adapter_runs_the_tile_row_kernel"
MUTANTS = [
    ("ROOT1 JAX: no ldcum reset before the ascent (cuasc_new 3.1)", JAXK,
     "    ldcum = jnp.bool_(False)\n", "", T_ROWS),
    ("ROOT1 numpy: no ldcum reset before the ascent", NPK,
     "    ldcum = False\n    if ktype == 1:\n        ikb = kcbot\n",
     "    if ktype == 1:\n        ikb = kcbot\n", T_NP),
    ("ROOT2: icum early return per column instead of per tile row", SA,
     "QVF0, QVB0, QFX0, XL0, ZNU0, dt, stepcu=int(stepcu), own_icum_mask=False,",
     "QVF0, QVB0, QFX0, XL0, ZNU0, dt, stepcu=int(stepcu), own_icum_mask=True,", T_ROWS),
    ("ROOT2: tile-row reduction over the wrong axis", SA,
     "    row_active = jnp.any(icum1.reshape(ny, nx), axis=1, keepdims=True)\n",
     "    row_active = jnp.any(icum1.reshape(ny, nx), axis=0, keepdims=True)\n", T_ROWS),
    ("ROOT2: row mask dropped (no early return at all)", SA,
     "    rates = [jnp.where(col_active[:, None], f, 0.0) for f in rates]\n",
     "    rates = list(rates)\n", T_ROWS),
    ("ROOT3 JAX: ihmin search keyed on the initial ihmin (never stops)", JAXK,
     "        llo1 = ldcum & (ktype == 1) & (ihmin == kcbot)\n",
     "        llo1 = ldcum & (ktype == 1) & (ihmin0 == kcbot)\n", T_ROWS),
    ("wiring: adapter calls the column kernel directly", SA,
     "    (rth, rqv, rqc, rqr, rqi, rqs, ru, rv, raincv) = tiedtke_tile_rows(\n",
     "    (rth, rqv, rqc, rqr, rqi, rqs, ru, rv, raincv) = tiedtke_column_jax(\n",
     T_WIRE),
]


def main(out, basetemp):
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(ROOT / "src"))
    results = []
    for name, rel, old, new, test in MUTANTS:
        path = ROOT / rel
        original = path.read_text()
        assert original.count(old) == 1, (name, original.count(old))
        try:
            path.write_text(original.replace(old, new))
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                   "--basetemp", basetemp, test], cwd=ROOT, env=env, capture_output=True,
                                  text=True, timeout=1800)
        finally:
            path.write_text(original)
        text = proc.stdout + proc.stderr
        last = text.strip().splitlines()[-1] if text.strip() else ""
        killed = proc.returncode == 1 and "AssertionError" in text and " error" not in last
        results.append(dict(mutant=name, file=rel, test=test, rc=proc.returncode, killed=bool(killed), tail=last))
        print(json.dumps(results[-1]), flush=True)
    Path(out).write_text(json.dumps(results, indent=1) + "\n")
    assert all(r["killed"] for r in results), [r["mutant"] for r in results if not r["killed"]]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
