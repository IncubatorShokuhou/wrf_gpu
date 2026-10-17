"""Genuine mutants for the c11 WDM6 small-grid firewall: each must be KILLED by an AssertionError of the named test
(no setup/collection errors, E202); the source file is restored afterwards.

usage: JAX_PLATFORMS=cpu python proofs/t_cfix_cu/c11_ptxas/mutants.py OUT_JSON BASETEMP
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SA = "src/gpuwrf/coupling/scan_adapters.py"
TEST = "tests/test_t_cfix_cu_wdm6_small_grid_firewall.py"
FENCED = "        _small_grid_firewall(mp(state.Nn), (ny, nx)), mp(state.Nc), mp(state.Nr),\n"
MUTANTS = [
    ("firewall removed", FENCED, "        mp(state.Nn), mp(state.Nc), mp(state.Nr),\n",
     TEST + "::test_small_grid_fences_exactly_the_nn_column_view"),
    ("firewall on the wrong operand (Nc)", FENCED,
     "        mp(state.Nn), _small_grid_firewall(mp(state.Nc), (ny, nx)), mp(state.Nr),\n",
     TEST + "::test_small_grid_fences_exactly_the_nn_column_view"),
    ("unconditional barrier (large grids change too)", FENCED,
     "        jax.lax.optimization_barrier(mp(state.Nn)), mp(state.Nc), mp(state.Nr),\n",
     TEST + "::test_large_grid_traces_no_barrier"),
    ("Nn operand perturbed at the call site", FENCED,
     "        _small_grid_firewall(mp(state.Nn), (ny, nx)) * 1.5, mp(state.Nc), mp(state.Nr),\n",
     TEST + "::test_nn_argument_values_unchanged"),
    ("firewall itself not an identity (layout_pin)", "        leaves[i] = leaf\n", "        leaves[i] = leaf * 1.5\n",
     TEST + "::test_firewall_is_value_identity_on_cpu", "src/gpuwrf/kernels/layout_pin.py"),
]


def main(out, basetemp):
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(ROOT / "src"))
    results = []
    for name, old, new, test, *rel in MUTANTS:
        path = ROOT / (rel[0] if rel else SA)
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
        results.append(dict(mutant=name, file=rel[0] if rel else SA, test=test, rc=proc.returncode, killed=bool(killed), tail=last))
        print(json.dumps(results[-1]), flush=True)
    Path(out).write_text(json.dumps(results, indent=1) + "\n")
    assert all(r["killed"] for r in results), [r["mutant"] for r in results if not r["killed"]]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
