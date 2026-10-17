"""F24 + F2-fix mutants (E202): each must be killed by a genuine AssertionError, 0 setup errors.

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
TEST = "tests/v040/test_t_calleraudit_f24_lsm_precip.py tests/v040/test_t_calleraudit_land_precip.py"
MUTANTS = [
    ("M9_seed_skips_mp0_convection_only", "src/gpuwrf/runtime/operational_mode.py",
     "    if not bool(namelist.run_physics):\n        return None\n    if _explicit_noahclassic(namelist):",
     "    if not bool(namelist.run_physics) or int(namelist.mp_physics) == 0:\n        return None\n    if _explicit_noahclassic(namelist):",
     "stateless_cumulus_raincv or seeds_land_precipitation"),
    ("M11_noahclassic_ignores_sr", "src/gpuwrf/coupling/noahclassic_surface_hook.py",
     "return jnp.maximum(rate, jnp.float32(0.0)), jnp.asarray(precipitation.sr, jnp.float32)",
     "return jnp.maximum(rate, jnp.float32(0.0)), jnp.zeros_like(rate)",
     "pristine_statements or assembler_consumes"),
    ("M12_px_precipitation_zero", "src/gpuwrf/coupling/pleim_xiu_surface_hook.py",
     "return jnp.maximum(jnp.float32(1.0e-3) * rate, jnp.float32(0.0))",
     "return jnp.zeros_like(rate)",
     "pristine_statements"),
    ("M13_restart_v6_refusal_removed", "src/gpuwrf/io/restart.py",
     'if version < 6 and carry.get("extra_fields", {}).get("noahclassic_land") is not None:',
     'if False and carry.get("extra_fields", {}).get("noahclassic_land") is not None:',
     "restart_roundtrip_and_pre_v6"),
    ("M14_thompson_sr_drops_ice", "src/gpuwrf/runtime/noahmp_precipitation.py",
     "sr = (p['snow'] + p['graupel'] + p['ice']) / (total + jnp.float32(1.e-12))",
     "sr = (p['snow'] + p['graupel']) / (total + jnp.float32(1.e-12))",
     "sr_channel_is_optional"),
    ("M15_generic_scheme_sr_ignored", "src/gpuwrf/coupling/scan_adapters.py",
     'sr = surface(tendency.diagnostics["sr"]) if "sr" in tendency.diagnostics else zero',
     "sr = zero",
     "carries_the_scheme_sr"),
]
results = []
for name, rel, old, new, selector in MUTANTS:
    path = TREE / rel
    original = path.read_text()
    assert original.count(old) >= 1, (name, original.count(old))
    path.write_text(original.replace(old, new, 1))  # first occurrence (M8: tiedtke_adapter)
    try:
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short",
                              "--basetemp", str(BASE / name), *TEST.split(), "-k", selector],
                             cwd=TREE, capture_output=True, text=True, timeout=1200,
                             env={**__import__("os").environ, "PYTHONPATH": str(TREE / "src")})
    finally:
        path.write_text(original)
    text = run.stdout + run.stderr
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    asserted = ("AssertionError" in text or "DID NOT RAISE" in text
                or re.search(r"\nE\s+assert ", text) is not None)  # pytest rewrite
    killed = run.returncode == 1 and " failed" in tail and asserted and " error" not in tail
    results.append(dict(mutant=name, file=rel, rc=run.returncode, killed=killed, summary=tail))
    print(name, killed, tail, flush=True)
OUT.write_text(json.dumps(results, indent=2) + "\n")
assert all(r["killed"] for r in results), results
