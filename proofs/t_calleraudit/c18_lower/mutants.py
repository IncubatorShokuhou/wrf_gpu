"""Deletion-sensitivity mutants for the c18-c24 fix (run on a private snapshot; restores every file)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TREE = Path(sys.argv[1]); OUT = Path(sys.argv[2]); BASE = Path(sys.argv[3])
PLUME = "src/gpuwrf/kernels/phys_mynn_plume.py"; DT = "src/gpuwrf/runtime/domain_tree.py"
T1 = "tests/test_t_calleraudit_mynn_passive_native.py"; T2 = "tests/test_t_calleraudit_lower_error_traceback.py"
MUTANTS = [
    ("M1_revert_passive_kwarg", PLUME, "def dmp_mf_columns_native(*args, interpret=True, passive=None, **kwargs):",
     "def dmp_mf_columns_native(*args, interpret=True, **kwargs):\n    passive = None", [T1 + "::test_native_entry_passive_fluxes_match_pristine_dmp_mf", T1 + "::test_release_mynn_root_step_lowers_with_number_scalars[28]"],
     r"TypeError: float\(\) argument must be a string or a real number, not 'dict'"),
    ("M2_lane_keeps_qni", PLUME, "lane = dict(kwargs, qni=jnp.asarray(values, dtype=jnp.float32))", "lane = dict(kwargs)",
     [T1 + "::test_native_entry_passive_fluxes_match_pristine_dmp_mf"], r"AssertionError"),
    ("M3_passive_dropped", PLUME, "        if passive:\n            from gpuwrf.kernels.phys_mynn_numbers import mf_flux_key",
     "        if False:\n            from gpuwrf.kernels.phys_mynn_numbers import mf_flux_key",
     [T1 + "::test_native_entry_passive_fluxes_match_pristine_dmp_mf", T1 + "::test_release_mynn_root_step_lowers_with_number_scalars[28]"],
     r"KeyError: 's_awqNc'|AssertionError"),
    ("M4_no_traceback_print", DT, "    traceback.print_exception(exc, file=sys.stderr)\n", "",
     [T2 + "::test_lower_error_text_prints_the_traceback"], r"AssertionError"),
    ("M5_fused_site_bypasses_helper", DT,
     "        except BaseException as exc:  # noqa: BLE001 - fail-open to normal fused jit\n            return None, None, _lower_error_text(exc)",
     "        except BaseException as exc:  # noqa: BLE001 - fail-open to normal fused jit\n            return None, None, f\"{type(exc).__name__}: {exc}\"",
     [T2 + "::test_both_lowering_sites_report_through_the_helper"], r"AssertionError"),
]
env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH="src")
results = []
for name, rel, old, new, tests, signature in MUTANTS:
    path = TREE / rel
    orig = path.read_text()
    assert orig.count(old) == 1, (name, orig.count(old))
    path.write_text(orig.replace(old, new))
    try:
        base = BASE / name
        base.parent.mkdir(parents=True, exist_ok=True)   # E202: basetemp parent must exist
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(base),
                               "--tb=short", *tests], cwd=TREE, env=env, capture_output=True, text=True)
    finally:
        path.write_text(orig)
    text = proc.stdout + proc.stderr
    summary = (text.strip().splitlines() or [""])[-1]
    setup_errors = len(re.findall(r"ERROR at setup|errors? in", text))
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", summary)) else 0
    killed = proc.returncode == 1 and failed == len(tests) and setup_errors == 0 and bool(re.search(signature, text))
    results.append(dict(mutant=name, file=rel, tests=tests, rc=proc.returncode, summary=summary,
                        signature=signature, signature_found=bool(re.search(signature, text)),
                        setup_errors=setup_errors, killed=killed))
    print(name, "KILLED" if killed else "SURVIVED/INVALID", summary, flush=True)
assert subprocess.run(["git", "diff", "--quiet"], cwd=TREE).returncode == 0, "tree not restored"
OUT.write_text(json.dumps(dict(tree=str(TREE), head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TREE, text=True).strip(),
                               mutants=results, killed=sum(r["killed"] for r in results), total=len(results)), indent=1) + "\n")
