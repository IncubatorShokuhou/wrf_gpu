"""Run semantic deletions on immutable host modules, under the real root pytest conftest.

Mutants live only in subprocess memory. No source mutation, forecast/GPU, or production atlas.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--only", nargs="+", help="rerun only changed guard artifacts; carry exact-source results for the rest")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    own = "tests/v45_waveops/test_waveops.py::"
    variants = [
        ("go-binding", "waves", "or any(type(token.get(k)) is not type(v) or token[k] != v for k, v in expected.items())",
         "or False", [own+"test_each_go_binding_refuses"]),
        ("registry-start", "registry", "METHOD_BEGIN", "",
         [own+"test_scheduler_registers_real_process_attempts[success-DONE_HELD]"]),
        ("registry-end", "registry", "METHOD_FINISH", "", [own+"test_real_registry_success_crash_abort_and_no_duplicate_end"]),
        ("retry-cap", "study", 'ledger["attempts"].get(key, 0) >= max_retries+1', "False",
         [own+"test_retry_and_case_hour_caps_never_launch_again"]),
        ("wall-cap", "study", "signal.setitimer(signal.ITIMER_REAL, max(0.001, wall-prep_wall))",
         "signal.setitimer(signal.ITIMER_REAL, 0)", [own+"test_wallclock_cap_interrupts_host_work"]),
        ("resume-lock", "study", "fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)", "None",
         [own+"test_concurrent_resume_cannot_double_start"]),
        ("harm-veto", "waves", "HARM", "",
         [own+"test_harm_veto_best_persists_and_stop_is_idempotent"]),
        ("objective-freeze", "waves", "OBJECTIVE", "", [own+"test_changed_objective_denominator_refuses_and_preserves_best"]),
        ("bundle-capability", "study", "BUNDLE", "", ["tests/test_optimize.py::test_bundle_launcher_runs_the_bundle_launcher_inside_the_consumer"]),
        ("durable-checkpoint", "study", "os.fsync(stream.fileno())", "None",
         [own+"test_atomic_checkpoint_is_durable_before_publication"]),
        ("incomplete-terminal-hold", "study", 'receipt.get("registry_run_id") and receipt.get("registry_complete") is not True',
         "False", [own+"test_terminal_failure_cannot_be_salvaged_as_success"]),
        ("terminal-completion-receipt", "registry", "self.complete = True", "self.complete = False",
         [own+"test_scheduler_registers_real_process_attempts[success-DONE_HELD]"]),
        ("dense-observation-cap", "continuous", "np.size(obs) > 4096", "False",
         [own+"test_dense_observation_cap_before_covariance"]),
        ("native-cloud-refusal", "continuous", 'scorer["kind"] == "cloud"', "False",
         [own+"test_dense_cloud_native_refused_before_model_work"]),
        ("cache-bookkeeping-namespace", "study", "ALISIOS_OPTIMIZE_REGISTRY", "GPUWRF_OPTIMIZE_REGISTRY",
         [own+"test_actual_member_bookkeeping_does_not_fragment_trace_key"]),
        ("optimizer-implementation-binding", "continuous", "OPTIMIZER_BINDING", "",
         [own+"test_resume_refuses_changed_optimizer_implementation"]),
    ]
    if a.only is not None:
        assert set(a.only) <= {v[0] for v in variants}
        variants = [v for v in variants if v[0] in a.only]
    results = []
    for name, module, old, new, tests in variants:
        path = a.source / f"src/gpuwrf/optimize/{module}.py"
        source = path.read_text()
        if old in ("METHOD_BEGIN", "METHOD_FINISH"):
            tree = ast.parse(source)
            function = next(f for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Invocation"
                            for f in c.body if isinstance(f, ast.FunctionDef)
                            and f.name == ("begin" if old == "METHOD_BEGIN" else "finish"))
            function.body = (ast.parse("return dict(env, ALISIOS_WRF_RUN_ID=self.run_id, "
                                      "ALISIOS_ATLAS_DIR=str(self.registry.atlas))").body
                             if old == "METHOD_BEGIN" else [ast.Return(ast.Constant(None))])
            changed = ast.unparse(ast.fix_missing_locations(tree))
        elif old == "OPTIMIZER_BINDING":
            tree = ast.parse(source)
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
            bindings = [n for n in function.body if isinstance(n, ast.Assign)
                        and isinstance(n.value, ast.Dict) and any(isinstance(k, ast.Constant)
                        and k.value == "optimizer_implementation" for k in n.value.keys)]
            assert len(bindings) == 1
            function.body = [ast.Pass() if n is bindings[0] else n for n in function.body]
            changed = ast.unparse(ast.fix_missing_locations(tree))
        elif old == "HARM":
            assert source.count('if row["admissible"]:') == 1
            assert source.count('if row.get("status") != "ADMISSIBLE":') == 1
            changed = source.replace('if row["admissible"]:', 'if True:').replace(
                'if row.get("status") != "ADMISSIBLE":', 'if False:')
        elif old == "OBJECTIVE":
            assert source.count('if objective is not None and self.state.get') == 1
            assert source.count('if objective is not None and best.get') == 1
            changed = source.replace('if objective is not None and self.state.get', 'if False and self.state.get').replace(
                'if objective is not None and best.get', 'if False and best.get')
        elif old == "BUNDLE":
            tree = ast.parse(source)
            guards = [n for n in ast.walk(tree) if isinstance(n, ast.If) and any(
                isinstance(b, ast.Raise) and 'bundle lacks per-member' in ast.unparse(b) for b in n.body)]
            assert len(guards) == 1
            guards[0].test = ast.Constant(False)
            changed = ast.unparse(ast.fix_missing_locations(tree))
        else:
            assert source.count(old) == 1, name
            changed = source.replace(old, new)
        xml = a.output / (name + ".xml")
        command = ["-q", "--tb=short", "--basetemp", str(a.output.parent / "pytest"),
                   "--junitxml=" + str(xml), *tests]
        bootstrap = (f"import sys; sys.path.insert(0,{str(a.source / 'scripts/v025')!r}); "
                     "import cpu_guard; sys.path.pop(0); "
                     f"import gpuwrf.optimize.{module} as m; "
                     f"exec(compile({changed!r},m.__file__,'exec'),m.__dict__); "
                     f"import pytest; raise SystemExit(pytest.main({command!r}))")
        env = dict(os.environ, PYTHONPATH=str(a.source / "src"), JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
        run = subprocess.run([sys.executable, "-c", bootstrap], cwd=a.source, env=env,
                             capture_output=True, text=True, timeout=150)
        (a.output / (name + ".log")).write_text(run.stdout+run.stderr)
        assert xml.is_file(), (name, run.returncode, run.stdout, run.stderr)
        cases = list(ET.parse(xml).getroot().iter("testcase"))
        failures = [c for c in cases if c.find("failure") is not None]
        errors = [c for c in cases if c.find("error") is not None]
        skips = [c for c in cases if c.find("skipped") is not None]
        assert run.returncode == 1 and failures and not errors and not skips, (name, run.stdout, run.stderr)
        assert all(any(s in c.find("failure").get("message", "") for s in
                       ("AssertionError", "DID NOT RAISE", "assert ", "Regex pattern did not match")) for c in failures), name
        results.append({"mutant": name, "assertion_failures": len(failures), "errors": 0, "skips": 0})
    (a.output / "RESULT.json").write_text(json.dumps({"status": "PASS", "mutants": results}, indent=2)+"\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
