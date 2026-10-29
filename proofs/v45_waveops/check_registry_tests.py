"""Root-conftest gate: 28 actual passes, zero skips, and assertion-only deletion kills.

No production ATLAS accesses. Use an immutable source archive and a private scratch directory.
"""
import argparse
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
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(args.source / "src"),
               PYTHONDONTWRITEBYTECODE="1", V05_REGISTRY_TEST_ROOT=str(args.output / "private-atlases"))
    env.pop("V05_REGISTRY_HELPER", None)
    variants = [("baseline", None, [])]
    text = (args.source / "scripts/alisios_registry.py").read_text()
    for name, old, new, tests in [
        ("delete-finally", "            self.end(terminal)", "            pass", [
            "test_success_finally_end_is_clean", "test_crash_finally_writes_failed_end"]),
        ("delete-precheck", "            rc = self._schema_check(argv)", "            rc = 0", [
            "test_precheck_refuses_before_storage_transport"]),
    ]:
        assert text.count(old) == 1
        mutant = args.output / (name + ".py")
        mutant.write_text(text.replace(old, new))
        variants.append((name, mutant, tests))
    results = []
    for name, mutant, tests in variants:
        xml = args.output / (name + ".xml")
        command = [sys.executable, "-m", "pytest", "-q", "-ra", "--basetemp",
                   str(args.output.parent / "pytest"), "--junitxml=" + str(xml)]
        command += ["tests/test_alisios_registry.py::RegistryTests::" + t for t in tests] if tests else [
            "tests/test_alisios_registry.py"]
        run = subprocess.run(command, cwd=args.source, env=env if mutant is None else
                             dict(env, V05_REGISTRY_HELPER=str(mutant)), capture_output=True, text=True, timeout=120)
        (args.output / (name + ".log")).write_text(run.stdout + run.stderr)
        cases = list(ET.parse(xml).getroot().iter("testcase"))
        failures = [c for c in cases if c.find("failure") is not None]
        errors = [c for c in cases if c.find("error") is not None]
        skips = [c for c in cases if c.find("skipped") is not None]
        assert not errors and not skips, (name, run.stdout, run.stderr)
        if mutant is None:
            assert run.returncode == 0 and len(cases) == 28 and not failures
        else:
            assert run.returncode == 1 and len(cases) == len(tests) and len(failures) == len(tests)
            assert all("AssertionError" in c.find("failure").get("message", "") for c in failures)
            mutant.unlink()
        results.append({"variant": name, "passed": len(cases)-len(failures), "assertion_failures": len(failures),
                        "errors": len(errors), "skipped": len(skips)})
    result = {"status": "PASS", "root_conftest": True, "production_reads": 0, "production_rows": 0,
              "variants": results}
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
