"""Reproduce CPU stub evidence + genuine source-deletion controls on a frozen tree.

Run with JAX_PLATFORMS=cpu, GPUWRF_JAX_CACHE=0, OPENBLAS_NUM_THREADS=1.
Arguments: immutable archive path, private lane output path.
No ALISIOS data, model forecast, GPU query, or file mutation in the archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive, out = args.archive.resolve(), args.output.resolve()
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get("GPUWRF_JAX_CACHE") != "0":
        raise RuntimeError("CPU-only/no-cache environment required")
    sys.path.insert(0, str(archive/"src"))
    from gpuwrf.optimize import twin

    # Existing gpuwrf/surface_params package bootstraps import JAX. Importing
    # a module is distinct from initializing a device backend (no query here).
    from jax._src import xla_bridge
    backend_dark = not xla_bridge.backends_are_initialized()
    assert backend_dark, "optional twin import unexpectedly initializes a backend"
    assert Path(twin.__file__).resolve() == archive/"src/gpuwrf/optimize/twin.py"
    tests = runpy.run_path(str(archive/"tests/v45_twin/test_station_twins.py"))
    start = time.perf_counter()
    report = tests["run"]()
    host_wall = time.perf_counter()-start
    assert report["scenarios"]["complete"]["status"] == "RECOVERED_SYNTHETIC_ONLY"
    assert report["scenarios"]["outages"]["status"] == "UNIDENTIFIABLE_FROZEN"
    out.mkdir(parents=True, exist_ok=True)
    # Function-level AST mutants avoid any changes to immutable source files.
    plugin = out/"v45_twin_mutant.py"
    plugin.write_text('''import ast, os

def pytest_sessionstart(session):
    # Root conftest must import cpu_guard BEFORE gpuwrf/JAX (the real guard).
    from gpuwrf.optimize import twin
    mode = os.environ["TWIN_MUTANT"]
    function = "_checked" if mode == "identity" else "run_twin"
    tree = ast.parse(open(twin.__file__).read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    targets = []
    for statement in ast.walk(node):
        if not isinstance(statement, ast.If):
            continue
        test = ast.unparse(statement.test)
        if (mode == "identity" and test == "batch.identity != identity"
                or mode == "rank" and test == "rank < p"):
            statement.test = ast.Constant(False)
            targets.append(statement)
    assert len(targets) == 1, "mutant did not match exactly one production guard"
    replacement = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(vars(twin))
    exec(compile(replacement, twin.__file__, "exec"), namespace)
    setattr(twin, function, namespace[function])
''')
    outcomes = {}
    for mode, selector, failures in (("identity", "test_wrong_identity_is_refused_before_eki", 6),
                                     ("rank", "test_deterministic_recovery_and_outage_rank_loss", 1)):
        env = dict(os.environ, TWIN_MUTANT=mode,
                   PYTHONPATH=os.pathsep.join([str(out), str(archive/"src")]))
        xml = out/f"mutant_{mode}.xml"
        argv = [sys.executable, "-m", "pytest", "-p", "v45_twin_mutant", "tests/v45_twin", "-q", "-k", selector,
                "--basetemp", str(out/"pytest"), "--junitxml", str(xml)]
        result = subprocess.run(argv, cwd=archive, env=env, capture_output=True, text=True, timeout=90)
        (out/f"mutant_{mode}.log").write_text(result.stdout+result.stderr)
        if not xml.exists():
            raise RuntimeError(f"mutant {mode}: setup rc={result.returncode}; see {out}/mutant_{mode}.log")
        suites = list(ET.parse(xml).getroot().iter("testsuite"))
        failed = sum(int(s.get("failures", 0)) for s in suites)
        errors = sum(int(s.get("errors", 0)) for s in suites)
        assert result.returncode == 1 and failed == failures and errors == 0, (mode, result.returncode, failed, errors)
        outcomes[mode] = {"rc": result.returncode, "genuine_test_failures": failed, "setup_errors": errors,
                          "selector": selector}
    # No backend initialization: source closure is an AST-only dependency walk.
    from gpuwrf.runtime.aot_cheap_key import _trace_reachable_source_files, _source_tree_content_digest

    package = archive/"src/gpuwrf"
    reachable = _trace_reachable_source_files(package)
    assert reachable is not None and package/"optimize/twin.py" not in reachable
    proof = {"scope": "CPU STUB ONLY; invented geometry; no WRF fidelity/real station/GPU claim",
             "import_backend_dark": backend_dark, "stub_wall_s": host_wall, "stub": report,
             "mutants": outcomes, "trace_source_files": len(reachable),
             "trace_source_digest": _source_tree_content_digest(package, reachable),
             "twin_outside_traced_source_closure": True,
             "sha256": {p: hashlib.sha256((archive/p).read_bytes()).hexdigest()
                        for p in ("src/gpuwrf/optimize/twin.py", "tests/v45_twin/test_station_twins.py")}}
    (out/"CPU_PROBE.json").write_text(json.dumps(proof, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "stub"}, indent=2))


if __name__ == "__main__":
    main()
