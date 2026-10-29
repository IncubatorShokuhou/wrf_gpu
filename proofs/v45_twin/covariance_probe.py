"""F1 source-regression/deletion and bounded host-admission cost; no model/GPU.

Args: immutable archive, private output directory, parent repository.
Run CPU-only, cache disabled, one BLAS thread, in the lane resource scope.
"""
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tracemalloc
import xml.etree.ElementTree as ET

import numpy as np

BASELINE = "3a8bea124e7555e65ef1eaad17fabbf209cf50b9"


def main():
    archive, out, repo = map(lambda p: Path(p).resolve(), sys.argv[1:])
    assert os.environ["JAX_PLATFORMS"] == "cpu" and os.environ["GPUWRF_JAX_CACHE"] == "0"
    sys.path.insert(0, str(archive/"src"))
    from gpuwrf.optimize import twin

    assert Path(twin.__file__).resolve() == archive/"src/gpuwrf/optimize/twin.py"
    original = subprocess.check_output(["git", "-C", str(repo), "show",
                                        BASELINE+":src/gpuwrf/optimize/twin.py"], text=True)
    # Extract the EXACT old admission statements, retaining its original bug.
    run = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == "run_twin")
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "gamma")
    end = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "calls")
    function = ast.parse("def baseline(sd, covariance):\n    m = len(sd)\n    return gamma, chol").body[0]
    function.body[1:1] = run.body[start:end]
    code = ast.unparse(ast.fix_missing_locations(function))
    namespace = {"np": np}
    exec(code, namespace)
    baseline = namespace["baseline"]
    out.mkdir(parents=True, exist_ok=True)
    plugin = out/"v45_covariance_mutant.py"
    plugin.write_text("import numpy as np\n"+code+"\n"
                      "def pytest_sessionstart(session):\n"
                      "    from gpuwrf.optimize import twin\n"
                      "    twin._observation_covariance = baseline\n")
    xml = out/"covariance_mutant.xml"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(out), str(archive/"src")]))
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "v45_covariance_mutant",
                             "tests/v45_twin", "-q", "-k", "test_critic_tiny_nonsymmetric_covariances",
                             "--basetemp", str(out/"pytest"), "--junitxml", str(xml)],
                            cwd=archive, env=env, capture_output=True, text=True, timeout=90)
    (out/"covariance_mutant.log").write_text(result.stdout+result.stderr)
    suites = list(ET.parse(xml).getroot().iter("testsuite"))
    failures = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    assert result.returncode == 1 and failures == 2 and errors == 0, (result.returncode, failures, errors)
    # The same SPD input exercises both admissions at the default row ceiling.
    sd = np.full(256, 0.05)
    covariance = np.diag(sd**2)
    covariance[0, 1] = covariance[1, 0] = 0.2*sd[0]*sd[1]
    costs = {}
    for name, function in (("baseline", baseline), ("candidate", twin._observation_covariance)):
        function(sd, covariance)
        samples = []
        for _ in range(21):
            begin = time.perf_counter()
            function(sd, covariance)
            samples.append(time.perf_counter()-begin)
        gc.collect()
        tracemalloc.start()
        admitted, chol = function(sd, covariance)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        np.testing.assert_array_equal(admitted, covariance)
        np.testing.assert_allclose(chol@chol.T, covariance, rtol=1e-14, atol=0)
        costs[name] = {"median_s": statistics.median(samples), "samples_s": samples,
                       "peak_traced_host_bytes": peak}
    proof = {"scope": "CPU-only covariance admission; synthetic matrix; shared-core timing, not a GPU benchmark",
             "baseline_commit": BASELINE, "baseline_admission_sha256": hashlib.sha256(code.encode()).hexdigest(),
             "candidate_sha256": hashlib.sha256(Path(twin.__file__).read_bytes()).hexdigest(),
             "baseline_mutant": {"genuine_failures": failures, "setup_errors": errors, "rc": result.returncode},
             "rows": len(sd), "costs": costs,
             "factorizations_per_admission": 1, "GPU_model_work": "NONE",
             "thresholds": "recovery/rank/misfit unchanged; numerical symmetry tolerance normalized by diagonal scales"}
    (out/"COVARIANCE_PROBE.json").write_text(json.dumps(proof, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in proof.items() if k != "costs"}, indent=2))
    print(json.dumps({name: {k: v for k, v in row.items() if k != "samples_s"} for name, row in costs.items()}, indent=2))


if __name__ == "__main__":
    main()
