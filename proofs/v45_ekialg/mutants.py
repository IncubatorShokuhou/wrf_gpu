"""Actual in-process source deletions; require numerical/refusal assertions, no setup errors/skips."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


VARIANTS = [
    ("anomaly-normalisation", "np.sqrt(dt / (n - 1))", "np.sqrt(dt / n)",
     "test_v45_ensemble_matches_independent_dense"),
    ("prior-augmentation", "if problem.tikhonov:  # augment", "if False:  # augment",
     "test_v45_ensemble_matches_independent_dense"),
    ("block-whitening", "np.linalg.solve(factor, part.T).T", "part",
     "test_v45_covariance_whitening_and_sampling_orientation"),
    ("finite-first", "if not all(np.all(np.isfinite(x)) for x in (theta, g, problem.obs, problem.prior_mean, problem.prior_cov)):",
     "if False:", "test_v45_finite_guard_precedes_covariance_and_reduction"),
    ("dense-guard", "_dense_guard(self.size)  # check BEFORE", "pass  # check BEFORE",
     "test_v45_dense_guard_precedes_allocation"),
    ("tempering", "np.sqrt(dt / (n - 1))", "np.sqrt(1 / (n - 1))",
     "test_v45_ensemble_matches_independent_dense"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--child", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.child is not None:
        # Root conftest requires cpu_guard before ANY gpuwrf/JAX import, even for NumPy-only tests.
        sys.path.insert(0, str(Path.cwd() / "scripts/v025"))
        import cpu_guard  # noqa: F401
        import pytest
        from gpuwrf.surface_params import eki
        name, old, new, test = VARIANTS[args.child]
        source = inspect.getsource(eki)
        assert source.count(old) == 1, (name, source.count(old))
        exec(compile(source.replace(old, new), eki.__file__, "exec"), eki.__dict__)
        raise SystemExit(pytest.main(["-q", f"tests/test_v45_eki_algebra.py::{test}",
                                     "--basetemp", str(args.output.parent / "pytest"),
                                     "--junitxml=" + str(args.output / (name+".xml"))]))
    results = []
    for index, (name, *_rest) in enumerate(VARIANTS):
        run = subprocess.run([sys.executable, __file__, str(args.output), "--child", str(index)],
                             capture_output=True, text=True, timeout=90)
        (args.output / (name+".log")).write_text(run.stdout + run.stderr)
        cases = list(ET.parse(args.output / (name+".xml")).getroot().iter("testcase"))
        failures = [c.find("failure") for c in cases if c.find("failure") is not None]
        errors = [c for c in cases if c.find("error") is not None]
        skips = [c for c in cases if c.find("skipped") is not None]
        assert run.returncode == 1 and failures and not errors and not skips, (name, run.returncode)
        assert all("AssertionError" in f.get("message", "") for f in failures), name
        results.append(dict(name=name, assertion_failures=len(failures), errors=0, skips=0))
    (args.output / "RESULT.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
