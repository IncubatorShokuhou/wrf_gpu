"""Deletion-sensitive consumer checks in a fresh process per mutant.

Exec a single textual mutation in the module namespace; never edit a snapshot.
"""
import argparse
import json
from pathlib import Path
import sys

MUTATIONS = {
    "wrong_loss_sign": ("(d - (s + b + w - loss))[:-1]", "(d - (s + b + w + loss))[:-1]",
                        "test_known_answer_profiles_closure_and_read_only"),
    "second_destagger": ("a = a[:-1]", "a = .5*(a[:-1] + a[1:])",
                         "test_known_answer_profiles_closure_and_read_only"),
    "missing_as_zero": ('return {"status": "UNAVAILABLE", "reason": reason, "units": units, "value": None}',
                        'return {"status": "AVAILABLE", "reason": reason, "units": units, "value": 0.}',
                        "test_missing_diagnostics_are_unavailable_not_zero"),
    "source_change_ignored": ("if [_stat(p) for p in paths] != sources:", "if False:",
                              "test_changed_history_refuses_and_never_opens_for_write"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mutant", choices=MUTATIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The project guard must precede JAX/package imports (E50).
    sys.path.insert(0, str(Path.cwd() / "scripts/v025"))
    import cpu_guard  # noqa: F401
    from gpuwrf.optimize import diagnostics as D
    import pytest

    old, new, test = MUTATIONS[args.mutant]
    source = Path(D.__file__).read_text()
    assert source.count(old) == 1, "mutation seam moved"
    exec(compile(source.replace(old, new), D.__file__, "exec"), D.__dict__)
    code = pytest.main(["-q", "tests/test_v45_diag_profiles.py::" + test,
                        "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/v45-diag/pytest",
                        "--junitxml=" + str(args.output)])
    print(json.dumps(dict(mutant=args.mutant, pytest_rc=code)))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
