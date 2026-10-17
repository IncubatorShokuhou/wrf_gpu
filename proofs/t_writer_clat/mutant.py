"""Restore the actual parent payload builder in both imported writer routes."""
from pathlib import Path
import ast
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/v025"), str(ROOT / "src")]
import cpu_guard
import pytest


class RestoreParent:
    def pytest_sessionstart(self, session):
        # conftest must establish import-time native/precision flags first.
        from gpuwrf.io import wrfout_writer as writer
        from gpuwrf.integration import nested_pipeline as nested
        raw = subprocess.check_output(["git", "show", "f737666b6:src/gpuwrf/io/wrfout_writer.py"], cwd=ROOT)
        tree = ast.parse(raw)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_wrfout_payload")
        namespace = dict(vars(writer))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<actual-unfixed-payload-builder>", "exec"), namespace)
        writer.prepare_wrfout_payload = namespace[fn.name]
        nested.prepare_wrfout_payload = namespace[fn.name]


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "tests/test_writer_clat_feedback.py", "-k", "not selected", *sys.argv[1:]],
                               plugins=[RestoreParent()]))
