"""Restore the old raw MP_RE selection; require a genuine caller failure."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

root, output = map(Path, sys.argv[1:3])
assert os.environ["JAX_PLATFORMS"] == "cpu"
sys.path[:0] = [str(root/"src"), str(root/"scripts/v025")]
import cpu_guard  # precedes JAX and pytest's later conftest import
from gpuwrf.coupling import ssib_surface_hook as hook
import pytest

old = subprocess.check_output(["git", "-C", str(root), "show",
                               "d85aad784:src/gpuwrf/coupling/ssib_surface_hook.py"], text=True)
node = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name == "ssib_radiation")
exec(compile(ast.Module(body=[node], type_ignores=[]), hook.__file__, "exec"), hook.__dict__)
code = pytest.main([str(root/"tests/test_t_ferrier_mpre.py") +
                    "::test_ssib_two_radiation_calls_without_mp_radii[1]", "-q",
                    "--basetemp", str(output.parent/"pytest"), "--junitxml", str(output)])
assert code == 1, code
suite = ET.parse(output).getroot().find("testsuite")
assert suite.attrib["failures"] == "1" and suite.attrib["errors"] == "0"
failure = suite.find("testcase/failure").text
assert "MP_RE requires held microphysics radii seeded before radiation" in failure
print("MUTANT KILLED: actual MP_RE caller refusal; zero setup errors", flush=True)
