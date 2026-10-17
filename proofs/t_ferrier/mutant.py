"""Delete both TIMEAN lowering fixes in memory; expect two lowering failures."""
import ast
import os
from pathlib import Path
import sys

root, output = map(Path, sys.argv[1:3])
assert os.environ["JAX_PLATFORMS"] == "cpu"
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts/v025"))
import cpu_guard  # must precede JAX, including pytest's later conftest import
from gpuwrf.physics.noahmp import groundwater_options as gw
import pytest

source = Path(gw.__file__).read_text()
fixed = "jnp.exp(jnp.asarray(-0.,jnp.float32)-timean)"
assert source.count(fixed) == 2
mutant = ast.parse(source.replace(fixed, "jnp.exp(-timean)"))
for node in mutant.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("groundwater_step", "_equilibrium_soilwater"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), gw.__file__, "exec"), gw.__dict__)
code = pytest.main([str(root / "tests/test_t_ferrier_groundwater_lower.py"), "-q",
                    "--basetemp", str(output.parent / "pytest"), "--junitxml", str(output)])
assert code == 1, code
import xml.etree.ElementTree as ET
suites = ET.parse(output).getroot().findall("testsuite")
assert sum(int(s.attrib["failures"]) for s in suites) == 2
assert sum(int(s.attrib["errors"]) for s in suites) == 0
failures = [f.text for f in ET.parse(output).getroot().iter("failure")]
assert all("TypedNdArray" in f and "AttributeError" in f for f in failures)
print("MUTANT KILLED: two product CUDA lowering failures; zero setup errors", flush=True)
