"""Delete the alias in memory and demand a real assertion failure (E39/E202)."""
from pathlib import Path
import inspect
import json
import sys
import textwrap

root, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "scripts/v025"))
import cpu_guard  # Before JAX, as in the test-suite entrypoint.
from gpuwrf.runtime import operational_mode as op
import pytest

original = op.OperationalNamelist.__post_init__
source = textwrap.dedent(inspect.getsource(original))
needle = '    if self.mp_physics == 17:\n        object.__setattr__(self, "mp_physics", 18)\n'
assert source.count(needle) == 1
scope = {}
exec(compile(source.replace(needle, ""), "<deleted-alias>", "exec"), original.__globals__, scope)
op.OperationalNamelist.__post_init__ = scope["__post_init__"]
exit_code = pytest.main([str(root / "tests/v040/test_nssl_legacy17.py") + "::test_api_normalizes_before_all_operator_selection",
                         "-q", "--tb=short", "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/s2-mpmore/pytest",
                         "--junitxml=" + str(output.with_suffix(".xml"))])
from xml.etree import ElementTree
suite = ElementTree.parse(output.with_suffix(".xml")).getroot().find("testsuite")
assert exit_code == 1 and suite.get("failures") == "1" and suite.get("errors") == "0", exit_code
failure = suite.find("testcase/failure")
assert "assert 17 == 18" in failure.text, failure.text
output.write_text(json.dumps(dict(mutant="remove mp17 normalization", killed=True, assertion=failure.get("message")), indent=2) + "\n")
