"""Confirm the producer-leak mutant fails numerical gates, independently of API-placement checks."""
from pathlib import Path
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

root, output = map(lambda s: Path(s).resolve(), sys.argv[1:3])
producer = root / "src/gpuwrf/physics/noahmp_coupler.py"
test = root / "tests/v025/t_cfix_mynn/test_f20_mynn_caller.py"
original_producer, original_test = producer.read_text(), test.read_text()
old = "**_mynn_surface_exner_kwargs(state, r))"
new = "**_mynn_surface_exner_kwargs(state, r), pbl_limits=True)"
source_only_check = '    assert "pbl_limits" not in calls[0]\n'
assert original_producer.count(old) == original_test.count(source_only_check) == 1
xml = output.with_suffix(".xml")
try:
    producer.write_text(original_producer.replace(old, new))
    test.write_text(original_test.replace(source_only_check, ""))
    process = subprocess.run(
        [sys.executable, "-m", "pytest", str(test), "-q", "-k", "test_extreme_fluxes_reach_cumulus_unlimited",
         "-p", "no:cacheprovider", "--basetemp", str(output.parent / "pytest_fix_behavioral"),
         "--junitxml", str(xml)], cwd=root, env=dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0"),
        capture_output=True, text=True, timeout=300)
finally:
    producer.write_text(original_producer)
    test.write_text(original_test)
output.with_suffix(".log").write_text(process.stdout + process.stderr)
suites = list(ET.parse(xml).getroot().iter("testsuite"))
record = {"scope": "M5 producer-leak mutation; only the source-placement assertion omitted from the test",
          "returncode": process.returncode,
          "failures": sum(int(s.attrib["failures"]) for s in suites),
          "errors": sum(int(s.attrib["errors"]) for s in suites),
          "tests": sum(int(s.attrib["tests"]) for s in suites),
          "numeric_operand_assertion": "assert np.array_equal(got[which], reference[which])" in process.stdout,
          "restored": producer.read_text() == original_producer and test.read_text() == original_test}
output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
assert record["returncode"] == 1 and record["tests"] == record["failures"] == 3
assert record["errors"] == 0 and record["numeric_operand_assertion"] and record["restored"]
