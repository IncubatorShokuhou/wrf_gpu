"""Genuine in-process deletion mutants; CPU-only, immutable model tree.
Usage: python p2_mutants.py SOURCE_ROOT OUTPUT_PREFIX
"""
import ast
import inspect
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root, prefix = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path[:0] = [str(root / "scripts/v025"), str(root / "src"), str(root / "tests")]
import cpu_guard  # must precede JAX
import pytest
from gpuwrf.tracers import runtime
from gpuwrf.io import wrfout_writer as writer, scheme_catalog as catalog

records = []
original_bound = runtime._real32_bound
original_batch = writer._build_wrfout_host_arrays
note = catalog._IMPLEMENTED_NOTES[("tracer_opt", 2)]
for mode, selection in (
    ("source-bound", ["tests/test_tracer_runtime.py", "-k", "accepted_json_source_overflow"]),
    ("history-stack", ["tests/test_tracer_history.py", "-k", "actual_host_batch_copies_only_requested"]),
    ("catalog-note", ["tests/test_scheme_catalog_fail_closed.py::test_implemented_tracer_reason_retains_qualification_disclosure"]),
):
    if mode == "source-bound":
        runtime._real32_bound = lambda value, description: value
    elif mode == "history-stack":
        function = ast.parse(inspect.getsource(original_batch)).body[0]
        for index, statement in enumerate(function.body):
            if isinstance(statement, ast.If) and ast.unparse(statement.test) == "not leaves":
                function.body.insert(index, ast.parse("visit(sources[0].tracers)").body[0])
                break
        else:
            raise AssertionError("production batch leaf gate not found")
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(module, writer.__file__, "exec"), writer.__dict__)
    else:
        catalog._IMPLEMENTED_NOTES.pop(("tracer_opt", 2))
    xml = prefix.with_name(prefix.name + "_" + mode + ".xml")
    status = pytest.main(["-q", "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/u2-tracers/pytest",
                          "--junitxml=" + str(xml), *selection])
    report = ET.parse(xml).getroot()
    failures, errors = len(report.findall(".//failure")), len(report.findall(".//error"))
    records.append({"mutation": mode, "pytest_status": int(status), "assertion_failures": failures,
                    "setup_errors": errors, "xml": str(xml)})
    assert int(status) == 1 and failures > 0 and errors == 0, records[-1]
    runtime._real32_bound = original_bound
    writer._build_wrfout_host_arrays = original_batch
    catalog._IMPLEMENTED_NOTES[("tracer_opt", 2)] = note
prefix.with_suffix(".json").write_text(json.dumps({"source": str(root), "gpu_executed": False,
    "mutants": records, "cache": "NONE", "files_modified": False}, indent=2) + "\n")
