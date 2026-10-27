"""Launcher E41 verdict (E230): parse wave_gate.json, never the gate tool's rc.

The fixture is the sealed v0.4.3 release-bundle receipt (real GPU cache, 14 entries, compact AOT d01/d02/
fused_d02 -> INCOMPLETE with STACK PASS); the other cases are minimal edits of that real receipt.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOL = HERE.parents[1] / "proofs/w44_thparams/e41_verdict.py"
spec = importlib.util.spec_from_file_location("thparams_e41_verdict", TOOL)
e41 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e41)
REAL = json.loads((HERE / "fixtures/e41_v043_bundle_wave_gate.json").read_text())


def write(tmp_path, gate):
    (tmp_path / "wave_gate.json").write_text(json.dumps(gate))
    return tmp_path


def measured(gate):
    for row in gate["entries"]:
        row["kinput_transpose_status"], row["kinput_transpose_over500"] = "MEASURED_RICH_JIT_HLO", []
    return gate


def test_real_bundle_receipt_is_incomplete_and_acceptable(tmp_path):
    record = e41.verdict(write(tmp_path, REAL))
    assert REAL["e41_gate"] == "INCOMPLETE" and record["e41_gate"] == "INCOMPLETE"
    assert record["acceptable"] and record["kinput_status"] == ["MEASURED_RICH_JIT_HLO", "UNKNOWN_AOT_METADATA"]
    assert record["max_stack_B"] <= 256 and record["n_entries"] == 14


def test_pass_is_acceptable(tmp_path):
    gate = measured(copy.deepcopy(REAL))
    gate["e41_gate"] = "PASS"
    assert e41.verdict(write(tmp_path, gate))["acceptable"]


def test_stack_fail_is_recorded_not_acceptable(tmp_path):
    gate = copy.deepcopy(REAL)
    gate["entries"][0]["elfs"][0]["stack_over256"] = [["spill_kernel", 528]]
    gate["e41_gate"], gate["hard_stack_gate"] = "FAIL", "FAIL"
    record = e41.verdict(write(tmp_path, gate))
    assert record["e41_gate"] == "FAIL" and not record["acceptable"]


@pytest.mark.parametrize("edit", ["hlo_error", "callgraph_unknown"])
def test_other_unknowns_are_not_acceptable(tmp_path, edit):
    gate = copy.deepcopy(REAL)
    if edit == "hlo_error":
        gate["entries"][0]["kinput_transpose_status"] = "UNKNOWN_HLO_ERROR"
    else:
        gate["entries"][0]["elfs"][0]["callgraph_unknown"] = [{"callee": "x"}]
    record = e41.verdict(write(tmp_path, gate))
    assert record["e41_gate"] == "INCOMPLETE" and not record["acceptable"]


def test_missing_receipt_is_recorded(tmp_path):
    record = e41.verdict(tmp_path)
    assert record["e41_gate"] == "MISSING" and not record["acceptable"]
    out = tmp_path / "verdict.json"
    assert e41.main([str(tmp_path), str(out)]) == 0 and json.loads(out.read_text())["e41_gate"] == "MISSING"


def test_inconsistent_receipt_is_refused(tmp_path):
    gate = copy.deepcopy(REAL)
    gate["e41_gate"] = "FAIL"  # FAIL without a STACK or callee/kInput flag cannot come from the gate tool
    assert e41.main([str(write(tmp_path, gate)), str(tmp_path / "v.json")]) == 1
    assert not (tmp_path / "v.json").exists()
