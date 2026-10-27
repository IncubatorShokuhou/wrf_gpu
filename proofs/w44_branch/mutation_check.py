"""Deletion controls on an immutable source snapshot, via in-memory mutants.

The snapshot is never edited. A kill requires a pytest assertion failure and no
setup/collection errors; a nonzero shell exit alone is not a scientific result.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("tree", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("mutant", choices=("identity", "schema", "edit", "finite", "array-hash", "consistency", "output-policy", "runtime-data", "reader-coverage"))
args = parser.parse_args()
assert os.environ.get("JAX_PLATFORMS") == "cpu"
sys.path.insert(0, str(args.tree / "scripts/v025"))
import cpu_guard  # noqa: F401 — required before any gpuwrf/JAX import
sys.path.insert(0, str(args.tree / "src"))
from gpuwrf.runtime import checkpoint_branch as branch
from gpuwrf.runtime import checkpoint_consistency as consistency
from gpuwrf.runtime import restart_store as storage
from gpuwrf.runtime import runtime_data_identity as data_identity

cases = {
    "identity": (branch, "branch_identity", "if expected != current:", "if False:", "test_undeclared_changes_refuse"),
    "schema": (branch, "branch_identity", 'if parent_manifest.get("branch_schema_version") != SCHEMA_VERSION:', "if False:", "test_old_schema_and_same_stream_refuse"),
    "edit": (branch, "apply_overrides", "result[key][field] = new", "result[key][field] = old", "test_all_nests_only_declared_arrays_change"),
    "finite": (consistency, "_finite_payload", "if not np.all(np.isfinite(value)):", "if False:", "test_null_branch_rejects_nonfinite_retained_leaf"),
    "array-hash": (branch, "_edit", 'if _file_digest(path) != operand["sha256"]:', "if False:", "test_array_hash_shape_and_race"),
    "consistency": (branch, "apply_overrides",
                    'reports[name] = validate(name, result, edits, changes.get("waivers", {}).get(name, {}), payload)',
                    'reports[name] = {"passed": True}',
                    "test_qv_edit_without_balance_waiver_refuses or test_negative_land_store_refuses or test_ph_edit_requires_waiver_and_retains_mu or test_state_skin_validation_cannot_be_hidden_by_retained_land"),
    "output-policy": (storage, "nested_identity", '"output_policy": _nested_output_policy(config, bundles),',
                      '"output_policy": {},', "test_empty_branch_refuses_each_output_flag"),
    "runtime-data": (storage, "nested_identity", '"runtime_data": runtime_data_identity(bundles),',
                     '"runtime_data": {},', "test_native_identity_binds_runtime_data_and_refuses_table_changes"),
    "reader-coverage": (data_identity, "check_reader_coverage", "if new:", "if False:",
                        "test_new_loader_read_of_unbound_file_fails_coverage or test_new_read_inside_existing_loader_is_not_hidden_by_module_whitelist"),
}
module, name, needle, replacement, selected = cases[args.mutant]
source = inspect.getsource(getattr(module, name))
assert source.count(needle) == 1
mutant = source.replace(needle, replacement)
exec(compile(mutant, f"<in-memory-mutant:{args.mutant}>", "exec"), module.__dict__)
args.out.mkdir(parents=True, exist_ok=True)
report = args.out / f"{args.mutant}.xml"
import pytest
testfile = "test_branch_consistency.py" if args.mutant == "consistency" else "test_checkpoint_branch.py"
if args.mutant == "output-policy":
    testfile = "test_restart_output_policy.py"
elif args.mutant in {"runtime-data", "reader-coverage"}:
    testfile = "test_runtime_data_identity.py"
rc = pytest.main([str(args.tree / "tests/v025/restart" / testfile), "-q", "-k", selected,
                  "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/w44-branch/pytest",
                  "--junitxml", str(report)])
if not report.exists():
    receipt = {"mutant": args.mutant, "killed": False, "pytest_rc": int(rc),
               "assertion_failures": 0, "setup_errors": 1, "reason": "pytest did not produce JUnit report"}
    (args.out / f"{args.mutant}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(1)
root = ET.parse(report).getroot()
failures, errors = root.findall(".//failure"), root.findall(".//error")
assertions = [item for item in failures if "AssertionError" in (item.text or "") or
              "DID NOT RAISE" in (item.text or "") or "Regex pattern did not match" in (item.text or "")]
killed = bool(assertions) and not errors
receipt = {"mutant": args.mutant, "killed": killed, "pytest_rc": int(rc),
           "assertion_failures": len(assertions), "setup_errors": len(errors),
           "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
           "mutant_sha256": hashlib.sha256(mutant.encode()).hexdigest(),
           "scope": "CPU structural deletion control; in-memory mutant of immutable source"}
(args.out / f"{args.mutant}.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
raise SystemExit(0 if killed else 1)
