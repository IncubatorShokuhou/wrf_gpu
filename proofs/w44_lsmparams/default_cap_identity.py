"""Pair both public APIs with the pinned main checker before the approved hunk."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--before", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
sys.path[:0] = [str(a.repo / "src"), str(a.repo / "tests/w44_override")]
from test_override import make_input
from gpuwrf.io import wrfinput_consistency as current
from gpuwrf.io.netcdf_lock import Dataset
spec = importlib.util.spec_from_file_location("w2_main_checker_before_cap", a.before)
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
a.out.mkdir(parents=True, exist_ok=True)
path = make_input(a.out / "wrfinput_d01")
with Dataset(path) as ds:
    fields, meta = current._file_view(ds)
    arrays = {name: np.asarray(value).copy() for name, value in fields.items()}


def capture(fn, args, kwargs):
    try:
        data = {"kind": "return", "report": fn(*args, **kwargs)}
    except Exception as exc:
        data = {"kind": "refusal", "type": type(exc).__name__, "message": str(exc), "report": getattr(exc, "report", None)}
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


rows = []
for label, changed, value, checkpoint in (
        ("unchanged", (), .2, False), ("soil_good", ("SMOIS",), .2, False),
        ("soil_bad", ("SMOIS",), .4, False), ("checkpoint", (), .2, True),
        ("checkpoint_soil", ("SMOIS",), .2, True), ("tslb", ("TSLB",), .2, False)):
    data = {name: value_.copy() for name, value_ in arrays.items()}
    data["SMOIS"][:, :, 0, 0] = value
    with Dataset(path, "r+") as ds:
        ds["SMOIS"][:, :, 0, 0] = value
    for api in ("validate_fields", "validate_wrfinput"):
        kwargs = {"changed_fields": changed}
        args = (data, copy.deepcopy(meta)) if api == "validate_fields" else (path,)
        if api == "validate_fields":
            kwargs["checkpoint"] = checkpoint
        old = capture(getattr(before, api), args, kwargs)
        for explicit_none in (False, True):
            new_kwargs = {**kwargs, **({"effective_smcmax": None} if explicit_none else {})}
            new = capture(getattr(current, api), args, new_kwargs)
            assert old == new, (label, api, explicit_none, old, new)
            rows.append({"case": label, "api": api, "explicit_none": explicit_none,
                         "report_sha256": hashlib.sha256(new).hexdigest(), "byte_equal": True})
report = {"source": str(a.repo), "before_source": str(a.before), "before_main_pin": "c4a829ece4",
          "before_source_sha256": hashlib.sha256(a.before.read_bytes()).hexdigest(),
          "rows": rows, "passed": all(row["byte_equal"] for row in rows), "model_executed": False}
(a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"paired_reports": len(rows), "byte_equal": report["passed"]}), flush=True)
