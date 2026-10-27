"""Pair legacy checkpoint-consistency outcomes with the pinned main callback."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--before", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
sys.path[:0] = [str(a.repo / "src"), str(a.repo / "tests/w44_override")]
from test_override import make_input
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.runtime import checkpoint_consistency as current
from gpuwrf.runtime.checkpoint_branch import apply_overrides
spec = importlib.util.spec_from_file_location("main_checkpoint_before_cap", a.before)
before = importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
a.out.mkdir(parents=True, exist_ok=True)
make_input(a.out / "wrfinput_d01")
with Dataset(a.out / "wrfinput_d01") as ds:
    f = {name: np.asarray(v[0]).copy() for name, v in ds.variables.items() if name != "Times"}
state = {leaf: f[key] for leaf, key in current.STATE_TO_WRF.items() if key in f}
state["theta"] = ((f["T"].astype(np.float64) + 300) * (1 + (461.6 / 287.) * state["qv"])).astype(np.float32)
for leaf, perturbation, base in (("p_total", "P", "PB"), ("ph_total", "PH", "PHB"), ("mu_total", "MU", "MUB")):
    state[leaf] = f[perturbation] + f[base]
base = SimpleNamespace(pb=f["PB"], phb=f["PHB"], mub=f["MUB"])
payload = {"state_fields": state, "noahmp_land_fields": {"smois": f["SMOIS"], "sh2o": f["SMOIS"].copy(), "tslb": f["TSLB"]},
           "extra_fields": {"base_state": base}, "scratch_fields": {}}
snapshot = {"carries": {"d01": payload}, "own_steps": {"d01": 100}, "driver_state": {"outputs": {"d01": []}}}


def capture(module, bundles, changes, source):
    try:
        _, result = apply_overrides(copy.deepcopy(source), changes, validate=module.checkpoint_validator(a.out, bundles))
        value = {"kind": "return", "report": result}
    except Exception as exc:
        value = {"kind": "refusal", "type": type(exc).__name__, "message": str(exc), "report": getattr(exc, "report", None)}
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


rows = []
for label, edits, waivers, empty_bundles, legacy in (
        ("empty", {}, {}, True, False), ("empty_with_namelist", {}, {}, False, False),
        ("legacy_hvt", {}, {}, False, True),
        ("noop_qv", {"qv": {"multiply": 1.}}, {}, False, False),
        ("changed_qv_refusal", {"qv": {"multiply": .99}}, {}, False, False),
        ("changed_qv_waiver", {"qv": {"multiply": .99}}, {"hydrostatic_consistency": "fixed-MU research branch"}, False, False)):
    nl = SimpleNamespace(sf_surface_physics=4, surface_controls={"hvt": np.ones((2, 3), np.float32)} if legacy else None)
    bundles = {} if empty_bundles else {"d01": SimpleNamespace(namelist=nl)}
    changes = {"schema_version": 1, "domains": {"d01": {"state": edits}}, "waivers": {"d01": waivers}}
    old, new = (capture(module, bundles, changes, snapshot) for module in (before, current))
    assert old == new, (label, old, new)
    rows.append({"case": label, "byte_equal": True, "report_sha256": hashlib.sha256(new).hexdigest()})
report = {"source": str(a.repo), "before_main_pin": "c4a829ece4", "before_sha256": hashlib.sha256(a.before.read_bytes()).hexdigest(),
          "rows": rows, "passed": True, "model_executed": False}
(a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"paired_reports": len(rows), "byte_equal": True}), flush=True)
