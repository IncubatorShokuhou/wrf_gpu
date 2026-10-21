"""Rule tests for verdict.py on mock scorer rows (no model values). usage: python3 -m pytest -q analysis/test_verdict.py"""
import json, math
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("verdict", Path(__file__).with_name("verdict.py"))
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)


def write(d, vals, lim=1.0, metric="rmse"):
    """vals: {field: {lead: value}}; all other leads 0."""
    d.mkdir(parents=True, exist_ok=True)
    for dom in ("d01", "d02", "d03"):
        fs = {}
        for f, per in vals.items():
            fs[f] = {"by_lead": [{"lead_h": t, "tolerance_result": {"checks": {
                metric: {"limit": lim, "value": per.get(t, 0.0) if dom == "d03" else 0.0},
                "finite_pair_fraction_min": {"limit": 1.0, "value": 1.0}}}} for t in range(73)]}
        (d / f"{dom}.json").write_text(json.dumps({"field_summaries": fs, "pairing": {"common_leads_h": list(range(73))}}))


def run(tmp, x, c, integ={"pass": True}):
    write(tmp / "x", x)
    if c is not None:
        write(tmp / "c", c)
    return V.verdict("mock", tmp / "x", (tmp / "c") if c is not None else None, integ)


def test_pass(tmp_path):
    r = run(tmp_path, {"T2": {30: 0.9}}, {"T2": {}})
    assert r["case_teacher"] == "TEACHER-OK" and r["domains"]["d03"]["bounded_worst"] == "PASS"


def test_strict24_ignores_floor(tmp_path):   # critic counterexample tau12 L=1 C=2 X=2
    r = run(tmp_path, {"T2": {12: 2.0}}, {"T2": {12: 2.0}})
    assert r["breaches"][0]["why"] == "strict24" and r["case_teacher"] == "FAIL"


def test_tau24_is_strict(tmp_path):
    r = run(tmp_path, {"T2": {24: 1.1}}, {"T2": {24: 5.0}})
    assert r["breaches"][0]["status"] == "FAIL"


def test_floor_exact_boundary(tmp_path):
    r = run(tmp_path, {"RAINNC": {50: 4.0}}, {"RAINNC": {50: 3.0}})       # X = 4/3 C exactly
    assert r["breaches"][0]["status"] == "FLOOR-LIMITED" and r["case_teacher"] == "TEACHER-OK"


def test_floor_exceeded_even_if_C_gt_L(tmp_path):
    r = run(tmp_path, {"RAINNC": {50: 4.01}}, {"RAINNC": {50: 3.0}})
    b = r["breaches"][0]
    assert b["status"] == "FAIL" and b["frozen_A3"] == "FLOOR-LIMITED"   # stricter than frozen A3's C>L clause


def test_no_twin_breach_fails(tmp_path):
    r = run(tmp_path, {"W": {60: 1.2}}, None)
    assert r["breaches"][0]["why"] == "no own twin" and r["case_teacher"] == "FAIL"


def test_nonfinite_fails(tmp_path):
    r = run(tmp_path, {"T2": {40: math.nan}}, {"T2": {40: 5.0}})
    assert r["breaches"][0]["why"] == "nonfinite"
    r = run(tmp_path / "b", {"T2": {40: 1.1}}, {"T2": {40: math.inf}})
    assert r["breaches"][0]["status"] == "FAIL"


def test_integrity_gates(tmp_path):
    assert run(tmp_path, {"T2": {}}, {"T2": {}}, {"pass": False})["case_teacher"] == "FAIL"
    assert run(tmp_path / "b", {"T2": {}}, {"T2": {}}, None)["case_teacher"] == "INCOMPLETE"
    assert run(tmp_path / "e", {"T2": {}}, {"T2": {}}, {"pass": None, "registry_verdict": "ERROR x"})["case_teacher"] == "INCOMPLETE"


def test_incomplete_leads(tmp_path):
    write(tmp_path / "x", {"T2": {}})
    j = tmp_path / "x" / "d02.json"; d = json.loads(j.read_text()); d["pairing"]["common_leads_h"] = list(range(72))
    j.write_text(json.dumps(d))
    assert V.verdict("m", tmp_path / "x", None, {"pass": True})["case_teacher"] == "INCOMPLETE"


def test_nonfinite_twin_never_silent(tmp_path):
    r = run(tmp_path, {"T2": {40: 0.5}}, {"T2": {40: math.nan}})
    assert r["breaches"] and r["breaches"][0]["why"] == "nonfinite" and r["case_teacher"] == "FAIL"
