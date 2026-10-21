"""teacher_qual.py on mock scorer rows (no model values). usage: python3 -m pytest -q analysis/test_teacher_qual.py"""
import json, subprocess, sys
from pathlib import Path
from test_verdict import V, write

HERE = Path(__file__).parent


def case(tmp, name, x, c, integ={"pass": True}):
    d = tmp / name
    write(d / "x", x)
    cd = None
    if c is not None:
        write(d / "c", c); cd = d / "c"
    (d / "VERDICT.json").write_text(json.dumps(V.verdict(name, d / "x", cd, integ)))
    return f"{name}=WN3:{d}:{cd or '-'}"


def run(tmp, *specs):
    subprocess.run([sys.executable, str(HERE / "teacher_qual.py"), str(tmp / "q.json"), str(tmp / "q.md"), *specs], check=True)
    return json.loads((tmp / "q.json").read_text())


def test_wind_qualified_and_blocks(tmp_path):
    q = run(tmp_path, case(tmp_path, "a", {"U10": {50: 1.2}, "T2": {10: 0.75}}, {"U10": {50: 1.0}, "T2": {}}))
    assert q["wind_tau_gt40_h"]["verdict"] == "QUALIFIED"
    d3 = q["cases"][0]["domains"]["d03"]
    assert d3["wind_tau_gt40"] == "FLOOR-LIMITED" and d3["classes"]["T2/Q2"]["max_X_over_L_0-23"] == 0.75


def test_wind_held_on_fail_after_40h_only(tmp_path):
    q = run(tmp_path, case(tmp_path, "a", {"U": {38: 0.5, 45: 1.5}}, None))
    assert q["wind_tau_gt40_h"]["verdict"] == "HELD" and q["cases"][0]["domains"]["d03"]["wind_tau_gt40"] == "FAIL"


def test_wind_ignores_breach_at_40(tmp_path):
    q = run(tmp_path, case(tmp_path, "a", {"V10": {30: 1.1}}, {"V10": {30: 1.0}}))   # tau 30 floor-limited, not > 40
    assert q["cases"][0]["domains"]["d03"]["wind_tau_gt40"] == "PASS"


def test_missing_case_holds(tmp_path):
    q = run(tmp_path, case(tmp_path, "a", {"U10": {}}, {"U10": {}}), f"b=ERA5:{tmp_path/'nope'}")
    assert q["wind_tau_gt40_h"]["verdict"] == "HELD" and q["cases"][1]["case_teacher"] == "INCOMPLETE"


def test_incomplete_integrity_holds(tmp_path):
    q = run(tmp_path, case(tmp_path, "a", {"U10": {}}, {"U10": {}}, None))
    assert q["wind_tau_gt40_h"]["verdict"] == "HELD"
