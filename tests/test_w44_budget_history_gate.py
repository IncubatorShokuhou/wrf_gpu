"""Committed real-data controls for the MYNN budget history gate (critic cr-w44-budget F1/A5).

Fixture: five real WN3 d03 day columns (0, 5, 13, 15, 24) at step 1 of the admitted CPU twin
P26_twin_day_final (actual side and the step inputs qv/qc/qi), the original public WRF driver
references and the measured 25-member original floors of P75_day_oracle, dt = 6 s. Gate rule
pre-registered at 49a3636970 plus the per-level k=4 quantization floor for RQV/RQC/RQI approved by
cr-w44-budget (00:37Z). Column 15 has RTHBLTEN ensemble-floor and RQIBLTEN quantization-floor
binding points; column 0 has the largest RQIBLTEN. No physics self-compare.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "proofs/w44_budget"))
from history_gate import EDMF, PBL, QUANT, TKE, evaluate, quantization_floor  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/w44_budget_history_gate_day_step1.npz"


@pytest.fixture(scope="module")
def arrays():
    with np.load(FIXTURE) as f:
        actual = {n: f["actual__" + n].copy() for n in TKE + EDMF + PBL}
        reference = {n: f["reference__" + n].copy() for n in TKE + EDMF + PBL}
        floors = {n: f["floor__" + n].copy() for n in ("Dtke", *PBL)}
        floors["inputs"] = {k: f["input__" + k].copy() for k in QUANT.values()}
        floors["dt"] = float(f["dt"])
    return actual, reference, floors


def _evaluate(actual, reference, floors):
    return evaluate(actual, reference, floors["Dtke"], pbl_floor={n: floors[n] for n in PBL},
                    inputs=floors["inputs"], dt=floors["dt"])


def test_admitted_real_columns_pass(arrays):
    actual, reference, floors = arrays
    result = _evaluate(actual, reference, floors)
    assert result["passed"] and result["closure"]["passed"]
    assert all(v["passed"] for v in result["terms"].values())


@pytest.mark.parametrize("name,mutate", [
    ("RQVBLTEN", lambda a: a * np.float32(1.5)),   # G1: survived the old absolute 1e-3 bound
    ("RQVBLTEN", np.zeros_like),                   # G2
    ("RQCBLTEN", lambda a: -a),                    # G3
    ("Dtke", lambda a: a * np.float32(2)),
    ("qBUOY", lambda a: a * np.float32(1.1)),
    ("RTHBLTEN", np.zeros_like),
    ("RUBLTEN", lambda a: -a),
    ("RVBLTEN", lambda a: a * np.float32(.5)),
    ("RQIBLTEN", lambda a: a * np.float32(1.5)),   # amendment controls (critic 00:37Z)
    ("RQIBLTEN", np.zeros_like),
    ("RQIBLTEN", lambda a: -a),
], ids=["G1_RQV_x1.5", "G2_RQV_zero", "G3_RQC_flip", "Dtke_x2", "qBUOY_x1.1", "RTH_zero", "RU_flip", "RV_x0.5",
        "RQI_x1.5", "RQI_zero", "RQI_flip"])
def test_scale_zero_and_sign_mutants_fail(arrays, name, mutate):
    actual, reference, floors = arrays
    assert np.count_nonzero(actual[name]), "a mutant on an all-zero term would be vacuous"
    bad = dict(actual)
    bad[name] = mutate(actual[name])
    result = _evaluate(bad, reference, floors)
    assert not result["passed"] and not result["terms"][name]["passed"]


def test_old_absolute_bound_was_vacuous_for_these_columns(arrays):
    # Documents F1: every G1-G3 error is far below the former absolute PBL bound 1e-3.
    actual, _, _ = arrays
    assert np.max(np.abs(actual["RQVBLTEN"] * np.float32(.5))) < 1e-3
    assert np.max(np.abs(2 * actual["RQCBLTEN"])) < 1e-3


def _binding_points(reference, floors, name):
    r = np.asarray(reference[name], np.float64)
    q = quantization_floor(floors["inputs"][QUANT[name]], r, floors["dt"])
    relative = .05 * np.max(np.abs(r), axis=-1)[:, None]
    ensemble = np.asarray(floors[name], np.float64)[:, None]
    return q, np.argwhere((q > relative) & (q > ensemble))


def test_ten_ulp_injection_at_a_quantization_binding_level_fails(arrays):
    actual, reference, floors = arrays
    q, points = _binding_points(reference, floors, "RQIBLTEN")
    assert len(points), "the fixture must contain a quantization-binding RQIBLTEN level"
    for column, level in points:
        bad = {k: v.copy() for k, v in actual.items()}
        step = np.float32(10 / 4 * q[column, level])            # 10 local ulp/dt (floor is 4 ulp/dt)
        bad["RQIBLTEN"][column, level] = np.float32(reference["RQIBLTEN"][column, level]) + step
        result = _evaluate(bad, reference, floors)
        assert not result["terms"]["RQIBLTEN"]["passed"]
        assert not result["terms"]["RQIBLTEN"]["columns"][column]["passed"]


def test_quantization_floor_definition():
    x = np.array([[1.385083e-06, 0.0, 1e-39]], np.float32)
    r = np.array([[1.8947807e-14, 0.0, 0.0]])
    q = quantization_floor(x, r, 6.0)
    m = max(abs(float(x[0, 0])), abs(float(x[0, 0]) + 1.8947807e-14 * 6.0))
    assert q[0, 0] == 4.0 * float(np.spacing(np.float32(m))) / 6.0   # = 4 local ulp(qi)/dt
    assert q[0, 1] == 0 and q[0, 2] == 0                         # zero and subnormal inputs: no floor


def test_quantization_floor_ignores_the_candidate(arrays):
    actual, reference, floors = arrays
    bad = {k: v.copy() for k, v in actual.items()}
    bad["RQIBLTEN"] = bad["RQIBLTEN"] * np.float32(1e6)
    q_ref, _ = _binding_points(reference, floors, "RQIBLTEN")
    assert np.array_equal(q_ref, quantization_floor(floors["inputs"]["qi"], reference["RQIBLTEN"], floors["dt"]))
    assert not _evaluate(bad, reference, floors)["terms"]["RQIBLTEN"]["passed"]
