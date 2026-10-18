"""Swiss twin OSSE gate arithmetic (proofs/a1_da/ens/osse_tables.py): an incomplete, duplicated or non-finite
report never passes (critic cr-w41-da-ens delta 12:45Z: gate_prospective passed on the 03Z row alone)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "proofs" / "a1_da" / "ens"))
import osse_tables as ot  # noqa: E402

V = ("T", "U", "V")


def _row(c, prior_sp, prior_rmse, post=True):
    r = {"cycle": c, "t_a": f"2023-01-15_{3 * (c + 1):02d}:00:00", "prior_spread": dict(zip(V, prior_sp)),
         "prior_rmse": dict(zip(V, prior_rmse)), "first_hour_retention": None}
    if post:
        r.update({"post_spread": {v: 0.5 for v in V}, "post_rmse": {v: 0.6 for v in V},
                  "written_spread": {v: 0.5 for v in V}})
    return r


def _arm():
    return {"rows": [_row(0, (1, 1, 1), (1, 1, 1)), _row(1, (1, 1, 1), (1, 1, 1)), _row(2, (1, 1, 1), (1, 1, 1)),
                     _row(3, (1, 1, 1), (1, 1, 1), post=False)]}


def test_complete_report_is_gated():
    g = ot.gate_prospective(_arm())
    assert g["PASS"] and g["G1"] and g["G2_no_compounding_decay"] and g["G3"] and g["G4p"]
    bad = _arm()
    bad["rows"][2]["prior_spread"] = {v: 3.0 for v in V}        # spread/RMSE 3 at 09Z -> G1 fails
    assert ot.gate_prospective(bad)["PASS"] is False


def test_incomplete_duplicate_or_nonfinite_never_passes():
    first_only = {"rows": _arm()["rows"][:1]}
    assert ot.gate_prospective(first_only) == {"PASS": False, "INCOMPLETE": ["cycles [0] != [0, 1, 2, 3]"]}
    dup = _arm()
    dup["rows"].append(copy.deepcopy(dup["rows"][1]))
    assert ot.gate_prospective(dup)["PASS"] is False and "INCOMPLETE" in ot.gate_prospective(dup)
    nan = _arm()
    nan["rows"][1]["post_rmse"]["U"] = float("nan")
    assert "INCOMPLETE" in ot.gate_prospective(nan)
    no_written = _arm()
    del no_written["rows"][0]["written_spread"]
    assert "INCOMPLETE" in ot.gate(no_written, _arm())


def test_registered_times_and_row_order():
    """Critic 12:59Z: rows must carry their registered valid time (03/06/09/12Z) and the G2 baseline is the cycle-0
    (03Z) prior whatever the row order."""
    wrong = _arm()
    for r in wrong["rows"]:
        r["t_a"] = "2023-01-15_03:00:00"                         # all four rows claim 03Z
    assert "INCOMPLETE" in ot.gate_prospective(wrong)
    missing = _arm()
    del missing["rows"][3]["t_a"]
    assert "INCOMPLETE" in ot.gate_prospective(missing)
    decay = _arm()
    decay["rows"][3]["prior_spread"] = {v: 0.4 for v in V}        # 12Z spread < 0.5 x 03Z -> G2 fails
    decay["rows"][3]["prior_rmse"] = {v: 0.4 for v in V}          # (G1 stays inside the band)
    ordered = ot.gate_prospective(decay)
    shuffled = {"rows": [decay["rows"][i] for i in (3, 1, 0, 2)]}   # 12Z row first
    assert ordered["G2_no_compounding_decay"] is False and ordered["PASS"] is False
    assert ot.gate_prospective(shuffled) == ordered


def test_critic_baseline_scenario_and_repeated_time():
    """cr-w41-da-ens delta2: 03Z spread 2, 06Z spread .7, other priors 1 -> the registered floor is 1 (0.5 x 03Z), so
    .7 fails in ANY row order; a row carrying another cycle's time (06Z twice) is refused."""
    a = _arm()
    a["rows"][0]["prior_spread"] = {v: 2.0 for v in V}
    a["rows"][0]["prior_rmse"] = {v: 2.0 for v in V}
    a["rows"][1]["prior_spread"] = {v: 0.7 for v in V}
    a["rows"][1]["prior_rmse"] = {v: 0.7 for v in V}
    for order in ((0, 1, 2, 3), (3, 0, 1, 2), (3, 2, 1, 0), (1, 3, 0, 2)):
        g = ot.gate_prospective({"rows": [a["rows"][i] for i in order]})
        assert g["G2_no_compounding_decay"] is False and g["PASS"] is False, order
    rep = _arm()
    rep["rows"][2]["t_a"] = rep["rows"][1]["t_a"]                 # 06Z repeated, 09Z missing
    assert "INCOMPLETE" in ot.gate_prospective(rep)
