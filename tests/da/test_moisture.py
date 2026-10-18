"""Counted moisture policy, water budget and admissibility gates (A1 R7/R10).

Hand references: dry-air layer mass from the hybrid full-level pressure written out by hand, added water =
clipped deficit x layer mass, the Bolton/WRF saturation formula evaluated directly. The R10 regression: one
negative member cell must no longer flip an otherwise identical analysis between admitted and refused.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwrf.assimilation.letkf import analysis as an
from gpuwrf.assimilation.letkf import config as dacfg
from gpuwrf.assimilation.letkf import moisture as mst
from gpuwrf.assimilation.letkf import obs as ob
from gpuwrf.assimilation.letkf import state_io

from _synth import PARAMS, ensemble, hx, hydrostatic_case, obs_at


def _geo(nz=3, ny=2, nx=2):
    return {"C3F": np.array([1.0, 0.6, 0.25, 0.0])[: nz + 1], "C4F": np.zeros(nz + 1), "P_TOP": 5000.0,
            "MUB": np.full((ny, nx), 95000.0)}


def test_layer_dry_mass_matches_hand_hybrid_pressure():
    geo = _geo()
    mu = np.array([[[100.0, -200.0], [0.0, 50.0]]])
    mass, label = mst.layer_dry_mass(geo, mu)
    mut = 95000.0 + mu[0]
    pf = [c * mut + 5000.0 for c in (1.0, 0.6, 0.25, 0.0)]
    hand = np.stack([(pf[k] - pf[k + 1]) / 9.81 for k in range(3)])
    assert np.allclose(mass[0], hand, rtol=1e-15) and label.startswith("WRF hybrid")
    assert np.isclose(mass[0].sum(axis=0), mut / 9.81).all()          # column dry mass = MU_total / g


def test_clip_budget_one_cell_hand_value():
    geo = _geo()
    mass, _ = mst.layer_dry_mass(geo, np.zeros((2, 2, 2)))
    qv = np.full((2, 3, 2, 2), 0.004)
    qv[1, 2, 0, 1] = -3.0e-5
    policy = mst.MoisturePolicy()
    out, raw, counts = mst.negative_step(qv, policy)
    assert counts["negative_cells"] == 1 and counts["clipped_cells"] == 1 and out.min() == 0.0
    budget = mst.water_budget(raw, out, mass, geo)
    added = 3.0e-5 * (0.25 * 95000.0) / 9.81          # deficit x layer-3 dry mass (C3F 0.25 -> 0)
    assert budget["added_column_max_kg_m2"] == pytest.approx(added, rel=1e-12)
    total_m1 = 0.004 * 95000.0 / 9.81 * 4 - 0.004 * (0.25 * 95000.0) / 9.81    # member 1 after the clip
    assert budget["added_fraction_by_member"][1] == pytest.approx(added / total_m1, rel=1e-12)
    assert budget["added_fraction_by_member"][0] == 0.0 and budget["columns_with_added_water"] == 1
    ok, reasons = mst.gate(policy, counts, budget)
    assert ok and not reasons
    strict = mst.MoisturePolicy(max_added_column_kg_m2=added / 2)
    ok, reasons = mst.gate(strict, counts, budget)
    assert not ok and "kg m-2" in reasons[0]
    frac = mst.MoisturePolicy(max_added_fraction=1e-9)
    assert not mst.gate(frac, counts, budget)[0]


def test_refuse_policy_keeps_values_and_refuses():
    qv = np.array([[0.001, -1e-9]])
    out, raw, counts = mst.negative_step(qv, mst.MoisturePolicy(negative="refuse"))
    assert np.array_equal(out, qv) and counts["negative_cells"] == 1
    ok, reasons = mst.gate(mst.MoisturePolicy(negative="refuse"), counts, None)
    assert not ok and "refuse" in reasons[0]


def test_nonfinite_is_always_refused():
    ok, reasons = mst.gate(mst.MoisturePolicy(), {"negative_cells": 0, "nonfinite_cells": 2},
                           {"added_column_max_kg_m2": 0.0, "added_fraction_max": 0.0})
    assert not ok and "non-finite" in reasons[0]
    for bad in ({"added_column_max_kg_m2": float("nan"), "added_fraction_max": 0.0},
                {"added_column_max_kg_m2": 0.0, "added_fraction_max": float("inf")}):
        ok, reasons = mst.gate(mst.MoisturePolicy(), {"negative_cells": 1, "nonfinite_cells": 0}, bad)
        assert not ok and "not finite" in reasons[0]


def test_log_transform_round_trip_and_positivity():
    q = np.array([0.0, 1e-9, 1e-6, 0.012])
    floor = 1e-7
    back = mst.from_log(mst.to_log(q, floor), floor)
    assert np.allclose(back, q, rtol=1e-12, atol=1e-18)
    assert (mst.from_log(np.array([-60.0, -30.0]), floor) >= -floor).all()


def test_policy_validation():
    with pytest.raises(ValueError):
        mst.MoisturePolicy(negative="drop")
    with pytest.raises(ValueError):
        mst.MoisturePolicy(supersaturation="rain")
    with pytest.raises(ValueError):
        mst.MoisturePolicy(max_added_fraction=-1.0)
    with pytest.raises(ValueError):
        mst.MoisturePolicy(negative="log", log_floor_kgkg=0.0)


def test_saturation_formula_and_keep_only_supersaturation():
    t = np.array([[[[290.0]]]])
    p = np.array([[[[90000.0]]]])
    theta = t / (p / 1.0e5) ** (287.0 / 1004.5) - 300.0
    es = 611.2 * np.exp(17.67 * (290.0 - 273.15) / (290.0 - 29.65))
    qsat = 287.0 / 461.6 * es / (90000.0 - es)
    assert mst.saturation_mixing_ratio(t, p)[0, 0, 0, 0] == pytest.approx(qsat, rel=1e-12)
    counts = mst.supersaturation_counts(np.full_like(t, qsat * 1.1), theta, p)
    assert counts["supersaturated_cells"] == 1 and counts["rh_max"] == pytest.approx(1.1, rel=1e-9)
    # cap/to_cloud would act after the boundary taper and change condensate the rebalance/writers do not carry
    for refused in ("cap", "to_cloud"):
        with pytest.raises(ValueError, match="refused in this build"):
            mst.MoisturePolicy(supersaturation=refused)
    with pytest.raises(dacfg.ConfigError, match="refused in this build"):
        dacfg.validate({"schema": "gpuwrf.da.v1", "t_a": "2023-01-15T00:00:00Z",
                        "ensemble": {"kind": "members", "member_files": ["a", "b"]},
                        "obs": {"sources": [{"format": "csv_surface", "path": "o.csv"}]},
                        "moisture": {"supersaturation": "to_cloud"},
                        "experiments": [{"name": "A", "assimilate": ["surface"]}]}, base_dir=Path("/nonexistent"))


def test_supersaturated_analysis_is_counted_and_never_changed():
    frames, geo = ensemble(seed=8)
    frames["QVAPOR"] = np.full_like(frames["QVAPOR"], 0.05)       # far above saturation everywhere
    obs = obs_at([(2, 16, 18)], [frames["T"][:, 2, 16, 18].mean() + 0.5], [0.5], frames, geo)
    xa, receipt = an.analyse(frames, geo, obs, PARAMS, model_equivalents=hx)
    assert receipt["moisture"]["supersaturation"]["supersaturated_cells"] > 0 and receipt["admissible"]
    taper_zone = np.s_[:, :, :5]                                     # untouched rows stay bitwise
    assert np.array_equal(xa["QVAPOR"][taper_zone], frames["QVAPOR"][taper_zone])


def test_rebalance_refuses_a_hydrometeor_change():
    geo, bg = hydrostatic_case()
    bg = {k: np.asarray(v, np.float64) for k, v in bg.items()}
    bg["QCLOUD"] = np.zeros_like(bg["QVAPOR"])
    dq = np.zeros_like(bg["QVAPOR"])
    dq[2:6, 6, 7] = 0.001
    an_ = dict(bg, QVAPOR=bg["QVAPOR"] - dq, QCLOUD=dq)              # a pure QV -> QC phase transfer
    with pytest.raises(ValueError, match=r"\['QCLOUD'\] changed"):
        state_io.rebalance_increment(bg, an_, geo, return_increments=True)


# ------------------------------------------------------------------------------------------- in the analysis

def _dry_case(obs_value=6.0e-5, spread=0.5, sigma=3.0e-5, seed=7):
    """Dry level (lognormal positive background) observed drier than the ensemble: the linear update pushes some
    members below zero. spread 0.5 / obs 6e-5 -> exactly one negative member cell; obs 6.5e-5 -> none."""
    frames, geo = ensemble(seed=seed, qv=2.0e-4, qv_spread=spread, qv_lognormal=True)
    assert frames["QVAPOR"].min() > 0
    o = obs_at([(1, 15, 17), (1, 14, 16)], [obs_value, obs_value], [sigma, sigma], frames, geo, var=ob.QV)
    return frames, geo, o


def test_analysis_clip_is_counted_budgeted_and_admitted():
    frames, geo, o = _dry_case()
    xa, receipt = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx)
    m = receipt["moisture"]
    assert m["negative"]["cells"] == 1, "fixture must produce one negative member cell"
    assert receipt["admissible"] and m["admissible"] and receipt["refusal_reasons"] == []
    assert 0 < m["budget"]["added_column_max_kg_m2"] < 1e-2 and m["dry_mass"].startswith("WRF hybrid")
    assert m["written"]["negative"] == 0 and np.asarray(xa["QVAPOR"]).min() >= 0


def test_large_deficit_is_refused_by_the_budget():
    frames, geo, o = _dry_case(obs_value=1.0e-6, spread=1.2, sigma=2.0e-5)
    _, receipt = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx)
    assert receipt["moisture"]["negative"]["cells"] > 100
    assert not receipt["admissible"] and any("kg m-2" in r for r in receipt["refusal_reasons"])
    assert receipt["counters"]["rebalance"] in ("off", "skipped (not admissible)")


def test_refuse_policy_and_tight_budget_refuse_with_reasons():
    frames, geo, o = _dry_case()
    _, refused = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx,
                            moisture=mst.MoisturePolicy(negative="refuse"))
    assert not refused["admissible"] and "policy 'refuse'" in refused["refusal_reasons"][0]
    _, tight = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx,
                          moisture=mst.MoisturePolicy(max_added_column_kg_m2=1.0e-12))
    assert not tight["admissible"] and any("kg m-2" in r for r in tight["refusal_reasons"])


def test_one_negative_cell_no_longer_flips_admissibility():
    """R10: two near-identical analyses (0 vs 1 negative member cell) get the same decision under the default
    counted policy; the old binary rule ('refuse') flips between them."""
    decisions = {}
    for value in (6.5e-5, 6.0e-5):
        frames, geo, o = _dry_case(obs_value=value)
        _, default = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx)
        _, binary = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx,
                               moisture=mst.MoisturePolicy(negative="refuse"))
        decisions[value] = (default["moisture"]["negative"]["cells"], default["admissible"], binary["admissible"])
    assert decisions[6.5e-5] == (0, True, True)
    assert decisions[6.0e-5] == (1, True, False)


def test_log_policy_analysis_is_positive_and_clips_at_most_the_floor():
    frames, geo, o = _dry_case(obs_value=1.0e-6, spread=1.2, sigma=2.0e-5)
    policy = mst.MoisturePolicy(negative="log")
    xa, receipt = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx, moisture=policy)
    assert receipt["admissible"] and np.asarray(xa["QVAPOR"]).min() >= 0
    assert receipt["moisture"]["negative"]["min_kgkg"] >= -policy.log_floor_kgkg
    assert receipt["moisture"]["budget"]["added_column_max_kg_m2"] < 1e-3


def test_background_negatives_outside_the_update_are_kept_bitwise():
    frames, geo, o = _dry_case(obs_value=1.0e-4)
    frames["QVAPOR"][:, 0, 0, 0] = -1.0e-7        # corner = taper 0: the analysis never touches it
    xa, receipt = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx)
    assert np.array_equal(xa["QVAPOR"][:, 0, 0, 0], frames["QVAPOR"][:, 0, 0, 0].astype(xa["QVAPOR"].dtype))
    assert receipt["moisture"]["negative"]["background_negative_cells_kept"] == 10
    assert receipt["admissible"] and receipt["moisture"]["written"]["inherited_negative"] == 10


# ------------------------------------------------------------------------------------------- analysis-mean wrfinput

def _mean_case(below):
    """Control + mean member increment for QVAPOR that ends ``below`` kg/kg under zero at the given cells."""
    from _synth import hydrostatic_case

    geo, ctrl = hydrostatic_case()
    n = 4
    bg = {"QVAPOR": np.repeat(ctrl["QVAPOR"][None], n, axis=0).astype(np.float32),
          "T": np.repeat(ctrl["T"][None], n, axis=0)}
    an_q = bg["QVAPOR"].astype(np.float64)
    for (k, j, i), value in below.items():
        an_q[:, k, j, i] = -value
    analysis = {"QVAPOR": an_q.astype(np.float32), "T": bg["T"]}
    return geo, ctrl, bg, analysis


def test_mean_wrfinput_clip_is_counted_and_budgeted():
    from gpuwrf.assimilation.letkf import state_io

    geo, ctrl, bg, an_ = _mean_case({(18, 6, 7): 2.0e-6})          # slightly below zero in one upper-level cell
    counters = {}
    out = state_io.analysis_mean_fields(ctrl, bg, an_, geo, names=("T", "QVAPOR"), counters=counters)
    assert counters["mean_QVAPOR_negative"] == 1 and counters["admissible"] and out["QVAPOR"][18, 6, 7] == 0.0
    mut = float(geo["MUB"][6, 7]) + float(ctrl["MU"][6, 7])
    layer = (float(geo["C3F"][18]) - float(geo["C3F"][19])) * mut / 9.81
    stored = float(np.float32(float(ctrl["QVAPOR"][18, 6, 7]) + (np.float64(np.float32(-2.0e-6))
                                                                  - float(ctrl["QVAPOR"][18, 6, 7]))))
    assert counters["mean_budget"]["added_column_max_kg_m2"] == pytest.approx(-stored * layer, rel=1e-5)
    assert "P" in out and "PH" in out                                 # rebalanced


def test_mean_wrfinput_budget_breach_and_refuse_are_refused_without_rebalance():
    from gpuwrf.assimilation.letkf import state_io

    geo, ctrl, bg, an_ = _mean_case({(2, 6, 7): 1.0e-2})           # a large low-level deficit
    counters = {}
    out = state_io.analysis_mean_fields(ctrl, bg, an_, geo, names=("T", "QVAPOR"), counters=counters)
    assert not counters["admissible"] and "kg m-2" in counters["refusal_reasons"][0] and "P" not in out
    geo, ctrl, bg, an_ = _mean_case({(18, 6, 7): 2.0e-6})
    counters = {}
    out = state_io.analysis_mean_fields(ctrl, bg, an_, geo, names=("T", "QVAPOR"), counters=counters,
                                        moisture=mst.MoisturePolicy(negative="refuse"))
    assert not counters["admissible"] and "refuse" in counters["refusal_reasons"][0] and "P" not in out


def test_budget_ignores_float32_storage_rounding():
    """float32 frames: a clean analysis (no negative cell) must report exactly zero added water."""
    frames, geo, o = _dry_case(obs_value=1.0e-4)
    frames = {k: np.asarray(v, np.float32) for k, v in frames.items()}
    xa, receipt = an.analyse(frames, geo, o, PARAMS, model_equivalents=hx)
    budget = receipt["moisture"]["budget"]
    assert receipt["moisture"]["negative"]["cells"] == 0 and xa["QVAPOR"].dtype == np.float32
    assert budget["added_column_max_kg_m2"] == 0.0 and budget["columns_with_added_water"] == 0
