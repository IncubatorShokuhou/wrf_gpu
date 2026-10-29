"""Scientific guard cases: a spatial tradeoff cannot hide behind a joint gain."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gpuwrf.optimize.multiobjective import assess as _assess, assess_file, command, markdown, _holm, _sign_p
from gpuwrf.optimize.reference import from_table, INSTANT_INTERVAL
from gpuwrf.optimize.evaluation import support_digest


def contract_for(obs, regions, scales, *, weights=None, margins=None, alpha=.05):
    """Explicit SYNTHETIC test authority fixture; never a real ALISIOS admission."""
    obs = obs.copy()
    if "case_id" not in obs:
        obs["case_id"] = "case"
    cases = {c: {"issue": pd.to_datetime(part.valid_time, utc=True).min().normalize().isoformat(),
                 "episode": c, "regime": "all"} for c,part in obs.groupby("case_id")}
    return {"authority": "ALISIOS", "contract_id": "SYNTHETIC_TEST_ONLY_NOT_AN_ADMISSION", "frozen": True,
            "support_sha256": support_digest(obs), "cases": cases, "regions": regions,
            "episode_weights": {c: 1 for c in cases}, "regime_weights": {"all": 1},
            "lead_bins": [{"name": "all", "start_hour": 0, "end_hour": 10000}],
            "scales": scales, "weights": weights or {v: 1 for v in scales},
            "margins": margins or {v: 0 for v in scales}, "alpha": alpha}


def assess(obs, candidates, reference, control, regions, **kwargs):
    contract = kwargs.pop("evaluation_contract", None)
    if contract is None:
        contract = contract_for(obs, regions, **{k:kwargs[k] for k in ("scales","weights","margins","alpha") if k in kwargs})
    if isinstance(reference, pd.DataFrame):
        reference = {"wn3_0p05": reference, "wn3_0p1": reference}
    return _assess(obs,candidates,reference,control,regions,evaluation_contract=contract,**kwargs)


def frames(days=24, variables=("t2", "wind", "rh", "cloud")):
    centers = {"t2": 20.0, "wind": 5.0, "rh": 50.0, "cloud": 0.5, "gust": 10.0}
    rows = [{"station_uid": station, "valid_time": time.isoformat(), "variable": v, "value": centers[v]}
            for time in pd.date_range("2026-01-01T12:00:00Z", periods=days, freq="D")
            for station in ("a", "b") for v in variables]
    truth = pd.DataFrame(rows)
    baseline = truth.copy()
    baseline["value"] += baseline.variable.map({"t2": 1, "wind": 1, "rh": 1, "cloud": .1, "gust":1})
    return truth, baseline


def reference_table(frame):
    table = frame.rename(columns={"station_uid": "location_id"}).copy()
    table["case_id"] = "case"
    table["variable"] = table.variable.map({"t2": "t2_c", "wind": "wind_speed_ms", "rh": "rh_pct", "cloud": "cloud_occurrence"})
    table["reference_role"] = "wn3_raw"
    table["units"] = table.variable.map({"t2_c": "degC", "wind_speed_ms": "m s-1", "rh_pct": "%", "cloud_occurrence": "1"})
    table["native_statistic"] = "ensemble_scalar_mean"
    table["native_interval"] = INSTANT_INTERVAL
    table["native_lead_hour"] = ((pd.to_datetime(table.valid_time, utc=True)-pd.Timestamp("2026-01-01T00:00Z")).dt.total_seconds()/3600).astype(int)
    table["operator_sha256"] = "a"*64
    table["calibrator_sha256"] = None
    table["available"] = True
    table["unavailable_reason"] = ""
    spec = {"path": "reference.parquet", "role": "wn3_raw", "issues": {"case": "2026-01-01T00:00:00Z"},
            "contracts": {v: {"native_statistic": "ensemble_scalar_mean", "native_interval": INSTANT_INTERVAL,
                               "operator_sha256": "a"*64, "calibrator_sha256": None} for v in sorted(set(table.variable))}}
    return table, spec


def evaluate(candidate=None, *, days=24, **kwargs):
    obs, base = frames(days)
    return assess(obs, {"candidate": base if candidate is None else candidate}, base, base,
                  {"a": "north", "b": "south"}, scales={"t2": 1, "wind": 1, "rh": 1, "cloud": .1},
                  repetitions=100, **kwargs)


def test_one_bad_region_overrides_joint_improvement_against_both_baselines():
    truth, base = frames()
    candidate = truth.copy()  # overall much better; north t2 becomes much worse
    candidate.loc[candidate.station_uid.eq("a") & candidate.variable.eq("t2"), "value"] += 3
    report = evaluate(candidate, parameters={"candidate": {"entrainment": 1.1}})
    assert report["scores"][0]["loss"] < 1
    assert not report["scores"][0]["admissible"]
    assert report["recommended"] is None
    bad = [r for r in report["tests"] if r["significantly_worse"]]
    assert {r["baseline"] for r in bad} == {"wn3_0p05", "wn3_0p1", "control"}
    assert {r["level"] for r in bad} == {"point", "region"}
    assert all(r["region"] == "north" and r["variable"] == "t2" for r in bad)
    cell = next(r for r in report["impact_matrix"] if r["variable"] == "t2" and r["region"] == "north")
    assert cell["effect"] == 2 and cell["interval95"] == [2, 2]
    assert cell["parameter"] == "entrainment" and not cell["admissible"]


def test_noise_only_and_exact_ties_are_admissible():
    _, base = frames()
    noise = base.copy()
    # Within every point/variable, alternating dates have opposite tiny errors.
    noise["value"] += np.repeat(np.tile([-.01, .01], 12), 8)
    result = evaluate(noise)
    assert result["recommended"] == "candidate"
    assert all(not t["significantly_worse"] for t in result["tests"])
    assert evaluate()["scores"][0]["admissible"]


def test_reference_and_control_are_independently_binding():
    obs, base = frames(24, ("t2",))
    worse = base.assign(value=base.value + 2)
    for reference, control, culprit in ((obs, worse, "reference"), (worse, obs, "control")):
        r = assess(obs, {"x": base}, reference, control, {"a": "r", "b": "r"}, scales={"t2": 1}, repetitions=100)
        assert not r["scores"][0]["admissible"]
        assert {t["baseline"] for t in r["tests"] if t["significantly_worse"]} == ({"wn3_0p05", "wn3_0p1"} if culprit == "reference" else {"control"})


@pytest.mark.parametrize("arm", ["observations", "candidate", "reference", "control"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_each_arm_is_refused_before_aggregation(arm, value):
    obs, base = frames(4, ("t2",))
    arms = {"observations": obs, "candidate": base.copy(), "reference": base.copy(), "control": base.copy()}
    arms[arm].loc[0, "value"] = value
    with pytest.raises(ValueError, match="NaN/Inf"):
        assess(arms["observations"], {"x": arms["candidate"]}, arms["reference"], arms["control"],
               {"a": "r", "b": "r"}, scales={"t2": 1}, repetitions=100)


@pytest.mark.parametrize("arm", ["candidate", "reference", "control"])
@pytest.mark.parametrize("change", ["station", "timestamp", "duplicate", "extra"])
def test_missing_or_extra_support_is_not_intersected_away(arm, change):
    obs, base = frames(4, ("t2",))
    arms = {k: base.copy() for k in ("candidate", "reference", "control")}
    frame = arms[arm]
    if change == "station":
        arms[arm] = frame[frame.station_uid.ne("b")]
    elif change == "timestamp":
        arms[arm] = frame.iloc[1:]
    elif change == "duplicate":
        arms[arm] = pd.concat([frame, frame.iloc[:1]])
    else:
        frame.loc[0, "valid_time"] = "2028-01-01T12:00:00Z"
    with pytest.raises(ValueError, match="support|duplicate"):
        assess(obs, {"x": arms["candidate"]}, arms["reference"], arms["control"],
               {"a": "r", "b": "r"}, scales={"t2": 1}, repetitions=100)


def test_missing_observation_station_and_insufficient_dates_are_not_admissible():
    obs, base = frames(4, ("t2",))
    with pytest.raises(ValueError, match="observation stations"):
        assess(obs[obs.station_uid.eq("a")], {"x": base}, base, base,
               {"a": "r", "b": "r"}, scales={"t2": 1}, repetitions=100)
    assert not evaluate(days=1)["scores"][0]["admissible"]


def test_holm_known_answers_and_family_includes_all_candidates():
    rows = [{"p_value": p} for p in (.01, .04, .03, 1)]
    _holm(rows, .05)
    assert [r["p_adjusted"] for r in rows] == pytest.approx([.04, .09, .09, 1])
    assert [r["significantly_worse"] for r in rows] == [True, False, False, False]
    obs, base = frames(4, ("t2",))
    r = assess(obs, {"z": base, "a": base}, base, base, {"a": "r", "b": "r"}, scales={"t2": 1}, repetitions=100)
    assert r["statistics"]["family_size"] == 18  # 2 candidates * (2 points+region) * 3 baselines
    assert [s["id"] for s in r["scores"]] == ["a", "z"]
    assert _sign_p(np.ones(10)) == pytest.approx(2**-10)
    assert _sign_p(np.zeros(10)) == 1
    # Exact baseline ties do not condition seven positive dates into a false
    # median-degradation discovery (the full 100-date median is still zero).
    assert _sign_p(np.r_[np.ones(7), np.zeros(93)]) == 1


def test_point_guard_catches_regression_hidden_inside_region():
    obs, base = frames(24, ("t2",))
    candidate = obs.copy()
    candidate.loc[candidate.station_uid.eq("a"), "value"] += 2
    result = assess(obs, {"x": candidate}, base, base, {"a": "r", "b": "r"}, scales={"t2": 1}, repetitions=100)
    assert not result["scores"][0]["admissible"]
    assert all(t["level"] == "point" for t in result["tests"] if t["significantly_worse"])


def test_margin_is_physical_units_and_weights_are_explicit():
    obs, base = frames(24, ("t2", "wind"))
    candidate = base.assign(value=base.value + .1)
    r = assess(obs, {"x": candidate}, base, base, {"a": "r", "b": "r"}, scales={"t2": 2, "wind": 1},
               weights={"t2": 3, "wind": 1}, margins={"t2": .2, "wind": .2}, repetitions=100)
    assert r["scores"][0]["admissible"]
    assert r["scores"][0]["loss"] == pytest.approx((3*1.1/2 + 1.1)/4)


def test_deterministic_under_candidate_row_and_region_order():
    obs, base = frames(8)
    args = dict(scales={"t2": 1, "wind": 1, "rh": 1, "cloud": .1}, repetitions=100,
                parameters={"a": {"p": 1, "q": 2}})
    one = assess(obs, {"z": base, "a": base}, base, base, {"a": "n", "b": "s"}, **args)
    two = assess(obs.iloc[::-1], {"a": base.iloc[::-1], "z": base.sample(frac=1, random_state=9)},
                 base.iloc[::-1], base.iloc[::-1], {"b": "s", "a": "n"}, **args)
    assert json.dumps(one, allow_nan=False) == json.dumps(two, allow_nan=False)
    assert all(r["joint_parameter_change"] for r in one["impact_matrix"] if r["candidate"] == "a")
    assert "associations" in markdown(one)


def test_distinct_cloud_pixels_and_station_registries():
    obs, base = frames(8, ("cloud", "wind"))
    obs = obs[(obs.variable.eq("cloud") & obs.station_uid.eq("a")) |
              (obs.variable.eq("wind") & obs.station_uid.eq("b"))]
    base = base.loc[obs.index]
    r = assess(obs, {"x": base}, base, base, {"a": "north", "b": "north"},
               scales={"cloud": .1, "wind": 1}, points={"cloud": ["a"], "wind": ["b"]}, repetitions=100)
    assert r["scores"][0]["admissible"] and r["statistics"]["family_size"] == 12


def test_different_hours_within_one_day_do_not_create_independent_blocks():
    obs, base = frames(24, ("t2",))
    obs["valid_time"] = base["valid_time"] = np.repeat(pd.date_range("2026-01-01T00:00Z", periods=24, freq="h"), 2)
    r = assess(obs, {"x": base}, obs, obs, {"a": "r", "b": "r"}, scales={"t2": 1}, repetitions=100)
    assert all(t["dates"] == 1 for t in r["tests"])
    assert not r["scores"][0]["admissible"]


@pytest.mark.parametrize("kwargs", [{"scales": {"t2": 0}}, {"scales": {"t2": float("nan")}},
    {"weights": {"t2": -1}}, {"margins": {"t2": -1}}, {"alpha": float("nan")},
    {"repetitions": True}, {"seed": -1}, {"parameters": {"x": {"p": float("inf")}}}])
def test_invalid_statistical_configuration(kwargs):
    obs, base = frames(4, ("t2",))
    options = {"scales": {"t2": 1}, "repetitions": 100, **kwargs}
    with pytest.raises(ValueError):
        assess(obs, {"x": base}, base, base, {"a": "r", "b": "r"}, **options)


def test_cli_assessment_real_files_json_md_and_no_overwrite(tmp_path):
    obs, base = frames(8, ("t2",))
    obs.to_csv(tmp_path/"obs.csv", index=False)
    base.to_csv(tmp_path/"forecast.csv", index=False)
    table, spec = reference_table(base)
    table.to_parquet(tmp_path/"reference.parquet", index=False)
    from gpuwrf.optimize.data import identity
    contract_path = tmp_path/"contract.json"
    contract_path.write_text(json.dumps(contract_for(obs, {"a":"r","b":"r"}, {"t2":1})))
    config = {"schema": "gpuwrf.optimize.multiobjective.v1", "observations": "obs.csv",
              "references": {"wn3_0p05":spec, "wn3_0p1":spec},
              "evaluation_contract": {"path":"contract.json", "sha256":identity(contract_path)["sha256"]},
              "control": "forecast.csv", "candidates": {"x": {"forecast": "forecast.csv", "parameters": {"p": 1}}},
              "regions": {"a": "r", "b": "r"}, "repetitions": 100}
    path = tmp_path/"assessment.json"
    path.write_text(json.dumps(config))
    result = assess_file(path)
    assert len(result["inputs"]) == 7 and result["recommended"] == "x"
    args = SimpleNamespace(assessment=path, output=tmp_path/"report.json")
    assert command(args) == 0
    assert json.loads(args.output.read_text())["recommended"] == "x"
    assert "| x | p | 1 | t2 | r | all | all | control |" in args.output.with_suffix(".md").read_text()
    assert command(args) == 2
    from gpuwrf.cli import build_parser
    parsed = build_parser().parse_args(["optimize", "assess", "--assessment", str(path), "--output", str(tmp_path/"new.json")])
    assert parsed.func(parsed) == 0


def test_protected_forecast_refused_before_read(tmp_path, monkeypatch):
    import gpuwrf.assimilation.letkf.member_driver as driver
    path = tmp_path/"assessment.json"
    path.write_text(json.dumps({"schema": "gpuwrf.optimize.multiobjective.v1", "observations": "protected.csv",
        "references": {"wn3_0p05":{"path":"ref.parquet"},"wn3_0p1":{"path":"ref.parquet"}},
        "evaluation_contract":{"path":"contract.json","sha256":"a"*64},"regions":{"a":"r"},
        "control": "ctl.csv", "candidates": {"x": {"forecast": "x.csv"}}}))
    calls = []
    def guard(where, paths, **kwargs):
        calls.append(where)
        if where.endswith("forecasts"):
            raise ValueError("protected file")
    monkeypatch.setattr(driver, "refuse_protected_inputs", guard)
    with pytest.raises(ValueError, match="protected file"):
        assess_file(path)
    assert calls == ["optimize assess", "optimize assess forecasts"]


@pytest.mark.parametrize("column,value,message", [
    ("available", False, "UNAVAILABLE"), ("available", "true", "UNAVAILABLE"),
    ("value", np.nan, "NaN/Inf"), ("value", np.inf, "NaN/Inf"),
    ("units", "K", "wrong units"), ("operator_sha256", "b"*64, "operator SHA"),
    ("calibrator_sha256", "b"*64, "unexpected calibrator"),
    ("native_lead_hour", 13, "native lead"), ("unavailable_reason", "missing", "unavailable_reason"),
    ("native_statistic", "vector_norm_mean", "UNAVAILABLE"),
    ("native_interval", "1h_mean", "UNAVAILABLE")])
def test_wn3_reference_contract_mutations_fail_closed(column, value, message):
    _, base = frames(4, ("t2",))
    table, spec = reference_table(base)
    table[column] = value
    with pytest.raises(ValueError, match=message):
        from_table(table, spec)


def test_wn3_calibrated_role_cannot_fall_back_to_raw():
    _, base = frames(4, ("t2",))
    table, spec = reference_table(base)
    spec["role"] = "wn3_calibrated"
    spec["contracts"]["t2_c"]["calibrator_sha256"] = "b"*64
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        from_table(table, spec)
    table["reference_role"] = "wn3_calibrated"
    table["calibrator_sha256"] = "b"*64
    assert len(from_table(table, spec)) == len(base)
    table["calibrator_sha256"] = "c"*64
    with pytest.raises(ValueError, match="calibrator SHA"):
        from_table(table, spec)


def test_structure_only_null_wn3_and_duplicates_cannot_score():
    _, base = frames(4, ("t2",))
    table, spec = reference_table(base)
    structure = table.assign(available=False, value=np.nan, unavailable_reason="FORMAT_ONLY")
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        from_table(structure, spec)
    with pytest.raises(ValueError, match="duplicate canonical"):
        from_table(pd.concat([table, table.iloc[:1]]), spec)


def test_overlapping_issues_are_distinct_keys():
    obs, base = frames(4, ("t2",))
    obs = pd.concat([obs.assign(case_id="one"), obs.assign(case_id="two")], ignore_index=True)
    base = pd.concat([base.assign(case_id="one"), base.assign(case_id="two")], ignore_index=True)
    r = assess(obs, {"x": base}, base, base, {"a": "r", "b": "r"}, scales={"t2": 1}, repetitions=100)
    assert r["support"]["rows"] == 16


def test_arrow_timestamp_unit_preserves_exact_support():
    obs, base = frames(4, ("t2",))
    microsecond = base.copy()
    microsecond["valid_time"] = pd.to_datetime(microsecond.valid_time, utc=True).astype("datetime64[us, UTC]")
    r = assess(obs, {"x": base}, microsecond, base, {"a": "r", "b": "r"}, scales={"t2": 1}, repetitions=100)
    assert r["recommended"] == "x"


def test_configuration_typos_cannot_silently_change_significance(tmp_path):
    path = tmp_path/"assessment.json"
    path.write_text(json.dumps({"schema": "gpuwrf.optimize.multiobjective.v1", "observations": "obs.csv",
        "reference": {"path": "ref.parquet"}, "control": "ctl.csv", "candidates": {"x": {"forecast": "x.csv"}},
        "regions": {"x": "r"}, "scales": {"t2": 1}, "alphaa": .01}))
    with pytest.raises(ValueError, match="unknown fields"):
        assess_file(path)


def test_two_wn3_resolutions_are_independently_binding_and_both_mandatory():
    obs, base = frames(24, ("t2",))
    regions = {"a":"r","b":"r"}
    for strong in ("wn3_0p05", "wn3_0p1"):
        refs = {name:(obs if name==strong else base.assign(value=base.value+1)) for name in ("wn3_0p05","wn3_0p1")}
        r = assess(obs,{"x":base},refs,base,regions,scales={"t2":1},repetitions=100)
        assert not r["scores"][0]["admissible"]
        assert {t["baseline"] for t in r["tests"] if t["significantly_worse"]} == {strong}
    with pytest.raises(ValueError,match="both WN3"):
        assess(obs,{"x":base},{"wn3_0p05":base},base,regions,scales={"t2":1},repetitions=100)


def test_frozen_pair_denominator_cannot_shrink_even_if_every_arm_drops_rows():
    obs, base = frames(24,("t2",))
    regions={"a":"r","b":"r"}
    contract=contract_for(obs,regions,{"t2":1})
    with pytest.raises(ValueError,match="pair denominator"):
        assess(obs.iloc[2:],{"x":base.iloc[2:]},base.iloc[2:],base.iloc[2:],regions,
               scales={"t2":1},evaluation_contract=contract,repetitions=100)


def test_fixed_episode_regime_weights_and_lead_regime_matrix():
    obs, base = frames(24,("t2",))
    obs=pd.concat([obs.assign(case_id="cold"),obs.assign(case_id="warm")],ignore_index=True)
    base=pd.concat([base.assign(case_id="cold"),base.assign(case_id="warm")],ignore_index=True)
    candidate=obs.copy()
    candidate["value"] += candidate.case_id.map({"cold":1.,"warm":9.})
    regions={"a":"r","b":"r"}
    contract=contract_for(obs,regions,{"t2":1})
    contract["cases"]["cold"]["regime"]="cold"
    contract["cases"]["warm"]["regime"]="warm"
    contract["regime_weights"]={"cold":1.,"warm":9.}
    r=assess(obs,{"x":candidate},base,base,regions,scales={"t2":1},evaluation_contract=contract,repetitions=100)
    assert r["scores"][0]["loss"] == pytest.approx(8.2)
    assert {t["regime"] for t in r["impact_matrix"]} == {"cold","warm"}
    assert {t["lead"] for t in r["impact_matrix"]} == {"all"}


def test_cloud_gain_cannot_average_away_gust_harm():
    obs, base = frames(24,("cloud","gust"))
    candidate=obs.copy()
    candidate.loc[candidate.variable.eq("gust"),"value"] += 2
    r=assess(obs,{"x":candidate},base,base,{"a":"r","b":"r"},scales={"cloud":.1,"gust":1},
             weights={"cloud":100,"gust":1},repetitions=100)
    assert r["scores"][0]["loss"] < 1 and not r["scores"][0]["admissible"]
    assert {t["variable"] for t in r["tests"] if t["significantly_worse"]} == {"gust"}


def test_simultaneous_intervals_disclose_unbounded_small_samples():
    r=evaluate(days=4)
    assert all(t["simultaneous_interval"]==[None,None] for t in r["tests"])
    assert all(t["interval_status"]=="UNBOUNDED_INSUFFICIENT_DATES" for t in r["tests"])
    assert r["statistics"]["simultaneous_confidence"] == .95
    obs, base=frames(24,("t2",))
    r=assess(obs,{"x":base},obs,obs,{"a":"r","b":"r"},scales={"t2":1},repetitions=100)
    assert all(t["simultaneous_interval"]==[1.,1.] for t in r["tests"])


def test_optimizer_cannot_override_external_gate_or_cohort_weights():
    obs, base=frames(24,("t2",))
    regions={"a":"r","b":"r"}
    contract=contract_for(obs,regions,{"t2":1})
    with pytest.raises(ValueError,match="optimizer cannot set gates"):
        assess(obs,{"x":base},base,base,regions,scales={"t2":1},alpha=.1,evaluation_contract=contract,repetitions=100)
    contract["episode_weights"]={}
    with pytest.raises(ValueError,match="episode_weights"):
        assess(obs,{"x":base},base,base,regions,scales={"t2":1},evaluation_contract=contract,repetitions=100)


def test_fixed_episode_weights_within_a_regime():
    obs, base=frames(24,("t2",))
    obs=pd.concat([obs.assign(case_id="a"),obs.assign(case_id="b")],ignore_index=True)
    base=pd.concat([base.assign(case_id="a"),base.assign(case_id="b")],ignore_index=True)
    candidate=obs.copy()
    candidate["value"] += candidate.case_id.map({"a":1.,"b":9.})
    regions={"a":"r","b":"r"}
    contract=contract_for(obs,regions,{"t2":1})
    contract["episode_weights"]={"a":1.,"b":9.}
    result=assess(obs,{"x":candidate},base,base,regions,scales={"t2":1},evaluation_contract=contract,repetitions=100)
    assert result["scores"][0]["loss"] == pytest.approx(8.2)


def test_objective_identity_binds_denominator_truth_baselines_and_contract_not_candidates():
    obs, base=frames(24,("t2",))
    regions={"a":"r","b":"r"}
    one=assess(obs,{"x":base},base,base,regions,scales={"t2":1},repetitions=100)
    two=assess(obs,{"y":obs},base,base,regions,scales={"t2":1},repetitions=100)
    assert one["objective_identity"]==two["objective_identity"]
    three=assess(obs,{"x":base},base.assign(value=base.value+1),base,regions,scales={"t2":1},repetitions=100)
    assert three["objective_identity"]!=one["objective_identity"]
    assert one["status"]=="EVALUATED" and one["scores"][0]["status"]=="ADMISSIBLE"
    assert evaluate(days=1)["scores"][0]["status"]=="INSUFFICIENT_DATES"
