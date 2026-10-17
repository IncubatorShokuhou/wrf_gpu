"""Analytic station errors, chronological mutants and fake scheduler integration.

Synthetic admission receipts below exercise binding, never certify real GPU physics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from gpuwrf.optimize import calibration
from gpuwrf.optimize.data import forecast, identity, metrics, paired, read_csv, validate, write_json
from gpuwrf.optimize.study import PHYSICS, config_id, evaluate, load_study, model_digest, namelist_text, schedule, score

REPO = Path(__file__).resolve().parents[1]


def station_file(root):
    path = root/"stations.csv"
    pd.DataFrame({"station_uid": ["a", "b"], "lat": [0.25, 0.75], "lon": [0.25, 0.75],
                  "elevation_m": [0, 0]}).to_csv(path, index=False)
    return path


def rows(days=range(1, 7), *, error=0, variable="t2"):
    return pd.DataFrame([{"station_uid": s, "valid_time": f"2026-01-{day:02d}T01:00:00Z",
                          "variable": variable, "value": day*2+offset+error}
                         for day in days for s, offset in (("a", 0), ("b", 1))])


def fake_case(root, day):
    from gpuwrf.io.netcdf_lock import Dataset
    case = root/f"case-{day}"
    case.mkdir()
    stamp = f"2026-01-{day:02d}"
    for name, times in (("wrfinput_d01", [f"{stamp}_00:00:00"]),
                        ("wrfbdy_d01", [f"{stamp}_00:00:00", f"{stamp}_03:00:00"])):
        with Dataset(case/name, "w") as ds:
            ds.createDimension("Time", len(times))
            ds.createDimension("DateStrLen", 19)
            ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(t) for t in times], dtype="S1")
    (case/"namelist.input").write_text(f"""&time_control
 start_year=2026, start_month=1, start_day={day}, start_hour=0,
 /
 &domains
 max_dom=1, time_step=3600,
 /
 &physics
 mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=4,
 ra_lw_physics=4, ra_sw_physics=4, cu_physics=0, radt=30, cudt=0,
 /
 &dynamics
 diff_opt=1, km_opt=4,
 /
""")
    return case


@pytest.fixture
def study_files(tmp_path):
    from gpuwrf import __version__
    stations = station_file(tmp_path)
    obs = tmp_path/"observations.csv"
    rows().to_csv(obs, index=False)
    cases = [{"id": f"c{d}", "date": f"2026-01-{d:02d}", "input_dir": str(fake_case(tmp_path, d))} for d in range(1, 7)]
    evidence = tmp_path/"SYNTHETIC_ONLY.txt"
    evidence.write_text("fake scheduler receipt; no GPU evidence or qualification claim")
    admissions = []
    for cu in (0, 1, 5, 93):
        physics = dict(zip(PHYSICS, (8, 5, 5, 4, 4, 4, cu)))
        receipt = {"schema": "gpuwrf.optimize.admission.v1", "physics": physics, "max_dom": 1,
                   "status": "GPU_SMOKED_ADMITTED", "model_sha256": model_digest(), "gpuwrf_version": __version__,
                   "evidence": [identity(evidence)]}
        path = tmp_path/f"admit-{cu}.json"
        write_json(path, receipt)
        admissions.append(str(path))
    study = {"schema": "gpuwrf.optimize.study.v1", "region": "synthetic", "objective": "T2", "hours": 3,
             "budget_case_hours": 42, "test_start": "2026-01-05", "cases": cases, "admissions": admissions,
             "stations": str(stations), "observations": str(obs)}
    path = tmp_path/"study.json"
    write_json(path, study)
    return path, study


def inventory(study, root):
    mapping = {}
    # Analytic training errors 0,1,5,93; held-out cu=0 error=7 to prove
    # the winner is selected before test observations, never by test performance.
    for candidate in study["candidates"]:
        cu = candidate["physics"]["cu_physics"][0]
        mapping[candidate["id"]] = {}
        for case in study["cases"]:
            day = int(case["date"][-2:])
            error = cu if day < 5 else (7 if cu == 0 else 0)
            path = root/f"{candidate['id']}-{case['id']}.csv"
            rows([day], error=error).to_csv(path, index=False)
            mapping[candidate["id"]][case["id"]] = [str(path)]
    path = root/"forecasts.json"
    write_json(path, {"schema": "gpuwrf.optimize.forecasts.v1", "forecasts": mapping})
    return path


def test_schedule_budget_and_no_singleton_batches():
    assert schedule(4, 4, 2, 3, 42) == ([{"candidates": 4, "train_dates": 2}, {"candidates": 2, "train_dates": 4}], 42)
    with pytest.raises(ValueError, match="42 case-hours"):
        schedule(4, 4, 2, 3, 41)
    for dates in range(2, 21):
        rounds, _ = schedule(8, dates, 2, 3, 10000)
        targets = [0]+[r["train_dates"] for r in rounds]
        assert all(b-a >= 2 for a, b in zip(targets, targets[1:]))


def test_admitted_selection_common_support_and_held_out(study_files, tmp_path):
    path, _ = study_files
    study = load_study(path)
    result = score(study, inventory(study, tmp_path))
    assert result["recommended"]["physics"]["cu_physics"] == [0]
    assert [r["loss"] for r in result["rounds"][0]["scores"]] == [0, 1, 5, 93]
    assert [r["loss"] for r in result["rounds"][1]["scores"]] == [0, 1]
    assert result["held_out"]["loss"] == 7
    assert result["held_out"]["uncertainty"]["interval95"] == [7, 7]
    assert result["evaluated_case_hours"] == result["planned_case_hours"] == 42
    # Only one candidate's test predictions were ever opened.
    used_test = [p for p in result["provenance"] if p.get("case") in ("c5", "c6")]
    assert len(used_test) == 2 and all(p["candidate"] == result["recommended"]["id"] for p in used_test)


@pytest.mark.parametrize("change", ["cpu", "source", "version", "evidence", "dates", "budget", "horizon", "unknown"])
def test_study_fail_closed(study_files, change):
    path, raw = study_files
    if change in ("cpu", "source", "version", "evidence"):
        for p in raw["admissions"]:
            receipt = json.loads(Path(p).read_text())
            if change == "cpu":
                receipt["status"] = "CPU_ORACLE_QUALIFIED"
            elif change == "source":
                receipt["model_sha256"] = "0"*64
            elif change == "version":
                receipt["gpuwrf_version"] = "old"
            else:
                receipt["evidence"][0]["sha256"] = "0"*64
            Path(p).write_text(json.dumps(receipt))
    elif change == "dates":
        raw["cases"][0]["date"] = "2026-01-02"
    elif change == "budget":
        raw["budget_case_hours"] = 41
    elif change == "horizon":
        raw["hours"] = 4
    else:
        raw["objective"] = "unknown"
    path.write_text(json.dumps(raw))
    with pytest.raises((ValueError, TypeError)):
        load_study(path)


def test_pairing_checked_against_operational_authority(study_files):
    path, raw = study_files
    for p in raw["admissions"]:
        receipt = json.loads(Path(p).read_text())
        receipt["physics"]["bl_pbl_physics"] = 1  # YSU with MYNN-SL=5 is refused.
        Path(p).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="no admitted"):
        load_study(path)


def test_search_space_does_not_admit_unqualified_options(study_files):
    path, raw = study_files
    raw["search_space"] = {k: [v] for k, v in dict(zip(PHYSICS, (18, 5, 5, 4, 4, 4, 0))).items()}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no admitted"):
        load_study(path)


def test_round_intersection_does_not_reward_missing_worst_error(study_files, tmp_path):
    path, _ = study_files
    study = load_study(path)
    inv = inventory(study, tmp_path)
    raw = json.loads(inv.read_text())
    candidate = next(c for c in study["candidates"] if c["physics"]["cu_physics"] == [1])
    p = Path(raw["forecasts"][candidate["id"]]["c1"][0])
    # All candidates must lose this exact row. Remaining station b still has support.
    frame = pd.read_csv(p).iloc[1:]
    frame.to_csv(p, index=False)
    result = score(study, inv)
    assert result["rounds"][0]["common_rows"] == 3
    assert all(r["metrics"][0]["n"] == 3 for r in result["rounds"][0]["scores"])


def test_equal_station_weight_and_mixed_units():
    p = validate(rows([1]))
    p = p.rename(columns={"value": "obs"})
    p["pred"] = p.obs+[0, 4]
    # Hand formula: station MSEs=0 and16; sqrt(mean)=sqrt(8).
    assert metrics(p, ["t2"], {"t2": 2})["loss"] == pytest.approx(np.sqrt(8)/2)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("variable", ["t2", "wind", "precip_rate", "solar"])
def test_nonfinite_not_hidden_by_score_or_calibration(value, variable):
    frame = rows([1], variable=variable)
    frame["value"] = frame.value.astype(float)
    if variable == "precip_rate":
        frame["interval_start"] = ["2026-01-01T00:00:00Z"]*2
    frame.loc[0, "value"] = value
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate(frame)


def test_calibration_bias_expanding_folds_and_future_apply(tmp_path):
    station = station_file(tmp_path)
    obs, pred = tmp_path/"obs.csv", tmp_path/"pred.csv"
    rows(error=0).to_csv(obs, index=False)
    rows(error=3).to_csv(pred, index=False)
    model = calibration.fit(station, obs, pred)
    assert all(c["intercept"] == -3 and c["slope"] == 1 for c in model["coefficients"])
    assert all(s["raw_rmse"] == 3 and s["corrected_rmse"] == 0 for s in model["cross_validation"]["scores"])
    assert [f["train_n"] for f in model["cross_validation"]["folds"]] == [6, 8, 10]
    mp = tmp_path/"model.json"
    write_json(mp, model)
    future = tmp_path/"future.csv"
    rows([7], error=3).to_csv(future, index=False)
    out = tmp_path/"corrected.csv"
    assert calibration.apply(mp, future, out)["n"] == 2
    np.testing.assert_equal(pd.read_csv(out).value.to_numpy(), [14, 15])
    with pytest.raises(ValueError, match="follow fitted_through"):
        calibration.apply(mp, pred, tmp_path/"leaked.csv")


def test_mos_independent_closed_form_and_no_future_leakage(tmp_path):
    station = station_file(tmp_path)
    obs, pred = tmp_path/"obs.csv", tmp_path/"pred.csv"
    truth = rows()
    truth["value"] = 2*truth.value+1
    truth.to_csv(obs, index=False)
    rows().to_csv(pred, index=False)
    model = calibration.fit(station, obs, pred, method="mos")
    assert all(c["slope"] == 2 and c["intercept"] == 1 for c in model["coefficients"])
    assert all(s["corrected_rmse"] == 0 for s in model["cross_validation"]["scores"])
    # Change only future truth: earlier folds' coefficients MUST stay unchanged.
    truth.loc[truth.valid_time.str.contains("01-06"), "value"] += 1000
    truth.to_csv(obs, index=False)
    changed = calibration.fit(station, obs, pred, method="mos")
    assert changed["cross_validation"]["folds"][:-1] == model["cross_validation"]["folds"][:-1]
    assert changed["cross_validation"]["folds"][-1]["coefficients"] == model["cross_validation"]["folds"][-1]["coefficients"]


@pytest.mark.parametrize("method", ["bias", "mos"])
def test_sparse_calibration_fit_refuses_unvalidated_station(tmp_path, method):
    station = station_file(tmp_path)
    obs, pred = tmp_path/"obs.csv", tmp_path/"pred.csv"
    rows().to_csv(obs, index=False)
    sparse = rows(error=3)
    # b has enough data for a fit, but no dates in any held-out fold.
    sparse = sparse.loc[sparse.station_uid.eq("a") | sparse.valid_time.le("2026-01-03T01:00:00Z")]
    sparse.to_csv(pred, index=False)
    with pytest.raises(ValueError, match="lack chronological held-out support: b/t2"):
        calibration.fit(station, obs, pred, method=method)
    from gpuwrf.cli import main
    output = tmp_path/"unqualified-model.json"
    assert main(["calibrate", "fit", "--stations", str(station), "--obs", str(obs),
                 "--forecasts", str(pred), "--method", method, "--output", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize("method", ["bias", "mos"])
def test_sparse_calibration_legacy_apply_checks_each_station(tmp_path, method):
    station = station_file(tmp_path)
    obs, pred = tmp_path/"obs.csv", tmp_path/"pred.csv"
    rows().to_csv(obs, index=False)
    rows(error=3).to_csv(pred, index=False)
    model = calibration.fit(station, obs, pred, method=method)
    # Legacy v1 files can contain b coefficients and a global CV label,
    # while only a has a held-out score. Neither makes b deployable.
    model["cross_validation"]["scores"] = [s for s in model["cross_validation"]["scores"] if s["station_uid"] == "a"]
    model_path = tmp_path/"legacy.json"
    write_json(model_path, model)
    future = tmp_path/"future-b.csv"
    rows([7], error=3).loc[lambda data: data.station_uid.eq("b")].to_csv(future, index=False)
    output = tmp_path/"unqualified.csv"
    with pytest.raises(ValueError, match="lack chronological held-out support: b/t2"):
        calibration.apply(model_path, future, output)
    assert not output.exists() and not output.with_suffix(".csv.json").exists()
    qualified = tmp_path/"future-a.csv"
    rows([7], error=3).loc[lambda data: data.station_uid.eq("a")].to_csv(qualified, index=False)
    assert calibration.apply(model_path, qualified, tmp_path/"qualified.csv")["n"] == 1


def test_support_constraint_and_bad_coefficient_refusals():
    pairs = validate(rows([1], variable="wind")).rename(columns={"value": "pred"})
    fit = [{"station_uid": s, "variable": "wind", "intercept": -10, "slope": 1} for s in ("a", "b")]
    value, clipped = calibration.correct(pairs, fit)
    assert value.tolist() == [0, 0] and clipped == 2
    fit[0]["intercept"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        calibration.correct(pairs, fit)


def test_canonical_solar_and_rain_interval_alignment(tmp_path):
    from gpuwrf.verify.observations import read_stations
    stations = read_stations(station_file(tmp_path))
    obs = rows([1], variable="precip_rate")
    obs["interval_start"] = "2026-01-01T00:00:00Z"
    prediction = obs.copy()
    got = paired(validate(obs), validate(prediction), stations, ["precip_rate"])
    np.testing.assert_equal(got.obs.to_numpy(), got.pred.to_numpy())
    solar = rows([1], variable="solar")
    p = tmp_path/"solar.csv"
    solar.to_csv(p, index=False)
    got, _ = forecast([p], stations)
    assert got.variable.eq("solar").all()


def test_namelist_serialization_roundtrip(tmp_path):
    from gpuwrf.io.namelist_parser import parse_namelist
    expected = {"physics": {"mp_physics": [8, 8], "x": [None, True], "quoted": "it's a path"}}
    path = tmp_path/"namelist.input"
    path.write_text(namelist_text(expected))
    assert parse_namelist(path) == expected


def test_offline_cli_blocks_jax_imports(tmp_path):
    script = """import importlib.abc, sys
sys.argv = ['gpuwrf', 'optimize', '--help']
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname == 'jax' or fullname.startswith('jax.'):
   raise AssertionError('offline optimizer imported JAX')
sys.meta_path.insert(0, Block())
from gpuwrf.cli import main
main(sys.argv[1:])
"""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0", "PYTHONPATH": str(REPO/"src")}
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "optimize" in result.stdout


def test_cli_plan_and_score(study_files, tmp_path):
    from gpuwrf.cli import main
    path, _ = study_files
    assert main(["optimize", "plan", "--study", str(path), "--output", str(tmp_path/"plan.json")]) == 0
    inv = inventory(load_study(path), tmp_path)
    out = tmp_path/"result.json"
    assert main(["optimize", "score", "--study", str(path), "--forecasts", str(inv), "--output", str(out)]) == 0
    assert json.loads(out.read_text())["held_out"]["loss"] == 7
    assert main(["optimize", "score", "--study", str(path), "--forecasts", str(inv), "--output", str(out)]) == 2


def test_run_real_ensemble_preparation_fake_scheduler(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.io.namelist_parser import parse_namelist
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.optimize.study import run

    path, _ = study_files
    study = load_study(path)
    calls = []
    def fake_launch(plan_path, **kwargs):
        plan = json.loads(Path(plan_path).read_text())
        calls.append(plan)
        for member in plan["members"]:
            cfg = parse_namelist(Path(member["case"])/"namelist.input")
            day = cfg["time_control"]["start_day"]
            cu = cfg["physics"]["cu_physics"]
            cu = cu[0] if isinstance(cu, list) else cu
            error = cu if day < 5 else (7 if cu == 0 else 0)
            output = Path(plan["out_root"])/member["id"]/"wrfout"
            output.mkdir(parents=True)
            with Dataset(output/f"wrfout_d01_2026-01-{day:02d}_01:00:00", "w") as ds:
                for dim, count in (("Time", 1), ("DateStrLen", 19), ("south_north", 2), ("west_east", 2)):
                    ds.createDimension(dim, count)
                ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(f"2026-01-{day:02d}_01:00:00")], dtype="S1")
                ds.setncatts({"GRID_ID": 1, "SIMULATION_START_DATE": f"2026-01-{day:02d}_00:00:00"})
                for name, values, units in (("XLAT", [[0, 0], [1, 1]], "degree_north"),
                                            ("XLONG", [[0, 1], [0, 1]], "degree_east"),
                                            ("LANDMASK", np.ones((2, 2)), ""),
                                            ("T2", 273.15+day*2+error+np.array([[-.5, .5], [.5, 1.5]]), "K")):
                    v = ds.createVariable(name, "f8", ("Time", "south_north", "west_east"))
                    v[:] = values
                    v.units = units
        return 0
    monkeypatch.setattr(workflows, "launch", fake_launch)
    output = tmp_path/"run"
    report = run(study, output)
    assert report["recommended"]["physics"]["cu_physics"] == [0]
    assert report["held_out"]["loss"] == pytest.approx(7)
    assert report["scheduled_case_hours"] == 42
    assert len(calls) == 7 and all(p["mode"] == "batch" for p in calls)
    assert (output/"result.json").is_file()


def test_failed_scheduler_never_publishes_recommendation(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import run
    path, _ = study_files
    monkeypatch.setattr(workflows, "launch", lambda *a, **k: 7)
    output = tmp_path/"failed"
    with pytest.raises(ValueError, match="failed with exit 7"):
        run(load_study(path), output)
    assert not (output/"result.json").exists()


def test_metric_rejects_one_nan_before_pandas_skipna():
    pairs = validate(rows([1])).rename(columns={"value": "obs"})
    pairs["pred"] = pairs.obs.astype(float)
    pairs.loc[0, "pred"] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        metrics(pairs, ["t2"], {"t2": 1})


def test_recorded_original_cpu_wrf_surface_reader(tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    fixture = REPO/"proofs/u2_optim/cpu_wrf_surface.npz"
    receipt = json.loads(fixture.with_suffix(".json").read_text())
    assert identity(fixture)["sha256"] == receipt["fixture"]["sha256"]
    with np.load(fixture, allow_pickle=False) as data:
        out = tmp_path/"recorded.nc"
        with Dataset(out, "w") as ds:
            for dim, count in (("Time", 2), ("DateStrLen", 19), ("south_north", 2), ("west_east", 2)):
                ds.createDimension(dim, count)
            ds.setncatts(receipt["attributes"])
            ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = data["Times"]
            for name in data.files:
                if name == "Times":
                    continue
                v = ds.createVariable(name, data[name].dtype, ("Time", "south_north", "west_east"))
                v[:] = data[name]
                v.units = receipt["units"][name]
        # Exact corner stations: independent expected nodes, not StationSampler.
        st = pd.DataFrame({"station_uid": ["corner0", "corner1"],
            "lat": [data["XLAT"][0, 0, 0], data["XLAT"][0, 1, 1]],
            "lon": [data["XLONG"][0, 0, 0], data["XLONG"][0, 1, 1]],
            "landmask": [data["LANDMASK"][0, 0, 0], data["LANDMASK"][0, 1, 1]],
            "elevation_m": [0, 0], "wind_height_m": [10, 10], "roughness_m": [.03, .03]})
        got, _ = forecast([out], st, method="nearest")
        for n, uid in enumerate(st.station_uid):
            t2 = got.loc[got.station_uid.eq(uid) & got.variable.eq("t2")].value.to_numpy()
            solar = got.loc[got.station_uid.eq(uid) & got.variable.eq("solar")].value.to_numpy()
            np.testing.assert_equal(t2, data["T2"][:, n, n].astype(float)-273.15)
            np.testing.assert_equal(solar, data["SWDOWN"][:, n, n].astype(float))
