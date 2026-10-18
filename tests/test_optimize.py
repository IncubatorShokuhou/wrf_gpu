"""Analytic station errors, chronological mutants and fake scheduler integration.

Synthetic admission receipts below exercise binding, never certify real GPU physics.
"""
from __future__ import annotations

import copy
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
            if name == "wrfinput_d01":
                for dim, size in (("west_east", 2), ("south_north", 2), ("bottom_top", 3)):
                    ds.createDimension(dim, size)
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
    for cu in (0, 1, 10, 11):  # released v0.4.0 tuples (c00, c36/c71-73, c30, c31)
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
    # Analytic training errors 0,1,10,11; held-out cu=0 error=7 to prove
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
    assert [r["loss"] for r in result["rounds"][0]["scores"]] == [0, 1, 10, 11]
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
    # Released RUC tuple (c33, Madrid with num_soil_layers=6) is still checked against this case's
    # namelist: without six soil layers the operational binding refuses it.
    receipt = json.loads(Path(raw["admissions"][0]).read_text())
    receipt["physics"]["sf_surface_physics"] = 3
    Path(raw["admissions"][0]).write_text(json.dumps(receipt))
    raw["admissions"] = raw["admissions"][:1]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no admitted"):
        load_study(path)


def test_unreleased_admission_is_refused(study_files):
    path, raw = study_files
    for p in raw["admissions"]:
        receipt = json.loads(Path(p).read_text())
        receipt["physics"]["bl_pbl_physics"] = 1  # YSU with MYNN-SL=5: no FINAL D6 PASS run
        Path(p).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="not released for gpuwrf optimize"):
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


def make_fake_launch(calls, *, clock=None, interrupt=None, fail=None, walls=None):
    """Fake scheduler: writes one analytic wrfout + launcher receipt per member (training error = cu)."""
    from gpuwrf.io.namelist_parser import parse_namelist
    from gpuwrf.io.netcdf_lock import Dataset

    compiled = set()  # configurations whose program a previous member compiled

    def fake_launch(plan_path, **kwargs):
        plan = json.loads(Path(plan_path).read_text())
        calls.append(plan)
        batch_wall = [0.0]
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
            failed = fail == (len(calls)-1, plan["members"].index(member))
            warm = cu in compiled
            compiled.add(cu)
            wall = 10.0 if walls is None else walls[1 if warm else 0]
            batch_wall[0] = max(batch_wall[0], wall)
            (output.parent/"receipt.json").write_text(json.dumps({"rc": 1 if failed else 0, "wall_s": wall,
                                                                  "aot_cache_warm": warm and walls is not None,
                                                                  "vmhwm_kb": 5_000_000, "vram_peak_mib": 3000}))
        if clock is not None:
            clock[0] += 100.0 if walls is None else batch_wall[0]
        if interrupt is not None and len(calls)-1 == interrupt:
            raise KeyboardInterrupt("GPU window ended")
        return 7 if fail is not None and fail[0] == len(calls)-1 else 0
    return fake_launch


def test_run_real_ensemble_preparation_fake_scheduler(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import run

    path, _ = study_files
    study = load_study(path)
    calls = []
    fake_launch = make_fake_launch(calls)
    monkeypatch.setattr(workflows, "launch", fake_launch)
    output = tmp_path/"run"
    report = run(study, output)
    assert report["recommended"]["physics"]["cu_physics"] == [0]
    assert report["held_out"]["loss"] == pytest.approx(7)
    assert report["scheduled_case_hours"] == 42
    # Every configuration of a rung goes to the launcher together, never one batch per configuration: rung 1
    # compiles each of the 4 configurations once (cold wave), then their second dates load warm; rung 2 = 2
    # survivors x 2 new dates (warm); then the winner's 2 held-out dates.
    assert [len(p["members"]) for p in calls] == [4, 4, 4, 2] and all(p["mode"] == "batch" for p in calls)
    first = {json.loads((Path(m["case"])/"member_manifest.json").read_text())["source_case"] for m in calls[0]["members"]}
    assert len({Path(f).parent.name for f in first}) == 4  # one member per configuration in the cold wave
    # Single-domain members get a fixed launcher pool (no C-auto plan exists for max_dom=1).
    pool = study["resources"]["member"]["pool_gib"]
    pools = [o[o.index("--pool-gib")+1] if "--pool-gib" in o else None for o in (p["launcher_options"] for p in calls)]
    assert pools == [str(pool)]*len(calls)
    # Host need: launcher default until a member peak exists, then 1.1 x the measured VmHWM (5.0 GB -> 5.5).
    hosts = [o[o.index("--host-gb-per-case")+1] if "--host-gb-per-case" in o else None
             for o in (p["launcher_options"] for p in calls)]
    assert hosts == [None, "5.5", "5.5", "5.5"]
    rungs = sorted((output/"rungs").glob("rung-*.json"))
    assert [json.loads(r.read_text())["survivors"] for r in rungs] == [
        [c["id"] for c in sorted(study["candidates"], key=lambda c: c["physics"]["cu_physics"][0])[:2]],
        [report["recommended"]["id"]]]
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


# --- v0.4.1 A6: released list, rungs, resumable chunks, bundle consumer, admission math -------------

def released_study(raw, path, **changes):
    raw = {**{k: v for k, v in raw.items() if k != "admissions"}, **changes}
    path.write_text(json.dumps(raw))
    return path


def test_release_list_binds_the_final_checkpoint():
    from gpuwrf.optimize import released
    data = released.load()
    checkpoint = json.loads((REPO/"proofs/r_docs040/RELEASE_CHECKPOINT.json").read_text())
    assert data["checkpoint"]["sha256"] == identity(REPO/"proofs/r_docs040/RELEASE_CHECKPOINT.json")["sha256"]
    final = {row["case"] for row in checkpoint["rows"]}
    assert len(final) == 66 and final <= set(data["by_run"])  # every FINAL composite qualifies its exact tuple
    assert {f"F20-{r['case']}" for r in checkpoint["default_F20"]["rows"]} <= set(data["by_run"])
    for row in checkpoint["rows"]:
        entry = data["by_run"][row["case"]]
        assert {k: entry["physics"][k][0] for k in PHYSICS} == {k: row["options"][k] for k in PHYSICS}
        run = next(r for r in entry["qualified_by"] if r["run"] == row["case"])
        assert run["D6"] == "PASS" and run["E41"] == row["final_receipt"]["E41"]
    assert len(data["configurations"]) == 47
    # ALISIOS WN3 d01 default tuple (KF on d01) is released with E41 PASS runs; YSU+MYNN-SL is not.
    kf = dict(zip(PHYSICS, ([8], [5], [5], [4], [4], [4], [1])))
    assert released.resolve(data, kf, 1, e41="PASS")["id"] == config_id(kf)
    with pytest.raises(ValueError, match="not released"):
        released.resolve(data, {**kf, "bl_pbl_physics": [1]}, 1)
    with pytest.raises(ValueError, match="not released"):
        released.resolve(data, kf, 2)  # same tuple, other domain count
    with pytest.raises(ValueError, match="no E41 PASS"):
        released.resolve(data, released.lookup(data, "c26")["physics"], 1, e41="PASS")


@pytest.mark.skipif(not Path("<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016").is_dir(), reason="F20 archive absent")
def test_release_list_rebuilds_byte_exact():
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0", "PYTHONPATH": str(REPO/"src")}
    result = subprocess.run([sys.executable, str(REPO/"proofs/w41_optimize/build_released.py"), "--check"],
                            cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "EXACT" in result.stdout, result.stdout+result.stderr


def test_release_list_refuses_other_versions_and_corruption(tmp_path, monkeypatch):
    import gpuwrf
    from gpuwrf.optimize import released
    data = json.loads(released.RELEASED_PATH.read_text())
    data["configurations"][0]["physics"]["mp_physics"] = [99]
    bad = tmp_path/"bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="corrupt"):
        released.load(bad)
    monkeypatch.setattr(gpuwrf, "__version__", "0.5.0")
    with pytest.raises(ValueError, match="does not apply"):
        released.load()


def test_configurations_by_released_name_without_admissions(study_files, tmp_path):
    path, raw = study_files
    study = load_study(released_study(raw, path, configurations=["c00", "c36", "c30", "c31"]))
    assert {c["status"] for c in study["candidates"]} == {"RELEASED_v0.4.0"}
    assert sorted(c["physics"]["cu_physics"][0] for c in study["candidates"]) == [0, 1, 10, 11]
    assert all(c["admission"] is None and c["release"]["best_E41"] == "PASS" for c in study["candidates"])
    # Disclosure: the case context is compared with the closest qualifying run, never hidden.
    context = study["candidates"][0]["context"]["c1"]
    assert context["run"] and isinstance(context["differences"], dict)
    result = score(study, inventory(study, tmp_path))
    assert result["recommended"]["physics"]["cu_physics"] == [0]
    for names, error in ((["c00", "c99"], "unknown released"), (["c00", "c26"], "no E41 PASS"),
                         (["c00", "F20-PROD"], "not released"), (["c00", "c00"], "twice"), ("c00", "must be")):
        released_study(raw, path, configurations=names, e41="PASS")
        with pytest.raises(ValueError, match=error):
            load_study(path)


def test_all_released_configurations_with_e41_filter(study_files):
    from gpuwrf.optimize import released
    path, raw = study_files
    study = load_study(released_study(raw, path, configurations="all", e41="PASS", budget_case_hours=100000))
    data = released.load()
    expected = {e["id"] for e in data["configurations"] if e["max_dom"] == 1 and e["best_E41"] == "PASS"}
    seen = {c["id"] for c in study["candidates"]} | {r["id"] for r in study["rejected"]}
    assert seen == expected and all(c["release"]["best_E41"] == "PASS" for c in study["candidates"])
    assert any("sf_surface_physics" in r["reason"] for r in study["rejected"])  # RUC without six soil layers


def test_explicit_rungs_tournament_and_refusals():
    # ALISIOS tournament shape: 8 configurations, 2 rungs (keep 4, then the winner), 2 held-out dates, 24 h.
    rounds, used = schedule(8, 4, 2, 24, 10000, rungs=[{"train_dates": 2, "keep": 4}, {"train_dates": 4, "keep": 1}])
    assert rounds == [{"candidates": 8, "train_dates": 2, "keep": 4}, {"candidates": 4, "train_dates": 4, "keep": 1}]
    assert used == (2 + 8*2 + 4*2)*24
    for rungs, error in (([{"train_dates": 2, "keep": 9}, {"train_dates": 4, "keep": 1}], "keep more"),
                         ([{"train_dates": 2, "keep": 4}, {"train_dates": 2, "keep": 1}], "increase"),
                         ([{"train_dates": 2, "keep": 4}, {"train_dates": 3, "keep": 1}], "every training date"),
                         ([{"train_dates": 2, "keep": 4}, {"train_dates": 4, "keep": 2}], "exactly one"),
                         ([{"train_dates": 4}], "list of")):
        with pytest.raises(ValueError, match=error):
            schedule(8, 4, 2, 24, 10000, rungs=rungs)
    with pytest.raises(ValueError, match="case-hours"):
        schedule(8, 4, 2, 24, used-1, rungs=[{"train_dates": 2, "keep": 4}, {"train_dates": 4, "keep": 1}])


def test_record_count_groups_are_reported(study_files, monkeypatch):
    from gpuwrf.io.netcdf_lock import Dataset
    path, raw = study_files
    case = Path(raw["cases"][0]["input_dir"])
    times = ["2026-01-01_00:00:00", "2026-01-01_03:00:00", "2026-01-01_06:00:00"]
    (case/"wrfbdy_d01").unlink()
    with Dataset(case/"wrfbdy_d01", "w") as ds:
        ds.createDimension("Time", len(times))
        ds.createDimension("DateStrLen", 19)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(t) for t in times], dtype="S1")
    from gpuwrf.optimize import study as study_module
    study = load_study(path)
    groups = study["program_groups"]
    assert sorted(len(g["cases"]) for g in groups["signatures"]) == [1, 5]
    expected = 1 if groups["record_count_independent"] else 2
    assert groups["cold_programs_upper_bound"] == expected*len(study["candidates"])
    monkeypatch.setattr(study_module, "record_count_independent", lambda: False)
    assert study_module.program_groups(study)["cold_programs_upper_bound"] == 2*len(study["candidates"])
    monkeypatch.setattr(study_module, "record_count_independent", lambda: True)
    assert study_module.program_groups(study)["cold_programs_upper_bound"] == len(study["candidates"])


def test_member_pool_and_admission_math():
    from gpuwrf.optimize.resources import admission, member_pool
    wn3 = member_pool({"west_east": 120, "south_north": 70, "bottom_top": 44})
    swiss = member_pool({"west_east": 42, "south_north": 42, "bottom_top": 44})
    prod_like = member_pool({"west_east": 120, "south_north": 70, "bottom_top": 50})
    # WN3 d01 is measured: 1.0 GiB sufficed (peak 1,744 MiB per process) -> 1.25 GiB pool, 2.25 GiB need.
    assert (wn3["pool_gib"], wn3["need_gib"]) == (1.25, 2.25) and wn3["measured"]["per_process_peak_mib"] == 1744
    # Unmeasured grids use the formula: Swiss floor 256 MiB -> ((256*1.41*1.5)+123)*1.2+512 MiB = 1309 MiB -> 1.5 GiB;
    # 120x70x50: 685*50/44 = 778 MiB -> ((778*1.41*1.5)+123)*1.2+512 = 2634 MiB -> 2.75 GiB.
    assert (swiss["pool_gib"], prod_like["pool_gib"]) == (1.5, 2.75)
    fit = admission(3.5, members=16, host_available_gb=51, host_gb_per_member=6)
    assert (fit["vram_parallel"], fit["host_parallel"], fit["parallel"]) == (7, 7, 7)
    assert admission(3.5, members=16, host_available_gb=51)["parallel"] == 2  # unmeasured host need 16 GB/case


def test_chunk_pause_and_resume_reproduce_the_uninterrupted_study(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import PAUSED, run
    study = load_study(study_files[0])
    calls = []
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls))
    full = run(study, tmp_path/"full")
    calls.clear()
    clock = [0.0]
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, clock=clock))
    out = tmp_path/"chunked"
    statuses = [run(study, out, chunk_minutes=1, member_wall_s=10, clock=lambda: clock[0])["status"]]
    changed = copy.deepcopy(study)
    changed["hours"] = 6
    launched = len(calls)
    with pytest.raises(ValueError, match="same study"):  # a paused study resumes only unchanged
        run(changed, out, resume=True, chunk_minutes=1, member_wall_s=10, clock=lambda: clock[0])
    assert len(calls) == launched  # refused before any member was launched
    for attempt in range(4):
        result = run(study, out, resume=True, chunk_minutes=1, member_wall_s=10, clock=lambda: clock[0])
        statuses.append(result.get("status"))
        if result["status"] == "PAUSED":
            assert json.loads((out/"progress.json").read_text())["status"] == "PAUSED"
            continue
        break
    assert statuses[:-1] == ["PAUSED"]*3 and len(statuses) == 4  # one batch per 1-minute chunk
    assert [len(p["members"]) for p in calls] == [4, 4, 4, 2]
    keys = ("recommended", "rounds", "held_out", "planned_case_hours")
    assert {k: result[k] for k in keys} == {k: full[k] for k in keys}
    assert PAUSED == 3


def test_interrupted_batch_members_are_salvaged_on_resume(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import run
    path, raw = study_files
    path.unlink()
    write_json(path, raw | {"budget_case_hours": 54})  # room to re-run the batch: salvage, not the budget, must spare it
    study = load_study(path)
    calls = []
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, interrupt=0))
    out = tmp_path/"run"
    with pytest.raises(KeyboardInterrupt):
        run(study, out)
    ledger = json.loads((out/"ledger.json").read_text())
    assert ledger["members"] == {} and "rc" not in ledger["batches"][0]
    calls.clear()
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls))
    report = run(study, out, resume=True)
    assert [len(p["members"]) for p in calls] == [4, 4, 2]  # the cold wave's four finished members were not rerun
    assert sum(bool(m.get("salvaged")) for m in json.loads((out/"ledger.json").read_text())["members"].values()) == 4
    assert report["recommended"]["physics"]["cu_physics"] == [0]


def test_changed_member_history_is_refused(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import run
    study = load_study(study_files[0])
    calls = []
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, interrupt=1))
    out = tmp_path/"run"
    with pytest.raises(KeyboardInterrupt):
        run(study, out)
    ledger = json.loads((out/"ledger.json").read_text())
    victim = Path(next(iter(ledger["members"].values()))["files"][0]["path"])
    victim.write_bytes(victim.read_bytes()+b"\0")
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls))
    with pytest.raises(ValueError, match="changed since it was recorded"):
        run(study, out, resume=True)


def test_bundle_launcher_runs_the_bundle_launcher_inside_the_consumer(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import bundle_launcher, run
    bundle = tmp_path/"P"
    (bundle/"src"/"scripts").mkdir(parents=True)
    (bundle/"sealed").mkdir()
    (bundle/"sealed"/"BUNDLE_MANIFEST.json").write_text("{}")
    (bundle/"sealed"/"BUNDLE.json").write_text("{}")
    (bundle/"bundle.env").write_text(f"export PYTHONPATH={bundle}/src/src\n")
    log = tmp_path/"consumer.log"
    (bundle/"src"/"scripts"/"run_parallel_cases.sh").write_text("#!/usr/bin/env bash\necho LAUNCHER \"$@\" >> "+str(log)+"\n")
    (bundle/"src"/"scripts"/"run_parallel_cases.sh").chmod(0o755)
    consumer = tmp_path/"bundle_consumer.sh"
    consumer.write_text("#!/usr/bin/env bash\necho CONSUMER PYTHONPATH=$PYTHONPATH \"$@\" >> "+str(log)+"\n"
                        "while [ \"$1\" != -- ]; do shift; done; shift; exec \"$@\"\n")
    consumer.chmod(0o755)
    out = tmp_path/"study"
    out.mkdir()
    script, record = bundle_launcher(out, bundle, consumer)
    assert subprocess.run([str(script), "--out-root", "x", "case"], check=False).returncode == 0
    consumer_line, launcher_line = log.read_text().splitlines()
    assert f"PYTHONPATH={bundle}/src/src --bundle {bundle} --layer {out}/bundle_layer --" in consumer_line
    assert launcher_line == "LAUNCHER --out-root x case"
    assert record["layer"] == str(out/"bundle_layer") and record["bundle_json"]["sha256"]
    with pytest.raises(ValueError, match="inside the bundle"):
        bundle_launcher(bundle/"jax", bundle, consumer)
    (bundle/"sealed"/"BUNDLE_MANIFEST.json").unlink()
    with pytest.raises(ValueError, match="sealed"):
        bundle_launcher(out, bundle, consumer)
    (bundle/"sealed"/"BUNDLE_MANIFEST.json").write_text("{}")
    # The optimizer hands every batch to that wrapper script.
    seen = []
    def fake(plan_path, launcher=None, **kwargs):
        seen.append(launcher)
        return make_fake_launch([])(plan_path)
    monkeypatch.setattr(workflows, "launch", fake)
    report = run(load_study(study_files[0]), tmp_path/"bundled", bundle=bundle, bundle_consumer=consumer)
    assert seen and all(Path(s).name == "bundle_launcher.sh" for s in seen)
    assert report["bundle"]["bundle"] == str(bundle.resolve())


def test_cli_released_listing(capsys):
    from gpuwrf.cli import main
    assert main(["optimize", "released", "--max-dom", "1", "--e41", "PASS", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["configurations"] and all(e["max_dom"] == 1 and e["best_E41"] == "PASS" for e in listing["configurations"])
    assert main(["optimize", "released"]) == 0
    assert "p-c8964de962e8" in capsys.readouterr().out


def test_driver_is_cpu_only_and_members_get_the_caller_platform(study_files, tmp_path, monkeypatch):
    """A caller platform the driver cannot initialize proves it never builds a backend with it (E50/E132)."""
    from gpuwrf.cli import main
    from gpuwrf.ens import workflows
    path, _ = study_files
    seen = []
    def fake(plan_path, launcher=None, env=None, **kwargs):
        seen.append(env)
        return make_fake_launch([])(plan_path)
    monkeypatch.setattr(workflows, "launch", fake)
    monkeypatch.setenv("JAX_PLATFORMS", "no_such_platform")
    assert main(["optimize", "run", "--study", str(path), "--output", str(tmp_path/"run")]) == 0
    assert seen and all(env["JAX_PLATFORMS"] == "no_such_platform" for env in seen)
    assert os.environ["JAX_PLATFORMS"] == "no_such_platform"  # restored after the command
    # Stand-in for any driver-side JAX use (e.g. an import-time jnp table, E46): with the pin it runs on the CPU;
    # without it the caller's platform would be initialised (here: an unknown one -> loud failure).
    script = """import sys, json
from gpuwrf.cli import main
from gpuwrf.optimize import study
real = study.load_study
def touching(path):
    import jax.numpy as jnp
    jnp.zeros(1).block_until_ready()
    return real(path)
study.load_study = touching
rc = main(sys.argv[1:])
bridge = sys.modules.get("jax._src.xla_bridge")
print(json.dumps({"rc": rc, "backends": sorted(getattr(bridge, "_backends", {}) or {})}))
"""
    env = {**os.environ, "JAX_PLATFORMS": "no_such_platform", "GPUWRF_JAX_CACHE": "0", "PYTHONPATH": str(REPO/"src")}
    result = subprocess.run([sys.executable, "-c", script, "optimize", "plan", "--study", str(path),
                             "--output", str(tmp_path/"plan.json")], cwd=REPO, env=env, capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["rc"] == 0 and set(out["backends"]) == {"cpu"}, result.stderr[-2000:]


def test_failed_member_keeps_the_others_and_resume_reruns_only_it(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import run
    path, raw = study_files
    path.unlink()
    write_json(path, raw | {"budget_case_hours": 45})  # planned 42 + one 3 h retry (retries are charged)
    study = load_study(path)
    calls = []
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, fail=(0, 3)))
    out = tmp_path/"run"
    with pytest.raises(ValueError, match="failed with exit 7"):
        run(study, out)
    assert len(json.loads((out/"ledger.json").read_text())["members"]) == 3 and not (out/"result.json").exists()
    calls.clear()
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls))
    report = run(study, out, resume=True)
    # the failed configuration's first date alone (cold), the four second dates (warm), rung 2, the held-out
    assert [len(p["members"]) for p in calls] == [1, 4, 4, 2]
    assert calls[0]["min_members"] == 1 and report["recommended"]["physics"]["cu_physics"] == [0]
    assert report["scheduled_case_hours"] == 45  # the failed attempt and its retry are both charged


def test_cold_first_compiles_each_configuration_once():
    from gpuwrf.optimize.study import cold_first
    a, b = {"id": "a"}, {"id": "b"}
    d1, d2, d3 = ({"id": n, "records": {"wrfbdy_d01": 13}} for n in ("d1", "d2", "d3"))
    other = {"id": "d4", "records": {"wrfbdy_d01": 9}}  # another record count is another program
    waves = cold_first([(a, d1), (a, d2), (b, d1), (b, d2), (a, other)], set())
    assert waves == [[(a, d1), (b, d1), (a, other)], [(a, d2), (b, d2)]]
    warm = {("a", json.dumps(d1["records"], sort_keys=True))}
    assert cold_first([(a, d3), (b, d3)], warm) == [[(b, d3)], [(a, d3)]]


def _serial_launch(calls, clock, wall):
    """Fake launcher that admits ONE member at a time although more were requested (host/VRAM bound)."""
    base = make_fake_launch(calls)

    def launch(plan_path, **kwargs):
        rc = base(plan_path, **kwargs)
        plan = json.loads(Path(plan_path).read_text())
        clock[0] += wall*len(plan["members"])
        root = Path(plan["out_root"])
        (root/"parallel_run.json").write_text(json.dumps({"peak_concurrency": 1, "vram_free_bytes_at_start": 30*2**30}))
        for member in plan["members"]:
            receipt = root/member["id"]/"receipt.json"
            receipt.write_text(json.dumps(json.loads(receipt.read_text()) | {"wall_s": wall, "concurrent_at_start": 0}))
        return rc
    return launch


def test_warm_waves_are_estimated_with_warm_member_walls(study_files, tmp_path, monkeypatch):
    """Cold members 500 s, warm 10 s, 10-minute chunks, one member at a time: after the first warm member the
    rest of the study fits one chunk; cold-wall estimates for warm waves would split it into single members."""
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import run
    study = load_study(study_files[0])
    calls, clock = [], [0.0]
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, clock=clock, walls=(500.0, 10.0)))
    statuses = []
    for attempt in range(8):
        result = run(study, tmp_path/"run", resume=attempt > 0, chunk_minutes=10, max_parallel=1, clock=lambda: clock[0])
        statuses.append(result.get("status"))
        if result.get("status") != "PAUSED":
            break
    # The unmeasured first member (default 3 h x 60 + 900 = 1,080 s > 600 s) runs alone, recorded as over the chunk;
    # each cold member then fills one chunk; the first warm member is estimated cold (no warm wall yet).
    assert statuses == ["PAUSED"]*4 + [statuses[-1]] and statuses[-1] != "PAUSED"
    assert [len(p["members"]) for p in calls] == [1, 1, 1, 1, 1, 3, 4, 2]
    ledger = json.loads((tmp_path/"run"/"ledger.json").read_text())
    assert "over_chunk" in ledger["batches"][0] and not any("over_chunk" in b for b in ledger["batches"][1:])


def test_chunk_size_follows_the_admitted_parallelism(study_files, tmp_path, monkeypatch):
    """Requested 4 in parallel, the launcher admits 1 (host/VRAM): no batch may assume the requested parallelism."""
    from gpuwrf.ens import workflows
    from gpuwrf.optimize import study as module
    study = load_study(study_files[0])
    calls, clock = [], [0.0]
    monkeypatch.setattr(workflows, "launch", _serial_launch(calls, clock, 400.0))
    monkeypatch.setattr(module, "mem_available_gb", lambda: 100.0)
    out = tmp_path/"run"
    first = module.run(study, out, chunk_minutes=10, max_parallel=4, member_wall_s=400, clock=lambda: clock[0])
    # Nothing measured yet -> one member at a time: one 400 s member fits the 600 s chunk, a second does not.
    assert first["status"] == "PAUSED" and clock[0] <= 600 and [len(p["members"]) for p in calls] == [1]
    clock[0] = 0.0
    second = module.run(study, out, resume=True, chunk_minutes=30, max_parallel=4, clock=lambda: clock[0])
    # Measured host need (5.5 GB) admits 4 -> a 3-member batch; the launcher ran it one at a time (peak 1 < 3),
    # so the following batch is sized for one member and the chunk still holds.
    assert second["status"] == "PAUSED" and clock[0] <= 1800 and [len(p["members"]) for p in calls] == [1, 3, 1]


def test_resume_refuses_changed_forecast_input_of_completed_members(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.optimize.study import run
    path, raw = study_files
    victim = Path(raw["cases"][0]["input_dir"])/"wrfinput_d01"
    with Dataset(victim, "a") as ds:
        ds.createVariable("T", "f4", ("bottom_top", "south_north", "west_east"))[:] = 300.0
    study = load_study(path)
    calls, clock = [], [0.0]
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, clock=clock))
    out = tmp_path/"run"
    assert run(study, out, chunk_minutes=1, member_wall_s=10, clock=lambda: clock[0])["status"] == "PAUSED"
    completed = json.loads((out/"ledger.json").read_text())["members"]
    assert any(key.endswith(f"|{raw['cases'][0]['id']}") for key in completed)
    size = victim.stat().st_size
    with Dataset(victim, "a") as ds:  # same path, same size, same dates and grid: only the values change
        ds.variables["T"][:] = 340.0
    assert victim.stat().st_size == size
    launched = len(calls)
    with pytest.raises(ValueError, match="same study"):
        run(load_study(path), out, resume=True, chunk_minutes=1, member_wall_s=10, clock=lambda: clock[0])
    assert len(calls) == launched and json.loads((out/"ledger.json").read_text())["members"] == completed


def test_forecast_input_changed_during_the_run_is_refused(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import run
    path, raw = study_files
    study = load_study(path)
    calls = []
    base = make_fake_launch(calls)
    later = Path(raw["cases"][1]["input_dir"])/"wrfinput_d01"

    def launch(plan_path, **kwargs):
        rc = base(plan_path, **kwargs)
        later.write_bytes(later.read_bytes())  # rewritten while members run (new mtime)
        os.utime(later, ns=(1, 1))
        return rc
    monkeypatch.setattr(workflows, "launch", launch)
    with pytest.raises(ValueError, match="changed while the study ran"):
        run(study, tmp_path/"run")
    assert len(calls) == 1


def test_failed_retries_are_charged_against_the_case_hour_budget(study_files, tmp_path, monkeypatch):
    from gpuwrf.ens import workflows
    from gpuwrf.optimize.study import run
    study = load_study(study_files[0])  # budget 42 case-hours = planned 42
    calls = []
    out = tmp_path/"run"
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls, fail=(0, 3)))
    with pytest.raises(ValueError, match="failed with exit 7"):
        run(study, out)
    monkeypatch.setattr(workflows, "launch", make_fake_launch(calls))
    with pytest.raises(ValueError, match="compute budget exhausted before launch"):
        run(study, out, resume=True)
    ledger = json.loads((out/"ledger.json").read_text())
    assert ledger["scheduled_case_hours"] <= study["budget_case_hours"] and not (out/"result.json").exists()
