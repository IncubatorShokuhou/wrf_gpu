"""Independent formulas and fake-member workflows, never forecast fidelity claims."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import pytest

from gpuwrf.ens.members import read_members
from gpuwrf.ens.statistics import crps, products, rank_histogram, reliability, spread_skill
from gpuwrf.ens.verification import verify_members, write_verification
from gpuwrf.ens.workflows import launch, prepare
from gpuwrf.ens.wrfout import write_products

REPO = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2026-01-01T00:00:00Z")


def write_wrf(path, *, offsets=(0, 1, 2), issue=0, value=0, variable="T2", volume=False):
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", len(offsets))
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 2)
        if volume:
            ds.createDimension("bottom_top", 2)
            ds.createVariable("ZNU", "f8", ("Time", "bottom_top"))[:] = [0.9, 0.2]
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([
            list((T0+pd.Timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S")) for h in offsets], dtype="S1")
        ds.setncatts({"GRID_ID": 1, "MAP_PROJ": 3, "DX": 1000., "DY": 1000.,
                     "SIMULATION_START_DATE": (T0+pd.Timedelta(hours=issue)).strftime("%Y-%m-%d_%H:%M:%S")})
        for name, data in (("XLAT", [[0, 0], [1, 1]]), ("XLONG", [[0, 1], [0, 1]]), ("LANDMASK", np.ones((2, 2)))):
            ds.createVariable(name, "f8", ("Time", "south_north", "west_east"))[:] = data
        dims = ("Time",)+(("bottom_top",) if volume else ())+ ("south_north", "west_east")
        field = ds.createVariable(variable, "f8", dims, fill_value=-9999)
        field.units = "K"
        field[:] = np.asarray(offsets).reshape((-1,)+(1,)*(len(dims)-1))+value
    return path


def members_json(path, files):
    path.write_text(json.dumps({"members": [{"id": f"m{i}", "files": [str(p)]} for i, p in enumerate(files)]}))
    return path


def test_hand_products_and_spread_skill():
    values = np.array([[0, 2], [2, 2], [4, 2]])
    got = products(values, thresholds=[0, 2, 4], quantiles=[0, 0.25, 0.5, 1])
    np.testing.assert_equal(got["mean"], [2, 2])
    np.testing.assert_equal(got["spread"], [2, 0])
    np.testing.assert_equal(got["quantile"], [[0, 2], [1, 2], [2, 2], [4, 2]])
    np.testing.assert_equal(got["probability"], [[2/3, 1], [1/3, 0], [0, 0]])
    assert spread_skill(values, [1, 3]) == pytest.approx({"n": 2, "rms_spread": np.sqrt(2),
                                                        "mean_rmse": 1, "spread_rmse_ratio": np.sqrt(2)})
    assert spread_skill(values, [2, 2])["spread_rmse_ratio"] is None


@pytest.mark.parametrize("n", [2, 3, 11])
@pytest.mark.parametrize("fair", [False, True])
def test_crps_independent_energy_sum(n, fair):
    random = np.random.default_rng(120+n)
    values = random.normal(size=(n, 2, 7))
    obs = random.normal(size=(2, 7))
    # Independent literal definition, deliberately not the sorted algorithm.
    expected = np.mean(np.abs(values-obs), axis=0)
    distances = sum(np.abs(a-b) for a in values for b in values)
    expected -= distances/(2*n*(n-1) if fair else 2*n*n)
    np.testing.assert_allclose(crps(values, obs, fair=fair), expected, rtol=2e-14, atol=2e-14)


def test_crps_hand_fair_and_large_offset():
    # M=2, x=(0,2), y=1: empirical 1 - 4/8 = 0.5; fair 1 - 4/4 = 0.
    assert crps([0, 2], 1) == 0.5
    assert crps([0, 2], 1, fair=True) == 0
    assert crps([1e12, 1e12+2], 1e12+1) == 0.5


def test_rank_ties_conserve_mass_and_are_permutation_invariant():
    values = [[0, 1, 2], [0, 2, 3], [0, 3, 4]]
    # all-tied y=0 => .25 each; y=2 tied once => ranks1/2; y=5 => rank3
    expected = [0.25, 0.75, 0.75, 1.25]
    np.testing.assert_equal(rank_histogram(values, [0, 2, 5]), expected)
    np.testing.assert_equal(rank_histogram(values[::-1], [0, 2, 5]), expected)
    assert rank_histogram(values, [0, 2, 5]).sum() == 3


def test_reliability_boundaries_empty_bins_and_counts():
    rows = reliability([0, 0.25, 0.5, 1], [0, 1, 0, 1], edges=[0, 0.25, 0.5, 0.75, 1])
    assert [r["n"] for r in rows] == [1, 1, 1, 1]
    assert [r["observed_frequency"] for r in rows] == [0, 1, 0, 1]
    assert rows[-1]["forecast_probability"] == 1
    empty = reliability([0, 1], [0, 1], edges=[0, 0.25, 0.5, 1])[1]
    assert empty["n"] == 0 and empty["forecast_probability"] is None and empty["observed_frequency"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("target", ["member", "obs", "threshold", "quantile", "probability", "event"])
def test_all_nonfinite_refused(bad, target):
    with pytest.raises(ValueError, match="NaN/Inf"):
        if target == "member":
            products([[0, bad], [1, 2]])
        elif target == "obs":
            crps([[0, 1], [1, 2]], [0, bad])
        elif target == "threshold":
            products([1, 2], thresholds=[bad])
        elif target == "quantile":
            products([1, 2], quantiles=[bad])
        elif target == "probability":
            reliability([0, bad], [0, 1])
        else:
            reliability([0, 1], [0, bad])


@pytest.mark.parametrize("function", [lambda: products([1]), lambda: products(np.empty((2, 0))),
    lambda: products([1, 2], quantiles=[1.1]), lambda: products([1, 2], thresholds=[1, 1]),
    lambda: crps([[1], [2]], 1), lambda: rank_histogram([[1], [2]], [1, 2]),
    lambda: reliability([1.1], [0]), lambda: reliability([0.1], [2]),
    lambda: reliability([0.1], [0], edges=[0.1, 1])])
def test_invalid_shapes_levels_events_refused(function):
    with pytest.raises(ValueError):
        function()


@pytest.mark.parametrize("volume", [False, True])
def test_grid_products_against_analytic_fields(tmp_path, volume):
    a = write_wrf(tmp_path/"a.nc", value=280, volume=volume)
    b = write_wrf(tmp_path/"b.nc", value=284, volume=volume)
    manifest = members_json(tmp_path/"members.json", [a, b])
    output = tmp_path/"ens.nc"
    result = write_products(manifest, "T2", output, thresholds=[282, 285], quantiles=[0.25, 0.5])
    assert result["frames"] == 3 and result["members"] == 2
    with Dataset(output) as ds:
        for i in range(3):
            np.testing.assert_equal(ds.variables["mean"][i], 282+i)
            np.testing.assert_allclose(ds.variables["spread"][i], np.sqrt(8))
            np.testing.assert_equal(ds.variables["quantile"][i, 0], 281+i)
        np.testing.assert_equal(ds.variables["probability"][:, 0], 0.5)
        assert ds.variables["mean"].units == "K" and ds.variables["probability"].units == "1"
        assert ds.spread_ddof == 1 and len(json.loads(ds.member_provenance)) == 2
    with pytest.raises(ValueError, match="overwrite"):
        write_products(manifest, "T2", output)


def test_lagged_valid_time_alignment_and_exclusions(tmp_path):
    a = write_wrf(tmp_path/"a.nc", offsets=[0, 1, 2], value=10)
    b = write_wrf(tmp_path/"b.nc", offsets=[1, 2, 3], issue=1, value=20)
    manifest = members_json(tmp_path/"members.json", [a, b])
    with pytest.raises(ValueError, match="valid times differ"):
        write_products(manifest, "T2", tmp_path/"exact.nc")
    write_products(manifest, "T2", tmp_path/"lag.nc", alignment="intersection")
    with Dataset(tmp_path/"lag.nc") as ds:
        np.testing.assert_equal(ds.variables["mean"][:, 0, 0], [16, 17])
        provenance = json.loads(ds.member_provenance)
        assert [row["excluded_times"] for row in provenance] == [1, 1]
        assert len({row["issue"] for row in provenance}) == 2


@pytest.mark.parametrize("change", ["units", "coordinates", "projection", "vertical", "nan", "inf", "fill", "domain", "duplicate_time"])
def test_grid_mismatch_nonfinite_and_masked_refused(tmp_path, change):
    a = write_wrf(tmp_path/"a.nc", value=280, volume=True)
    b = write_wrf(tmp_path/"b.nc", value=284, volume=True)
    with Dataset(b, "a") as ds:
        if change == "units":
            ds.variables["T2"].units = "degC"
        elif change == "coordinates":
            ds.variables["XLAT"][1, 0, 0] += 0.01  # later frame, not only first
        elif change == "projection":
            ds.DX = 2000
        elif change == "vertical":
            ds.variables["ZNU"][1, 0] = 0.8
        elif change == "domain":
            ds.GRID_ID = 2
        elif change == "duplicate_time":
            ds.variables["Times"][1] = ds.variables["Times"][0]
        else:
            ds.variables["T2"][1, 0, 0, 0] = {"nan": np.nan, "inf": np.inf, "fill": -9999}[change]
    manifest = members_json(tmp_path/"members.json", [a, b])
    with pytest.raises(ValueError):
        write_products(manifest, "T2", tmp_path/"bad.nc")
    assert not (tmp_path/"bad.nc").exists() and not list(tmp_path.glob(".ensemble-*.nc"))


def test_member_duplicates_refused(tmp_path):
    a = write_wrf(tmp_path/"a.nc")
    manifest = members_json(tmp_path/"members.json", [a, a])
    with pytest.raises(ValueError, match="duplicate member"):
        read_members(manifest)


def station_inputs(tmp_path, *, rain=False):
    station = tmp_path/"stations.csv"
    pd.DataFrame([dict(station_uid="s:1", lat=0.5, lon=0.5, elevation_m=0)]).to_csv(station, index=False)
    def rows(values, start=0):
        return pd.DataFrame([dict(station_uid="s:1", valid_time=(T0+pd.Timedelta(hours=i+start)).isoformat(),
            interval_start=(T0+pd.Timedelta(hours=i+start-(1 if rain else 0))).isoformat(),
            variable="precip_rate" if rain else "t2", value=value) for i, value in enumerate(values)])
    obs, a, b = tmp_path/"obs.csv", tmp_path/"a.csv", tmp_path/"b.csv"
    rows([1, 2, 3], start=1 if rain else 0).to_csv(obs, index=False)
    rows([0, 2, 4], start=1 if rain else 0).to_csv(a, index=False)
    rows([2, 2], start=1 if rain else 0).to_csv(b, index=False)
    manifest = members_json(tmp_path/"members.json", [a, b])
    return manifest, station, obs


@pytest.mark.parametrize("rain", [False, True])
def test_station_pairing_matches_shared_verify_and_hand_scores(tmp_path, rain):
    manifest, station, obs = station_inputs(tmp_path, rain=rain)
    report, table = verify_members(manifest, station, obs, variable="precip_rate" if rain else "t2", thresholds=[1])
    assert report["n"] == 2 and report["member_count"] == 2
    assert report["crps"] == 0.25 and report["fair_crps"] == 0
    assert report["support"]["member_rows"] == {"m0": 3, "m1": 2}
    assert report["spread_skill"]["rms_spread"] == 1
    assert table["mean"].tolist() == [1, 2]
    assert report["events"][0]["brier_score"] == 0.125
    assert report["units"] == ("mm/h" if rain else "degC")
    write_verification(report, table, tmp_path/"report")
    assert all((tmp_path/"report"/name).is_file() for name in ("ensemble.json", "paired.csv", "reliability.svg", "rank_histogram.svg", "spread_skill.svg"))


def test_station_wrfout_extraction_uses_verify_sampler(tmp_path):
    manifest, stations, obs = station_inputs(tmp_path)
    files = [write_wrf(tmp_path/f"m{i}.nc", value=273.15+2*i) for i in range(2)]
    manifest = members_json(manifest, files)
    report, table = verify_members(manifest, stations, obs)
    assert report["n"] == 3
    np.testing.assert_allclose(table["mean"], [1, 2, 3], atol=3e-14)
    assert report["extraction"]["m0"]["method"] == "bilinear"
    assert report["extraction"]["m0"]["sampling"][0]["weights"] == [0.25]*4


def test_close_levels_preserve_identifiers_and_values_in_exported_csv(tmp_path):
    station = tmp_path/"stations.csv"
    pd.DataFrame([dict(station_uid="s:1", lat=0.5, lon=0.5, elevation_m=0)]).to_csv(station, index=False)
    paths = []
    lower, upper = 1.00000015, 1.00000025
    for name, value in (("a", lower), ("b", upper), ("obs", 1.00000018)):
        path = tmp_path/f"{name}.csv"
        pd.DataFrame([dict(station_uid="s:1", valid_time=T0.isoformat(), variable="t2", value=value)]).to_csv(path, index=False)
        paths.append(path)
    thresholds = [1.0000001, 1.0000002]
    quantiles = [0.5000001, 0.5000002]
    inventory = members_json(tmp_path/"members.json", paths[:2])
    report, table = verify_members(inventory, station, paths[2], thresholds=thresholds, quantiles=quantiles)
    output = tmp_path/"report"
    write_verification(report, table, output)
    exported = pd.read_csv(output/"paired.csv", float_precision="round_trip")
    probability_columns = [name for name in exported if name.startswith("probability_gt_")]
    quantile_columns = [name for name in exported if name.startswith("quantile_")]
    assert len(probability_columns) == len(thresholds) and len(quantile_columns) == len(quantiles)
    assert [float(name.removeprefix("probability_gt_")) for name in probability_columns] == thresholds
    assert [float(name.removeprefix("quantile_")) for name in quantile_columns] == quantiles
    np.testing.assert_equal(exported[probability_columns].iloc[0].to_numpy(), [1, 0.5])
    expected = np.array([lower+(upper-lower)*q for q in quantiles])
    # A two-member linear interpolation is an independent reference. The close
    # results differ by ~45 ulp, well above CSV/input arithmetic roundoff.
    np.testing.assert_array_less(np.abs(exported[quantile_columns].iloc[0].to_numpy()-expected), 3*np.spacing(expected))
    assert exported[quantile_columns].iloc[0, 0] < exported[quantile_columns].iloc[0, 1]
    metadata = json.loads((output/"ensemble.json").read_text())
    assert [event["threshold"] for event in metadata["events"]] == thresholds
    assert metadata["quantile_levels"] == quantiles


@pytest.mark.parametrize("change", ["nan", "inf", "unknown_station", "no_overlap", "direction"])
def test_station_refusals(tmp_path, change):
    manifest, station, obs = station_inputs(tmp_path)
    data = pd.read_csv(obs)
    if change in ("nan", "inf"):
        data["value"] = data.value.astype(float)
        data.loc[0, "value"] = np.nan if change == "nan" else np.inf
    elif change == "unknown_station":
        data["station_uid"] = "unknown"
    elif change == "no_overlap":
        data["valid_time"] = "2030-01-01T00:00:00Z"
        data["interval_start"] = data.valid_time
        data = data.iloc[:1]
    data.to_csv(obs, index=False)
    with pytest.raises(ValueError):
        verify_members(manifest, station, obs, variable="wind_direction" if change == "direction" else "t2")


def fake_case(tmp_path, name, issue=0, dt=3600, coverage=30):
    case = tmp_path/name
    case.mkdir()
    write_wrf(case/"wrfinput_d01", offsets=[issue], issue=issue)
    write_wrf(case/"wrfbdy_d01", offsets=[issue, issue+coverage], issue=issue)
    stamp = T0+pd.Timedelta(hours=issue)
    (case/"namelist.input").write_text(f"""&time_control
 start_year={stamp.year}, start_month={stamp.month}, start_day={stamp.day},
 start_hour={stamp.hour}, start_minute=0, start_second=0,
 /
 &domains
 max_dom=1, time_step={dt},
 /
 &physics
 mp_physics=8,
 /
""")
    return case


def test_physics_shares_inputs_and_launch_delegates_scheduler(tmp_path, monkeypatch):
    case = fake_case(tmp_path, "case")
    n0, n1 = tmp_path/"p0.nml", tmp_path/"p1.nml"
    n0.write_text((case/"namelist.input").read_text())
    n1.write_text(n0.read_text().replace("mp_physics=8", "mp_physics=6"))
    # A history in the source case must not be linked or hashed as model input.
    (case/"wrfout_d01_old").write_text("old history")
    plan = prepare("physics", tmp_path/"prepared", case=case, namelists=[n0, n1], hours=3, gpus="0,1", max_per_gpu=2)
    assert len(plan["members"]) == 2 and plan["common_window"][0] == T0.isoformat()
    for member in plan["members"]:
        dest = Path(member["case"])
        assert (dest/"wrfinput_d01").is_symlink()
        assert (dest/"wrfinput_d01").resolve() == (case/"wrfinput_d01").resolve()
        assert not (dest/"wrfout_d01_old").exists()
        assert not (dest/"namelist.input").is_symlink()
        provenance = json.loads((dest/"member_manifest.json").read_text())
        native = json.loads(Path(provenance["source_manifest"]["path"]).read_text())
        assert native["schema"] == "GpuwrfReproducibility"
        assert "python_source_sha256" in native["gpuwrf"]
    assert len(list((tmp_path/"prepared"/"source_manifests").glob("*.json"))) == 1
    calls = []
    fake = tmp_path/"run_parallel_cases.sh"
    fake.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.delenv("GPUWRF_GPU_LOCK_FD", raising=False)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append((argv, kw)) or type("Result", (), {"returncode": 7})())
    assert launch(tmp_path/"prepared"/"plan.json", dry_run=True, launcher=fake) == 7
    argv = calls[0][0]
    assert "--dry-run" in argv and argv[argv.index("--gpus")+1] == "0,1"
    assert argv[-4:] == ["--", "--domains-from-namelist", "--hours", "3"]
    assert calls[0][1]["pass_fds"] == ()


def test_lagged_and_batch_members(tmp_path):
    cases = [fake_case(tmp_path, "a", 0), fake_case(tmp_path, "b", 6), fake_case(tmp_path, "c", 12)]
    plan = prepare("lagged", tmp_path/"lagged", cases=cases[::-1], hours=24)
    assert plan["product_alignment"] == "intersection"
    assert plan["common_window"] == [(T0+pd.Timedelta(hours=12)).isoformat(), (T0+pd.Timedelta(hours=24)).isoformat()]
    assert [m["issue"] for m in plan["members"]] == [(T0+pd.Timedelta(hours=i)).isoformat() for i in (0, 6, 12)]
    batch = prepare("batch", tmp_path/"batch", cases=[cases[0], cases[2]], hours=3)
    assert batch["mode"] == "batch" and batch["common_window"] is None


@pytest.mark.parametrize("change", ["input", "namelist", "manifest", "source_manifest"])
def test_prepared_input_drift_refused_before_scheduler(tmp_path, change):
    cases = [fake_case(tmp_path, "a"), fake_case(tmp_path, "b", 6)]
    plan = prepare("batch", tmp_path/"prepared", cases=cases, hours=3)
    member = Path(plan["members"][0]["case"])
    path = {"input": cases[0]/"wrfinput_d01", "namelist": member/"namelist.input", "manifest": member/"member_manifest.json",
            "source_manifest": tmp_path/"prepared"/"source_manifests"/"source-000.json"}[change]
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        launch(tmp_path/"prepared"/"plan.json", launcher=tmp_path/"unused")


@pytest.mark.parametrize("change", ["irregular", "no_overlap", "short_boundary", "wrong_clock", "dt", "duplicate"])
def test_invalid_workflow_refused_before_prepare(tmp_path, change):
    cases = [fake_case(tmp_path, "a"), fake_case(tmp_path, "b", 6), fake_case(tmp_path, "c", 13 if change == "irregular" else 12)]
    hours = 3 if change in ("no_overlap", "dt") else 24
    if change == "short_boundary":
        with Dataset(cases[0]/"wrfbdy_d01", "a") as ds:
            ds.variables["Times"][1] = np.array(list("2026-01-01_01:00:00"), dtype="S1")
    if change in ("wrong_clock", "dt"):
        path = cases[0]/"namelist.input"
        path.write_text(path.read_text().replace("start_hour=0", "start_hour=1") if change == "wrong_clock"
                        else path.read_text().replace("time_step=3600", "time_step=7"))
    if change == "duplicate":
        cases[1] = cases[0]
    with pytest.raises(ValueError):
        prepare("lagged", tmp_path/"refused", cases=cases, hours=hours)
    assert not (tmp_path/"refused").exists()


def test_physics_geometry_edit_refused(tmp_path):
    case = fake_case(tmp_path, "a")
    modified = tmp_path/"geometry.nml"
    modified.write_text((case/"namelist.input").read_text().replace("time_step=3600", "time_step=1800"))
    with pytest.raises(ValueError, match="only &physics"):
        prepare("physics", tmp_path/"bad", case=case, namelists=[case/"namelist.input", modified], hours=3)


def test_cli_help_is_backend_dark_and_products_run(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO/"src"), JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    code = """import sys
from gpuwrf.cli import main
try:
    main(['ens', '--help'])
except SystemExit as exc:
    assert exc.code == 0
assert not any(k.startswith('gpuwrf.physics') or k.startswith('gpuwrf.ens.statistics') for k in sys.modules)
bridge = sys.modules.get('jax._src.xla_bridge')
assert not getattr(bridge, '_backends', {})
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    a, b = write_wrf(tmp_path/"a.nc", value=280), write_wrf(tmp_path/"b.nc", value=284)
    manifest = members_json(tmp_path/"members.json", [a, b])
    command = [sys.executable, "-m", "gpuwrf.cli", "ens", "products", "--members", str(manifest),
               "--variable", "T2", "--output", str(tmp_path/"cli.nc")]
    result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["frames"] == 3


def test_prepare_is_backend_dark_and_cli_wired(tmp_path):
    cases = [fake_case(tmp_path, "a"), fake_case(tmp_path, "b", 6)]
    env = dict(os.environ, PYTHONPATH=str(REPO/"src"), JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    code = """import sys
from gpuwrf.cli import main
assert main(sys.argv[1:]) == 0
bridge = sys.modules.get('jax._src.xla_bridge')
assert not getattr(bridge, '_backends', {})
assert not any(k.startswith('gpuwrf.physics') for k in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code, "ens", "prepare", "--mode", "batch", "--cases",
                             *map(str, cases), "--hours", "3", "--output-dir", str(tmp_path/"batch")],
                            cwd=REPO, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_modified_plan_cannot_change_work_or_inject_scheduler_commands(tmp_path):
    cases = [fake_case(tmp_path, "a"), fake_case(tmp_path, "b", 6)]
    plan = prepare("batch", tmp_path/"prepared", cases=cases, hours=3)
    path = tmp_path/"prepared"/"plan.json"
    plan["cli_options"] = ["--domains-from-namelist", "--hours", "24"]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="forecast hours"):
        launch(path)
    plan["cli_options"][-1] = "3"
    plan["launcher_options"] += ["--cli-json", "arbitrary"]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="scheduler options"):
        launch(path)


def test_fake_members_end_to_end_through_actual_m1_scheduler(tmp_path, monkeypatch):
    case = fake_case(tmp_path, "source")
    n1 = tmp_path/"alternative.input"
    n1.write_text((case/"namelist.input").read_text().replace("mp_physics=8", "mp_physics=6"))
    prepare("physics", tmp_path/"prepared", case=case, namelists=[case/"namelist.input", n1], hours=3, gpus="0,1")
    fake_cli = tmp_path/"fake_cli.py"
    fake_cli.write_text('''import json,os,pathlib,shutil,sys
a=sys.argv
source=pathlib.Path(a[a.index('--input-dir')+1])
out=pathlib.Path(a[a.index('--output-dir')+1]);out.mkdir(parents=True,exist_ok=True)
shutil.copyfile(source/'wrfinput_d01',out/'wrfout_d01_2026-01-01_00:00:00')
(out/'fake_binding.json').write_text(json.dumps({'gpu':os.environ['CUDA_VISIBLE_DEVICES']}))
print(json.dumps({'all_domains_finite':True,'all_outputs_present':True}))
''')
    runner = tmp_path/"scheduler.py"
    runner.write_text(f'''import importlib.util,json,os,sys
spec=importlib.util.spec_from_file_location('parallel_cases',{str(REPO/'scripts/parallel_cases.py')!r})
pc=importlib.util.module_from_spec(spec);spec.loader.exec_module(pc)
os.environ['GPUWRF_GPU_LOCK_HELD']='1'  # simulated devices/CLI ONLY
pc.vram_by_pid=lambda:{{}}
devices=lambda:[dict(index=str(i),uuid='FAKE-'+str(i),name='Fake device',compute_cap='8.9',total_bytes=8*1024**3,free_bytes=8*1024**3) for i in range(2)]
argv=sys.argv[1:]
argv[0:0]=['--gpu-log-interval','0','--host-gb-per-case','0.001','--min-mem-available-gb','0','--cli-json',json.dumps([sys.executable,{str(fake_cli)!r}])]
sys.exit(pc.main(argv,sizer=lambda ds,a,e:[dict(need_bytes=1024) for d in ds],devices=devices,vram=lambda:{{}}))
''')
    wrapper = tmp_path/"run_parallel_cases.sh"
    # Exec retains the normal scheduler argv and environment into the fake
    # injection harness; no actual NVIDIA tools or CUDA backend are invoked.
    import shlex
    wrapper.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(runner))} \"$@\"\n")
    wrapper.chmod(0o755)
    monkeypatch.setenv("PYTHONPATH", str(REPO/"src"))
    monkeypatch.delenv("GPUWRF_GPU_LOCK_FD", raising=False)
    for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
        monkeypatch.delenv(key, raising=False)
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(key, raising=False)
    assert launch(tmp_path/"prepared/plan.json", launcher=wrapper) == 0
    members = []
    bindings = []
    for directory in sorted((tmp_path/"prepared/runs").glob("member-*")):
        receipt = json.loads((directory/"receipt.json").read_text())
        assert receipt["rc"] == 0
        bindings.append(json.loads((directory/"wrfout/fake_binding.json").read_text())["gpu"])
        members.append(directory/"wrfout/wrfout_d01_2026-01-01_00:00:00")
    assert sorted(bindings) == ["0", "1"]
    inventory = members_json(tmp_path/"members.json", members)
    result = write_products(inventory, "T2", tmp_path/"products.nc")
    assert result["members"] == 2 and result["frames"] == 1
    with Dataset(tmp_path/"products.nc") as ds:
        np.testing.assert_equal(ds.variables["mean"][:], 0)
        np.testing.assert_equal(ds.variables["spread"][:], 0)


def test_launch_preserves_only_validated_official_descriptor(tmp_path, monkeypatch):
    from gpuwrf.ens import workflows

    cases = [fake_case(tmp_path, "a"), fake_case(tmp_path, "b", 6)]
    prepare("batch", tmp_path/"prepared", cases=cases, hours=3)
    fake = tmp_path/"fake_launcher"
    fake.write_text("fake")
    lock = tmp_path/"fake_gpu.lock"
    lock.write_text("fake lock, no GPU")
    monkeypatch.setattr(workflows, "GPU_LOCK_FILE", lock)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(kw) or type("Result", (), {"returncode": 0})())
    with lock.open() as handle:
        monkeypatch.setenv("GPUWRF_GPU_LOCK_FD", str(handle.fileno()))
        assert launch(tmp_path/"prepared/plan.json", launcher=fake) == 0
        assert calls[-1]["pass_fds"] == (handle.fileno(),)
    with fake.open() as unrelated:
        monkeypatch.setenv("GPUWRF_GPU_LOCK_FD", str(unrelated.fileno()))
        with pytest.raises(ValueError, match="official lock"):
            launch(tmp_path/"prepared/plan.json", launcher=fake)
    for bad in ("0", "2", "-1", "bad"):
        monkeypatch.setenv("GPUWRF_GPU_LOCK_FD", bad)
        with pytest.raises(ValueError, match="descriptor"):
            launch(tmp_path/"prepared/plan.json", launcher=fake)


@pytest.mark.parametrize("action", ["products", "verify", "prepare"])
def test_offline_cli_never_imports_jax(tmp_path, action):
    manifest, stations, obs = station_inputs(tmp_path)
    if action == "products":
        manifest = members_json(manifest, [write_wrf(tmp_path/"a.nc", value=280), write_wrf(tmp_path/"b.nc", value=284)])
        args = ["ens", "products", "--members", str(manifest), "--variable", "T2", "--output", str(tmp_path/"ens.nc")]
    elif action == "verify":
        args = ["ens", "verify", "--members", str(manifest), "--stations", str(stations), "--obs", str(obs),
                "--output-dir", str(tmp_path/"verification")]
    else:
        cases = [fake_case(tmp_path, "a"), fake_case(tmp_path, "b", 6)]
        args = ["ens", "prepare", "--mode", "batch", "--cases", *map(str, cases), "--hours", "3", "--output-dir", str(tmp_path/"prepared")]
    code = '''import importlib.abc,sys
class RefuseJax(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='jax' or fullname.startswith('jax.'):
            raise AssertionError('Offline ens must not import JAX')
sys.meta_path.insert(0,RefuseJax())
sys.argv=['gpuwrf',*sys.argv[1:]]
from gpuwrf.cli import main
assert main(sys.argv[1:])==0
assert not any(k=='jax' or k.startswith('jax.') for k in sys.modules)
'''
    env = dict(os.environ, PYTHONPATH=str(REPO/"src"), JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    result = subprocess.run([sys.executable, "-c", code, *args], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
