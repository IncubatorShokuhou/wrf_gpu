"""Independent arithmetic/analytic cases for offline station verification."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from netCDF4 import Dataset

from gpuwrf.verify.engine import verify, write_report
from gpuwrf.verify.observations import read_forecast_csv, read_observations, read_stations, validate_rows
from gpuwrf.verify.providers import fetch_meteostat, meteostat_rows
from gpuwrf.verify.sampling import StationSampler, earth_wind
from gpuwrf.verify.scores import aggregate_intervals, precip_scores, scalar_scores, score_sources
from gpuwrf.verify.wrfout import extract_wrfout

T0 = pd.Timestamp("2026-01-01T00:00:00Z")
GOLDEN = Path(__file__).parent/"fixtures"/"station_scores_golden.json"


def records(variable, values, *, uid="test:1", hours=None, start_hour=1):
    return pd.DataFrame([{"station_uid": uid, "valid_time": T0+pd.Timedelta(hours=i+start_hour),
                          "variable": variable, "value": value,
                          "interval_start": T0+pd.Timedelta(hours=i+start_hour-(hours or 0))}
                         for i, value in enumerate(values)])


def stations(lat=0.5, lon=0.5, **extra):
    return pd.DataFrame([{"station_uid": "test:1", "lat": lat, "lon": lon, "elevation_m": 200,
                          "landmask": 1, "wind_height_m": 10, "roughness_m": 0.03, **extra}])


def write_wrf(path, *, u=None, v=None, rain=(0, 2, 5), start=0, **attributes):
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", len(rain))
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 2)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array([list((T0+pd.Timedelta(hours=start+i)).strftime("%Y-%m-%d_%H:%M:%S")) for i in range(len(rain))], dtype="S1")
        ds.setncattr("MAP_PROJ", 3)
        ds.setncattr("GRID_ID", 1)
        ds.setncattr("SIMULATION_START_DATE", "2026-01-01_00:00:00")
        for key, value in attributes.items():
            ds.setncattr(key, value)
        fields = {"XLAT": ("degree_north", [[0, 0], [1, 1]]), "XLONG": ("degree_east", [[0, 1], [0, 1]]),
                  "LANDMASK": ("1", np.ones((2, 2))), "HGT": ("m", np.full((2, 2), 100)),
                  "T2": ("K", [[280, 282], [284, 286]]),
                  "U10": ("m s-1", np.full((2, 2), 3) if u is None else u),
                  "V10": ("m s-1", np.full((2, 2), 4) if v is None else v),
                  "RAINC": ("mm", np.zeros((2, 2))), "RAINNC": ("mm", np.zeros((2, 2))),
                  "TEST_GUST": ("m s-1", np.full((2, 2), 9))}
        for name, (units, values) in fields.items():
            field = ds.createVariable(name, "f8", ("Time", "south_north", "west_east"))
            field.units = units
            field[:] = values
        ds.variables["RAINNC"][:] = np.array(rain)[:, None, None]
    return path


def test_hand_computed_golden():
    golden = json.loads(GOLDEN.read_text())
    assert scalar_scores([2, 2, 1, 5], [1, 2, 3, 4]) == pytest.approx(golden["scalar"])
    assert precip_scores([1, 1, 0, 1, 0], [1, 1, 1, 0, 0], 1) == pytest.approx(golden["precip"])
    obs = pd.concat([records("wind", [5, 5, 4]), records("wind_direction", [350, 10, 90])])
    pred = pd.concat([records("wind", [5, 5, 8]), records("wind_direction", [10, 350, 180])])
    result = score_sources(obs, {"cpu": pred})
    direction = next(row for row in result["metrics"] if row["variable"] == "wind_direction")
    for key, value in golden["direction"].items():
        assert direction[key] == pytest.approx(value)
    assert direction["weak_wind_excluded"] == 1


def test_vector_wind_and_common_support():
    obs = pd.concat([records("wind", [5, 5, 5]), records("wind_direction", [0, 0, 0]), records("t2", [1, 2, 3])])
    candidate = pd.concat([records("wind", [5, 5, 5]), records("wind_direction", [0, 90, 180]), records("t2", [2, 3, 100])])
    baseline = obs[~((obs.variable == "t2") & (obs.valid_time == T0+pd.Timedelta(hours=3)))].copy()
    result = score_sources(obs, {"gpu": candidate, "cpu": baseline})
    wind = next(row for row in result["metrics"] if row["source"] == "gpu" and row["variable"] == "wind")
    assert wind["rmse"] == 0
    assert wind["vector_rmse"] == pytest.approx(math.sqrt(50))  # vector squared errors 0,50,100
    assert wind["vector_n"] == 3
    temperature = [row for row in result["metrics"] if row["variable"] == "t2"]
    assert [row["n"] for row in temperature] == [2, 2]
    assert [row["rmse"] for row in temperature] == [1, 0]
    comparison = next(row for row in result["comparisons"] if row["variable"] == "t2")
    assert comparison["rmse_ratio"] is None


def test_explicit_components_preserve_scalar_speed_arm():
    obs = pd.concat([records("wind", [5]), records("wind_direction", [0])])
    pred = records("wind", [5])
    pred["u_ms"], pred["v_ms"] = 3, -1  # separately interpolated vector has magnitude sqrt(10), not 5
    result = score_sources(obs, {"a": pred})
    wind = next(row for row in result["metrics"] if row["variable"] == "wind")
    assert wind["rmse"] == 0
    assert wind["vector_rmse"] == 5  # sqrt(3²+4²)


@pytest.mark.parametrize("component", ["u_ms", "v_ms"])
def test_partial_wind_component_pair_refused(tmp_path, component):
    data = pd.concat([records("wind", [5]), records("wind_direction", [0])])
    data[component] = 3
    path = tmp_path/"partial.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="both u_ms and v_ms"):
        read_forecast_csv(path)
    with pytest.raises(ValueError, match="both u_ms and v_ms"):
        score_sources(data, {"candidate": data})


@pytest.mark.parametrize("component", ["u_ms", "v_ms"])
def test_declared_wind_component_pair_must_be_finite(component):
    data = records("wind", [5])
    data["u_ms"], data["v_ms"] = 3, 4
    data[component] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_rows(data)


def test_direction_threshold_is_common_to_every_source():
    obs = pd.concat([records("wind", [5, 5]), records("wind_direction", [0, 0])])
    candidate = pd.concat([records("wind", [5, 5]), records("wind_direction", [0, 180])])
    baseline = pd.concat([records("wind", [5, 4]), records("wind_direction", [0, 90])])
    result = score_sources(obs, {"a": candidate, "b": baseline})
    assert [row["n"] for row in result["metrics"] if row["variable"] == "wind_direction"] == [1, 1]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_truth_forecast_and_score_refused(bad):
    for p, o in (([bad], [1]), ([1], [bad])):
        with pytest.raises(ValueError, match="NaN/Inf"):
            scalar_scores(p, o)
        with pytest.raises(ValueError, match="NaN/Inf"):
            precip_scores(p, o, 1)
    for source in (True, False):
        good, corrupt = records("t2", [1]), records("t2", [bad])
        with pytest.raises(ValueError, match="NaN/Inf"):
            score_sources(good if source else corrupt, {"a": corrupt if source else good})


def test_precip_no_events_and_bad_threshold():
    value = precip_scores([0, 0], [0, 0], 1)
    assert value["ets"] is None and value["fbi"] is None
    for threshold in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            precip_scores([0], [0], threshold)


def test_precip_integrates_exact_windows_gaps_are_unavailable():
    native = records("precip_rate", [1, 2, 3], hours=1)
    target = records("precip_rate", [0], hours=3, start_hour=3)
    aggregated = aggregate_intervals(native, target)
    assert aggregated.value.tolist() == [2]  # 6 mm / 3 h
    assert aggregate_intervals(native.drop(index=1), target).empty
    overlap = native.copy()
    overlap.loc[1, "interval_start"] = T0
    with pytest.raises(ValueError, match="overlapping"):
        aggregate_intervals(overlap, target)


def test_gust_windows_are_part_of_common_support():
    obs = records("gust", [9], hours=1)
    pred = records("gust", [8])  # instantaneous, no 1 h maximum
    row = next(row for row in score_sources(obs, {"a": pred})["metrics"] if row["variable"] == "gust")
    assert row["n"] == 0


def test_obsdb_adapter_qc_circular_mean_and_gust(tmp_path):
    path = tmp_path/"obs.csv"
    pd.DataFrame({"station_id": ["test:1"]*7,
                  "ts_utc": ["2026-01-01 01:10:00", "2026-01-01 01:20:00"]*3+["2026-01-01 01:30:00"],
                  "variable": ["t2_c", "t2_c", "wind_dir_deg", "wind_dir_deg", "wind_gust_ms", "wind_gust_ms", "t2_c"],
                  "value": [10, 12, 350, 10, 5, 8, 100], "qc_flag": [0]*6+[1]}).to_csv(path, index=False)
    rows, audit = read_observations(path, "alisios")
    assert audit["qc_excluded"] == 1
    values = dict(zip(rows.variable, rows.value))
    assert values["t2"] == 11 and values["gust"] == 8
    assert min(abs(values["wind_direction"]), abs(values["wind_direction"]-360)) < 1e-10
    gust = rows[rows.variable == "gust"].iloc[0]
    assert gust.valid_time == T0+pd.Timedelta(hours=2)
    assert gust.interval_start == T0+pd.Timedelta(hours=1)


def test_obsdb_precip_requires_declared_duration(tmp_path):
    path = tmp_path/"obs.csv"
    pd.DataFrame({"station_id": ["test:1"], "ts_utc": ["2026-01-01 03:00:00"],
                  "variable": ["precip_mm"], "value": [6], "qc_flag": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="precip_hours"):
        read_observations(path, "alisios")
    rows, _ = read_observations(path, "alisios", precip_hours=3)
    assert rows.value.tolist() == [2]


@pytest.mark.parametrize("native,values", [
    ("wind_speed_ms", [-2, 12]), ("wind_gust_ms", [-2, 12]),
    ("wind_dir_deg", [400, 20]), ("wind_dir_deg", [-20, 20]),
    ("precip_rate_mmh", [-2, 12]), ("precip_mm", [-2, 12]),
])
def test_native_bounds_refused_before_hourly_aggregation(tmp_path, native, values):
    path = tmp_path/"invalid-native.csv"
    pd.DataFrame({"station_id": ["test:1"]*2, "ts_utc": ["2026-01-01 01:10", "2026-01-01 01:20"],
                  "variable": [native]*2, "value": values, "qc_flag": [0, 0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="native"):
        read_observations(path, "alisios")


@pytest.mark.parametrize("native,bad,good,canonical", [
    ("wind_speed_ms", -2, 12, "wind"), ("wind_gust_ms", -2, 12, "gust"),
    ("wind_dir_deg", 400, 20, "wind_direction"),
])
def test_native_bounds_apply_after_counted_qc_exclusions(tmp_path, native, bad, good, canonical):
    path = tmp_path/"qc-native.csv"
    pd.DataFrame({"station_id": ["test:1"]*2, "ts_utc": ["2026-01-01 01:10", "2026-01-01 01:20"],
                  "variable": [native]*2, "value": [bad, good], "qc_flag": [1, 0]}).to_csv(path, index=False)
    rows, audit = read_observations(path, "alisios")
    assert audit["qc_excluded"] == 1
    assert rows.variable.tolist() == [canonical]
    assert rows.value.tolist() == pytest.approx([good])


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_declared_issue_time_requires_every_row(tmp_path, missing):
    data = records("t2", [1, 2])
    data["issue_time"] = ["2026-01-01T00:00:00Z", missing]
    path = tmp_path/"missing-issue.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="issue_time must be non-missing"):
        read_forecast_csv(path)


def test_issue_metadata_optional_or_single_complete_issue(tmp_path):
    data = records("t2", [1, 2])
    path = tmp_path/"forecast.csv"
    data.to_csv(path, index=False)
    assert len(read_forecast_csv(path)) == 2
    data["issue_time"] = "2026-01-01T00:00:00Z"
    data.to_csv(path, index=False)
    assert len(read_forecast_csv(path)) == 2


def test_schema_refuses_duplicates_timezone_negative_and_missing_interval():
    good = records("t2", [1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(pd.concat([good, good]))
    good["valid_time"] = ["2026-01-01 01:00:00"]
    with pytest.raises(ValueError, match="UTC/offset"):
        validate_rows(good)
    with pytest.raises(ValueError, match="positive"):
        validate_rows(records("precip_rate", [1]))
    with pytest.raises(ValueError, match="negative"):
        validate_rows(records("wind", [-1]))


def test_station_metadata_aliases_and_default_assumptions(tmp_path):
    path = tmp_path/"stations.csv"
    pd.DataFrame({"station_id": ["s:1"], "lat": [28], "lon": [-16], "altitude_m": [100], "altitude_verified_m": [110],
                  "meteostat_id": ["00123"]}).to_csv(path, index=False)
    row = read_stations(path).iloc[0]
    assert row.elevation_m == 110 and row.wind_height_m == 10
    assert row.meteostat_id == "00123"


def test_alisios_issued_wind_export_and_source_selection(tmp_path):
    path = tmp_path/"pred.csv"
    wind = records("wind", [5]).rename(columns={"value": "prediction_value"})
    wind["prediction_dir_deg"] = 90
    wind["prediction_u_ms"], wind["prediction_v_ms"] = -4, 0
    wind["source_id"] = "wrf"
    wind.to_csv(path, index=False)
    data = read_forecast_csv(path)
    assert set(data.variable) == {"wind", "wind_direction"}
    assert data[data.variable.eq("wind_direction")].value.tolist() == [90]
    assert data[data.variable.eq("wind")].u_ms.tolist() == [-4]
    other = wind.copy()
    other["source_id"], other["valid_time"], other["interval_start"] = "aifs", T0, T0
    pd.concat([wind, other]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one forecast source"):
        read_forecast_csv(path)


def test_bilinear_analytic_plane_land_nearest_and_outside():
    lat, lon = np.array([[0, 0], [1, 1]]), np.array([[0, 1], [0, 1]])
    field = 2*lon+4*lat+10
    sampler = StationSampler(lat, lon, stations(0.25, 0.75), landmask=np.ones((2, 2)))
    assert sampler.sample(field)[0] == pytest.approx(12.5)  # f(.75,.25), independent analytic plane
    coast = StationSampler(lat, lon, stations(), landmask=[[1, 0], [1, 0]])
    assert coast.sample([[10, 100], [30, 100]])[0] == 20
    near = StationSampler(lat, lon, stations(0.1, 0.9), landmask=[[1, 0], [1, 0]], method="nearest")
    assert near.sample([[10, 100], [30, 100]])[0] == 10
    with pytest.raises(ValueError, match="outside"):
        StationSampler(lat, lon, stations(5, 5), landmask=np.ones((2, 2)))
    with pytest.raises(ValueError, match="land/sea"):
        StationSampler(lat, lon, stations(), landmask=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="NaN/Inf"):
        coast.sample([[10, np.nan], [30, 100]])


def test_curvilinear_bilinear_and_dateline():
    # Known bilinear cell mapping at x=.3,y=.6: lon=.3+.2*.3*.6, lat=.6+.1*.3*.6
    lat, lon = [[0, 0], [1, 1.1]], [[0, 1], [0, 1.2]]
    sampler = StationSampler(lat, lon, stations(0.618, 0.336), land_aware=False)
    assert sampler.sample([[0, 1], [2, 3]])[0] == pytest.approx(1.5)
    sampler = StationSampler([[0, 0], [1, 1]], [[179, -179], [179, -179]], stations(0.5, 180), land_aware=False)
    assert sampler.sample([[0, 2], [2, 4]])[0] == 2


def test_wind_rotation_ncar_reference_equations():
    u, v = np.ones((2, 2)), np.zeros((2, 2))
    lat, lon = np.full((2, 2), 45), np.full((2, 2), 90)
    # Polar stereographic cone=1: 90 degree clockwise rotation at lon-stand=90.
    ue, ve = earth_wind(u, v, lat, lon, {"MAP_PROJ": 2, "STAND_LON": 0, "TRUELAT1": 60})
    np.testing.assert_allclose(ue, 0, atol=1e-15)
    np.testing.assert_allclose(ve, -1)
    ue, ve = earth_wind(u, v, lat, lon, {}, cosalpha=np.zeros((2, 2)), sinalpha=np.ones((2, 2)))
    np.testing.assert_allclose(ue, 0)
    np.testing.assert_allclose(ve, 1)
    with pytest.raises(ValueError, match="metadata"):
        earth_wind(u, v, lat, lon, {})


def test_wrfout_extraction_units_lapse_wind_height_and_rain(tmp_path):
    path = write_wrf(tmp_path/"wrfout.nc")
    rows, audit = extract_wrfout([path], stations(), lapse_rate=0.0065, gust_variable="TEST_GUST", gust_hours=1)
    assert rows[rows.variable == "t2"].value.tolist() == pytest.approx([9.2]*3)  # mean 283 K -273.15 -100m*.0065
    assert rows[rows.variable == "wind"].value.tolist() == [5]*3
    assert rows[rows.variable == "precip_rate"].value.tolist() == [2, 3]
    assert rows[rows.variable == "gust"].value.tolist() == [9]*3
    assert audit["frames"] == 3
    height, _ = extract_wrfout([path], stations(wind_height_m=20), wind_height="neutral-log")
    expected = 5*math.log(20/.03)/math.log(10/.03)
    assert height[height.variable == "wind"].value.iloc[0] == pytest.approx(expected)


def test_wrfout_speed_interpolated_before_vector_norm(tmp_path):
    path = write_wrf(tmp_path/"wrfout.nc", u=[[5, -5], [5, -5]], v=[[0, 0], [0, 0]])
    rows, audit = extract_wrfout([path], stations())
    wind = rows[rows.variable == "wind"]
    assert wind.value.tolist() == [5]*3
    assert wind.u_ms.tolist() == [0]*3 and wind.v_ms.tolist() == [0]*3
    assert rows[rows.variable == "wind_direction"].empty  # atan2(0,0) is undefined
    assert audit["undefined_vector_direction_excluded"] == 3


def test_precip_reset_cannot_hide_under_bilinear_weights(tmp_path):
    path = write_wrf(tmp_path/"wrfout.nc", rain=(0, 2))
    with Dataset(path, "r+") as ds:
        ds.variables["RAINNC"][0] = [[10, 0], [0, 0]]
        ds.variables["RAINNC"][1] = [[0, 10], [10, 10]]
    # Interpolated total increases 2.5 -> 7.5, but one contributor reset.
    with pytest.raises(ValueError, match="reset/decrease"):
        extract_wrfout([path], stations())


def test_wrfout_scalar_speed_precedes_rounded_rotation(tmp_path):
    path = write_wrf(tmp_path/"wrfout.nc")
    with Dataset(path, "r+") as ds:
        for name, value in (("COSALPHA", 1+2e-8), ("SINALPHA", 0)):
            ds.createVariable(name, "f8", ("Time", "south_north", "west_east"))[:] = value
    rows, _ = extract_wrfout([path], stations())
    wind = rows[rows.variable == "wind"]
    assert wind.value.tolist() == [5]*3  # original hypot(3,4), not rotated norm
    assert (np.hypot(wind.u_ms, wind.v_ms) > 5).all()


@pytest.mark.parametrize("change,match", [("reset", "reset/decrease"), ("nan", "NaN/Inf"), ("units", "units"), ("bucket", "bucketed")])
def test_wrfout_refuses_bad_fields(tmp_path, change, match):
    path = write_wrf(tmp_path/"wrfout.nc", rain=(0, 2, 1) if change == "reset" else (0, 2, 5))
    with Dataset(path, "r+") as ds:
        if change == "nan":
            ds.variables["T2"][1, 0, 0] = np.nan
        if change == "units":
            ds.variables["T2"].units = "degC"
        if change == "bucket":
            ds.createVariable("I_RAINNC", "i4", ("Time", "south_north", "west_east"))[:] = 1
    with pytest.raises(ValueError, match=match):
        extract_wrfout([path], stations())


def test_meteostat_metric_conversion_and_missing():
    native = pd.DataFrame({"temp": [20, np.nan], "wspd": [36, 18], "wdir": [90, 180], "prcp": [2, 0]},
                          index=pd.date_range("2026-01-01", periods=2, freq="h"))
    rows, missing = meteostat_rows(native, "test:1")
    assert rows[rows.variable == "wind"].value.tolist() == [10, 5]
    assert rows[rows.variable == "t2"].value.tolist() == [20] and missing == 1
    assert rows[rows.variable == "precip_rate"].value.tolist() == [2, 0]


def test_optional_provider_never_requests_model_fills(monkeypatch):
    calls = []
    native = pd.DataFrame({"temp": [20], "wspd": [36]}, index=[T0])
    class TimeSeries:
        def fetch(self, **kwargs):
            calls.append(kwargs)
            return native
    def hourly(station, start, end, **kwargs):
        assert station == "00123"  # string lookup resolves native provider identifiers
        calls.append(kwargs)
        return TimeSeries()
    fake = SimpleNamespace(hourly=hourly, Station=lambda **kwargs: kwargs, Provider=SimpleNamespace(METAR="metar", ISD_LITE="isd_lite"), UnitSystem=SimpleNamespace(METRIC="metric"))
    monkeypatch.setitem(sys.modules, "meteostat", fake)
    data, audit = fetch_meteostat(stations(meteostat_id="00123"), T0, T0+pd.Timedelta(hours=1), provider="metar")
    assert calls == [{"timezone": "UTC", "providers": ["metar"]}, {"fill": False, "clean": True, "units": "metric"}]
    assert audit["model_substitutes"] is False and len(data) == 2


@pytest.mark.parametrize("native", [None, pd.DataFrame()])
def test_optional_provider_no_data_is_cleanly_refused(monkeypatch, native):
    fake = SimpleNamespace(hourly=lambda *args, **kwargs: SimpleNamespace(fetch=lambda **options: native),
                           Provider=SimpleNamespace(METAR="metar", ISD_LITE="isd_lite"), UnitSystem=SimpleNamespace(METRIC="metric"))
    monkeypatch.setitem(sys.modules, "meteostat", fake)
    with pytest.raises(ValueError, match="no observation measurements"):
        fetch_meteostat(stations(meteostat_id="00123"), T0, T0+pd.Timedelta(hours=1))


def test_cli_end_to_end_and_safe_html(tmp_path):
    station = tmp_path/"stations.csv"
    stations().to_csv(station, index=False)
    obs, pred, base = (tmp_path/name for name in ("obs.csv", "pred.csv", "base.csv"))
    records("t2", [1, 2, 3, 4]).to_csv(obs, index=False)
    records("t2", [2, 2, 1, 5]).to_csv(pred, index=False)
    records("t2", [3, 4, 5, 6]).to_csv(base, index=False)
    output = tmp_path/"report"
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-m", "gpuwrf.cli", "verify", "--stations", str(station), "--obs", str(obs),
               "--model", str(pred), "--compare", str(base), "--output-dir", str(output), "--candidate-name", "<script>alert(1)</script>"]
    result = subprocess.run(command, cwd=repo, env={**os.environ, "PYTHONPATH": str(repo/"src")}, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((output/"verification.json").read_text())
    temperature = next(row for row in report["metrics"] if row["variable"] == "t2")
    assert temperature["rmse"] == pytest.approx(math.sqrt(1.5))
    assert len(report["inputs"]) == 4 and all(len(item["sha256"]) == 64 for item in report["inputs"])
    assert "<script>" not in (output/"verification.html").read_text()
    assert (output/"verification.md").exists()
    records("t2", [np.inf]).to_csv(pred, index=False)
    result = subprocess.run(command, cwd=repo, env={**os.environ, "PYTHONPATH": str(repo/"src")}, text=True, capture_output=True, timeout=30)
    assert result.returncode == 2 and "NaN/Inf" in result.stderr


def test_engine_real_file_sampling_against_analytic_fields(tmp_path):
    station, obs = tmp_path/"stations.csv", tmp_path/"obs.csv"
    stations().to_csv(station, index=False)
    # Exact independent 283 K plane-centre value; no lapse by default.
    records("t2", [9.85, 9.85, 9.85], start_hour=0).to_csv(obs, index=False)
    wrf = write_wrf(tmp_path/"wrfout.nc")
    report = verify(station, obs, [wrf])
    row = next(row for row in report["metrics"] if row["variable"] == "t2")
    assert row["rmse"] < 1e-12
    assert report["extraction"]["candidate"]["lapse_rate_K_per_m"] == 0


def test_report_rejects_nonfinite_before_writing(tmp_path):
    report = score_sources(records("t2", [1]), {"a": records("t2", [1])})
    report["bad"] = np.nan
    with pytest.raises(ValueError):
        write_report(report, tmp_path/"report")
    assert not (tmp_path/"report"/"verification.json").exists()


def test_parser_registration_keeps_backend_and_data_imports_dark():
    repo = Path(__file__).resolve().parents[1]
    script = """
import sys
from gpuwrf.cli import build_parser
import jax
assert not jax._src.xla_bridge.backends_are_initialized()
build_parser().parse_args(['verify','--help'])
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=repo,
                            env={**os.environ, "PYTHONPATH": str(repo/"src")}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    script = """
import sys
from gpuwrf.cli import build_parser
import jax
before = set(sys.modules)
build_parser()
assert not jax._src.xla_bridge.backends_are_initialized()
assert 'gpuwrf.verify.engine' not in sys.modules
assert 'gpuwrf.verify.observations' not in sys.modules
assert not any(name.startswith(('pandas','netCDF4')) for name in set(sys.modules)-before)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=repo,
                            env={**os.environ, "PYTHONPATH": str(repo/"src")}, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
