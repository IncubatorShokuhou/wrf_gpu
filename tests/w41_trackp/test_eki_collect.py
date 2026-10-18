"""EKI station operator: member wrfouts -> predictions matrix in one fixed key order; ALISIOS reads refused unguarded."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from netCDF4 import Dataset

from gpuwrf.surface_params import cli as spc
from gpuwrf.surface_params.stations import collect, is_alisios_data

T0 = pd.Timestamp("2026-01-01T00:00:00Z")


def _wrf(path, t2_offset, hours=3):
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", hours)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", 2)
        ds.createDimension("west_east", 2)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array([list((T0 + pd.Timedelta(hours=i)).strftime("%Y-%m-%d_%H:%M:%S")) for i in range(hours)],
                            dtype="S1")
        ds.setncattr("MAP_PROJ", 3)
        ds.setncattr("GRID_ID", 1)
        ds.setncattr("SIMULATION_START_DATE", "2026-01-01_00:00:00")
        fields = {"XLAT": ("degree_north", [[0, 0], [1, 1]]), "XLONG": ("degree_east", [[0, 1], [0, 1]]),
                  "LANDMASK": ("1", np.ones((2, 2))), "HGT": ("m", np.full((2, 2), 100)),
                  "T2": ("K", np.array([[280, 282], [284, 286]]) + t2_offset),
                  "U10": ("m s-1", np.full((2, 2), 3.0 + t2_offset)), "V10": ("m s-1", np.full((2, 2), 4)),
                  "RAINC": ("mm", np.zeros((2, 2))), "RAINNC": ("mm", np.zeros((2, 2)))}
        for name, (units, values) in fields.items():
            var = ds.createVariable(name, "f8", ("Time", "south_north", "west_east"))
            var.units = units
            var[:] = values
    return path


def _inputs(tmp_path, offsets=(0.0, 0.5, 1.0)):
    dirs = []
    for k, off in enumerate(offsets):
        d = tmp_path / f"member_{k}"
        d.mkdir()
        _wrf(d / "wrfout_d01_2026-01-01_00:00:00", off)
        dirs.append(d)
    stations = tmp_path / "stations.csv"
    pd.DataFrame([{"station_uid": "s:1", "lat": .5, "lon": .5, "elevation_m": 100, "landmask": 1,
                   "wind_height_m": 10, "roughness_m": .03},
                  {"station_uid": "s:2", "lat": .2, "lon": .7, "elevation_m": 100, "landmask": 1,
                   "wind_height_m": 10, "roughness_m": .03}]).to_csv(stations, index=False)
    obs = tmp_path / "obs.csv"
    rows = [{"station_uid": s, "valid_time": (T0 + pd.Timedelta(hours=h)).isoformat(), "variable": v,
             "value": val, "interval_start": (T0 + pd.Timedelta(hours=h)).isoformat()}
            for s in ("s:1", "s:2") for h in (1, 2) for v, val in (("t2", 10.0 + h), ("wind", 5.0))]
    rows.append({"station_uid": "s:1", "valid_time": (T0 + pd.Timedelta(hours=9)).isoformat(), "variable": "t2",
                 "value": 1.0, "interval_start": (T0 + pd.Timedelta(hours=9)).isoformat()})  # no model support
    pd.DataFrame(rows).to_csv(obs, index=False)
    return dirs, stations, obs


def test_collect_common_support_fixed_order(tmp_path):
    dirs, stations, obs = _inputs(tmp_path)
    members = [sorted(d.glob("wrfout_d01_*")) for d in dirs]
    pred, y, sig, keys = collect(members, stations, obs, sigma={"t2": 1.0, "wind": 1.5})
    assert pred.shape == (3, 8) and y.shape == (8,) and len(keys) == 8
    assert keys == sorted(keys)
    t2 = [i for i, k in enumerate(keys) if k[2] == "t2"]
    np.testing.assert_allclose(pred[1, t2] - pred[0, t2], 0.5, atol=1e-9)  # member T2 offsets survive sampling
    assert set(sig[t2]) == {1.0}
    with pytest.raises(ValueError, match="sigma required"):
        collect(members, stations, obs, sigma={"t2": 1.0})


def test_collect_cli_writes_eki_inputs(tmp_path):
    dirs, stations, obs = _inputs(tmp_path)
    out = tmp_path / "collected"
    argv = ["eki", "collect", "--stations", str(stations), "--obs", str(obs), "--sigma", "t2=1.0", "--sigma",
            "wind=1.5", "--output-dir", str(out)]
    for d in dirs:
        argv += ["--member-dir", str(d)]
    assert spc.main_args(argv) == 0
    pred = np.load(out / "predictions.npy")
    data = np.load(out / "obs.npz")
    assert pred.shape[0] == 3 and pred.shape[1] == data["y"].size == data["sigma"].size
    assert len(json.loads((out / "keys.json").read_text())) == pred.shape[1]


def test_alisios_observations_refused_without_guard(tmp_path):
    dirs, stations, _obs = _inputs(tmp_path)
    fake = "<DATA_ROOT>/alisios/__trackp_never_opened__/obs.csv"
    assert is_alisios_data(fake)
    members = [sorted(d.glob("wrfout_d01_*")) for d in dirs]
    with pytest.raises(ValueError, match="alisios_guard"):
        collect(members, stations, fake, sigma={"t2": 1.0, "wind": 1.5})
