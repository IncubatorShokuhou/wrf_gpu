"""sfclay=0 history vs original CPU-WRF, through the real NetCDF writer."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
import pytest

from gpuwrf.integration import daily_pipeline as daily
from gpuwrf.io.disabled_surface_history import HELD_SURFACE_FIELDS, load_disabled_surface_history
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.auxhist_stream import AuxhistStreamConfig
from gpuwrf.io.wrfout_writer import bind_wrfout_domain_authority

from test_initial_history_wrf_t0 import _writer
from test_m7_netcdf_writer import synthetic_case, authenticated_source_grid

FIXTURE = Path(__file__).resolve().parents[1] / "proofs/t_c01nophys/held_surface.npz"


def _input(path, arrays, *, missing_th2=False):
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", 4)
        ds.createDimension("west_east", 5)
        for name in HELD_SURFACE_FIELDS:
            if missing_th2 and name == "TH2":
                continue
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = arrays[f"input_{name}"]


def _loaded(tmp_path, *, missing_th2=False):
    with np.load(FIXTURE) as archive:
        arrays = {k: archive[k] for k in archive.files}
    _input(tmp_path / "wrfinput_d01", arrays, missing_th2=missing_th2)
    run = SimpleNamespace(
        namelist={"physics": {"sf_sfclay_physics": [0]}},
        wrfinput_file=lambda _: tmp_path / "wrfinput_d01", history_files=lambda _: [],
    )
    fields, _ = daily._load_static_latlon_writer_diagnostics(run, "d01", grid=SimpleNamespace(ny=4, nx=5))
    return arrays, fields


@pytest.mark.parametrize("hour", [0, 1, 2, 24])
def test_native_written_surface_is_byte_exact_original_wrf(tmp_path, monkeypatch, hour):
    def changed_diagnostics(*a, **kw):
        return {name: np.full((4, 5), 123.0, np.float32) for name in HELD_SURFACE_FIELDS}

    writer, state = _writer(tmp_path, changed_diagnostics)
    arrays, fields = _loaded(tmp_path)
    # Real writer receives an evolved atmosphere and a deliberately contradictory
    # output-time surface map; only original WRF's held fields are authoritative.
    state.theta += 5.0
    state.p_total += 1000.0
    writer.writer_diagnostics = {"d01": fields}
    nml = SimpleNamespace(**writer.bundles["d01"].namelist, sf_sfclay_physics=0)
    writer.bundles["d01"].namelist = nml
    # Lead-zero's existing loader additionally needs TSK (not held by this fix).
    with Dataset(tmp_path / "wrfinput_d01", "a") as ds:
        ds.createVariable("TSK", "f4", ("Time", "south_north", "west_east"))[0] = state.t_skin
    monkeypatch.delenv("GPUWRF_DYN_RK_FP32", raising=False)
    result = writer("d01", hour * 200, state)
    with Dataset(result["wrfout"]) as ds:
        for name in HELD_SURFACE_FIELDS:
            actual, expected = np.asarray(ds[name][0]), arrays[f"h{hour}_{name}"]
            assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes(), name


def test_shared_loader_and_merge_hold_only_disabled_surface_fields(tmp_path):
    arrays, initial = _loaded(tmp_path)
    changed = {name: np.full((4, 5), 123.0, np.float32) for name in HELD_SURFACE_FIELDS}
    changed["SWDOWN"] = np.full((4, 5), 456.0, np.float32)
    changed["TSK"] = np.full((4, 5), 301.0, np.float32)
    held = daily._merge_output_diagnostics(initial, changed, namelist=SimpleNamespace(sf_sfclay_physics=0))
    for name in HELD_SURFACE_FIELDS:
        np.testing.assert_array_equal(held[name], arrays[f"h24_{name}"])
        assert held[name] is initial[name]  # no per-frame array copy/device traffic
    assert held["SWDOWN"] is changed["SWDOWN"]
    assert held["TSK"] is changed["TSK"]
    normal = daily._merge_output_diagnostics(initial, changed, namelist=SimpleNamespace(sf_sfclay_physics=5))
    for name in HELD_SURFACE_FIELDS:
        assert normal[name] is changed[name]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", HELD_SURFACE_FIELDS)
def test_nonfinite_initialized_surface_is_rejected(tmp_path, field, bad):
    _loaded(tmp_path)
    with Dataset(tmp_path / "wrfinput_d01", "a") as ds:
        ds[field][0, 1, 1] = bad
    with pytest.raises(ValueError, match=field):
        load_disabled_surface_history(tmp_path / "wrfinput_d01", (4, 5))


def test_missing_th2_keeps_allocator_zero(tmp_path):
    _, fields = _loaded(tmp_path, missing_th2=True)
    np.testing.assert_array_equal(fields["TH2"], np.zeros((4, 5), np.float32))


def test_surface_shape_mismatch_rejected(tmp_path):
    _loaded(tmp_path)
    with pytest.raises(ValueError, match="PSFC"):
        load_disabled_surface_history(tmp_path / "wrfinput_d01", (5, 4))


@pytest.mark.parametrize("async_output", [False, True])
def test_daily_main_and_subhour_streams_hold_original_wrf_fields(tmp_path, monkeypatch, async_output):
    arrays, fields = _loaded(tmp_path)
    state, grid, values = synthetic_case()
    nml = SimpleNamespace(**values, sf_sfclay_physics=0)
    case = daily.DailyCase(
        state=state, grid=grid, namelist=nml,
        run_start=datetime(2023, 1, 15, tzinfo=timezone.utc), metadata={},
        writer_diagnostics=fields,
        writer_domain_authority=bind_wrfout_domain_authority("d01", authenticated_source_grid(grid, "d01"), grid),
    )
    monkeypatch.setattr(daily, "_surface_diagnostics_for_output", lambda *a, **k: {
        name: np.full((4, 5), 123.0, np.float32) for name in HELD_SURFACE_FIELDS})

    def advance(current, namelist, hours):
        current.psfc += 100.0
        current.t2 += 0.25
        current.theta += 0.25
        return current

    config = daily.DailyPipelineConfig(
        run_id="held-surface", domain="d01", hours=2, output_dir=tmp_path / "daily",
        proof_dir=tmp_path / "proof", score=False, refresh_land_state_hourly=False,
        async_output=async_output, auxhist=AuxhistStreamConfig(
            stream_id=1, interval_minutes=15, variables=HELD_SURFACE_FIELDS),
    )
    monkeypatch.delenv("GPUWRF_DYN_RK_FP32", raising=False)
    result = daily._run_forecast_sequence(
        config, output_dir=config.output_dir, forecast_fn=advance,
        case_builder=lambda _: (case, tmp_path),
    )
    paths = [*result.output_files, *result.auxhist_files]
    assert len(paths) == 10
    for path in paths:
        with Dataset(path) as actual:
            for name in HELD_SURFACE_FIELDS:
                assert np.asarray(actual[name][0]).tobytes() == arrays[f"h2_{name}"].tobytes(), (path.name, name)
