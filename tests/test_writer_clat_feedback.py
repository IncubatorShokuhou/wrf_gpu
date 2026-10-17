"""The actual nested output routes retain computational latitude after feedback."""
from datetime import datetime, timezone
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.integration import daily_pipeline as daily, nested_pipeline as nested
from gpuwrf.io import gen2_accessor, wrf_history_metadata, wrfout_writer as output
from test_m7_netcdf_writer import synthetic_case, authenticated_source_grid, writer_authority
from test_initial_history_wrf_t0 import _wrfinput


def make_writer(tmp_path, monkeypatch, mode, pipeline):
    state, grid, nl = synthetic_case()
    shape = (grid.ny, grid.nx)
    xlat = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1024 + 28
    clat = np.nextafter(xlat, np.float32(90))
    static = {"XLAT": xlat, "XLONG": np.full(shape, -16, np.float32)}
    source = authenticated_source_grid(grid, "d01")
    monkeypatch.setattr(gen2_accessor, "Gen2Run", lambda _: SimpleNamespace(
        namelist={}, grid=lambda _: source))
    monkeypatch.setattr(daily, "_load_static_latlon_writer_diagnostics", lambda *a, **kw: (static, {}))
    monkeypatch.setattr(wrf_history_metadata, "wrf_history_global_attributes", lambda *a, **kw: {})
    nl = SimpleNamespace(use_noahmp=False, sf_sfclay_physics=5)
    # Real constructor must merge the producer override into static diagnostics.
    writer = nested._PerDomainWrfoutWriter(
        output_dir=tmp_path / "out", input_dir=tmp_path,
        run_start=datetime(2026, 2, 28, tzinfo=timezone.utc),
        bundles={"d01": SimpleNamespace(grid=grid, namelist=nl,
                  writer_static_overrides={"CLAT": clat})},
        output_cadence_steps={"d01": 67}, dt_by_domain={"d01": 54.})
    writer._surface_diagnostics_for_output = lambda *a, **kw: {}
    writer._full_variable_set = mode == "full_compatibility"
    writer._variable_subset = ("CLAT", "XLAT", "XLONG") if mode == "selected" else None
    _wrfinput(tmp_path / "wrfinput_d01", grid.ny, grid.nx)
    snapshots = []
    if pipeline:
        writer._output_pipeline = SimpleNamespace(submit=snapshots.append)
    assert writer.writer_diagnostics["d01"]["CLAT"] is clat
    return writer, state, clat, xlat, snapshots


@pytest.mark.parametrize("mode", ["full", "full_compatibility", "selected"])
@pytest.mark.parametrize("pipeline", [False, True], ids=["sync", "materialize"])
@pytest.mark.parametrize("step", [0, 67, 1600], ids=["t0", "h1_18s", "h24"])
def test_actual_nested_writer_retains_feedback_clat(tmp_path, monkeypatch, mode, pipeline, step):
    writer, state, expected, xlat, snapshots = make_writer(tmp_path, monkeypatch, mode, pipeline)
    before = {name: value.tobytes() for name, value in vars(state).items() if isinstance(value, np.ndarray)}
    result = writer("d01", step, state)
    if pipeline:
        assert len(snapshots) == 1
        snapshot = snapshots[0]
        snapshot.writer._materialize_and_submit(
            name=snapshot.name, own_step=snapshot.own_step, carry=snapshot.carry,
            valid_time=snapshot.valid_time, lead_seconds=snapshot.lead_seconds,
            lead_hours=snapshot.lead_hours, path=snapshot.path)
    with Dataset(result["wrfout"]) as nc:
        assert nc["CLAT"].dtype == np.dtype(np.float32)
        assert np.asarray(nc["CLAT"][0]).tobytes() == expected.tobytes()
        assert np.asarray(nc["XLAT"][0]).tobytes() == xlat.tobytes()
    assert before == {name: value.tobytes() for name, value in vars(state).items() if isinstance(value, np.ndarray)}


@pytest.mark.parametrize("mode", ["full", "full_compatibility"])
def test_absent_clat_preserves_legacy_fallback(tmp_path, monkeypatch, mode):
    writer, state, _, xlat, _ = make_writer(tmp_path, monkeypatch, mode, False)
    del writer.writer_diagnostics["d01"]["CLAT"]
    result = writer("d01", 67, state)
    with Dataset(result["wrfout"]) as nc:
        assert np.asarray(nc["CLAT"][0]).tobytes() == xlat.tobytes()


@pytest.mark.parametrize("alias", ["CLAT", "clat"])
def test_state_producer_and_selected_batch(tmp_path, monkeypatch, alias):
    state, grid, nl = synthetic_case()
    held = jnp.full(state.psfc.shape, np.float32(28.123456))
    setattr(state, alias, held)
    calls = []
    device_get = jax.device_get
    def capture(values):
        calls.append(tuple(id(value) for value in values))
        return device_get(values)
    monkeypatch.setattr(jax, "device_get", capture)
    payload = output.prepare_wrfout_payload(state, grid, nl, tmp_path / "state.nc",
        domain="d01", domain_authority=writer_authority(grid, "d01"),
        run_start=datetime(2026, 2, 28), valid_time=datetime(2026, 2, 28, 1),
        lead_hours=1, variable_subset=("CLAT",))
    assert payload.fields["CLAT"].tobytes() == np.asarray(held).tobytes()
    assert len(calls) == 1 and id(held) in calls[0]
