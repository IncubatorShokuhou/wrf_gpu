"""CPU-WRF inventories, metadata and real samples qualify history transcription."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.microphysics_history import SCHEME_FIELDS, UNPROVIDED_FIELDS
from gpuwrf.io.wrfout_writer import (
    FULL_WRFOUT_VARIABLES, prepare_wrfout_payload, write_prepared_wrfout,
)
from test_m7_netcdf_writer import synthetic_case, writer_authority

PROOF = Path(__file__).resolve().parents[1] / "proofs/t_mpout"
SCHEMA = json.loads((PROOF / "cpu_history_schema.json").read_text())
# Independently audited adapter -> WRF caller names; do not use the writer's map
# to construct its own expected values (a swapped alias must fail this gate).
LEAVES = {
    "QNCLOUD": "Nc", "QNCCN": "Nn", "QNSNOW": "Ns", "QNGRAUPEL": "Ng",
    "QHAIL": "qh", "QNHAIL": "Nh", "QVGRAUPEL": "qvolg", "QVHAIL": "qvolh",
    "QNDROP": "Nc", "QNWFA": "nwfa", "QNIFA": "nifa",
    "QNWFA2D": "nwfa2d", "QNIFA2D": "nifa2d",
}
START = datetime(2023, 1, 15, tzinfo=timezone.utc)


def sources(mp):
    state, grid, nl = synthetic_case()
    nl["mp_physics"] = mp
    row = SCHEMA[f"mp{mp}"]
    with np.load(PROOF / "cpu_history_samples.npz") as samples:
        expected = {name: samples[f"mp{mp}__{name}"].copy() for name in row["missing_schema"]}
    for name, value in expected.items():
        if name in LEAVES:
            setattr(state, LEAVES[name], value)
    return state, grid, nl, expected


def prepare(tmp_path, mp, *, subset=None, full=True, diagnostics=None):
    state, grid, nl, expected = sources(mp)
    authority = writer_authority(grid, "d01")
    payload = prepare_wrfout_payload(
        state, grid, nl, tmp_path / "history.nc", domain="d01",
        domain_authority=authority, valid_time=START, lead_hours=24,
        run_start=START, full_variable_set=full, variable_subset=subset,
        diagnostics=diagnostics,
    )
    return payload, authority, expected


@pytest.mark.parametrize("mp", (40, 24, 28, 16, 26, 18))
@pytest.mark.parametrize("mode", ("full", "operational", "subset"))
def test_cpu_inventory_values_and_metadata(tmp_path, mp, mode):
    row = SCHEMA[f"mp{mp}"]
    subset = tuple(row["missing_schema"]) if mode == "subset" else None
    payload, authority, expected = prepare(tmp_path, mp, subset=subset, full=mode != "operational")
    write_prepared_wrfout(payload, expected_domain="d01", expected_domain_authority=authority,
                         variable_subset=subset)
    with Dataset(payload.target) as ds:
        provided = set(row["missing_schema"]) - set(UNPROVIDED_FIELDS.get(mp, ()))
        assert set(SCHEME_FIELDS[mp]) == set(row["missing_schema"])
        assert set(row["missing_schema"]) - set(ds.variables) == set(UNPROVIDED_FIELDS.get(mp, ()))
        if mode == "full":
            assert set(row["inventory"]) - set(ds.variables) == set(UNPROVIDED_FIELDS.get(mp, ()))
        for name in provided:
            var = ds[name]
            np.testing.assert_array_equal(var[0], expected[name], err_msg=name)
            assert var.dimensions == tuple(row["missing_schema"][name]["dimensions"])
            assert var.dtype == np.dtype("float32")
            for key, value in row["missing_schema"][name]["attrs"].items():
                assert var.getncattr(key) == value, (name, key)
        if mp == 18 and mode == "full":
            assert "QNCLOUD" not in ds.variables  # NSSL's caller names Nc QNDROP.


@pytest.mark.parametrize("mp", (18,))
def test_explicit_held_diagnostics_are_written_without_invented_defaults(tmp_path, mp):
    _, _, _, expected = sources(mp)
    diagnostics = {name: expected[name] + np.float32(i + 1)
                   for i, name in enumerate(UNPROVIDED_FIELDS[mp])}
    payload, authority, _ = prepare(tmp_path, mp, diagnostics=diagnostics)
    write_prepared_wrfout(payload, expected_domain="d01", expected_domain_authority=authority)
    with Dataset(payload.target) as ds:
        for name, value in diagnostics.items():
            np.testing.assert_array_equal(ds[name][0], value, err_msg=name)
            for key, reference in SCHEMA[f"mp{mp}"]["missing_schema"][name]["attrs"].items():
                assert ds[name].getncattr(key) == reference, (name, key)


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("mp,name", ((40, "QNSNOW"), (28, "QNWFA"), (18, "QNDROP")))
def test_nonfinite_new_sources_are_rejected(tmp_path, bad, mp, name):
    _, _, _, expected = sources(mp)
    value = expected[name].copy()
    value.flat[0] = bad
    with pytest.raises(ValueError, match="nonfinite microphysics history field " + name):
        prepare(tmp_path, mp, diagnostics={name: value})


def test_subset_materializes_selected_state_alias_once(tmp_path, monkeypatch):
    import jax
    import gpuwrf.io.wrfout_writer as writer
    state, grid, nl, expected = sources(18)
    state.Nc = jax.numpy.asarray(state.Nc)
    state.Nh = jax.numpy.full(state.Nc.shape, np.nan)
    real_get = jax.device_get
    calls = []

    def get(values):
        calls.append(tuple(values))
        return real_get(values)

    monkeypatch.setattr(jax, "device_get", get)
    authority = writer_authority(grid, "d01")
    payload = writer.prepare_wrfout_payload(state, grid, nl, tmp_path / "subset.nc",
        domain="d01", domain_authority=authority, valid_time=START, run_start=START,
        lead_hours=24, full_variable_set=True, variable_subset=("QNDROP",))
    assert len(calls) == 1 and len(calls[0]) == 1 and calls[0][0] is state.Nc
    np.testing.assert_array_equal(payload.fields["QNDROP"], expected["QNDROP"])
    assert set(payload.fields) == {"QNDROP"}


def test_missing_optional_source_is_not_fabricated(tmp_path):
    state, grid, nl = synthetic_case()
    nl["mp_physics"] = 18
    state.Nc = None
    authority = writer_authority(grid, "d01")
    payload = prepare_wrfout_payload(state, grid, nl, tmp_path / "missing.nc",
        domain="d01", domain_authority=authority, valid_time=START, run_start=START,
        lead_hours=24, full_variable_set=True)
    assert "QNDROP" not in payload.fields


@pytest.mark.parametrize("mp", (40, 24, 28, 16, 26, 18))
def test_production_output_callback_preserves_species(tmp_path, mp):
    from test_initial_history_wrf_t0 import _writer
    writer, _ = _writer(tmp_path, lambda *args, **kwargs: {})
    state, _, nl, expected = sources(mp)
    writer.bundles["d01"].namelist = nl
    writer._full_variable_set = True
    result = writer("d01", 200, SimpleNamespace(state=state))
    with Dataset(result["wrfout"]) as ds:
        for name in set(expected) - set(UNPROVIDED_FIELDS.get(mp, ())):
            np.testing.assert_array_equal(ds[name][0], expected[name], err_msg=name)


def test_default_full_inventory_is_unchanged(tmp_path):
    # Resolve the control before preparing: mp8 adds no scheme-specific slots.
    state, grid, nl = synthetic_case()
    authority = writer_authority(grid, "d01")
    nl["mp_physics"] = 8
    payload = prepare_wrfout_payload(state, grid, nl, tmp_path / "control.nc",
        domain="d01", domain_authority=authority, valid_time=START, run_start=START,
        lead_hours=24, full_variable_set=True)
    write_prepared_wrfout(payload, expected_domain="d01", expected_domain_authority=authority)
    with Dataset(payload.target) as ds:
        assert tuple(ds.variables) == FULL_WRFOUT_VARIABLES
