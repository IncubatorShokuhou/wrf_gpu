"""Original-WRF Solar values/metadata through public and nested history seams.

Flux values come from the pristine interpolation oracle. The metadata fixture
must be frozen from the private original-CPU-WRF Solar history, including the
iofields-enabled SWDDIR/GSW; Registry descriptions alone are not a final gate.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset
import pytest

from gpuwrf.diagnostics.solar import history_fields, seed_solar
from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
from test_initial_history_wrf_t0 import _writer, _wrfinput
from test_m7_netcdf_writer import writer_authority
from test_solar_restart import case  # Reuse the exact schema-state/original-Solar fixture.


PROOFS = Path(__file__).resolve().parents[1] / "proofs/u2_diag"
NAMES = ("SWDOWN", "SWDDIR", "SWDDNI", "SWDDIF", "GSW")
NEW_NAMES = NAMES[1:]
ATTRS = ("FieldType", "MemoryOrder", "description", "units", "stagger", "coordinates")
# Genuine product mutation target for the root's later immutable-snapshot arm:
# delete only the solar_interpolation overlay in
# _PerDomainWrfoutWriter._materialize_and_submit, then run the nonzero node:
# test_actual_nested_solar_overlay_preserves_profile_flux[200]. Its old M9
# SWDOWN conflicts with every Solar value, and cannot pass as a dry sample.
OVERLAY_MUTANT_NODE = "tests/test_solar_history.py::test_actual_nested_solar_overlay_preserves_profile_flux[200]"


def _cpu_metadata():
    path = PROOFS / "solar_history_metadata.json"
    payload = json.loads(path.read_text())
    assert payload["reference_file"] and set(payload["fields"]) == set(NAMES)
    assert payload.get("source_kind") == "original CPU-WRF history header"
    assert payload.get("values_read") is False
    digest = hashlib.sha256(json.dumps(payload["fields"], sort_keys=True).encode()).hexdigest()
    assert digest == payload["metadata_sha256"] == "a86d343c36148944c541b21afad40a62673a2b3eb494b764941010f3ed424789"
    for attrs in payload["fields"].values():
        assert set(attrs) == set(ATTRS)
    return payload["fields"]


def _check_schema(ds, expected):
    metadata = _cpu_metadata()
    for name in NAMES:
        variable = ds[name]
        assert variable.dimensions == ("Time", "south_north", "west_east")
        assert np.dtype(variable.dtype) == np.dtype("float32")
        for attr in ATTRS:
            assert variable.getncattr(attr) == metadata[name][attr], (name, attr)
        actual = np.asarray(variable[0])
        value = np.asarray(expected[name], np.float32)
        assert actual.shape == value.shape == (4, 4)
        assert actual.tobytes() == value.tobytes(), name


def _public_write(path, grid, nl, carry, *, subset=None, full=False, diagnostics=None):
    return write_wrfout_netcdf(carry.state, grid, nl, path, domain="d01",
        domain_authority=writer_authority(grid, "d01"), lead_hours=1.,
        valid_time="2026-06-03_01:00:00", run_start="2026-06-03_00:00:00",
        diagnostics=history_fields(carry.solar_interpolation) if diagnostics is None else diagnostics,
        variable_subset=subset, full_variable_set=full)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("subset", [None, NAMES])
def test_public_full_and_subset_write_original_solar_values_metadata(case, tmp_path, full, subset):
    grid, nl, carry = case
    path = _public_write(tmp_path / "wrfout", grid, nl, carry, subset=subset, full=full)
    with Dataset(path) as ds:
        _check_schema(ds, history_fields(carry.solar_interpolation))


def _nested_writer(tmp_path, grid, nl, resolve):
    writer, _ = _writer(tmp_path, resolve)
    writer.bundles["d01"] = SimpleNamespace(namelist=nl, grid=grid)
    writer.domain_authorities["d01"] = writer_authority(grid, "d01")
    _wrfinput(tmp_path / "wrfinput_d01", grid.ny, grid.nx)
    return writer


@pytest.mark.parametrize("own_step", [0, 200])
def test_actual_nested_solar_overlay_preserves_profile_flux(case, tmp_path, own_step):
    grid, nl, carry = case
    shape = carry.state.t_skin.shape
    group = seed_solar(carry.state, xp=np) if own_step == 0 else carry.solar_interpolation
    expected = history_fields(group)
    held_profile = (np.zeros(shape, np.float32) if own_step == 0
                    else np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + np.float32(77.))
    m9_down = np.full(shape, np.float32(987654.25), np.float32)
    if own_step:
        assert not np.array_equal(np.asarray(expected["SWDOWN"]), held_profile)
        assert not np.array_equal(np.asarray(expected["SWDOWN"]), m9_down)
    def resolve(*args, **kwargs):
        assert own_step != 0, "WRF pre-solve history must not trigger an M9 resolve"
        return {"SWDOWN": m9_down, "SWDNB": held_profile}
    writer = _nested_writer(tmp_path, grid, nl, resolve)
    sample = SimpleNamespace(state=carry.state, solar_interpolation=group,
                             radiation_diagnostics=SimpleNamespace(swdown=held_profile))
    result = writer("d01", own_step, sample)
    with Dataset(result["wrfout"]) as ds:
        _check_schema(ds, expected)
        np.testing.assert_array_equal(ds["SWDNB"][0], held_profile)
    # Output cannot replace or mutate the held profile with interpolated SWDOWN.
    np.testing.assert_array_equal(sample.radiation_diagnostics.swdown, held_profile)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("subset", [None, ("T2",)])
def test_actual_callback_rejects_every_bad_solar_leaf_even_unrequested(case, tmp_path, name, bad, subset):
    grid, nl, carry = case
    obj = carry.solar_interpolation
    bad_flux = np.asarray(getattr(obj.fluxes, name.lower())).copy()
    bad_flux[1, 2] = np.float32(bad)
    damaged = obj._replace(fluxes=obj.fluxes._replace(**{name.lower(): bad_flux}))
    writer = _nested_writer(tmp_path, grid, nl, lambda *a, **kwargs: {})
    writer._variable_subset = subset
    with pytest.raises(ValueError, match="nonfinite|non-finite"):
        writer("d01", 200, SimpleNamespace(state=carry.state, solar_interpolation=damaged))
    assert not list(writer.output_dir.glob("wrfout*"))


@pytest.mark.parametrize("full", [False, True])
def test_solar_off_adds_no_optional_sun_fields(case, tmp_path, full):
    grid, nl, carry = case
    nl = replace(nl, swint_opt=0)
    carry = carry.replace(solar_interpolation=None)
    baseline_down = np.full(carry.state.t_skin.shape, 510.25, np.float32)
    held_profile = np.full(carry.state.t_skin.shape, 600.75, np.float32)
    writer = _nested_writer(tmp_path, grid, nl,
                            lambda *a, **kwargs: {"SWDOWN": baseline_down, "SWDNB": held_profile})
    writer._full_variable_set = full
    output = writer("d01", 200, SimpleNamespace(state=carry.state, solar_interpolation=None))
    with Dataset(output["wrfout"]) as ds:
        assert not any(name in ds.variables for name in NEW_NAMES)
        np.testing.assert_array_equal(ds["SWDOWN"][0], baseline_down)
        np.testing.assert_array_equal(ds["SWDNB"][0], held_profile)
