"""Solar E78: exact independent-source carry, clocks and legacy coexistence."""
from dataclasses import replace
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.diagnostics.solar import SolarCarry, SolarInterpolationState, SolarSurfaceFluxes
from gpuwrf.diagnostics.solar_config import CONFIG_KEYS, configuration, validate_options, validate_restart
from gpuwrf.io import restart as rp, wrfrst_netcdf as nc
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.validation.tier2 import make_ideal_grid
from test_online_diagnostics import populated as populated_nwp
from test_v0110_wrfrst_netcdf import _state


GROUP = "solar_interpolation"
SCHEMA = "GPUWRF_SOLAR_INTERPOLATION_SCHEMA_VERSION"
CONFIG = "GPUWRF_SOLAR_INTERPOLATION_CONFIG"


def _source_group():
    fixture = json.loads((Path(__file__).resolve().parents[1] / "proofs/u2_diag/solar_oracle.json").read_text())
    # An active pristine-WRF update/interpolation snapshot, not port output.
    snapshot = next(np.asarray(values, np.float32) for case in fixture["cases"]
                    for values in case["expected"] if np.any(np.asarray(values[9]) > 0))
    leaves = tuple(jnp.asarray(value.reshape(4, 4)) for value in snapshot)
    assert len(leaves) == 12
    return SolarCarry(SolarInterpolationState(*leaves[:7]), SolarSurfaceFluxes(*leaves[7:]))


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_SFC_WSPD", "0")
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re, "_MP_RE", False)
    grid = make_ideal_grid(5, 4, 4)  # (nz, ny, nx), matching the 16-cell oracle
    state = _state(grid)
    nl = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics,
                             swint_opt=1, use_noahmp=True, sf_surface_physics=4,
                             dt_s=18., radiation_interval_s=600., radiation_cadence_steps=33)
    carry = initial_operational_carry(state).replace(solar_interpolation=_source_group())
    return grid, nl, carry


def _write(path, fmt, grid, nl, carry):
    if fmt == "pickle":
        rp.write_restart(carry, nl, grid, 17, path)
    else:
        nc.write_wrfrst_carry(carry, grid, nl, path, valid_time="2026-06-03_00:05:06",
                             run_start="2026-06-03_00:00:00", step_index=17)


def _read(path, fmt, nl=None):
    return (rp.read_restart(path, namelist=nl) if fmt == "pickle"
            else nc.read_wrfrst_carry(path, namelist=nl))[0]


def _exact(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for before, after in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        before, after = np.asarray(before), np.asarray(after)
        assert after.dtype == before.dtype == np.dtype("float32")
        assert after.shape == before.shape == (4, 4)
        assert after.tobytes() == before.tobytes()


def test_normalized_bound_flat_and_raw_configuration(case):
    _, nl, _ = case
    cfg = configuration(nl)
    assert tuple(cfg) == CONFIG_KEYS
    assert cfg["sf_surface_physics"] == 4
    assert cfg == configuration(replace(nl, sf_surface_physics=None)) == configuration(cfg)
    raw = {"physics": {"swint_opt": [1], "ra_sw_physics": 4, "ra_lw_physics": 4,
                        "sf_surface_physics": 4, "radt": [10.]},
           "domains": {"time_step": 18}}
    assert configuration(raw) == cfg
    assert validate_options(nl) == cfg
    assert (cfg["dt_s"], cfg["radiation_interval_s"], cfg["radiation_cadence_steps"]) == (18., 600., 33)
    # Nominal radiation midpoint clock and discrete call cadence are distinct.
    assert cfg["radiation_interval_s"] != cfg["dt_s"] * cfg["radiation_cadence_steps"]
    fallback = validate_options(replace(nl, radiation_interval_s=0.))
    assert fallback["radiation_interval_s"] == 594.


@pytest.mark.parametrize("key,value", [
    ("swint_opt", 2), ("swint_opt", 1.5), ("ra_sw_physics", 1), ("ra_lw_physics", 0),
    ("sf_surface_physics", 2), ("solar_opt", 1), ("topo_shading", 1), ("slope_rad", 1),
    ("sf_urban_physics", 1), ("sf_lake_physics", 1), ("dt_s", 0.), ("dt_s", np.nan),
    ("dt_s", np.inf), ("radiation_interval_s", -1.), ("radiation_interval_s", np.inf),
    ("radiation_cadence_steps", 0), ("radiation_cadence_steps", 1.5),
])
def test_active_scheme_and_clock_refusal(case, key, value):
    _, nl, _ = case
    cfg = {**configuration(nl), key: value}
    with pytest.raises(ValueError, match="Solar"):
        validate_options(cfg)


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_exact_solar_restart_from_original_operator(case, tmp_path, fmt):
    grid, nl, carry = case
    path = tmp_path / fmt
    _write(path, fmt, grid, nl, carry)
    _exact(carry.solar_interpolation, _read(path, fmt, nl).solar_interpolation)
    _exact(carry.solar_interpolation, _read(path, fmt).solar_interpolation)
    if fmt == "pickle":
        schema = pickle.loads(path.read_bytes())["carry"]
        assert schema["solar_interpolation_schema_version"] == 1
        assert schema["solar_interpolation_config"] == configuration(nl)
        assert schema["online_diagnostics_schema_version"] == 1
        assert "lake_schema_version" in schema
    else:
        with Dataset(path) as ds:
            assert ds.getncattr(SCHEMA) == 1
            assert json.loads(ds.getncattr(CONFIG)) == configuration(nl)
            assert ds.GPUWRF_ONLINE_DIAGNOSTICS_SCHEMA_VERSION == 1


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
@pytest.mark.parametrize("damage", ["missing", "precision", "shape"])
def test_write_refuses_damaged_solar_carry(case, tmp_path, fmt, damage):
    grid, nl, carry = case
    group = carry.solar_interpolation
    value = group.parameters.coszen_ref
    bad = None if damage == "missing" else value.astype(jnp.float64) if damage == "precision" else value[:, :-1]
    group = group._replace(parameters=group.parameters._replace(coszen_ref=bad))
    with pytest.raises(ValueError, match="Solar.*E78"):
        _write(tmp_path / fmt, fmt, grid, nl, carry.replace(solar_interpolation=group))


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
@pytest.mark.parametrize("damage", ["schema", "version", "config", "config_key", "group", "leaf"])
def test_read_refuses_missing_active_solar_schema(case, tmp_path, fmt, damage):
    grid, nl, carry = case
    path = tmp_path / fmt
    _write(path, fmt, grid, nl, carry)
    if fmt == "pickle":
        payload = pickle.loads(path.read_bytes())
        schema = payload["carry"]
        if damage == "schema":
            schema.pop("solar_interpolation_schema_version")
        elif damage == "version":
            schema["solar_interpolation_schema_version"] = 2
        elif damage == "config":
            schema.pop("solar_interpolation_config")
        elif damage == "config_key":
            schema["solar_interpolation_config"].pop("dt_s")
        elif damage == "group":
            schema["extra_fields"].pop(GROUP)
            schema["carry_field_order"].remove(GROUP)
        else:
            group = schema["extra_fields"][GROUP]
            schema["extra_fields"][GROUP] = group._replace(fluxes=group.fluxes._replace(swddni=None))
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path, "a") as ds:
            if damage == "schema":
                ds.delncattr(SCHEMA)
            elif damage == "version":
                ds.setncattr(SCHEMA, 2)
            elif damage == "config":
                ds.delncattr(CONFIG)
            elif damage == "config_key":
                cfg = json.loads(ds.getncattr(CONFIG))
                cfg.pop("dt_s")
                ds.setncattr(CONFIG, json.dumps(cfg))
            elif damage == "group":
                for name in ("GPUWRF_OPTIONAL_CARRY_FIELD_ORDER", "GPUWRF_OPTIONAL_CARRY_KIND"):
                    values = json.loads(ds.getncattr(name))
                    values.pop(GROUP)
                    ds.setncattr(name, json.dumps(values))
            else:
                ds.renameVariable("GPUWRF_SOLAR_INTERPOLATION_LEAF_009", "REMOVED_SOLAR_DNI")
    with pytest.raises(ValueError, match="Solar.*E78|missing exact optional carry variables"):
        _read(path, fmt, nl)


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
@pytest.mark.parametrize("changes", [
    {"dt_s": 20.}, {"radiation_interval_s": 660.}, {"radiation_cadence_steps": 34},
    {"swint_opt": 0}, {"ra_sw_physics": 1},
])
def test_resume_selection_and_independent_clocks_must_match(case, tmp_path, fmt, changes):
    grid, nl, carry = case
    path = tmp_path / fmt
    _write(path, fmt, grid, nl, carry)
    with pytest.raises(ValueError, match="Solar.*E78"):
        _read(path, fmt, replace(nl, **changes))


def _remove_solar_markers(path, fmt):
    if fmt == "pickle":
        payload = pickle.loads(path.read_bytes())
        schema = payload["carry"]
        schema.pop("solar_interpolation_schema_version")
        schema.pop("solar_interpolation_config")
        schema["extra_fields"].pop(GROUP)
        schema["carry_field_order"].remove(GROUP)
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path, "a") as ds:
            ds.delncattr(SCHEMA)
            ds.delncattr(CONFIG)
            for attr in ("GPUWRF_OPTIONAL_CARRY_FIELD_ORDER", "GPUWRF_OPTIONAL_CARRY_KIND"):
                values = json.loads(ds.getncattr(attr))
                values.pop(GROUP)
                ds.setncattr(attr, json.dumps(values))


@pytest.mark.parametrize("fmt", ["pickle", "netcdf"])
def test_inactive_old_solar_checkpoint_keeps_nwp_carry_and_its_guard(case, tmp_path, fmt):
    grid, nl, carry = case
    nwp, radar = populated_nwp(carry.state)
    nl = replace(nl, swint_opt=0, nwp_diagnostics=1, do_radar_ref=1)
    carry = carry.replace(solar_interpolation=None, nwp_diagnostics=nwp, radar_reflectivity=radar)
    path = tmp_path / fmt
    _write(path, fmt, grid, nl, carry)
    _remove_solar_markers(path, fmt)
    restored = _read(path, fmt, nl)
    assert restored.solar_interpolation is None
    for a, b in zip(jax.tree.leaves(nwp), jax.tree.leaves(restored.nwp_diagnostics), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(restored.radar_reflectivity, radar)
    with pytest.raises(ValueError, match="Solar.*E78"):
        _read(path, fmt, replace(nl, swint_opt=1))
    # Removing only the unrelated NWP marker must still trigger its old E78
    # guard; Solar legacy compatibility cannot weaken the earlier schema.
    if fmt == "pickle":
        payload = pickle.loads(path.read_bytes())
        payload["carry"].pop("online_diagnostics_schema_version")
        path.write_bytes(pickle.dumps(payload))
    else:
        with Dataset(path, "a") as ds:
            ds.delncattr("GPUWRF_ONLINE_DIAGNOSTICS_SCHEMA_VERSION")
    with pytest.raises(ValueError, match="online diagnostics schema.*E78"):
        _read(path, fmt, nl)


def test_saved_config_cannot_disagree_with_pickle_namelist(case, tmp_path):
    grid, nl, carry = case
    path = tmp_path / "pickle"
    _write(path, "pickle", grid, nl, carry)
    payload = pickle.loads(path.read_bytes())
    payload["namelist"] = replace(payload["namelist"], radiation_interval_s=601.)
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="Solar.*cadence/selection differs.*E78"):
        rp.read_restart(path)


def test_config_only_guard_handles_malformed_records_and_no_reseed(case):
    _, nl, carry = case
    cfg = configuration(nl)
    for saved in ([cfg], {**cfg, "dt_s": None}, {**cfg, "unexpected": 1}):
        with pytest.raises(ValueError, match="Solar.*E78"):
            validate_restart(carry.solar_interpolation, saved, 1, (4, 4), requested=nl)
    with pytest.raises(ValueError, match="Solar.*E78"):
        validate_restart(None, None, None, (4, 4), requested=nl)
    validate_restart(None, None, None, (4, 4), requested=replace(nl, swint_opt=0))
