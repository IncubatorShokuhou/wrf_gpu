"""E78 grouped physics must not bypass actual checkpoint admission/restoration."""
from collections import UserDict
from datetime import datetime
from types import SimpleNamespace
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.io.restart_schema import validate_held_state
from gpuwrf.nesting.scalar_species import required_extra_boundary_records
from gpuwrf.runtime.checkpoint import read_checkpoint, write_checkpoint
from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state


def grouped_namelist():
    return {"physics": {"mp_physics": 24, "use_mp_re": 0},
            "boundary_config": {"nested_frozen_wrf_boundary_bundle": True}}


@pytest.mark.parametrize("namelist,expected", [
    ({"mp_physics": 24, "boundary_config": {"nested_frozen_wrf_boundary_bundle": True}}, ("qh_bdy",)),
    ({**grouped_namelist(), "mp_physics": 8}, ("qh_bdy",)),
    (UserDict({"physics": UserDict({"mp_physics": 24}),
               "boundary_config": UserDict({"nested_frozen_wrf_boundary_bundle": True})}), ("qh_bdy",)),
    (SimpleNamespace(mp_physics=24, boundary_config=SimpleNamespace(nested_frozen_wrf_boundary_bundle=True)), ("qh_bdy",)),
    ({**grouped_namelist(), "boundary_config": {"nested_frozen_wrf_boundary_bundle": False}}, ()),
    ({**grouped_namelist(), "boundary_config": None}, ()),
    ({"mp_physics": 8, "boundary_config": {"nested_frozen_wrf_boundary_bundle": True}}, ()),
])
def test_schema_selection_matches_physics_group_and_preserves_child_predicate(namelist, expected):
    assert required_extra_boundary_records(namelist) == expected
    fields = {name: object() for name in ("sfc_wspd", "re_cloud", "re_ice", "re_snow", *expected)}
    validate_held_state(fields, namelist)
    if expected:
        fields.pop("qh_bdy")
        with pytest.raises(ValueError, match="extra scalar boundary.*qh_bdy"):
            validate_held_state(fields, namelist)


@pytest.fixture
def checkpoint_fields():
    grid = build_flat_grid(nx=8, ny=8, nz=3, dx_m=3000.)
    state = build_neutral_state(grid).ensure_conditional_leaves(mp_physics=24)
    # This regression does not depend on the optional surface diagnostic flag.
    state = state.replace(sfc_wspd=jnp.zeros_like(state.mu_total, dtype=jnp.float32))
    assert state.qh is not None and state.qh_bdy is None
    return grid, state


def assert_missing_hail_refused(operation):
    try:
        operation()
    except ValueError as exc:
        assert "extra scalar boundary" in str(exc) and "qh_bdy" in str(exc)
    else:
        raise AssertionError("grouped mp24 live-child restart admitted missing qh_bdy")


def test_grouped_checkpoint_writer_refuses_missing_hail_record(checkpoint_fields, tmp_path):
    grid, state = checkpoint_fields
    path = tmp_path / "invalid.pkl"
    assert_missing_hail_refused(lambda: write_checkpoint(state, grouped_namelist(), grid, 1, path))
    assert not path.exists()


def test_grouped_checkpoint_reader_refuses_missing_hail_record(checkpoint_fields, tmp_path):
    grid, state = checkpoint_fields
    record = jnp.arange(2 * state.qv_bdy[0].size, dtype=jnp.float32).reshape(2, *state.qv_bdy.shape[1:])
    state = state.replace(qh_bdy=record)
    path = tmp_path / "restart.pkl"
    write_checkpoint(state, grouped_namelist(), grid, 1, path)
    restored, *_ = read_checkpoint(path)
    np.testing.assert_array_equal(restored.qh_bdy, record)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    # An existing bad payload can retain the field name with an absent value.
    payload["state_fields"]["qh_bdy"] = None
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    assert_missing_hail_refused(lambda: read_checkpoint(path))


def test_grouped_netcdf_writer_refuses_missing_hail_record(checkpoint_fields, tmp_path):
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state
    grid, state = checkpoint_fields
    path = tmp_path / "invalid.nc"
    assert_missing_hail_refused(lambda: write_wrfrst_state(
        state=state, grid=grid, namelist=grouped_namelist(), path=path,
        valid_time=datetime(2026, 1, 1), run_start=datetime(2026, 1, 1), step_index=1))
    assert not path.exists()
