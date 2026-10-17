"""Spectral namelist/auxinput10/RK/restart admission and source ownership."""
from dataclasses import replace
from datetime import datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

from gpuwrf.assimilation.fdda_spectral import (
    SpectralFddaOptions, SpectralFddaInputs, SpectralFddaCarry, initial_spectral_carry,
)
from gpuwrf.coupling import fdda_rk
from gpuwrf.io.fdda_namelist import fdda_namelist_errors, grid_fdda_options
from gpuwrf.io.namelist_binding import require_namelist_honoured
from gpuwrf.io.restart_schema import validate_spectral_fdda
from gpuwrf.io.wrffdda import load_grid_fdda

jax.config.update("jax_enable_x64", True)


def legacy_helpers():
    path = Path(__file__).with_name("test_o2_fdda_namelist_io.py")
    spec = importlib.util.spec_from_file_location("spectral_grid_io_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(nz=3, ny=4, nx=5):
    shape = (1, nz, ny, nx)
    vals = {}
    for name, s in (("u", (1, nz, ny, nx+1)), ("v", (1, nz, ny+1, nx)), ("t", shape),
                    ("q", shape), ("ph", (1, nz+1, ny, nx))):
        for suff in ("old", "new"):
            vals[f"{name}_{suff}"] = jnp.ones(s, jnp.float32)
    return SpectralFddaInputs(**vals, options=SpectralFddaOptions(interval_m=60), record_steps=(0,))


def test_binding_registry_defaults_and_prejax_honour():
    h = legacy_helpers()
    nl = h.nl(grid_fdda=[2, 2], guv=[3e-4, 1e-4], gfdda_interval_m=[60, 30],
              xwavenum=[2, 3], ywavenum=[4, 5], dk_zfac_t=[6, 7], ktrop=30, gph=[2e-4, 4e-4])
    require_namelist_honoured(nl, ("d01", "d02"))
    a, b = (grid_fdda_options(nl, d, dt) for d, dt in (("d01", 18.), ("d02", 6.)))
    assert (a["xwavenum"], b["xwavenum"]) == (2, 3)
    assert (a["gph"], b["gph"]) == (2e-4, 4e-4)
    assert a["ktrop"] == b["ktrop"] == 30  # Registry scalar, not max_domains
    assert b["interval_m"] == 30 and b["dk_zfac_t"] == 7
    defaults = grid_fdda_options(h.nl(grid_fdda=[2]), "d01", 18.)
    assert defaults["dk_zfac_uv"] == defaults["dk_zfac_ph"] == 1
    assert defaults["xwavenum"] == defaults["ywavenum"] == defaults["gph"] == 0
    assert not fdda_namelist_errors(h.nl(grid_fdda=[2], ktrop=2), ("d01",))  # low KTROP is warning in WRF
    assert fdda_namelist_errors(h.nl(grid_fdda=[2], if_zfac_ph=[1], dk_zfac_ph=[0]), ("d01",))
    assert fdda_namelist_errors(h.nl(grid_fdda=[2], gph=[float("nan")]), ("d01",))


def test_reader_requires_and_loads_staggered_perturbation_geopotential(tmp_path):
    h = legacy_helpers()
    start = datetime(2023, 1, 15)
    path = tmp_path / "wrffdda_d01"
    h._write_wrffdda(path, [start])
    nl = h.nl(grid_fdda=[2])
    kw = dict(run_start=start, dt_s=18., nz=3, ny=4, nx=5, run_seconds=60.)
    with pytest.raises(ValueError, match="PH_NDG_OLD"):
        load_grid_fdda(tmp_path, nl, "d01", **kw)
    with netCDF4.Dataset(path, "a") as ds:
        ds.createDimension("bottom_top_stag", 4)
        for suffix, value in (("OLD", 17.), ("NEW", 23.)):
            var = ds.createVariable(f"PH_NDG_{suffix}", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))
            var[:] = value
    result = load_grid_fdda(tmp_path, nl, "d01", **kw)
    assert isinstance(result, SpectralFddaInputs)
    assert result.ph_old.shape == (1, 4, 4, 5) and result.ph_old.dtype == jnp.float32
    np.testing.assert_array_equal(result.ph_old, 17.)
    np.testing.assert_array_equal(result.ph_new, 23.)
    assert result.record_steps == (0,)


def test_reader_accepts_wrf_packed_mass_dimensions(tmp_path):
    """Pristine WRF packed fdda3d serializes U/V/PH on the mass footprint."""
    h = legacy_helpers()
    start = datetime(2023, 1, 15)
    path = tmp_path / "wrffdda_d01"
    with netCDF4.Dataset(path, "w") as ds:
        for n, size in (("Time", None), ("DateStrLen", 19), ("bottom_top", 3),
                        ("south_north", 4), ("west_east", 5)):
            ds.createDimension(n, size)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0] = np.frombuffer(b"2023-01-15_00:00:00", dtype="S1")
        for field in ("U", "V", "T", "Q", "PH"):
            for suffix in ("OLD", "NEW"):
                v = ds.createVariable(f"{field}_NDG_{suffix}", "f4", ("Time", "bottom_top", "south_north", "west_east"))
                v[0] = 17 if suffix == "OLD" else 23
    got = load_grid_fdda(tmp_path, h.nl(grid_fdda=[2]), "d01", run_start=start,
                         dt_s=18., nz=3, ny=4, nx=5, run_seconds=60.)
    assert got.u_old.shape == (1, 3, 4, 6)
    assert got.v_old.shape == (1, 3, 5, 5)
    assert got.ph_old.shape == (1, 4, 4, 5)
    np.testing.assert_array_equal(got.u_old[..., :-1], 17)
    np.testing.assert_array_equal(got.u_old[..., -1], 0)
    np.testing.assert_array_equal(got.v_old[..., :-1, :], 17)
    np.testing.assert_array_equal(got.v_old[..., -1, :], 0)
    np.testing.assert_array_equal(got.ph_old[:, :-1], 17)
    np.testing.assert_array_equal(got.ph_old[:, -1], 0)


def test_restart_schema_roundtrip_and_missing_leaf_refusal(monkeypatch):
    inp = inputs()
    nl = SimpleNamespace(data_assimilation=inp)
    carry = initial_spectral_carry(inp, jnp.float32)
    leaves, tree = jax.tree.flatten(carry)
    assert len(leaves) == 15 and all(a.dtype == jnp.float32 for a in leaves)
    restored = jax.tree.unflatten(tree, [np.asarray(a) for a in leaves])
    validate_spectral_fdda(restored, nl)
    for a, b in zip(jax.tree.leaves(restored), leaves):
        np.testing.assert_array_equal(a, b)
    for field in ("ph_old", "ph_new", "rphndgdten", "rqvndgdten"):
        with pytest.raises(ValueError, match="E78"):
            validate_spectral_fdda(replace(carry, **{field: None}), nl)
    with pytest.raises(ValueError, match="E78"):
        validate_spectral_fdda(None, nl)
    with pytest.raises(ValueError, match="schema"):
        SpectralFddaCarry.tree_unflatten(0, leaves)
    monkeypatch.setattr(fdda_rk, "_real_glue", lambda: True)
    assert isinstance(fdda_rk.seed_fdda_carry(nl), SpectralFddaCarry)


def test_ph_source_is_delivered_to_real_rk_physics_lane():
    from gpuwrf.runtime import operational_mode as om
    shape = (3, 4, 5)
    z = jnp.zeros(shape, jnp.float32)
    metrics = SimpleNamespace(c1h=jnp.ones(3, jnp.float32), c2h=jnp.zeros(3, jnp.float32), precision="fp32")
    owned = (jnp.zeros((3, 4, 6)), jnp.zeros((3, 5, 5)), z, z, jnp.full((4, 4, 5), .125))
    result = om._source_leaf_dry_tendencies(jnp.full((4, 5), 9e4), z, None, None, None, None,
        z, jnp.full(shape, 300.), metrics, jnp.float32, real_glue=True, fdda_tendf=owned)
    np.testing.assert_array_equal(result.ph_tendf, .125)


def test_actual_restart_formats_preserve_all_spectral_leaves(tmp_path):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.io.restart import write_restart, read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint_with_runtime_state
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry

    path = Path(__file__).resolve().parents[2] / "test_p0_5_restart_full_carry.py"
    spec = importlib.util.spec_from_file_location("spectral_restart_helpers", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    grid = GridSpec.canary_3km_template()
    state = helper._state(grid)
    # The fixture is storage-only. Populate every independently enabled held
    # diagnostic rather than disabling the release's E78 protection.
    shape = (grid.nz, grid.ny, grid.nx)
    state = state.replace(re_cloud=jnp.zeros(shape), re_ice=jnp.zeros(shape), re_snow=jnp.zeros(shape),
                          sfc_wspd=jnp.zeros((grid.ny, grid.nx)))
    inp = inputs(grid.nz, grid.ny, grid.nx)
    group = initial_spectral_carry(inp, jnp.float32).replace(
        rundgdten=jnp.full((grid.nz, grid.ny, grid.nx+1), .2, jnp.float32),
        rphndgdten=jnp.full((grid.nz+1, grid.ny, grid.nx), .125, jnp.float32))
    carry = initial_operational_carry(state).replace(fdda=group)
    nl = OperationalNamelist(grid=grid, tendencies=None, metrics=grid.metrics, dt_s=18.,
                              acoustic_substeps=4, data_assimilation=inp)
    pickle_path = tmp_path / "spectral.pkl"
    write_restart(carry, nl, grid, 137, pickle_path)
    restored, restored_nl, _, step = read_restart(pickle_path)
    assert step == 137 and restored_nl.data_assimilation.record_steps == (0,)
    nc_path = tmp_path / "spectral.nc"
    write_wrfrst_carry(carry, grid, nl, nc_path, valid_time="2023-01-15_00:41:06",
                       run_start="2023-01-15_00:00:00", step_index=137)
    restored_nc, meta = read_wrfrst_carry(nc_path, namelist=nl)
    assert meta["step_index"] == 137
    checkpoint_path = tmp_path / "checkpoint.pkl"
    with pytest.raises(ValueError, match="E78"):
        write_checkpoint(state, nl, grid, 137, checkpoint_path)
    write_checkpoint(state, nl, grid, 137, checkpoint_path, runtime_state={"fdda": group})
    *_, runtime = read_checkpoint_with_runtime_state(checkpoint_path)
    for back in (restored.fdda, restored_nc.fdda, runtime["fdda"]):
        assert isinstance(back, SpectralFddaCarry)
        for a, b in zip(jax.tree.leaves(group), jax.tree.leaves(back)):
            np.testing.assert_array_equal(a, b)
            assert np.asarray(a).dtype == np.asarray(b).dtype
