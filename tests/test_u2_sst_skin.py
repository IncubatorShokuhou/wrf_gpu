"""Literal WRF skin physics plus caller timing, no-ice admission and E78 restart."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.sst_skin_hook import SSTSkinState, apply_sst_skin, sst_skin_update, validate_restart

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("compiled", [False, True])
def test_24step_pristine_wrf_real_trajectory(compiled):
    fx = json.loads((ROOT / "proofs/u2_land/skin_oracle.json").read_text())
    x = np.asarray(fx["inputs"], np.float32).T
    skin = SSTSkinState(jnp.asarray(x[5]), jnp.asarray(x[8]), jnp.asarray(x[9]))
    step = lambda skin,inputs: sst_skin_update(*inputs[:8], skin, inputs[10,0])
    if compiled:
        step = jax.jit(step)
    ref = np.asarray(fx["output"])
    for k in range(24):
        skin = step(skin,jnp.asarray(x))
        actual = np.stack([np.asarray(skin.sstsk), np.asarray(skin.dtw)], -1)
        assert actual.dtype == np.float32 and np.isfinite(actual).all() and np.isfinite(ref[k]).all()
        np.testing.assert_allclose(actual[:,0], ref[k,:,0], atol=fx["tolerance"]["sstsk_atol"], rtol=0.)
        np.testing.assert_allclose(actual[:,1], ref[k,:,1], atol=fx["tolerance"]["dtw_atol"], rtol=fx["tolerance"]["dtw_rtol"])


@pytest.mark.parametrize("which", ["absent", "dtype", "shape", "nan", "inf", "version", "selection"])
def test_restart_fails_closed(which):
    base = jnp.full((2,3), 290., jnp.float32)
    skin = SSTSkinState(base, base, jnp.zeros_like(base))
    version, enabled = 1, True
    if which == "absent": skin = None
    if which == "dtype": skin = skin._replace(dtw=np.zeros((2,3), np.float64))
    if which == "shape": skin = skin._replace(dtw=np.zeros((2,4), np.float32))
    if which in ("nan", "inf"): skin = skin._replace(sstsk=np.full((2,3), float(which), np.float32))
    if which == "version": version = None
    if which == "selection": enabled = False
    with pytest.raises(ValueError, match="sst_skin.*restart"):
        validate_restart(skin, enabled, version, (2,3))


def test_surface_driver_uses_previous_fluxes_and_distinct_bulk_sst():
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state
    grid = build_flat_grid(nx=3, ny=2, nz=3, dx_m=3000.)
    state = build_neutral_state(grid)
    row = lambda x: jnp.full((2,3), x, jnp.float32)
    state = state.replace(xland=row(2.), t_skin=row(299.), ustar=row(.2), hfx=row(20.), qfx=row(.00002))
    skin = SSTSkinState(row(290.), row(291.), row(.1))
    rad = SimpleNamespace(glw=row(350.), swnorm=row(700.), swup_topographic=row(200.), surface_emissivity=row(.98))
    carry = initial_operational_carry(state).replace(sst_skin=skin, radiation_diagnostics=rad)
    nl = SimpleNamespace(sst_skin=1, dt_s=18.)
    # The separately oracle-qualified kernel must receive stored bulk SST,
    # previous fluxes and net down-up radiation at this caller boundary.
    expected = sst_skin_update(state.xland, rad.glw, row(500.), state.hfx, state.qfx,
                               row(290.), state.ustar, row(.98), skin, 18.)
    result = apply_sst_skin(carry, nl)
    np.testing.assert_array_equal(result.state.t_skin, expected.sstsk)
    np.testing.assert_array_equal(result.sst_skin.bulk_sst, row(290.))
    lower = {"SST": row(288.)}
    from gpuwrf.io.lower_boundary import apply_lower_boundary_fields
    changed = apply_sst_skin(apply_lower_boundary_fields(carry,lower), nl, lower)
    np.testing.assert_array_equal(changed.sst_skin.bulk_sst, row(288.))
    assert not np.array_equal(changed.state.t_skin, result.state.t_skin)

    # WRF's prescribed-SST phase holds TSK for invalid SST, and the skin
    # phase must not replace that hold with the raw invalid SST record.
    invalid={"SST":row(0.)}
    held=apply_sst_skin(apply_lower_boundary_fields(carry,invalid),nl,invalid)
    expected_held=sst_skin_update(state.xland,rad.glw,row(500.),state.hfx,state.qfx,
                                  state.t_skin,state.ustar,row(.98),skin._replace(bulk_sst=row(0.)),18.)
    np.testing.assert_array_equal(held.state.t_skin,expected_held.sstsk)


def test_operational_restart_preserves_skin_and_rejects_missing_group(tmp_path):
    import pickle
    from test_noahmp_checkpoint_v2 import _namelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state
    from gpuwrf.io.restart import write_restart, read_restart
    grid = build_flat_grid(nx=3, ny=2, nz=3, dx_m=3000.)
    state = build_neutral_state(grid)
    base = jnp.full((2,3), 290., jnp.float32)
    skin = SSTSkinState(base, base + 1., jnp.full_like(base,.1))
    carry = initial_operational_carry(state).replace(sst_skin=skin)
    nl = replace(_namelist(grid), sst_skin=1, sst_skin_initial=skin)
    target = tmp_path / "skin.pkl"
    write_restart(carry, nl, grid, 1, target)
    resumed, _, _, _ = read_restart(target)
    for old, new in zip(skin, resumed.sst_skin):
        np.testing.assert_array_equal(old, new)
    with target.open("rb") as f: payload = pickle.load(f)
    payload["carry"]["extra_fields"]["sst_skin"] = None
    with target.open("wb") as f: pickle.dump(payload,f)
    with pytest.raises(ValueError, match="sst_skin.*restart"):
        read_restart(target)


def test_off_carry_uses_released_pytree_prefix():
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state
    carry = initial_operational_carry(build_neutral_state(build_flat_grid(nx=3, ny=2, nz=3, dx_m=3000.)))
    children, aux = carry.tree_flatten()
    # All absent option groups retain the schema preceding main's MMF append.
    assert len(children) == list(carry.__dataclass_fields__).index('noahmp_mmf')
    assert type(carry).tree_unflatten(aux, children).sst_skin is None


def test_actual_netcdf_restart_and_wrf_native_skin_fields(tmp_path):
    from gpuwrf.contracts.grid import GridSpec
    from test_v0110_wrfrst_netcdf import _state
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry
    from gpuwrf.io.netcdf_lock import Dataset
    grid = GridSpec.canary_3km_template()
    state = _state(grid)
    base = jnp.full(state.t_skin.shape, 290., jnp.float32)
    skin = SSTSkinState(base, base + 1., jnp.full_like(base, .1))
    carry = initial_operational_carry(state).replace(sst_skin=skin)
    path = tmp_path / "skin.nc"
    write_wrfrst_carry(carry, grid, {"sst_skin":1}, path,
                      valid_time="2026-06-03_00:00:18", run_start="2026-06-03_00:00:00", step_index=1)
    restored, _ = read_wrfrst_carry(path, namelist={"sst_skin":1})
    for old, new in zip(skin, restored.sst_skin):
        np.testing.assert_array_equal(old, new)
    with Dataset(path,"r") as ds:
        for field, expected in zip(("SST","SSTSK","DTW"), skin):
            np.testing.assert_array_equal(ds[field][0], expected)
    with Dataset(path,"a") as ds:
        ds.GPUWRF_SST_SKIN_SCHEMA_VERSION=0
    with pytest.raises(ValueError, match="sst_skin.*restart"):
        read_wrfrst_carry(path)


def test_state_checkpoint_cannot_drop_active_skin(tmp_path):
    from gpuwrf.contracts.grid import GridSpec
    from test_noahmp_checkpoint_v2 import _state, _namelist
    from gpuwrf.runtime.checkpoint import write_checkpoint
    grid = GridSpec.canary_3km_template()
    nl = replace(_namelist(grid), sst_skin=1)
    with pytest.raises(ValueError, match="sst_skin.*restart"):
        write_checkpoint(_state(grid), nl, grid, 1, tmp_path/"bad.pkl")
