"""Restart/output mechanics; seeded numerical fidelity is the separate WRF gate."""
from dataclasses import replace
from datetime import datetime

import jax
import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.stochastic_config import bind_stochastic
from gpuwrf.physics.stochastic_driver import initial_stochastic, advance_stochastic,grid_spacing


def _config(vertical):
    controls = dict(skebs=1,sppt=1,rand_perturb=1,spp=1,nens=7,
                    skebs_vertstruc=vertical,sppt_vertstruc=vertical,rand_pert_vertstruc=vertical,
                    vertstruc_spp_conv=vertical,vertstruc_spp_pbl=vertical,vertstruc_spp_lsm=vertical)
    controls.update({"lengthscale_"+key:6000. for key in
                     ("sppt","rand_pert","spp_conv","spp_pbl","spp_lsm")})
    return bind_stochastic({"stoch":controls},"d01")


def _same(a,b):
    left,lt = jax.tree.flatten(a)
    right,rt = jax.tree.flatten(b)
    assert lt == rt
    for x,y in zip(left,right):
        x,y = np.asarray(x),np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()


def _grid():
    from gpuwrf.contracts.grid import GridSpec
    import jax.numpy as jnp
    grid = GridSpec.canary_3km_template()
    return replace(grid,projection=replace(grid.projection,nx=12,ny=12),
        terrain=replace(grid.terrain,shape=(12,12)),terrain_height=jnp.zeros((12,12)),metrics=None)


@pytest.mark.parametrize("vertical",[0,1])
@pytest.mark.parametrize("format",["netcdf","pickle"])
def test_complete_stochastic_restart_continuity(tmp_path,vertical,format):
    from test_noahmp_checkpoint_v2 import _state,_namelist
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry,read_wrfrst_stochastic_seeds
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    from gpuwrf.io.stochastic_schema import seed_arrays
    grid = _grid()
    config = _config(vertical)
    namelist = replace(_namelist(grid),stochastic_config=config)
    state = _state(grid)
    stoch = initial_stochastic(config,grid,18,datetime(2023,1,15))
    dx,dy = grid_spacing(grid)
    step = jax.jit(lambda c:advance_stochastic(c,config,dx=dx,dy=dy))
    stoch = step(step(stoch))
    carry = initial_operational_carry(state).replace(stochastic=stoch)
    path = tmp_path/"restart"
    if format == "netcdf":
        write_wrfrst_carry(carry,grid,namelist,path,valid_time="2023-01-15_00:00:36",
                          run_start="2023-01-15_00:00:00",step_index=2)
        restored,_ = read_wrfrst_carry(path,namelist=namelist)
        seeds = read_wrfrst_stochastic_seeds(path)
        assert set(seeds) == set(seed_arrays(stoch))
        for name,seed in seed_arrays(stoch).items():
            np.testing.assert_array_equal(seeds[name],seed)
    else:
        write_checkpoint(state,namelist,grid,2,path,runtime_state=carry)
        *_,restored = read_checkpoint_with_runtime_state(path)
    _same(stoch,restored.stochastic)
    _same(step(stoch),step(restored.stochastic))
    # An active old checkpoint cannot silently create a new random stream.
    with pytest.raises(ValueError,match="stochastic.*E78"):
        if format == "netcdf":
            write_wrfrst_carry(carry.replace(stochastic=None),grid,namelist,tmp_path/"missing",
                valid_time="2023-01-15_00:00:36",run_start="2023-01-15_00:00:00",step_index=2)
        else:
            write_checkpoint(state,namelist,grid,2,tmp_path/"missing",runtime_state=carry.replace(stochastic=None))


def test_stochastic_history_exact_native_fields_and_off_schema(tmp_path):
    from test_m7_netcdf_writer import synthetic_case,authenticated_source_grid
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf,bind_wrfout_domain_authority
    from gpuwrf.io.stochastic_schema import history_fields,PATTERN_FIELDS,SEED_NAMES,CLIP_FIELDS
    state,grid,nml = synthetic_case(12,12,10)
    config = _config(1)
    carry = initial_stochastic(config,grid,18,datetime(2023,1,15))
    carry = advance_stochastic(carry,config,dx=3000.,dy=3000.)
    fields = history_fields(carry,grid.nz,grid.ny,grid.nx)
    kwargs = dict(domain="d01",domain_authority=bind_wrfout_domain_authority("d01",authenticated_source_grid(grid,"d01"),grid),
                  valid_time="2023-01-15_00:00:18",lead_hours=.005,run_start="2023-01-15_00:00:00")
    nml["stochastic_config"] = config
    write_wrfout_netcdf(state,grid,nml,tmp_path/"on",diagnostics=fields,**kwargs)
    with Dataset(tmp_path/"on") as ds:
        for name in (*PATTERN_FIELDS,*SEED_NAMES,*CLIP_FIELDS):
            np.testing.assert_array_equal(np.asarray(ds[name][0]),fields[name])
        assert len(ds.dimensions["num_stoch_levels"]) == 11
    nml.pop("stochastic_config")
    write_wrfout_netcdf(state,grid,nml,tmp_path/"off",**kwargs)
    with Dataset(tmp_path/"off") as ds:
        assert "num_stoch_levels" not in ds.dimensions
        assert not set(PATTERN_FIELDS).intersection(ds.variables)


@pytest.mark.parametrize("damage",["missing-plan","nonfinite-plan","wrong-plan-shape","missing-wind-component"])
def test_restart_rejects_corrupt_live_stochastic_plans(damage):
    import jax.numpy as jnp
    from gpuwrf.io.stochastic_schema import validate_stochastic
    cfg=_config(1)
    carry=initial_stochastic(cfg,_grid(),18,datetime(2023,1,15))
    validate_stochastic(carry,{"stochastic_config":cfg})
    if damage=="missing-plan":
        broken=carry._replace(vertical=None)
    elif damage=="nonfinite-plan":
        broken=carry._replace(vertical=(carry.vertical[0].at[0,0,0].set(jnp.nan),*carry.vertical[1:]))
    elif damage=="wrong-plan-shape":
        broken=carry._replace(vertical=(carry.vertical[0][:1],*carry.vertical[1:]))
    else:
        patterns=list(carry.patterns)
        patterns[0]=patterns[0]._replace(field=patterns[0].field[:1])
        broken=carry._replace(patterns=tuple(patterns))
    with pytest.raises(ValueError,match="stochastic.*E78"):
        validate_stochastic(broken,{"stochastic_config":cfg})
