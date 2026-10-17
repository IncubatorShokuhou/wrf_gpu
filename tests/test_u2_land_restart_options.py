"""Static Noah-MP selections/table constants are restart identity (E78)."""
from dataclasses import replace
import pickle

import pytest


@pytest.mark.parametrize("format",["pickle","netcdf","checkpoint"])
def test_options_roundtrip_and_missing_or_changed_selection(tmp_path,format):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_noahmp_checkpoint_v2 import _namelist
    from test_v0110_wrfrst_netcdf import _state
    grid=GridSpec.canary_3km_template()
    state=_state(grid)
    carry=initial_operational_carry(state)
    nl=replace(_namelist(grid),noahmp_options=NoahMPOptions(dveg=8,opt_btr=3,opt_rsf=4,rsurf_snow=50.))
    requested=replace(nl,noahmp_options=NoahMPOptions(dveg=9))
    path=tmp_path/format
    if format=="pickle":
        from gpuwrf.io.restart import write_restart,read_restart
        write_restart(carry,nl,grid,1,path)
        restored,*_=read_restart(path,namelist=nl)
        with pytest.raises(ValueError,match="restart options differ"):read_restart(path,namelist=requested)
    elif format=="netcdf":
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
        from gpuwrf.io.netcdf_lock import Dataset
        write_wrfrst_carry(carry,grid,nl,path,valid_time="2026-06-03_00:00:18",run_start="2026-06-03_00:00:00",step_index=1)
        restored,_=read_wrfrst_carry(path,namelist=nl)
        with pytest.raises(ValueError,match="restart options differ"):read_wrfrst_carry(path,namelist=requested)
        with Dataset(path,"a") as ds:ds.delncattr("GPUWRF_NOAHMP_OPTIONS")
        with pytest.raises(ValueError,match="restart options differ"):read_wrfrst_carry(path,namelist=nl)
    else:
        from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint
        write_checkpoint(state,nl,grid,1,path,runtime_state=carry)
        read_checkpoint(path)
        with path.open("rb") as handle:payload=pickle.load(handle)
        assert payload["noahmp_options"]["dveg"]==8
        del payload["noahmp_options"]
        with path.open("wb") as handle:pickle.dump(payload,handle)
        with pytest.raises(ValueError,match="restart options differ"):read_checkpoint(path)


def test_table_parameter_change_is_restart_identity():
    from types import SimpleNamespace
    from gpuwrf.io.noahmp_options import NoahMPOptions,restart_configuration,validate_option_restart
    nl=SimpleNamespace(noahmp_options=NoahMPOptions(opt_rsf=4,rsurf_snow=50.))
    saved=restart_configuration(nl)
    with pytest.raises(ValueError,match="selection/table"):
        validate_option_restart(saved,SimpleNamespace(noahmp_options=NoahMPOptions(opt_rsf=4,rsurf_snow=51.)))


@pytest.mark.parametrize('format',('pickle','netcdf','checkpoint'))
def test_resolved_table_values_and_missing_identity_are_refused(tmp_path,format):
    import jax.numpy as jnp
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_noahmp_checkpoint_v2 import _namelist
    from test_v0110_wrfrst_netcdf import _state
    grid=GridSpec.canary_3km_template();state=_state(grid);carry=initial_operational_carry(state)
    nl=replace(_namelist(grid),noahmp_options=NoahMPOptions(dveg=8),noahmp_energy_params=jnp.asarray([1.],jnp.float32))
    changed=replace(nl,noahmp_energy_params=jnp.asarray([2.],jnp.float32));path=tmp_path/format
    if format=='pickle':
        from gpuwrf.io.restart import write_restart,read_restart
        write_restart(carry,nl,grid,1,path)
        read_restart(path,namelist=nl)
        with pytest.raises(ValueError,match='bound table'):read_restart(path,namelist=changed)
        with path.open('rb') as f:payload=pickle.load(f)
        del payload['noahmp_bound_tables']
        with path.open('wb') as f:pickle.dump(payload,f)
        with pytest.raises(ValueError,match='bound table'):read_restart(path,namelist=nl)
    elif format=='netcdf':
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
        from gpuwrf.io.netcdf_lock import Dataset
        write_wrfrst_carry(carry,grid,nl,path,valid_time='2026-06-03_00:00:18',run_start='2026-06-03_00:00:00',step_index=1)
        read_wrfrst_carry(path,namelist=nl)
        with pytest.raises(ValueError,match='bound table'):read_wrfrst_carry(path,namelist=changed)
        with Dataset(path,'a') as ds:ds.delncattr('GPUWRF_NOAHMP_BOUND_TABLES')
        with pytest.raises(ValueError,match='bound table'):read_wrfrst_carry(path,namelist=nl)
    else:
        from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint
        write_checkpoint(state,nl,grid,1,path,runtime_state=carry)
        read_checkpoint(path)
        with path.open('rb') as f:payload=pickle.load(f)
        payload['namelist']=changed
        with path.open('wb') as f:pickle.dump(payload,f)
        with pytest.raises(ValueError,match='bound table'):read_checkpoint(path)
