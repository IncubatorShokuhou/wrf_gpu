"""Exact restart and full-driver requirements for SIMGM positive-depth groundwater."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("format",["pickle","netcdf","checkpoint"])
def test_groundwater_restart_roundtrip(tmp_path,format):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.physics.noahmp.groundwater_options import GroundwaterState
    from test_noahmp_checkpoint_v2 import _namelist,_land
    from test_v0110_wrfrst_netcdf import _state
    grid=GridSpec.canary_3km_template();state=_state(grid);ny,nx=state.t_skin.shape
    group=GroundwaterState(*(jnp.full((ny,nx),v,jnp.float32) for v in (2.49,4901.,4901.)))
    carry=initial_operational_carry(state).replace(noahmp_groundwater=group,noahmp_land=_land(ny,nx))
    nl=replace(_namelist(grid),noahmp_options=NoahMPOptions(opt_run=1,fsatmx=.38,timean=10.5))
    path=tmp_path/format
    if format=="pickle":
        from gpuwrf.io.restart import write_restart,read_restart
        write_restart(carry,nl,grid,1,path);restored,*_=read_restart(path,namelist=nl)
        with pytest.raises(ValueError,match="groundwater restart|restart options differ"):read_restart(path,namelist=replace(nl,noahmp_options=None))
    elif format=="netcdf":
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
        from gpuwrf.io.netcdf_lock import Dataset
        write_wrfrst_carry(carry,grid,nl,path,valid_time="2026-06-03_00:00:18",run_start="2026-06-03_00:00:00",step_index=1)
        restored,_=read_wrfrst_carry(path,namelist=nl)
        with Dataset(path,"r") as ds:
            for name,value in group._asdict().items():np.testing.assert_array_equal(ds[name.upper()][0],value)
        with Dataset(path,"a") as ds:ds.delncattr("GPUWRF_NOAHMP_GROUNDWATER_SCHEMA_VERSION")
        with pytest.raises(ValueError,match="groundwater restart schema"):read_wrfrst_carry(path,namelist=nl)
    else:
        from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
        write_checkpoint(state,nl,grid,1,path,runtime_state=carry,land_state=carry.noahmp_land)
        *_,restored=read_checkpoint_with_runtime_state(path)
    for a,b in zip(group,restored.noahmp_groundwater):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize("problem",["absent","version","nan","dtype","shape","inactive"])
def test_groundwater_restart_fails_closed(problem):
    from types import SimpleNamespace
    from gpuwrf.physics.noahmp.groundwater_options import GroundwaterState
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.io.noahmp_groundwater import validate_restart
    group=GroundwaterState(*(np.zeros((2,3),np.float32) for _ in range(3)))
    nl=SimpleNamespace(noahmp_options=NoahMPOptions(opt_run=1,fsatmx=.38,timean=10.5));version=1
    if problem=="absent":group=None
    if problem=="version":version=None
    if problem=="nan":group=group._replace(wa=np.full((2,3),np.nan,np.float32))
    if problem=="dtype":group=group._replace(wa=np.zeros((2,3),np.float64))
    if problem=="shape":group=group._replace(wa=np.zeros((3,3),np.float32))
    if problem=="inactive":nl.noahmp_options=None
    with pytest.raises(ValueError,match="groundwater restart"):validate_restart(group,nl,version,(2,3))


@pytest.mark.parametrize("run",[1,2])
def test_run_options_reach_actual_noah_driver(monkeypatch,run):
    import importlib.util
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL","1")
    monkeypatch.setenv("GPUWRF_NOAHMP_COLUMN_KERNELS","0")
    jax.clear_caches()
    path=ROOT/"tests/v025/b_noahmp/test_snow_wrf_order_oracle.py"
    spec=importlib.util.spec_from_file_location("groundwater_columns",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    columns=json.loads((path.parent/"fixtures/swiss_snow_wrf_order.json").read_text())["columns"][:1];c=module._build(columns)
    from gpuwrf.physics.noahmp.groundwater_options import GroundwaterState
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    import gpuwrf.physics.noahmp.groundwater_options as gw
    evolved=[]
    original=gw.groundwater_step
    def record(*args,**kwargs):
        value=original(*args,**kwargs)
        evolved.append(value[0])
        return value
    monkeypatch.setattr(gw,"groundwater_step",record)
    initial=GroundwaterState(*(jnp.full((1,1),v,jnp.float32) for v in (2.5,4900.,4900.)))
    result=noah_mp_step(c["land"],c["forcing"],c["static"],c["dt"],energy_params=c["ep"],rad_params=c["rp"],
        options=NoahMPOptions(opt_run=run,opt_inf=2,fsatmx=.38,timean=10.5),history=True,
        **({"groundwater":initial} if run==1 else {}))
    assert len(result)==(3 if run==1 else 2)
    assert ("QIN" if run==1 else "ZWT") in result[1].history
    if run==1:
        assert isinstance(result[2],GroundwaterState)
        # The cold Swiss column's flux is below the storage's REAL4 ULP.
        # Check that the actual GROUNDWATER result reaches the driver output.
        assert len(evolved)==1
        for a,b in zip(evolved[0],result[2]):np.testing.assert_array_equal(a,b)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result))
    jax.clear_caches()
