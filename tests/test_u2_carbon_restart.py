"""Dynamic carbon pools are exact, mandatory full-carry restart state."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("format",["pickle","netcdf","checkpoint"])
@pytest.mark.parametrize("crop",[False,True])
def test_six_carbon_pools_roundtrip_and_missing_marker(tmp_path,format,crop):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.physics.noahmp.carbon import CarbonState
    from test_noahmp_checkpoint_v2 import _namelist,_land
    from test_v0110_wrfrst_netcdf import _state
    grid=GridSpec.canary_3km_template();state=_state(grid);ny,nx=state.t_skin.shape
    pools=CarbonState(*(jnp.full((ny,nx),value,jnp.float32) for value in (60.,500.,66.,500.,1000.,1000.)))
    if crop:
        from gpuwrf.physics.noahmp.crop import CropCarbonState
        pools=CropCarbonState(*pools,jnp.full((ny,nx),10.,jnp.float32),jnp.full((ny,nx),720.,jnp.float32),jnp.full((ny,nx),4,jnp.int32))
    carry=initial_operational_carry(state).replace(noahmp_carbon=pools,noahmp_land=_land(ny,nx))
    nl=replace(_namelist(grid),noahmp_options=NoahMPOptions(opt_crop=1) if crop else NoahMPOptions(dveg=2))
    path=tmp_path/format
    if format=="pickle":
        from gpuwrf.io.restart import write_restart,read_restart
        write_restart(carry,nl,grid,1,path)
        restored,*_=read_restart(path,namelist=nl)
        with pytest.raises(ValueError,match="carbon restart|restart options differ"):read_restart(path,namelist=replace(nl,noahmp_options=None))
    elif format=="netcdf":
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
        from gpuwrf.io.netcdf_lock import Dataset
        write_wrfrst_carry(carry,grid,nl,path,valid_time="2026-06-03_00:00:18",run_start="2026-06-03_00:00:00",step_index=1)
        restored,_=read_wrfrst_carry(path,namelist=nl)
        with Dataset(path,"r") as ds:
            for name,value in pools._asdict().items():np.testing.assert_array_equal(ds[name.upper()][0],value)
        with Dataset(path,"a") as ds:ds.delncattr("GPUWRF_NOAHMP_CARBON_SCHEMA_VERSION")
        with pytest.raises(ValueError,match="carbon restart schema"):read_wrfrst_carry(path,namelist=nl)
    else:
        from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
        write_checkpoint(state,nl,grid,1,path,runtime_state=carry,land_state=carry.noahmp_land)
        *_,restored=read_checkpoint_with_runtime_state(path)
    for a,b in zip(pools,restored.noahmp_carbon):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize("problem",["absent","version","nan","dtype","shape","inactive"])
def test_carbon_restart_fails_closed(problem):
    from types import SimpleNamespace
    from gpuwrf.physics.noahmp.carbon import CarbonState
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.io.noahmp_carbon import validate_restart
    pools=CarbonState(*(np.zeros((2,3),np.float32) for _ in range(6)))
    nl=SimpleNamespace(noahmp_options=NoahMPOptions(dveg=5));version=1
    if problem=="absent":pools=None
    if problem=="version":version=None
    if problem=="nan":pools=pools._replace(wood=np.full((2,3),np.nan,np.float32))
    if problem=="dtype":pools=pools._replace(wood=np.zeros((2,3),np.float64))
    if problem=="shape":pools=pools._replace(wood=np.zeros((3,3),np.float32))
    if problem=="inactive":nl.noahmp_options=None
    with pytest.raises(ValueError,match="carbon restart"):validate_restart(pools,nl,version,(2,3))


@pytest.mark.parametrize('columns',[False,True])
def test_dynamic_carbon_reaches_actual_noah_driver(monkeypatch,columns):
    import importlib.util
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL","1")
    monkeypatch.setenv("GPUWRF_NOAHMP_COLUMN_KERNELS","1" if columns else "0")
    jax.clear_caches()
    path=ROOT/"tests/v025/b_noahmp/test_snow_wrf_order_oracle.py"
    spec=importlib.util.spec_from_file_location("carbon_columns",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    columns=json.loads((path.parent/"fixtures/swiss_snow_wrf_order.json").read_text())["columns"][:1];c=module._build(columns)
    from gpuwrf.physics.noahmp.carbon import CarbonState,CarbonParameters
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    from gpuwrf.physics.noahmp.tables import _parse_mptable_block
    from gpuwrf.config.paths import wrf_run_dir
    tables=_parse_mptable_block((wrf_run_dir()/"MPTABLE.TBL").read_text(),"noahmp_modis_parameters")
    category=int(np.asarray(c["static"].ivgtyp)[0,0])
    p=CarbonParameters(*(jnp.full((1,1),tables[key.upper()][category-1],jnp.float32) for key in CarbonParameters._fields))
    pools=CarbonState(c["land"].lai.astype(jnp.float32)*(1000./jnp.maximum(p.sla,1.)),jnp.full((1,1),500.,jnp.float32),
                      c["land"].sai.astype(jnp.float32)*(jnp.float32(1000.)/jnp.float32(3.)),
                      *(jnp.full((1,1),value,jnp.float32) for value in (500.,1000.,1000.)))
    result=noah_mp_step(c["land"],c["forcing"],c["static"],c["dt"],energy_params=c["ep"],rad_params=c["rp"],
                        options=NoahMPOptions(dveg=2),carbon=(p,pools),history=True)
    assert len(result)==3 and "GPP" in result[1].history
    assert isinstance(result[2],CarbonState)
    assert any(not np.array_equal(a,b) for a,b in zip(pools,result[2]))
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result))
    jax.clear_caches()
