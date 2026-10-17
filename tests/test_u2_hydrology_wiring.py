"""Actual native Noah driver and fail-closed restart for runoff/tile selections."""
from dataclasses import replace
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('run,tdrn',[(6,0),(7,0),(8,0),(3,1),(3,2)])
def test_actual_hydrology_binding_and_noah_driver(tmp_path,monkeypatch,run,tdrn):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.io.noahmp_hydrology import bind_hydrology
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    from gpuwrf.contracts.grid import GridSpec
    from test_noahmp_checkpoint_v2 import _namelist
    monkeypatch.setenv('GPUWRF_NOAHMP_NATIVE_REAL','1');monkeypatch.setenv('GPUWRF_NOAHMP_COLUMN_KERNELS','0');jax.clear_caches()
    path=ROOT/'tests/v025/b_noahmp/test_snow_wrf_order_oracle.py'
    spec=importlib.util.spec_from_file_location('hydrology_columns',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    columns=json.loads((path.parent/'fixtures/swiss_snow_wrf_order.json').read_text())['columns'][-2:]
    columns=[dict(c,isltyp=1,vegtyp=12) for c in columns];c=mod._build(columns)
    land=c['land'].replace(tslb=jnp.full((4,1,2),290.,jnp.float32),smois=jnp.full((4,1,2),.33,jnp.float32),sh2o=jnp.full((4,1,2),.33,jnp.float32),
        tv=jnp.full((1,2),290.,jnp.float32),tg=jnp.full((1,2),290.,jnp.float32),tah=jnp.full((1,2),290.,jnp.float32))
    forcing=c['forcing']._replace(sfctmp=jnp.full((1,2),290.,jnp.float32),julian=jnp.float32(200.))
    opts=NoahMPOptions(opt_run=run,opt_infdv=3 if run==8 else 0,opt_tdrn=tdrn,drain_layer_opt=4 if tdrn else None)
    nl=replace(_namelist(GridSpec.canary_3km_template()),noahmp_static=c['static'],noahmp_land=land,noahmp_options=opts)
    with Dataset(tmp_path/'wrfinput_d01','w') as ds:
        for key,n in (('Time',1),('south_north',1),('west_east',2)):ds.createDimension(key,n)
        ds.createVariable('TD_FRACTION','f4',('Time','south_north','west_east'))[:]=1.
    nl=bind_hydrology(nl,tmp_path,'d01',state=SimpleNamespace(t_skin=land.t_skin))
    kwargs={'tile_drain':jnp.zeros((1,2),jnp.float32)} if tdrn else {}
    result=noah_mp_step(land,forcing,nl.noahmp_static,18.,energy_params=nl.noahmp_energy_params,rad_params=nl.noahmp_rad_params,options=nl.noahmp_options,history=True,**kwargs)
    assert len(result)==(3 if tdrn else 2)
    if tdrn:
        assert result[2].dtype==jnp.float32 and np.any(np.asarray(result[2])>0.),'Inactive tile drainage caller'
        assert 'ZWT' in result[1].history
        with pytest.raises(ValueError,match='QTDRAIN'):noah_mp_step(land,forcing,nl.noahmp_static,18.,options=nl.noahmp_options)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result))
    jax.clear_caches()


@pytest.mark.parametrize('format',['pickle','netcdf','checkpoint','native_store'])
def test_qtdrain_exact_restart(tmp_path,format):
    from gpuwrf.contracts.grid import GridSpec
    from gpuwrf.io.noahmp_options import NoahMPOptions
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_noahmp_checkpoint_v2 import _namelist,_land
    from test_v0110_wrfrst_netcdf import _state
    grid=GridSpec.canary_3km_template();state=_state(grid);ny,nx=state.t_skin.shape
    group=jnp.full((ny,nx),1.25,jnp.float32)
    carry=initial_operational_carry(state).replace(noahmp_tile_drain=group,noahmp_land=_land(ny,nx))
    nl=replace(_namelist(grid),noahmp_options=NoahMPOptions(opt_tdrn=2,drain_layer_opt=4))
    path=tmp_path/format
    if format=='pickle':
        from gpuwrf.io.restart import write_restart,read_restart
        write_restart(carry,nl,grid,1,path);restored,*_=read_restart(path,namelist=nl)
        with pytest.raises(ValueError,match='tile drainage restart|restart options differ'):read_restart(path,namelist=replace(nl,noahmp_options=None))
    elif format=='netcdf':
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
        from gpuwrf.io.netcdf_lock import Dataset
        write_wrfrst_carry(carry,grid,nl,path,valid_time='2026-06-03_00:00:18',run_start='2026-06-03_00:00:00',step_index=1)
        restored,_=read_wrfrst_carry(path,namelist=nl)
        with Dataset(path,'r') as ds:np.testing.assert_array_equal(ds['QTDRAIN'][0],group)
        with Dataset(path,'a') as ds:ds.delncattr('GPUWRF_NOAHMP_TILE_DRAIN_SCHEMA_VERSION')
        with pytest.raises(ValueError,match='tile drainage restart'):read_wrfrst_carry(path,namelist=nl)
    elif format=='checkpoint':
        from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
        write_checkpoint(state,nl,grid,1,path,runtime_state=carry,land_state=carry.noahmp_land)
        *_,restored=read_checkpoint_with_runtime_state(path)
    else:
        from gpuwrf.runtime.restart_store import RestartStore,_validate_snapshot
        from gpuwrf.io.noahmp_options import restart_configuration
        import pickle
        store=RestartStore(path,10**8,2,0)
        identity={'noahmp_options_config':{'d01':restart_configuration(nl)}}
        generation,_=store.save({'d01':carry},{'d01':1},{'d01':18.},identity)
        snapshot,_=store.read(generation,expected_identity=identity)
        restored=snapshot['carries']['d01']
        with (generation/'snapshot.pkl').open('rb') as stream:raw=pickle.load(stream)
        manifest=json.loads((generation/'manifest.json').read_text())
        raw['carries']['d01'].pop('noahmp_tile_drain_schema_version')
        with pytest.raises(ValueError,match='tile drainage restart'):_validate_snapshot(raw,manifest)
    np.testing.assert_array_equal(group,restored.noahmp_tile_drain)


@pytest.mark.parametrize('problem',['absent','version','nan','dtype','shape','inactive'])
def test_tile_restart_fails_closed(problem):
    from gpuwrf.io.noahmp_hydrology import validate_tile_restart
    from gpuwrf.io.noahmp_options import NoahMPOptions
    group=np.zeros((2,3),np.float32);nl=SimpleNamespace(noahmp_options=NoahMPOptions(opt_tdrn=1,drain_layer_opt=4));version=1
    if problem=='absent':group=None
    if problem=='version':version=None
    if problem=='nan':group[:]=np.nan
    if problem=='dtype':group=group.astype(np.float64)
    if problem=='shape':group=group.T
    if problem=='inactive':nl.noahmp_options=None
    with pytest.raises(ValueError,match='tile drainage restart'):validate_tile_restart(group,nl,version,(2,3))
