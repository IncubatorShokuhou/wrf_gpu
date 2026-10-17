"""Original Registry inventory at the State, transport, restart and file boundaries."""
import dataclasses
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXTRAS = dict(qi2='QICE2', Ni2='QNICE2', qi3='QICE3', Ni3='QNICE3',
              qvoli='QVOLI', qaoli='QAOLI', qvoli2='QVOLI2', qaoli2='QAOLI2',
              qvoli3='QVOLI3', qaoli3='QAOLI3')
HISTORY = {**EXTRAS, 'Ni':'QNICE', 'Nr':'QNRAIN', 're_cloud':'RE_CLOUD', 're_ice':'RE_ICE'}
MOIST = ('qv','qc','qr','qi','qi2','qi3')
SCALAR = ('Ni','Nr','qvoli','qaoli','Ni2','qvoli2','qaoli2','Ni3','qvoli3','qaoli3')


def fixture(name='test_rrtm_lw_operational_wiring.py'):
    spec = importlib.util.spec_from_file_location('jensen_'+name.replace('.','_'), ROOT/'tests'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_transport_and_adapter(monkeypatch):
    from gpuwrf.contracts import state as sc
    from gpuwrf.contracts.physics_registry import state_leaves_for_mp
    from gpuwrf.physics.jensen_ishmael import adapter, column
    from gpuwrf.runtime import operational_mode as op
    monkeypatch.setattr(sc, '_gpu_device', lambda: jax.devices('cpu')[0])
    f=fixture(); grid=f._grid(nz=8,ny=3,nx=4)
    dormant=f._state(grid)
    assert all(getattr(dormant,n) is None for n in EXTRAS)
    state=dormant.ensure_conditional_leaves(mp_physics=55)
    nl=dataclasses.replace(f._namelist(grid),mp_physics=55,run_physics=True,
                           moist_adv_opt=1,scalar_adv_opt=1)
    assert set(state_leaves_for_mp(55))==set((*MOIST,*SCALAR,'re_cloud','re_ice'))
    assert op._advected_scalar_species(nl)==MOIST
    assert op._nested_number_scalar_species(nl)==SCALAR
    assert op._diffused_scalar_species(nl)==MOIST+SCALAR
    assert set(op._mixing_scalar_species(state,nl))==set(MOIST+SCALAR)
    assert set(op._root_flow_only(nl)['flow_only'])==set(EXTRAS)
    captured=[]
    def step(fields,dt,it):
        captured.append((fields,dt,it))
        output={n:value for n,value in fields.items() if n not in ('p','dz')}
        output.update(effc=jnp.full_like(fields['qv'],20.e-6),effi=jnp.full_like(fields['qv'],50.e-6))
        for cat in (1,2,3):
            for name in ('di','rho','vm','phi'):
                output[name+str(cat)] = jnp.full_like(fields['qv'],cat/16)
        z=jnp.zeros(fields['qv'].shape[:-1],jnp.float32)
        return output,dict(rainstep=z+3,snowstep=z+1)
    monkeypatch.setattr(column,'step',step)
    result=op._apply_post_rk_microphysics(state,nl,step_index=jnp.int32(7),return_precipitation=True)
    assert len(captured)==1 and int(captured[0][2])==7
    fields,_,_=captured[0]
    assert set(fields)==set(adapter.output_mapping())-{'effc','effi'}|{'th','p','dz'}
    assert all(value.dtype==jnp.float32 for value in fields.values())
    # WRF total precipitation is separated exactly once into liquid/solid State accumulators.
    updated=result[0]
    np.testing.assert_array_equal(updated.rain_acc-state.rain_acc,2)
    np.testing.assert_array_equal(updated.snow_acc-state.snow_acc,1)
    for name in EXTRAS:np.testing.assert_array_equal(getattr(updated,name),getattr(state,name))
    with pytest.raises(ValueError,match='habit moments'):
        adapter.jensen_adapter(state.replace(qvoli3=None),18.,grid)


def test_admission_and_radiation_flags():
    from gpuwrf.io.namelist_check import validate_operational_namelist, refuse_single_domain_mp_nested, NotOperationallyWiredError
    from gpuwrf.io.namelist_binding import require_namelist_honoured, bind_namelist_options
    from gpuwrf.physics.p3.radiation import mp_re_active
    f=fixture();grid=f._grid(nz=8,ny=3,nx=4)
    raw={'physics':{'mp_physics':[55],'cu_physics':[0]}}
    validate_operational_namelist(raw);require_namelist_honoured(raw,('d01',))
    assert bind_namelist_options(f._namelist(grid),raw,'d01').mp_physics==55
    assert mp_re_active(mp_physics=55,use_mp_re=1,ra_lw_physics=4,ra_sw_physics=4)
    with pytest.raises(NotOperationallyWiredError):refuse_single_domain_mp_nested(raw,2)


def test_third_ice_enters_moist_loading_and_native_hydrostatic(monkeypatch):
    """Regression for the original sum over every active moist Registry member."""
    from gpuwrf.contracts import state as sc
    from gpuwrf.dynamics.acoustic_wrf import moisture_coupling_factors
    from gpuwrf.kernels.phys_hydrostatic_real import hydrostatic_profiles
    from gpuwrf.coupling.physics_couplers import _cloud_fraction_columns
    monkeypatch.setattr(sc,'_gpu_device',lambda:jax.devices('cpu')[0])
    f=fixture();grid=f._grid(nz=8,ny=3,nx=4)
    state=f._state(grid).ensure_conditional_leaves(mp_physics=55)
    state=state.replace(**{n:jnp.zeros_like(state.qv) for n in MOIST[:4]+('qs','qg','qi2')},
                        qi3=jnp.full_like(state.qv,1.e-6,dtype=jnp.float32))
    for value in moisture_coupling_factors(state):
        np.testing.assert_allclose(value,1./(1.+1.e-6),rtol=1.e-7,atol=0)
    np.testing.assert_allclose(_cloud_fraction_columns(state),.1,rtol=1.e-7,atol=0)
    nz,ny,nx=4,2,3;zero=jnp.zeros((nz,ny,nx),jnp.float32)
    moist=(zero,)*6+(zero+.01,zero+.02)
    pressure,faces=hydrostatic_profiles(jnp.full((ny,nx),1000,jnp.float32),moist,
        jnp.ones(nz,jnp.float32),jnp.zeros(nz,jnp.float32),jnp.full(nz,-.25,jnp.float32),
        jnp.float32(100),interpret=True)
    expected=np.empty((nz+1,ny,nx),np.float32);expected[-1]=100
    R=np.float32
    for k in range(nz-1,-1,-1):expected[k]=expected[k+1]-R(R(R(1)+R(R(.01)+R(.02)))*R(1000))*R(-.25)
    np.testing.assert_array_equal(faces,expected)
    np.testing.assert_array_equal(pressure,R(.5)*(expected[:-1]+expected[1:]))


def test_lossless_restart_and_missing_habit_refusal(tmp_path,monkeypatch):
    from gpuwrf.contracts import state as sc
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state,read_wrfrst_state
    from gpuwrf.io.restart_schema import validate_held_state
    monkeypatch.setattr(sc,'_gpu_device',lambda:jax.devices('cpu')[0])
    f=fixture();grid=f._grid(nz=8,ny=3,nx=4);nl=dataclasses.replace(f._namelist(grid),mp_physics=55)
    state=f._state(grid).ensure_conditional_leaves(mp_physics=55)
    state=state.replace(**{name:jnp.full_like(state.qv,(i+1)/128,dtype=jnp.float32)
                          for i,name in enumerate(HISTORY)})
    write_checkpoint(state,nl,grid,2,tmp_path/'restart.pkl')
    a,na,_,step=read_checkpoint(tmp_path/'restart.pkl');assert na.mp_physics==55 and step==2
    write_wrfrst_state(state,grid,nl,tmp_path/'restart.nc',valid_time='2026-07-25_18:00:36',run_start='2026-07-25_18:00:00',step_index=2)
    b,_=read_wrfrst_state(tmp_path/'restart.nc')
    for got in (a,b):
        for name in HISTORY:
            x,y=np.asarray(getattr(state,name)),np.asarray(getattr(got,name))
            assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),name
    fields={name:getattr(state,name) for name in state.__slots__}
    for name in HISTORY:
        with pytest.raises(ValueError,match='restart Jensen '+name):
            validate_held_state({**fields,name:None},nl)
        bad=np.asarray(fields[name]).copy();bad.flat[0]=np.nan
        with pytest.raises(ValueError,match='restart Jensen '+name):
            validate_held_state({**fields,name:bad},nl)


@pytest.mark.parametrize('mode',('full','operational','subset'))
def test_actual_emitted_inventory_and_original_metadata(mode,tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout
    from gpuwrf.config.paths import wrf_root
    f=fixture('test_m7_netcdf_writer.py');state,grid,nl=f.synthetic_case();nl['mp_physics']=55
    expected={leaf:np.full(state.qv.shape,(i+1)/128,np.float32) for i,leaf in enumerate(HISTORY)}
    for leaf,value in expected.items():setattr(state,leaf,value)
    original=(wrf_root()/'Registry/Registry.EM_COMMON').read_text()
    start=datetime(2023,1,15,tzinfo=timezone.utc);authority=f.writer_authority(grid,'d01')
    omitted={'QSNOW','QGRAUP','QNCLOUD','QNCCN','QNSNOW','QNGRAUPEL','REFL_10CM'}
    subset=tuple((*HISTORY.values(),*omitted)) if mode=='subset' else None
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'history.nc',domain='d01',domain_authority=authority,
        valid_time=start,run_start=start,lead_hours=0,full_variable_set=mode!='operational',variable_subset=subset)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority,variable_subset=subset)
    with Dataset(payload.target) as ds:
        assert not omitted&set(ds.variables)
        for leaf,name in HISTORY.items():
            var=ds[name];np.testing.assert_array_equal(var[0],expected[leaf])
            assert var.dtype==np.dtype('float32') and var.dimensions==('Time','bottom_top','south_north','west_east')
            metadata=re.search(r'"'+name+r'"\s+"([^"]*)"\s+"([^"]*)"',original)
            assert metadata is not None,name
            # Retained number-field ABI follows original CPU-WRF history
            # metadata, including the existing category-2/3 number fields.
            units=metadata[2].replace('#',' ') if name in ('QNICE','QNRAIN','QNICE2','QNICE3') else metadata[2]
            assert var.description==metadata[1] and var.units==units,name
            assert var.MemoryOrder=='XYZ' and var.FieldType==104


@pytest.mark.parametrize('leaf,name',tuple(HISTORY.items()))
@pytest.mark.parametrize('bad',(np.nan,np.inf,-np.inf))
def test_nonfinite_history_refused(leaf,name,bad,tmp_path):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    f=fixture('test_m7_netcdf_writer.py');state,grid,nl=f.synthetic_case();nl['mp_physics']=55
    value=np.zeros(state.qv.shape,np.float32);value.flat[0]=bad;setattr(state,leaf,value)
    start=datetime(2023,1,15,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='nonfinite microphysics history field '+name):
        prepare_wrfout_payload(state,grid,nl,tmp_path/'bad.nc',domain='d01',domain_authority=f.writer_authority(grid,'d01'),
            valid_time=start,run_start=start,lead_hours=0,full_variable_set=True,variable_subset=(name,))
