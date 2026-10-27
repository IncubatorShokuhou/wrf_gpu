"""Goddard admission, cadence, conditional radii and restart are deletion-sensitive."""
from dataclasses import replace
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import goddard_radiation as goddard
from gpuwrf.io.goddard_config import validate_geometry
from gpuwrf.io.namelist_check import UnsupportedSchemeError,validate_operational_namelist
from gpuwrf.io.scheme_catalog import SupportStatus,classify_scheme
from gpuwrf.runtime import operational_mode as om
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv('GPUWRF_MYNN_SFC_WSPD','0')
    grid=make_ideal_grid(4,3,5)
    state=_state(grid)
    from gpuwrf.coupling.physics_couplers import RRTMGRadiationStatic
    z=jnp.zeros((3,4),jnp.float32)
    static=RRTMGRadiationStatic(jnp.full_like(z,28),jnp.full_like(z,-16),z,z,z,z,jnp.ones_like(z))
    nl=om.OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,use_noahmp=False,
        sf_surface_physics=0,bl_pbl_physics=0,sf_sfclay_physics=0,cu_physics=0,
        ra_lw_physics=5,ra_sw_physics=5,radiation_static=static)
    carry=om._initial_carry_for_run(state,nl)
    return grid,nl,carry


def test_goddard_admitted_and_honestly_qualified():
    validate_operational_namelist({'physics':{'ra_lw_physics':[5],'ra_sw_physics':[5],'mp_physics':[8],'use_mp_re':[1]}})
    for key in ('ra_lw_physics','ra_sw_physics'):
        result=classify_scheme(key,5)
        assert result.status is SupportStatus.IMPLEMENTED
        assert 'CPU-oracle-qualified' in result.reason
        assert 'GPU/coupled-forecast qualification pending' in result.reason
        assert 'Experimental' in result.reason and 'taux' in result.reason


def test_both_real_loader_binding_seams_and_honour(monkeypatch,tmp_path):
    from test_namelist_binding import _load_nested,_build_daily_case
    from gpuwrf.io.namelist_binding import namelist_honour_report
    nested=tmp_path/'nested';nested.mkdir()
    bundles=_load_nested(monkeypatch,nested,{'ra_lw_physics':'5,5,5','ra_sw_physics':'5,5,5'})
    for bundle in bundles.values():
        assert (bundle.namelist.ra_lw_physics,bundle.namelist.ra_sw_physics,bundle.namelist.mp_physics)==(5,5,8)
    daily=tmp_path/'daily';daily.mkdir()
    nl=_build_daily_case(monkeypatch,daily,' mp_physics=8,\n ra_lw_physics=5,\n ra_sw_physics=5,\n radt=10,\n use_mp_re=1,\n')
    assert (nl.ra_lw_physics,nl.ra_sw_physics,nl.mp_physics)==(5,5,8)
    errors,_warnings=namelist_honour_report({'physics':{'mp_physics':[8],'ra_lw_physics':[5],'ra_sw_physics':[5],'radt':[10],'use_mp_re':[1]}},('d01',))
    assert errors==[]


@pytest.mark.parametrize('key,value',[('ra_lw_physics',4),('ra_sw_physics',4),('mp_physics',6),('mp_physics',28),
    ('use_mp_re',0),('gsfcrad_gocart_coupling',1),('aer_ra_feedback',1),('topo_shading',1),('slope_rad',1),('swint_opt',1)])
def test_unsupported_caller_fails_before_execution(key,value):
    options={'ra_lw_physics':5,'ra_sw_physics':5,'mp_physics':8,'use_mp_re':1,key:value}
    with pytest.raises(UnsupportedSchemeError):validate_operational_namelist({'physics':options})


def test_original_equator_oob_is_explicitly_refused():
    with pytest.raises(ValueError,match='equator'):
        validate_geometry(SimpleNamespace(radiation_static=SimpleNamespace(xlat_deg=np.array([[0]],np.float32))))
    from gpuwrf.physics.ra_goddard_common import sounding_interp
    assert np.isnan(np.asarray(jax.jit(sounding_interp)(jnp.float32(0),jnp.int32(100)))).any()


def test_seeded_radii_and_held_cadence(case,monkeypatch):
    _grid,nl,carry=case
    assert type(carry.radiation_diagnostics) is goddard.GoddardRadiationDiagnostics
    assert all(getattr(carry.state,k) is not None for k in ('re_cloud','re_ice','re_snow'))
    calls=[]
    def fake(state,grid,**kw):
        calls.append(kw)
        return jnp.ones_like(carry.rthraten),carry.radiation_diagnostics._replace(sw_heating_negative_count=jnp.uint32(2))
    monkeypatch.setattr(goddard,'paired_radiation',fake)
    fresh=om._refresh_rrtmg_driver(carry,nl,0.,True,None)
    held=om._refresh_rrtmg_driver(fresh,nl,18.,False,None)
    assert len(calls)==1 and calls[0]['solar_lead_seconds']==.5*nl.dt_s*nl.radiation_cadence_steps
    assert fresh.rthraten is held.rthraten
    assert fresh.radiation_diagnostics is held.radiation_diagnostics
    assert int(held.radiation_diagnostics.sw_heating_negative_count)==2


def test_off_optional_common_diagnostics_are_absent(case):
    _grid,nl,carry=case
    assert nl.rad_diag==0
    diag=carry.radiation_diagnostics
    assert diag.rad_diag is None and diag.urban_radiation is None
    for name in goddard.unproduced_optional_fields():
        assert getattr(diag,name) is None,name
    assert diag.cloud_fraction.shape==carry.state.qv.shape
    for name in goddard._COMMON_SURFACE_FIELDS:
        value=np.asarray(getattr(diag,name))
        assert value.shape==carry.state.t_skin.shape and value.dtype==np.float32
        assert np.isfinite(value).all() and not value.any()


@pytest.mark.parametrize('future',['shadow_height_m','future_optional_diagnostic'])
def test_future_optional_common_fields_are_none_before_allocation(case,monkeypatch,future):
    from collections import namedtuple
    from gpuwrf.coupling import physics_couplers as pc
    _grid,_nl,carry=case
    base=pc.RRTMGRadiationDiagnostics
    # A previously future field can become a real optional common field.
    # Exercise its absence directly rather than constructing a duplicate name.
    extended=base if future in base._fields else namedtuple(
        base.__name__,(*base._fields,future),defaults=(*base._field_defaults.values(),None))
    extra=tuple(name for name in goddard.GoddardRadiationDiagnostics._fields if name not in base._fields)
    monkeypatch.setattr(pc,'RRTMGRadiationDiagnostics',extended)
    monkeypatch.setattr(goddard,'GoddardRadiationDiagnostics',namedtuple(
        'GoddardRadiationDiagnostics',(*extended._fields,*extra),defaults=(None,)))
    original=jnp.zeros_like;allocations=[]
    def counted(*args,**kwargs):
        allocations.append(np.shape(args[0]))
        return original(*args,**kwargs)
    monkeypatch.setattr(jnp,'zeros_like',counted)
    diag=goddard.initial_diagnostics(carry.state)
    assert future in goddard.unproduced_optional_fields()
    assert getattr(diag,future) is None and diag.rad_diag is None
    assert len(allocations)==len(goddard._COMMON_SURFACE_FIELDS)+len(goddard.HISTORY_FIELDS)+1


def test_goddard_refuses_every_foreign_optional_plane(case):
    from gpuwrf.io.restart_schema import validate_goddard_radiation,GODDARD_RADIATION_SCHEMA_VERSION
    _grid,nl,carry=case
    for name in goddard.unproduced_optional_fields():
        group=carry.radiation_diagnostics._replace(**{name:jnp.zeros_like(carry.state.t_skin)})
        with pytest.raises(ValueError,match='Goddard.*optional.*E78'):
            validate_goddard_radiation(group,nl,GODDARD_RADIATION_SCHEMA_VERSION)


def exact(a,b):
    aa,at=jax.tree.flatten(a);bb,bt=jax.tree.flatten(b)
    assert at==bt
    for x,y in zip(aa,bb,strict=True):
        x,y=np.asarray(x),np.asarray(y)
        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()


@pytest.mark.parametrize('fmt',['pickle','netcdf'])
def test_radiation_restart_every_byte_and_missing_marker_refused(case,tmp_path,fmt):
    from gpuwrf.io import restart as rp,wrfrst_netcdf as nc
    from gpuwrf.io.netcdf_lock import Dataset
    grid,nl,carry=case
    carry=carry.replace(radiation_diagnostics=carry.radiation_diagnostics._replace(sw_heating_negative_count=jnp.uint32(7)))
    path=tmp_path/'restart'
    if fmt=='pickle':
        rp.write_restart(carry,nl,grid,2,path)
        exact(carry,rp.read_restart(path)[0])
        payload=pickle.loads(path.read_bytes());payload['carry'].pop('goddard_radiation_schema_version')
        path.write_bytes(pickle.dumps(payload))
        with pytest.raises(ValueError,match='Goddard'):rp.read_restart(path)
    else:
        nc.write_wrfrst_carry(carry,grid,nl,path,valid_time='2026-06-03_00:30:00',run_start='2026-06-03_00:00:00',step_index=2)
        exact(carry,nc.read_wrfrst_carry(path,namelist=nl)[0])
        with Dataset(path,'a') as ds:ds.delncattr('GPUWRF_GODDARD_RADIATION_SCHEMA_VERSION')
        with pytest.raises(ValueError,match='Goddard'):nc.read_wrfrst_carry(path,namelist=nl)


def test_store_restart_and_missing_counter_refused(case,tmp_path):
    from gpuwrf.runtime.restart_store import RestartStore
    from gpuwrf.io.restart_schema import validate_goddard_radiation, GODDARD_RADIATION_SCHEMA_VERSION
    _grid,nl,carry=case
    store=RestartStore(tmp_path/'store',10**8,2,0)
    generation,_=store.save({'d01':carry},{'d01':2},{'d01':18.},{'scheme':'goddard'})
    exact(carry,store.read(generation)[0]['carries']['d01'])
    with pytest.raises(ValueError,match='Goddard'):
        validate_goddard_radiation(carry.radiation_diagnostics._replace(sw_heating_negative_count=None),nl,GODDARD_RADIATION_SCHEMA_VERSION)


@pytest.mark.parametrize('field,value',[
    ('glw',np.zeros((1,),np.float32)),('swddir',np.zeros((3,4),np.float64)),
    ('cloud_fraction',np.zeros((1,3,4),np.float32)),
    ('shadow_mask',np.zeros((3,4),np.float32)),('sw_heating_negative_count',np.int32(0)),
    ('cod2d_out',np.full((3,4),np.nan,np.float32)),
    ('sw_clear_sfc_down',np.zeros((3,4),np.float32)),
])
@pytest.mark.parametrize('fmt',['pickle','netcdf','store'])
def test_malformed_held_schema_rejected_at_write(case,tmp_path,field,value,fmt):
    from gpuwrf.io import restart as rp,wrfrst_netcdf as nc
    from gpuwrf.runtime.restart_store import RestartStore
    grid,nl,carry=case
    carry=carry.replace(radiation_diagnostics=carry.radiation_diagnostics._replace(**{field:value}))
    with pytest.raises(ValueError,match='Goddard'):
        if fmt=='pickle':rp.write_restart(carry,nl,grid,2,tmp_path/'bad')
        elif fmt=='netcdf':nc.write_wrfrst_carry(carry,grid,nl,tmp_path/'bad',valid_time='2026-06-03_00:30:00',run_start='2026-06-03_00:00:00',step_index=2)
        else:RestartStore(tmp_path/'store',10**8,2,0).save({'d01':carry},{'d01':2},{'d01':18.},{'scheme':'goddard'})


@pytest.mark.parametrize('fmt',['pickle','store'])
@pytest.mark.parametrize('field,value',[
    ('glw',np.zeros((1,),np.float32)),('swddir',np.zeros((3,4),np.float64)),
    ('cloud_fraction',np.zeros((1,3,4),np.float32)),
    ('shadow_mask',np.zeros((3,4),np.float32)),('sw_heating_negative_count',np.int32(0)),
    ('cod2d_out',np.full((3,4),np.nan,np.float32)),('swddni',None),
])
def test_authenticated_malformed_payload_rejected_at_read(case,tmp_path,fmt,field,value):
    from gpuwrf.io import restart as rp
    from gpuwrf.runtime.restart_store import RestartStore
    from test_online_diagnostics_restart_store import rewrite
    grid,nl,carry=case
    def damage(payload):
        group=payload['extra_fields']['radiation_diagnostics']
        payload['extra_fields']['radiation_diagnostics']=group._replace(**{field:value})
    if fmt=='pickle':
        path=tmp_path/'restart';rp.write_restart(carry,nl,grid,2,path)
        payload=pickle.loads(path.read_bytes());damage(payload['carry']);path.write_bytes(pickle.dumps(payload))
        with pytest.raises(ValueError,match='Goddard'):rp.read_restart(path)
    else:
        store=RestartStore(tmp_path/'store',10**8,2,0)
        generation,_=store.save({'d01':carry},{'d01':2},{'d01':18.},{'scheme':'goddard'})
        rewrite(generation,lambda payload,manifest:damage(payload))
        with pytest.raises(ValueError,match='Goddard'):store.read(generation)


@pytest.mark.parametrize('damage',['dtype','shape','nonfinite','group_marker','config_off'])
def test_netcdf_stored_schema_checked_before_device_conversion(case,tmp_path,damage):
    from gpuwrf.io import wrfrst_netcdf as nc
    from gpuwrf.io.netcdf_lock import Dataset
    grid,nl,carry=case;healthy=tmp_path/'healthy';bad=tmp_path/'bad'
    nc.write_wrfrst_carry(carry,grid,nl,healthy,valid_time='2026-06-03_00:30:00',run_start='2026-06-03_00:00:00',step_index=2)
    paths=jax.tree.leaves_with_path(carry.radiation_diagnostics)
    index=next(i for i,(p,_) in enumerate(paths) if p[0].name=='glw')
    target=nc._optional_variable_name('radiation_diagnostics',f'leaf_{index:03d}')
    with Dataset(healthy) as src,Dataset(bad,'w') as dst:
        import json
        global_attrs={k:src.getncattr(k) for k in src.ncattrs()}
        if damage=='group_marker':
            for k in ('GPUWRF_GODDARD_RADIATION_SCHEMA_VERSION','GPUWRF_GODDARD_RADIATION_CONFIG','GPUWRF_TREE_RADIATION_DIAGNOSTICS'):
                global_attrs.pop(k)
            for attr,value in (('GPUWRF_OPTIONAL_CARRY_KIND','none'),('GPUWRF_OPTIONAL_CARRY_FIELD_ORDER',[])):
                fields=json.loads(global_attrs[attr]);fields['radiation_diagnostics']=value;global_attrs[attr]=json.dumps(fields)
        if damage=='config_off':
            for k in ('GPUWRF_MP_RE_CONFIG','GPUWRF_GODDARD_RADIATION_CONFIG'):
                config=json.loads(global_attrs[k]);config.update(ra_lw_physics=4,ra_sw_physics=4);global_attrs[k]=json.dumps(config)
        dst.setncatts(global_attrs)
        for k,dim in src.dimensions.items():dst.createDimension(k,None if dim.isunlimited() else len(dim))
        if damage=='shape':dst.createDimension('bad_goddard',1)
        for k,var in src.variables.items():
            if damage=='group_marker' and k.startswith('GPUWRF_RADIATION_DIAGNOSTICS_'):continue
            dtype=np.float64 if k==target and damage=='dtype' else var.dtype
            dims=('bad_goddard',) if k==target and damage=='shape' else var.dimensions
            attrs={k:var.getncattr(k) for k in var.ncattrs()};fill=attrs.pop('_FillValue',None)
            dest=dst.createVariable(k,dtype,dims,fill_value=fill);dest.setncatts(attrs)
            dest[...]=(np.zeros(1,np.float32) if k==target and damage=='shape' else
                       np.full(var.shape,np.nan,np.float32) if k==target and damage=='nonfinite' else var[...])
    with pytest.raises(ValueError,match='Goddard'):nc.read_wrfrst_carry(bad)


@pytest.mark.parametrize('fmt',['pickle','store'])
@pytest.mark.parametrize('damage',['group_marker','config_off'])
def test_saved_selection_refuses_deleted_group_or_mismatched_pair(case,tmp_path,fmt,damage):
    from gpuwrf.io import restart as rp
    from gpuwrf.runtime.restart_store import RestartStore
    from gpuwrf.physics.rrtmg_mp_re import mp_re_config
    from test_online_diagnostics_restart_store import rewrite
    grid,nl,carry=case
    def drop(payload):
        payload['extra_fields']['radiation_diagnostics']=None
        payload.pop('goddard_radiation_schema_version')
    if fmt=='pickle':
        path=tmp_path/'restart';rp.write_restart(carry,nl,grid,2,path);payload=pickle.loads(path.read_bytes())
        if damage=='group_marker':drop(payload['carry'])
        else:payload['namelist']=replace(payload['namelist'],ra_lw_physics=4,ra_sw_physics=4)
        path.write_bytes(pickle.dumps(payload))
        with pytest.raises(ValueError,match='Goddard'):rp.read_restart(path)
    else:
        store=RestartStore(tmp_path/'store',10**8,2,0)
        config={'mp_re_config':{'d01':mp_re_config(nl)}}
        generation,_=store.save({'d01':carry},{'d01':2},{'d01':18.},config)
        exact(carry,store.read(generation)[0]['carries']['d01'])
        def mutate(payload,manifest):
            if damage=='group_marker':drop(payload)
            else:manifest['identity']['mp_re_config']['d01'].update(ra_lw_physics=4,ra_sw_physics=4)
        rewrite(generation,mutate)
        with pytest.raises(ValueError,match='Goddard'):store.read(generation)
