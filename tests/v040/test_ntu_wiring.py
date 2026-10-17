"""NTU original Registry, caller, history, restart and keyed table boundaries."""
import dataclasses
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.ntu.registry import (
    EXTRA_SCALARS, SURFACE_LEAVES, input_variables, output_mapping, moist_species, scalar_species)

ROOT = Path(__file__).resolve().parents[2]


def test_inventory_is_the_original_wrf_package():
    original=(ROOT/'data/wrf_pristine/WRF/Registry/Registry.EM_COMMON').read_text()
    package=next(line for line in original.splitlines() if re.search(r'mp_physics==56\s',line))
    moist=re.search(r'moist:([^;]+)',package)[1].split(',')
    scalar=re.search(r'scalar:([^;]+)',package)[1].split(',')
    translate={'qnc':'Nc','qnr':'Nr','qni':'Ni','qns':'Ns','qng':'Ng','qnh':'Nh'}
    assert set(moist)==set(moist_species()) and len(moist)==7
    assert {translate.get(n,'ntu_'+n) for n in scalar}==set(scalar_species()) and len(scalar)==21
    for name,variable in input_variables().items():
        declaration=re.search(r'(?ms)^state\s+real\s+'+name[4:]+r'\s+ikjftb\s+scalar.*?"([^"]+)"',original)
        assert declaration is not None and declaration[1]==variable


def fixture(name='test_rrtm_lw_operational_wiring.py'):
    spec = importlib.util.spec_from_file_location('ntu_' + name.replace('.', '_'), ROOT/'tests'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_transport_caller_and_precipitation(monkeypatch):
    from gpuwrf.physics.ntu import adapter, column
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.contracts.physics_registry import state_leaves_for_mp
    f = fixture(); grid = f._grid(nz=8, ny=3, nx=4)
    dormant = f._state(grid)
    assert all(getattr(dormant, n) is None for n in (*EXTRA_SCALARS, *SURFACE_LEAVES))
    state = dormant.ensure_conditional_leaves(mp_physics=56)
    nl = dataclasses.replace(f._namelist(grid), mp_physics=56, ccnty=4, run_physics=True,
                             moist_adv_opt=1, scalar_adv_opt=1)
    assert len(state_leaves_for_mp(56)) == 28
    assert op._advected_scalar_species(nl) == moist_species()
    assert op._nested_number_scalar_species(nl) == scalar_species()
    assert op._diffused_scalar_species(nl) == moist_species() + scalar_species()
    assert set(op._mixing_scalar_species(state, nl)) == set((*moist_species(), *scalar_species()))
    assert set(op._root_flow_only(nl, state)['flow_only']) == {'qh', 'Nc', 'Ns', 'Ng', 'Nh', *EXTRA_SCALARS}
    with_record = state.replace(ntu_fi_bdy=jnp.ones((1, 4, 5, 8, 5), jnp.float32))
    assert 'ntu_fi' not in op._root_flow_only(nl, with_record)['flow_only']
    calls = []
    def step(fields, dt, *, ccnty, xp, output_names):
        calls.append((fields, ccnty))
        result = {n: fields[n] for n in output_mapping()}
        z = jnp.zeros(fields['th'].shape[:-1], jnp.float32)
        result.update(th=fields['th'], rainnc=z+21, snownc=z+7, grapnc=z+5, hailnc=z+6,
                      sr=z+18/21, clodncv=z+1, rainncv=z+2, icencv=z+3, snowncv=z+4,
                      grapncv=z+5, hailncv=z+6)
        return {n: result[n] for n in output_names}
    monkeypatch.setattr(column, 'step', step)
    updated, water = op._apply_post_rk_microphysics(state, nl, return_precipitation=True)
    assert len(calls) == 1 and calls[0][1] == 4
    fields = calls[0][0]
    assert all(x.dtype == jnp.float32 for x in fields.values())
    np.testing.assert_array_equal(fields['w'], jnp.moveaxis(state.w.astype(jnp.float32), 0, -1))
    np.testing.assert_array_equal(fields['pii'], jnp.moveaxis((state.p.astype(jnp.float32)/jnp.float32(100000))**jnp.float32(2/7), 0, -1))
    for name, amount in [('rain_acc',3),('ice_acc',3),('snow_acc',4),('graupel_acc',5),('hail_acc',6),('ntu_rainnc',21),('ntu_snownc',7)]:
        np.testing.assert_array_equal(getattr(updated,name), amount)
    np.testing.assert_array_equal(water['rain'], 2)
    np.testing.assert_array_equal(water['hail'], 6)
    from gpuwrf.runtime.noahmp_precipitation import precipitation_from_step
    held=precipitation_from_step(water,jnp.zeros_like(state.rain_acc),18)
    for value,amount in ((held.prcpnonc,2),(held.prcpsnow,4),(held.prcpgrpl,5),(held.prcphail,6)):
        np.testing.assert_array_equal(value,jnp.float32(amount)/jnp.float32(18))
    with pytest.raises(ValueError, match='initialized prognostics'):
        adapter.ntu_adapter(state.replace(ntu_fi=None), 18, grid)


def test_ccnty_binding_both_pipeline_shared_binder_and_static_aux():
    from gpuwrf.io.namelist_binding import require_namelist_honoured, bind_namelist_options
    from gpuwrf.io.namelist_check import validate_operational_namelist, refuse_single_domain_mp_nested, NotOperationallyWiredError
    f=fixture(); grid=f._grid()
    for ccnty in (1,2,3,4):
        raw={'physics':{'mp_physics':[56], 'ccnty':[ccnty], 'cu_physics':[0]}}
        validate_operational_namelist(raw); require_namelist_honoured(raw, ('d01',))
        nl=bind_namelist_options(f._namelist(grid), raw, 'd01')
        assert nl.mp_physics == 56 and nl.ccnty == ccnty
        leaves, tree=jax.tree.flatten(nl)
        assert jax.tree.unflatten(tree,leaves).ccnty == ccnty
        with pytest.raises(NotOperationallyWiredError): refuse_single_domain_mp_nested(raw,2)
    from gpuwrf.io.namelist_binding import NamelistNotHonouredError
    for invalid in (0,5):
        with pytest.raises(NamelistNotHonouredError,match='ccnty'):
            require_namelist_honoured({'physics':{'mp_physics':[56],'ccnty':[invalid]}},('d01',))


def test_restart_two_formats_and_fail_closed(tmp_path):
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state, read_wrfrst_state
    from gpuwrf.io.restart_schema import validate_held_state
    f=fixture(); grid=f._grid(); nl=dataclasses.replace(f._namelist(grid),mp_physics=56,ccnty=3)
    state=f._state(grid).ensure_conditional_leaves(mp_physics=56)
    names=(*output_mapping().values(),*SURFACE_LEAVES,'hail_acc')
    state=state.replace(**{n:jnp.full_like(getattr(state,n),(i+1)/128) for i,n in enumerate(names)})
    write_checkpoint(state,nl,grid,2,tmp_path/'restart.pkl')
    a, na, _, step=read_checkpoint(tmp_path/'restart.pkl'); assert step==2 and na.ccnty==3
    write_wrfrst_state(state,grid,nl,tmp_path/'restart.nc',valid_time='2026-07-25_18:00:36',run_start='2026-07-25_18:00:00',step_index=2)
    b,_=read_wrfrst_state(tmp_path/'restart.nc')
    from gpuwrf.io.restart import write_restart, read_restart
    from gpuwrf.runtime.operational_state import initial_operational_carry
    write_restart(initial_operational_carry(state),nl,grid,2,tmp_path/'carry.pkl')
    c=read_restart(tmp_path/'carry.pkl')[0].state
    for got in (a,b,c):
        for name in names:
            x,y=np.asarray(getattr(state,name)),np.asarray(getattr(got,name))
            assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),name
    fields={n:getattr(state,n) for n in state.__slots__}
    for name in names:
        with pytest.raises(ValueError, match='restart NTU '+name): validate_held_state({**fields,name:None},nl)
        bad=np.asarray(fields[name]).copy(); bad.flat[0]=np.nan
        with pytest.raises(ValueError, match='restart NTU '+name): validate_held_state({**fields,name:bad},nl)


@pytest.mark.parametrize('mode', ('full','operational','subset'))
def test_written_original_fields_metadata_and_totals(mode,tmp_path):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload,write_prepared_wrfout
    f=fixture('test_m7_netcdf_writer.py'); state,grid,nl=f.synthetic_case(); nl['mp_physics']=56
    expected={leaf:np.full(state.qv.shape,(i+1)/128,np.float32) for i,leaf in enumerate(input_variables())}
    for leaf,value in expected.items(): setattr(state,leaf,value)
    state.ntu_rainnc=np.full(state.qv.shape[-2:],21,np.float32)
    state.ntu_snownc=np.full(state.qv.shape[-2:],7,np.float32)
    state.ntu_sr=np.full(state.qv.shape[-2:],.8,np.float32)
    original=(ROOT/'data/wrf_pristine/WRF/Registry/Registry.EM_COMMON').read_text()
    start=datetime(2023,1,15,tzinfo=timezone.utc); authority=f.writer_authority(grid,'d01')
    subset=(*input_variables().values(),'RAINNC','SNOWNC','SR') if mode=='subset' else None
    payload=prepare_wrfout_payload(state,grid,nl,tmp_path/'history.nc',domain='d01',domain_authority=authority,
        valid_time=start,run_start=start,lead_hours=0,full_variable_set=mode!='operational',variable_subset=subset)
    write_prepared_wrfout(payload,expected_domain='d01',expected_domain_authority=authority,variable_subset=subset)
    with Dataset(payload.target) as ds:
        assert 'QNCCN' not in ds.variables
        for leaf,name in input_variables().items():
            var=ds[name]; np.testing.assert_array_equal(var[0],expected[leaf])
            meta=re.search(r'"'+name+r'"\s+"([^"]*)"\s+"([^"]*)"',original)
            assert var.description==meta[1] and var.units==meta[2]
            assert var.dtype==np.dtype('float32') and var.dimensions==('Time','bottom_top','south_north','west_east')
        np.testing.assert_array_equal(ds['RAINNC'][0],21)
        np.testing.assert_array_equal(ds['SNOWNC'][0],7)
        np.testing.assert_array_equal(ds['SR'][0],np.float32(.8))


def test_actual_table_values_and_ccnty_change_aot_key(monkeypatch):
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    from gpuwrf.runtime.aot_cheap_key import cheap_key
    from gpuwrf.kernels import phys_ntu
    f=fixture(); nl=dataclasses.replace(f._namelist(f._grid()),mp_physics=56,
                                      ra_sw_physics=0,ra_lw_physics=0)
    @jax.jit
    def fn(x,nl): return x+jnp.asarray(phys_ntu.selected_tables(np.float32,nl.ccnty)['dnc0'])
    x=jnp.ones((),jnp.float32)
    before=baked_table_hash(fn,nl); key=cheap_key(fn,(x,nl),{},nl)
    old=phys_ntu.selected_tables
    def changed(dtype,ccnty):
        d=dict(old(dtype,ccnty)); d['dnc0']=d['dnc0']*np.dtype(dtype).type(1.01); return d
    monkeypatch.setattr(phys_ntu,'selected_tables',changed)
    assert baked_table_hash(fn,nl)!=before
    assert cheap_key(fn,(x,nl),{},nl)!=key
    assert cheap_key(fn,(x,dataclasses.replace(nl,ccnty=4)),{},dataclasses.replace(nl,ccnty=4))!=key
