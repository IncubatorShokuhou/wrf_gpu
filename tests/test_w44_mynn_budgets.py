"""MYNN history selection, program identity and checkpoint admission controls.

Sentinel histories test wiring and persistence; physics evidence lives in the
pristine-driver day/night receipts, never in these structural IO tests.
"""
from dataclasses import replace
import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset
import pytest

from gpuwrf.diagnostics import mynn_budget as H
from gpuwrf.io import restart as R,wrfrst_netcdf as N
from gpuwrf.io.mynn_budget_options import validate_options
from gpuwrf.io.namelist_binding import bind_namelist_options,namelist_honour_report
from gpuwrf.io.scheme_catalog import classify_control,SupportStatus
from gpuwrf.runtime.aot_cheap_key import static_config_hash
from gpuwrf.runtime.restart_store import RestartStore
from test_solar_restart import case,_write,_read
from test_online_diagnostics_restart_store import identity as original_identity
from gpuwrf.io.mynn_budget_history import METADATA,pack_history


def selected(case,tke=1,output=1,edmf=1):
    grid,nl,carry=case
    nl=replace(nl,tke_budget=tke,bl_mynn_output=output,bl_mynn_edmf=edmf)
    values=tuple(jnp.full((grid.ny*grid.nx,grid.nz),i+.25,jnp.float32) for i in range(21))
    history=H.from_columns(values,grid.ny,grid.nx,H.configuration(nl)) if tke or output else None
    return grid,nl,carry.replace(mynn_budget=history)


@pytest.mark.parametrize('tke,output,edmf',[(0,0,1),(1,0,1),(0,1,1),(1,1,1),(1,1,0),(0,0,0)])
def test_binding_pytree_and_keys(case,tke,output,edmf):
    _,base,_=case
    raw={'physics':dict(tke_budget=[0,tke],bl_mynn_output=[0,output],bl_mynn_edmf=[1,edmf])}
    bound=bind_namelist_options(base,raw,'d02')
    assert H.configuration(bound)==dict(tke_budget=tke,bl_mynn_output=output,bl_mynn_edmf=edmf)
    leaves,definition=jax.tree.flatten(bound)
    rebuilt=jax.tree.unflatten(definition,leaves)
    assert H.configuration(rebuilt)==H.configuration(bound)
    default=bind_namelist_options(base,{'physics':{}},'d02')
    assert (static_config_hash(bound)==static_config_hash(default))==((tke,output,edmf)==(0,0,1))


def test_named_refusals_before_backend():
    for raw in ({'tke_budget':2},{'bl_mynn_output':-1},{'bl_mynn_edmf':2},
                {'tke_budget':1,'bl_pbl_physics':1},
                {'bl_mynn_edmf':1,'bl_mynn_edmf_mom':0}):
        with pytest.raises(ValueError):validate_options({'physics':raw})
        errors,_=namelist_honour_report({'physics':raw},('d01',))
        assert errors
    assert validate_options({'bl_mynn_edmf':0,'bl_mynn_edmf_mom':0})['bl_mynn_edmf']==0
    for name in ('tke_budget','bl_mynn_output','bl_mynn_edmf'):
        assert classify_control(name,0).status==SupportStatus.IMPLEMENTED
        assert classify_control(name,1).status==SupportStatus.IMPLEMENTED
        assert classify_control(name,2).status==SupportStatus.RECOGNIZED_FAIL_CLOSED


@pytest.mark.parametrize('fmt',['pickle','netcdf'])
@pytest.mark.parametrize('tke,output',[(1,0),(0,1),(1,1)])
def test_exact_checkpoint_history_roundtrip(case,tmp_path,fmt,tke,output):
    grid,nl,carry=selected(case,tke,output)
    path=tmp_path/'checkpoint'
    _write(path,fmt,grid,nl,carry)
    restored=_read(path,fmt,nl)
    assert jax.tree.structure(restored.mynn_budget)==jax.tree.structure(carry.mynn_budget)
    for left,right in zip(jax.tree.leaves(carry.mynn_budget),jax.tree.leaves(restored.mynn_budget),strict=True):
        a,b=np.asarray(left),np.asarray(right)
        assert a.dtype==b.dtype==np.float32 and a.shape==b.shape and a.tobytes()==b.tobytes()


@pytest.mark.parametrize('fmt',['pickle','netcdf'])
def test_legacy_off_checkpoint_refuses_on_request(case,tmp_path,fmt):
    grid,nl,carry=case;path=tmp_path/'legacy_off'
    _write(path,fmt,grid,nl,carry)
    _read(path,fmt,nl)
    with pytest.raises(ValueError,match='MYNN|budget'):_read(path,fmt,replace(nl,tke_budget=1))


def test_missing_schema_or_corrupted_history_refuses_before_restore(case):
    _,nl,carry=selected(case)
    good=R._carry_to_payload(carry,mynn_config=H.configuration(nl))
    H.validate_payload(good,nl)
    for what in ('marker','selection','group','nonfinite'):
        bad=copy.deepcopy(good)
        if what=='marker':bad.pop('mynn_budget_schema_version')
        if what=='selection':bad.pop('mynn_budget_config')
        if what=='group':bad['extra_fields']['mynn_budget']=None
        if what=='nonfinite':
            h=bad['extra_fields']['mynn_budget']
            bad['extra_fields']['mynn_budget']=h._replace(tke=(np.full_like(h.tke[0],np.nan),*h.tke[1:]))
        with pytest.raises(ValueError):R._restore_carry(bad)
    with pytest.raises(ValueError):H.validate_payload(good,replace(nl,bl_mynn_edmf=0))


def test_netcdf_missing_marker_fails(case,tmp_path):
    grid,nl,carry=selected(case);path=tmp_path/'nc'
    _write(path,'netcdf',grid,nl,carry)
    with Dataset(path,'r+') as dataset:dataset.delncattr('GPUWRF_MYNN_BUDGET_SCHEMA_VERSION')
    with pytest.raises(ValueError,match='MYNN|budget'):_read(path,'netcdf',nl)


def test_native_store_binds_selected_program(case,tmp_path):
    _,nl,carry=selected(case)
    ident=original_identity(nl)
    ident['mynn_budget_config']={'d01':H.configuration(nl)}
    store=RestartStore(tmp_path/'store',10**7,2,0)
    generation,_=store.save({'d01':carry},{'d01':17},{'d01':nl.dt_s},ident)
    snapshot,_=store.read(generation,expected_identity=ident)
    got=snapshot['carries']['d01'].mynn_budget
    for a,b in zip(jax.tree.leaves(carry.mynn_budget),jax.tree.leaves(got),strict=True):
        assert np.asarray(a).tobytes()==np.asarray(b).tobytes()


def test_registry_metadata_matches_independent_inventory():
    from pathlib import Path
    record=json.loads((Path(H.__file__).parents[3]/'proofs/w44_budget/receipts/P03_inventory.json').read_text())
    assert set(record['fields'])==set(METADATA)
    for name,metadata in METADATA.items():
        expected=record['fields'][name]
        assert metadata==(expected['description'],expected['units'],expected['staggering'].replace('-',''))


def test_host_writer_uses_all_held_planes_and_original_top(case,tmp_path):
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
    from test_m7_netcdf_writer import writer_authority
    grid,nl,carry=selected(case)
    nl=replace(nl,swint_opt=0)
    fields=H.history_fields(carry.mynn_budget)
    for name in tuple(METADATA)[:5]:fields[name]=fields[name].at[-1].set(0.)
    path=tmp_path/'wrfout'
    write_wrfout_netcdf(carry.state,grid,nl,path,domain='d01',domain_authority=writer_authority(grid,domain='d01'),
        lead_hours=1.,valid_time='2026-06-03_01:00:00',run_start='2026-06-03_00:00:00',
        diagnostics=fields,variable_subset=tuple(METADATA))
    with Dataset(path) as dataset:
        for name,(description,units,stagger) in METADATA.items():
            actual=dataset[name]
            assert actual.description==description and actual.units==units and actual.stagger==stagger
            value=np.asarray(actual[0]);expected=np.asarray(fields[name])
            if stagger=='Z':
                assert actual.dimensions[1]=='bottom_top_stag' and value.shape[0]==grid.nz+1
                assert not np.count_nonzero(value[-1]);value=value[:-1]
            else:assert actual.dimensions[1]=='bottom_top'
            assert value.dtype==np.float32 and value.tobytes()==expected.tobytes()


def test_writer_refuses_dropped_nonfinite_or_disabled_budget(case):
    grid,nl,carry=selected(case)
    good=H.history_fields(carry.mynn_budget)
    for name in tuple(METADATA)[:5]:good[name]=good[name].at[-1].set(0.)
    shape=(grid.nz,grid.ny,grid.nx)
    for name in good:
        for kind in ('missing','nan'):
            bad=dict(good)
            if kind=='missing':bad.pop(name)
            else:bad[name]=jnp.full(shape,jnp.nan,jnp.float32)
            with pytest.raises(ValueError):pack_history(bad,nl,shape,host_value=np.asarray)
    with pytest.raises(ValueError):pack_history(good,replace(nl,tke_budget=0),shape,('qWT',),host_value=np.asarray)
    assert pack_history(None,replace(nl,tke_budget=0,bl_mynn_output=0),shape,host_value=np.asarray)=={}


@pytest.mark.parametrize('stream',['full','subset'])
def test_edmf_off_omits_wrf_package_mynn_dmp_edmf(case,tmp_path,stream):
    # Registry.EM_COMMON:3190 package mynn_dmp_edmf bl_mynn_edmf==1 -> ztop_plume,maxmf,maxwidth:
    # CPU-WRF history has none of the three on a bl_mynn_edmf=0 domain (critic cr-w44-budget F3).
    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
    from gpuwrf.io.mynn_budget_history import DMP_EDMF_HISTORY
    from test_m7_netcdf_writer import writer_authority
    grid,nl,carry=case
    names=(*DMP_EDMF_HISTORY,'QKE','T')
    present={}
    for edmf in (1,0):
        path=tmp_path/f'wrfout_edmf{edmf}'
        write_wrfout_netcdf(carry.state,grid,replace(nl,swint_opt=0,bl_mynn_edmf=edmf),path,domain='d01',
            domain_authority=writer_authority(grid,domain='d01'),lead_hours=1.,valid_time='2026-06-03_01:00:00',
            run_start='2026-06-03_00:00:00',full_variable_set=True,variable_subset=names if stream=='subset' else None)
        with Dataset(path) as dataset:present[edmf]=set(dataset.variables)
    assert set(DMP_EDMF_HISTORY)<=present[1]
    assert not set(DMP_EDMF_HISTORY)&present[0]
    assert present[1]-present[0]==set(DMP_EDMF_HISTORY)
