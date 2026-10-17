"""Previous QSNOWXY cannot disappear through any public restart format."""
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
import pickle
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.io.noahmp_glacier_options import GlacierStatic
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io import glacier_schema
from test_p0_5_restart_full_carry import _state,_land,_namelist
from test_s2_glacier_wiring import setup


@pytest.fixture
def initial():
    grid=GridSpec.canary_3km_template();land=_land(grid)
    q=jnp.full((grid.ny,grid.nx),.00123,jnp.float32);land=land.replace(glacier_qsnow=q)
    _,_,static,_,_=setup()
    static=static.replace(ivgtyp=jnp.full(q.shape,15,jnp.int32),
        glacier=GlacierStatic(tuple(range(q.size)),(),q.shape))
    nl=replace(_namelist(grid),noahmp_static=static)
    return grid,nl,initial_operational_carry(_state(grid),noahmp_land=land)


def same(expected,actual):
    a=np.asarray(expected.glacier_qsnow);b=np.asarray(actual.glacier_qsnow)
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def test_runtime_checkpoint_roundtrip_and_missing_rate_refused(initial,tmp_path):
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_land_state
    grid,nl,c=initial;p=tmp_path/'runtime.pkl'
    write_checkpoint(c.state,nl,grid,2,p,land_state=c.noahmp_land)
    same(c.noahmp_land,read_checkpoint_with_land_state(p)[-1])
    payload=pickle.loads(p.read_bytes());payload['noahmp_land_state'].pop('glacier_qsnow')
    payload['noahmp_land_field_order'].remove('glacier_qsnow');payload['glacier_schema_version']=None
    p.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError,match='QSNOWXY'):read_checkpoint_with_land_state(p)


def test_full_restart_roundtrip_and_missing_rate_refused(initial,tmp_path):
    from gpuwrf.io.restart import write_restart,read_restart
    grid,nl,c=initial;p=tmp_path/'full.pkl';write_restart(c,nl,grid,2,p)
    same(c.noahmp_land,read_restart(p)[0].noahmp_land)
    payload=pickle.loads(p.read_bytes());fields=payload['carry']
    fields['noahmp_land_fields'].pop('glacier_qsnow');fields['noahmp_land_field_order'].remove('glacier_qsnow')
    fields['glacier_schema_version']=None;p.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError,match='QSNOWXY'):read_restart(p)


def test_netcdf_roundtrip_and_deleted_variable_refused(initial,tmp_path):
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry,noahmp_land_extension_name
    grid,nl,c=initial;p=tmp_path/'wrfrst.nc'
    write_wrfrst_carry(c,grid,nl,p,valid_time='2026-10-14_00:00:36',run_start='2026-10-14_00:00:00',step_index=2)
    same(c.noahmp_land,read_wrfrst_carry(p,namelist=nl)[0].noahmp_land)
    with Dataset(p,'r+') as d:
        d.renameVariable(noahmp_land_extension_name('glacier_qsnow'),'deleted_glacier_rate')
    with pytest.raises(ValueError,match='missing exact optional carry'):read_wrfrst_carry(p,namelist=nl)


def test_authenticated_store_roundtrip_and_resealed_deleted_group_refused(initial,tmp_path):
    from gpuwrf.runtime.restart_store import RestartStore
    grid,nl,c=initial;store=RestartStore(tmp_path/'store',1000000000,2,0)
    identity={'glacier_config':{'d01':{'required':True,'shape':[grid.ny,grid.nx]}}}
    generation,_=store.save({'d01':c},{'d01':2},{'d01':18.},identity)
    snap,_=store.read(generation,expected_identity=identity);same(c.noahmp_land,snap['carries']['d01'].noahmp_land)
    p=generation/'snapshot.pkl';snap=pickle.loads(p.read_bytes());fields=snap['carries']['d01']
    fields['noahmp_land_fields'].pop('glacier_qsnow');fields['noahmp_land_field_order'].remove('glacier_qsnow')
    fields['glacier_schema_version']=None;p.write_bytes(pickle.dumps(snap))
    m=generation/'manifest.json';manifest=json.loads(m.read_text());manifest['payload_bytes']=p.stat().st_size
    manifest['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='QSNOWXY'):store.read(generation,expected_identity=identity)


@pytest.mark.parametrize('bad',[None,np.zeros((2,3),np.float32),np.full((1,2),np.nan,np.float32),np.zeros((1,2),np.int32)])
def test_glacier_schema_rejects_missing_shape_nonfinite_and_dtype(bad):
    land,_,_,_,_=setup();fields={k:getattr(land,k) for k in glacier_schema.field_order(land)}
    fields['glacier_qsnow']=bad
    with pytest.raises(ValueError,match='QSNOWXY'):
        glacier_schema.validate(fields,tuple(fields),1,active=True,shape=(1,2))
