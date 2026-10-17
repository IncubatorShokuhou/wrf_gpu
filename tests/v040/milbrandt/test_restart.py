"""Public Milbrandt restart APIs: exact IO and authenticated malformed-carry rejection.

Structural evidence only; these patterned small-grid fields are not a forecast.
"""
from dataclasses import replace
import json
import hashlib
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.test_online_diagnostics import case


@pytest.fixture
def mp9case(case):
    grid,nl,carry=case
    nl=replace(nl,mp_physics=9,nwp_diagnostics=0,do_radar_ref=0,prec_acc_dt=0.)
    state=carry.state.ensure_conditional_leaves(mp_physics=9)
    updates={n:jnp.full(state.qv.shape,(i+1)*1.2345,jnp.float32)
             for i,n in enumerate(('qh','Nc','Nr','Ni','Ns','Ng','Nh'))}
    updates['hail_acc']=jnp.full(state.t_skin.shape,9.8765,jnp.float32)
    state=state.replace(_cast=False,**updates)
    carry=carry.replace(state=state,nwp_diagnostics=None,precipitation_buckets=None,
        radar_reflectivity=jnp.arange(state.qv.size,dtype=jnp.float32).reshape(state.qv.shape))
    return grid,nl,carry


def identity(nl):
    from tests.test_online_diagnostics_restart_store import identity as base
    return {**base(nl),'mp_re_config':{'d01':{'mp_physics':9,'use_mp_re':0}}}


def exact(a,b):
    assert jax.tree.structure(a)==jax.tree.structure(b)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        x,y=np.asarray(x),np.asarray(y)
        assert (x.dtype,x.shape,x.tobytes())==(y.dtype,y.shape,y.tobytes())


def save(fmt,path,grid,nl,carry):
    if fmt=='pickle':
        from gpuwrf.io.restart import write_restart
        return write_restart(carry,nl,grid,7,path)
    if fmt=='netcdf':
        from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry
        return write_wrfrst_carry(carry,grid,nl,path,valid_time='2026-01-01_00:02:06',
                                 run_start='2026-01-01_00:00:00',step_index=7)
    from gpuwrf.runtime.restart_store import RestartStore
    return RestartStore(path,10**7,2,0).save({'d01':carry},{'d01':7},{'d01':nl.dt_s},identity(nl))[0]


@pytest.mark.parametrize('fmt',['pickle','netcdf','native'])
def test_public_restart_roundtrip_exact(mp9case,tmp_path,fmt):
    grid,nl,carry=mp9case
    path=save(fmt,tmp_path/'restart',grid,nl,carry)
    if fmt=='pickle':
        from gpuwrf.io.restart import read_restart
        restored=read_restart(path)[0]
    elif fmt=='netcdf':
        from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry
        restored=read_wrfrst_carry(path,namelist=nl)[0]
    else:
        from gpuwrf.runtime.restart_store import RestartStore
        restored=RestartStore(tmp_path/'restart',10**7,2,0).read(path,expected_identity=identity(nl))[0]['carries']['d01']
    exact(carry,restored)


@pytest.mark.parametrize('fmt',['pickle','netcdf','native'])
@pytest.mark.parametrize('bad',['qh','Nh','hail_acc','radar'])
def test_public_restart_save_rejects_missing_active_carry(mp9case,tmp_path,fmt,bad):
    grid,nl,carry=mp9case
    damaged=(carry.replace(radar_reflectivity=None) if bad=='radar' else
             carry.replace(state=carry.state.replace(**{bad:None})))
    with pytest.raises(ValueError,match='E78|schema|missing'):
        save(fmt,tmp_path/'broken',grid,nl,damaged)


@pytest.mark.parametrize('bad',['qh','Nh','hail_acc','radar'])
def test_native_read_rejects_rehashed_missing_carry(mp9case,tmp_path,bad):
    from gpuwrf.runtime.restart_store import RestartStore
    grid,nl,carry=mp9case
    path=save('native',tmp_path/'native',grid,nl,carry)
    payload=path/'snapshot.pkl';blob=pickle.loads(payload.read_bytes());c=blob['carries']['d01']
    if bad=='radar':c['extra_fields']['radar_reflectivity']=None
    else:
        c['state_fields'].pop(bad);c['state_field_order'].remove(bad)
    raw=pickle.dumps(blob);payload.write_bytes(raw)
    mpath=path/'manifest.json';m=json.loads(mpath.read_text())
    m.update(payload_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='E78'):
        RestartStore(tmp_path/'native',10**7,2,0).read(path,device=False)


def test_state_checkpoint_requires_held_radar(mp9case,tmp_path):
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    grid,nl,carry=mp9case
    with pytest.raises(ValueError,match='full runtime_state'):
        write_checkpoint(carry.state,nl,grid,7,tmp_path/'missing.pkl')
    write_checkpoint(carry.state,nl,grid,7,tmp_path/'full.pkl',runtime_state=carry)
    state,_,_,step,runtime=read_checkpoint_with_runtime_state(tmp_path/'full.pkl')
    assert step==7
    exact(state,carry.state);exact(runtime,carry)
