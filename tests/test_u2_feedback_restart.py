"""Production restart-boundary probes; structural controls, no physics claim."""
from dataclasses import replace
from types import SimpleNamespace
import jax.numpy as jnp
import pytest
from gpuwrf.contracts.state import BaseState
from gpuwrf.runtime.history_accumulators import PackedFields
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.runtime.restart_store import RestartStore
from gpuwrf.io.nesting_options import require_feedback_restart
from gpuwrf.coupling.feedback_aux import FLUXES
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state
from test_online_diagnostics_restart_store import identity

@pytest.fixture
def seed():
    grid=make_ideal_grid(5,4,5)
    state=_state(grid)
    nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,dt_s=54.)
    carry=initial_operational_carry(state)
    base=BaseState(pb=jnp.zeros_like(state.qv),phb=jnp.zeros_like(state.w),
        mub=jnp.zeros_like(state.mu_total),t0=jnp.zeros_like(state.qv),theta_base=jnp.zeros_like(state.qv),
        alb=jnp.ones(state.qv.shape,jnp.float32))
    history=PackedFields.pack({k:jnp.ones(state.t_skin.shape,jnp.float32) for k in ('TH2',*FLUXES)})
    return grid,nl,carry.replace(base_state=base,feedback_history=history)

def two_way_identity(nl):
    first,second=identity(nl),identity(replace(nl,dt_s=18.))
    return {'feedback':True,**{key:{'d01':value['d01'],'d02':second[key]['d01']} for key,value in first.items()}}

def test_public_store_healthy_exact_groups(seed,tmp_path):
    _,nl,carry=seed
    bound=two_way_identity(nl)
    store=RestartStore(tmp_path,10**7,2,0)
    generation,_=store.save({'d01':carry,'d02':carry},{'d01':1,'d02':3},{'d01':54.,'d02':18.},bound)
    restored=store.read(generation,expected_identity=bound)[0]['carries']
    require_feedback_restart(restored)
    assert set(restored['d01'].feedback_history)=={'TH2',*FLUXES}
    assert restored['d01'].base_state.alb.shape==carry.state.qv.shape

@pytest.mark.parametrize('damage',('missing_TH2','alb_shape','alb_dtype','history_shape','history_dtype'))
def test_public_store_refuses_malformed_active_feedback(seed,tmp_path,damage):
    _,nl,carry=seed
    if damage=='missing_TH2':
        history=PackedFields.pack({k:v for k,v in carry.feedback_history.items() if k!='TH2'})
        carry=carry.replace(feedback_history=history)
    elif damage=='alb_shape':carry=carry.replace(base_state=carry.base_state.replace(alb=jnp.ones((1,),jnp.float32)))
    elif damage=='alb_dtype':carry=carry.replace(base_state=carry.base_state.replace(alb=carry.base_state.alb.astype(jnp.float64)))
    elif damage=='history_shape':carry=carry.replace(feedback_history=PackedFields.pack({k:v[:1,:1] for k,v in carry.feedback_history.items()}))
    else:carry=carry.replace(feedback_history=PackedFields.pack({k:v.astype(jnp.float64) for k,v in carry.feedback_history.items()}))
    bound=two_way_identity(nl)
    with pytest.raises(ValueError,match='E78'):
        store=RestartStore(tmp_path,10**7,2,0)
        generation,_=store.save({'d01':carry,'d02':seed[2]},{'d01':1,'d02':3},{'d01':54.,'d02':18.},bound)
        restored=store.read(generation,expected_identity=bound)[0]['carries']
        require_feedback_restart(restored)

@pytest.mark.parametrize('feedback',(False,True))
def test_native_single_domain_resume_with_inactive_feedback(seed,tmp_path,monkeypatch,feedback):
    from tests.v025.restart.test_single_domain_restart import _install_driver,DT
    grid,nl,carry=seed
    # A single domain never creates ALB/feedback_history in _load_domains.
    carry=carry.replace(base_state=None,feedback_history=None)
    pipeline,storage,_=_install_driver(monkeypatch,(grid,nl,carry),[])
    original=pipeline._load_domains
    def loaded(*a,**kw):
        h,b,m,start,dts,c=original(*a,**kw)
        return h,b,{'nesting':{'feedback':feedback,'smooth_option':2,'spec_zone':1}},start,dts,c
    monkeypatch.setattr(pipeline,'_load_domains',loaded)
    bound={'test':'inactive-feedback','output_dir':str((tmp_path/'out').resolve()),'feedback':feedback}
    monkeypatch.setattr(storage,'nested_identity',lambda *a:bound)
    monkeypatch.setattr(pipeline,'_require_native_base_state',lambda *a:None)
    def admitted(*a,**kw):raise RuntimeError('healthy single-domain carry admitted')
    monkeypatch.setattr(pipeline,'_output_cadence_steps_by_domain',admitted)
    store=RestartStore(tmp_path/'store',10**7,2,0)
    generation,_=store.save({'d01':carry},{'d01':1},{'d01':DT},bound,driver_state={'forecast_hours':3.})
    config=pipeline.NestedPipelineConfig(tmp_path/'in',tmp_path/'out',tmp_path/'proof',3.,1,
        feedback=feedback,resume_checkpoint=generation,checkpoint_reserve_bytes=0,aot_prefetch=False)
    with pytest.raises(RuntimeError,match='healthy single-domain carry admitted'):
        pipeline.execute_nested_pipeline(config)


# Keep the critic's eight production-boundary repros above unchanged. Extend
# their structural fixture to cover authenticated reads and both file formats.
import base64
import hashlib
import json
import pickle
import jax
import numpy as np
from gpuwrf.io.feedback_restart import configuration
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io import restart as pickle_io, wrfrst_netcdf as netcdf_io

DAMAGE=('missing_TH2','alb_shape','alb_dtype','history_shape','history_dtype')


def damaged(carry,kind):
    history=carry.feedback_history
    if kind.startswith('missing_'):
        name=kind.removeprefix('missing_')
        names=tuple(n for n in history if n!=name)
        return carry.replace(feedback_history=PackedFields(names,np.stack([np.asarray(history[n]) for n in names])))
    if kind.startswith('alb_'):
        alb=np.ones((1,),np.float32) if kind=='alb_shape' else np.asarray(carry.base_state.alb,np.float64)
        return carry.replace(base_state=carry.base_state.replace(alb=alb))
    data=np.asarray(history.data)
    data=data[...,:1,:1] if kind=='history_shape' else data.astype(np.float64)
    return carry.replace(feedback_history=PackedFields(history.names,data))


def write_file(fmt,path,grid,nl,carry,**kw):
    if fmt=='pickle':return pickle_io.write_restart(carry,nl,grid,1,path,**kw)
    return netcdf_io.write_wrfrst_carry(carry,grid,nl,path,
        valid_time='2026-02-28_00:00:54',run_start='2026-02-28_00:00:00',step_index=1,**kw)


def read_file(fmt,path,**kw):
    return pickle_io.read_restart(path,**kw) if fmt=='pickle' else netcdf_io.read_wrfrst_carry(path,**kw)


@pytest.mark.parametrize('fmt',('pickle','netcdf'))
@pytest.mark.parametrize('kind',DAMAGE)
def test_advertised_formats_refuse_malformed_active_write(seed,tmp_path,fmt,kind):
    grid,nl,carry=seed
    with pytest.raises(ValueError,match='E78'):
        write_file(fmt,tmp_path/f'restart.{fmt}',grid,nl,damaged(carry,kind))


def corrupt_file(fmt,path,carry,kind):
    if fmt=='pickle':
        payload=pickle.loads(path.read_bytes())
        if kind=='schema_version':payload['carry']['feedback_schema']['version']=0
        elif kind=='schema_missing':payload['carry'].pop('feedback_schema')
        else:
            broken=damaged(carry,kind)
            group='base_state' if kind.startswith('alb_') else 'feedback_history'
            payload['carry']['extra_fields'][group]=jax.tree.map(np.asarray,getattr(broken,group))
        path.write_bytes(pickle.dumps(payload))
        return
    with Dataset(path,'a') as ds:
        if kind=='schema_version':
            config=json.loads(ds.GPUWRF_FEEDBACK_SCHEMA);config['version']=0
            ds.GPUWRF_FEEDBACK_SCHEMA=json.dumps(config)
            return
        if kind=='schema_missing':ds.delncattr('GPUWRF_FEEDBACK_SCHEMA');return
        broken=damaged(carry,kind)
        group='base_state' if kind.startswith('alb_') else 'feedback_history'
        leaves,structure=jax.tree.flatten(getattr(broken,group))
        index=5 if group=='base_state' else 0
        ds.setncattr('GPUWRF_TREE_'+group.upper(),base64.b64encode(pickle.dumps(structure)).decode())
        name=netcdf_io._optional_variable_name(group,f'leaf_{index:03d}')
        ds.renameVariable(name,name+'_original')
        data=np.asarray(leaves[index]);dims=[]
        for axis,size in enumerate(data.shape):
            dim=f'damaged_feedback_{axis}';ds.createDimension(dim,size);dims.append(dim)
        ds.createVariable(name,data.dtype,tuple(dims))[...]=data


@pytest.mark.parametrize('fmt',('pickle','netcdf'))
@pytest.mark.parametrize('kind',(*DAMAGE,'schema_version','schema_missing'))
def test_advertised_formats_refuse_before_device_restoration(seed,tmp_path,monkeypatch,fmt,kind):
    grid,nl,carry=seed;path=tmp_path/f'restart.{fmt}'
    write_file(fmt,path,grid,nl,carry);corrupt_file(fmt,path,carry,kind)
    def device_forbidden(*a,**kw):raise AssertionError('device restoration happened before schema refusal')
    if fmt=='pickle':monkeypatch.setattr(pickle_io,'_restore_carry',device_forbidden)
    else:monkeypatch.setattr(netcdf_io.jnp,'asarray',device_forbidden)
    with pytest.raises(ValueError,match='E78'):read_file(fmt,path)


@pytest.mark.parametrize('kind',(*DAMAGE,'schema_version','schema_missing','missing_SHDMAX'))
def test_authenticated_native_read_refuses_before_device(seed,tmp_path,monkeypatch,kind):
    import gpuwrf.runtime.restart_store as store_io
    grid,nl,carry=seed
    fields=dict(carry.feedback_history.items());fields['SHDMAX']=jnp.ones(carry.state.t_skin.shape,jnp.float32)
    carry=carry.replace(feedback_history=PackedFields.pack(fields))
    schema=configuration(carry.base_state,carry.feedback_history,carry.state.qv.shape,carry.state.t_skin.shape)
    bound=two_way_identity(nl);bound['feedback_schema']={'d01':schema,'d02':schema}
    store=RestartStore(tmp_path,10**7,2,0)
    generation,_=store.save({'d01':carry,'d02':carry},{'d01':1,'d02':3},{'d01':54.,'d02':18.},bound)
    path=generation/'snapshot.pkl';snapshot=pickle.loads(path.read_bytes());payload=snapshot['carries']['d01']
    if kind=='schema_version':payload['feedback_schema']['version']=0
    elif kind=='schema_missing':payload.pop('feedback_schema')
    else:
        broken=damaged(carry,kind);group='base_state' if kind.startswith('alb_') else 'feedback_history'
        payload['extra_fields'][group]=jax.tree.map(np.asarray,getattr(broken,group))
    data=pickle.dumps(snapshot);path.write_bytes(data)
    manifest_path=generation/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest.update(payload_bytes=len(data),sha256=hashlib.sha256(data).hexdigest());manifest_path.write_text(json.dumps(manifest))
    def device_forbidden(*a,**kw):raise AssertionError('device restoration happened before schema refusal')
    monkeypatch.setattr(store_io,'_restore_carry',device_forbidden)
    with pytest.raises(ValueError,match='E78'):store.read(generation,expected_identity=bound)


@pytest.mark.parametrize('fmt',('pickle','netcdf'))
def test_complete_initialized_field_binding_and_exact_roundtrip(seed,tmp_path,fmt):
    grid,nl,carry=seed
    fields=dict(carry.feedback_history.items());fields['SHDMAX']=jnp.full(carry.state.t_skin.shape,.875,jnp.float32)
    carry=carry.replace(feedback_history=PackedFields.pack(fields))
    schema=configuration(carry.base_state,carry.feedback_history,carry.state.qv.shape,carry.state.t_skin.shape)
    with pytest.raises(ValueError,match='E78'):
        write_file(fmt,tmp_path/'missing',grid,nl,damaged(carry,'missing_SHDMAX'),feedback_schema=schema)
    path=tmp_path/'healthy';write_file(fmt,path,grid,nl,carry,feedback_schema=schema)
    restored=read_file(fmt,path,feedback_schema=schema)[0]
    before,bt=jax.tree.flatten(carry);after,at=jax.tree.flatten(restored)
    assert bt==at
    for a,b in zip(before,after,strict=True):
        a,b=np.asarray(a),np.asarray(b)
        assert (a.shape,a.dtype,a.tobytes())==(b.shape,b.dtype,b.tobytes())
    wrong=dict(schema,history_fields=[n for n in schema['history_fields'] if n!='SHDMAX'])
    with pytest.raises(ValueError,match='E78'):read_file(fmt,path,feedback_schema=wrong)


@pytest.mark.parametrize('fmt',('pickle','netcdf'))
def test_legacy_inactive_format_has_no_feedback_schema_requirement(seed,tmp_path,fmt):
    grid,nl,carry=seed;carry=carry.replace(base_state=carry.base_state.replace(alb=None),feedback_history=None)
    path=tmp_path/'inactive';write_file(fmt,path,grid,nl,carry)
    restored=read_file(fmt,path)[0]
    assert restored.base_state.alb is None and restored.feedback_history is None
