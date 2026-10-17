"""QNSE admission and lossless held-memory restart; no forecast-fidelity claim."""
from dataclasses import replace
import hashlib
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.qnse_adapter import initial_qnse_carry
from gpuwrf.coupling.physics_dispatch import resolve_physics_suite
from gpuwrf.io.namelist_binding import bound_options
from gpuwrf.io.restart_schema import QNSE_SCHEMA_VERSION,validate_qnse
from gpuwrf.io import restart as rp,wrfrst_netcdf as nc
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.runtime import checkpoint as cp
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_state import initial_operational_carry,seed_history_diagnostics
from gpuwrf.runtime.restart_store import RestartStore
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv('GPUWRF_MYNN_SFC_WSPD','0')
    from gpuwrf.physics import rrtmg_mp_re
    monkeypatch.setattr(rrtmg_mp_re,'_MP_RE',False)
    grid=make_ideal_grid(4,3,5);state=_state(grid)
    nl=OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,bl_pbl_physics=4,sf_sfclay_physics=4)
    q=initial_qnse_carry(state)
    # Distinct nonzero memory detects accidental cold starts or leaf swaps.
    q=type(q)(*(jnp.arange(a.size,dtype=jnp.float32).reshape(a.shape)/1024+i for i,a in enumerate(q,1)))
    history=seed_history_diagnostics(state,qnsesfc=True)
    history=history._replace(**{key:jnp.full(state.t_skin.shape,i+1.,jnp.float32)
        for i,key in enumerate(history._fields) if getattr(history,key) is not None})
    return grid,nl,initial_operational_carry(state).replace(qnse=q,history_diagnostics=history)


def exact(a,b):
    aa,at=jax.tree.flatten(a);bb,bt=jax.tree.flatten(b)
    assert at==bt
    for x,y in zip(aa,bb,strict=True):
        x,y=np.asarray(x),np.asarray(y)
        assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()


def identity(nl):
    return {'qnse_config':{'d01':{key:int(getattr(nl,key)) for key in
                                  ('bl_pbl_physics','sf_sfclay_physics','mfshconv')}}}


def save_read(path,fmt,grid,nl,carry,*,save=True):
    if fmt=='pickle':
        if save:rp.write_restart(carry,nl,grid,2,path)
        return rp.read_restart(path)[0]
    if fmt=='netcdf':
        if save:nc.write_wrfrst_carry(carry,grid,nl,path,valid_time='2026-06-03_00:30:00',run_start='2026-06-03_00:00:00',step_index=2)
        return nc.read_wrfrst_carry(path,namelist=nl)[0]
    if fmt=='checkpoint':
        if save:cp.write_checkpoint(carry.state,nl,grid,2,path,runtime_state=carry)
        return cp.read_checkpoint_with_runtime_state(path)[4]
    store=RestartStore(path,10**8,2,0)
    generation=(store.save({'d01':carry},{'d01':2},{'d01':54.},identity(nl))[0]
                if save else store.latest(expected_identity=identity(nl)))
    return store.read(generation,expected_identity=identity(nl))[0]['carries']['d01']


@pytest.mark.parametrize('mf',(0,1))
@pytest.mark.parametrize('fmt',('pickle','netcdf','checkpoint','store'))
def test_qnse_all_public_restarts_keep_every_byte(case,tmp_path,fmt,mf):
    grid,nl,carry=case;nl=replace(nl,mfshconv=mf)
    exact(carry,save_read(tmp_path/fmt,fmt,grid,nl,carry))


@pytest.mark.parametrize('damage',('group','leaf','dtype','shape','nonfinite'))
@pytest.mark.parametrize('fmt',('pickle','netcdf','checkpoint','store'))
def test_qnse_public_save_refuses_incomplete_memory(case,tmp_path,fmt,damage):
    grid,nl,carry=case;q=carry.qnse
    q={'group':None,'leaf':q._replace(rc_mf=None),'dtype':q._replace(akhs=q.akhs.astype(jnp.float64)),
       'shape':q._replace(rc_mf=q.rc_mf[:2]),'nonfinite':q._replace(qz0=q.qz0.at[0,0].set(jnp.nan))}[damage]
    with pytest.raises(ValueError,match='QNSE.*E78'):
        save_read(tmp_path/fmt,fmt,grid,nl,carry.replace(qnse=q))


@pytest.mark.parametrize('damage',('missing','dtype','shape','nonfinite','vertical_extent'))
@pytest.mark.parametrize('fmt',('pickle','netcdf','checkpoint','store'))
def test_qnse_public_save_requires_output_cache_and_actual_domain(case,tmp_path,fmt,damage):
    grid,nl,carry=case;h=carry.history_diagnostics
    if damage=='vertical_extent':
        q=carry.qnse
        carry=carry.replace(qnse=q._replace(**{k:getattr(q,k)[:-1] for k in ('rc_mf','exch_h','exch_m')}))
    else:
        history={'missing':None,'dtype':h._replace(t2=h.t2.astype(jnp.float64)),
                 'shape':h._replace(t2=h.t2[:2,:2]),'nonfinite':h._replace(q2=h.q2.at[0,0].set(jnp.nan))}[damage]
        carry=carry.replace(history_diagnostics=history)
    with pytest.raises(ValueError,match='QNSE.*E78'):
        save_read(tmp_path/fmt,fmt,grid,nl,carry)


@pytest.mark.parametrize('fmt',('pickle','netcdf','checkpoint','store'))
def test_qnse_public_read_requires_schema_marker(case,tmp_path,fmt):
    grid,nl,carry=case;path=tmp_path/fmt
    save_read(path,fmt,grid,nl,carry)
    if fmt=='netcdf':
        with Dataset(path,'a') as ds:ds.delncattr('GPUWRF_QNSE_SCHEMA_VERSION')
    else:
        if fmt=='store':
            store=RestartStore(path,10**8,2,0);generation=store.latest();path=generation/'snapshot.pkl'
        data=pickle.loads(path.read_bytes())
        group=(data['carry'] if fmt=='pickle' else data['carries']['d01'] if fmt=='store' else data)
        group.pop('qnse_schema_version')
        path.write_bytes(pickle.dumps(data))
        if fmt=='store':
            manifest_path=generation/'manifest.json';manifest=json.loads(manifest_path.read_text())
            manifest.update(payload_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='QNSE.*E78'):
        if fmt=='store':store.read(generation,device=False)
        else:save_read(path,fmt,grid,nl,carry,save=False)


@pytest.mark.parametrize('fmt',('pickle','netcdf','checkpoint','store'))
def test_qnse_public_read_refuses_dropped_surface_output_cache(case,tmp_path,fmt):
    grid,nl,carry=case;path=tmp_path/fmt
    save_read(path,fmt,grid,nl,carry)
    if fmt=='netcdf':
        with Dataset(path,'a') as ds:
            fields=json.loads(ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER)
            kinds=json.loads(ds.GPUWRF_OPTIONAL_CARRY_KIND)
            fields['history_diagnostics']=[];kinds['history_diagnostics']='none'
            ds.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER=json.dumps(fields)
            ds.GPUWRF_OPTIONAL_CARRY_KIND=json.dumps(kinds)
    else:
        if fmt=='store':
            store=RestartStore(path,10**8,2,0);generation=store.latest();path=generation/'snapshot.pkl'
        data=pickle.loads(path.read_bytes())
        if fmt=='checkpoint':data['runtime_state']=data['runtime_state'].replace(history_diagnostics=None)
        else:
            group=data['carry'] if fmt=='pickle' else data['carries']['d01']
            group['extra_fields']['history_diagnostics']=None
        path.write_bytes(pickle.dumps(data))
        if fmt=='store':
            mp=generation/'manifest.json';manifest=json.loads(mp.read_text())
            manifest.update(payload_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            mp.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='QNSE.*E78'):
        if fmt=='store':store.read(generation,device=False)
        else:save_read(path,fmt,grid,nl,carry,save=False)


@pytest.mark.parametrize('bl,sf',((4,5),(5,4),(4,2),(2,4)))
def test_qnse_pair_is_mandatory(bl,sf):
    with pytest.raises(ValueError,match='pairing'):
        resolve_physics_suite({'bl_pbl_physics':bl,'sf_sfclay_physics':sf})


def test_registry_mass_flux_default_and_non_qnse_reset():
    for domain in ('d01','d02'):
        assert bound_options({'physics':{'bl_pbl_physics':[4,4],'sf_sfclay_physics':[4,4]}},domain)['mfshconv']==1
    assert bound_options({'physics':{'bl_pbl_physics':4,'mfshconv':0}},'d01')['mfshconv']==0
    assert bound_options({'physics':{'bl_pbl_physics':5,'mfshconv':1}},'d01')['mfshconv']==0


@pytest.mark.parametrize('mf',(0,1))
@pytest.mark.parametrize('lsm',(1,2,7,'1','2','7'))
def test_qnse_api_refuses_unqualified_land_callers(case,mf,lsm):
    from gpuwrf.runtime.operational_mode import _resolve_operational_suite
    _,nl,_=case
    with pytest.raises(ValueError,match='QNSE land coupling'):
        _resolve_operational_suite(replace(nl,mfshconv=mf,sf_surface_physics=lsm,use_noahmp=False))


@pytest.mark.parametrize('mf',(0,1))
@pytest.mark.parametrize('lsm',(1,2,7))
def test_qnse_namelist_refuses_unqualified_land_callers(mf,lsm):
    from gpuwrf.io.namelist_check import validate_supported_namelist,UnsupportedNamelistOption
    config={'physics':{'bl_pbl_physics':4,'sf_sfclay_physics':4,'mfshconv':mf,'sf_surface_physics':lsm}}
    with pytest.raises(UnsupportedNamelistOption,match='QNSE post-land'):
        validate_supported_namelist(config)


@pytest.mark.parametrize('mf',(0,1))
@pytest.mark.parametrize('lsm',(1,2,7))
@pytest.mark.parametrize('pipeline',('daily','nested'))
def test_qnse_both_loaders_refuse_unqualified_land_callers(monkeypatch,tmp_path,mf,lsm,pipeline):
    from gpuwrf.io.namelist_binding import NamelistNotHonouredError
    from test_namelist_binding import _load_nested,_build_daily_case
    with pytest.raises(NamelistNotHonouredError,match='QNSE land coupling'):
        if pipeline=='nested':
            _load_nested(monkeypatch,tmp_path,{'bl_pbl_physics':'4,4,4','sf_sfclay_physics':'4,4,4',
                'sf_surface_physics':f'{lsm},{lsm},{lsm}'},extra_physics=f' mfshconv = {mf},{mf},{mf},\n')
        else:
            _build_daily_case(monkeypatch,tmp_path,f' bl_pbl_physics = 4,\n sf_sfclay_physics = 4,\n mfshconv = {mf},\n sf_surface_physics = {lsm},\n')


@pytest.mark.parametrize('mf',(0,1))
@pytest.mark.parametrize('lsm',(0,4))
def test_qnse_supported_land_admission_is_retained(case,mf,lsm):
    from gpuwrf.io.namelist_check import validate_supported_namelist
    from gpuwrf.io.namelist_binding import require_namelist_honoured
    from gpuwrf.runtime.operational_mode import _resolve_operational_suite
    _,nl,_=case
    config={'physics':{'bl_pbl_physics':4,'sf_sfclay_physics':4,'mfshconv':mf,'sf_surface_physics':lsm}}
    validate_supported_namelist(config)
    require_namelist_honoured(config,('d01',))
    assert _resolve_operational_suite(replace(nl,mfshconv=mf,sf_surface_physics=lsm,use_noahmp=lsm==4)).pbl.option==4


@pytest.mark.parametrize('mf',(0,1))
def test_qnse_land_refusal_broadcasts_and_identifies_affected_domain(mf):
    from gpuwrf.io.namelist_check import validate_supported_namelist,UnsupportedNamelistOption
    config={'physics':{'bl_pbl_physics':[5,4],'sf_sfclay_physics':[5,4],'mfshconv':mf,'sf_surface_physics':[1,2]}}
    with pytest.raises(UnsupportedNamelistOption) as exc:
        validate_supported_namelist(config)
    failures=[x for x in exc.value.selections if x.key=='sf_surface_physics']
    assert len(failures)==1 and failures[0].domain_index==2
    config['physics'].update(bl_pbl_physics=4,sf_sfclay_physics=4,sf_surface_physics=[4,7])
    with pytest.raises(UnsupportedNamelistOption) as exc:
        validate_supported_namelist(config)
    assert [x.domain_index for x in exc.value.selections if x.key=='sf_surface_physics']==[2]


@pytest.mark.parametrize('mf',(0,1))
def test_runtime_mass_flux_control_and_post_land_flux_handoff(case,monkeypatch,mf):
    # Wiring test only: independent pristine columns provide numerical fidelity.
    # Distinct tagged rates/fluxes detect a bound-but-ignored selector, dropped
    # MF seed, or accidentally reusing the earlier surface flux after the LSM.
    from gpuwrf.coupling import qnse_adapter as adapter
    grid,nl,carry=case
    state=carry.state.replace(hfx=jnp.full(carry.state.hfx.shape,123.,jnp.float32))
    held=carry.qnse._replace(lh=jnp.full(carry.qnse.lh.shape,234.,jnp.float32))
    nz,ny,nx=state.theta.shape;shape=(nz,ny*nx);calls=[]
    def columns(scheme,inputs,**options):
        calls.append(scheme)
        if scheme=='mf':
            assert set(options['requested'])=={'RTHBLTEN','RQVBLTEN','rc_mf'}
            return dict(RTHBLTEN=jnp.full(shape,.001,jnp.float32),
                        RQVBLTEN=jnp.full(shape,1.e-7,jnp.float32),
                        rc_mf=jnp.full(shape,1.e-5,jnp.float32))
        assert scheme=='pbl'
        np.testing.assert_array_equal(inputs['hfx'],np.full(ny*nx,123.,np.float32))
        np.testing.assert_array_equal(inputs['elflx'],np.full(ny*nx,234.,np.float32))
        zero=jnp.zeros(shape,jnp.float32)
        result={name:inputs.get(name,zero) for name in
                ('RTHBLTEN','RQVBLTEN','RUBLTEN','RVBLTEN','RQCBLTEN')}
        result.update(tke=inputs['tke'],EL_MYJ=zero,EXCH_H=inputs['EXCH_H'],EXCH_M=inputs['EXCH_M'])
        result.update({name:inputs[name] for name in ('thz0','qz0','ct','qsfc')})
        result['PBLH']=jnp.ones(ny*nx,jnp.float32)
        return result
    monkeypatch.setattr(adapter,'qnse_columns',columns)
    _,saved,raw=adapter.qnse_pbl_step(state,held,18.,grid,mfshconv=mf,apply_tendencies=False)
    assert calls==(['mf','pbl'] if mf else ['pbl'])
    np.testing.assert_array_equal(raw['RTHBLTEN'],np.full(state.theta.shape,.001 if mf else 0.,np.float32))
    if mf:np.testing.assert_array_equal(saved.rc_mf,np.full(state.theta.shape,1.e-5,np.float32))
    else:exact(saved.rc_mf,held.rc_mf)


def test_qnse_seeds_surface_output_cache_even_when_global_flag_is_off(case,monkeypatch):
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run
    _,nl,carry=case
    monkeypatch.setenv('GPUWRF_HISTORY_INSTEP_DIAG','0')
    nl=replace(nl,mp_physics=0,cu_physics=0,ra_lw_physics=0,ra_sw_physics=0,use_noahmp=False)
    result=_initial_carry_for_run(carry.state,nl)
    assert result.history_diagnostics is not None
    assert all(np.asarray(a).dtype==np.float32 for a in jax.tree.leaves(result.history_diagnostics))


@pytest.mark.parametrize('noahmp',(False,True))
def test_qnse_single_surface_solve_populates_output_cache(case,monkeypatch,noahmp):
    # Structural cache wiring, separate from the pristine numerical oracle.
    from types import SimpleNamespace
    from gpuwrf.coupling import qnse_adapter as adapter
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import seed_history_diagnostics
    grid,nl,carry=case
    nl=replace(nl,mp_physics=0,cu_physics=0,ra_lw_physics=0,ra_sw_physics=0,use_noahmp=noahmp)
    carry=carry.replace(history_diagnostics=seed_history_diagnostics(carry.state,qnsesfc=True))
    names=('t2','q2','u10','v10','lh','psfc','hfx','u10e','v10e');calls=[]
    values={key:jnp.full(carry.state.t_skin.shape,i+1.,jnp.float32) for i,key in enumerate(names)}
    def surface(state,held,*args,**options):
        calls.append('surface')
        return state,held._replace(lh=values['lh']),SimpleNamespace(**values)
    def pbl(state,held,*args,**options):
        calls.append('pbl')
        return state,held,{key:jnp.zeros(state.theta.shape,jnp.float32) for key in
            ('RTHBLTEN','RQVBLTEN','RUBLTEN','RVBLTEN','RQCBLTEN')}
    if noahmp:
        zero=jnp.zeros(carry.state.t_skin.shape,jnp.float32)
        carry=carry.replace(radiation_diagnostics=SimpleNamespace(swnorm=zero,glw=zero,coszen=zero))
        monkeypatch.setattr(op,'_noahmp_params',lambda *a:(None,None))
        monkeypatch.setattr(op,'_lane_noahmp_static',lambda *a:None)
        monkeypatch.setattr(op,'_noahmp_clock',lambda *a:None)
        def land(state,held,*args,**options):
            calls.append('land')
            assert options['surface_diagnostics'].u10e is values['u10e']
            # The real land producer overlays ordinary diagnostics, without
            # owning QNSESFC's special winds. Distinct tags expose stale writes.
            return state,held,{key:value+20 for key,value in values.items()
                               if key not in ('u10e','v10e')}
        monkeypatch.setattr(op,'noahmp_surface_step',land)
    monkeypatch.setattr(adapter,'qnse_surface_step',surface)
    monkeypatch.setattr(adapter,'qnse_pbl_step',pbl)
    result=op._physics_step_forcing(carry,nl,0.,run_radiation=False)
    assert calls==(['surface','land','pbl'] if noahmp else ['surface','pbl'])
    for key in names:
        actual=getattr(result.carry.history_diagnostics,key)
        if actual is not None:
            np.testing.assert_array_equal(actual,values[key]+(20 if noahmp and key not in ('u10e','v10e') else 0))


def test_qnse_adapter_retains_actual_equivalent_roughness_outputs(monkeypatch):
    # Retention check only; original REAL4/compiled-envelope tests independently
    # qualify these unchanged equations. No second surface solve is introduced.
    from gpuwrf.coupling import qnse_adapter as adapter
    from test_v013_myj_janjic_operational import _grid,_state
    grid=_grid();state=_state(grid)
    # Restart byte tags in ``case`` deliberately are not atmospheric inputs.
    state=state.replace(xland=state.xland.at[:,0].set(2),
                        roughness_m=jnp.full(state.t_skin.shape,.5,jnp.float32))
    kernel=adapter.qnse_columns;outputs=[]
    def capture(*args,**options):
        out=kernel(*args,**options);outputs.append(out);return out
    monkeypatch.setattr(adapter,'qnse_columns',capture)
    _,_,diag=adapter.qnse_surface_step(state,adapter.initial_qnse_carry(state),18.,grid,first_timestep=True)
    assert len(outputs)==1
    for name in ('u10e','v10e'):
        actual=np.asarray(getattr(diag,name))
        assert actual.dtype==np.float32 and np.isfinite(actual).all()
        np.testing.assert_array_equal(actual,outputs[0][name].reshape(state.t_skin.shape))
        np.testing.assert_array_equal(actual[:,0],np.asarray(getattr(diag,name[:-1]))[:,0])
    assert np.any(np.asarray(diag.u10e)[:,1:]!=np.asarray(diag.u10)[:,1:])


@pytest.mark.parametrize('pipeline',('daily','nested'))
@pytest.mark.parametrize('mf',(0,1))
def test_both_real_loader_seams_bind_qnse(monkeypatch,tmp_path,pipeline,mf):
    from test_namelist_binding import _load_nested,_build_daily_case
    if pipeline=='nested':
        nl=_load_nested(monkeypatch,tmp_path,{'bl_pbl_physics':'4,4,4','sf_sfclay_physics':'4,4,4'},
                        extra_physics=f' mfshconv = {mf},{mf},{mf},\n')['d02'].namelist
    else:
        nl=_build_daily_case(monkeypatch,tmp_path,f' bl_pbl_physics = 4,\n sf_sfclay_physics = 4,\n mfshconv = {mf},\n')
    assert (nl.bl_pbl_physics,nl.sf_sfclay_physics,nl.mfshconv)==(4,4,mf)
