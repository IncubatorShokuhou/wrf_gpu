"""Distinct old-KF oracle, CPS recurrence, own NCA lifecycle and restart gate."""
import dataclasses
from datetime import datetime,timezone
import functools
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from proofs.v040.sas_family.evidence_gate import promoted_fields, require_finite
import test_v040_sas94 as sas94
from gpuwrf.physics.cumulus_kf99 import kf99_column,update_w0avg99
from gpuwrf.coupling import kf99_adapter as adapter
from gpuwrf.runtime.operational_mode import _initial_carry_for_run,_resolve_operational_suite

jax.config.update('jax_enable_x64',True)
FIX=Path(__file__).resolve().parents[1]/'proofs/v040/sas_family/fixtures'


def test_complete_original_fixture_provenance_active_and_failed_trigger_regimes():
    manifest=json.loads((FIX/'kf99_manifest.json').read_text())
    for name,sha in manifest['fixtures'].items():
        assert hashlib.sha256((FIX/name).read_bytes()).hexdigest()==sha
        with np.load(FIX/name) as d:
            assert all(np.isfinite(a).all() for a in d.values())
            assert np.count_nonzero(d['RAINCV'])==2 and np.count_nonzero(d['RAINCV']==0.)==3


def test_full_promoted_pristine_old_kfcps_ten_outputs():
    with np.load(FIX/'kf99_r8.npz') as d:
        require_finite(**{key:d[key] for key in d})
        def col(*args):return kf99_column(*args[:8],float(d['DT'][0]),float(d['DX'][0]),pi_phy=args[8],mode='r8')
        out=jax.jit(jax.vmap(col))(*[d[k] for k in ('T','QV','P','DZ','RHO','W0AVG','U','V','PI')])
        expected={key:d[key] for key in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RAINCV','PRATEC','NCA','TIMEC')}
        fields=promoted_fields(out,expected,exact=('NCA','TIMEC'))
        for key,result in fields.items():
            assert result['pass'], (key,result)
        require_finite(**{key:np.asarray(value) for key,value in out.items()})
        assert not np.any(out['SOURCE_FATAL']) and not np.any(out['BORROW_EVENTS'])


@pytest.mark.parametrize('mode',('r4','r8'))
def test_cps_full_w_faces_running_mean_against_actual_original(mode):
    dtype=np.float32 if mode=='r4' else np.float64
    with np.load(FIX/f'kf99_{mode}.npz') as d:
        # The original driver explicitly sets W=pre-update W0AVG; top face=0.
        initial=d['W'].astype(dtype)
        faces=np.concatenate((initial,np.zeros((5,1),dtype)),axis=1)
        actual=np.asarray(jax.jit(functools.partial(update_w0avg99,stepcu=5,mode=mode))(
            initial,faces,dtype(d['DT'][0])))
        expected=d['W0AVG'].astype(dtype)
        if mode=='r4':
            # Two implementations of the source's <=6-operation FIR each
            # contribute gamma_6 rounding error. Input/operation FTZ can add
            # at most five minimum-normal values for STEPCU>=1. This bound
            # derives from original inputs, not measured candidate outputs.
            u=np.finfo(np.float32).eps/2.;gamma6=6.*u/(1.-6.*u)
            scale=(9.*np.abs(initial.astype(np.float64))
                   +.5*(np.abs(faces[:,:-1].astype(np.float64))+np.abs(faces[:,1:].astype(np.float64))))/10.
            bound=2.*gamma6*scale+5.*np.finfo(np.float32).tiny
            assert np.all(np.abs(actual.astype(np.float64)-expected)<=bound)
        else:np.testing.assert_allclose(actual,expected,rtol=1.e-14,atol=0.)


def setup():
    grid,state,nl=sas94._setup()
    return grid,state,dataclasses.replace(nl,cu_physics=99,cudt_minutes=.9)


def test_old_clear_threshold_and_hold_are_distinct_from_eta(monkeypatch):
    grid,state,nl=setup()
    held=tuple(jnp.full_like(x,1.e-6 if i<6 else .001) for i,x in enumerate(adapter.initial_kf99_tendencies(state)))
    mean,nca=adapter.initial_kf99_carry(state)
    # Step 2 is outside the 3-step driver cadence. advance_ppt still consumes
    # held rates and decrements NCA, retaining PRATEC even when six rates clear.
    for count,clear in ((36.,False),(18.,False),(9.,False),(4.5,True),(-24.,False)):
        out,(_,next_nca),next_rates=adapter.kf99_cadence_step(state,state,(mean,jnp.full_like(nca,count)),held,
            18.,grid,itimestep=2,stepcu=3,cudt=.9,mp_physics=8)
        np.testing.assert_allclose(np.asarray(out.rainc_acc)-state.rainc_acc,.018,rtol=1e-6)
        np.testing.assert_array_equal(next_nca,jnp.full_like(nca,count-18. if count>0. else count))
        for i,rate in enumerate(next_rates):np.testing.assert_array_equal(rate,jnp.zeros_like(rate) if clear and i<6 else held[i])
    with pytest.raises(ValueError,match='E78'):
        adapter.kf99_cadence_step(state,state,None,held,18.,grid,itimestep=1,stepcu=3,cudt=.9,mp_physics=8)


def test_old_kfcps_half_step_gate_runs_equality_and_preserves_held_rates(monkeypatch):
    grid,state,nl=setup()
    mean,nca=adapter.initial_kf99_carry(state)
    held=tuple(jnp.full_like(x,.001*(i+1)) for i,x in enumerate(adapter.initial_kf99_tendencies(state)))
    def kernel(*args,**kw):
        zeros=jnp.zeros_like(args[0])
        return {**{name:zeros for name in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN')},
                'PRATEC':jnp.asarray(0.,jnp.float32),
                'NCA':jnp.where(kw['nca']>9.,kw['nca'],jnp.asarray(-24.,jnp.float32))}
    monkeypatch.setattr(adapter,'kf99_column',kernel)
    for value,kept in ((10.,True),(9.,False),(-24.,False)):
        rates,_,next_nca=adapter.kf99_rates(state,18.,grid,mean,jnp.full_like(nca,value),held,stepcu=3,cudt=.9)
        for i,rate in enumerate(rates):np.testing.assert_array_equal(rate,held[i] if kept else jnp.zeros_like(rate))
        np.testing.assert_array_equal(next_nca,jnp.full_like(nca,value if kept else -24.))


def test_actual_phy_prep_inputs_against_independent_cpu_exner_fixture(monkeypatch):
    from types import SimpleNamespace
    root=Path(__file__).resolve().parents[1]
    raw=json.loads((root/'tests/v025/b_phys/fixtures/mynn_phy_exner_h1.json').read_text())
    raw={k:np.asarray(raw[k],np.float32) for k in ('p_dyn','p_hyd','pi_phy','theta','qv','qc','qi')}
    n,k=raw['p_dyn'].shape
    field=lambda a:jnp.asarray(a.T[:,:,None],jnp.float32)
    theta=raw['theta']*(np.float32(1.)+np.float32(461.6/287.)*raw['qv'])
    z=np.broadcast_to(np.arange(k+1,dtype=np.float32)[:,None,None]*100.,(k+1,n,1))
    state=SimpleNamespace(p=field(raw['p_dyn']),theta=field(theta),qv=field(raw['qv']),
        ph=jnp.asarray(z*np.float32(9.81)),u=jnp.ones((k,n,2),jnp.float32),
        v=jnp.ones((k,n+1,1),jnp.float32),w=jnp.zeros((k+1,n,1),jnp.float32))
    hp=field(raw['p_hyd']);ps=jnp.full((n,1),100000.,jnp.float32)
    monkeypatch.setattr(adapter,'_wrf_hydrostatic_pressure_profiles_from_state',lambda *a,**kw:(hp,jnp.concatenate((ps[None],hp)),ps))
    monkeypatch.setattr(adapter,'_wrf_phy_prep_rho_from_state',lambda *a,**kw:jnp.full((k,n,1),.9,jnp.float32))
    # Returning observed arguments avoids leaking a vmap tracer into Python.
    def kernel(*a,**kw):
        return dict(RTHCUTEN=a[0],RQVCUTEN=kw['pi_phy'],RQCCUTEN=a[2],RQRCUTEN=a[3],
                    RQICUTEN=a[4],RQSCUTEN=a[5],PRATEC=jnp.float32(0.),NCA=kw['nca'])
    monkeypatch.setattr(adapter,'kf99_column',kernel)
    grid=SimpleNamespace(metrics=object(),projection=SimpleNamespace(dx_m=9000.))
    mean,nca=adapter.initial_kf99_carry(state)
    rates,_,_=adapter.kf99_rates(state,18.,grid,mean,nca,adapter.initial_kf99_tendencies(state))
    unpack=lambda a:np.moveaxis(np.asarray(a),0,-1).reshape(n,k)
    np.testing.assert_allclose(unpack(rates[1]),raw['pi_phy'],rtol=2e-7,atol=0.)
    np.testing.assert_array_equal(unpack(rates[2]),raw['p_hyd'])
    dry=theta/(np.float32(1.)+np.float32(461.6/287.)*raw['qv'])
    np.testing.assert_allclose(unpack(rates[0]),dry*raw['pi_phy'],rtol=3e-7,atol=0.)
    np.testing.assert_array_equal(unpack(rates[4]),np.full((n,k),.9,np.float32))


def test_bound_operational_carry_and_all_restart_formats_refuse_deleted_pair(tmp_path):
    from gpuwrf.io.namelist_binding import bind_namelist_options,require_namelist_honoured
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.runtime.restart_store import RestartStore,nested_identity
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig
    from types import SimpleNamespace
    grid,state,nl=setup()
    require_namelist_honoured({'physics':{'cu_physics':[99,99]}},('d01','d02'))
    nl=bind_namelist_options(nl,{'physics':{'cu_physics':[99,99]}},'d01');_resolve_operational_suite(nl)
    carry=_initial_carry_for_run(state,nl)
    assert len(carry.cumulus_tendencies)==7
    assert all(np.asarray(a).dtype==np.float32 for a in (*carry.cumulus_carry,*carry.cumulus_tendencies))
    carry=carry.replace(cumulus_carry=(carry.cumulus_carry[0]+.125,carry.cumulus_carry[1]+500.),
                        cumulus_tendencies=tuple(x+.001*(i+1) for i,x in enumerate(carry.cumulus_tendencies)))
    def exact(restored):
        for a,b in zip(jax.tree.leaves(carry),jax.tree.leaves(restored),strict=True):np.testing.assert_array_equal(a,b)
    p=tmp_path/'checkpoint'
    write_checkpoint(carry.state,nl,grid,4,p,runtime_state=carry)
    exact(read_checkpoint_with_runtime_state(p)[4])
    write_restart(carry,nl,grid,4,p);exact(read_restart(p)[0])
    p=tmp_path/'wrfrst'
    write_wrfrst_carry(carry,grid,nl,p,valid_time=datetime(2023,1,15,0,1,12,tzinfo=timezone.utc),
        run_start=datetime(2023,1,15,tzinfo=timezone.utc),lead_hours=.02,step_index=4)
    exact(read_wrfrst_carry(p,namelist=nl)[0])
    config=NestedPipelineConfig(tmp_path,tmp_path/'out',tmp_path/'proof',hours=1,max_dom=1)
    identity=nested_identity(config,{'d01':SimpleNamespace(namelist=nl)},datetime(2023,1,15,tzinfo=timezone.utc))
    store=RestartStore(tmp_path/'store',10**7,2,0)
    generation,_=store.save({'d01':carry},{'d01':4},{'d01':18.},identity)
    exact(store.read(generation,expected_identity=identity)[0]['carries']['d01'])
    for missing in ('cumulus_carry','cumulus_tendencies'):
        bad=carry.replace(**{missing:None})
        with pytest.raises(ValueError,match='E78'):write_checkpoint(carry.state,nl,grid,4,p,runtime_state=bad)
        with pytest.raises(ValueError,match='E78'):store.save({'d01':bad},{'d01':4},{'d01':18.},identity)
