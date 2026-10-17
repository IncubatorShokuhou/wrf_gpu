"""Distinct old OSAS native oracle, genuine CPU-WRF caller and held cadence."""
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import test_v040_sas94 as sas94
from gpuwrf.coupling import osas95_adapter as adapter
from gpuwrf.physics.cumulus_osas95 import cu_osas95_columns
from gpuwrf.physics.osas_rng import initial_osas_key,osas_draws

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'proofs/v040/sas_family/fixtures'


def test_recorded_draw_fixture_provenance_and_active_regimes():
    manifest=json.loads((FIX/'osas95_manifest.json').read_text())
    for name,sha in manifest['fixtures'].items():
        assert hashlib.sha256((FIX/name).read_bytes()).hexdigest()==sha
        with np.load(FIX/name) as data:
            assert all(np.isfinite(x).all() for x in data.values())
            assert len(data['DRAW_XKT2'])==455 and np.all((data['DRAW_XKT2']>=0.)&(data['DRAW_XKT2']<1.))
            assert np.count_nonzero(data['RAINCV'])>=190
            assert np.count_nonzero(data['RAINCV']==0.)>=250


@pytest.mark.parametrize('dt,stepcu',((54.,5),(18.,1)))
def test_native_pristine_recorded_draws(dt,stepcu):
    name=f'osas95_dt{int(dt)}_stepcu{stepcu}_shallow1_r4.npz'
    with np.load(FIX/name) as expected:
        out=jax.jit(functools.partial(cu_osas95_columns,dt=dt,stepcu=stepcu,literals='r4'))(
            *sas94._column_args(),draws=expected['DRAW_XKT2'])
        for k,a in out.items():
            a=np.asarray(a);b=expected[k]
            assert a.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all()
            if k in ('HBOT','HTOP'):np.testing.assert_array_equal(a,b)
            else:
                assert np.all(np.abs(a.astype(np.float64)-b)<=np.spacing(np.maximum(np.abs(a),np.abs(b)))),k
                assert np.mean(a==b)>=.999,k


def test_actual_caller_reads_independent_cpu_exner_and_hydro_pressure(monkeypatch):
    row=json.loads((ROOT/'tests/v025/b_phys/fixtures/mynn_phy_exner_h1.json').read_text())
    raw={k:np.asarray(row[k],np.float32) for k in ('p_dyn','p_hyd','pi_phy','theta','qv','qc','qi')}
    n,k=raw['p_dyn'].shape
    field=lambda a:jnp.asarray(a.T[:,:,None],jnp.float32)
    theta=raw['theta']*(np.float32(1.)+np.float32(461.6/287.)*raw['qv'])
    zface=np.broadcast_to(np.arange(k+1,dtype=np.float32)[:,None,None]*100.,(k+1,n,1))
    state=SimpleNamespace(p=field(raw['p_dyn']),theta=field(theta),qv=field(raw['qv']),
        qc=field(raw['qc']),qi=field(raw['qi']),ph=jnp.asarray(zface*np.float32(9.81)),
        u=jnp.ones((k,n,2),jnp.float32),v=jnp.ones((k,n+1,1),jnp.float32),
        w=jnp.zeros((k+1,n,1),jnp.float32),xland=jnp.ones((n,1),jnp.float32))
    ps=jnp.full((n,1),100000.,jnp.float32);hp=field(raw['p_hyd'])
    monkeypatch.setattr(adapter,'_wrf_hydrostatic_pressure_profiles_from_state',lambda *a,**kw:(hp,jnp.concatenate((ps[None],hp)),ps))
    monkeypatch.setattr(adapter,'_wrf_phy_prep_rho_from_state',lambda *a,**kw:jnp.ones((k,n,1),jnp.float32))
    captured={}
    def kernel(*a,**kw):
        captured.update(a=a,kw=kw)
        return {**{key:jnp.zeros((n,k),jnp.float32) for key in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN')},
                'PRATEC':jnp.zeros(n,jnp.float32)}
    monkeypatch.setattr(adapter,'cu_osas95_columns',kernel)
    key=initial_osas_key(3)
    adapter.osas95_rates(state,18.,SimpleNamespace(metrics=object()),rng_key=key,itimestep=123,stepcu=3)
    a=captured['a']
    np.testing.assert_allclose(a[5],raw['pi_phy'],rtol=2e-7,atol=0.)
    np.testing.assert_array_equal(a[4],raw['p_hyd'])
    dry=theta/(np.float32(1.)+np.float32(461.6/287.)*raw['qv'])
    np.testing.assert_allclose(a[0],dry*raw['pi_phy'],rtol=3e-7,atol=0.)
    np.testing.assert_array_equal(captured['kw']['draws'],osas_draws(key,123,jnp.arange(n,dtype=jnp.uint32)))


def test_cadence_holds_original_rates_between_calls(monkeypatch):
    grid,state,nl=sas94._setup()
    rates=adapter.initial_osas95_tendencies(state)
    held=tuple(jnp.full_like(x,.001*(i+1)) for i,x in enumerate(rates))
    monkeypatch.setattr(adapter,'osas95_rates',lambda *a,**kw:tuple(rates[i] for i in (0,1,2,4,6)))
    for step,refresh in ((1,True),(2,False),(3,True),(4,False)):
        out,new=adapter.osas95_cadence_step(state,state,held,18.,grid,stepcu=3,itimestep=step,rng_key=initial_osas_key())
        for a,b in zip(new,rates if refresh else held):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='E78'):
        adapter.osas95_cadence_step(state,state,held,18.,grid,stepcu=3,itimestep=2,rng_key=None)


def test_production_slot_clock_continues_counter_draws_after_checkpoint(monkeypatch,tmp_path):
    import dataclasses
    from gpuwrf.runtime.operational_mode import _initial_carry_for_run,_physics_step_forcing
    from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
    grid,state,nl=sas94._setup()
    nl=dataclasses.replace(nl,cu_physics=95,osas_domain_id=2)
    def kernel(*a,draws,**kw):
        n,k=a[0].shape
        return {**{key:jnp.zeros((n,k),jnp.float32) for key in ('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN')},
                'PRATEC':jnp.asarray(draws,jnp.float32)*jnp.float32(.001)}
    monkeypatch.setattr(adapter,'cu_osas95_columns',kernel)
    carry=_initial_carry_for_run(state,nl)
    for i in range(2):carry=_physics_step_forcing(carry,nl,18.*i,run_radiation=False).carry
    path=tmp_path/'resume.pkl'
    write_checkpoint(carry.state,nl,grid,2,path,runtime_state=carry)
    _,resumed_nl,_,step,resumed=read_checkpoint_with_runtime_state(path)
    assert step==2
    for i in range(step,6):
        carry=_physics_step_forcing(carry,nl,18.*i,run_radiation=False).carry
        resumed=_physics_step_forcing(resumed,resumed_nl,18.*i,run_radiation=False).carry
        for a,b in zip(jax.tree.leaves(carry),jax.tree.leaves(resumed),strict=True):
            np.testing.assert_array_equal(a,b)
    want=osas_draws(initial_osas_key(2),6,jnp.arange(4,dtype=jnp.uint32)).reshape(2,2)
    np.testing.assert_array_equal(carry.cumulus_tendencies[6],np.asarray(want,jnp.float32)*np.float32(.001))
