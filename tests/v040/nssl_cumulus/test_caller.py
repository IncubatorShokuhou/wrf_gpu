"""Actual post-RK NSSL call receives held, cleared/decoupled KF Q*CUTEN.

The RK kernel is replaced by a deterministic final-mass change; the real
physics preparation, advance_ppt/phy_prep_part2 and post-RK call execute.
Independent stage/full-driver fixtures cover the scientific number update.
"""
import dataclasses
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def fixture_module():
    spec = importlib.util.spec_from_file_location('nssl_caller_state',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    return f


@pytest.mark.parametrize('nca',[54.,108.])
@pytest.mark.parametrize('with_precip',[False,True])
@pytest.mark.parametrize('step_index',[1,2])
def test_actual_post_rk_uses_final_mass_rates(monkeypatch,nca,with_precip,step_index):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.runtime.noahmp_precipitation import seed_precipitation
    from gpuwrf.diagnostics import census
    from gpuwrf.physics.nssl2mom import adapter
    f=fixture_module()
    grid=f._grid(ny=3,nx=4,nz=8)
    state=f._state(grid)
    nml=dataclasses.replace(f._namelist(grid,ra_sw_physics=0,ra_lw_physics=0),
        run_physics=True,use_noahmp=False,mp_physics=18,cu_physics=1,bl_pbl_physics=0,
        sf_sfclay_physics=0,rad_rk_tendf=1,run_boundary=False,dt_s=54.,cumulus_cadence_steps=6)
    rates=tuple(jnp.full_like(state.theta,(i+1)*1e-8) for i in range(6))+(jnp.full_like(state.t_skin,.002),)
    carry=initial_operational_carry(state).replace(
        cumulus_carry=(jnp.zeros_like(state.theta),jnp.full_like(state.t_skin,nca)),
        cumulus_tendencies=rates,census=census.initial_census(),
        noahmp_precipitation=seed_precipitation(state) if with_precip else None)
    monkeypatch.setenv('GPUWRF_KF_TEND_RK_WRF','1')
    monkeypatch.setenv('GPUWRF_PHYS_TEND_RK_WRF','1')
    monkeypatch.setenv('GPUWRF_MP_WRF_ORDER','1')
    monkeypatch.setattr(op,'kf_adapter',lambda st,dt,w,n,**kw:(kw['held_tendencies'],w,n))
    final_mu=state.mu_total+1234.
    monkeypatch.setattr(op,'_rk_scan_step',lambda c,nml,**kw:c.replace(state=c.state.replace(mu_total=final_mu)))
    calls=[]
    def capture(st,dt,grid,**kw):
        calls.append(kw)
        zero=jnp.zeros_like(st.t_skin)
        return st,dict(rainncv=zero,rain_liquid=zero,hailncv=zero,snowncv=zero,grplncv=zero,sr=zero)
    monkeypatch.setattr(adapter,'nssl2mom_adapter',capture)
    op._physics_boundary_step(carry,nml,jnp.asarray(step_index),run_radiation=False)
    assert len(calls)==1
    assert bool(calls[0]['first_step']) == (step_index == 1)
    cu=calls[0]['cumulus_tendencies']
    assert set(cu)=={'qr','qs'}
    mass0=grid.metrics.c1h[:,None,None]*state.mu_total[None]+grid.metrics.c2h[:,None,None]
    massf=grid.metrics.c1h[:,None,None]*final_mu[None]+grid.metrics.c2h[:,None,None]
    for q,i in [('qr',3),('qs',5)]:
        expected=np.zeros_like(rates[i]) if nca==54. else mass0*rates[i]/massf
        assert np.isfinite(np.asarray(cu[q])).all()
        np.testing.assert_allclose(cu[q],expected,rtol=2e-15,atol=1e-25)


def test_adapter_passes_column_rates_without_reconstruction(monkeypatch):
    from gpuwrf.physics.nssl2mom import adapter
    from gpuwrf.contracts import state as sc
    monkeypatch.setattr(sc,'_gpu_device',lambda:jax.devices('cpu')[0])
    f=fixture_module()
    grid=f._grid(ny=3,nx=4,nz=8)
    state=f._state(grid).ensure_conditional_leaves(mp_physics=18)
    rate=jnp.arange(8*3*4,dtype=state.qr.dtype).reshape(8,3,4)*1e-8
    seen=[]
    def column(fields,dt,C,prec,**kw):
        seen.append(kw['cumulus_tendencies'])
        zero=jnp.zeros((3,4),state.theta.dtype)
        return fields,dict(rain_liquid=zero,hailncv=zero,snowncv=zero,grplncv=zero),{}
    monkeypatch.setattr(adapter,'nssl2mom_column',column)
    adapter.nssl2mom_adapter(state,18.,grid,cumulus_tendencies={'qr':rate,'qs':-rate})
    assert len(seen)==1
    # Explicit State [k,j,i] -> NSSL column [j,i,k] mapping.
    np.testing.assert_array_equal(seen[0]['qr'],np.transpose(rate,(1,2,0)))
    np.testing.assert_array_equal(seen[0]['qs'],-np.transpose(rate,(1,2,0)))


def test_morrison_radius_default_matches_compiled_pristine_predicate(monkeypatch):
    from gpuwrf.physics import rrtmg_mp_re as re
    receipt=json.loads((ROOT/'proofs/v040/nssl_cumulus/radius_audit.json').read_text())
    mp40=[r for r in receipt['rows'] if r[0]==40]
    assert mp40==[[40,0,0,0,0],[40,1,0,0,0]]
    # Exercise the enabled radius route; FAST_DEFAULTS=0 alone is insufficient
    # to catch an accidental future addition of mp40 to the activation rule.
    monkeypatch.setattr(re,'_MP_RE',True)
    assert re.mp_re_active(mp_physics=8,use_mp_re=1)
    for use_re in (0,1):
        assert not re.mp_re_active(mp_physics=40,use_mp_re=use_re)


def test_physics_disabled_does_not_require_kf_rates(monkeypatch):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry
    f=fixture_module()
    grid=f._grid(ny=3,nx=4,nz=8)
    state=f._state(grid)
    nml=dataclasses.replace(f._namelist(grid,ra_sw_physics=0,ra_lw_physics=0),
        run_physics=False,mp_physics=18,cu_physics=1,run_boundary=False)
    carry=initial_operational_carry(state)
    assert carry.cumulus_tendencies is None
    monkeypatch.setattr(op,'_rk_scan_step',lambda c,*a,**kw:c)
    got=op._physics_boundary_step(carry,nml,jnp.asarray(1),run_radiation=False)
    np.testing.assert_array_equal(got.state.qr,state.qr)
