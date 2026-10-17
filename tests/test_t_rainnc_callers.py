"""Pristine WRF caller conventions; independent fixture and active rates."""
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import scan_adapters as sa
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_v014_dry_source_leaf_wiring import _grid,_state,_cpu_tendencies

FIXTURE=Path(__file__).resolve().parents[1]/'proofs/t_rainnc/caller_oracle.json'


def _tiedtke(dtype,monkeypatch):
    row=json.loads(FIXTURE.read_text())['rows']['32' if dtype==jnp.float32 else '64']
    grid=_grid(nz=3,ny=1,nx=1);state=_state(grid)
    col=lambda v:jnp.asarray(v,dtype).reshape(3,1,1)
    state=state.replace(_cast=False,theta=col(row['theta_m']),qv=col(row['qv']),
        p=col(row['p']),qc=jnp.full_like(state.qc,1.e-4),qi=jnp.full_like(state.qi,2.e-5))
    seen={}
    def column(*args,**kw):
        # Test-only host observation of this mocked column boundary; never
        # installed in the model or a measured forecast/timestep program.
        def observe(t,rho):
            seen.update(temperature=np.asarray(t).copy(),rho=np.asarray(rho).copy())
        jax.debug.callback(observe,args[0],args[7])
        zero=jnp.zeros_like(args[0])
        rates={n:zero for n in ('RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','RUCUTEN','RVCUTEN')}
        rates.update(RTHCUTEN=jnp.asarray([.001,-.002,.003],dtype),
                     RQVCUTEN=jnp.asarray(row['qv_rate'],dtype),RAINCV=jnp.asarray(.25,dtype),
                     ICUM1=jnp.bool_(True))  # t-cfix-cu c06 ROOT2 ABI: convective tile row
        return rates
    monkeypatch.setattr(sa,'tiedtke_column_jax',column)
    # Actual original phy_prep coefficient boundary: rho=(1+QV)/ALT.
    # Its dedicated grid/EOS source tests already qualify ALT; this test checks
    # that CU6 selects that total-density producer rather than Thompson's dry rho.
    total_rho=jnp.asarray([1.2,1.0,.8],dtype).reshape(3,1,1)
    monkeypatch.setattr(sa,'_wrf_phy_prep_rho_from_state',lambda *a,**k:total_rho)
    updated=sa.tiedtke_adapter(state,18.,grid)
    jax.block_until_ready(updated)
    return row,state,updated,seen,total_rho


@pytest.mark.parametrize('dtype',(jnp.float32,jnp.float64))
def test_tiedtke_receives_original_dry_temperature(dtype,monkeypatch):
    row,state,updated,seen,_=_tiedtke(dtype,monkeypatch)
    expected=np.asarray(row['temperature'])
    # The legacy column facade computes Exner in DOUBLE; the REAL caller has
    # one rounded theta division and one rounded multiplication. Eight input
    # ULPs cover this independently counted seam, not a fitted tolerance.
    atol=8*np.spacing(expected.astype(np.float32)).astype(np.float64) if dtype==jnp.float32 else 0.
    np.testing.assert_array_less(np.abs(np.asarray(seen['temperature'])-expected),
        atol+2.e-14*np.abs(expected),err_msg='CU6 must receive WRF dry T_PHY')
    np.testing.assert_array_equal(updated.qc,state.qc)
    np.testing.assert_array_equal(updated.qi,state.qi)


@pytest.mark.parametrize('dtype',(jnp.float32,jnp.float64))
def test_tiedtke_converts_dry_tendency_to_moist_carry(dtype,monkeypatch):
    row,state,updated,_,_=_tiedtke(dtype,monkeypatch)
    r=np.asarray(row['theta_m_rate'],dtype=np.float32 if dtype==jnp.float32 else np.float64).reshape(3,1,1)
    expected=(np.asarray(state.theta)+np.asarray(18.,r.dtype)*r).astype(np.asarray(updated.theta).dtype)
    np.testing.assert_allclose(updated.theta,expected,rtol=2*np.finfo(expected.dtype).eps,atol=0,
                               err_msg='CU6 must apply original conv_t_tendf_to_moist')
    np.testing.assert_allclose(updated.rainc_acc-state.rainc_acc,.25,rtol=0,atol=0)


def test_tiedtke_receives_phy_prep_total_density(monkeypatch):
    _,_,_,seen,expected=_tiedtke(jnp.float64,monkeypatch)
    np.testing.assert_array_equal(np.asarray(seen['rho']),np.asarray(expected).reshape(3),
                                  err_msg='CU6 must use original phy_prep total density')


@pytest.mark.parametrize('step',(1,2,3,200))
def test_acm2_actual_runtime_passes_wrf_own_step(monkeypatch,step):
    grid=_grid(nz=3,ny=2,nx=2);state=_state(grid)
    n=om.OperationalNamelist.from_grid(grid,dt_s=18.,tendencies=_cpu_tendencies(grid),metrics=grid.metrics,force_fp64=True)
    n=dataclasses.replace(n,mp_physics=0,cu_physics=0,bl_pbl_physics=7,sf_sfclay_physics=0,
                          sf_surface_physics=0,use_noahmp=False,ra_sw_physics=0,ra_lw_physics=0)
    seen=[]
    cols=jnp.zeros((4,3),state.theta.dtype);flat=jnp.zeros((4,),state.theta.dtype)
    forcing=dict(ncol=4,nz=3,ny=2,nx=2,**{k:cols for k in
        ('u_cols','v_cols','theta_cols','T_cols','qv_cols','rho_cols','dz_cols')},
        **{k:flat for k in ('pblh','ust','hfx','qfx','wspd')})
    monkeypatch.setattr(sa,'_pbl_surface_forcing',lambda *a,**kw:forcing)
    def acm(*a,**kw):
        seen.append(kw['xtime'])
        return dict(theta=cols,qv=cols,u=cols,v=cols,pblh=flat,exch_h=jnp.zeros((4,4)))
    monkeypatch.setattr(sa,'acm2_columns',acm)
    carry=initial_operational_carry(state)
    om._physics_step_forcing(carry,n,(step-1)*18.,run_radiation=False)
    # Original module_pbl_driver.F:1593: XTIME=itimestep (not clock minutes).
    np.testing.assert_array_equal(np.asarray(seen),[step],err_msg='ACM2 XTIME must be actual own step')
