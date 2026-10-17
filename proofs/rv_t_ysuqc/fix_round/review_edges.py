"""Independent caller-seam review. Synthetic sentinels test wiring, not fidelity."""
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
from gpuwrf.runtime import operational_mode as om
from gpuwrf.runtime.operational_state import initial_operational_carry
from test_v013_operational_smoke import _grid, _base_state, _namelist

OUT = Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-ysuqc/fix_round')


def test_current_radiation_reaches_ysu_on_refresh_step(monkeypatch):
    grid = _grid(nz=12, ny=3, nx=3)
    ns = _namelist(grid, mp_physics=0, cu_physics=0, bl_pbl_physics=1,
        sf_sfclay_physics=1, sf_surface_physics=0, use_noahmp=False,
        ra_lw_physics=1, ra_sw_physics=1, rad_rk_tendf=1)
    carry = initial_operational_carry(_base_state(grid))
    carry = carry.replace(rthraten=jnp.full_like(carry.rthraten, -2.e-5))
    assert carry.radiation_diagnostics is None
    calls, seen = [], {}
    def radiation(st, *args, **kwargs):
        calls.append('radiation')
        return jnp.full_like(st.theta, -1.e-4)
    monkeypatch.setattr(om, 'rrtm_lw_theta_tendency', radiation)
    monkeypatch.setattr(om, 'dudhia_sw_theta_tendency', radiation)
    def pbl(st, dt, grid, **kwargs):
        calls.append('pbl')
        seen['rthraten'] = np.asarray(kwargs['rthraten'])
        zero = jnp.zeros((9,12),jnp.float32)
        return st, {key:zero for key in ('u','v','theta','qv','qc','qi')}
    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS,1,pbl)
    forcing = om._physics_step_forcing(carry,ns,0.,run_radiation=True,first_timestep=True)
    current = np.asarray(forcing.carry.rthraten)
    (OUT/'radiation_edge.json').write_text(json.dumps(dict(calls=calls,
        pbl_rthraten=float(seen['rthraten'].flat[0]),current_rthraten=float(current.flat[0])),indent=2)+'\n')
    np.testing.assert_array_equal(seen['rthraten'],current,err_msg='YSU must consume this radiation step RTHRATEN')


def test_ssib_cloud_sources_enter_rk_once(monkeypatch,tmp_path):
    from test_s2_ssib_wiring import _case
    from gpuwrf.coupling import ssib_surface_hook as hook
    from gpuwrf.coupling import scan_adapters as adapters
    import gpuwrf.physics.pbl_ysu as ysu
    grid,state,ns,carry = _case(tmp_path)
    ns = dataclasses.replace(ns,rad_rk_tendf=1)
    # Isolate only the unchanged land/surface work. Retain the production YSU
    # hook, cloud-rate application and operational RK handoff.
    z = jnp.zeros((2,3),jnp.float32)
    carry = carry.replace(radiation_diagnostics=SimpleNamespace(
        swdown=z,glw=z,coszen=z,cloud_fraction=jnp.zeros_like(state.theta)))
    monkeypatch.setattr(om,'_refresh_rrtmg_driver',lambda c,*a,**k:c)
    monkeypatch.setattr(om,'_phys_tend_rk_wrf_enabled',lambda:True)
    monkeypatch.setattr(hook,'ssib_day',lambda *a,**k:1.)
    monkeypatch.setattr(hook,'ssib_radiation_surface',lambda *a,**k:None)
    surface = {key:jnp.ones_like(z) for key in ('fm','fh','br','u10','v10')}
    monkeypatch.setattr(adapters,'sfclay_revised_mm5_adapter',lambda st,*a,**k:(st,surface))
    history = carry.history_diagnostics._asdict()
    monkeypatch.setattr(hook,'ssib_surface_step',lambda st,land,*a,**k:(st,land,history))
    rates = {key:jnp.full((6,6),value,jnp.float32) for key,value in
        dict(u=0.,v=0.,theta=0.,qv=0.,qc=1.e-7,qi=2.e-7).items()}
    def columns(*args,**kwargs):
        return dict(rates,pblh=jnp.full((6,),432.,jnp.float32))
    monkeypatch.setattr(ysu,'ysu_columns',columns)
    forcing = om._physics_step_forcing(carry,ns,0.,run_radiation=False,first_timestep=True)
    raw = forcing.dry_tendencies.moist_tendf or {}
    evidence = {key:dict(state_max_delta=float(np.max(np.abs(np.asarray(getattr(forcing.state,key))-np.asarray(getattr(state,key))))),
                        raw_present=key in raw) for key in ('qc','qi')}
    (OUT/'ssib_edge.json').write_text(json.dumps(evidence,indent=2)+'\n')
    for key in ('qc','qi'):
        np.testing.assert_array_equal(getattr(forcing.state,key),getattr(state,key),
            err_msg=f'SSiB YSU {key}: Euler update must be restored before RK')
        assert key in raw, f'SSiB YSU {key}: missing RK source'


def test_generic_cloud_sources_enter_rk_once(monkeypatch):
    grid = _grid(nz=12,ny=3,nx=3)
    state = _base_state(grid)
    ns = _namelist(grid,mp_physics=0,cu_physics=0,bl_pbl_physics=1,
        sf_sfclay_physics=1,sf_surface_physics=0,use_noahmp=False,
        ra_lw_physics=0,ra_sw_physics=0,rad_rk_tendf=1)
    values = dict(u=0.,v=0.,theta=0.,qv=0.,qc=1.e-7,qi=2.e-7)
    rates = {key:jnp.full((9,12),value,jnp.float32) for key,value in values.items()}
    monkeypatch.setattr(om,'_phys_tend_rk_wrf_enabled',lambda:True)
    def pbl(st,*a,**k):
        return st.replace(qc=st.qc+1.,qi=st.qi+2.),rates
    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS,1,pbl)
    forcing = om._physics_step_forcing(initial_operational_carry(state),ns,0.,
        run_radiation=False,first_timestep=True)
    mass = np.asarray(ns.metrics.c1h)[:,None,None]*np.asarray(state.mu_total)[None]+np.asarray(ns.metrics.c2h)[:,None,None]
    for key in ('qc','qi'):
        np.testing.assert_array_equal(getattr(forcing.state,key),getattr(state,key))
        np.testing.assert_allclose(np.asarray(forcing.dry_tendencies.moist_tendf[key]),mass*values[key],rtol=8*np.finfo(np.float32).eps,atol=0.)
