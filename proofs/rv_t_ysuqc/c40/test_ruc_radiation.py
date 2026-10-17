"""Independent RUC producer/caller check; constructed state, real radiation/bundle.

This tests the physical flux interface, not a coupled forecast fidelity gate.
"""
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
import json
from conftest import cpu_pallas_interpret
import jax.numpy as jnp
import numpy as np
import pytest

OUT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-ysuqc/c40_review')

class Stop(Exception):
    pass

def test_ruc_real_rrtmg_pair_and_gsw_caller(cpu_pallas_interpret,tmp_path,monkeypatch):
    from test_s2_ssib_wiring import _case
    from test_o2_ruccli import _input
    from gpuwrf.io.ruc_land_init import load_ruc_land
    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.coupling import ruc_surface_hook as hook
    from gpuwrf.runtime import operational_mode as om
    assert pc._rrtmg_real_enabled(), 'Run with both release radiation REAL-entry flags enabled'
    grid,state,nml,_=_case(tmp_path)
    # Use actual c33 atmospheric columns/vertical metrics. The tiny neutral
    # six-level construction produced nonfinite SW and is retained as failed
    # evidence; it cannot establish the real RUC producer contract.
    from gpuwrf.io.netcdf_lock import Dataset
    from test_v013_operational_smoke import _grid, _base_state
    case=Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c33/cpu/wrfout_d01_2026-10-13_12:00:00')
    with Dataset(case) as ds:
        nz=len(ds.dimensions['bottom_top'])
        grid=_grid(nz=nz,ny=2,nx=3)
        state=_base_state(grid)
        def field(name,ny=2,nx=3):
            return jnp.asarray(ds[name][0,...,20:20+ny,20:20+nx],jnp.float32)
        qv=field('QVAPOR');pressure=field('P')+field('PB');height=field('PH')+field('PHB')
        values=dict(u=field('U',2,4),v=field('V',3,3),theta=(field('T')+300.)*(1.+jnp.float32(461.6/287.)*qv),
            qv=qv,qc=field('QCLOUD'),qi=field('QICE'),qs=field('QSNOW'),qr=field('QRAIN'),qg=field('QGRAUP'),
            p=pressure,p_total=pressure,p_perturbation=field('P'),
            ph=height,ph_total=height,ph_perturbation=field('PH'),
            mu=field('MU')+field('MUB'),mu_total=field('MU')+field('MUB'),mu_perturbation=field('MU'),
            t_skin=field('TSK'),xland=field('XLAND'))
        state=state.replace(_cast=False,**values)
        metric={key.lower():jnp.asarray(ds[key][0],jnp.float32) for key in ('C1H','C2H','C3H','C4H','DNW','FNM','FNP')}
        metric['p_top']=jnp.asarray(ds['P_TOP'][0],jnp.float32)
        grid=dataclasses.replace(grid,metrics=dataclasses.replace(grid.metrics,**metric))
    _input(tmp_path/'wrfinput_d01',6)
    static,land=load_ruc_land(tmp_path,'d01',num_soil_layers=6,dt=18.,p1=np.full((2,3),90000.))
    z=np.zeros((2,3),np.float32)
    rad_static=pc.RRTMGRadiationStatic(xlat_deg=z+42.,xlong_deg=z,terrain_height_m=z,
        slope_rad=z,slope_azimuth_rad=z,sina=z,cosa=z+1.)
    nml=dataclasses.replace(nml,sf_surface_physics=3,sf_sfclay_physics=5,bl_pbl_physics=5,
        ra_sw_physics=4,ra_lw_physics=4,ssib_static=None,ssib_land=None,ssib_rad=None,
        grid=grid,metrics=grid.metrics,ruc_static=static,ruc_land=land,
        time_utc=datetime(2026,10,13,12,tzinfo=timezone.utc),radiation_static=rad_static)
    carry=om._initial_carry_for_run(state,nml)
    diag=pc.rrtmg_radiation_diagnostics(state,grid,time_utc=nml.time_utc,
        radiation_static=rad_static,land_state=hook.ruc_radiation_surface(land),slope_rad=0)
    down,normal,up,topup=map(np.asarray,(diag.swdown,diag.swnorm,diag.swup,diag.swup_topographic))
    assert np.isfinite(down).all() and np.isfinite(up).all()
    assert down.min()>100.,down
    np.testing.assert_array_equal(normal,down)
    np.testing.assert_array_equal(topup,up)
    carry=carry.replace(radiation_diagnostics=diag)
    seen={}
    def step(*args,**kwargs):
        seen.update(kwargs)
        raise Stop
    monkeypatch.setattr(hook,'ruc_surface_step',step)
    with pytest.raises(Stop):
        om._physics_step_forcing(carry,nml,0.,run_radiation=False,first_timestep=True)
    actual=np.asarray(seen['radiation'].gsw)
    np.testing.assert_array_equal(actual,down-up)
    np.testing.assert_array_equal(np.asarray(seen['radiation'].glw),np.asarray(diag.glw))
    (OUT/'ruc_real_producer.json').write_text(json.dumps(dict(
        scope='actual c33 noon atmospheric crop+vertical metrics, prepared 6-soil 2x3 RUC bundle, real RRTMG; interface proof, not forecast fidelity',
        swdown_min=float(down.min()),swdown_max=float(down.max()),
        swnorm_equals_swdown=True,topographic_up_equals_up=True,
        ruc_gsw_min=float(actual.min()),ruc_gsw_max=float(actual.max()),
        caller_equals_swdown_minus_swup=True,slope_rad=0),indent=2)+'\n')
