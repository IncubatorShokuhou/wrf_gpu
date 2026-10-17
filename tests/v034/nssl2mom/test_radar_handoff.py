"""Original NSSL radar S5 path, cadence and retained host publication."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import oracle_io as o
from test_driver_io import fields_in
from test_wiring import _state
from gpuwrf.physics.nssl2mom import driver
from gpuwrf.physics.nssl2mom.constants import get_constants
from gpuwrf.physics.nssl2mom.indices import FP64
from gpuwrf.diagnostics.online import configuration, radar_from_context, validate_options

ROOT=Path(__file__).resolve().parents[3]
jax.config.update('jax_enable_x64',True)
FUNCTIONS={}


@pytest.mark.parametrize('mp',[18,22])
@pytest.mark.parametrize('depth',[0,1,44])
def test_selected_depth_matches_original_and_keeps_prognostics(mp,depth):
    sp=(o.load_case('fp64',4) if mp==18 else
        json.loads((ROOT/'proofs/s2_mpmore/nssl22/fixtures/fp64_4.json').read_text()))
    fields={k:jnp.asarray(v,jnp.float64) for k,v in fields_in(sp).items()}
    nz=fields['qv'].shape[-1]
    depth=nz if depth==44 else depth
    dt=sp['scalars']['DT']
    if mp not in FUNCTIONS:
        FUNCTIONS[mp]=jax.jit(lambda f,d:driver.nssl2mom_column(f,dt,get_constants('fp64'),FP64,
            first_step=sp['scalars']['ITIMESTEP']==1,hail_enabled=mp==18,radar_depth=d))
    out,pr,diag=FUNCTIONS[mp](fields,jnp.int32(depth))
    expected=np.asarray(sp['columns']['DBZ_OUT'])
    assert np.count_nonzero(expected)>0
    if depth:
        np.testing.assert_allclose(np.asarray(diag['dbz'])[...,:depth],expected[:depth],rtol=1e-10,atol=1e-14)
    assert np.count_nonzero(np.asarray(diag['dbz'])[...,depth:])==0
    # Regression-only: diagnostic selection must not alter prognostic work.
    off=FUNCTIONS[mp](fields,jnp.int32(0))
    for actual,baseline in zip(jax.tree.leaves((out,pr)),jax.tree.leaves(off[:2])):
        assert np.asarray(actual).tobytes()==np.asarray(baseline).tobytes()


@pytest.mark.parametrize('mp',[17,18,22])
def test_wrf_forced_allocation_and_caller_cadence(mp):
    nl=SimpleNamespace(mp_physics=mp,do_radar_ref=0,nwp_diagnostics=0,run_physics=True,
                       dt_s=54.,diagnostic_history_interval=60.)
    assert configuration(nl)['do_radar_ref']==1
    validate_options(nl,specified=True,nested=False,periodic=False)
    expected=np.asarray(o.load_case('fp64',4)['columns']['DBZ_OUT'],np.float32)
    context=np.broadcast_to(expected[:,None,None],(len(expected),4,4)).copy()
    previous=np.full_like(context,99.)
    call=jax.jit(lambda step:radar_from_context(jnp.asarray(previous),jnp.asarray(context),nl,step,1))
    for step in (1,66,67,68):
        got=np.asarray(call(jnp.int32(step)))
        want=previous.copy()
        if step==1:want[0,1:-1,1:-1]=context[0,1:-1,1:-1]
        if step==67:want[:,1:-1,1:-1]=context[:,1:-1,1:-1]
        np.testing.assert_array_equal(got,want)


def test_adapter_returns_original_radar_and_post_rk_preserves_controls(monkeypatch):
    from gpuwrf.physics.nssl2mom import adapter
    from gpuwrf.runtime import operational_mode as op
    from test_rrtm_lw_operational_wiring import _grid,_namelist
    sp=o.load_case('fp64',4)
    state=_state(sp).replace(re_cloud=None,re_ice=None,re_snow=None)
    # The fixture column used DN=rho_d*(1+qv); WRF's caller passes the dry 1./(al+alb)
    # (t-classd, tests/test_t_classd_mp_rho_callers.py). Feed the fixture DN through the seam.
    from gpuwrf.coupling import scan_adapters as sa
    production=sa._wrf_moist_prep_den_delz
    def fixture_density(st,grid,dtype):
        rho,delz=production(st,grid,dtype)
        return jnp.broadcast_to(jnp.asarray(sp['columns']['DN'],dtype)[:,None,None],rho.shape),delz
    monkeypatch.setattr(sa,'_wrf_moist_prep_den_delz',fixture_density)
    got=jax.jit(lambda st:adapter.nssl2mom_adapter(st,sp['scalars']['DT'],first_step=True,
                                                 radar_depth=jnp.int32(st.qv.shape[0])))(state)
    monkeypatch.setattr(sa,'_wrf_moist_prep_den_delz',production)
    np.testing.assert_allclose(np.asarray(got[2])[:,0,0],sp['columns']['DBZ_OUT'],rtol=1e-3,atol=1e-4)
    seen=[]
    def capture(st,dt,grid,**kw):
        seen.append(kw)
        z=jnp.zeros(st.t_skin.shape)
        return st,dict(rainncv=z,rain_liquid=z,hailncv=z,snowncv=z,grplncv=z,sr=z),got[2]
    monkeypatch.setattr(adapter,'nssl2mom_adapter',capture)
    nl=replace(_namelist(_grid()),mp_physics=18,dt_s=54.,diagnostic_history_interval=60.)
    rates={'qr':jnp.ones_like(state.qv),'qs':jnp.ones_like(state.qv)*2}
    for step,depth in ((1,1),(66,0),(67,state.qv.shape[0])):
        actual=op._apply_post_rk_microphysics(state,nl,return_precipitation=True,
            return_radar_context=True,step_index=jnp.int32(step),cumulus_tendencies=rates)
        assert int(seen[-1]['radar_depth'])==depth
        assert seen[-1]['cumulus_tendencies'] is rates
        assert bool(seen[-1]['first_step'])==(step==1)
        assert actual[2] is got[2]
