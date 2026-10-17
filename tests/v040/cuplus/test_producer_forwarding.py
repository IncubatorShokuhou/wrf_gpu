"""Producer forwarding/default preservation; physics truth is the WRF oracles."""
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from test_v022_camuw_pbl import _grid,_state


def assert_same(a,b):
    left=jax.tree.leaves(a);right=jax.tree.leaves(b)
    assert len(left)==len(right)
    for x,y in zip(left,right,strict=True):
        np.testing.assert_array_equal(np.asarray(x),np.asarray(y))


def test_surface_forwards_this_calls_zol_without_a_second_solve(monkeypatch):
    from gpuwrf.coupling import physics_couplers as c
    grid=_grid();state=_state(grid)
    producer=c.surface_layer_with_diagnostics(c._surface_column_view(state,grid))
    count=[]
    def one(*args,**kwargs):count.append(1);return producer
    monkeypatch.setattr(c,'surface_layer_with_diagnostics',one)
    control=c.surface_adapter(state,10.,grid)
    result,zol=c.surface_adapter(state,10.,grid,return_cumulus_inputs=True)
    assert len(count)==2  # one per call, no extra solve for the diagnostic
    assert_same(control,result)
    assert zol.dtype==jnp.float32
    np.testing.assert_array_equal(np.asarray(zol),np.asarray(producer.zol,np.float32))


def test_noahmp_forwards_pre_lsm_zol_preserving_existing_returns(monkeypatch):
    from test_noahmp_coupler import _build
    from gpuwrf.physics import noahmp_coupler as c
    state,land,static,rad,clock=_build()
    producer=c.surface_layer_with_diagnostics(state,snowh=land.snowh)
    count=[]
    def one(*args,**kwargs):count.append(1);return producer
    monkeypatch.setattr(c,'surface_layer_with_diagnostics',one)
    control=c.noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=10.)
    result=c.noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=10.,return_cumulus_inputs=True)
    assert len(count)==2
    assert len(result)==len(control)+1
    for name,value in control[0].__dict__.items():
        if value is not None:np.testing.assert_array_equal(np.asarray(value),np.asarray(getattr(result[0],name)))
    assert_same(control[1:],result[1:-1])
    np.testing.assert_array_equal(np.asarray(result[-1]),np.asarray(producer.zol,np.float32))


def test_myj_forwards_existing_kpbl_and_raw_rates(monkeypatch):
    from gpuwrf.physics import myj_adapters as c
    grid=_grid();state=_state(grid);nz,ny,nx=state.theta.shape
    profile=jnp.full((ny*nx,nz),.00001,jnp.float64)
    output={key:profile for key in ('RUBLTEN','RVBLTEN','RTHBLTEN','RQVBLTEN','RQCBLTEN','RQIBLTEN')}
    output.update(TKE_MYJ=jnp.ones_like(profile),PBLH=jnp.full((ny*nx,),950.),KPBL=jnp.full((ny*nx,),3,jnp.int32))
    count=[]
    def one(*args,**kwargs):count.append(1);return output
    monkeypatch.setattr(c,'myj_columns',one)
    control=c.myj_pbl_adapter(state,10.,grid)
    result,inputs=c.myj_pbl_adapter(state,10.,grid,return_cumulus_inputs=True)
    raw,rates,raw_inputs=c.myj_pbl_adapter(state,10.,grid,return_tendencies=True,return_cumulus_inputs=True)
    assert len(count)==3
    for name in ('u','v','theta','qv','qke'):assert_same(getattr(control,name),getattr(result,name))
    for name in ('u','v','theta','qv','qc','qi'):assert_same(getattr(state,name),getattr(raw,name))
    assert set(rates)=={'rublten','rvblten','rthblten','rqvblten','rqcblten','rqiblten'}
    assert all(a.dtype==jnp.float32 for a in rates.values())
    np.testing.assert_array_equal(np.asarray(result.pblh),np.full((ny,nx),950.))
    assert_same(inputs,raw_inputs)
    np.testing.assert_array_equal(np.asarray(inputs[1]),np.full((ny,nx),3,np.int32))


def test_janjic_forwards_this_calls_physical_heat_and_vapor_flux(monkeypatch):
    from gpuwrf.physics import myj_adapters as c
    grid=_grid();state=_state(grid)
    producer=c._run_janjic(c._myj_geometry(state));calls=[]
    def one(*args,**kwargs):calls.append(1);return producer
    monkeypatch.setattr(c,'_run_janjic',one)
    control=c.janjic_sfclay_adapter(state,10.,grid)
    result=c.janjic_sfclay_adapter(state,10.,grid,return_cumulus_inputs=True)
    assert len(calls)==2
    for name in ('ustar','theta_flux','qv_flux','tau_u','tau_v','rhosfc','fltv'):
        assert_same(getattr(control,name),getattr(result,name))
    for name in ('hfx','qfx'):
        expected=np.asarray(producer[name]).reshape(state.t_skin.shape).astype(np.asarray(getattr(state,name)).dtype)
        np.testing.assert_array_equal(np.asarray(getattr(result,name)),expected)
