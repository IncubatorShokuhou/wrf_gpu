"""E11: native radiation_driver SGS inputs reach both legacy solver ABIs.

Solvers are transparent input witnesses, not heating/fidelity references.
The real shared cloud producer runs, including non-MYNN policy and thresholds.
"""
from datetime import datetime, timezone
from types import SimpleNamespace as NS
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.coupling import flg_gfdl_radiation as port


@pytest.mark.parametrize('scheme',[7,99])
@pytest.mark.parametrize('pbl',[5,1])
@pytest.mark.parametrize('lead',[0.,18.])
def test_native_sgs_condensate_reaches_solver(monkeypatch,scheme,pbl,lead):
    # Native strict thresholds, including equality and inactive cloud fraction.
    qc=np.array([0.,.5e-6,1e-6,0.,0.,2e-6],np.float32)
    qi=np.array([0.,.5e-8,1e-8,0.,0.,2e-8],np.float32)
    cf=np.array([.5,.5,.5,.001,0.,.5],np.float32)
    field=lambda a:jnp.asarray(a,jnp.float32).reshape(6,1,1)
    z=field(np.zeros(6,np.float32));s=jnp.ones((1,1),jnp.float32)
    state=NS(theta=field(np.full(6,290,np.float32)),p=field(np.full(6,80000,np.float32)),
        qv=field(np.full(6,.002,np.float32)),qc=field(qc),qi=field(qi),qi2=None,qi3=None,
        qr=z,qs=z,qg=z,qc_bl=field(np.full(6,1e-4,np.float32)),
        qi_bl=field(np.full(6,2e-5,np.float32)),cldfra_bl=field(cf),t_skin=280*s,xland=s)
    pc=port.pc
    monkeypatch.setattr(pc,'sgs_cloud_enabled',lambda:True)
    monkeypatch.setattr(pc,'_column_dz_from_state',lambda *_a,**_k:jnp.ones((1,6),jnp.float32))
    monkeypatch.setattr(pc,'_rho_from_state',lambda *_a,**_k:jnp.ones_like(z))
    monkeypatch.setattr(pc,'_wrf_phy_prep_rho_from_state',lambda *_a,**_k:jnp.ones_like(z))
    monkeypatch.setattr(pc,'_surface_radiation_properties',lambda *_a,**_k:(.2*s,.97*s))
    monkeypatch.setattr(pc,'_radiation_static_for_grid',lambda *_a,**_k:None)
    monkeypatch.setattr(pc,'_grid_lat_lon',lambda *_a,**_k:(45*s,8*s))
    native_view=pc._rrtmg_column_inputs
    def view(*args,**kwargs):
        # Exercise the actual shared WRF cloud producer. Only irrelevant
        # geometry is supplied here, rather than a fake already-merged cloud.
        result=native_view(args[0],None,**kwargs)
        lw=result[1]
        lw.pressure_interfaces=jnp.full((1,7),80000,jnp.float32)
        lw.temperature_interfaces=jnp.full((1,7),270,jnp.float32)
        return result[0],lw,*result[2:]
    monkeypatch.setattr(pc,'_rrtmg_column_inputs',view)
    # paired_radiation separately requires supplied native geographical fields.
    original_static=pc._radiation_static_for_grid
    def static(*a,**k):return NS(xlat_deg=45*s,xlong_deg=8*s) if a[1] is not None else original_static(*a,**k)
    monkeypatch.setattr(pc,'_radiation_static_for_grid',static)
    nl=NS(ra_lw_physics=scheme,bl_pbl_physics=pbl,mp_physics=8,radiation_interval_s=36.,
        dt_s=18.,radiation_cadence_steps=2,grid=NS(metrics=None),
        time_utc=datetime(2026,1,15,12,tzinfo=timezone.utc),radiation_static=None,
        noahmp_static=NS(shdfac=s),gfdl_grid=NS(solver=lambda:None))
    # WRF phys/module_radiation_driver.F:1418-1430, REAL4 additions. This
    # independent expected mapping does not call the port's merge helper.
    expected_qc=qc.copy();expected_qi=qi.copy()
    if pbl==5:
        for k in range(6):
            if qc[k]<np.float32(1e-6) and cf[k]>np.float32(.001):expected_qc[k]+=np.float32(1e-4)
            if qi[k]<np.float32(1e-8) and cf[k]>np.float32(.001):expected_qi[k]+=np.float32(2e-5)
    if scheme==7:
        monkeypatch.setattr(port.flg,'preprocess_column',lambda *a:(a[5][:-1],a[7][:-1]))
        def witness(pre,pi):
            c,i=pre
            return dict(total_rate=c+10*i,swdn=jnp.float32(0),gsw=i.sum(),glw=c.sum(),
                sw_up_top=jnp.float32(0),lw_up_top=jnp.float32(0),direct=jnp.float32(0),
                diffuse=jnp.float32(0),dni=jnp.float32(0))
        monkeypatch.setattr(port.flg,'_column_from_preprocessed',witness)
    else:
        def witness(_solver,c,_clock,_old,**kwargs):
            out={k:jnp.float32(0) for k in ('rswin','rswinc','rswtoa','rlwtoa','czmean',
                'acfrcv','acfrst','cfracl','cfracm','cfrach')}
            out.update(glw=c['qc'].sum(),gsw=c['qi'].sum(),ncfrcv=jnp.int32(0),ncfrst=jnp.int32(0),
                tendl=c['qc'][None,::-1,None],tends=c['qi'][None,::-1,None],
                cldfra=jnp.zeros((1,7,1),jnp.float32))
            return out
        monkeypatch.setattr(port,'native_column',witness)
    _,diag,_=port.paired_radiation(state,nl,lead,gfdl=port.initial_gfdl(state) if scheme==99 else None)
    np.testing.assert_allclose(diag.glw,np.sum(expected_qc,dtype=np.float32),rtol=2e-7,atol=0)
    np.testing.assert_allclose(diag.gsw,np.sum(expected_qi,dtype=np.float32),rtol=2e-7,atol=0)
    # WRF saves/restores prognostic QC/QI; a radiation view must not mutate it.
    np.testing.assert_array_equal(np.asarray(state.qc).ravel(),qc)
    np.testing.assert_array_equal(np.asarray(state.qi).ravel(),qi)
