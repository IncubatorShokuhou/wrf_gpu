"""Observation-nudging inputs and source lifetime at the WRF physics/RK seam."""
import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.assimilation.obs_driver import ObservationInputs, initial_observation_carry, observation_driver


def observation_nudging_active(namelist):
    return isinstance(getattr(namelist,'observation_nudging',None),ObservationInputs)


def surface_observation_aux(view,diag):
    """surface_driver's ratios and REGIME from this call's actual SFCLAY result.

    Capture before the LSM overwrites TH2/U10/V10. T_PHY uses nonhydrostatic
    pressure; P_PHY in the ratio is hydrostatic, so replacing it by theta is
    not the original calculation. Inputs are column-last, as used by SFCLAY.
    """
    dtype=jnp.asarray(diag.u10).dtype
    r=lambda x:jnp.asarray(x,dtype)  # noqa: E731
    u,v=r(view.u)[...,0],r(view.v)[...,0]
    p=r(view.p)[...,0]
    if view.t_air is None:raise ValueError('obs nudging requires the WRF T_PHY surface input')
    t=r(view.t_air)[...,0]
    ur=jnp.where(jnp.abs(diag.u10)>r(1e-10),u/diag.u10,r(1.2))
    vr=jnp.where(jnp.abs(diag.v10)>r(1e-10),v/diag.v10,r(1.2))
    cp=r(np.float32(np.float32(287)/np.float32(1004.5)) if dtype==jnp.float32 else 287/1004.5)
    tr=(t*(r(100000)*r(.001)/(p/jax.lax.optimization_barrier(r(1000))))**cp)/r(diag.th2)
    return ur,vr,tr,r(diag.regime)


def seed_observation_carry(namelist,state):
    from gpuwrf.coupling.fdda_rk import fdda_dtype
    return initial_observation_carry(namelist.observation_nudging,state.theta.shape,fdda_dtype())


def observation_step_sources(carry,before,pblh,surface_aux,base_state,namelist,lead_seconds):
    from gpuwrf.assimilation.fdda_grid import wrf_muu_muv
    from gpuwrf.coupling.fdda_rk import _metrics,_phy_prep_inputs,_real_glue
    from gpuwrf.coupling.physics_couplers import _mynn_column_from_state
    from gpuwrf.physics.mynn_pbl import _wrf_zw
    if carry is None:raise ValueError('obs_nudge_opt=1 restart lacks held observation carry (E78)')
    if surface_aux is None:raise ValueError('obs_nudge_opt=1 lacks this step\'s SFCLAY ratios/regime')
    real=_real_glue();dtype=jnp.float32 if real else jnp.float64
    r=lambda x:jnp.asarray(x,dtype)  # noqa: E731
    metrics=_metrics(namelist,real)
    theta,zw=_phy_prep_inputs(before,base_state,dtype,real)
    mu2=r(before.mu_perturbation)
    mub=r(base_state.mub) if base_state is not None else r(before.mu_total)-mu2
    mut=mub+mu2
    muu,muv=wrf_muu_muv(mu2,mub)
    # First/last faces are unused by nudob's root ownership. Preserve their
    # correct extent without creating a second full 3-D mass field.
    muu=jnp.pad(muu,((0,0),(1,1)),mode='edge')
    muv=jnp.pad(muv,((1,1),(0,0)),mode='edge')
    # GET_PBLH's actual MYNN geometry and PBLH output, not a separate PBL solve.
    column=_mynn_column_from_state(before,namelist.grid)
    heights=_wrf_zw(column.dz)
    k=jnp.arange(1,heights.shape[-1]+1)
    found=(k>=2)&(k<heights.shape[-1])&(heights>=r(pblh)[...,None])
    first=jnp.min(jnp.where(found,k,heights.shape[-1]+1),axis=-1)
    kpbl=jnp.where(first<=heights.shape[-1],first-1,2).astype(jnp.int32)
    shape=mut.shape
    ur,vr,tr,regime=(r(a).reshape(shape) for a in surface_aux)
    if base_state is None:raise ValueError('obs nudging requires explicit WRF base-state pressure')
    lead=jnp.asarray(lead_seconds,jnp.float64)
    xtime=r(lead/jax.lax.optimization_barrier(jnp.asarray(60,jnp.float64)))
    ktau=jnp.rint(lead/float(namelist.dt_s)).astype(jnp.int32)
    return observation_driver(carry,namelist.observation_nudging,ktau=ktau,xtime=xtime,
        dt_s=float(namelist.dt_s),dx=float(namelist.grid.projection.dx_m),
        u=r(before.u),v=r(before.v),tb=theta-r(300),qv=r(before.qv),
        pbase=r(base_state.pb),pp=r(before.p_perturbation),z=r(.5)*(zw[:-1]+zw[1:]),
        uratx=ur,vratx=vr,tratx=tr,kpbl=kpbl,regime=regime,pblh=r(pblh),
        terrain=r(namelist.grid.terrain_height),mut=mut,muu=muu,muv=muv,
        c1h=metrics.c1h,c2h=metrics.c2h,msfux=metrics.msfux,msfuy=metrics.msfuy,
        msfvx=metrics.msfvx,msfvy=metrics.msfvy,msftx=metrics.msftx,msfty=metrics.msfty)
