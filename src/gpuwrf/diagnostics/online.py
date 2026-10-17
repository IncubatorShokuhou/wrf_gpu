"""Caller lifetimes for opt-in WRF NWP and Thompson radar diagnostics."""
from collections.abc import Mapping
import math

import jax
import jax.numpy as jnp


def configuration(namelist):
    if isinstance(namelist, Mapping):
        physics=namelist.get("physics",namelist)
        time=namelist.get("time_control",namelist)
        mp=physics.get('mp_physics',8)
        mp=mp[0] if isinstance(mp,(list,tuple)) else mp
        return dict(nwp_diagnostics=int(time.get("nwp_diagnostics",0)),
                    do_radar_ref=_radar_setting(physics.get("do_radar_ref",0),mp),
                    history_interval=float(time.get("history_interval",namelist.get("diagnostic_history_interval",60.))))
    return dict(nwp_diagnostics=int(getattr(namelist,"nwp_diagnostics",0)),
                do_radar_ref=_radar_setting(getattr(namelist,"do_radar_ref",0),getattr(namelist,"mp_physics",8)),
                history_interval=float(getattr(namelist,"diagnostic_history_interval",60.)))


def _radar_setting(value,mp):
    value=int(value)
    # Force MY2's unconditional diagnostic while preserving invalid selectors
    # so validate_options can still reject them for every scheme.
    # check_a_mundo also allocates radar for NSSL_2MOM (legacy17/19/21/22
    # normalize to18 in WRF); P3 and Jensen (mp55) packages allocate refl_10cm and
    # their drivers fill it every call. The public do_radar_ref value stays unchanged.
    return 1 if int(mp) in (9,17,18,19,21,22,50,51,52,53,55) and value in (0,1) else value


def validate_options(namelist, *, specified, nested, periodic):
    cfg=configuration(namelist)
    if cfg["nwp_diagnostics"] not in (0,1) or cfg["do_radar_ref"] not in (0,1):
        raise ValueError("nwp_diagnostics and do_radar_ref support only 0/1")
    if int(namelist.mp_physics) in (9,17,18,19,21,22,50,51,52,53,55) and not cfg['nwp_diagnostics']:
        if not namelist.run_physics:
            raise ValueError('Microphysics radar requires physics enabled')
        return  # WRF MY2 diagnoses every step, independently of history alarms.
    if not (cfg["nwp_diagnostics"] or cfg["do_radar_ref"]):
        return
    if cfg["nwp_diagnostics"] and not cfg["do_radar_ref"]:
        raise ValueError("NWP diagnostics require do_radar_ref=1, as enforced by WRF check_a_mundo")
    if not math.isfinite(cfg["history_interval"]) or cfg["history_interval"] <= 0:
        raise ValueError("online diagnostics require a positive finite history_interval")
    mp=int(namelist.mp_physics)
    if not namelist.run_physics or mp not in (8,9,10,40) or (mp in (9,10,40) and cfg['nwp_diagnostics']):
        raise ValueError("online NWP diagnostics require Thompson mp_physics=8; radar-only also admits "
                         "Morrison mp_physics=10/40; physics must be enabled")
    if namelist.dt_s <= 0 or cfg["history_interval"]*60 < namelist.dt_s:
        raise ValueError("online diagnostics require positive fixed dt no larger than history interval")
    if cfg["nwp_diagnostics"] and (periodic or not (specified or nested)):
        raise ValueError("NWP helicity currently requires a full specified/nested nonperiodic domain")


def history_due(step, dt, minutes):
    # Is_alarm_tstep: current+dt crosses the next ring (module_domain.F2484).
    clock=jnp.asarray(step,jnp.float64)*float(dt)
    period=float(minutes)*60.
    return jnp.floor(clock/period)>jnp.floor((clock-float(dt))/period)


def ph_parts(state, base):
    ph=jnp.asarray(state.ph_perturbation,jnp.float32)
    phb=(jnp.asarray(base.phb,jnp.float32) if base is not None else
         (jnp.asarray(state.ph_total,jnp.float64)-jnp.asarray(state.ph_perturbation,jnp.float64)).astype(jnp.float32))
    return ph,phb


def _prep_fields(state,base,namelist, *, moist):
    """Literal calc_p_rho_phi geometry and phy/moist_physics_prep mediation.

    Keep perturbation/base differences separate; the intermediate al rounding
    before adding alb is part of WRF's REAL arithmetic, not an algebraic no-op.
    """
    from gpuwrf.coupling.physics_couplers import _wrf_phy_prep_p8w
    from gpuwrf.runtime.operational_mode import _base_state_from_totals
    base=_base_state_from_totals(state) if base is None else base
    r=lambda a:jnp.asarray(a,jnp.float32)
    m=namelist.metrics
    ph,phb=ph_parts(state,base)
    mub=r(base.mub)[None]
    mut=jax.lax.optimization_barrier(r(base.mub)+r(state.mu_perturbation))[None]
    dbase=phb[1:]-phb[:-1]
    if int(namelist.hypsometric_opt)==2:
        def alpha(dph,mass):
            top=r(m.p_top).reshape(())
            up=r(m.c3f)[1:,None,None]*mass+r(m.c4f)[1:,None,None]+top
            down=r(m.c3f)[:-1,None,None]*mass+r(m.c4f)[:-1,None,None]+top
            half=r(m.c3h)[:,None,None]*mass+r(m.c4h)[:,None,None]+top
            return dph/half/jnp.log(down/up)
        alb=alpha(dbase,mub)
        dph=((ph[1:]-ph[:-1])+phb[1:])-phb[:-1]
        al=jax.lax.optimization_barrier(alpha(dph,mut)-alb)
    elif int(namelist.hypsometric_opt)==1:
        c1,c2=r(m.c1h)[:,None,None],r(m.c2h)[:,None,None]
        alb=-dbase/(r(m.dnw)[:,None,None]*(c1*mub+c2))
        al=jax.lax.optimization_barrier(-r(1.)/(c1*mut+c2)*(alb*(c1*r(state.mu_perturbation)[None])+
                      r(m.rdnw)[:,None,None]*(ph[1:]-ph[:-1])))
    else:
        raise ValueError("online diagnostics require WRF hypsometric_opt=1/2")
    alt=al+alb
    rho=r(1.)/alt if moist else r(1.)/alt*(r(1.)+r(state.qv))
    p=r(state.p_perturbation)+r(base.pb)
    p8w=_wrf_phy_prep_p8w(p,(ph+phb)/r(9.81),m.fnm,m.fnp)
    return p8w,rho


def nwp_inputs(entry,pre_mp,base,namelist,spec_zone):
    old=_prep_fields(entry,base,namelist,moist=False)
    new=_prep_fields(pre_mp,base,namelist,moist=True)
    if not spec_zone:
        return new
    ny,nx=entry.t_skin.shape
    y,x=jnp.arange(ny)[:,None],jnp.arange(nx)[None,:]
    inside=(y>=spec_zone)&(y<ny-spec_zone)&(x>=spec_zone)&(x<nx-spec_zone)
    return tuple(jnp.where(inside[None],n,o) for n,o in zip(new,old))


def before_rk(carry,namelist):
    from gpuwrf.diagnostics.nwp import update_helicity
    s,m=carry.state,namelist.metrics
    ph,phb=ph_parts(s,carry.base_state)
    return carry.replace(nwp_diagnostics=update_helicity(carry.nwp_diagnostics,
        u=s.u,v=s.v,w=s.w,ph=ph,phb=phb,msfux=m.msfux,msfvy=m.msfvy,
        terrain=namelist.grid.terrain_height,rdx=1./namelist.grid.projection.dx_m,
        rdy=1./namelist.grid.projection.dy_m,dn=m.dn,dnw=m.dnw,fnm=m.fnm,fnp=m.fnp,
        cf1=m.cf1,cf2=m.cf2,cf3=m.cf3))


def radar_from_context(previous,context,namelist,step,spec_zone):
    mp=int(getattr(namelist,'mp_physics',8))
    if mp in (9,17,18,19,21,22,50,51,52,53,55):
        # MY2/P3/Jensen supply Zet every step; NSSL supplies its selected S5 levels.
        result=context
        if mp in (17,18,19,21,22):
            full=history_due(step,namelist.dt_s,namelist.diagnostic_history_interval)
            bottom=jnp.equal(step,1) | bool(namelist.nwp_diagnostics)
            levels=jnp.arange(previous.shape[0])[:,None,None]
            result=jnp.where(full | (bottom & (levels==0)),context,previous)
        if spec_zone:
            ny,nx=previous.shape[-2:]
            y,x=jnp.arange(ny)[:,None],jnp.arange(nx)[None,:]
            inside=(y>=spec_zone)&(y<ny-spec_zone)&(x>=spec_zone)&(x<nx-spec_zone)
            result=jnp.where(inside[None],result,previous)
        return result
    if mp in (10,40):
        # MP_MORR_TWO_MOMENT[_AERO]: full-column refl10cm_hm only when diagflag (history
        # alarm; nwp_diagnostics is refused), MAX(-35.,dBZ); otherwise refl_10cm is retained.
        from gpuwrf.diagnostics.morrison_radar import morrison_refl_10cm
        nz,ny,nx=previous.shape
        def morrison():
            value=morrison_refl_10cm(*context)
            return jnp.moveaxis(value.reshape(ny,nx,nz),-1,0).astype(previous.dtype)
        result=jax.lax.cond(history_due(step,namelist.dt_s,namelist.diagnostic_history_interval),
                            morrison,lambda:previous)
        if spec_zone:
            y,x=jnp.arange(ny)[:,None],jnp.arange(nx)[None,:]
            inside=(y>=spec_zone)&(y<ny-spec_zone)&(x>=spec_zone)&(x<nx-spec_zone)
            result=jnp.where(inside[None],result,previous)
        return result
    from gpuwrf.diagnostics.thompson_radar import thompson_reflectivity
    nz=previous.shape[0]
    def compute(depth):
        return jnp.moveaxis(thompson_reflectivity(*context,ke_diag=depth),-1,0)
    ordinary=(lambda:compute(1)) if namelist.nwp_diagnostics else (lambda:previous)
    result=jax.lax.cond(history_due(step,namelist.dt_s,namelist.diagnostic_history_interval),
                        lambda:compute(nz),ordinary)
    if spec_zone:
        ny,nx=previous.shape[-2:]
        y,x=jnp.arange(ny)[:,None],jnp.arange(nx)[None,:]
        inside=(y>=spec_zone)&(y<ny-spec_zone)&(x>=spec_zone)&(x<nx-spec_zone)
        result=jnp.where(inside[None],result,previous)
    return result


def after_step(carry,namelist,step,inputs):
    from gpuwrf.diagnostics.nwp import update_nwp
    s=carry.state
    ph,phb=ph_parts(s,carry.base_state)
    p8w,rho=inputs
    refl=carry.radar_reflectivity
    if refl is None:
        refl=jnp.zeros_like(s.qv,dtype=jnp.float32)
    h=carry.history_diagnostics
    return carry.replace(nwp_diagnostics=update_nwp(carry.nwp_diagnostics,
        u10=h.u10,v10=h.v10,w=s.w,p8w=p8w,znw=namelist.grid.eta_levels,qg=s.qg,
        rho=rho,ph=ph,phb=phb,refl_10cm=refl,itimestep=step,dt=namelist.dt_s,
        history_interval=namelist.diagnostic_history_interval))


def validate_restart(nwp,radar,config,version,shape):
    from gpuwrf.diagnostics.nwp import validate_restart_nwp
    if config["nwp_diagnostics"] and not config["do_radar_ref"]:
        raise ValueError("restart NWP diagnostics require do_radar_ref=1 (E78)")
    active=bool(config["nwp_diagnostics"] or config["do_radar_ref"] or nwp is not None or radar is not None)
    if active and version!=1:
        raise ValueError("restart online diagnostics schema is missing (E78)")
    if bool(nwp is not None)!=bool(config["nwp_diagnostics"]):
        raise ValueError("restart NWP carry and selection disagree (E78)")
    if bool(radar is not None)!=bool(config["do_radar_ref"]):
        raise ValueError("restart radar carry and selection disagree (E78)")
    validate_restart_nwp(nwp,config["nwp_diagnostics"],version,shape[-2:])
    if config["do_radar_ref"] and radar is None:
        raise ValueError("restart lacks held REFL_10CM (E78)")
    if radar is not None and (getattr(radar,"dtype",None)!=jnp.float32 or radar.shape!=tuple(shape)):
        raise ValueError("restart REFL_10CM differs from the REAL mass grid (E78)")
