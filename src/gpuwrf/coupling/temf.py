"""Opt-in TEMF pair, WRF caller operands and resident Registry/history carry."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.physics.bl_temf import temf_column
from gpuwrf.physics.sf_temf import temf_surface

TEMF_PROFILES = ('te_temf','kh_temf','km_temf','shf_temf','qf_temf','uw_temf','vw_temf',
                 'wupd_temf','mf_temf','thup_temf','qtup_temf','qlup_temf','cf3d_temf')
TEMF_SURFACE = ('hd_temf','lcl_temf','hct_temf','cfm_temf','wm_temf')
SURFACE_HELD = (*TEMF_SURFACE,'pblh','kpbl','exch_temf','flhc','flqc','chs','chs2','cqs2',
                'u10','v10','t2','th2','q2','lh','psfc')


class TemfCarry(NamedTuple):
    profiles: dict
    surface: dict


def initial_temf_carry(state):
    nz, ny, nx = state.theta.shape
    profiles = {k:jnp.full((ny*nx,nz), 1e-3 if k=='te_temf' else 0., jnp.float32) for k in TEMF_PROFILES}
    surface = {k:jnp.zeros((ny*nx,), jnp.int32 if k=='kpbl' else jnp.float32) for k in SURFACE_HELD}
    surface['pblh'] = jnp.asarray(state.pblh,jnp.float32).reshape(-1)
    return TemfCarry(profiles,surface)


def phy_inputs(state, grid):
    """Literal REAL phy_prep views, including the physical (not EOS) density."""
    from gpuwrf.coupling.physics_couplers import (
        _wrf_phy_prep_p8w, _wrf_phy_prep_rho_from_state, _wrf_phy_rv_over_rd)
    r = jnp.float32
    nz,ny,nx=state.theta.shape
    cols=lambda a:jnp.moveaxis(jnp.asarray(a,r),0,-1).reshape(ny*nx,-1)
    qv=jnp.asarray(state.qv,r)
    th=jnp.asarray(state.theta,r)/(r(1.)+_wrf_phy_rv_over_rd(r)*qv)
    p=jnp.asarray(state.p,r)
    ex=(p/r(1e5))**(r(287.)/r(1004.5))
    zw=jnp.asarray(state.ph,r)/r(9.81)
    z=r(.5)*(zw[:-1]+zw[1:])
    u,v=jnp.asarray(state.u,r),jnp.asarray(state.v,r)
    if grid is None or getattr(grid,'metrics',None) is None:
        raise ValueError('TEMF requires WRF grid metrics for phy_prep density/pressure/Coriolis')
    # first_rk_step_part1.F:1130/1132: PBL P_PHY=grid%p_hyd, P8W=grid%p_hyd_w; PI_PHY stays non-hydrostatic.
    from gpuwrf.coupling.physics_couplers import _wrf_hydrostatic_pressure_profiles_from_state
    p_hyd,p8w,_psfc=_wrf_hydrostatic_pressure_profiles_from_state(state,grid.metrics,output_dtype=r)
    p8w=jnp.asarray(p8w,r)
    rho=_wrf_phy_prep_rho_from_state(state,grid.metrics,output_dtype=r)
    # The extra z memory entry is used only in a discarded top dzt assignment.
    zmemory=jnp.concatenate((z,z[-1:]+(z[-1:]-z[-2:-1])),axis=0)
    return dict(u=cols(r(.5)*(u[:,:,:-1]+u[:,:,1:])),v=cols(r(.5)*(v[:,:-1,:]+v[:,1:,:])),
                temperature=cols(th*ex),theta=cols(th),qv=cols(qv),qc=cols(state.qc),
                pressure=cols(p_hyd),p8w=cols(p8w[:-1]),exner=cols(ex),rho=cols(rho),z=cols(zmemory),
                terrain=jnp.asarray(grid.terrain_height,r).reshape(-1),
                roughness=jnp.asarray(state.roughness_m,r).reshape(-1),
                coriolis=jnp.asarray(grid.metrics.f,r).reshape(-1),
                psfc=jnp.asarray(p8w[0],r).reshape(-1))


def surface_step(state, grid, carry):
    """Surface driver: update TE(1), WM and physical/kinematic surface fields."""
    f=phy_inputs(state,grid)
    from gpuwrf.coupling.physics_couplers import _wrf_hydrostatic_pressure_from_state
    # first_rk_step_part1.F:633,637 sends p_hyd/p_hyd_w to surface_driver;
    # PI_PHY still comes from the nonhydrostatic pressure. PBL uses p_phy.
    phyd,psfc=_wrf_hydrostatic_pressure_from_state(state,grid.metrics,output_dtype=jnp.float32)
    f['pressure']=jnp.moveaxis(phyd,0,-1).reshape(f['u'].shape)
    f['psfc']=psfc.reshape(-1)
    kw={k:f[k][:,0] for k in ('u','v','theta','qv','pressure','exner','rho','z')}
    kw.update({k:f[k] for k in ('terrain','roughness','coriolis','psfc')})
    kw.update(tsk=jnp.asarray(state.t_skin,jnp.float32).reshape(-1),
              mavail=jnp.asarray(state.mavail,jnp.float32).reshape(-1),
              hfx=jnp.asarray(state.hfx,jnp.float32).reshape(-1),
              te_temf=carry.profiles['te_temf'][:,0],hd_temf=carry.surface['hd_temf'],wm_temf=carry.surface['wm_temf'])
    out=temf_surface(**kw)
    ny,nx=state.t_skin.shape
    back=lambda a:jnp.asarray(a,jnp.float32).reshape(ny,nx)
    rho=kw['rho']
    thflux=out['hfx']/(rho*jnp.float32(1004.5))
    qflux=out['qfx']/rho
    angle=jnp.arctan2(kw['v'],kw['u'])
    def put(k,a):
        return back(a).astype(getattr(state,k).dtype)
    next_state=state.replace(
        ustar=put('ustar',out['ust']),theta_flux=put('theta_flux',thflux),qv_flux=put('qv_flux',qflux),
        tau_u=put('tau_u',-jnp.cos(angle)*out['ust']**2),tau_v=put('tau_v',-jnp.sin(angle)*out['ust']**2),
        rhosfc=put('rhosfc',rho),fltv=put('fltv',(1.+.608*kw['qv'])*thflux+.608*kw['theta']*qflux),
        hfx=put('hfx',out['hfx']),qfx=put('qfx',out['qfx']),qsfc=put('qsfc',out['qsfc']))
    profiles=dict(carry.profiles,te_temf=carry.profiles['te_temf'].at[:,0].set(out['te_temf']))
    surface=dict(carry.surface,**{k:out[k] for k in SURFACE_HELD if k in out},psfc=kw['psfc'])
    return next_state,TemfCarry(profiles,surface)


class SurfaceDiagnostics(NamedTuple):
    fluxes: object
    hfx: object
    lh: object
    u10: object
    v10: object
    t2: object
    th2: object
    q2: object
    qsfc: object
    mol: object
    znt: object
    chs: object
    chs2: object
    cqs2: object
    ch: object
    cm: object


def noahmp_surface_diagnostics(state,temf,land):
    """Reuse TEMF instead of recomputing MYNN-SL inside Noah-MP.

    WRF Noah-MP CHXY/CMXY are its own retained coefficients; TEMF does not
    write those fields. They must not be replaced with invented MYNN values.
    """
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    ny,nx=state.t_skin.shape
    back=lambda k:temf.surface[k].reshape(ny,nx)
    sf=SurfaceFluxes(ustar=state.ustar,theta_flux=state.theta_flux,qv_flux=state.qv_flux,
                     tau_u=state.tau_u,tau_v=state.tau_v,rhosfc=state.rhosfc,fltv=state.fltv,
                     xland=state.xland,wspd=None)
    return SurfaceDiagnostics(sf,state.hfx,back('lh'),back('u10'),back('v10'),back('t2'),back('th2'),
                              back('q2'),state.qsfc,state.mol,state.roughness_m,back('chs'),back('chs2'),
                              back('cqs2'),land.ch,land.cm)


def pbl_step(state,grid,dt,carry):
    """Return PBL State, updated TEMF carry, and literal uncoupled WRF rates."""
    from gpuwrf.coupling.scan_adapters import _add_a2c_u_increment,_add_a2c_v_increment,_theta_m_tendency_from_dry
    kw=phy_inputs(state,grid)
    kw.pop('psfc')
    ncol=kw['u'].shape[0]
    kw.update(tskold=jnp.asarray(state.t_skin,jnp.float32).reshape(-1),
              hfx=jnp.asarray(state.hfx,jnp.float32).reshape(-1),qfx=jnp.asarray(state.qfx,jnp.float32).reshape(-1),
              dt=jnp.full((ncol,),dt,jnp.float32),carry=dict(carry.profiles,pblh=carry.surface['pblh'],kpbl=carry.surface['kpbl']))
    out=jax.vmap(lambda args:temf_column(**args))(kw)
    nz,ny,nx=state.theta.shape
    back=lambda a:jnp.moveaxis(a.reshape(ny,nx,nz),-1,0)
    rates={k:back(out[k]) for k in ('rublten','rvblten','rthblten','rqvblten','rqcblten')}
    r=jnp.float32
    tht=_theta_m_tendency_from_dry(rates['rthblten'],rates['rqvblten'],state.theta,state.qv,r)
    cast=lambda k,a:jnp.asarray(a).astype(getattr(state,k).dtype)
    next_state=state.replace(
        u=cast('u',_add_a2c_u_increment(state.u,r(dt)*rates['rublten'])),
        v=cast('v',_add_a2c_v_increment(state.v,r(dt)*rates['rvblten'])),
        theta=cast('theta',state.theta+r(dt)*tht),qv=cast('qv',state.qv+r(dt)*rates['rqvblten']),
        qc=cast('qc',state.qc+r(dt)*rates['rqcblten']),pblh=cast('pblh',out['pblh'].reshape(ny,nx)),
        ustar=cast('ustar',out['ust'].reshape(ny,nx)))
    profiles={k:out[k] for k in TEMF_PROFILES}
    surface=dict(carry.surface,**{k:out[k] for k in SURFACE_HELD if k in out})
    return next_state,TemfCarry(profiles,surface),rates


def history_fields(carry,shape):
    ny,nx=shape
    fields={k:jnp.moveaxis(v.reshape(ny,nx,-1),-1,0) for k,v in carry.profiles.items()}
    fields.update({k:carry.surface[k].reshape(ny,nx) for k in TEMF_SURFACE})
    fields.update({k.upper():carry.surface[k].reshape(ny,nx) for k in ('u10','v10','t2','th2','q2','lh','psfc')})
    return fields
