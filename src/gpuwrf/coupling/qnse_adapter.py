"""WRF QNSESFC -> land -> optional MFSHCONVPBL -> QNSEPBL coupling.

The surface solve executes once. Its INOUT exchange/surface memory is carried
across steps and restart; PBL reads that solve and the updated physical land
fluxes. REAL4 is explicit throughout this adapter.
"""
from typing import NamedTuple

import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import (
    _u_mass, _v_mass, _wrf_phy_prep_p8w, _wrf_phy_prep_rho_from_state,
    _add_a2c_u_increment, _add_a2c_v_increment, _theta_m_tendency_from_dry,
    _wrf_hydrostatic_pressure_profiles_from_state,
)
from gpuwrf.kernels.phys_qnse_columns import qnse_columns
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes


class QnseCarry(NamedTuple):
    akhs: object
    akms: object
    thz0: object
    qz0: object
    uz0: object
    vz0: object
    ct: object
    z0base: object
    lh: object
    rc_mf: object
    exch_h: object
    exch_m: object


class QnseSurfaceDiagnostics(NamedTuple):
    fluxes: object
    hfx: object
    lh: object
    u10: object
    v10: object
    t2: object
    q2: object
    qsfc: object
    mol: object
    znt: object
    chs: object
    chs2: object
    cqs2: object
    cpm: object
    ch: object
    psfc: object
    u10e: object
    v10e: object


def initial_qnse_carry(state):
    zero=jnp.zeros(state.t_skin.shape,jnp.float32)
    profile=jnp.zeros(state.theta.shape,jnp.float32)
    return QnseCarry(zero,zero,zero,zero,zero,zero,zero,
                     jnp.asarray(state.roughness_m,jnp.float32),zero,profile,profile,profile)


def _geometry(state,grid):
    nz,ny,nx=state.theta.shape
    f=jnp.float32
    qv=jnp.asarray(state.qv,f)
    theta=jnp.asarray(state.theta,f)/(f(1.)+(f(461.6)/f(287.))*qv)
    pressure=jnp.asarray(state.p,f)
    exner=(pressure/f(1.e5))**(f(287.)/f(1004.5))
    z=jnp.asarray(state.ph_total,f)/f(9.81)
    metrics=getattr(grid,'metrics',None)
    if metrics is None:
        raise ValueError('QNSE requires WRF grid metrics for phy_prep P8W/RHO')
    # first_rk_step_part1.F:633/637,1130/1132: P_PHY=grid%p_hyd, P8W=grid%p_hyd_w (hydrostatic).
    p_hyd,pint,_psfc=_wrf_hydrostatic_pressure_profiles_from_state(state,metrics,output_dtype=f)
    pint=jnp.asarray(pint,f)
    profiles=dict(theta=theta,temperature=theta*exner,qv=qv,qc=jnp.asarray(state.qc,f),
                  p_mid=p_hyd,p_int=pint,exner=exner,dz=z[1:]-z[:-1],
                  u=_u_mass(state,dtype=f),v=_v_mass(state,dtype=f),tke=f(.5)*jnp.asarray(state.qke,f),
                  rho=_wrf_phy_prep_rho_from_state(state,metrics,output_dtype=f))
    profiles={k:a.reshape(a.shape[0],ny*nx) for k,a in profiles.items()}
    return profiles,z[0],jnp.asarray(metrics.f,f), (nz,ny,nx)


def qnse_surface_step(state,carry,dt,grid,*,first_timestep=False,with_psfc=True):
    if carry is None:raise ValueError('QNSE surface carry is missing (E78)')
    g,ht,_,shape=_geometry(state,grid)
    nz,ny,nx=shape
    flat=lambda a:jnp.asarray(a,jnp.float32).reshape(ny*nx)
    inputs={k:g[k] for k in ('u','v','temperature','theta','qv','qc','p_mid','dz')}
    inputs.update(q2=g['tke'],tsk=flat(state.t_skin),xland=flat(state.xland),
                  z0base=flat(carry.z0base),psfc=g['p_int'][0],znt=flat(state.roughness_m),
                  ustar=flat(state.ustar),mavail=flat(state.mavail),qsfc=flat(state.qsfc),
                  ht=flat(ht),dt=dt,itimestep=jnp.where(first_timestep,jnp.float32(1),jnp.float32(2)),
                  hfx=flat(state.hfx),qfx=flat(state.qfx),scm_force_flux=0)
    inputs.update({k:flat(getattr(carry,k)) for k in ('thz0','qz0','uz0','vz0','akhs','akms')})
    out=qnse_columns('sfc',inputs)
    r=lambda a:a.reshape(ny,nx)
    surface={k:r(v) for k,v in out.items()}
    rho=g['p_mid'][0]/(jnp.float32(287.)*g['temperature'][0])
    thflux=surface['hfx']/(r(rho)*jnp.float32(1004.5))
    qflux=surface['qfx']/r(rho)
    fluxes=SurfaceFluxes(ustar=surface['ustar'],theta_flux=thflux,qv_flux=qflux,
        tau_u=r(out['akms']*(g['u'][0]-out['uz0'])),tau_v=r(out['akms']*(g['v'][0]-out['vz0'])),
        rhosfc=r(rho),fltv=thflux,xland=jnp.asarray(state.xland,jnp.float32),
        t_skin=jnp.asarray(state.t_skin,jnp.float32),
        wspd=r(jnp.maximum(jnp.sqrt(g['u'][0]**2+g['v'][0]**2),jnp.float32(.001))))
    updates={k:getattr(fluxes,k).astype(getattr(state,k).dtype)
             for k in ('ustar','theta_flux','qv_flux','tau_u','tau_v','rhosfc','fltv')}
    updates.update({key:surface[source].astype(getattr(state,key).dtype) for key,source in
                    [('hfx','hfx'),('qfx','qfx'),('qsfc','qsfc'),('roughness_m','znt'),('pblh','pblh')]})
    next_carry=carry._replace(**{k:surface[k] for k in ('thz0','qz0','uz0','vz0','akhs','akms','ct')},lh=surface['flx_lh'])
    diag=QnseSurfaceDiagnostics(fluxes,surface['hfx'],surface['flx_lh'],surface['u10'],surface['v10'],
        surface['t02'],surface['q02'],surface['qsfc'],jnp.asarray(state.mol,jnp.float32),
        surface['znt'],surface['chs'],surface['chs2'],surface['cqs2'],surface['cpm'],surface['chs'],
        r(g['p_int'][0]) if with_psfc else None,surface['u10e'],surface['v10e'])
    return state.replace(**updates),next_carry,diag


def qnse_pbl_step(state,carry,dt,grid,*,mfshconv=1,snow=0.,sice=0.,apply_tendencies=True):
    """Return state, held memory and the five ORIGINAL dry WRF source arrays."""
    if carry is None:raise ValueError('QNSE PBL carry is missing (E78)')
    if mfshconv not in (0,1):raise ValueError('QNSE mfshconv must be 0 or 1')
    g,ht,corf,shape=_geometry(state,grid)
    nz,ny,nx=shape
    flat=lambda a:jnp.broadcast_to(jnp.asarray(a,jnp.float32),(ny,nx)).reshape(ny*nx)
    sources={}
    if mfshconv:
        mf_inputs={k:g[k] for k in ('theta','qv','p_mid','exner','dz','rho','tke','u','v')}
        mf_inputs.update(hfx=flat(state.hfx),qfx=flat(state.qfx),rc_mf=carry.rc_mf.reshape(nz,ny*nx),dt=dt,stepbl=1)
        # OMIXUV=False and WRF's final RQC assignment make the other three
        # rates identically zero. Export only consumed fields: 3 instead of
        # 19 full profiles; diagnostic columns remain available to the oracle.
        mf=qnse_columns('mf',mf_inputs,requested=('RTHBLTEN','RQVBLTEN','rc_mf'))
        sources={k:mf[k] for k in ('RTHBLTEN','RQVBLTEN')}
        carry=carry._replace(rc_mf=mf['rc_mf'].reshape(shape))
    inputs={k:g[k] for k in ('u','v','temperature','theta','qv','qc','p_mid','p_int','exner','dz','tke')}
    inputs.update(xland=flat(state.xland),ustar=flat(state.ustar),hfx=flat(state.hfx),elflx=flat(carry.lh),
                  qsfc=flat(state.qsfc),corf=flat(corf),snow=flat(snow),sice=flat(sice),chklowq=1.,
                  dt=dt,stepbl=1,ht=flat(ht),EXCH_H=carry.exch_h.reshape(nz,ny*nx),
                  EXCH_M=carry.exch_m.reshape(nz,ny*nx),**sources)
    inputs.update({k:flat(getattr(carry,k)) for k in ('akhs','akms','thz0','qz0','uz0','vz0','ct')})
    out=qnse_columns('pbl',inputs)
    raw={k:out[k].reshape(shape) for k in ('RUBLTEN','RVBLTEN','RTHBLTEN','RQVBLTEN','RQCBLTEN')}
    updates=dict(qke=(jnp.float32(2.)*out['tke'].reshape(shape)).astype(state.qke.dtype),
                 pblh=out['PBLH'].reshape(ny,nx).astype(state.pblh.dtype),
                 qsfc=out['qsfc'].reshape(ny,nx).astype(state.qsfc.dtype))
    if getattr(state,'el_pbl',None) is not None:updates['el_pbl']=out['EL_MYJ'].reshape(shape).astype(state.el_pbl.dtype)
    carry=carry._replace(thz0=out['thz0'].reshape(ny,nx),qz0=out['qz0'].reshape(ny,nx),
        ct=out['ct'].reshape(ny,nx),exch_h=out['EXCH_H'].reshape(shape),exch_m=out['EXCH_M'].reshape(shape))
    if apply_tendencies:
        f=jnp.float32
        updates.update(u=_add_a2c_u_increment(state.u,f(dt)*raw['RUBLTEN']).astype(state.u.dtype),
                       v=_add_a2c_v_increment(state.v,f(dt)*raw['RVBLTEN']).astype(state.v.dtype),
                       theta=(state.theta+f(dt)*_theta_m_tendency_from_dry(raw['RTHBLTEN'],raw['RQVBLTEN'],
                           jnp.asarray(state.theta,f),jnp.asarray(state.qv,f),f)).astype(state.theta.dtype),
                       qv=(state.qv+f(dt)*raw['RQVBLTEN']).astype(state.qv.dtype),
                       qc=(state.qc+f(dt)*raw['RQCBLTEN']).astype(state.qc.dtype))
    updated=state.replace(**updates)
    if state.cam_kvh3d is not None:
        from gpuwrf.physics.cammg.pbl import retain_mass
        updated=retain_mass(updated,exch_h=out['EXCH_H'].reshape(shape),tke=out['tke'].reshape(shape))
    return updated,carry,raw
