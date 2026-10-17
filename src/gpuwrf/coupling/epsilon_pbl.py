"""EEPS/KEPS carry and caller wiring (experimental compiler-zero semantics)."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.nesting.scalar_species import epsilon_pbl_scalars
from gpuwrf.contracts.precision import EPSILON_PBL_DIAGNOSTICS


class EpsilonSurfaceForcing(NamedTuple):
    """Actual surface-driver outputs; RMOL is 1/L, FM/FHH are resistances."""
    rmol: jax.Array
    wspd: jax.Array
    br: jax.Array
    fm: jax.Array
    fhh: jax.Array


class EpsilonPblDiagnostics(NamedTuple):
    exch_h: jax.Array
    exch_m: jax.Array
    wu: jax.Array
    wv: jax.Array
    wt: jax.Array
    wq: jax.Array


class EpsilonNoahSurface(NamedTuple):
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
    psfc: object


def epsilon_surface_step(state,grid,*,land=None,return_full=False):
    """One revised-MM5 call; retain its RMOL/WSPD/FM/FH for PBL and Noah-MP.

    This route currently qualifies sf_sfclay_physics=1. It does not reinterpret
    MYNN's PSIM/PSIH as FM/FHH or silently use a different surface-layer scheme.
    """
    from gpuwrf.coupling.physics_couplers import (_u_mass,_v_mass,_dry_theta_view,
        _wrf_hydrostatic_pressure_from_state)
    from gpuwrf.physics.sfclay_revised_mm5 import SfclayRevisedMM5ColumnState,sfclay_revised_mm5_run
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    with jax.enable_x64(False):
        real=lambda x:jnp.asarray(x,jnp.float32)
        theta=real(_dry_theta_view(state));p=real(state.p)
        pi=(p/100000.)**(jnp.float32(287.)/jnp.float32(1004.5))
        phyd,psfc=_wrf_hydrostatic_pressure_from_state(state,grid.metrics,output_dtype=jnp.float32)
        dz=(real(state.ph[1])-real(state.ph[0]))/jnp.float32(9.81)
        column=SfclayRevisedMM5ColumnState(real(_u_mass(state)[0]),real(_v_mass(state)[0]),
            theta[0]*pi[0],real(state.qv[0]),real(phyd[0]),dz,real(psfc),real(state.t_skin),
            real(state.xland),real(state.lakemask),real(state.mavail),real(state.roughness_m),
            real(state.ustar),real(state.mol),real(state.hfx),real(state.qfx),real(state.qsfc),
            real(state.pblh),jnp.float32(grid.projection.dx_m),jnp.float32(50.))
        out=sfclay_revised_mm5_run(column)
        updates={k:real(getattr(out,v)) for k,v in {'ustar':'ust','theta_flux':'theta_flux','qv_flux':'qv_flux',
                 'tau_u':'tau_u','tau_v':'tau_v','rhosfc':'rhosfc','fltv':'fltv','hfx':'hfx','qfx':'qfx',
                 'qsfc':'qsfc','mol':'mol','roughness_m':'znt'}.items()}
        next_state=state.replace(_cast=False,**updates)
        forcing=EpsilonSurfaceForcing(*(real(getattr(out,k)) for k in ('rmol','wspd','br','fm','fh')))
        sf=SurfaceFluxes(ustar=next_state.ustar,theta_flux=next_state.theta_flux,qv_flux=next_state.qv_flux,
            tau_u=next_state.tau_u,tau_v=next_state.tau_v,rhosfc=next_state.rhosfc,
            # Noah's optional WSPD leaf is MYNN's State.sfc_wspd feedback.
            # EEPS consumes the actual WSPD directly from `forcing` above.
            fltv=next_state.fltv,xland=next_state.xland,wspd=None)
        # No-LSM history also reads these actual revised-MM5 outputs.
        noah=EpsilonNoahSurface(sf,next_state.hfx,real(out.lh),real(out.u10),real(out.v10),real(out.t2),
            real(out.th2),real(out.q2),next_state.qsfc,next_state.mol,next_state.roughness_m,
            real(out.chs),real(out.chs2),real(out.cqs2),
            None if land is None else land.ch,None if land is None else land.cm,real(psfc))
    return (next_state,forcing,noah,out) if return_full else (next_state,forcing,noah)


def seed_epsilon_pbl_state(state, bl_pbl_physics):
    """WRF EEPSINIT / KEPS step1: physical minima; Registry scalar init zero."""
    bl=int(bl_pbl_physics)
    if bl not in (16,17): return state
    if state.epsilon_pbl is not None:
        return state.replace(_cast=False,qke=state.qke.astype(jnp.float32))
    def full(v): return jnp.full_like(state.qv,v,dtype=jnp.float32)
    updates=dict(qke=full(1.e-4),epsilon_pbl=full(1.e-7),epsilon_pbl_step=jnp.int32(0))
    if bl==17: updates.update(tpe_pbl=full(1.e-7),pr_pbl=full(1.))
    updates.update({name:full(0.) for name in epsilon_pbl_scalars(bl)})
    updates.update({name:full(0.) for name in EPSILON_PBL_DIAGNOSTICS})
    return state.replace(_cast=False,**updates)


def initial_epsilon_diagnostics(state):
    return EpsilonPblDiagnostics(*(jnp.zeros_like(state.qv,dtype=jnp.float32) for _ in range(6)))


def epsilon_pbl_adapter(state,dt,grid,*,scheme,step,rthraten,surface_forcing):
    """WRF phy_prep operands -> original column driver -> update_phy_ten.

    Surface inputs must be from the surface call that produced this step's
    HFX/QFX/UST. No second surface solve or substitution is performed here.
    """
    from gpuwrf.coupling.physics_couplers import (
        _u_mass,_v_mass,_dry_theta_view,_theta_m_tendency_from_dry,
        _add_a2c_u_increment,_add_a2c_v_increment,_wrf_phy_prep_rho_from_state,_wrf_phy_prep_p8w,
        _wrf_hydrostatic_pressure_from_state,_wrf_hydrostatic_pressure_profiles_from_state,
    )
    from gpuwrf.physics.pbl_eeps import eeps_column
    from gpuwrf.physics.pbl_keps import keps_column
    from gpuwrf.io.pbl_epsilon_schema import validate_epsilon_state,EPSILON_PBL_SCHEMA_VERSION
    validate_epsilon_state(state,{'bl_pbl_physics':int(scheme)},EPSILON_PBL_SCHEMA_VERSION,check_values=False)
    nz,ny,nx=state.qv.shape;ncol=ny*nx
    # Persist the WRF call counter in State so daily State-only chunk callbacks
    # and State checkpoints cannot repeat KEPS's first-two-call initialization.
    step=state.epsilon_pbl_step+jnp.int32(1)
    def cols(x,n=nz):return jnp.moveaxis(jnp.asarray(x,jnp.float32),0,-1).reshape(ncol,n)
    def flat(x):return jnp.asarray(x,jnp.float32).reshape(ncol)
    def back(x):return jnp.moveaxis(x.reshape(ny,nx,nz),-1,0)
    with jax.enable_x64(False):
        theta=jnp.asarray(_dry_theta_view(state),jnp.float32)
        p=jnp.asarray(state.p,jnp.float32)
        exner=(p/100000.)**(jnp.float32(287.)/jnp.float32(1004.5))
        z=jnp.asarray(state.ph,jnp.float32)/jnp.float32(9.81)
        dz=z[1:]-z[:-1]
        rho=_wrf_phy_prep_rho_from_state(state,grid.metrics,output_dtype=jnp.float32)
        common=dict(u=cols(_u_mass(state)),v=cols(_v_mass(state)),theta=cols(theta),qv=cols(state.qv),
                    qc=cols(state.qc),rho=cols(rho),dz=cols(dz),exner=cols(exner),raten=cols(rthraten),
                    tke=cols(state.qke),epsilon=cols(state.epsilon_pbl),
                    dt=jnp.full((ncol,),dt,jnp.float32),step=jnp.full((ncol,),step,jnp.int32),
                    hfx=flat(state.hfx),qfx=flat(state.qfx),ustar=flat(state.ustar))
        f=surface_forcing
        if int(scheme)==16:
            _,psfc=_wrf_hydrostatic_pressure_from_state(state,grid.metrics,output_dtype=jnp.float32)
            p_hyd,_p8w,_psfc=_wrf_hydrostatic_pressure_profiles_from_state(state,grid.metrics,output_dtype=jnp.float32)
            common.update(temperature=cols(theta*exner),pressure=cols(p_hyd),qi=cols(state.qi),qr=cols(state.qr),
                          qs=cols(state.qs),qg=cols(state.qg),tke_adv=cols(state.pek_adv),epsilon_adv=cols(state.pep_adv),
                          rmol=flat(f.rmol),wspd=flat(f.wspd),tsk=flat(state.t_skin),qsfc=flat(state.qsfc),
                          psfc=flat(psfc),dx=jnp.full((ncol,),grid.projection.dx_m,jnp.float32))
            out=jax.vmap(lambda kw:eeps_column(**kw))(common)
            updates={'pek_adv':back(out['tke_adv']),'pep_adv':back(out['epsilon_adv'])}
        else:
            _p_hyd,p8w,_psfc=_wrf_hydrostatic_pressure_profiles_from_state(state,grid.metrics,output_dtype=jnp.float32)
            common.update(p8w=cols(jnp.asarray(p8w,jnp.float32),nz+1),tpe=cols(state.tpe_pbl),prandtl=cols(state.pr_pbl),
                          tke_adv=cols(state.tke_adv),epsilon_adv=cols(state.diss_adv),tpe_adv=cols(state.tpe_adv),
                          roughness=flat(state.roughness_m),br=flat(f.br),fm=flat(f.fm),fhh=flat(f.fhh))
            out=jax.vmap(lambda kw:keps_column(**kw))(common)
            updates=dict(tpe_pbl=back(out['tpe']),pr_pbl=back(out['prandtl']),tke_adv=back(out['tke_adv']),
                         diss_adv=back(out['epsilon_adv']),tpe_adv=back(out['tpe_adv']))
        rq=back(out['rqvblten'])
        th=_theta_m_tendency_from_dry(back(out['rthblten']),rq,state.theta,state.qv,jnp.float32)
        updates.update(qke=back(out['tke']),epsilon_pbl=back(out['epsilon']),epsilon_pbl_step=step,pblh=out['pblh'].reshape(ny,nx),
                       u=_add_a2c_u_increment(state.u,dt*back(out['rublten'])),
                       v=_add_a2c_v_increment(state.v,dt*back(out['rvblten'])),
                       theta=state.theta+dt*th,qv=state.qv+dt*rq,
                       qc=state.qc+dt*back(out['rqcblten']),qi=state.qi+dt*back(out['rqiblten']))
        diag=EpsilonPblDiagnostics(*(back(out[k]) for k in EpsilonPblDiagnostics._fields))
        updates.update(dict(zip(EPSILON_PBL_DIAGNOSTICS,diag,strict=True)))
    rates={name:back(out[name]) for name in ('rublten','rvblten','rthblten','rqvblten','rqcblten','rqiblten')}
    updated=state.replace(**updates)
    if state.cam_kvh3d is not None:
        from gpuwrf.physics.cammg.pbl import retain_mass
        updated=retain_mass(updated,exch_h=diag.exch_h,tke=back(out['tke']) if int(scheme)==17 else None)
    return updated,diag,rates
