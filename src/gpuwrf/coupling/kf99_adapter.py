"""Old-KF ARW phy_prep, held rates and the source-specific NCA lifecycle.

Input fields are from physics time-n. The existing step-entry scalar coupling
seam is retained; coupled-forecast D6 qualification remains pending. The old
advance_ppt clears six rates at NINT(NCA/DT)<=0; KF-eta uses <=1 instead.
"""
import jax
import jax.numpy as jnp
from gpuwrf.contracts.physics_registry import moist_species_for_mp
from gpuwrf.coupling.physics_couplers import (
    _dry_theta_view,_u_mass,_v_mass,_wrf_hydrostatic_pressure_profiles_from_state,
    _wrf_phy_prep_rho_from_state,_theta_m_tendency_from_dry,_output_dtype,
)
from gpuwrf.physics.cumulus_kf99 import kf99_column,update_w0avg99
from gpuwrf.coupling.kf_rk import kf_add_a2a

REAL=jnp.float32


def initial_kf99_carry(state):
    return jnp.zeros(state.theta.shape,REAL),jnp.full(state.theta.shape[1:],-100.,REAL)


def initial_kf99_tendencies(state):
    return (jnp.zeros(state.theta.shape,REAL),)*6+(jnp.zeros(state.theta.shape[1:],REAL),)


def kf99_rates(state,dt,grid,w0avg,nca,held,*,stepcu=1,cudt=0.,mp_physics=8):
    nz,ny,nx=state.theta.shape
    def cols(a):
        a=jnp.asarray(a,REAL)
        return jnp.moveaxis(a,0,-1).reshape(ny*nx,a.shape[0])
    def back(a):return jnp.moveaxis(a.reshape(ny,nx,nz),-1,0)
    p=jnp.asarray(state.p,REAL)
    pi=(p/jnp.asarray(1.e5,REAL))**jnp.asarray(2./7.,REAL)
    temperature=jnp.asarray(_dry_theta_view(state),REAL)*pi
    ph=jnp.asarray(state.ph,REAL)/jnp.asarray(9.81,REAL)
    dz=ph[1:]-ph[:-1]  # no non-WRF minimum-thickness clamp
    pressure,_,_=_wrf_hydrostatic_pressure_profiles_from_state(state,grid.metrics,output_dtype=REAL)
    rho=_wrf_phy_prep_rho_from_state(state,grid.metrics,output_dtype=REAL)
    wnext=update_w0avg99(cols(w0avg),cols(state.w),dt,stepcu=stepcu,cudt=cudt)
    live=jnp.asarray(nca,REAL).reshape(ny*nx)
    species=moist_species_for_mp(int(mp_physics))
    def column(*a):
        return kf99_column(*a[:8],dt,float(grid.projection.dx_m),pi_phy=a[8],nca=a[9],
            warm_rain=int(mp_physics)==1,qi_flag='qi' in species,qs_flag='qs' in species,qr_flag='qr' in species)
    out=jax.vmap(column)(cols(temperature),cols(state.qv),cols(pressure),cols(dz),cols(rho),
                        wnext,cols(_u_mass(state)),cols(_v_mass(state)),cols(pi),live)
    names=('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN')
    rates=tuple(back(out[name]) for name in names)+(out['PRATEC'].reshape(ny,nx),)
    # KFCPS leaves all old rates untouched only for NCA>0.5*DT; equality runs.
    active=jnp.asarray(nca,REAL)>jnp.asarray(.5*float(dt),REAL)
    rates=tuple(jnp.where(active[None],old,new) for old,new in zip(held[:6],rates[:6],strict=True))+(jnp.where(active,held[6],rates[6]),)
    rates=tuple(rate if name in species or i==0 else jnp.zeros_like(rate)
                for i,(name,rate) in enumerate(zip(('theta','qv','qc','qr','qi','qs'),rates[:6],strict=True)))+(rates[6],)
    return rates,back(wnext),out['NCA'].reshape(ny,nx)


def kf99_refresh(entry,persistent,held,dt,grid,*,itimestep,stepcu,cudt,mp_physics):
    """cumulus_driver gate + KFCPS: the rates consumed this step, W0AVG and the post-call NCA."""
    if persistent is None or held is None:
        raise ValueError('old KF requires W0AVG/NCA and seven held restart rates (E78)')
    mean,nca=persistent
    step=jnp.asarray(itimestep,jnp.int32);stepcu=max(1,int(stepcu))
    run=(step==1)|(stepcu==1)|(jnp.mod(step,stepcu)==0)|(float(cudt)==0.)
    return jax.lax.cond(run,
        lambda _:kf99_rates(entry,dt,grid,mean,nca,held,stepcu=stepcu,cudt=cudt,mp_physics=mp_physics),
        lambda _:(held,mean,nca),None)


def kf99_advance_ppt(nca,dt):
    """advance_ppt KFSCHEME (module_physics_addtendc.F:2304-2322): clear mask and NCA-=DT."""
    clear=(nca>0.)&(jnp.floor(nca/jnp.asarray(dt,REAL)+jnp.asarray(.5,REAL))<=0.)
    return clear,jnp.where(nca>0.,nca-jnp.asarray(dt,REAL),nca)


def kf99_cadence_step(entry,state,persistent,held,dt,grid,*,itimestep,stepcu,cudt,mp_physics,
                      specified_or_nested=False,periodic_x=False):
    rates,mean,nca=kf99_refresh(entry,persistent,held,dt,grid,itimestep=itimestep,stepcu=stepcu,
                                cudt=cudt,mp_physics=mp_physics)
    applied=tuple(jnp.where(jnp.isfinite(rate),
        kf_add_a2a(rate,specified_or_nested=specified_or_nested,periodic_x=periodic_x),rate)
        for rate in rates[:6])
    names=('theta','qv','qc','qr','qi','qs')
    updates={name:(getattr(state,name)+float(dt)*rate).astype(_output_dtype(state,name))
             for name,rate in zip(names,applied,strict=True)}
    theta_rate=_theta_m_tendency_from_dry(applied[0],applied[1],entry.theta,entry.qv,state.theta.dtype)
    updates['theta']=(state.theta+float(dt)*theta_rate).astype(_output_dtype(state,'theta'))
    updates['rainc_acc']=(state.rainc_acc+float(dt)*jnp.asarray(rates[6],state.rainc_acc.dtype)).astype(state.rainc_acc.dtype)
    # advance_ppt accumulates rain before clearing scalar rates; PRATEC is held.
    clear,next_nca=kf99_advance_ppt(nca,dt)
    next_rates=tuple(jnp.where(clear[None],jnp.asarray(0.,REAL),rate) for rate in rates[:6])+(rates[6],)
    return state.replace(**updates),(mean,next_nca),next_rates
