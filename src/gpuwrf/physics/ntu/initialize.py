"""Cold NTU_INIT/INIT_AEROSOL initialization, original REAL operation order.

Inputs use (..., mass level), geopotential has one extra interface, and ALT
is exactly the inverse-density argument of original physics_init. The final
unused interface follows the project's explicit compiler-zero policy.
"""
import numpy as np
from .tables import selected_tables


def aerosols(pressure,inverse_density,ph,phb,qv,xland,*,ccnty=2,xp=np):
    p=xp.asarray(pressure);r=np.dtype(p.dtype).type
    alt=xp.asarray(inverse_density,dtype=p.dtype);qv=xp.asarray(qv,dtype=p.dtype)
    if alt.shape!=p.shape or qv.shape!=p.shape:raise ValueError('NTU initialization mass-grid shapes differ')
    ph=xp.asarray(ph,dtype=p.dtype);phb=xp.asarray(phb,dtype=p.dtype)
    expected=p.shape[:-1]+(p.shape[-1]+1,)
    if ph.shape!=expected or phb.shape!=expected:raise ValueError('NTU initialization needs staggered PH/PHB')
    if np.shape(xland)!=p.shape[:-1]:raise ValueError('NTU initialization XLAND shape differs')
    c=selected_tables(p.dtype,ccnty)
    height=(phb+ph)/c['g']
    dz=(height[...,1:]-height[...,:-1])[...,::-1]
    rho=(r(1)/alt*(r(1)+qv))[...,::-1]
    # Original INIT_AEROSOL:1597–1604, sulfate component. The IN component
    # is overridden by BACKIN in :1628–1633 (ID_DUST=0, ID_IN=2).
    mass=r(0)
    for im in range(3):
        xmas=c['zccn'][im,0]*xp.power(c['cnmod'][im,0],r(3))*xp.exp(r(4.5)*xp.power(c['cnstd'][im,0],r(2)))
        mass=mass+xmas
    mass=mass*c['c4pi3']*c['dnas'][0]*c['sens'][0]
    dry=mass/rho
    nuclei=r(4.e5)/rho
    # NTU_INIT fills only KTS:KTE-1; original INIT_AEROSOL then reads its
    # zero-initialized extra pressure. K_PBL=KTE, capped to KTE-2. Thus two
    # physical bottom layers plus the unused interface form the held PBL.
    # This is an upstream experimental disclosure, not a new PBL diagnosis.
    z=xp.zeros(p.shape[:-1],dtype=p.dtype);dz0=xp.zeros_like(z)
    sulfate_scale=r(1)/c['ash'][0]
    nuclei_scale=r(1)/c['ash'][1]*r(.5)
    # Host initialization only: in-place per-plane writes avoid NZ full-field
    # concatenate copies. No timestep code calls this function.
    dry=dry.copy();nuclei=nuclei.copy()
    for k in range(p.shape[-1]-3,-1,-1):
        z=z+(dz0+dz[...,k])*r(.5);dz0=dz[...,k]
        dry[...,k]=dry[...,k]*xp.exp(-sulfate_scale*z)
        nuclei[...,k]=nuclei[...,k]*xp.exp(-nuclei_scale*z)
    dry=dry[...,::-1];nuclei=nuclei[...,::-1]
    return dict(qdcn=dry,qtcn=dry.copy(),qccn=xp.zeros_like(dry),qrcn=xp.zeros_like(dry),qnin=nuclei)


def cold_state(state, base, metrics, *, ccnty=2, hypsometric_opt=2):
    """Host-only physics_init; start_em passes grid%ALT, PH/PHB and QV.

    ALT is the dycore's dry-air specific volume, not inverse moist density.
    Obtain the same calc_p_rho_phi diagnostic as start_em.F:841/1038; NTU_INIT
    itself multiplies its inverse by (1+QV). Restart loaders never call this.
    """
    import jax
    import jax.numpy as jnp
    from gpuwrf.dynamics.acoustic_wrf import diagnose_pressure_al_alt
    from gpuwrf.coupling.physics_couplers import _to_columns, _from_columns
    state = state.ensure_conditional_leaves(mp_physics=56)
    _, _, alt = diagnose_pressure_al_alt(state, base, metrics, hypsometric_opt=hypsometric_opt)
    host = lambda a: np.asarray(jax.device_get(_to_columns(a)), dtype=np.float32)
    values = aerosols(host(state.p), host(alt), host(state.ph_perturbation),
                      host(base.phb), host(state.qv),
                      np.asarray(jax.device_get(state.xland), np.float32), ccnty=ccnty)
    return state.replace(**{'ntu_' + n: _from_columns(jnp.asarray(a)) for n, a in values.items()})
