"""Optional WRF MYNN research outputs; no second turbulence/plume solve.

The opt-in twin calls these helpers; production column returns stay unchanged.
Nothing here feeds a prognostic update. WRF 4.7.1 module_bl_mynnedmf.F:1137-1157
normalizes the unused top and destaggers shear/buoyancy; :6470-6489 supplies
the plume means. Registry units are preserved separately from rate semantics.
"""
from __future__ import annotations

import jax.numpy as jnp

TKE_NAMES = ('qSHEAR','qBUOY','qDISS','qWT','Dtke')
EDMF_NAMES = ('edmf_a','edmf_w','edmf_qt','edmf_thl','edmf_ent','edmf_qc',
              'sub_thl3D','sub_sqv3D','det_thl3D','det_sqv3D')
PBL_NAMES = ('RTHBLTEN','RQVBLTEN','RQCBLTEN','RQIBLTEN','RUBLTEN','RVBLTEN')


def tke_history(state, qke_before, qke_after, turb, qwt, qdiss, flux, dt, *, mf=None, pdk=None):
    """Return the five WRF rates on kts:kte, with the unused kte set to zero."""
    from gpuwrf.physics import mynn_pbl as P
    ust, dz0 = flux.ustar, state.dz[...,0]
    rmol = -P.KARMAN * P.mynn_constant('GTR',state.theta.dtype) * flux.fltv / jnp.maximum(ust**3,1.e-6)
    zet = jnp.clip(.5*dz0*rmol,-20.,20.)
    phi_m = P._phim_puhales(zet) if pdk is None else None
    shear, buoy = turb['qshear'], turb['qbuoy']
    if mf is not None:
        # WRF's production is present even when edmf_tke=0 disables MF transport.
        production = jnp.abs(mf['edmf_w'])**3 * mf['edmf_a'] / (P.B1*jnp.maximum(state.el,.1))
        buoy = buoy + .5*production
    buoy0 = 4.*(-ust**3*zet/(P.KARMAN*dz0))-buoy[...,1]
    if pdk is None:
        shear0 = 4.*(ust**3*phi_m/(P.KARMAN*dz0))-shear[...,1]
    else:
        # Predictor surface production uses the ACTUAL selected phi parameters.
        # Recover its pmz contribution; do not freeze CPHM_UNST/am_st defaults.
        surface_pdk = pdk[...,0]+pdk[...,1]
        shear0 = surface_pdk+4.*ust**3*zet/(P.KARMAN*dz0)-shear[...,1]
    shear = jnp.concatenate((shear0[...,None],shear[...,1:]),axis=-1)
    buoy = jnp.concatenate((buoy0[...,None],buoy[...,1:]),axis=-1)
    zero = jnp.zeros_like(qke_after[...,:1])
    destagger = lambda value:jnp.concatenate((.5*(value[...,:-1]+value[...,1:]),zero),axis=-1)
    unused_top = lambda value:jnp.concatenate((value[...,:-1],zero),axis=-1)
    return (destagger(shear),destagger(buoy),unused_top(qdiss),unused_top(qwt),
            unused_top(.5*(qke_after-qke_before)/dt))


def edmf_history(state, mf):
    """Original standard env_subs=false makes the four sub/det outputs zero."""
    zero = jnp.zeros_like(state.theta)
    if mf is None:
        return (zero,)*len(EDMF_NAMES)
    return tuple(mf[name] for name in EDMF_NAMES[:6])+(zero,)*4


def mean_tendencies(before, after, dt):
    """Raw dry-column PBL sources; the operational adapter owns A-to-C coupling."""
    return tuple((getattr(after,name)-getattr(before,name))/dt
                 for name in ('theta','qv','qc','qi','u','v'))
