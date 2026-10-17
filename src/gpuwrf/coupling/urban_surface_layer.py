"""Selected revised-MM5 surface layer for BEP/BEM + YSU/BouLac only.

The shared Noah-MP adapter historically calls MYNN's surface layer directly.
The new urban path must consume sf_sfclay_physics=1, including its held BR,
PSIM/PSIH and WSPD, rather than recomputing MYNN in the PBL caller.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class UrbanSurfaceLayer(NamedTuple):
    br: object
    psim: object
    psih: object
    wspd: object
    u10: object
    v10: object


def revised_surface_layer(state, *, return_pbl_resistances=False):
    from gpuwrf.physics.sfclay_revised_mm5 import SfclayRevisedMM5ColumnState,sfclay_revised_mm5_run
    from gpuwrf.physics.surface_layer import SurfaceLayerDiagnostics
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    def take(name,default=0.):
        value=getattr(state,name,None)
        if value is None:value=default
        value=jnp.asarray(value,jnp.float32)
        return value[...,0] if value.ndim==3 else value
    t_air=take('t_air')
    with jax.enable_x64(False):
        column=SfclayRevisedMM5ColumnState(take('u'),take('v'),t_air,take('qv'),take('p'),take('dz'),
            take('psfc'),take('t_skin'),take('xland'),take('lakemask'),take('mavail',1.),
            take('roughness_m',.1),take('ustar',.1),take('mol'),take('hfx'),take('qfx'),take('qsfc',-1.),
            take('pblh',1000.),take('dx_m',3000.),jnp.float32(50.))
        out=sfclay_revised_mm5_run(column)
    flux=SurfaceFluxes(ustar=out.ust,theta_flux=out.theta_flux,qv_flux=out.qv_flux,tau_u=out.tau_u,tau_v=out.tau_v,
        rhosfc=out.rhosfc,fltv=out.fltv,xland=take('xland'),wspd=out.wspd,t_skin=take('t_skin'),psfc=take('psfc'))
    diag = SurfaceLayerDiagnostics(flux,out.hfx,out.lh,out.u10,out.v10,out.th2,out.t2,out.q2,out.qsfc,
        out.mol,out.rmol,out.zol,out.regime,out.psim,out.psih,out.br,out.znt,out.chs2,out.cqs2,out.chs)
    return (diag,(out.fm,out.fh)) if return_pbl_resistances else diag
