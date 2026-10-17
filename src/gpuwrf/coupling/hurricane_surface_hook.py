"""Scheme-specific WRF hurricane surface fluxes for sfclay=1,5,91."""
import jax
import jax.numpy as jnp


def surface_step(state, carry, namelist, *, first_timestep=False, spp_pattern=None, return_pbl_forcing=False):
    from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
    from gpuwrf.physics.surface_layer import SurfaceLayerDiagnostics, _surface_layer_impl
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    view = _build_column_view(state,namelist.grid)
    option = int(namelist.isftcflx)
    sf = int(namelist.sf_sfclay_physics)
    if carry.hurricane_ustm is None:
        raise ValueError('isftcflx needs resident USTM (E78)')
    if sf == 5:
        diag, ustm = _surface_layer_impl(view,first_timestep,jnp.float32,
            snowh=carry.noahmp_land.snowh if carry.noahmp_land is not None else None,
            isftcflx=option,ustm=carry.hurricane_ustm,_return_hurricane=True,
            **({'spp_pattern':spp_pattern} if spp_pattern is not None else {}))
    else:
        # WRF phy_prep fields, not the legacy adapter's p(kts) PSFC proxy.
        a = lambda x:jnp.asarray(x,jnp.float32)
        inputs = [a(x[...,0]) for x in (view.u,view.v,view.t_air,view.qv,view.p,view.dz)]
        if sf == 1:
            from gpuwrf.physics.sfclay_revised_mm5 import SfclayRevisedMM5ColumnState,sfclay_revised_mm5_run
            column = SfclayRevisedMM5ColumnState(*inputs,a(view.psfc),a(view.t_skin),a(view.xland),
                a(view.lakemask),a(view.mavail),a(view.roughness_m),a(view.ustar),a(view.mol),a(view.hfx),
                a(view.qfx),a(view.qsfc),a(view.pblh),jnp.full_like(view.t_skin,namelist.grid.projection.dx_m),jnp.full_like(view.t_skin,50))
            out = sfclay_revised_mm5_run(column,isftcflx=option)._asdict()
        elif sf == 91:
            from gpuwrf.physics.sfclay_old_mm5 import sfclay_old_mm5_columns
            arrays = inputs+[a(x) for x in (view.psfc,view.t_skin,view.xland,view.roughness_m,
                view.ustar,view.mol,view.qsfc,view.pblh,view.mavail,view.lakemask)]+[
                jnp.full_like(view.t_skin,namelist.grid.projection.dx_m),a(view.hfx),a(view.qfx)]
            shape=state.t_skin.shape
            out = sfclay_old_mm5_columns(*(x.reshape(-1) for x in arrays),isftcflx=option)
            out = {k:v.reshape(shape) for k,v in out.items()}
        else:
            raise ValueError('isftcflx=1/2 supported only by WRF sf_sfclay_physics=1/5/91')
        rho=a(view.rho[...,0])
        cpm=jnp.float32(1004.5)*(1.+jnp.float32(.8)*inputs[3])
        wind=jnp.maximum(jnp.sqrt(inputs[0]**2+inputs[1]**2),.1)
        flt=out['hfx']/(rho*cpm)
        flq=out['qfx']/rho
        theta=inputs[2]*(jnp.float32(100000)/inputs[4])**(jnp.float32(287)/jnp.float32(1004.5))
        ep1=jnp.float32(461.6)/jnp.float32(287)-1.
        fluxes=SurfaceFluxes(out['ust'],flt,flq,-out['ust']**2*inputs[0]/wind,
            -out['ust']**2*inputs[1]/wind,rho,(1.+ep1*inputs[3])*flt+ep1*theta*flq,xland=a(view.xland))
        diag=SurfaceLayerDiagnostics(fluxes,*(out[k] for k in ('hfx','lh','u10','v10','th2','t2','q2','qsfc','mol','rmol','zol','regime','psim','psih','br','znt')),
                                    chs2=out['chs2'],cqs2=out['cqs2'])
        # Both MM5 callers also advance the nongusty USTM, even though their
        # dissipative-heating additions are commented out in pristine WRF.
        ustm=.5*carry.hurricane_ustm+.5*jnp.float32(.4)*jnp.sqrt(inputs[0]**2+inputs[1]**2)/out['fm']
    # Optional path is explicitly REAL; no f64 carry escapes from MM5 tables.
    diag=jax.tree.map(lambda x:x.astype(jnp.float32) if hasattr(x,'dtype') and jnp.issubdtype(x.dtype,jnp.floating) else x,diag)
    flux=diag.fluxes
    updates=dict(ustar=flux.ustar,theta_flux=flux.theta_flux,qv_flux=flux.qv_flux,tau_u=flux.tau_u,
                 tau_v=flux.tau_v,rhosfc=flux.rhosfc,fltv=flux.fltv,hfx=diag.hfx,
                 qfx=diag.lh/jnp.float32(2.5e6),qsfc=diag.qsfc,mol=diag.mol,roughness_m=diag.znt)
    state=state.replace(**{k:jnp.asarray(v,getattr(state,k).dtype) for k,v in updates.items()})
    carry=carry.replace(hurricane_ustm=jnp.asarray(ustm,jnp.float32))
    if not namelist.use_noahmp and carry.history_diagnostics is not None:
        carry=carry.replace(history_diagnostics=carry.history_diagnostics.write({
            **{k:getattr(diag,k) for k in ('hfx','lh','t2','q2','u10','v10')},'psfc':view.psfc}))
    if return_pbl_forcing:
        if sf != 1:
            raise ValueError('selected revised-MM5 PBL handoff requires sfclay=1')
        from gpuwrf.coupling.surface_handoff import surface_handoff
        return state,carry,diag,surface_handoff(out,state.t_skin)
    return state,carry,diag
