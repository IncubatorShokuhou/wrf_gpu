"""WRF surface_driver -> ocean_driver -> OML1D caller wiring."""
import jax.numpy as jnp


def ocean_surface_step(entry, state, carry, namelist):
    """After the LSM; fluxes are current, U_PHY/V_PHY are step-entry winds."""
    from gpuwrf.physics.ocean_mixed_layer import step
    if carry.ocean is None:
        raise ValueError('sf_ocean_physics=1 missing resident OML carry (E78)')
    history=carry.history_diagnostics
    if history is None or history.lh is None:
        raise ValueError('OML requires current surface_driver LH, not a QFX reconstruction (E78)')
    rad = carry.radiation_diagnostics
    if rad is not None:
        # radiation_driver GSW is absorbed, not incident SWDOWN/SWNORM.
        gsw = rad.swnorm-rad.swup_topographic
        glw = rad.glw
        if carry.solar_interpolation is not None:
            gsw=carry.solar_interpolation.fluxes.swdown*(1.-carry.noahmp_land.albedo)
    elif carry.noahmp_rad is not None:
        gsw = carry.noahmp_rad[0]*(1.-carry.noahmp_land.albedo)
        glw = carry.noahmp_rad[1]
    else:
        raise ValueError('OML requires held surface radiation')
    ocean = step(carry.ocean,hfx=state.hfx,lh=history.lh,
                 gsw=gsw,glw=glw,u=(entry.u[0,:,:-1]+entry.u[0,:,1:])*jnp.float32(.5),
                 v=(entry.v[0,:-1,:]+entry.v[0,1:,:])*jnp.float32(.5),
                 ust=state.ustar,f=namelist.grid.metrics.f,
                 emiss=carry.noahmp_land.emiss if carry.noahmp_land is not None else jnp.float32(.98),
                 dt=namelist.dt_s,gamma=namelist.oml_gamma,
                 relaxation_time=namelist.oml_relaxation_time,water=state.xland>1.5)
    tsk = jnp.where(state.xland>1.5,ocean.tml,state.t_skin)
    land = carry.noahmp_land
    if land is not None:
        land = land.replace(t_skin=jnp.where(state.xland>1.5,ocean.tml,land.t_skin))
    return state.replace(t_skin=tsk),carry.replace(ocean=ocean,noahmp_land=land)


def history(carry):
    return {name.upper():value for name,value in carry._asdict().items()}
