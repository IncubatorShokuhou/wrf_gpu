"""Original mp38 caller for the experimental CPU-oracle-qualified column."""
import jax.numpy as jnp


def thompson_hail_adapter(state, dt, grid=None, *, return_precipitation=False):
    from gpuwrf.coupling.physics_couplers import (
        P0_PA, R_D_OVER_CP, _dry_theta_view, _to_columns, _from_columns,
        density_from_pressure_temperature,
        _state_from_thompson_aero_output, _theta_m_from_microphysics,
    )
    from gpuwrf.physics.thompson_hail.column import (
        ThompsonHailColumnState, step_thompson_hail_column_with_precip,
    )
    from gpuwrf.physics.thompson_aero_column import apply_surface_aerosol_emission

    # physics_init.F:988-1024 excludes THOMPSONGH from has_reqc/i/s.
    state = state.ensure_conditional_leaves(mp_physics=38, use_mp_re=0)
    real = jnp.float32
    th = jnp.asarray(_dry_theta_view(state), real)
    p, qv = jnp.asarray(state.p, real), jnp.asarray(state.qv, real)
    exner = (p / jnp.asarray(P0_PA, real)) ** jnp.asarray(R_D_OVER_CP, real)
    temperature = th * exner
    # mp_gt_driver :1234 recomputes DRY density from the prepared T/P/QV;
    # the external total phy_prep density used by Goddard is not this input.
    rho = density_from_pressure_temperature(p, temperature, qv)
    height = jnp.asarray(state.ph, real) / jnp.asarray(9.81, real)
    dz = height[1:] - height[:-1]  # Original phy_prep: no masking clamp.
    col = lambda value: _to_columns(jnp.asarray(value, real))
    column = ThompsonHailColumnState(
        *(col(getattr(state, name)) for name in ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Nc", "nwfa", "nifa")),
        col(temperature), col(p), col(rho), Ng=col(state.Ng),
        dz=col(dz), w=col(state.w[:-1]), qvolg=col(state.qvolg),
    )
    out, precip = step_thompson_hail_column_with_precip(column, float(dt))
    # mp_gt_driver :1320-1321 applies held emissions AFTER column physics.
    out = apply_surface_aerosol_emission(out, state.nwfa2d, state.nifa2d, float(dt))
    # Original wrapper :1299/:1308: REAL left-associated total and last-step SR.
    surface = {name: jnp.asarray(precip[name], real) for name in ("rain", "snow", "graupel", "ice")}
    total = ((surface["rain"] + surface["snow"]) + surface["graupel"]) + surface["ice"]
    frozen = (surface["snow"] + surface["graupel"]) + surface["ice"]
    sr = frozen / (total + jnp.asarray(1.e-12, real))
    precip = {**precip, "total": total, "sr": sr}
    theta_new = _from_columns(out.T) / exner
    theta = _theta_m_from_microphysics(th, state.qv, theta_new, _from_columns(out.qv), state.theta.dtype)
    # The legacy mp28 reassembler explicitly widens accumulator additions.
    # mp38's original caller :1299/:1302 instead rounds EACH REAL addition,
    # starting with the previous total (not previous + pre-summed RAINNCV).
    rainnc = jnp.asarray(state.thompson_rainnc, real)
    for name in ("rain", "snow", "graupel", "ice"):
        rainnc = rainnc + surface[name]
    snownc = (jnp.asarray(state.thompson_snownc, real) + surface["snow"]) + surface["ice"]
    buckets = {name + "_acc": (jnp.asarray(getattr(state, name + "_acc"), real) + amount).astype(getattr(state, name + "_acc").dtype)
               for name, amount in surface.items()}
    result = _state_from_thompson_aero_output(state, out, None).replace(
        theta=theta, qvolg=_from_columns(out.qvolg), Ns=state.Ns,
        thompson_sr=sr, thompson_rainnc=rainnc, thompson_snownc=snownc, **buckets)
    return (result, precip) if return_precipitation else result


def thompson_hail_coldstart_init(state):
    """Original host-only aerosol initialization; held emissions survive restarts."""
    import numpy as np
    from gpuwrf.physics.thompson_hail.init import initial_profiles

    state = state.ensure_conditional_leaves(mp_physics=38, use_mp_re=0)
    # thompson_init :489/527 uses independent MAXVAL checks, not ABS or a
    # joint CCN/IN presence test. The last tile edge is excluded there.
    nwfa, nifa = np.asarray(state.nwfa), np.asarray(state.nifa)
    absent_ccn = np.max(nwfa[:, :-1, :-1]) < np.float32(1.e-15)
    absent_in = np.max(nifa[:, :-1, :-1]) < np.float32(1.e-15)
    if not (absent_ccn or absent_in):
        return state
    z_w = np.asarray(state.ph, dtype=np.float32) / np.float32(9.81)
    hgt = np.moveaxis(np.float32(.5) * (z_w[:-1] + z_w[1:]), 0, -1)
    ccn, ice, emission = initial_profiles(hgt)
    updates = {}
    if absent_ccn:
        updates.update(nwfa=np.moveaxis(ccn, -1, 0), nwfa2d=emission)
    if absent_in:
        updates["nifa"] = np.moveaxis(ice, -1, 0)
    return state.replace(**updates)
