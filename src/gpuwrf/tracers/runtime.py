"""WRF non-Chem tracers and explicitly configured passive mass tracers.

Resolved transport uses the production scalar kernels and acoustic sumflux.
The optional mass tracers use kg/kg dry air; WRF TR17 fields are dimensionless.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

WRF_NAMES = tuple(f"tr17_{n}" for n in range(1, 9))


def names(namelist):
    config = getattr(namelist, "tracer_config", None)
    return ((WRF_NAMES if namelist.tracer_opt == 2 else ())
            + (() if config is None else tuple(s.name for s in config.species)))


def refuse_unwired_feedback(wrf_namelist, domains):
    """The current two-way registry does not feed back stacked tracers."""
    from gpuwrf.io.namelist_binding import namelist_value
    if any(namelist_value(wrf_namelist, "tracer_opt", domain) == 2
           or namelist_value(wrf_namelist, "gpuwrf_tracer_file", domain)
           for domain in domains):
        raise ValueError("passive tracers currently require feedback=0; two-way tracer feedback is not wired")


def validate_options(namelist):
    if namelist.tracer_opt not in (0, 2):
        raise ValueError("non-Chem WRF supports tracer_opt=0/2; modes 1/3/4 require WRF-Chem")
    if namelist.tracer_adv_opt not in (0, 1, 2):
        raise ValueError("tracer_adv_opt supports 0 (original), 1 (PD), 2 (monotonic)")
    if namelist.tracer_pblmix not in (0, 1):
        raise ValueError("tracer_pblmix supports 0/1")
    if not names(namelist):
        return
    if not namelist.use_flux_advection or not namelist.use_vertical_solver or namelist.rk_order != 3:
        raise ValueError("passive tracers require WRF RK3 acoustic sumflux transport")
    if namelist.diff_opt not in (0, 1) or (namelist.diff_opt == 1 and namelist.km_opt != 4):
        raise ValueError("passive tracers support diff_opt=0 or diff_opt=1, km_opt=4")
    if namelist.diff_opt == 1 and namelist.bl_pbl_physics == 0 and namelist.kvdif != 0:
        raise ValueError("passive tracers with bl=0 currently require kvdif=0")
    if namelist.tracer_pblmix and namelist.bl_pbl_physics not in (0, 5):
        raise ValueError("passive tracer PBL mixing currently supports MYNN (bl_pbl_physics=5) or bl=0")
    if namelist.tracer_pblmix and namelist.bl_pbl_physics == 5 and not namelist.rad_rk_tendf:
        raise ValueError("passive tracer MYNN mixing requires WRF physics source-leaf coupling (rad_rk_tendf=1)")
    if (not np.isfinite(namelist.dt_s) or namelist.dt_s <= 0
            or namelist.dt_s > float(np.finfo(np.float32).max)):
        raise ValueError("passive tracers require positive time_step")
    validate_source_constants(namelist)


def _real32_bound(value, description):
    # Admission arithmetic is host-only. Never narrow an overflowing product
    # first, and never clamp a configured source into the representable range.
    if not np.isfinite(value) or abs(value) > float(np.finfo(np.float32).max):
        raise ValueError(f"passive tracer {description} exceeds finite REAL32 range")
    return value


def _source_scalar(value, description):
    value = _real32_bound(float(value), description)
    if value < 0:
        raise ValueError(f"passive tracer {description} must be nonnegative")
    return float(np.float32(value))


def validate_source_constants(namelist):
    """Trace-time/static refusal; adds no device operations or host transfers."""
    cfg = namelist.tracer_config
    if cfg is None:
        return
    dt = float(np.float32(_real32_bound(float(namelist.dt_s), "time_step")))
    area = _real32_bound(float(namelist.grid.projection.dx_m)
                         * float(namelist.grid.projection.dy_m), "grid-cell area")
    totals = [0.0] * len(cfg.species)
    for spec in cfg.species:
        # These products precede the exponential/division in the REAL32 kernel.
        _real32_bound(_source_scalar(spec.decay_rate_s, "decay_rate_s") * dt, "decay exponent")
        _real32_bound(_source_scalar(spec.deposition_velocity_m_s, "deposition_velocity_m_s") * dt, "deposition distance")
    for emission in cfg.emissions:
        rate = (_source_scalar(emission.rate_kg_s, "rate_kg_s") if emission.is_point else
                _real32_bound(_source_scalar(emission.flux_kg_m2_s, "flux_kg_m2_s") * area, "area emission rate"))
        duration = _source_scalar(emission.end_s, "end_s") - _source_scalar(emission.start_s, "start_s")
        _real32_bound(min(dt, duration) * rate, "per-step emitted mass")
        cells = (emission.y1 - emission.y0) * (emission.x1 - emission.x0)
        totals[emission.species_index] += duration * rate * cells
        _real32_bound(totals[emission.species_index], "scheduled emission budget")


def validate_source_geometry(state, namelist):
    """Resolve mapped-cell source bounds once at run admission, never in a scan."""
    cfg = namelist.tracer_config
    if cfg is None:
        return
    validate_source_constants(namelist)
    mass = np.asarray(layer_mass(state, namelist))
    area = np.float32(namelist.grid.projection.dx_m * namelist.grid.projection.dy_m) / (
        np.asarray(namelist.metrics.msftx, np.float32) * np.asarray(namelist.metrics.msfty, np.float32))
    dz = (np.asarray(state.ph_total[1], np.float32)
          - np.asarray(state.ph_total[0], np.float32)) / np.float32(9.81)
    if (not np.isfinite(mass).all() or np.any(mass <= 0) or not np.isfinite(area).all()
            or np.any(area <= 0) or not np.isfinite(dz).all() or np.any(dz <= 0)):
        raise ValueError("passive tracer source geometry requires finite positive cell mass, area and depth")
    rain = (np.asarray(state.rain_acc, np.float64) + np.asarray(state.rainc_acc, np.float64))
    if not np.isfinite(rain).all():
        raise ValueError("passive tracer precipitation must be finite")
    if state.tracer_budget is None:
        raise ValueError("active custom tracers require cumulative source/sink budgets")
    offset = 8 if namelist.tracer_opt == 2 else 0
    dt = float(np.float32(namelist.dt_s))
    for index, spec in enumerate(cfg.species):
        _real32_bound(_source_scalar(spec.decay_rate_s, "decay_rate_s") * dt
            + _source_scalar(spec.deposition_velocity_m_s, "deposition_velocity_m_s") * dt / float(dz.min())
            + _source_scalar(spec.wet_scavenging_per_mm, "wet_scavenging_per_mm") * float(np.abs(rain).max()), "resolved loss exponent")
        added = np.zeros(area.shape, np.float64)
        for emission in cfg.emissions:
            if emission.species_index != index:
                continue
            ys, xs = slice(emission.y0, emission.y1), slice(emission.x0, emission.x1)
            rate = (_source_scalar(emission.rate_kg_s, "rate_kg_s") if emission.is_point else
                    _source_scalar(emission.flux_kg_m2_s, "flux_kg_m2_s") * area[ys, xs].astype(np.float64))
            duration = _source_scalar(emission.end_s, "end_s") - _source_scalar(emission.start_s, "start_s")
            _real32_bound(float(np.max(rate)), "mapped-cell emission rate")
            added[ys, xs] += duration * rate
        q = np.asarray(state.tracers[index + offset], np.float64)
        budget = np.asarray(state.tracer_budget[index + offset], np.float64)
        _real32_bound(float(added.sum()) + float(np.abs(budget).max()), "resolved emission budget")
        _real32_bound(float(np.max(np.abs(q[0]) + added / mass[0])), "resolved emitted concentration")
        # Any one loss budget can receive the entire current inventory.
        _real32_bound(float(np.sum(np.abs(q) * mass)) + float(added.sum())
                      + float(np.abs(budget).max()), "resolved loss budget")


def bind_inputs(namelist, state, wrf_namelist, domain, run_dir):
    """Read once before scanning; custom file paths are relative to the case."""
    from gpuwrf.io.namelist_binding import namelist_value
    from gpuwrf.tracers.config import load_config
    file = namelist_value(wrf_namelist, "gpuwrf_tracer_file", domain)
    if file:
        if not isinstance(file,str):
            raise ValueError("gpuwrf_tracer_file must be a path string")
        path = Path(str(file).strip("'\""))
        if not path.is_absolute():
            path = Path(run_dir) / path
        namelist = replace(namelist, tracer_config=load_config(path, shape=state.qv.shape),
                           gpuwrf_tracer_file=str(path))
    validate_options(namelist)
    count = len(names(namelist))
    if not count:
        return namelist, state
    fields = np.zeros((count,) + state.qv.shape, np.float32)
    if namelist.tracer_opt == 2:
        # real.exe's template is used only when the input lacks ALL TR17s.
        # Partial input is corruption, never an excuse to silently reinitialize.
        from gpuwrf.io.netcdf_lock import Dataset
        path = Path(run_dir) / f"wrfinput_{domain}"
        with Dataset(path) as ds:
            present = tuple(n in ds.variables for n in WRF_NAMES)
            if any(present) and not all(present):
                raise ValueError("wrfinput has an incomplete TR17 tracer group")
            if all(present):
                for i, name in enumerate(WRF_NAMES):
                    a = np.asarray(ds.variables[name][0], np.float32)
                    if a.shape != state.qv.shape or not np.isfinite(a).all():
                        raise ValueError(f"invalid wrfinput tracer {name}")
                    fields[i] = a
            else:
                nz, ny, nx = state.qv.shape
                # WRF ids=jds=1, ide=nx+1; convert integer center to zero-based.
                cx, cy = (nx + 2) // 2 - 1, (ny + 2) // 2 - 1
                if nx < 9 or ny < 9:
                    raise ValueError("tracer_opt=2 real.exe template requires at least 9x9 mass cells")
                fields[:4, 0, cy-4:cy+5, cx-4:cx+5] = 1
    updates = {"tracers": jnp.asarray(fields)}
    if namelist.tracer_config is not None:
        updates["tracer_budget"] = jnp.zeros((count, 4), jnp.float32)
    config = namelist.boundary_config
    if namelist.run_boundary and config.nested_frozen_wrf_boundary_bundle and not config.force_geopotential:
        width = int(state.qv_bdy.shape[2]) if state.qv_bdy.ndim == 5 else 1
        updates["tracer_bdy"] = jnp.zeros((count, 2, 4, width, state.qv.shape[0], max(state.qv.shape[1:])), jnp.float32)
    from gpuwrf.tracers.history import register_specs
    register_specs(namelist)
    return namelist, state.replace(**updates)


def seed_carry(carry, namelist):
    validate_options(namelist)
    if names(namelist):
        if carry.state.tracers is None:
            raise ValueError("active tracers require initialized State.tracers")
        validate_source_geometry(carry.state, namelist)
        carry = carry.replace(tracer_tendencies=jnp.zeros_like(carry.state.tracers))
    return carry


def inherit_parent(state, namelist, parent, parent_namelist, parent_grid, child_grid, child_meta, wrf_namelist, domain, run_dir):
    """WRF nest-start SINT inheritance, followed by the selected input stream."""
    from gpuwrf.nesting.scalar_initialization import _time_control_value
    from gpuwrf.nesting.boundary_construction import build_nest_force_weights
    from gpuwrf.nesting.interp import interp_sint_full
    from gpuwrf.io.netcdf_lock import Dataset
    if names(namelist) != names(parent_namelist):
        raise ValueError("parent/child passive tracers require the same ordered species list")
    if not names(namelist):
        return state
    read_file = bool(_time_control_value(wrf_namelist,"input_from_file",domain,False))
    stream = int(_time_control_value(wrf_namelist,"fine_input_stream",domain,0))
    keep = set()
    if read_file and stream == 0 and namelist.tracer_opt == 2:
        with Dataset(Path(run_dir)/f"wrfinput_{domain}") as ds:
            keep = {n for n in WRF_NAMES if n in ds.variables}
    weights = None
    fields = []
    for i, name in enumerate(names(namelist)):
        if name in keep:
            fields.append(state.tracers[i])
        else:
            if weights is None:
                weights = build_nest_force_weights(parent_grid_ratio=int(child_meta.parent_grid_ratio),
                    i_parent_start=int(child_meta.i_parent_start), j_parent_start=int(child_meta.j_parent_start),
                    parent_grid=parent_grid,child_grid=child_grid)
            fields.append(interp_sint_full(parent.tracers[i],weights.mass,
                parent_grid_ratio=int(child_meta.parent_grid_ratio)).astype(jnp.float32))
    return state.replace(tracers=jnp.stack(fields))


def frozen_tendencies(origin, namelist, *, muts, xkhh, pbl_rates, lead_seconds):
    """WRF RK1-frozen tracer_tend: physics, diffusion, nested relax/spec."""
    from gpuwrf.runtime.operational_mode import _acoustic_lateral_bc_flags
    from gpuwrf.dynamics.explicit_diffusion import horizontal_diffusion_coord_scalar_tendency
    from gpuwrf.dynamics.explicit_diffusion import wrf_sixth_order_scalar_tendf
    from gpuwrf.coupling.boundary_apply import nested_scalar_boundary_tendencies
    metrics = namelist.metrics
    # pbl_driver's tracer_pblmix switch and first_rk_step_part1's BL=0
    # dispatch must win over any held rate supplied by a restart/API caller.
    if not namelist.tracer_pblmix or namelist.bl_pbl_physics == 0:
        pbl_rates = None
    _, specified, nested = _acoustic_lateral_bc_flags(namelist)
    dtype = jnp.float32
    mass = metrics.c1h.astype(dtype)[:, None, None] * muts.astype(dtype)[None] + metrics.c2h.astype(dtype)[:, None, None]
    physics_mass = metrics.c1h.astype(dtype)[:, None, None] * origin.mu_total.astype(dtype)[None] + metrics.c2h.astype(dtype)[:, None, None]
    result = []
    for i, q in enumerate(origin.tracers):
        sc = jnp.zeros_like(q) if pbl_rates is None else pbl_rates[i] * physics_mass
        if xkhh is not None and not namelist.tracer_mix2_off:
            sc = sc + horizontal_diffusion_coord_scalar_tendency(
                q, xkhh.astype(dtype), mass,
                dx_m=float(namelist.grid.projection.dx_m), dy_m=float(namelist.grid.projection.dy_m),
                msftx=metrics.msftx if specified or nested else None,
                msfty=metrics.msfty if specified or nested else None,
                msfux=metrics.msfux if specified or nested else None,
                msfuy=metrics.msfuy if specified or nested else None,
                msfvx=metrics.msfvx if specified or nested else None,
                msfvy=metrics.msfvy if specified or nested else None,
                nonperiodic_owned=bool(specified or nested)).astype(dtype)
        if namelist.diff_6th_opt and not namelist.tracer_mix6_off:
            sc = sc + wrf_sixth_order_scalar_tendf(
                q, muts.astype(dtype), c1=metrics.c1h.astype(dtype), c2=metrics.c2h.astype(dtype),
                msftx=metrics.msftx.astype(dtype), msfty=metrics.msfty.astype(dtype),
                dt=float(namelist.dt_s)/3, diff_6th_factor=float(namelist.diff_6th_factor),
                monotonic=namelist.diff_6th_opt == 2, specified_or_nested=bool(specified or nested)).astype(dtype)
        if nested:
            if origin.tracer_bdy is None:
                raise ValueError("nested tracers require parent boundary records")
            # Reuse the pristine-qualified REAL scalar boundary primitive. The
            # proxy exists at trace time and does not alter physical QC.
            proxy = origin.replace(qc=q, qc_bdy=origin.tracer_bdy[i])
            bdy = nested_scalar_boundary_tendencies(proxy, lead_seconds, metrics,
                float(namelist.dt_s), namelist.boundary_config, species=("qc",))[0]
            spec = int(namelist.boundary_config.spec_zone)
            if spec:
                sc = sc.at[:, :spec, :].set(0).at[:, -spec:, :].set(0)
                sc = sc.at[:, :, :spec].set(0).at[:, :, -spec:].set(0)
            sc = sc + bdy
        result.append(sc)
    return jnp.stack(result)


def stage_update(state, origin, frozen, namelist, *, rk_step, dt_rk, velocities):
    """Literal tracer loop: final PD preload, map-scaled advection, RK update."""
    from gpuwrf.dynamics.flux_advection import advect_moisture_scalars
    from gpuwrf.coupling.boundary_apply import flow_dep_bdy
    from gpuwrf.runtime.operational_mode import _acoustic_lateral_bc_flags
    metrics = namelist.metrics
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    mass_old = f32(metrics.c1h)[:, None, None] * f32(origin.mu_total)[None] + f32(metrics.c2h)[:, None, None]
    mass_new = f32(metrics.c1h)[:, None, None] * f32(state.mu_total)[None] + f32(metrics.c2h)[:, None, None]
    final_pd = rk_step == 3 and namelist.tracer_adv_opt in (1, 2)
    # module_em.F:1907 rk_update_scalar_pd: retain the coupled numerator's
    # REAL operation order before the limiter reads this preloaded old field.
    old = ((mass_old * origin.tracers + f32(dt_rk) * frozen) / mass_old
           if final_pd else origin.tracers)
    transport = replace(velocities, **{n: f32(getattr(velocities, n))
        for n in ("ru", "rv", "rom", "ru_full", "rv_full", "msftx", "msfux", "msfvy", "msfvx")
        if getattr(velocities, n) is not None})
    adv = advect_moisture_scalars(tuple(state.tracers), tuple(old) if final_pd else None,
        transport, moist_adv_opt=namelist.tracer_adv_opt, is_final_rk_stage=rk_step == 3,
        mut=f32(state.mu_total), mu_old=f32(origin.mu_total), c1=f32(metrics.c1h), c2=f32(metrics.c2h),
        rdx=1/float(namelist.grid.projection.dx_m), rdy=1/float(namelist.grid.projection.dy_m),
        rdzw=f32(metrics.rdnw), fzm=f32(metrics.fnm), fzp=f32(metrics.fnp), dt=float(namelist.dt_s),
        msfty=f32(metrics.msfty), force_native_pd=True)
    _, specified, nested = _acoustic_lateral_bc_flags(namelist)
    spec = int(namelist.boundary_config.spec_zone) if specified or nested else 0
    updates = []
    for i, tend in enumerate(adv):
        tend = tend * f32(metrics.msfty)[None]
        if spec:
            interior = jnp.zeros_like(tend)
            tend = interior.at[:, spec:-spec, spec:-spec].set(tend[:, spec:-spec, spec:-spec])
        if not final_pd:
            tend = tend + frozen[i]
        q = (mass_old * old[i] + f32(dt_rk) * tend) / mass_new
        if specified:
            if transport.ru_full is None or transport.rv_full is None:
                raise ValueError("specified tracer boundaries require full acoustic mass fluxes")
            q = flow_dep_bdy(q, transport.ru_full, transport.rv_full, namelist.boundary_config)
        updates.append(q)
    return state.replace(tracers=jnp.stack(updates))


def layer_mass(state, namelist):
    """Dry-air kg/cell from hybrid pressure thickness and physical grid area."""
    metrics = namelist.metrics
    mass = (jnp.asarray(metrics.c1h, jnp.float32)[:, None, None] * state.mu_total.astype(jnp.float32)[None]
            + jnp.asarray(metrics.c2h, jnp.float32)[:, None, None])
    area = np.float32(namelist.grid.projection.dx_m * namelist.grid.projection.dy_m) / (
        metrics.msftx.astype(jnp.float32) * metrics.msfty.astype(jnp.float32))
    return mass * (-metrics.dnw.astype(jnp.float32)[:, None, None]) / np.float32(9.81) * area[None]


def sources_and_sinks(state, before, namelist, start_s):
    """Opt-in split source/sink step; exact constant-coefficient loss and budget.

    Emissions are added at the end of their overlapping step interval. The
    loss is integrated on the transported field; emitted material begins loss
    in the following step. No positivity clamps or tracer feedback to weather.
    """
    cfg = namelist.tracer_config
    if cfg is None:
        return state
    validate_source_constants(namelist)
    count = len(names(namelist))
    offset = 8 if namelist.tracer_opt == 2 else 0
    mass = layer_mass(state, namelist)
    area = np.float32(namelist.grid.projection.dx_m * namelist.grid.projection.dy_m) / (
        namelist.metrics.msftx.astype(jnp.float32) * namelist.metrics.msfty.astype(jnp.float32))
    dz0 = (state.ph_total[1].astype(jnp.float32) - state.ph_total[0].astype(jnp.float32)) / np.float32(9.81)
    dt = np.float32(namelist.dt_s)
    q = state.tracers
    budget = state.tracer_budget
    for i, spec in enumerate(cfg.species, offset):
        # Normalize coefficients before products and normalize exposures before
        # their sum. In particular, a finite wet coefficient times finite rain
        # may exceed REAL32: the analytic survival tends to zero, while sink
        # shares must remain finite rather than producing Inf/Inf budgets.
        decay_dose = np.float32(spec.decay_rate_s) * dt
        deposition_distance = np.float32(spec.deposition_velocity_m_s) * dt
        wet_coefficient = np.float32(spec.wet_scavenging_per_mm)
        scale = np.float32(max(float(decay_dose), float(deposition_distance),
                               float(wet_coefficient)) or 1.)
        decay = jnp.full_like(q[i], decay_dose / scale)
        dry = jnp.zeros_like(q[i]).at[0].set((deposition_distance / scale) / dz0)
        wet_components = tuple(jnp.broadcast_to((wet_coefficient / scale) * increment[None], q[i].shape)
            for increment in (state.rain_acc.astype(jnp.float32) - before.rain_acc.astype(jnp.float32),
                              state.rainc_acc.astype(jnp.float32) - before.rainc_acc.astype(jnp.float32)))
        largest = jnp.maximum(jnp.maximum(jnp.abs(decay), jnp.abs(dry)),
                              jnp.maximum(jnp.abs(wet_components[0]), jnp.abs(wet_components[1])))
        divisor = jnp.where(largest == 0, np.float32(1), largest)
        exposures = jnp.stack((decay / divisor, dry / divisor,
                              wet_components[0] / divisor + wet_components[1] / divisor))
        total = jnp.sum(exposures, axis=0)
        # Overflow here has a defined analytic limit exp(-Inf)=0; it never
        # enters a budget ratio and no concentration or budget is clamped.
        fraction = -jnp.expm1(-(scale * largest * total))
        loss = q[i] * fraction
        denominator = jnp.where(total == 0, np.float32(1), total)
        weights = exposures / denominator[None]
        removed = jnp.sum(weights * (mass * loss)[None], axis=(1, 2, 3))
        updated = q[i] - loss
        emitted = jnp.asarray(0, jnp.float32)
        for emission in cfg.emissions:
            if emission.species_index != i - offset:
                continue
            overlap = jnp.maximum(np.float32(0), jnp.minimum(jnp.asarray(start_s, jnp.float32) + dt,
                np.float32(emission.end_s)) - jnp.maximum(jnp.asarray(start_s, jnp.float32), np.float32(emission.start_s)))
            ys, xs = slice(emission.y0, emission.y1), slice(emission.x0, emission.x1)
            amount = overlap * (np.float32(emission.rate_kg_s) if emission.is_point else
                                np.float32(emission.flux_kg_m2_s) * area[ys, xs])
            updated = updated.at[0, ys, xs].add(amount / mass[0, ys, xs])
            emitted = emitted + jnp.sum(amount)
        q = q.at[i].set(updated)
        budget = budget.at[i].add(jnp.concatenate((emitted[None], removed)))
    return state.replace(tracers=q, tracer_budget=budget)


def final_boundary(state, namelist, lead_seconds):
    from gpuwrf.runtime.operational_mode import _nested_frozen_wrf_boundary_active
    if not _nested_frozen_wrf_boundary_active(namelist):
        return state
    from gpuwrf.coupling.boundary_apply import _scalar_record_value_rate_real4, _full_ring_target_from_leaf
    result = []
    nz, ny, nx = state.qv.shape
    y, x = jnp.arange(ny)[:, None], jnp.arange(nx)[None]
    mask = jnp.minimum(jnp.minimum(y, ny-1-y), jnp.minimum(x, nx-1-x)) < namelist.boundary_config.spec_zone
    mass = (namelist.metrics.c1h.astype(jnp.float32)[:, None, None] * state.mu_total.astype(jnp.float32)[None]
            + namelist.metrics.c2h.astype(jnp.float32)[:, None, None])
    for q, bdy in zip(state.tracers, state.tracer_bdy, strict=True):
        value = _scalar_record_value_rate_real4(bdy, lead_seconds, namelist.boundary_config.update_cadence_s)[0]
        target = _full_ring_target_from_leaf(value, nz, ny, nx, jnp.float32)
        result.append(jnp.where(mask[None], target / mass, q))
    return state.replace(tracers=jnp.stack(result))
