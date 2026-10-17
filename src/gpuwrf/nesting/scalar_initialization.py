"""WRF nest-start inheritance for extra moist/scalar fields (initialization only).

med_nest_initial -> med_interp_domain interpolates uncoupled Registry 'd'
fields with interp_fcn/SINT. The child input stream then overwrites fields it
contains. A missing input field retains the interpolated parent value.
"""
from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.io.namelist_binding import namelist_value
from gpuwrf.nesting.boundary_construction import build_nest_force_weights
from gpuwrf.nesting.interp import interp_sint_full
from gpuwrf.nesting.scalar_species import extra_boundary_species

_INITIALIZED_MP = frozenset((5, 7, 9, 15, 18, 24, 26, 28, 38, 40))
_VARIABLES = {
    "qt": "CWM", "qrimef": "QRIMEF",
    "qh": "QHAIL", "Nc": "QNCLOUD", "Nn": "QNCCN", "Ns": "QNSNOW", "Ng": "QNGRAUPEL",
    "Nh": "QNHAIL", "qvolg": "QVGRAUPEL", "qvolh": "QVHAIL", "nwfa": "QNWFA", "nifa": "QNIFA",
}


def _time_control_value(namelist, key, domain, default):
    """Fortran per-domain defaults; an omitted tail does not repeat entry one."""
    raw = namelist.get("time_control", {}).get(key)
    index = int(domain[1:]) - 1
    if raw is None:
        return default
    values = raw if isinstance(raw, (list, tuple)) else (raw,)
    return values[index] if index < len(values) else default


def initialize_extra_scalar_fields(run, domain, state, grid, *, parent_case=None, parent_state=None):
    """Initialize selected extra fields without changing the released mp=8 path.

    Input overrides are read first so an interpolation that would immediately
    be overwritten is not materialized. No other nest-initialization operation
    consumes these extra fields between interpolation and input in WRF.
    """
    selected = namelist_value(run.namelist, "mp_physics", domain)
    mp = 8 if selected is None else int(selected)
    if mp not in _INITIALIZED_MP:
        return state
    state = state.ensure_conditional_leaves(mp_physics=mp)
    names = extra_boundary_species(mp)
    weights = None
    child_meta = run.grid(domain) if parent_case is not None else None
    from_file = _time_control_value(run.namelist, "input_from_file", domain, False)
    read_input = parent_case is None or bool(from_file)
    stream = int(_time_control_value(run.namelist, "fine_input_stream", domain, 0))
    updates = {}
    for name in names:
        variable = "QNDROP" if mp == 18 and name == "Nc" else _VARIABLES[name]
        value = None
        # These Registry entries all carry i0 (main input stream) only.
        if read_input and (parent_case is None or stream == 0):
            try:
                value = run.load_wrfinput(domain, variable, lazy=False)
            except KeyError:
                pass
        if value is None and parent_case is not None:
            # Parent physics initialization precedes child opening in WRF.
            # In particular mp28 aerosols may have been seeded after raw input.
            source_state = parent_case.state if parent_state is None else parent_state
            parent_field = getattr(source_state, name, None)
            if parent_field is None:
                raise ValueError(f"{domain}: parent has no {name} for Registry nest-start interpolation")
            if weights is None:
                weights = build_nest_force_weights(
                    parent_grid_ratio=int(child_meta.parent_grid_ratio),
                    i_parent_start=int(child_meta.i_parent_start),
                    j_parent_start=int(child_meta.j_parent_start),
                    parent_grid=parent_case.grid, child_grid=grid,
                )
            value = interp_sint_full(jnp.asarray(parent_field, jnp.float32), weights.mass,
                                     parent_grid_ratio=int(child_meta.parent_grid_ratio))
        if value is not None:
            value = jnp.asarray(value, jnp.float32)
            if value.shape != state.qv.shape:
                raise ValueError(f"{domain}: {variable} shape {value.shape} != mass grid {state.qv.shape}")
            updates[name] = value
    if mp in (28, 38) and read_input:
        # wif_input_opt/=2: mp_gt_driver zeroes NBCA on every microphysics column
        # (module_mp_thompson.F:1336-1338); only a zero input keeps the unported
        # specified-zone transport of QNBCA identically zero (QNBCA history = +0).
        import numpy as np
        for variable in ("QNBCA", "QNBCA2D"):
            try:
                value = run.load_wrfinput(domain, variable, lazy=False)
            except KeyError:
                continue
            if np.any(np.asarray(value) != 0):
                raise ValueError(f"{domain}: mp_physics={mp} qualifies only zero {variable} with default wif_input_opt=1")
    if mp in (28, 38) and read_input:
        surface_fields = (("nwfa2d", "QNWFA2D"), ("nifa2d", "QNIFA2D"))
        if mp == 38:
            surface_fields += (("thompson_rainnc", "RAINNC"), ("thompson_snownc", "SNOWNC"))
        for name, variable in surface_fields:
            try:
                value = run.load_wrfinput(domain, variable, lazy=False)
            except KeyError:
                continue
            value = jnp.asarray(value, jnp.float32)
            if value.shape != state.rain_acc.shape:
                raise ValueError(f"{domain}: {variable} shape {value.shape} != surface grid {state.rain_acc.shape}")
            updates[name] = value
    # This is input initialization, before the operational precision adapter.
    # Preserve WRF REAL values/storage even if a legacy fixture seeded Ns/Ng as f64.
    return state.replace(_cast=False, **updates) if updates else state
