"""History/auxhist fields for the explicit per-domain tracer selection."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np


def register_specs(namelist):
    from gpuwrf.tracers.runtime import names
    from gpuwrf.io.wrfout_writer import WRFOUT_VARIABLE_SPECS, _spec, XYZ
    selected = names(namelist)
    for name in selected:
        wrf = name.startswith("tr17_")
        spec = _spec(name, XYZ, "XYZ", name if wrf else f"Passive tracer {name}",
            "Dimensionless" if wrf else "kg kg-1", coordinates="XLONG XLAT XTIME")
        if name in WRFOUT_VARIABLE_SPECS and WRFOUT_VARIABLE_SPECS[name] != spec:
            raise ValueError(f"passive tracer {name} collides with an existing output field")
        WRFOUT_VARIABLE_SPECS[name] = spec
    return selected


@dataclass(frozen=True)
class TracerHistorySelection:
    names: tuple[str, ...]
    stack: object = None
    volumes: tuple = ()
    finite: object = None


@jax.jit
def _finite_on_device(values):
    # Keep the elementwise predicate inside the reduction program rather than
    # materializing a full-sized boolean mask as a separate eager result.
    return jnp.all(jnp.isfinite(values))


def select_history(state, namelist, requested=None):
    """Select volumes before the host batch; retain views for identity lookup."""
    selected = register_specs(namelist)
    if state.tracers is None:
        if selected:
            raise ValueError("active tracer history is missing State.tracers")
        return TracerHistorySelection(())
    # Shape is host metadata. Do not coerce the device stack to check it.
    if state.tracers.shape != (len(selected),) + state.qv.shape:
        raise ValueError("invalid tracer history shape or nonfinite values")
    indices = tuple(i for i, name in enumerate(selected) if requested is None or name in requested)
    if len(indices) == len(selected):
        # Full output already needs every value: one stack transfer and host
        # finite check avoid N slice launches and an additional device reduction.
        return TracerHistorySelection(selected, stack=state.tracers)
    if isinstance(state.tracers, np.ndarray):
        finite = np.isfinite(state.tracers).all()
    else:
        finite = _finite_on_device(state.tracers)
    return TracerHistorySelection(tuple(selected[i] for i in indices),
                                 volumes=tuple(state.tracers[i] for i in indices), finite=finite)


def history_fields(state, namelist, requested=None, *, selection=None):
    from gpuwrf.io.wrfout_writer import _host_materialized_value
    if selection is None:
        # Standalone callers keep the same batched transfer discipline.
        selection = select_history(state, namelist, requested)
        stack, volumes, finite = jax.device_get((selection.stack, selection.volumes, selection.finite))
    else:
        stack = None if selection.stack is None else _host_materialized_value(selection.stack)
        volumes = tuple(_host_materialized_value(value) for value in selection.volumes)
        finite = None if selection.finite is None else _host_materialized_value(selection.finite)
    if stack is not None:
        values = np.asarray(stack)
        if not np.isfinite(values).all():
            raise ValueError("invalid tracer history shape or nonfinite values")
        return dict(zip(selection.names, values, strict=True))
    if finite is not None and not bool(np.asarray(finite)):
        raise ValueError("invalid tracer history shape or nonfinite values")
    return {name: np.asarray(value) for name, value in zip(selection.names, volumes, strict=True)}
