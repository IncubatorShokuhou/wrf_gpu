"""Opt-in Thompson operand adapter; OFF retains the original adapter object."""
from __future__ import annotations

from gpuwrf.coupling.physics_couplers import (
    _thompson_column_from_state, _state_from_thompson_output,
    _thompson_tendency_side_channel, thompson_adapter,
)
from gpuwrf.kernels.phys_thompson_full import full_column


def selected_adapter(namelist, default_adapter=thompson_adapter):
    """Return the original OFF function, never a wrapper with an extra frame."""
    parameters = getattr(namelist, "thompson_parameters", None)
    if parameters is None:
        return default_adapter
    from gpuwrf.physics.thompson_parameters import require_native, require_runtime_qualified
    require_native(); require_runtime_qualified(parameters)
    import os
    if int(namelist.mp_physics) != 8 or not namelist.run_physics or os.environ.get("GPUWRF_MICROPHYSICS_WRF_ORDER", "1") != "1":
        raise ValueError("runtime Thompson adapter requires active WRF-ordered mp_physics=8")
    from functools import partial
    return partial(thompson_runtime_adapter, parameters=parameters)


def thompson_runtime_adapter(state, dt, grid=None, *, parameters,
                             return_tendencies=False, return_precipitation=False,
                             return_radar_context=False):
    """Same column preparation/copyback as thompson_adapter, with REAL operands.

    The namelist binder admits only native mp=8 explicit sedimentation. This
    path uses the existing full-column kernel, including vapor copyback and
    precipitation. It preserves the State and precipitation return contract.
    """
    column = _thompson_column_from_state(state, grid)
    if return_radar_context:
        out, precip, (raw_qv, radar_ng) = full_column(
            column, dt, parameters=parameters, return_radar_context=True)
    else:
        out, precip = full_column(column, dt, parameters=parameters)
    nt_c = parameters.get("nt_c")
    if nt_c is not None and state.re_cloud is not None:
        from gpuwrf.physics.thompson_nt_c import effective_radii
        from gpuwrf.coupling.physics_couplers import _from_columns
        # The existing copyback is otherwise unchanged. Suppress its fixed-Nt
        # radius block locally, then run the original equation once with Nt_c.
        no_radii = state.replace(re_cloud=None, re_ice=None, re_snow=None)
        next_state = _state_from_thompson_output(no_radii, out, precip)
        radii = effective_radii(out, nt_c)
        next_state = next_state.replace(**dict(zip(("re_cloud", "re_ice", "re_snow"), map(_from_columns, radii))))
    else:
        next_state = _state_from_thompson_output(state, out, precip)
    if return_radar_context:
        from gpuwrf.diagnostics.thompson_radar import ThompsonRadarInputs
        return next_state, precip, ThompsonRadarInputs(raw_qv, out.qr, out.Nr, out.qs, out.qg,
                                                       radar_ng, out.T, out.p)
    if return_tendencies:
        return next_state, _thompson_tendency_side_channel(state, out, dt)
    if return_precipitation:
        return next_state, precip
    return next_state
