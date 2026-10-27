"""Exact WRF ``spec_bdyupdate`` walk rates for a specified root (o5-coast, GPUWRF_ROOT_SPEC_STAGE_CLOCK=walk).

Kept out of ``boundary_apply`` on purpose: Pallas/Triton custom calls embed SOURCE LINE locations, so code
inserted above ``boundary_apply._scalar_couple_kernel`` changed the raw HLO of the nest force-down program
even with the walk OFF (FINDINGS E109, w44-release 2026-10-24).  Mode selection (``ROOT_SPEC_WALK``,
``root_spec_stage_clock_mode``, ``root_spec_stage_lead``) stays in ``boundary_apply`` below that kernel.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.contracts.state import State
from gpuwrf.coupling.boundary_apply import BoundaryConfig, specified_boundary_tendency


class RootWalkRates(NamedTuple):
    """WRF ``spec_bdy_dry`` ring tendencies of a specified root (coupled, per second)."""

    mu: jax.Array      # (ny, nx)        MU_BT (uncoupled)
    theta: jax.Array   # (nz, ny, nx)    T_BT  = d/dt[(c1h*MT+c2h)*(theta_m-T0)]
    ph: jax.Array      # (nz+1, ny, nx)  PH_BT = d/dt[(c1f*MT+c2f)*ph']
    u: jax.Array       # (nz, ny, nx+1)  U_BT  = d/dt[(c1h*MTu+c2h)*u/msfuy]
    v: jax.Array       # (nz, ny+1, nx)  V_BT  = d/dt[(c1h*MTv+c2h)*v/msfvx]


def _root_side_strips_2d(field, width: int, length: int, fill: float):
    """``(4, width, 1, length)`` W/E/S/N strips of a 2-D field, b=0 outermost (d02_replay._field_sides_3d)."""

    def pad(a):
        return jnp.pad(a, ((0, 0), (0, int(length) - int(a.shape[1]))), constant_values=fill)

    sides = (
        field[:, :width].T,
        jnp.flip(field[:, -width:], axis=1).T,
        field[:width, :],
        jnp.flip(field[-width:, :], axis=0),
    )
    return jnp.stack([pad(s) for s in sides])[:, :, None, :]


def _root_face_mass(total, kind: str, ny: int, nx: int):
    """calc_mu_staggered on the record mass strips ``(time, 4, width, 1, len)`` for u or v faces.

    Faces normal to a side (W/E u, S/N v): face b averages cells b and b-1, the outermost face takes the
    edge cell.  Faces along a side (S/N u, W/E v): n cells give n+1 faces with one-sided ends.  Same algebra
    as the wrfbdy decoupling (d02_replay mass_u_2d/mass_v_2d) and module_big_step_utilities_em.F:403.
    """

    length = int(total.shape[-1])

    def across(s):
        prev = jnp.concatenate([s[:, :1], s[:, :-1]], axis=1)
        return 0.5 * (s + prev)

    def along(s, n):
        cells = s[..., :n]
        ext = jnp.concatenate([cells[..., :1], cells, cells[..., n - 1:n]], axis=-1)
        faces = 0.5 * (ext[..., 1:] + ext[..., :-1])
        pad = [(0, 0)] * (faces.ndim - 1) + [(0, length - (n + 1))]
        return jnp.pad(faces, pad, constant_values=1.0)

    w, e, s, n = (total[:, i] for i in range(4))
    if kind == "u":
        sides = (across(w), across(e), along(s, nx), along(n, nx))
    else:
        sides = (along(w, ny), along(e, ny), across(s), across(n))
    return jnp.stack(sides, axis=1)


def root_walk_rates(state: State, lead_seconds, metrics: DycoreMetrics, config: BoundaryConfig, *,
                    theta_offset: float, nz: int, ny: int, nx: int, dtype) -> RootWalkRates:
    """Coupled wrfbdy record tendencies for the exact WRF spec-zone walk of a specified root (o5-coast).

    The root leaves hold record k DECOUPLED by that record's own mass ``MT_k = MUB + MU_B(k)``
    (d02_replay.load_wrfbdy_boundary_leaves).  Re-coupling every record with the same mass restores the
    real.exe ``couple`` records (module_big_step_utilities_em.F:524) up to the leaf's storage rounding, and
    ``specified_boundary_tendency`` returns ``(rec[l+1]-rec[l])/interval`` scattered to the spec ring with
    WRF's bracket and corner ownership -- the arrays ``spec_bdy_dry`` hands to ``spec_bdyupdate``.
    """

    if state.mu_bdy is None or state.mub_bdy is None:
        raise ValueError("root spec walk needs the root mu_bdy/mub_bdy record leaves")
    cadence = float(config.update_cadence_s)
    leaf_dtype = state.theta_bdy.dtype
    mt = state.mu_bdy.astype(leaf_dtype) + state.mub_bdy.astype(leaf_dtype)   # (time, 4, width, 1, len)
    width, length = int(mt.shape[2]), int(mt.shape[-1])
    col = lambda a: a.astype(leaf_dtype)[None, None, None, :, None]
    c1h, c2h, c1f, c2f = (col(getattr(metrics, n)) for n in ("c1h", "c2h", "c1f", "c2f"))
    nzt, nzw = int(state.theta_bdy.shape[3]), int(state.ph_bdy.shape[3])
    theta_c = (state.theta_bdy - jnp.asarray(theta_offset, leaf_dtype)) * (c1h[..., :nzt, :] * mt + c2h[..., :nzt, :])
    ph_c = state.ph_bdy * (c1f[..., :nzw, :] * mt + c2f[..., :nzw, :])
    msfu = _root_side_strips_2d(metrics.msfuy.astype(leaf_dtype), width, length, 1.0)[None]
    msfv = _root_side_strips_2d(metrics.msfvx.astype(leaf_dtype), width, length, 1.0)[None]
    nzu = int(state.u_bdy.shape[3])
    u_c = state.u_bdy * (c1h[..., :nzu, :] * _root_face_mass(mt, "u", ny, nx) + c2h[..., :nzu, :]) / msfu
    v_c = state.v_bdy * (c1h[..., :nzu, :] * _root_face_mass(mt, "v", ny, nx) + c2h[..., :nzu, :]) / msfv

    def rate(records, z_len, y_len, x_len):
        return specified_boundary_tendency(records, lead_seconds, cadence, z_len=z_len, y_len=y_len,
                                           x_len=x_len, dtype=dtype, config=config)

    return RootWalkRates(
        mu=rate(state.mu_bdy, 1, ny, nx)[0],
        theta=rate(theta_c, nz, ny, nx),
        ph=rate(ph_c, nz + 1, ny, nx),
        u=rate(u_c, nz, ny, nx + 1),
        v=rate(v_c, nz, ny + 1, nx),
    )
