"""Device-resident prognostic state and tendency contracts."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import argparse
import os
from pathlib import Path
from typing import Iterable

import jax
from jax import config
import jax.numpy as jnp

from .ice_diagnostics import ICE_DIAGNOSTIC_LEAVES, ice_diagnostic_leaves
from .grid import GridSpec
from .precision import (
    DEFAULT_DTYPES, GWDO_DIAGNOSTIC_LEAVES,
    GWDO_VOLUME_DIAGNOSTIC_LEAVES, GWDO_SURFACE_DIAGNOSTIC_LEAVES, P3_STATE_LEAVES,
    EPSILON_PBL_SCALARS, EPSILON_PBL_LEAVES, JENSEN_STATE_LEAVES,
    NTU_STATE_LEAVES, NTU_SCALAR_LEAVES, NTU_SURFACE_LEAVES,
    FERRIER_CARRY_LEAVES,
    GODDARD4ICE_CARRY_LEAVES, GODDARD4ICE_HISTORY_LEAVES, THOMPSON_HAIL_SURFACE_LEAVES,
    WRITER_PROFILE_LEAVES,
)


from gpuwrf.contracts.cammg_fields import STATE_FIELDS as CAMMG_STATE_FIELDS, MASS_FIELDS as CAMMG_MASS_FIELDS, INTERFACE_FIELDS as CAMMG_INTERFACE_FIELDS, SURFACE_FIELDS as CAMMG_SURFACE_FIELDS

configure_jax_x64()


def mynn_sfc_wspd_enabled() -> bool:
    """Opt-in WRF surface-layer WSPD handoff to MYNN's implicit boundary."""
    return os.environ.get("GPUWRF_MYNN_SFC_WSPD", "0") == "1"


def _gpu_device() -> jax.Device:
    """Selected JAX device; retain the helper name for existing callers."""

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        return jax.devices()[0]  # Portable JAX backend (CPU/TPU/Metal).
    return _require_supported_gpu_architecture(devices[0])


def _zeros(shape: tuple[int, ...], field: str, device: jax.Device) -> jax.Array:
    """Allocate one state/tendency field on the selected device during init only."""

    create = jnp.ones if field == "f_rimef_phy" else jnp.zeros
    return jax.device_put(create(shape, dtype=DEFAULT_DTYPES.dtype_for(field)), device)


HAIL_MP_PHYSICS_OPTIONS: frozenset[int] = frozenset({7, 17, 18, 19, 24, 26, 27, 38})
HAIL_CONDITIONAL_LEAVES: tuple[str, ...] = ("qh", "Nh", "qvolg", "qvolh", "hail_acc")
AEROSOL_CONDITIONAL_LEAVES: tuple[str, ...] = ("nwfa", "nifa")
EXTRA_SCALAR_BOUNDARY_SPECIES: tuple[str, ...] = (
    "qh", "Nc", "Nn", "Ns", "Ng", "Nh", "qvolg", "qvolh", "nwfa", "nifa",
    *EPSILON_PBL_SCALARS,
    "qt", "qrimef",
    *NTU_SCALAR_LEAVES,
)
SCALAR_BOUNDARY_OPTIONAL_LEAVES: tuple[str, ...] = (
    "qc_bdy",
    "qr_bdy",
    "qi_bdy",
    "qs_bdy",
    "qg_bdy",
    "Ni_bdy",
    "Nr_bdy",
)
CONDITIONAL_STATE_LEAVES: tuple[str, ...] = (
    "qh",
    "Nh",
    "qvolg",
    "qvolh",
    "nwfa",
    "nifa",
    "hail_acc",
    *SCALAR_BOUNDARY_OPTIONAL_LEAVES,
    *(f"{name}_bdy" for name in EXTRA_SCALAR_BOUNDARY_SPECIES),
    "sfc_wspd",
    *P3_STATE_LEAVES,
    "tracers", "tracer_bdy", "tracer_budget",
    *FERRIER_CARRY_LEAVES,
    *GODDARD4ICE_CARRY_LEAVES, *GODDARD4ICE_HISTORY_LEAVES, *THOMPSON_HAIL_SURFACE_LEAVES,
    *JENSEN_STATE_LEAVES,
    *WRITER_PROFILE_LEAVES,
    *NTU_STATE_LEAVES,
    *CAMMG_STATE_FIELDS,
    *ICE_DIAGNOSTIC_LEAVES,
)


def conditional_state_leaves_for_mp(
    mp_physics: int | None = 8,
    *,
    include_all_conditional: bool = False,
) -> tuple[str, ...]:
    """Return State leaves that must exist for a static microphysics selection."""

    mp = 8 if mp_physics is None else int(mp_physics)
    if include_all_conditional:
        # Preserve the legacy scan tail, but CAM-MG histories/optics belong
        # only to Registry's mp11 package (including its retained PBL work).
        cammg = CAMMG_STATE_FIELDS if mp == 11 else ()
        return ("qh", "Nh", "qvolg", "qvolh", "nwfa", "nifa", "hail_acc", *P3_STATE_LEAVES, *FERRIER_CARRY_LEAVES, *GODDARD4ICE_CARRY_LEAVES, *GODDARD4ICE_HISTORY_LEAVES, *THOMPSON_HAIL_SURFACE_LEAVES, *JENSEN_STATE_LEAVES, *NTU_STATE_LEAVES, *cammg, *ICE_DIAGNOSTIC_LEAVES)
    if mp == 11:
        return CAMMG_STATE_FIELDS
    if mp == 9:
        return ("qh", "Nh", "hail_acc")
    if mp == 19:
        return ("qh", "qvolg", "hail_acc")
    if mp == 22:
        # WRF nssl_hail_on=0, nssl_density_on=1: graupel volume only.
        return ("qvolg",)
    if mp == 5:
        return ("qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy", "ferrier_sr")
    if mp == 15:
        return ("qrimef", "ferrier_sr")
    if mp == 7:
        return ("qh", "hail_acc", *GODDARD4ICE_CARRY_LEAVES, *GODDARD4ICE_HISTORY_LEAVES)
    if mp == 38:
        return ("qvolg", "nwfa", "nifa", *THOMPSON_HAIL_SURFACE_LEAVES)
    if mp in HAIL_MP_PHYSICS_OPTIONS:
        return HAIL_CONDITIONAL_LEAVES
    if mp == 28:
        return AEROSOL_CONDITIONAL_LEAVES
    if mp in (50, 51, 52, 53):
        common = ("qir", "qib", "mp_th_old", "mp_qv_old", *ice_diagnostic_leaves(mp))
        return common + (("qi2", "Ni2", "qir2", "qib2") if mp == 52 else ("qzi",) if mp == 53 else ())
    if mp == 55:
        return ("qi2", "Ni2", *JENSEN_STATE_LEAVES, *ice_diagnostic_leaves(mp))
    if mp == 56:
        return ("qh", "Nh", "hail_acc", *NTU_STATE_LEAVES)
    return ()


def is_conditional_state_leaf(name: str) -> bool:
    return name in CONDITIONAL_STATE_LEAVES


def _state_field_shapes(
    grid: GridSpec,
    *,
    mp_physics: int | None = 8,
    include_all_conditional: bool = False,
    gwd_opt: int = 0,
) -> dict[str, tuple[int, ...]]:
    """Returns the SoA field-shape contract for one static physics selection."""

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    mass_3d = (nz, ny, nx)
    surface_2d = (ny, nx)
    boundary_side = max(nx + 1, ny + 1)
    boundary_width = 5
    boundary_mass = (1, 4, boundary_width, nz, boundary_side)
    boundary_face = (1, 4, boundary_width, nz + 1, boundary_side)
    boundary_surface = (1, 4, boundary_width, 1, boundary_side)
    shapes = {
        "u": (nz, ny, nx + 1),
        "v": (nz, ny + 1, nx),
        "w": (nz + 1, ny, nx),
        "theta": mass_3d,
        "qv": mass_3d,
        # v0.20 S1: legacy total aliases p/ph/mu removed (they duplicated the
        # totals); the State p/ph/mu properties re-expose p_total/ph_total/mu_total.
        "p_total": mass_3d,
        "p_perturbation": mass_3d,
        "ph_total": (nz + 1, ny, nx),
        "ph_perturbation": (nz + 1, ny, nx),
        "mu_total": surface_2d,
        "mu_perturbation": surface_2d,
        "qc": mass_3d,
        "qr": mass_3d,
        "qi": mass_3d,
        "qs": mass_3d,
        "qg": mass_3d,
        "Ni": mass_3d,
        "Nr": mass_3d,
        "Ns": mass_3d,
        "Ng": mass_3d,
        "qke": mass_3d,
        "ustar": surface_2d,
        "theta_flux": surface_2d,
        "qv_flux": surface_2d,
        "tau_u": surface_2d,
        "tau_v": surface_2d,
        "rhosfc": surface_2d,
        "fltv": surface_2d,
        "t_skin": surface_2d,
        "soil_moisture": surface_2d,
        "xland": surface_2d,
        "lakemask": surface_2d,
        "mavail": surface_2d,
        "roughness_m": surface_2d,
        "lu_index": surface_2d,
        "rain_acc": surface_2d,
        "snow_acc": surface_2d,
        "graupel_acc": surface_2d,
        "ice_acc": surface_2d,
        "u_bdy": boundary_mass,
        "v_bdy": boundary_mass,
        "theta_bdy": boundary_mass,
        "qv_bdy": boundary_mass,
        "ph_bdy": boundary_face,
        "mu_bdy": boundary_surface,
        "w_bdy": boundary_face,
        "p_bdy": boundary_mass,
        "pb_bdy": boundary_mass,
        "phb_bdy": boundary_face,
        "mub_bdy": boundary_surface,
        # v0.6.0 S0 additive physics leaves (append-only): WDM6 Nc/Nn number
        # concentrations on mass points; cumulus rainc_acc on surface points.
        "Nc": mass_3d,
        "Nn": mass_3d,
        "rainc_acc": surface_2d,
        # v0.15 MYNN SGS-cloud leaves (append-only): closure-2.6 prognostic
        # total-water variance qsq plus the mym_condensation CASE(2) subgrid
        # cloud state (qc_bl/qi_bl/cldfra_bl) the icloud_bl radiation merge and
        # the SGS-aware thlv buoyancy consume.
        "qsq": mass_3d,
        "qc_bl": mass_3d,
        "qi_bl": mass_3d,
        "cldfra_bl": mass_3d,
        # B39 MYNN surface-layer carry (REAL-locked, zero = WRF registry init).
        "mol": surface_2d,
        "hfx": surface_2d,
        "qfx": surface_2d,
        "qsfc": surface_2d,
        "pblh": surface_2d,
        "el_pbl": mass_3d,
        "maxmf": surface_2d,
        "maxwidth": surface_2d,
        "ztop_plume": surface_2d,
    }
    if int(gwd_opt) == 1 or include_all_conditional:
        shapes.update({name: mass_3d for name in GWDO_VOLUME_DIAGNOSTIC_LEAVES})
        shapes.update({name: surface_2d for name in GWDO_SURFACE_DIAGNOSTIC_LEAVES})
    if mynn_sfc_wspd_enabled():
        shapes["sfc_wspd"] = surface_2d
    active = set(
        conditional_state_leaves_for_mp(
            mp_physics,
            include_all_conditional=include_all_conditional,
        )
    )
    if active & set(HAIL_CONDITIONAL_LEAVES):
        # v0.17 ADR-032 graupel/hail substrate, allocated only for hail-family
        # static microphysics selections.
        shapes.update({name: mass_3d for name in ("qh", "Nh", "qvolg", "qvolh") if name in active})
    if active & set(AEROSOL_CONDITIONAL_LEAVES):
        # v0.16 aerosol-aware Thompson (mp=28) prognostic aerosol numbers.
        shapes.update({
            "nwfa": mass_3d,
            "nifa": mass_3d,
        })
    if "hail_acc" in active:
        # v0.17 hail microphysics surface-precip accumulator.
        shapes["hail_acc"] = surface_2d
    shapes.update({name: mass_3d for name in P3_STATE_LEAVES if name in active})
    shapes.update({name: mass_3d for name in JENSEN_STATE_LEAVES if name in active})
    shapes.update({name: mass_3d for name in NTU_SCALAR_LEAVES if name in active})
    shapes.update({name: surface_2d for name in NTU_SURFACE_LEAVES if name in active})
    shapes.update({name: mass_3d for name in ICE_DIAGNOSTIC_LEAVES if name in active})
    shapes.update({name: mass_3d for name in FERRIER_CARRY_LEAVES if name in active})
    if "ferrier_sr" in active:
        shapes["ferrier_sr"] = surface_2d
    shapes.update({name: mass_3d for name in GODDARD4ICE_CARRY_LEAVES + GODDARD4ICE_HISTORY_LEAVES if name in active})
    shapes.update({name: surface_2d for name in THOMPSON_HAIL_SURFACE_LEAVES if name in active})
    if "goddard_sr" in active:
        shapes["goddard_sr"] = surface_2d
    shapes.update({name: mass_3d for name in CAMMG_MASS_FIELDS if name in active})
    shapes.update({name: (grid.nz + 1, grid.ny, grid.nx) for name in CAMMG_INTERFACE_FIELDS if name in active})
    shapes.update({name: surface_2d for name in CAMMG_SURFACE_FIELDS if name in active})
    return shapes


def _leaf_nbytes(leaves: Iterable[jax.Array]) -> int:
    """Computes persistent byte totals from pytree leaves for the spacetime budget."""

    return int(sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in leaves))


def _as_dtype(value, dtype):
    """Cast concrete/traced arrays while leaving abstract lowering args intact."""

    if getattr(value, "dtype", None) == dtype:
        return value
    if hasattr(value, "astype"):
        return value.astype(dtype)
    return jnp.asarray(value, dtype=dtype)


@jax.tree_util.register_pytree_node_class
class BaseState:
    """Read-only WRF base-state fields separated from prognostic State.

    These fields may vary over terrain but do not belong in the high-frequency
    prognostic timestep carry. They are explicit so c2 pressure/geopotential
    helpers do not infer static WRF quantities from reduced c1 state.
    """

    __slots__ = ("pb", "phb", "mub", "t0", "theta_base", "alb")

    def __init__(
        self,
        pb: jax.Array,
        phb: jax.Array,
        mub: jax.Array,
        t0: jax.Array,
        theta_base: jax.Array,
        alb: jax.Array | None = None,
    ) -> None:
        self.pb = pb
        self.phb = phb
        self.mub = mub
        self.t0 = t0
        self.theta_base = theta_base
        # Opt-in two-way feedback makes ALB independent of PHB/MUB.
        # None preserves the released base leaves and reconstruction path.
        self.alb = alb

    @classmethod
    def zeros(cls, grid: GridSpec) -> "BaseState":
        """Allocates base-state placeholders once on the first visible GPU."""

        device = _gpu_device()
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        return cls(
            _zeros((nz, ny, nx), "p", device),
            _zeros((nz + 1, ny, nx), "ph", device),
            _zeros((ny, nx), "mu", device),
            _zeros((nz, ny, nx), "theta", device),
            _zeros((nz, ny, nx), "theta", device),
        )

    @property
    def mu_base(self) -> jax.Array:
        """Compatibility alias for the WRF base dry-column mass ``mub``."""

        return self.mub

    def replace(self, **updates) -> "BaseState":
        """Returns an updated base-state pytree with explicit field names."""

        values = {name: getattr(self, name, None) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def bytes(self) -> int:
        """Reports persistent base-state bytes for proof-object generation."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents base-state arrays as JAX leaves."""

        return tuple(getattr(self, name, None) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds BaseState after JAX transformations."""

        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class BoundaryState:
    """Time-interpolated lateral forcing separated from prognostic State.

    The current M6 State still carries legacy six-leaf boundary arrays for
    compatibility. c2 code should use this object so future boundary replay can
    add pressure, base-pressure, and vertical-velocity forcing without widening
    the prognostic timestep state.
    """

    __slots__ = ("u", "v", "w", "theta", "qv", "p", "pb", "ph", "phb", "mu", "mub")

    def __init__(
        self,
        u: jax.Array,
        v: jax.Array,
        w: jax.Array,
        theta: jax.Array,
        qv: jax.Array,
        p: jax.Array,
        pb: jax.Array,
        ph: jax.Array,
        phb: jax.Array,
        mu: jax.Array,
        mub: jax.Array,
    ) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.p = p
        self.pb = pb
        self.ph = ph
        self.phb = phb
        self.mu = mu
        self.mub = mub

    @classmethod
    def zeros(cls, grid: GridSpec, *, n_times: int = 1) -> "BoundaryState":
        """Allocates a complete boundary forcing schema once on the first GPU."""

        device = _gpu_device()
        nz = grid.nz
        boundary_side = max(grid.nx + 1, grid.ny + 1)
        shape_mass = (n_times, 4, nz, boundary_side)
        shape_face = (n_times, 4, nz + 1, boundary_side)
        shape_mu = (n_times, 4, 1, boundary_side)
        return cls(
            _zeros(shape_mass, "u", device),
            _zeros(shape_mass, "v", device),
            _zeros(shape_face, "w", device),
            _zeros(shape_mass, "theta", device),
            _zeros(shape_mass, "qv", device),
            _zeros(shape_mass, "p", device),
            _zeros(shape_mass, "p", device),
            _zeros(shape_face, "ph", device),
            _zeros(shape_face, "ph", device),
            _zeros(shape_mu, "mu", device),
            _zeros(shape_mu, "mu", device),
        )

    @classmethod
    def from_legacy_state(cls, state: "State") -> "BoundaryState":
        """Builds the new boundary object from the current six legacy leaves."""

        return cls(
            u=state.u_bdy,
            v=state.v_bdy,
            w=getattr(
                state,
                "w_bdy",
                jnp.zeros((state.u_bdy.shape[0], 4, state.ph_bdy.shape[-2], state.u_bdy.shape[-1]), dtype=state.w.dtype),
            ),
            theta=state.theta_bdy,
            qv=state.qv_bdy,
            p=getattr(state, "p_bdy", jnp.zeros_like(state.theta_bdy, dtype=state.p_perturbation.dtype)),
            pb=getattr(state, "pb_bdy", jnp.zeros_like(state.theta_bdy, dtype=state.p_total.dtype)),
            ph=state.ph_bdy,
            phb=getattr(state, "phb_bdy", jnp.zeros_like(state.ph_bdy, dtype=state.ph_total.dtype)),
            mu=state.mu_bdy,
            mub=getattr(state, "mub_bdy", jnp.zeros_like(state.mu_bdy, dtype=state.mu_total.dtype)),
        )

    def replace(self, **updates) -> "BoundaryState":
        """Returns an updated boundary pytree with explicit field names."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def bytes(self) -> int:
        """Reports persistent boundary bytes for proof-object generation."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents boundary arrays as JAX leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds BoundaryState after JAX transformations."""

        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class Tendencies:
    """Pytree of preallocated tendency buffers matching every prognostic state field."""

    __slots__ = ("u", "v", "w", "theta", "qv", "p", "ph", "mu")

    def __init__(self, u, v, w, theta, qv, p, ph, mu) -> None:
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        self.p = p
        self.ph = ph
        self.mu = mu

    @classmethod
    def zeros(cls, grid: GridSpec) -> "Tendencies":
        """Allocates all tendency buffers once; reused by the timestep scan carry."""

        device = _gpu_device()
        nz, ny, nx = grid.nz, grid.ny, grid.nx
        return cls(
            _zeros((nz, ny, nx + 1), "u", device),
            _zeros((nz, ny + 1, nx), "v", device),
            _zeros((nz + 1, ny, nx), "w", device),
            _zeros((nz, ny, nx), "theta", device),
            _zeros((nz, ny, nx), "qv", device),
            _zeros((nz, ny, nx), "p", device),
            _zeros((nz + 1, ny, nx), "ph", device),
            _zeros((ny, nx), "mu", device),
        )

    def replace(self, **updates) -> "Tendencies":
        """Returns an updated pytree with explicit field names; mirrors State.replace."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def bytes(self) -> int:
        """Reports persistent tendency-buffer bytes for proof-object generation."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents tendency arrays as JAX scan carry leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds Tendencies after JAX transformations."""

        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class State:
    """Pytree of GPU-resident WRF-shaped prognostic and coupling fields.

    Units and staggering:
    - `u`, `v`, `w`: m s^-1 on Arakawa C-grid faces.
    - `theta`: K, `qv/qc/qr/qi/qs/qg`: kg kg^-1 on mass points.
    - `p`/`p_total`: Pa total pressure on mass points; `p_perturbation`
      is WRF perturbation pressure. It is diagnostically refreshed inside
      the dycore after acoustic `ph/theta/mu` changes and is used by c2 PGF
      terms.
    - `ph`/`ph_total`: m2 s^-2 total geopotential on vertical faces;
      `ph_perturbation` is WRF perturbation geopotential advanced by the
      nonhydrostatic acoustic dycore.
    - `mu`/`mu_total`: Pa column dry mass on mass points; `mu_perturbation`
      is the perturbation dry-column mass relative to `BaseState.mub`.
    - `Ni/Nr/Ns/Ng`: m^-3 number concentrations on mass points.
    - `qh`: kg kg^-1 hail mixing ratio; `Nh`: kg^-1 hail number;
      `qvolg/qvolh`: m^3 kg^-1 predicted-density graupel/hail particle volume
      on mass points (ADR-032 hail substrate; inert/zero unless a hail MP
      scheme is selected).
    - `qke`: m2 s^-2 MYNN turbulent kinetic energy on mass points.
    - `ustar`: m s^-1, `theta_flux`: K m s^-1, `qv_flux`: kg kg^-1 m s^-1,
      `tau_u/tau_v`: m2 s^-2, `rhosfc`: kg m^-3, `fltv`: K m s^-1,
      `t_skin`: K, `soil_moisture`: m3 m^-3 on surface mass points.
    - `xland`, `lakemask`, `mavail`: prescribed land/water and moisture
      availability fields from Gen2 `wrfinput_d02`; `roughness_m`: prescribed
      or derived surface roughness length in m; `lu_index`: int32 WRF
      land-use category on mass points from `wrfinput_d02`.
    - `rain_acc/snow_acc/graupel_acc/ice_acc`: mm accumulated precipitation
      on surface mass points.
    - `*_bdy`: time-varying lateral forcing as
      `(time, side=W/E/S/N, bdy_width, z-like, padded side index)`.
      Legacy four-dimensional leaves without `bdy_width` are still accepted
      by the boundary application adapter.
    """

    __slots__ = (
        "u",
        "v",
        "w",
        "theta",
        "qv",
        # v0.20 S1 liveness reduction: the legacy total aliases ``p``/``ph``/``mu``
        # were bitwise-identical duplicates of ``p_total``/``ph_total``/``mu_total``
        # (kept in sync at every ``replace`` + equal at every construction site).
        # They are REMOVED from the hot carry (3 fewer full-grid arrays per step)
        # and re-exposed as read-only properties aliasing the totals, so every
        # reader is unchanged and fp64_default stays BIT-IDENTICAL. (The TOTAL
        # itself cannot be removed bit-identically -- the dycore recovers the base
        # as ``total - perturbation`` each RK stage and reads ``p_total`` directly;
        # that perturbation-authoritative rewrite is v0.20 S4, gated behind the
        # additive fp32 mode.)
        "p_total",
        "p_perturbation",
        "ph_total",
        "ph_perturbation",
        "mu_total",
        "mu_perturbation",
        "qc",
        "qr",
        "qi",
        "qs",
        "qg",
        "Ni",
        "Nr",
        "Ns",
        "Ng",
        "qke",
        "ustar",
        "theta_flux",
        "qv_flux",
        "tau_u",
        "tau_v",
        "rhosfc",
        "fltv",
        "t_skin",
        "soil_moisture",
        "xland",
        "lakemask",
        "mavail",
        "roughness_m",
        "rain_acc",
        "snow_acc",
        "graupel_acc",
        "ice_acc",
        "u_bdy",
        "v_bdy",
        "theta_bdy",
        "qv_bdy",
        "ph_bdy",
        "mu_bdy",
        "w_bdy",
        "p_bdy",
        "pb_bdy",
        "phb_bdy",
        "mub_bdy",
        "lu_index",
        # --- v0.6.0 S0 additive physics leaves (append-only; manager patch) ---
        # WDM6 cloud-droplet (Nc) + CCN (Nn) number concentrations (m^-3) and the
        # cumulus precipitation accumulator (rainc_acc, mm -> RAINC). Appended AT
        # THE END so the prefix of every existing leaf keeps its pytree position;
        # ``__init__`` gives them ``None`` defaults (-> zeros) so a pre-v0.6.0
        # ``cls(*children)`` flatten with the old leaf count still reconstructs.
        "Nc",
        "Nn",
        "rainc_acc",
        # --- v0.15 MYNN SGS-cloud leaves (append-only; same reconstruction
        # contract as the v0.6.0 block above) ---
        "qsq",
        "qc_bl",
        "qi_bl",
        "cldfra_bl",
        # --- v0.17 ADR-032 graupel/hail (qh) substrate leaves (append-only;
        # same reconstruction contract as the blocks above) ---
        # qh   = hail mixing ratio (kg kg^-1; WRF moist QHAIL)
        # Nh   = hail number concentration (kg^-1; WRF scalar QNHAIL)
        # qvolg = graupel particle volume (m^3 kg^-1; WRF scalar QVGRAUPEL)
        # qvolh = hail particle volume (m^3 kg^-1; WRF scalar QVHAIL)
        # Appended after the prior additive blocks so every existing leaf keeps
        # its pytree position. They stay ``None`` unless a hail MP scheme is
        # selected, so the default mp=8 path does not allocate their arrays.
        "qh",
        "Nh",
        "qvolg",
        "qvolh",
        # --- v0.16 additive aerosol-aware Thompson (mp=28) leaves ---
        # WRF QNWFA/QNIFA water-/ice-friendly aerosol number concentrations
        # (kg^-1). Appended after the v0.6.0 + v0.15 + v0.17 ADR-032 hail
        # additions so every existing leaf keeps its pytree position. They stay
        # ``None`` unless aerosol-aware Thompson (mp=28) is selected.
        "nwfa",
        "nifa",
        # v0.17 hail surface-precip accumulator (append-only historical tail).
        # hail_acc = accumulated grid-scale hail (mm; WRF HAILNC). It stays
        # ``None`` unless a hail MP scheme is selected.
        "hail_acc",
        # v0.22 optional wrfbdy scalar boundary leaves.  These stay ``None`` unless
        # standalone/native lateral forcing provides the matching WRF scalar strips.
        "qc_bdy",
        "qr_bdy",
        "qi_bdy",
        "qs_bdy",
        "qg_bdy",
        "Ni_bdy",
        "Nr_bdy",
        # B39 MYNN surface-layer carry: WRF grid%MOL, HFX, QFX, QSFC, PBLH that
        # surface_driver hands to SFCLAY_mynn on the next step. Appended at the
        # end (pytree prefix unchanged); REAL-locked, ``None`` -> zeros.
        "mol",
        "hfx",
        "qfx",
        "qsfc",
        "pblh",
        # WRF history MYNN diagnostics; held from the existing PBL call.
        "el_pbl",
        "maxmf",
        "maxwidth",
        "ztop_plume",
        # WRF GWDO history; absent unless gwd_opt=1, held between calls.
        "dtaux3d",
        "dtauy3d",
        "dusfcg",
        "dvsfcg",
        # BP90: exact in-step SFCLAY_mynn WSPD, including WSTAR/VSGD.
        "sfc_wspd",
        # Held Thompson calc_effectRad diagnostics [m], never transported.
        # Absent when GPUWRF_RRTMG_MP_RE is off; seeded before the scan.
        "re_cloud",
        "re_ice",
        "re_snow",
        # WRF km_opt=2 LES TKE (grid%tke_2, REAL, restart-only). Absent unless
        # diff_opt=2/km_opt=2; seeded (zeros, WRF Registry init) before the scan.
        "tke",
        # Extra Registry moist/scalar force-down records. None outside the
        # selected live-child scheme; append-only to preserve old slot order.
        *(f"{name}_bdy" for name in EXTRA_SCALAR_BOUNDARY_SPECIES[:10]),
        "sms_br", "sms_rmol", "sms_wspd", "sms_gamu", "sms_gamv",
        "sms_ustm",
        *P3_STATE_LEAVES,
        *(f"{name}_bdy" for name in EPSILON_PBL_SCALARS),
        *EPSILON_PBL_LEAVES,
        "tracers", "tracer_bdy", "tracer_budget",
        "qt_bdy", "qrimef_bdy",
        *FERRIER_CARRY_LEAVES,
        *GODDARD4ICE_CARRY_LEAVES, *GODDARD4ICE_HISTORY_LEAVES, *THOMPSON_HAIL_SURFACE_LEAVES,
        *JENSEN_STATE_LEAVES,
        *WRITER_PROFILE_LEAVES,
        *NTU_STATE_LEAVES,
        *(f"{name}_bdy" for name in NTU_SCALAR_LEAVES),
        *CAMMG_STATE_FIELDS,
        *ICE_DIAGNOSTIC_LEAVES,
    )

    def __init__(
        self,
        u: jax.Array,
        v: jax.Array,
        w: jax.Array,
        theta: jax.Array,
        qv: jax.Array,
        p_total: jax.Array,
        p_perturbation: jax.Array,
        ph_total: jax.Array,
        ph_perturbation: jax.Array,
        mu_total: jax.Array,
        mu_perturbation: jax.Array,
        qc: jax.Array,
        qr: jax.Array,
        qi: jax.Array,
        qs: jax.Array,
        qg: jax.Array,
        Ni: jax.Array,
        Nr: jax.Array,
        Ns: jax.Array,
        Ng: jax.Array,
        qke: jax.Array,
        ustar: jax.Array,
        theta_flux: jax.Array,
        qv_flux: jax.Array,
        tau_u: jax.Array,
        tau_v: jax.Array,
        rhosfc: jax.Array,
        fltv: jax.Array,
        t_skin: jax.Array,
        soil_moisture: jax.Array,
        xland: jax.Array,
        lakemask: jax.Array,
        mavail: jax.Array,
        roughness_m: jax.Array,
        rain_acc: jax.Array,
        snow_acc: jax.Array,
        graupel_acc: jax.Array,
        ice_acc: jax.Array,
        u_bdy: jax.Array,
        v_bdy: jax.Array,
        theta_bdy: jax.Array,
        qv_bdy: jax.Array,
        ph_bdy: jax.Array,
        mu_bdy: jax.Array,
        w_bdy: jax.Array | None = None,
        p_bdy: jax.Array | None = None,
        pb_bdy: jax.Array | None = None,
        phb_bdy: jax.Array | None = None,
        mub_bdy: jax.Array | None = None,
        lu_index: jax.Array | None = None,
        # --- v0.6.0 S0 additive physics leaves (append-only; manager patch) ---
        Nc: jax.Array | None = None,
        Nn: jax.Array | None = None,
        rainc_acc: jax.Array | None = None,
        # --- v0.15 MYNN SGS-cloud leaves (append-only) ---
        qsq: jax.Array | None = None,
        qc_bl: jax.Array | None = None,
        qi_bl: jax.Array | None = None,
        cldfra_bl: jax.Array | None = None,
        # --- v0.17 ADR-032 graupel/hail (qh) substrate leaves (append-only) ---
        qh: jax.Array | None = None,
        Nh: jax.Array | None = None,
        qvolg: jax.Array | None = None,
        qvolh: jax.Array | None = None,
        # --- v0.16 additive aerosol-aware Thompson (mp=28) leaves ---
        nwfa: jax.Array | None = None,
        nifa: jax.Array | None = None,
        # v0.17 hail surface-precip accumulator (append-only historical tail).
        hail_acc: jax.Array | None = None,
        # v0.22 optional WRF scalar boundary leaves from wrfbdy.
        qc_bdy: jax.Array | None = None,
        qr_bdy: jax.Array | None = None,
        qi_bdy: jax.Array | None = None,
        qs_bdy: jax.Array | None = None,
        qg_bdy: jax.Array | None = None,
        Ni_bdy: jax.Array | None = None,
        Nr_bdy: jax.Array | None = None,
        mol: jax.Array | None = None,
        hfx: jax.Array | None = None,
        qfx: jax.Array | None = None,
        qsfc: jax.Array | None = None,
        pblh: jax.Array | None = None,
        el_pbl: jax.Array | None = None,
        maxmf: jax.Array | None = None,
        maxwidth: jax.Array | None = None,
        ztop_plume: jax.Array | None = None,
        dtaux3d: jax.Array | None = None,
        dtauy3d: jax.Array | None = None,
        dusfcg: jax.Array | None = None,
        dvsfcg: jax.Array | None = None,
        # v0.20 S1: legacy total aliases accepted for call-site back-compat ONLY
        # (they are no longer pytree leaves; the read-only p/ph/mu properties below
        # re-expose the authoritative totals). A caller that still passes
        # ``p=``/``ph=``/``mu=`` has its value folded into the matching total
        # (they were always bitwise-equal); the total wins when both are given.
        p: jax.Array | None = None,
        ph: jax.Array | None = None,
        mu: jax.Array | None = None,
        sfc_wspd: jax.Array | None = None,
        re_cloud: jax.Array | None = None,
        re_ice: jax.Array | None = None,
        re_snow: jax.Array | None = None,
        tke: jax.Array | None = None,
        qh_bdy: jax.Array | None = None,
        Nc_bdy: jax.Array | None = None,
        Nn_bdy: jax.Array | None = None,
        Ns_bdy: jax.Array | None = None,
        Ng_bdy: jax.Array | None = None,
        Nh_bdy: jax.Array | None = None,
        qvolg_bdy: jax.Array | None = None,
        qvolh_bdy: jax.Array | None = None,
        nwfa_bdy: jax.Array | None = None,
        nifa_bdy: jax.Array | None = None,
        sms_br: jax.Array | None = None,
        sms_rmol: jax.Array | None = None,
        sms_wspd: jax.Array | None = None,
        sms_gamu: jax.Array | None = None,
        sms_gamv: jax.Array | None = None,
        sms_ustm: jax.Array | None = None,
        qir: jax.Array | None = None,
        qib: jax.Array | None = None,
        qi2: jax.Array | None = None,
        Ni2: jax.Array | None = None,
        qir2: jax.Array | None = None,
        qib2: jax.Array | None = None,
        qzi: jax.Array | None = None,
        mp_th_old: jax.Array | None = None,
        mp_qv_old: jax.Array | None = None,
        pek_adv_bdy: jax.Array | None = None,
        pep_adv_bdy: jax.Array | None = None,
        tke_adv_bdy: jax.Array | None = None,
        diss_adv_bdy: jax.Array | None = None,
        tpe_adv_bdy: jax.Array | None = None,
        epsilon_pbl: jax.Array | None = None,
        tpe_pbl: jax.Array | None = None,
        pr_pbl: jax.Array | None = None,
        pek_adv: jax.Array | None = None,
        pep_adv: jax.Array | None = None,
        tke_adv: jax.Array | None = None,
        diss_adv: jax.Array | None = None,
        tpe_adv: jax.Array | None = None,
        epsilon_exch_h: jax.Array | None = None,
        epsilon_exch_m: jax.Array | None = None,
        epsilon_wu: jax.Array | None = None,
        epsilon_wv: jax.Array | None = None,
        epsilon_wt: jax.Array | None = None,
        epsilon_wq: jax.Array | None = None,
        epsilon_pbl_step: jax.Array | None = None,
        tracers: jax.Array | None = None,
        tracer_bdy: jax.Array | None = None,
        tracer_budget: jax.Array | None = None,
        qt_bdy: jax.Array | None = None,
        qrimef_bdy: jax.Array | None = None,
        qt: jax.Array | None = None,
        f_ice_phy: jax.Array | None = None,
        f_rain_phy: jax.Array | None = None,
        f_rimef_phy: jax.Array | None = None,
        qrimef: jax.Array | None = None,
        ferrier_sr: jax.Array | None = None,
        acphysc: jax.Array | None = None,
        acphyse: jax.Array | None = None,
        acphysd: jax.Array | None = None,
        acphyss: jax.Array | None = None,
        acphysm: jax.Array | None = None,
        acphysf: jax.Array | None = None,
        re_cloud_gsfc: jax.Array | None = None,
        re_rain_gsfc: jax.Array | None = None,
        re_ice_gsfc: jax.Array | None = None,
        re_snow_gsfc: jax.Array | None = None,
        re_graupel_gsfc: jax.Array | None = None,
        re_hail_gsfc: jax.Array | None = None,
        goddard_sr: jax.Array | None = None,
        phys_tot: jax.Array | None = None,
        nwfa2d: jax.Array | None = None,
        nifa2d: jax.Array | None = None,
        thompson_sr: jax.Array | None = None,
        thompson_rainnc: jax.Array | None = None,
        thompson_snownc: jax.Array | None = None,
        qvoli: jax.Array | None = None,
        qaoli: jax.Array | None = None,
        qvoli2: jax.Array | None = None,
        qaoli2: jax.Array | None = None,
        qi3: jax.Array | None = None,
        Ni3: jax.Array | None = None,
        qvoli3: jax.Array | None = None,
        qaoli3: jax.Array | None = None,
        rimi: jax.Array | None = None,
        exch_tke: jax.Array | None = None,
        ntu_qdcn: jax.Array | None = None,
        ntu_qtcn: jax.Array | None = None,
        ntu_qccn: jax.Array | None = None,
        ntu_qrcn: jax.Array | None = None,
        ntu_qnin: jax.Array | None = None,
        ntu_fi: jax.Array | None = None,
        ntu_fs: jax.Array | None = None,
        ntu_vi: jax.Array | None = None,
        ntu_vs: jax.Array | None = None,
        ntu_vg: jax.Array | None = None,
        ntu_ai: jax.Array | None = None,
        ntu_as: jax.Array | None = None,
        ntu_ag: jax.Array | None = None,
        ntu_ah: jax.Array | None = None,
        ntu_i3m: jax.Array | None = None,
        ntu_rainnc: jax.Array | None = None,
        ntu_snownc: jax.Array | None = None,
        ntu_sr: jax.Array | None = None,
        ntu_qdcn_bdy: jax.Array | None = None,
        ntu_qtcn_bdy: jax.Array | None = None,
        ntu_qccn_bdy: jax.Array | None = None,
        ntu_qrcn_bdy: jax.Array | None = None,
        ntu_qnin_bdy: jax.Array | None = None,
        ntu_fi_bdy: jax.Array | None = None,
        ntu_fs_bdy: jax.Array | None = None,
        ntu_vi_bdy: jax.Array | None = None,
        ntu_vs_bdy: jax.Array | None = None,
        ntu_vg_bdy: jax.Array | None = None,
        ntu_ai_bdy: jax.Array | None = None,
        ntu_as_bdy: jax.Array | None = None,
        ntu_ag_bdy: jax.Array | None = None,
        ntu_ah_bdy: jax.Array | None = None,
        ntu_i3m_bdy: jax.Array | None = None,
        rh_old_mp: jax.Array | None = None,
        lcd_old_mp: jax.Array | None = None,
        cldfra_old_mp: jax.Array | None = None,
        cldfra_mp: jax.Array | None = None,
        cldfra_mp_all: jax.Array | None = None,
        cldfra_conv: jax.Array | None = None,
        cldfrai: jax.Array | None = None,
        cldfral: jax.Array | None = None,
        wsedl3d: jax.Array | None = None,
        lradius: jax.Array | None = None,
        iradius: jax.Array | None = None,
        cam_kvh3d: jax.Array | None = None,
        cam_tke_pbl: jax.Array | None = None,
        cam_turbtype3d: jax.Array | None = None,
        cam_smaw3d: jax.Array | None = None,
        cam_rainnc: jax.Array | None = None,
        cam_snownc: jax.Array | None = None,
        cam_sr: jax.Array | None = None,
        cam_fatal: jax.Array | None = None,
        cam_snowh: jax.Array | None = None,
        d_ice: jax.Array | None = None,
        rho_ice: jax.Array | None = None,
        v_ice: jax.Array | None = None,
        phi_ice: jax.Array | None = None,
        d_ice2: jax.Array | None = None,
        rho_ice2: jax.Array | None = None,
        v_ice2: jax.Array | None = None,
        phi_ice2: jax.Array | None = None,
        d_ice3: jax.Array | None = None,
        rho_ice3: jax.Array | None = None,
        v_ice3: jax.Array | None = None,
        phi_ice3: jax.Array | None = None,
    ) -> None:
        self.rimi = None if rimi is None else jnp.asarray(rimi, jnp.float32)
        self.exch_tke = None if exch_tke is None else jnp.asarray(exch_tke, jnp.float32)
        self.tracers = None if tracers is None else jnp.asarray(tracers, dtype=jnp.float32)
        self.tracer_bdy = None if tracer_bdy is None else jnp.asarray(tracer_bdy, dtype=jnp.float32)
        self.tracer_budget = None if tracer_budget is None else jnp.asarray(tracer_budget, dtype=jnp.float32)
        self.u = u
        self.v = v
        self.w = w
        self.theta = theta
        self.qv = qv
        # v0.20 S1: fold the legacy aliases into the authoritative totals (they
        # were always bitwise-equal); the total takes precedence when both are
        # supplied. The read-only p/ph/mu properties re-expose them for readers.
        self.p_total = p_total if p_total is not None else p
        self.p_perturbation = p_perturbation
        self.ph_total = ph_total if ph_total is not None else ph
        self.ph_perturbation = ph_perturbation
        self.mu_total = mu_total if mu_total is not None else mu
        self.mu_perturbation = mu_perturbation
        self.qc = qc
        self.qr = qr
        self.qi = qi
        self.qs = qs
        self.qg = qg
        self.Ni = Ni
        self.Nr = Nr
        self.Ns = Ns
        self.Ng = Ng
        self.qke = qke
        self.ustar = ustar
        self.theta_flux = theta_flux
        self.qv_flux = qv_flux
        self.tau_u = tau_u
        self.tau_v = tau_v
        self.rhosfc = rhosfc
        self.fltv = fltv
        self.t_skin = t_skin
        self.soil_moisture = soil_moisture
        self.xland = xland
        self.lakemask = lakemask
        self.mavail = mavail
        self.roughness_m = roughness_m
        self.rain_acc = rain_acc
        self.snow_acc = snow_acc
        self.graupel_acc = graupel_acc
        self.ice_acc = ice_acc
        self.u_bdy = u_bdy
        self.v_bdy = v_bdy
        self.w_bdy = jnp.zeros_like(ph_bdy, dtype=w.dtype) if w_bdy is None else w_bdy
        self.theta_bdy = theta_bdy
        self.qv_bdy = qv_bdy
        self.p_bdy = jnp.zeros_like(theta_bdy, dtype=p_perturbation.dtype) if p_bdy is None else p_bdy
        self.pb_bdy = jnp.zeros_like(theta_bdy, dtype=p_total.dtype) if pb_bdy is None else pb_bdy
        self.ph_bdy = ph_bdy
        self.phb_bdy = jnp.zeros_like(ph_bdy, dtype=ph_total.dtype) if phb_bdy is None else phb_bdy
        self.mu_bdy = mu_bdy
        self.mub_bdy = jnp.zeros_like(mu_bdy, dtype=mu_total.dtype) if mub_bdy is None else mub_bdy
        self.lu_index = (
            jnp.zeros_like(xland, dtype=jnp.int32)
            if lu_index is None
            else _as_dtype(lu_index, jnp.int32)
        )
        # v0.6.0 additive physics leaves. ``None`` -> zeros so existing call sites
        # and pre-v0.6.0 pytree flattens (old leaf count) still construct: Nc/Nn are
        # 3-D mass-point number concentrations (templated on Ni); rainc_acc is the
        # 2-D cumulus precip accumulator (templated on rain_acc). The cast keeps the
        # ADR-007 precision matrix (Nc/Nn FP32-gated, rainc_acc FP64) consistent.
        self.Nc = (
            jnp.zeros_like(qc, dtype=DEFAULT_DTYPES.dtype_for("Nc"))
            if Nc is None
            else _as_dtype(Nc, DEFAULT_DTYPES.dtype_for("Nc"))
        )
        self.Nn = (
            jnp.zeros_like(qc, dtype=DEFAULT_DTYPES.dtype_for("Nn"))
            if Nn is None
            else _as_dtype(Nn, DEFAULT_DTYPES.dtype_for("Nn"))
        )
        self.rainc_acc = (
            jnp.zeros_like(rain_acc, dtype=DEFAULT_DTYPES.dtype_for("rainc_acc"))
            if rainc_acc is None
            else _as_dtype(rainc_acc, DEFAULT_DTYPES.dtype_for("rainc_acc"))
        )
        # v0.15 MYNN SGS-cloud leaves. ``None`` -> zeros (the WRF cold start:
        # qsq=0, no BL cloud) so existing call sites and pre-v0.15 flattens
        # still construct. qsq templates on qke (FP64 turbulence family);
        # qc_bl/qi_bl/cldfra_bl template on qc (FP32-gated diagnostics).
        self.qsq = (
            jnp.zeros_like(qke, dtype=DEFAULT_DTYPES.dtype_for("qsq"))
            if qsq is None
            else _as_dtype(qsq, DEFAULT_DTYPES.dtype_for("qsq"))
        )
        self.qc_bl = (
            jnp.zeros_like(qc, dtype=DEFAULT_DTYPES.dtype_for("qc_bl"))
            if qc_bl is None
            else _as_dtype(qc_bl, DEFAULT_DTYPES.dtype_for("qc_bl"))
        )
        self.qi_bl = (
            jnp.zeros_like(qc, dtype=DEFAULT_DTYPES.dtype_for("qi_bl"))
            if qi_bl is None
            else _as_dtype(qi_bl, DEFAULT_DTYPES.dtype_for("qi_bl"))
        )
        self.cldfra_bl = (
            jnp.zeros_like(qc, dtype=DEFAULT_DTYPES.dtype_for("cldfra_bl"))
            if cldfra_bl is None
            else _as_dtype(cldfra_bl, DEFAULT_DTYPES.dtype_for("cldfra_bl"))
        )
        # v0.17/v0.16 conditional additive leaves. ``None`` means the selected
        # static microphysics scheme does not carry the leaf, so no array is
        # allocated and JAX omits it from the pytree leaves. Enabling a hail or
        # aerosol scheme must pass/ensure concrete arrays before the scan starts.
        self.qh = None if qh is None else _as_dtype(qh, DEFAULT_DTYPES.dtype_for("qh"))
        self.Nh = None if Nh is None else _as_dtype(Nh, DEFAULT_DTYPES.dtype_for("Nh"))
        self.qvolg = None if qvolg is None else _as_dtype(qvolg, DEFAULT_DTYPES.dtype_for("qvolg"))
        self.qvolh = None if qvolh is None else _as_dtype(qvolh, DEFAULT_DTYPES.dtype_for("qvolh"))
        self.nwfa = None if nwfa is None else _as_dtype(nwfa, DEFAULT_DTYPES.dtype_for("nwfa"))
        self.nifa = None if nifa is None else _as_dtype(nifa, DEFAULT_DTYPES.dtype_for("nifa"))
        self.hail_acc = None if hail_acc is None else _as_dtype(hail_acc, DEFAULT_DTYPES.dtype_for("hail_acc"))
        self.qc_bdy = None if qc_bdy is None else _as_dtype(qc_bdy, DEFAULT_DTYPES.dtype_for("qc_bdy"))
        self.qr_bdy = None if qr_bdy is None else _as_dtype(qr_bdy, DEFAULT_DTYPES.dtype_for("qr_bdy"))
        self.qi_bdy = None if qi_bdy is None else _as_dtype(qi_bdy, DEFAULT_DTYPES.dtype_for("qi_bdy"))
        self.qs_bdy = None if qs_bdy is None else _as_dtype(qs_bdy, DEFAULT_DTYPES.dtype_for("qs_bdy"))
        self.qg_bdy = None if qg_bdy is None else _as_dtype(qg_bdy, DEFAULT_DTYPES.dtype_for("qg_bdy"))
        self.Ni_bdy = None if Ni_bdy is None else _as_dtype(Ni_bdy, DEFAULT_DTYPES.dtype_for("Ni_bdy"))
        self.Nr_bdy = None if Nr_bdy is None else _as_dtype(Nr_bdy, DEFAULT_DTYPES.dtype_for("Nr_bdy"))
        # B39 MYNN surface-layer carry: REAL-locked, ``None`` -> zeros (WRF
        # registry initial value; wrfinput carries none of them).
        for name, value in (("mol", mol), ("hfx", hfx), ("qfx", qfx), ("qsfc", qsfc), ("pblh", pblh)):
            dtype = DEFAULT_DTYPES.dtype_for(name)
            object.__setattr__(
                self, name,
                jnp.zeros_like(xland, dtype=dtype) if value is None else _as_dtype(value, dtype),
            )
        for name, value in (("dtaux3d", dtaux3d), ("dtauy3d", dtauy3d),
                            ("dusfcg", dusfcg), ("dvsfcg", dvsfcg)):
            object.__setattr__(self, name, None if value is None else _as_dtype(value, DEFAULT_DTYPES.dtype_for(name)))
        # Allocate before tracing so the ON carry has a stable pytree. WRF's
        # Registry seed is zero; the surface call writes WSPD before MYNN.
        if sfc_wspd is None and mynn_sfc_wspd_enabled():
            sfc_wspd = jnp.zeros_like(xland, dtype=DEFAULT_DTYPES.dtype_for("sfc_wspd"))
        self.sfc_wspd = None if sfc_wspd is None else _as_dtype(sfc_wspd, DEFAULT_DTYPES.dtype_for("sfc_wspd"))

        # WRF Registry starts these history diagnostics at zero. Required
        # restart leaves: old payloads fail the E78 missing-field schema gate.
        for name, value, template in (("el_pbl", el_pbl, qke), ("maxmf", maxmf, xland),
                                      ("maxwidth", maxwidth, xland), ("ztop_plume", ztop_plume, xland)):
            dtype = DEFAULT_DTYPES.dtype_for(name)
            object.__setattr__(self, name, jnp.zeros_like(template, dtype=dtype)
                               if value is None else _as_dtype(value, dtype))
        for name, value in (("re_cloud", re_cloud), ("re_ice", re_ice), ("re_snow", re_snow)):
            object.__setattr__(self, name, None if value is None else _as_dtype(value, DEFAULT_DTYPES.dtype_for(name)))
        object.__setattr__(self, "tke", None if tke is None else _as_dtype(tke, DEFAULT_DTYPES.dtype_for("tke")))
        for name, value in (
            ("qh", qh_bdy), ("Nc", Nc_bdy), ("Nn", Nn_bdy), ("Ns", Ns_bdy),
            ("Ng", Ng_bdy), ("Nh", Nh_bdy), ("qvolg", qvolg_bdy),
            ("qvolh", qvolh_bdy), ("nwfa", nwfa_bdy), ("nifa", nifa_bdy),
            ("pek_adv", pek_adv_bdy), ("pep_adv", pep_adv_bdy), ("tke_adv", tke_adv_bdy),
            ("diss_adv", diss_adv_bdy), ("tpe_adv", tpe_adv_bdy),
            ("qt", qt_bdy), ("qrimef", qrimef_bdy),
        ):
            object.__setattr__(self, f"{name}_bdy", None if value is None else _as_dtype(value, jnp.float32))
        for name, value in (("sms_br",sms_br),("sms_rmol",sms_rmol),("sms_wspd",sms_wspd),
                            ("sms_gamu",sms_gamu),("sms_gamv",sms_gamv),("sms_ustm",sms_ustm)):
            object.__setattr__(self,name,None if value is None else _as_dtype(value,jnp.float32))
        for name, value in zip(P3_STATE_LEAVES, (qir, qib, qi2, Ni2, qir2, qib2, qzi, mp_th_old, mp_qv_old)):
            object.__setattr__(self, name, None if value is None else _as_dtype(value, jnp.float32))
        for name, value in zip(ICE_DIAGNOSTIC_LEAVES,
                               (d_ice, rho_ice, v_ice, phi_ice, d_ice2, rho_ice2, v_ice2, phi_ice2,
                                d_ice3, rho_ice3, v_ice3, phi_ice3), strict=True):
            object.__setattr__(self, name, None if value is None else _as_dtype(value, jnp.float32))
        for name, value in zip(JENSEN_STATE_LEAVES, (qvoli,qaoli,qvoli2,qaoli2,qi3,Ni3,qvoli3,qaoli3),strict=True):
            object.__setattr__(self,name,None if value is None else _as_dtype(value,jnp.float32))
        for name, value in zip(NTU_STATE_LEAVES, (ntu_qdcn, ntu_qtcn, ntu_qccn, ntu_qrcn, ntu_qnin, ntu_fi, ntu_fs, ntu_vi, ntu_vs, ntu_vg, ntu_ai, ntu_as, ntu_ag, ntu_ah, ntu_i3m, ntu_rainnc, ntu_snownc, ntu_sr), strict=True):
            object.__setattr__(self, name, None if value is None else _as_dtype(value, jnp.float32))
        for name, value in (('ntu_qdcn', ntu_qdcn_bdy),('ntu_qtcn', ntu_qtcn_bdy),('ntu_qccn', ntu_qccn_bdy),('ntu_qrcn', ntu_qrcn_bdy),('ntu_qnin', ntu_qnin_bdy),('ntu_fi', ntu_fi_bdy),('ntu_fs', ntu_fs_bdy),('ntu_vi', ntu_vi_bdy),('ntu_vs', ntu_vs_bdy),('ntu_vg', ntu_vg_bdy),('ntu_ai', ntu_ai_bdy),('ntu_as', ntu_as_bdy),('ntu_ag', ntu_ag_bdy),('ntu_ah', ntu_ah_bdy),('ntu_i3m', ntu_i3m_bdy)):
            object.__setattr__(self, f"{name}_bdy", None if value is None else _as_dtype(value, jnp.float32))
        for name, value in zip(EPSILON_PBL_LEAVES,
                               (epsilon_pbl,tpe_pbl,pr_pbl,pek_adv,pep_adv,tke_adv,diss_adv,tpe_adv,
                                epsilon_exch_h,epsilon_exch_m,epsilon_wu,epsilon_wv,epsilon_wt,epsilon_wq,
                                epsilon_pbl_step),strict=True):
            object.__setattr__(self,name,None if value is None else _as_dtype(value,DEFAULT_DTYPES.dtype_for(name)))
        for name, value in (("qt", qt), ("f_ice_phy", f_ice_phy), ("f_rain_phy", f_rain_phy),
                            ("f_rimef_phy", f_rimef_phy), ("qrimef", qrimef), ("ferrier_sr", ferrier_sr)):
            object.__setattr__(self, name, None if value is None else _as_dtype(value, jnp.float32))
        for name, value in (("acphysc", acphysc), ("acphyse", acphyse), ("acphysd", acphysd),
                            ("acphyss", acphyss), ("acphysm", acphysm), ("acphysf", acphysf),
                            ("re_cloud_gsfc", re_cloud_gsfc), ("re_rain_gsfc", re_rain_gsfc),
                            ("re_ice_gsfc", re_ice_gsfc), ("re_snow_gsfc", re_snow_gsfc),
                            ("re_graupel_gsfc", re_graupel_gsfc), ("re_hail_gsfc", re_hail_gsfc),
                            ("goddard_sr", goddard_sr), ("phys_tot", phys_tot), ("nwfa2d", nwfa2d), ("nifa2d", nifa2d), ("thompson_sr", thompson_sr),
                            ("thompson_rainnc", thompson_rainnc), ("thompson_snownc", thompson_snownc)):
            object.__setattr__(self, name, None if value is None else _as_dtype(value, jnp.float32))

        for name, value in zip(CAMMG_STATE_FIELDS, (rh_old_mp, lcd_old_mp, cldfra_old_mp, cldfra_mp, cldfra_mp_all, cldfra_conv, cldfrai, cldfral, wsedl3d, lradius, iradius, cam_kvh3d, cam_tke_pbl, cam_turbtype3d, cam_smaw3d, cam_rainnc, cam_snownc, cam_sr, cam_fatal, cam_snowh), strict=True):
            object.__setattr__(self, name, None if value is None else _as_dtype(value, jnp.int32 if name == "cam_fatal" else jnp.float32))

    # --- v0.20 S1 legacy total aliases (read-only properties; not pytree leaves) ---
    # ``p``/``ph``/``mu`` were bitwise-identical duplicates of the totals carried
    # alongside them in the hot loop. They are removed from ``__slots__`` (3 fewer
    # full-grid arrays in the scan carry) and re-exposed here so every existing
    # reader (``state.p`` etc.) is unchanged and fp64_default stays bit-identical.
    # Writers go through ``replace(p=...)``, which redirects to the total.
    @property
    def p(self) -> jax.Array:
        return self.p_total

    @property
    def ph(self) -> jax.Array:
        return self.ph_total

    @property
    def mu(self) -> jax.Array:
        return self.mu_total

    @classmethod
    def zeros(cls, grid: GridSpec, *, mp_physics: int | None = 8, gwd_opt: int = 0) -> "State":
        """Allocates the State leaves required by one static microphysics option."""

        device = _gpu_device()
        return cls(**{
            field: _zeros(shape, field, device)
            for field, shape in _state_field_shapes(grid, mp_physics=mp_physics, gwd_opt=gwd_opt).items()
        })

    @classmethod
    def from_init(cls, grid: GridSpec, ic: Path, *, mp_physics: int | None = 8) -> "State":
        """Keeps the future IC-loading call shape while M3 only supports zero init."""

        del ic
        return cls.zeros(grid, mp_physics=mp_physics)

    def active_field_names(self) -> tuple[str, ...]:
        """Return slot names that currently carry arrays in pytree order."""

        return tuple(name for name in self.__slots__ if getattr(self, name) is not None)

    def ensure_conditional_leaves(
        self,
        *,
        mp_physics: int | None = 8,
        include_all_conditional: bool = False,
        use_mp_re: int = 1,
        ra_lw_physics: int = 4,
        ra_sw_physics: int = 4,
    ) -> "State":
        """Materialize static-scheme conditional leaves once before a timestep scan."""

        active = conditional_state_leaves_for_mp(
            mp_physics, include_all_conditional=include_all_conditional
        )
        updates: dict[str, jax.Array] = {}
        from gpuwrf.physics.rrtmg_mp_re import mp_re_active, RE_CLOUD_BG, RE_ICE_BG, RE_SNOW_BG
        if int(mp_physics or 0) in (50,51,52,53,55):
            # Registry P3 carries only cloud/ice radii; has_reqs=0 in WRF.
            # physics_init.F:1268-1290 seeds RE_QC/QI_BG when flags are set;
            # the later p3_main clear-column diagnostics (10/25um) differ.
            radius_active = int(use_mp_re)==1 and int(ra_lw_physics)==4 and int(ra_sw_physics)==4
            for name, background in (("re_cloud", RE_CLOUD_BG if radius_active else 0.),
                                     ("re_ice", RE_ICE_BG if radius_active else 0.)):
                if getattr(self, name) is None:
                    updates[name] = jnp.full_like(self.qc, background, dtype=jnp.float32)
            if self.re_snow is not None:
                updates['re_snow'] = None
        elif mp_re_active(use_mp_re=int(use_mp_re), mp_physics=int(mp_physics or 0),
                        ra_lw_physics=int(ra_lw_physics), ra_sw_physics=int(ra_sw_physics)):
            for name, background in (("re_cloud", RE_CLOUD_BG), ("re_ice", RE_ICE_BG), ("re_snow", RE_SNOW_BG)):
                if getattr(self, name) is None:
                    updates[name] = jnp.full_like(self.qc, 0 if int(mp_physics) == 7 else background, dtype=jnp.float32)
        else:
            for name in ("re_cloud", "re_ice", "re_snow"):
                if getattr(self, name) is not None:
                    updates[name] = None
        if "qh" in active and self.qh is None:
            updates["qh"] = jnp.zeros_like(self.qc, dtype=DEFAULT_DTYPES.dtype_for("qh"))
        if "Nh" in active and self.Nh is None:
            updates["Nh"] = jnp.zeros_like(self.qc, dtype=DEFAULT_DTYPES.dtype_for("Nh"))
        if "qvolg" in active and self.qvolg is None:
            updates["qvolg"] = jnp.zeros_like(self.qc, dtype=DEFAULT_DTYPES.dtype_for("qvolg"))
        if "qvolh" in active and self.qvolh is None:
            updates["qvolh"] = jnp.zeros_like(self.qc, dtype=DEFAULT_DTYPES.dtype_for("qvolh"))
        if "nwfa" in active and self.nwfa is None:
            updates["nwfa"] = jnp.zeros_like(self.qc, dtype=DEFAULT_DTYPES.dtype_for("nwfa"))
        if "nifa" in active and self.nifa is None:
            updates["nifa"] = jnp.zeros_like(self.qc, dtype=DEFAULT_DTYPES.dtype_for("nifa"))
        if "hail_acc" in active and self.hail_acc is None:
            updates["hail_acc"] = jnp.zeros_like(self.rain_acc, dtype=DEFAULT_DTYPES.dtype_for("hail_acc"))
        for name in P3_STATE_LEAVES:
            if name in active and getattr(self, name) is None:
                updates[name] = jnp.zeros_like(self.qc, dtype=jnp.float32)
        for name in ICE_DIAGNOSTIC_LEAVES:
            if name in active and getattr(self, name) is None:
                updates[name] = jnp.zeros_like(self.qc, dtype=jnp.float32)
        for name in JENSEN_STATE_LEAVES:
            if name in active and getattr(self,name) is None:
                updates[name]=jnp.zeros_like(self.qc,dtype=jnp.float32)
        for name in FERRIER_CARRY_LEAVES:
            if name in active and getattr(self, name) is None:
                create = jnp.ones_like if name == "f_rimef_phy" else jnp.zeros_like
                updates[name] = create(self.rain_acc if name == "ferrier_sr" else self.qc, dtype=jnp.float32)
        for name in GODDARD4ICE_CARRY_LEAVES + GODDARD4ICE_HISTORY_LEAVES:
            if name in active and getattr(self, name) is None:
                updates[name] = jnp.zeros_like(self.rain_acc if name == "goddard_sr" else self.qc, dtype=jnp.float32)
        for name in NTU_STATE_LEAVES:
            if name in active and getattr(self, name) is None:
                updates[name] = jnp.zeros_like(self.rain_acc if name in NTU_SURFACE_LEAVES else self.qc, dtype=jnp.float32)
        for name in THOMPSON_HAIL_SURFACE_LEAVES:
            if name in active and getattr(self, name) is None:
                updates[name] = jnp.zeros_like(self.rain_acc, dtype=jnp.float32)
        for name in CAMMG_STATE_FIELDS:
            if name in active and getattr(self, name) is None:
                template = self.ph if name in CAMMG_INTERFACE_FIELDS else self.rain_acc if name in CAMMG_SURFACE_FIELDS else self.qc
                updates[name] = jnp.zeros_like(template, dtype=jnp.int32 if name == "cam_fatal" else jnp.float32)
        return self if not updates else self.replace(**updates)

    def replace(self, *, _cast: bool = True, **updates) -> "State":
        """Returns an updated pytree with explicit field names for JAX functional steps.

        ``p_total`` is authoritative; ``p_perturbation`` is a delta the caller
        maintains explicitly against the current ``BaseState.pb``. Updating
        ``p_perturbation`` alone does not auto-recompute ``p_total`` because
        ``pb`` is not visible to ``State.replace``.

        By default each updated value is cast to the *current* field dtype so
        ordinary functional updates cannot silently change precision.  Pass
        ``_cast=False`` to let the supplied dtype stand -- this is how
        ``_enforce_operational_precision(force_fp64=True)`` actually upcasts a
        mixed-precision real-case state to fp64 (otherwise the canonicalisation
        below pins fields back to their loaded dtype and force_fp64 is a no-op;
        Sprint U P0-1 / GPT firm-rule confirm-close).
        """

        # v0.20 S1: ``p``/``ph``/``mu`` are read-only aliases of the totals (no
        # longer pytree leaves). Redirect any legacy-alias write to the
        # authoritative total leaf so the historical ``replace(p=...)`` contract is
        # preserved; the total wins when both are supplied (they were always
        # bitwise-equal).
        for legacy, total in (("p", "p_total"), ("ph", "ph_total"), ("mu", "mu_total")):
            if legacy in updates:
                legacy_value = updates.pop(legacy)
                if total not in updates:
                    updates[total] = legacy_value

        values = {name: getattr(self, name) for name in self.__slots__}
        for name, value in updates.items():
            current = values[name]
            if _cast and current is not None and hasattr(current, "dtype") and hasattr(value, "astype"):
                value = value.astype(current.dtype)
            elif _cast and current is None and value is not None and hasattr(value, "astype"):
                value = value.astype(DEFAULT_DTYPES.dtype_for(name))
            values[name] = value

        def sync_total_perturbation(total: str, perturbation: str) -> None:
            """Maintain the perturbation as a delta when the total alone is updated.

            Mirrors the historical alias-sync arithmetic EXACTLY (bit-identical):
            updating the total without an explicit perturbation shifts the
            perturbation by the same increment, so the caller's
            ``total = base + perturbation`` relation is preserved across the
            update; an explicit perturbation update takes precedence.
            """

            if total in updates and perturbation not in updates:
                old_total = getattr(self, total)
                values[perturbation] = values[perturbation] + (values[total] - old_total)

        sync_total_perturbation("p_total", "p_perturbation")
        sync_total_perturbation("ph_total", "ph_perturbation")
        sync_total_perturbation("mu_total", "mu_perturbation")
        obj = object.__new__(type(self))
        for name in self.__slots__:
            object.__setattr__(obj, name, values[name])
        return obj

    def bytes(self) -> int:
        """Reports persistent state bytes for the spacetime budget."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return _leaf_nbytes(leaves)

    def tree_flatten(self):
        """Presents state arrays as JAX scan carry leaves."""

        return tuple(getattr(self, name) for name in self.__slots__), None

    def __setstate__(self, state):
        """Unpickles States written before a slot existed (B39 leaves -> REAL zeros)."""

        dict_state, slot_state = state if isinstance(state, tuple) else (state, None)
        for source in (dict_state, slot_state):
            for name, value in (source or {}).items():
                object.__setattr__(self, name, value)
        for name in (*GWDO_DIAGNOSTIC_LEAVES, "sfc_wspd", "re_cloud", "re_ice", "re_snow", "tke",
                     *WRITER_PROFILE_LEAVES,
                     *(f"{n}_bdy" for n in EXTRA_SCALAR_BOUNDARY_SPECIES),
                     "sms_br", "sms_rmol", "sms_wspd", "sms_gamu", "sms_gamv", "sms_ustm", *P3_STATE_LEAVES, *JENSEN_STATE_LEAVES, *NTU_STATE_LEAVES, *FERRIER_CARRY_LEAVES, *GODDARD4ICE_CARRY_LEAVES, *GODDARD4ICE_HISTORY_LEAVES, *THOMPSON_HAIL_SURFACE_LEAVES, *EPSILON_PBL_LEAVES, "tracers", "tracer_bdy", "tracer_budget", *CAMMG_STATE_FIELDS, *ICE_DIAGNOSTIC_LEAVES):
            if not hasattr(self, name):
                object.__setattr__(self, name, None)
        for name in ("mol", "hfx", "qfx", "qsfc", "pblh"):
            if not hasattr(self, name):
                object.__setattr__(
                    self, name, jnp.zeros_like(self.xland, dtype=DEFAULT_DTYPES.dtype_for(name))
                )

        # Raw State fixtures use zero Registry diagnostics; restart payload
        # readers validate their explicit field order before constructing State.
        for name in ("el_pbl", "maxmf", "maxwidth", "ztop_plume"):
            if not hasattr(self, name):
                template = self.qke if name == "el_pbl" else self.xland
                object.__setattr__(self, name, jnp.zeros_like(template, dtype=DEFAULT_DTYPES.dtype_for(name)))

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuilds State after JAX transformations.

        This is the EXACT structural inverse of ``tree_flatten`` and must NOT
        re-run ``__init__``'s value canonicalisation (``jnp.asarray(..., int32)``
        dtype casts, ``None`` -> ``zeros_like`` defaults). Two reasons:

        1. **Correctness contract.** ``tree_flatten`` already emitted exactly one
           leaf per ``__slots__`` field, each already in its canonical dtype.
           Reconstruction must write those same leaves straight back so the
           round-trip is the identity. Routing them through ``cls(*children)``
           re-canonicalises (e.g. re-casts ``lu_index``), which can perturb a
           leaf's abstract value and make a JAX scan's carry-in / carry-out
           treedefs (or shardings) compare unequal -- the bug that surfaced as a
           per-chunk recompile and, once the carry was device-committed, as a
           hard ``TypeError`` (see point 2).
        2. **Sentinel tolerance.** JAX legitimately calls ``tree_unflatten`` with
           placeholder leaves that are NOT arrays -- notably
           ``equality_errors_pytreedef`` builds ``tree_unflatten(td, [Leaf]*n)``
           to FORMAT a treedef-mismatch message. ``cls(*children)`` then runs
           ``jnp.asarray(<Leaf>, dtype=int32)`` -> ``int(<Leaf>)`` ->
           ``TypeError: ... not 'Leaf'``, masking the real mismatch and aborting
           tracing before any work runs. Assigning children verbatim accepts any
           leaf object, so JAX can report the underlying structure diff instead
           of crashing in our ``__init__``.
        """

        del aux
        obj = object.__new__(cls)
        for name in cls.__slots__:
            object.__setattr__(obj, name, None)
        for name, value in zip(cls.__slots__, children):
            object.__setattr__(obj, name, value)
        return obj


def _self_test() -> None:
    """Runs a small allocation check for the sprint validation command."""

    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    leaves, _ = jax.tree_util.tree_flatten((state, tendencies))
    platforms = {leaf.devices().pop().platform for leaf in leaves}
    if platforms != {"gpu"}:
        raise RuntimeError(f"expected all leaves on gpu, got {sorted(platforms)}")
    print(f"ok state_bytes={state.bytes()} tendency_bytes={tendencies.bytes()} device=gpu")


def main(argv: list[str] | None = None) -> int:
    """CLI exists only to serve the contract's explicit self-test command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    parser.print_help()
    return 0


# Appended at the end of the module so no existing line moves (source locations in traced programs stay identical).
_GPU_ARCHITECTURE_CHECKED: dict[tuple[object, ...], object] = {}


def _require_supported_gpu_architecture(device: jax.Device) -> jax.Device:
    """Refuse a CUDA GPU below sm_80 before any state lands on it (runtime/gpu_preflight.py).

    Pure Python at allocation time, memoized per device and opt-in value: no traced program changes.
    """

    from gpuwrf.runtime.gpu_preflight import UNSUPPORTED_GPU_ENV, check_jax_device_architecture

    key = (getattr(device, "id", None), str(getattr(device, "device_kind", "")),
           os.environ.get(UNSUPPORTED_GPU_ENV, ""))
    if key not in _GPU_ARCHITECTURE_CHECKED:
        _GPU_ARCHITECTURE_CHECKED[key] = check_jax_device_architecture(device)
    return device


if __name__ == "__main__":
    raise SystemExit(main())
