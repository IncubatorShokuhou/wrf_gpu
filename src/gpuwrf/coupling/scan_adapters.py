"""v0.6.0 State<->scheme SCAN ADAPTERS for the operational forecast loop.

This module supplies the per-scheme ``State -> State`` adapters that the
operational scan (``runtime.operational_mode._physics_boundary_step``) routes
through the dispatcher (``coupling.physics_dispatch``). Each adapter mirrors the
established v0.2.0 coupler contract (``coupling.physics_couplers.thompson_adapter``
/ ``surface_adapter``): slice the resident :class:`State` into the scheme's
column-kernel inputs, call the WRF-savepoint-parity-passed kernel, then reassemble
``State`` from the kernel output (applying the scheme's
``PhysicsTendency.state_replacements`` / ``accumulator_increments`` per the frozen
``contracts.physics_interfaces`` contract, in WRF call order).

Scope (HONEST tracability, audited 2026-06-03 -- see the scan-wire handoff):

* **Microphysics** -- Kessler (1), Purdue-Lin (2), WSM3 (3), WSM5 (4), WSM6 (6),
  Morrison (10), WDM6 (16) are pure
  ``jnp`` column kernels batched over ``(ncol, nlev)`` returning a
  ``PhysicsTendency`` of ``state_replacements``. They are jit/vmap-traceable and
  wire into the GPU scan as drop-in microphysics-slot adapters (the same slot
  ``thompson_adapter`` fills for mp=8).
* **Surface layer** -- revised-MM5 (1) and Pleim-Xiu (7) ``*_run`` paths are
  vectorized ``jnp`` and write the SAME B2 surface-flux handles
  (``ustar``/``theta_flux``/``qv_flux``/``tau_u``/``tau_v``/``rhosfc``/``fltv``)
  that ``surface_adapter`` writes for sf_sfclay=5, so they drop into the
  surface-layer slot.
* **Cumulus** -- KF (1) and BMJ (2) are jit/vmap-traceable per-column kernels;
  their adapters vmap the column step over the grid and thread scheme-specific
  persistent carry through the operational carry's additive ``cumulus_carry``
  leaf. Tendencies (``RTHCUTEN``/``RQ*CUTEN``) are applied as ``state += dt*tend``
  and ``RAINCV`` accumulates into ``rainc_acc`` (mm).
* **PBL** -- YSU (1) / ACM2 (7) are the v0.6.0 ``jax.lax.scan``-traceable / vmap-
  batched rewrites of the host-NumPy single-column kernels
  (``physics.pbl_{ysu,acm2}.{ysu,acm2}_columns``). Their adapters re-derive the
  per-cell surface forcing the kernels consume via the revised-MM5 surface layer
  (``surface_layer_with_diagnostics``) and apply the PBL momentum increment
  A2C-averaged onto the C-grid faces (WRF ``add_a2c_u``/``add_a2c_v``), exactly
  like the MYNN adapter. Per-case savepoint parity re-passes on the traceable path
  (``pbl_gpuop_report.json``).

NOT wired here (kept fail-closed in the scan with scheme-specific reasons; the
dispatcher selects them, ``runtime.operational_mode._resolve_operational_suite``
rejects them loudly):

* **Noah-classic (2) land** -- wired in ``coupling.noahclassic_surface_hook`` as
  the land-surface slot analogue of ``coupling.noahmp_surface_hook``. It is not a
  table entry here because it threads a land carry rather than a plain
  ``State -> State`` adapter.
* **Grell-Freitas (3) cumulus** -- faithful CPU-NumPy reference port
  (``gpu_runnable=False``); excluded from the GPU scan by design (GPU-batching of
  the sequential 16-member closure ensemble + beta-PDF gamma is a post-0.9.0 TODO).
* **New-Tiedtke (16) cumulus** -- v0.23 F2 OPERATIONAL: faithful traceable JAX
  kernel (``cumulus_ntiedtke_jax``, machine-precision vs the fp64 WRF oracle
  savepoints ``proofs/v013/savepoints/cumulus/ntiedtke_case_*.json``) wired via
  adapter is wired yet, so it stays fail-closed.

  (NOTE: modified-Tiedtke ``cu=6`` IS wired -- it is the v0.6.0 GPU-batched jit/vmap
  ``tiedtke_adapter`` below, a stateless ``State -> State`` scan adapter in
  ``CU_STATELESS_SCAN_ADAPTERS``; only GF(3) and New-Tiedtke(16) remain unwired.)

Every adapter is a pure ``State -> State`` (or ``(State, carry) -> (State, carry)``
for KF) function, allocates nothing at import, and writes at the live State dtype
(so ``force_fp64`` stays truly fp64 through physics; fp32-defeat fix).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from gpuwrf.contracts.state import State
from gpuwrf.kernels.layout_pin import small_grid_firewall as _small_grid_firewall
from gpuwrf.coupling.goddard4ice_adapter import goddard4ice_adapter
from gpuwrf.coupling.thompson_hail_adapter import thompson_hail_adapter
from gpuwrf.coupling.physics_couplers import (
    GRAVITY_M_S2,
    P0_PA,
    R_D_OVER_CP,
    _add_a2c_u_increment,
    _add_a2c_v_increment,
    _column_dz_from_state,
    _dry_theta_view,
    _output_dtype,
    _rho_from_state,
    _surface_column_view,
    _temperature_from_theta,
    _theta_m_from_microphysics,
    _theta_m_tendency_from_dry,
    _u_mass,
    _v_mass,
    _w_mass,
    _wrf_phy_prep_rho_from_state,
    _wrf_hydrostatic_pressure_profiles_from_state,
    _wrf_phy_prep_p8w,
    WRF_PHYSICS_G,
)
from gpuwrf.coupling.wrf_phy_prep import wrf_phy_prep
from gpuwrf.physics.microphysics_goddard import goddard_physics_tendency
from gpuwrf.physics.microphysics_kessler import kessler_physics_tendency
from gpuwrf.physics.microphysics_lin import lin_physics_tendency
from gpuwrf.physics.microphysics_morrison import morrison_tendency
from gpuwrf.physics.microphysics_morrison_aero import morrison_aero_tendency
from gpuwrf.physics.microphysics_sbu_ylin import sbu_ylin_physics_tendency
from gpuwrf.physics.microphysics_wdm5 import wdm5_physics_tendency
from gpuwrf.physics.microphysics_wdm6 import wdm6_physics_tendency
from gpuwrf.physics.microphysics_wdm7 import wdm7_physics_tendency
from gpuwrf.physics.microphysics_wsm3 import wsm3_physics_tendency
from gpuwrf.physics.microphysics_wsm5 import wsm5_physics_tendency
from gpuwrf.physics.microphysics_wsm6 import wsm6_physics_tendency
from gpuwrf.physics.microphysics_wsm7 import wsm7_physics_tendency
# BMJ tables initialize on first CU2 use, rather than on every scheme's import.
from gpuwrf.physics.cumulus_kf import step_kf_column
from gpuwrf.physics.cumulus_tiedtke_jax import tiedtke_column_jax
from gpuwrf.physics.cumulus_ntiedtke_jax import ntiedtke_column_jax
from gpuwrf.coupling.scalesas_adapter import scalesas_rates
from gpuwrf.coupling.sas94_adapter import sas94_rates
from gpuwrf.coupling.osas95_adapter import osas95_rates
from gpuwrf.coupling.kf99_adapter import kf99_rates
from gpuwrf.coupling.native_sas_adapter import ksas14_rates, nsas96_rates
from gpuwrf.physics._gf_jax import gfdrv_batched
from gpuwrf.physics.pbl_acm2 import acm2_columns
from gpuwrf.physics.pbl_boulac import TEMIN as BOULAC_TKE_MIN, boulac_columns
from gpuwrf.physics.pbl_ysu import ysu_columns
from gpuwrf.physics.bl_mrf import mrf_columns
from gpuwrf.physics.bl_gfs import bl_gfs_wrf_columns, gfs_columns
from gpuwrf.physics.bl_gbm import gbm_columns
from gpuwrf.physics.bl_shinhong import shinhong_columns
from gpuwrf.physics.bl_camuw import CamUwCarry, camuw_column, initial_camuw_carry
from gpuwrf.physics.sfclay_pleim_xiu import step_pxsfclay_column
from gpuwrf.physics.sfclay_revised_mm5 import step_sfclay_revised_mm5_column
from gpuwrf.physics.sfclay_old_mm5 import sfclay_old_mm5_columns
from gpuwrf.physics.sfclay_gfs import sf_gfs_columns
from gpuwrf.physics.surface_constants import CP_D, EP1, R_D
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

CP_DRY = 1004.0  # J kg^-1 K^-1 (matches R_D_OVER_CP = 287/1004)


# --- shared helpers -----------------------------------------------------------

def _exner_columns(p_columns: jax.Array) -> jax.Array:
    """WRF Exner function ``pi = (p/p0)**(Rd/cp)`` on column arrays."""

    return (jnp.maximum(p_columns, 1.0) / P0_PA) ** R_D_OVER_CP


def _wrf_moist_prep_den_delz(state: State, grid, dtype):
    """WSM DEN/DELZ exactly as the post-RK caller moist_physics_prep_em forms them.

    solve_em.F:3750 -> module_big_step_utilities_em.F:5480/5488:
    ``z_at_w=(ph+phb)/g`` and ``dz8w=z_at_w(k+1)-z_at_w(k)`` in REAL with WRF
    g=9.81; :5542 ``rho=1./(al+alb)`` is the DRY density of calc_p_rho_phi
    (no 1+qv factor; that total form belongs to the earlier phy_prep).
    Without hybrid metrics (isolated callers) the legacy dry EOS density stays.
    """

    z_at_w = jnp.asarray(state.ph_total, jnp.float32) / jnp.float32(WRF_PHYSICS_G)
    delz = (z_at_w[1:] - z_at_w[:-1]).astype(dtype)
    metrics = getattr(grid, "metrics", None)
    if metrics is None:
        return _rho_from_state(state), delz
    alt = _wrf_phy_prep_rho_from_state(state, metrics, output_dtype=jnp.float32, return_alt=True)
    return (jnp.float32(1.0) / alt).astype(dtype), delz


def _mp_in(field3d: jax.Array, ny: int, nx: int, nz: int) -> jax.Array:
    """State leaf ``(nz, ny, nx)`` -> MP-kernel batch ``(ncol=ny*nx, nlev=nz)``.

    The WRF MP column kernels (``kessler_run`` / ``wsm6_run`` / ``morrison_run`` /
    ``wdm6_run``) are ``jax.vmap`` over ONE leading column axis: profiles are
    ``(ncol, nlev)`` (vertical trailing). The operational State is ``(nz, ny, nx)``;
    flatten the two spatial axes to a single batch axis (cf. MYNN's
    ``_flatten_columns_to_batch``).
    """

    return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)


def _apply_mp_replacements(state: State, tendency, *, ny: int, nx: int, nz: int) -> State:
    """Reassemble State from an MP ``PhysicsTendency`` (batch-major kernel output).

    The MP kernels return ``state_replacements`` keyed by State leaf as
    ``(ncol=ny*nx, nlev=nz)`` arrays and per-call mm ``accumulator_increments`` as
    ``(ncol,)``. Their ``rain_acc`` increment is WRF's all-phase RAINNCV;
    snow/graupel/hail are overlapping subsets (e.g. module_mp_wdm6.F:987-1012).
    State and its writer use DISJOINT channels, so remove frozen fallout from
    that increment before accumulating liquid rain. Replacements overwrite
    the leaf; accumulators ``+=``. Writes at the live State dtype.
    """

    def _back3d(value):  # (ncol, nlev) -> (nz, ny, nx)
        return jnp.moveaxis(jnp.asarray(value).reshape(ny, nx, nz), -1, 0)

    updates: dict[str, jax.Array] = {}
    for leaf, value in tendency.state_replacements.items():
        updates[leaf] = _back3d(value).astype(_output_dtype(state, leaf))
    for acc, inc in tendency.accumulator_increments.items():
        dtype = _output_dtype(state, acc)
        prev = jnp.asarray(getattr(state, acc), dtype=dtype)
        amount = jnp.asarray(inc, dtype=dtype).reshape(ny, nx)
        if acc == "rain_acc":
            for frozen in ("snow_acc", "graupel_acc", "ice_acc", "hail_acc"):
                if frozen in tendency.accumulator_increments:
                    amount = amount - jnp.asarray(
                        tendency.accumulator_increments[frozen], dtype=dtype
                    ).reshape(ny, nx)
        updates[acc] = (prev + amount).astype(
            _output_dtype(state, acc)
        )
    # WSM7 produces WRF radiation radii as diagnostics. Hold them only when
    # physics_init's resolved selection allocated the radius carry.
    for name in ("re_cloud", "re_ice", "re_snow"):
        if getattr(state, name) is not None and name in tendency.diagnostics:
            updates[name] = _back3d(tendency.diagnostics[name]).astype(jnp.float32)
    return state.replace(**updates)


#: Public instantaneous precipitation slots each WRF 4.7.1 microphysics CALL fills
#: (module_microphysics_driver.F CASE blocks). Unfilled slots keep their Registry
#: initial 0 and still reach module_surface_driver.F:3283 / noahmpdrv.F:779-783
#: (e.g. SBU-YLin passes RAINNCV only, :2470-2500). RAINNCV is all-phase.
MP_PUBLIC_PRECIP_SLOTS: dict[int, tuple[str, ...]] = {
    1: ("rainncv",), 13: ("rainncv",),
    3: ("rainncv", "snowncv"), 4: ("rainncv", "snowncv"), 14: ("rainncv", "snowncv"),
    2: ("rainncv", "snowncv", "graupelncv"), 6: ("rainncv", "snowncv", "graupelncv"),
    10: ("rainncv", "snowncv", "graupelncv"), 16: ("rainncv", "snowncv", "graupelncv"),
    40: ("rainncv", "snowncv", "graupelncv"), 97: ("rainncv", "snowncv", "graupelncv"),
    24: ("rainncv", "snowncv", "graupelncv", "hailncv"), 26: ("rainncv", "snowncv", "graupelncv", "hailncv"),
}


def _mp_public_precipitation(tendency, ny: int, nx: int, mp_code: int) -> dict:
    """WRF's per-call RAINNCV/SNOWNCV/GRAUPELNCV/HAILNCV for the land-surface caller.

    Instantaneous kernel amounts (never differences of rounded totals). Kessler's
    public RAINNCV is the LAST sedimentation sub-step only (module_mp_kessler.F
    DO WHILE nfall: RAINNCV=ppt*1000 while RAINNC sums every sub-step); the kernel
    keeps that slot in ``diagnostics['rainncv']``.
    """
    slots = MP_PUBLIC_PRECIP_SLOTS[mp_code]
    inc = tendency.accumulator_increments
    surface = lambda value: jnp.asarray(value).reshape(ny, nx)  # noqa: E731
    total = surface(tendency.diagnostics["rainncv"] if "rainncv" in tendency.diagnostics else inc["rain_acc"])
    zero = jnp.zeros_like(total)
    pick = lambda slot, acc: surface(inc[acc]) if slot in slots and acc in inc else zero  # noqa: E731
    snow, graupel, hail = pick("snowncv", "snow_acc"), pick("graupelncv", "graupel_acc"), pick("hailncv", "hail_acc")
    # SR: the scheme's own frozen fraction (WRF SR=SR); Kessler/SBU-YLin never set it (Registry 0).
    sr = surface(tendency.diagnostics["sr"]) if "sr" in tendency.diagnostics else zero
    return dict(rain=total - snow - graupel - hail, snow=snow, graupel=graupel, ice=zero, hail=hail, total=total, sr=sr)


def _with_mp_precipitation(updated: State, tendency, ny: int, nx: int, mp_code: int, return_precipitation: bool):
    return (updated, _mp_public_precipitation(tendency, ny, nx, mp_code)) if return_precipitation else updated


# --- Microphysics adapters (mp_physics) ---------------------------------------

def ferrier5_adapter(state: State, dt: float, grid=None) -> State:
    from gpuwrf.physics.ferrier.adapter import ferrier_adapter
    return ferrier_adapter(state, dt, grid, option=5)


def ferrier15_adapter(state: State, dt: float, grid=None) -> State:
    from gpuwrf.physics.ferrier.adapter import ferrier_adapter
    return ferrier_adapter(state, dt, grid, option=15)

def kessler_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=1 Kessler warm-rain microphysics ``State -> State`` scan adapter."""

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2  # (nz+1, ny, nx)
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = kessler_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(rho), mp(pii), mp(z_mass), mp(dz), float(dt),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    return _with_mp_precipitation(updated.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, "theta"))), tend, ny, nx, 1, return_precipitation)


def lin_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=2 Purdue-Lin single-moment 6-class microphysics scan adapter.

    Lin's terminal-velocity sedimentation is Courant-limited on the geometric
    LEVEL HEIGHT ``z`` (not just the thickness ``dz``), so this adapter passes
    the mass-level height (mean of the bounding interface heights from ``ph/g``)
    in addition to ``dz``. The scheme is theta-based (no Exner conversion of the
    moist update) and returns the 6 moist species + surface precip.
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2  # (nz+1, ny, nx)
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = lin_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        mp(pii), mp(rho), mp(state.p), mp(z_mass), mp(dz), float(dt),
        ht=interface_z[0].reshape(ny * nx),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    # Lin consumes/returns dry TH_PHY (module_mp_lin.F:194/244), while State
    # carries theta_m. Mirror moist_physics_prep/finish_em at this seam.
    return _with_mp_precipitation(updated.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, 'theta'))), tend, ny, nx, 2, return_precipitation)


def sbu_ylin_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=13 SBU-YLin single-moment 5-class microphysics scan adapter.

    Like Purdue-Lin, SBU-YLin's adaptive semi-Lagrangian sedimentation is
    Courant-limited on the geometric LEVEL HEIGHT ``z`` (the mass-level height
    from ``ph/g``), so this adapter passes both the mass-level height and ``dz``,
    plus the surface height ``ht`` (lowest interface ``ph[0]/g``). The diagnostic
    riming-intensity profile ``ri3d`` is retained from the existing scheme
    output; history must not reconstruct it from the later updated state.
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2  # (nz+1, ny, nx)
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    ht = interface_z[0].reshape(ny * nx)  # surface height per column
    tend = sbu_ylin_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs),
        mp(pii), mp(rho), mp(state.p), mp(z_mass), mp(dz), ht, float(dt),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    return _with_mp_precipitation(updated.replace(rimi=jnp.moveaxis(tend.diagnostics["ri3d"].reshape(ny, nx, nz), -1, 0).astype(jnp.float32),
        theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, "theta"))), tend, ny, nx, 13, return_precipitation)


def wsm6_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=6 WSM6 single-moment 6-class microphysics scan adapter."""

    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho, dz = _wrf_moist_prep_den_delz(state, grid, state.theta.dtype)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    tend = wsm6_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    return _with_mp_precipitation(updated.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, "theta"))), tend, ny, nx, 6, return_precipitation)


def goddard_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=97 Goddard GCE single-moment 3-ice microphysics scan adapter.

    Faithful column port of WRF ``phys/module_mp_gsfcgce.F:gsfcgce`` (operational
    ``ihail=0``/``ice2=0``/``itaobraun=1``/``new_ice_sat=2`` call). Like Lin,
    Goddard's ``fall_flux`` sedimentation is Courant-limited on the geometric
    LEVEL HEIGHT ``z`` (mass-level height from ``ph/g``), so this adapter passes
    ``z_mass`` in addition to ``dz``. The kernel is theta-based and returns the 6
    moist species + surface rain/snow/graupel precip. No new prognostic state.
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2  # (nz+1, ny, nx)
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = goddard_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        mp(pii), mp(rho), mp(state.p), mp(z_mass), mp(dz), float(dt),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    return _with_mp_precipitation(updated.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, "theta"))), tend, ny, nx, 97, return_precipitation)


def wsm7_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=24 WSM7 single-moment 7-class microphysics scan adapter.

    WSM7 = WSM6 (rain/snow/graupel) plus a SEPARATE precipitating hail class. It
    consumes/produces the ADR-032 hail mixing-ratio leaf ``qh`` and accumulates
    grid-scale hail into the v0.17 ``hail_acc`` surface accumulator (WRF HAILNC).
    WRF dry-theta caller preparation with ``qh`` threaded; the
    generic ``_apply_mp_replacements`` writes back qh (state replacement) and
    hail_acc (accumulator increment) -- both are real State leaves in v0.17.
    """

    del grid
    state = state.ensure_conditional_leaves(
        mp_physics=24, use_mp_re=int(state.re_cloud is not None))
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho = _rho_from_state(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = wsm7_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg), mp(state.qh),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    return _with_mp_precipitation(updated.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, "theta"))), tend, ny, nx, 24, return_precipitation)


def wsm3_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=3 WSM3 simple-ice microphysics ``State -> State`` scan adapter."""

    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho, dz = _wrf_moist_prep_den_delz(state, grid, state.theta.dtype)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    tend = wsm3_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(_w_mass(state)), mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    return _with_mp_precipitation(updated.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, "theta"))), tend, ny, nx, 3, return_precipitation)


def wsm5_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=4 WSM5 single-moment 5-class microphysics ``State -> State`` scan adapter."""

    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho, dz = _wrf_moist_prep_den_delz(state, grid, state.theta.dtype)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    tend = wsm5_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    return _with_mp_precipitation(updated.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, "theta"))), tend, ny, nx, 4, return_precipitation)


def _morrison_radar_context(tend, state, ny, nx, nz):
    """refl10cm_hm arguments: MP-exit qv1d,qr1d,nr1d,qs1d,ns1d,qg1d,ng1d,t1d and p1d (ncol, nz)."""
    out = tend.state_replacements
    return tuple(out[k] for k in ("qv", "qr", "Nr", "qs", "Ns", "qg", "Ng")) + (
        tend.diagnostics["t_exit"], _mp_in(state.p, ny, nx, nz))


def _with_morrison_side_outputs(updated, tend, state, ny, nx, nz, mp_code, return_precipitation, return_radar):
    """``State[, public precipitation slots][, radar context]`` (the NSSL/Milbrandt/P3 order)."""
    side = ()
    if return_precipitation:
        side += (_mp_public_precipitation(tend, ny, nx, mp_code),)
    if return_radar:
        side += (_morrison_radar_context(tend, state, ny, nx, nz),)
    return (updated, *side) if side else updated


def morrison_adapter(state: State, dt: float, grid=None, *, return_precipitation=False, return_radar=False) -> State:
    """mp=10 Morrison two-moment microphysics scan adapter.

    ``return_precipitation=True`` also returns WRF's public RAINNCV/SNOWNCV/GRAUPELNCV
    slots; ``return_radar=True`` also returns the MP-exit radar context (WRF
    ``diagflag .and. do_radar_ref==1`` call of ``refl10cm_hm``). Both may be requested.
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = morrison_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        mp(state.Ni), mp(state.Ns), mp(state.Nr), mp(state.Ng),
        mp(pii), mp(state.p), mp(dz), mp(_w_mass(state)), float(dt),
        **({"radar": True} if return_radar else {}),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    updated = updated.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, "theta")))
    return _with_morrison_side_outputs(updated, tend, state, ny, nx, nz, 10, return_precipitation, return_radar)


def morrison_aero_adapter(state: State, dt: float, grid=None, *, return_precipitation=False,
                          return_radar=False) -> State:
    """mp=40 aerosol-aware Morrison scan adapter (WRF aercu_opt=0 point).

    Same inputs as the mp=10 adapter plus the WRF ``qnc`` scalar (``Nc``),
    which the scheme overwrites with its constant-droplet output (WRF
    wrapper ``NC(i,k,j)=nc1d(k)``).  aercu_opt>0 is rejected upstream
    (needs cu_physics=11 MSKF + CESM aerosol input, not ported).
    """

    del grid
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    interface_z = state.ph / GRAVITY_M_S2
    dz = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    tend = morrison_aero_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        mp(state.Ni), mp(state.Ns), mp(state.Nr), mp(state.Ng), mp(state.Nc),
        mp(pii), mp(state.p), mp(dz.astype(state.theta.dtype)),
        mp(_w_mass(state).astype(state.theta.dtype)), float(dt),
        **({"radar": True} if return_radar else {}),
    )
    tend.validate_keys()
    updated = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    # moist_physics_prep/finish_em: the wrapper consumes dry TH_PHY and
    # returns dry theta; the prognostic carry remains moist theta_m.
    updated = updated.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, updated.theta, updated.qv, _output_dtype(state, "theta")))
    return _with_morrison_side_outputs(updated, tend, state, ny, nx, nz, 40, return_precipitation, return_radar)


def wdm5_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=14 WDM5 double-moment warm-rain 5-class microphysics scan adapter.

    WDM5 reuses the WDM6 number leaves ``Nc`` (cloud-droplet number) and ``Nn``
    (CCN); no new State leaf is introduced. The kernel returns ``Nc``/``Nr`` as
    replacements and ``Nn`` as a diagnostic; this adapter materializes ``Nn`` into
    the State leaf so the CCN prognostic actually evolves (it would otherwise be
    lost). WDM5 has NO graupel and NO land/sea ``qcr`` switch, so (unlike WDM6) it
    takes no ``slmsk`` argument.
    """

    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho, dz = _wrf_moist_prep_den_delz(state, grid, state.theta.dtype)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    tend = wdm5_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs),
        mp(state.Nn), mp(state.Nc), mp(state.Nr),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt),
    )
    tend.validate_keys()
    next_state = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    # Nn is returned in diagnostics (the kernel reuses the WDM6 number layout but
    # returns Nn as a diagnostic); thread it into the State leaf so CCN evolves.
    nn = tend.diagnostics.get("Nn")
    if nn is not None:
        nn3d = jnp.moveaxis(jnp.asarray(nn).reshape(ny, nx, nz), -1, 0)
        next_state = next_state.replace(Nn=nn3d.astype(_output_dtype(state, "Nn")))
    return _with_mp_precipitation(next_state.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, next_state.theta, next_state.qv, _output_dtype(state, "theta"))), tend, ny, nx, 14, return_precipitation)


def wdm6_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=16 WDM6 double-moment warm-rain microphysics scan adapter.

    WDM6 introduces the additive State leaves ``Nc`` (cloud-droplet number) and
    ``Nn`` (CCN). The kernel returns ``Nc``/``Nr`` as replacements and ``Nn`` as a
    diagnostic; this adapter materializes ``Nn`` into the State leaf so the CCN
    prognostic actually evolves (it would otherwise be lost). ``slmsk`` follows
    WRF ``xland`` (1 land / 2 water). The scheme consumes dry physics theta;
    moist_physics_finish_em restores the prognostic moist theta afterwards.
    """

    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho, dz = _wrf_moist_prep_den_delz(state, grid, state.theta.dtype)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    slmsk_2d = jnp.where(jnp.asarray(state.xland) < 1.5, 1.0, 2.0)
    # One scalar land/sea mask per column, as in the WRF wdm6 wrapper.
    slmsk = jnp.asarray(slmsk_2d).reshape(ny * nx)
    # e41-class (c11, Swiss 42x42): Nn reaches WDM6 column-major (root flow_dep fence), and its column-view
    # transpose became the hero of ONE kInput fusion that absorbed the WDM6 column body (1613 HLO instr,
    # 23 outputs, 102 cloned callees; ptxas > 57 min / 25.9 GiB). Fencing only that view keeps the transpose
    # its own small fusion. Identity op; grids above SMALL_GRID_COLUMNS trace nothing new.
    tend = wdm6_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg),
        _small_grid_firewall(mp(state.Nn), (ny, nx)), mp(state.Nc), mp(state.Nr),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt), slmsk,
    )
    tend.validate_keys()
    next_state = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    # Nn is returned in diagnostics (the kernel predates the additive leaf); thread
    # it into the State leaf so the CCN prognostic evolves.
    nn = tend.diagnostics.get("Nn")
    if nn is not None:
        nn3d = jnp.moveaxis(jnp.asarray(nn).reshape(ny, nx, nz), -1, 0)
        next_state = next_state.replace(Nn=nn3d.astype(_output_dtype(state, "Nn")))
    return _with_mp_precipitation(next_state.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, next_state.theta, next_state.qv, _output_dtype(state, "theta"))), tend, ny, nx, 16, return_precipitation)


def wdm7_adapter(state: State, dt: float, grid=None, *, return_precipitation=False) -> State:
    """mp=26 WDM7 double-moment warm-rain + single-moment hail scan adapter.

    WDM7 = WDM6 (double-moment cloud/rain number Nc/Nr + CCN Nn) plus a SEPARATE
    single-moment precipitating hail class ``qh`` (no hail number). Identical
    prep to ``wdm6_adapter`` (slmsk land/sea mask, Exner, dz) with ``qh`` threaded
    and the ``hail_acc`` surface accumulator (WRF HAILNC) advanced by the generic
    ``_apply_mp_replacements``. As in WDM6, the kernel returns ``Nn`` as a
    diagnostic; this adapter materializes it into the State leaf so CCN evolves.
    """

    state = state.ensure_conditional_leaves(
        mp_physics=26, use_mp_re=int(state.re_cloud is not None))
    nz, ny, nx = state.theta.shape
    mp = lambda f: _mp_in(f, ny, nx, nz)  # noqa: E731
    theta_dry = _dry_theta_view(state)
    rho, dz = _wrf_moist_prep_den_delz(state, grid, state.theta.dtype)
    pii = (jnp.maximum(state.p, 1.0) / P0_PA) ** R_D_OVER_CP
    # WDM7 kernel uses slmsk=1 land / 2 water (it tests slmsk==2.0 for the water
    # qc0 autoconversion threshold, matching proofs/v013_wdm7 SLMSK); derive it
    # from xland (1 land / 2 water).
    slmsk = jnp.where(jnp.asarray(state.xland) < 1.5, 1.0, 2.0).reshape(ny * nx)
    tend = wdm7_physics_tendency(
        mp(theta_dry), mp(state.qv), mp(state.qc), mp(state.qr),
        mp(state.qi), mp(state.qs), mp(state.qg), mp(state.qh),
        mp(state.Nn), mp(state.Nc), mp(state.Nr),
        mp(pii), mp(rho), mp(state.p), mp(dz), float(dt), slmsk,
    )
    tend.validate_keys()
    next_state = _apply_mp_replacements(state, tend, ny=ny, nx=nx, nz=nz)
    nn = tend.diagnostics.get("Nn")
    if nn is not None:
        nn3d = jnp.moveaxis(jnp.asarray(nn).reshape(ny, nx, nz), -1, 0)
        next_state = next_state.replace(Nn=nn3d.astype(_output_dtype(state, "Nn")))
    return _with_mp_precipitation(next_state.replace(theta=_theta_m_from_microphysics(
        theta_dry, state.qv, next_state.theta, next_state.qv, _output_dtype(state, "theta"))), tend, ny, nx, 26, return_precipitation)


# --- Surface-layer adapters (sf_sfclay_physics) -------------------------------

def _surface_layer_bottom_inputs(state: State):
    """Lowest-model-level 2-D inputs the column surface-layer kernels consume."""

    u_mass = _u_mass(state)
    v_mass = _v_mass(state)
    T = _temperature_from_theta(state.theta, state.p)
    interface_z = state.ph.astype(jnp.float64) / GRAVITY_M_S2
    dz_full = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    # Lowest model level (k=0). WRF passes the half-layer reference height; the
    # kernel's WRF wrapper uses 0.5*dz of the lowest layer for the surface stencil,
    # but the column entry takes the lowest full-layer dz like the savepoint cases.
    return {
        "u": u_mass[0],
        "v": v_mass[0],
        "temperature": T[0],
        "qv": state.qv[0],
        "pressure": state.p[0],
        "dz": dz_full[0],
        "psfc": state.p[0],  # surface pressure proxy from lowest-level pressure
    }


def _apply_surface_flux_replacements(state: State, tendency) -> State:
    """Write the B2 surface-flux handles a surface-layer scheme produced."""

    updates: dict[str, jax.Array] = {}
    for leaf, value in tendency.state_replacements.items():
        updates[leaf] = jnp.asarray(value).astype(_output_dtype(state, leaf))
    return state.replace(**updates)


def sfclay_revised_mm5_adapter(state: State, dt: float, grid=None, *, history=False, surface_inputs=None) -> State:
    """sf_sfclay=1 revised-MM5 surface layer scan adapter (writes B2 flux handles)."""

    del dt
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    inp = _surface_layer_bottom_inputs(state) if surface_inputs is None else surface_inputs
    result = step_sfclay_revised_mm5_column(
        inp["u"], inp["v"], inp["temperature"], inp["qv"], inp["pressure"], inp["dz"],
        psfc=inp["psfc"],
        tsk=state.t_skin,
        xland=state.xland,
        lakemask=state.lakemask,
        mavail=state.mavail,
        znt=jnp.maximum(state.roughness_m, 1.0e-4),
        ust=jnp.maximum(state.ustar, 1.0e-3),
        dx=dx,
    )
    updated = _apply_surface_flux_replacements(state, result.tendency)
    if history:
        values = result.diagnostics.surface_layer
        held={name: jnp.asarray(values[key],getattr(state,name).dtype)
              for name,key in (("hfx","HFX"),("qfx","QFX"),("qsfc","QSFC"),("mol","MOL"),("sfc_wspd","WSPD"))
              if getattr(state,name,None) is not None}
        updated=updated.replace(**held)
        return updated, {name.lower(): values[name] for name in ("T2", "Q2", "U10", "V10", "LH", "HFX", "FM", "FH", "BR", "WSPD")}
    return updated


def pleim_xiu_sfclay_adapter(state: State, dt: float, grid=None) -> State:
    """sf_sfclay=7 Pleim-Xiu surface layer scan adapter (writes B2 flux handles)."""

    del dt
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    inp = _surface_layer_bottom_inputs(state)
    theta_bottom = state.theta[0]
    result = step_pxsfclay_column(
        inp["u"], inp["v"], inp["temperature"], inp["qv"], inp["pressure"], inp["dz"],
        theta=theta_bottom,
        psfc=inp["psfc"],
        tsk=state.t_skin,
        xland=state.xland,
        mavail=state.mavail,
        znt=jnp.maximum(state.roughness_m, 1.0e-4),
        ust=jnp.maximum(state.ustar, 1.0e-3),
        dx=dx,
    )
    return _apply_surface_flux_replacements(state, result.tendency)


# v0.13 Tier-3 surface-layer ports. Both produce HFX/QFX/USTAR/U10/V10 directly
# (the WRF in-place style); the helper below converts those to the SAME B2
# kinematic flux handles (theta_flux/qv_flux/tau_u/tau_v/rhosfc/fltv) that the
# revised-MM5 / MYNN adapters write, so they drop into the surface-layer slot.

def _flux_handles_from_hfx_qfx(state: State, *, hfx, qfx, ust, cpm) -> State:
    """Map a scheme's HFX/QFX/USTAR onto the B2 kinematic surface handles.

    Mirrors ``sfclay_revised_mm5.py`` (theta_flux=hfx/(rho*cpm), qv_flux=qfx/rho,
    tau_u/tau_v=-ust^2*u/|wind|, fltv=(1+ep1*qv)*theta_flux + ep1*thx*qv_flux).
    """

    inp = _surface_layer_bottom_inputs(state)
    T0 = _temperature_from_theta(state.theta, state.p)[0]
    rhox = state.p[0] / (R_D * T0 * (1.0 + EP1 * state.qv[0]))
    theta_flux = hfx / jnp.maximum(rhox * cpm, 1.0e-12)
    qv_flux = qfx / jnp.maximum(rhox, 1.0e-12)
    u_b = inp["u"]
    v_b = inp["v"]
    wind = jnp.maximum(jnp.sqrt(u_b * u_b + v_b * v_b), 0.1)
    tau_u = -(ust * ust) * u_b / wind
    tau_v = -(ust * ust) * v_b / wind
    thx = T0 * (P0_PA / state.p[0]) ** R_D_OVER_CP
    fltv = (1.0 + EP1 * state.qv[0]) * theta_flux + EP1 * thx * qv_flux
    updates = {
        "ustar": ust, "theta_flux": theta_flux, "qv_flux": qv_flux,
        "tau_u": tau_u, "tau_v": tau_v, "rhosfc": rhox, "fltv": fltv,
    }
    return state.replace(**{k: jnp.asarray(v).astype(_output_dtype(state, k))
                            for k, v in updates.items()})


def sfclay_old_mm5_adapter(state: State, dt: float, grid=None) -> State:
    """sf_sfclay=91 old-MM5 surface layer scan adapter (writes B2 flux handles)."""

    del dt
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    inp = _surface_layer_bottom_inputs(state)
    z = jnp.zeros_like(state.t_skin)
    out = sfclay_old_mm5_columns(
        inp["u"], inp["v"], inp["temperature"], inp["qv"], inp["pressure"], inp["dz"],
        inp["psfc"], state.t_skin, state.xland,
        jnp.maximum(state.roughness_m, 1.0e-4), jnp.maximum(state.ustar, 1.0e-3),
        z,                                       # mol carry (0 = no prior unstable)
        jnp.full_like(state.t_skin, -1.0),       # qsfc<0 -> recomputed from tsk
        jnp.full_like(state.t_skin, 1000.0),     # pblh (default, as revised-MM5)
        state.mavail, state.lakemask,
        jnp.full_like(state.t_skin, dx),
        z, z,                                    # hfx_in/qfx_in (0 on entry)
        isfflx=1,
    )
    cpm = CP_D * (1.0 + 0.8 * state.qv[0])
    return _flux_handles_from_hfx_qfx(
        state, hfx=out["hfx"], qfx=out["qfx"], ust=out["ust"], cpm=cpm
    )


def gfs_sfclay_adapter(state: State, dt: float, grid=None) -> State:
    """sf_sfclay=3 NCEP-GFS surface layer scan adapter (writes B2 flux handles)."""

    del dt, grid
    inp = _surface_layer_bottom_inputs(state)
    # GFS derives its reference height Z1 from the hydrostatic thickness
    # -RD*Tv*log(p_mid/psfc)/g, so it needs a genuine surface pressure ABOVE the
    # lowest-level pressure (psfc==p_mid -> Z1=0 -> NaN). Derive psfc by adding the
    # hydrostatic weight of the half lowest layer below the mass point:
    # psfc = p_mid + rho * g * 0.5 * dz.
    T0 = inp["temperature"]
    rhox = inp["pressure"] / (R_D * T0 * (1.0 + EP1 * inp["qv"]))
    psfc = inp["pressure"] + rhox * GRAVITY_M_S2 * 0.5 * inp["dz"]
    out = sf_gfs_columns(
        inp["u"], inp["v"], inp["temperature"], inp["qv"], inp["pressure"],
        psfc, state.t_skin, state.xland,
        jnp.maximum(state.roughness_m, 1.0e-4), jnp.maximum(state.ustar, 1.0e-3),
        isfflx=1,
    )
    return _flux_handles_from_hfx_qfx(
        state, hfx=out["hfx"], qfx=out["qfx"], ust=out["ust"], cpm=out["cpm"]
    )


# --- Cumulus adapter (cu_physics) ---------------------------------------------

def _wrf_cumulus_prep(state: State, grid):
    """WRF phy_prep inputs of the legacy cumulus drivers (cumulus_driver.F args, first_rk_step_part1.F:1379-1400).

    Dry ``T=t_phy``, ``PI=pi_phy``, total ``RHO``, ``DZ8W`` (g=9.81), ``P=p_hyd``, ``P8W=p_hyd_w`` and
    ``PSFC=p_hyd_w(1)`` (``coupling.wrf_phy_prep``). Without dycore metrics (isolated kernel tests) the
    pressure/density members keep the legacy non-hydrostatic assembler and Thompson's density.
    """

    prep = wrf_phy_prep(state, getattr(grid, "metrics", None))
    if prep["p_hyd"] is not None:
        p, p8w, rho = prep["p_hyd"], prep["p_hyd_w"], prep["rho"]
    else:
        p = state.p.astype(jnp.float64)
        p8w = jnp.concatenate([p[:1], 0.5 * (p[:-1] + p[1:]), p[-1:]], axis=0)
        rho = _rho_from_state(state)
    return dict(T=prep["t_phy"], pii=_exner_columns(state.p), rho=rho, dz=jnp.maximum(prep["dz8w"], 1.0),
                p=p, p8w=p8w, psfc=prep["psfc"], z_at_w=prep["z_at_w"])


def kf_cps_species_gate(rqc, rqr, rqi, rqs, *, f_qr: bool, f_qi: bool, f_qs: bool):
    """KF_eta_CPS output writes by species flag (module_cu_kfeta.F:493-526).

    .NOT.F_QR: RQRCUTEN=0, RQCCUTEN=DQRDT+DQCDT; RQICUTEN/RQSCUTEN are written only
    if F_QI/F_QS (zero-initialised otherwise).  Static flags: no ops when all true.
    """

    if not f_qr:
        rqc, rqr = rqr + rqc, jnp.zeros_like(rqr)
    if not f_qi:
        rqi = jnp.zeros_like(rqi)
    if not f_qs:
        rqs = jnp.zeros_like(rqs)
    return rqc, rqr, rqi, rqs


def kf_adapter(state: State, dt: float, w0avg, nca, *, grid=None,
               stepcu=5, cudt=0.0, held_tendencies=None, return_tendencies=False,
               mp_physics=None):
    """cu=1 Kain-Fritsch cumulus scan adapter.

    Returns ``(next_state, w0avg_next, nca_next)``. KF is a per-column kernel; the
    adapter vmaps it over the ``(ny*nx)`` grid columns. Its ``w0avg`` running mean
    of vertical velocity and ``nca`` cloud-relaxation countdown are persistent
    state threaded through the operational carry's ``cumulus_carry`` leaf.
    Tendencies are applied ``state += dt*tend`` (WRF ``RTHCUTEN``/``RQ*CUTEN`` are
    rates); ``RAINCV`` (mm/step) accumulates into ``rainc_acc``.

    ``mp_physics`` selects WRF's microphysics-species flags: ``warm_rain`` for Kessler
    (module_physics_init.F:4479-4481) and F_QR/F_QI/F_QS = moist-array membership
    (KF_eta_CPS module_cu_kfeta.F:211-216 -> KF_eta_PARA); KF_eta_CPS writes RQICUTEN/
    RQSCUTEN only if F_QI/F_QS and folds DQRDT into RQCCUTEN when .NOT.F_QR
    (:493-525).  ``None`` keeps the mixed-phase all-species profile (Thompson-class MP).
    """

    from gpuwrf.physics.cumulus_kf import kf_carry_dtype

    real = kf_carry_dtype()  # BP55: WRF REAL on the native KF path, legacy f64 otherwise
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    nz, ny, nx = state.theta.shape
    # P0/F1: WRF cumulus_driver passes KF the phy_prep TOTAL density rho = (1+qv)/alt
    # (module_big_step_utilities_em.F:4856; module_cumulus_driver.F KF_ETA_CPS RHO=rho).  Grid-backed callers
    # rebuild alt exactly as MYNN does; analytic callers without metrics use the same total density from the
    # equation of state (dry EOS density times 1+qv).
    if getattr(grid, "metrics", None) is not None:
        rho = _wrf_phy_prep_rho_from_state(state, grid.metrics, output_dtype=real)
    else:
        rho = _rho_from_state(state) * (1.0 + jnp.asarray(state.qv, real))
    theta_dry = _dry_theta_view(state)   # P0: KF works on WRF phy_prep dry theta / t_phy
    phyd_wrf = real == jnp.float32 and os.environ.get("GPUWRF_KF_PHYD_WRF", "0") == "1"
    if phyd_wrf:
        if getattr(grid, "metrics", None) is None:
            raise ValueError("KF_PHYD_WRF requires WRF hybrid-pressure metrics")
        # first_rk_step_part1.F:1380 passes P_HYD and the independent PI_PHY.
        # This is the same c1/c2, moisture and DNW producer as phy_prep:4946-4968.
        p_kf, _p8w, _psfc = _wrf_hydrostatic_pressure_profiles_from_state(
            state, grid.metrics, output_dtype=real)
        pi_phy = (jnp.asarray(state.p, real) / jnp.asarray(1.e5, real)) ** jnp.asarray(2. / 7., real)
        T = jnp.asarray(theta_dry, real) * pi_phy
        interface_z = state.ph.astype(real) / jnp.asarray(9.81, real)
    else:
        p_kf, pi_phy = state.p, None
        T = _temperature_from_theta(theta_dry, state.p)
        interface_z = state.ph.astype(real) / GRAVITY_M_S2
    dz_full = jnp.maximum(interface_z[1:] - interface_z[:-1], 1.0)
    w_mass = _w_mass(state)

    # Reshape to (ncol, nz) with ncol = ny*nx and vmap the per-column KF step.
    def _cols(field3d):
        cols = jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)
        return cols if real == jnp.float64 else cols.astype(real)

    T_c = _cols(T)
    qv_c = _cols(state.qv)
    p_c = _cols(p_kf)
    pi_c = _cols(pi_phy) if pi_phy is not None else None
    dz_c = _cols(dz_full)
    rho_c = _cols(rho)
    u_c = _cols(_u_mass(state))
    v_c = _cols(_v_mass(state))
    w_c = _cols(w_mass)
    w0avg_c = _cols(jnp.asarray(w0avg, real))
    nca_c = jnp.asarray(nca, real).reshape(ny * nx)

    if mp_physics is None:
        species_flags = {}
        f_qr = f_qi = f_qs = True
    else:
        from gpuwrf.contracts.physics_registry import moist_species_for_mp
        species = moist_species_for_mp(int(mp_physics))
        f_qr, f_qi, f_qs = ("qr" in species), ("qi" in species), ("qs" in species)
        species_flags = dict(warm_rain=int(mp_physics) == 1, f_qi=f_qi, f_qs=f_qs)

    def _one(T0, QV0, P0, DZQ, RHOE, w0a, U0, V0, w_col, nca0, pi0):
        res = step_kf_column(
            T0, QV0, P0, DZQ, RHOE, w0a, U0, V0, float(dt), dx,
            w=w_col, nca=nca0, stepcu=stepcu, cudt=cudt, pi_phy=pi0, **species_flags,
        )
        st = res.tendency.state_tendencies
        cc = res.carry.cumulus
        return (
            st["theta"], st["qv"], st["qc"], st["qr"], st["qi"], st["qs"],
            res.tendency.accumulator_increments["rainc_acc"],
            res.diagnostics.cumulus["pratec"],
            cc["w0avg"], cc["nca"],
        )

    (rth, rqv, rqc, rqr, rqi, rqs, raincv, pratec, w0avg_next_c, nca_next_c) = jax.vmap(_one)(
        T_c, qv_c, p_c, dz_c, rho_c, w0avg_c, u_c, v_c, w_c, nca_c, pi_c
    )

    def _back(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    rqc, rqr, rqi, rqs = kf_cps_species_gate(rqc, rqr, rqi, rqs, f_qr=f_qr, f_qi=f_qi, f_qs=f_qs)
    rates = tuple(_back(rate) for rate in (rth, rqv, rqc, rqr, rqi, rqs))
    rain_rate = pratec.reshape(ny, nx)
    if held_tendencies is not None:
        # KF_eta leaves existing tendencies/PRATEC untouched for active clouds.
        active = jnp.asarray(nca) >= 0.5 * float(dt)
        rates = tuple(jnp.where(active[None], old, new)
                      for old, new in zip(held_tendencies[:6], rates, strict=True))
        rain_rate = jnp.where(active, held_tendencies[6], rain_rate)
    w0avg_next = jnp.moveaxis(w0avg_next_c.reshape(ny, nx, nz), -1, 0)
    nca_next = nca_next_c.reshape(ny, nx)
    if return_tendencies:
        return rates + (rain_rate,), w0avg_next, nca_next
    rth, rqv, rqc, rqr, rqi, rqs = rates
    dt_f = float(dt)
    rain_increment = raincv.reshape(ny, nx) if held_tendencies is None else dt_f * rain_rate
    # P0: RTHCUTEN/RQVCUTEN are DRY-theta / vapour tendencies; WRF conv_t_tendf_to_moist turns them into the
    # theta_m tendency at the OLD state, which this direct step-entry update applies over dt (no dt^2 cross term).
    qv_next = (state.qv + dt_f * rqv).astype(_output_dtype(state, "qv"))
    theta_dtype = jnp.asarray(state.theta).dtype
    theta_next = (
        jnp.asarray(state.theta, theta_dtype)
        + dt_f * _theta_m_tendency_from_dry(rth, rqv, state.theta, state.qv, theta_dtype)
    ).astype(_output_dtype(state, "theta"))
    next_state = state.replace(
        theta=theta_next,
        qv=qv_next,
        qc=(state.qc + dt_f * rqc).astype(_output_dtype(state, "qc")),
        qr=(state.qr + dt_f * rqr).astype(_output_dtype(state, "qr")),
        qi=(state.qi + dt_f * rqi).astype(_output_dtype(state, "qi")),
        qs=(state.qs + dt_f * rqs).astype(_output_dtype(state, "qs")),
        rainc_acc=(
            jnp.asarray(state.rainc_acc, jnp.float64) + rain_increment
        ).astype(_output_dtype(state, "rainc_acc")),
    )
    return next_state, w0avg_next, nca_next


def initial_kf_carry(state: State):
    """Seed the KF ``(w0avg, nca)`` carry: zero w-mean, nca=-100 (no active cloud).

    WRF W0AVG/NCA are REAL: float32 on the native KF path (BP55), legacy f64 otherwise."""

    from gpuwrf.physics.cumulus_kf import kf_carry_dtype

    real = kf_carry_dtype()
    nz, ny, nx = state.theta.shape
    w0avg = jnp.zeros((nz, ny, nx), dtype=real)
    nca = jnp.full((ny, nx), -100.0, dtype=real)
    return (w0avg, nca)


def _cu_condensate_flags(mp_physics) -> tuple[bool, bool]:
    """(F_QC, F_QI): WRF moist-array membership of QC/QI for the selected microphysics.

    The cumulus drivers write RQCCUTEN/RQICUTEN only ``IF (F_QC)``/``IF (F_QI)``
    (module_cu_tiedtke.F:418-438, module_cu_ntiedtke.F:254-272, G3DRV
    module_cu_g3.F:541-563) and phy_cu_ten couples them only if P_QC/P_QI exist
    (module_physics_addtendc.F:1382-1390, :1606-1613).  ``None`` = both present.
    """

    if mp_physics is None:
        return True, True
    from gpuwrf.contracts.physics_registry import moist_species_for_mp

    species = moist_species_for_mp(int(mp_physics))
    return "qc" in species, "qi" in species


def tiedtke_tile_rows(T_c, qv_c, qc_c, qi_c, p_c, p8w_c, dz_c, rho_c, pii_c, u_c, v_c, w_c,
                      qvften_c, qvpblten_c, qfx_c, xland_c, znu_c, dt, *, ny, nx, stepcu=1):
    """CU_TIEDTKE over one tile: the column kernel vmapped over ``(ny*nx)`` row-major columns.

    module_cu_tiedtke.F:959 ``if(icum.eq.0) return`` is a TILE-ROW decision (CU_TIEDTKE calls TIECNV
    once per j with i=its..ite): every column of a row with any first-ascent convection gets the second
    ascent (and may start mid-level convection there); a row without any gets no tendencies.  Rows =
    the full x extent, as in the serial original CPU-WRF (one tile); an MPI/OpenMP patch split of a row
    can change this decision in CPU-WRF itself.  Returns uncoupled RTH/RQV/RQC/RQR/RQI/RQS/RU/RVCUTEN
    ``(ncol, nz)`` and RAINCV ``(ncol,)``.
    """

    def _one(T0, QV0, QC0, QI0, P0, P8W0, DZ0, RHO0, PI0, U0, V0, W0,
             QVF0, QVB0, QFX0, XL0, ZNU0):
        out = tiedtke_column_jax(
            T0, QV0, QC0, QI0, P0, P8W0, DZ0, RHO0, PI0, U0, V0, W0,
            QVF0, QVB0, QFX0, XL0, ZNU0, dt, stepcu=int(stepcu), own_icum_mask=False,
        )
        return (out["RTHCUTEN"], out["RQVCUTEN"], out["RQCCUTEN"], out["RQRCUTEN"],
                out["RQICUTEN"], out["RQSCUTEN"], out["RUCUTEN"], out["RVCUTEN"],
                out["RAINCV"], out["ICUM1"])

    (*rates, raincv, icum1) = jax.vmap(_one)(
        T_c, qv_c, qc_c, qi_c, p_c, p8w_c, dz_c, rho_c, pii_c, u_c, v_c, w_c,
        qvften_c, qvpblten_c, qfx_c, xland_c, znu_c,
    )
    row_active = jnp.any(icum1.reshape(ny, nx), axis=1, keepdims=True)
    col_active = jnp.broadcast_to(row_active, (ny, nx)).reshape(ny * nx)
    rates = [jnp.where(col_active[:, None], f, 0.0) for f in rates]
    return (*rates, jnp.where(col_active, raincv, 0.0))


def tiedtke_adapter(
    state: State,
    dt: float,
    grid=None,
    *,
    return_raincv: bool = False,
    stepcu: int = 1,
    qvften: jax.Array | None = None,
    qvpblten: jax.Array | None = None,
    mp_physics: int | None = None,
    return_rates: bool = False,
) -> State:
    """cu=6 modified-Tiedtke cumulus ``State -> State`` scan adapter.

    ``return_rates=True`` (F9): ``(rates, RAINCV)`` with the uncoupled RTHCUTEN /
    RQV/RQC/RQICUTEN / mass-point RU/RVCUTEN that phy_cu_ten adds for TIEDTKE
    (module_physics_addtendc.F:1583-1618), for the caller's RK source coupling.

    Tiedtke is a per-column kernel (``cumulus_tiedtke_jax.tiedtke_column_jax``); the
    adapter vmaps it over the ``(ny*nx)`` grid columns. Unlike KF, modified-Tiedtke
    carries NO persistent cumulus state, so this is a plain ``State -> State``
    adapter (like the microphysics adapters).

    Tendencies are applied ``state += dt*tend`` (WRF ``RTHCUTEN``/``RQ*CUTEN`` are
    rates over the cumulus step); ``RAINCV`` (mm/step) accumulates into
    ``rainc_acc``. The cumulus call cadence ``stepcu`` defaults to 1 in the
    operational scan (the cumulus slot is invoked every dynamics step here; the
    savepoint oracle uses WRF's STEPCU=5 -- the kernel multiplies dt by stepcu
    internally, so ``stepcu=1, dt=dt`` is the every-step coupling).

    The Tiedtke-specific inputs not in the frozen B2 contract are assembled here,
    pure-``jnp``, no host transfer:
      * ``P8W`` interface pressure (nz+1): interior faces = mean of adjacent
        levels, edges by zero-gradient (same assembler the YSU/ACM2 adapter uses).
      * ``ZNU`` eta at mass levels: WRF eta relation ``(p - p_top)/(p_sfc - p_top)``
        per column (used only by the below-cloud rain-evaporation coefficient).
      * ``QVFTEN`` / ``QVPBLTEN`` advective / PBL moisture forcing: threaded by
        the operational runtime from the WRF flux-form qv-advection diagnostic
        and the PBL-slot qv increment.  Direct adapter calls may omit them, in
        which case they remain zero for isolated kernel tests.
      * ``QFX`` surface moisture flux from the B2 kinematic ``qv_flux`` handle.
    """

    nz, ny, nx = state.theta.shape
    # cumulus_driver.F:1397 receives phy_prep's dry T_PHY and total RHO;
    # State.theta is theta_m when use_theta_m=1 (big_step_utilities:4834/4856).
    metrics = getattr(grid, "metrics", None)
    rho = (_wrf_phy_prep_rho_from_state(state, metrics, output_dtype=state.theta.dtype)
           if metrics is not None else _rho_from_state(state) * (1.0 + state.qv))
    T = _temperature_from_theta(_dry_theta_view(state), state.p)
    pii = _exner_columns(state.p)
    # cumulus_driver.F:1404 PCPS=p_hyd, P8W=p_hyd_w, DZ8W with g=9.81 (phy_prep).
    prep = _wrf_cumulus_prep(state, grid)
    dz_full, p, p8w = prep["dz"], prep["p"], prep["p8w"]
    w_mass = _w_mass(state)
    eta = getattr(grid, "eta_levels", None)
    if eta is not None and int(jnp.shape(eta)[0]) == nz + 1:
        # cumulus_driver.F:1519 ZNU = grid%znu (half-level eta, znu(k) = (znw(k)+znw(k+1))/2 as
        # real.exe builds it); tiedtke.F:388 sig1 -> cevapcu (:2343).
        eta = jnp.asarray(eta, jnp.float64)
        znu = jnp.broadcast_to((0.5 * (eta[:-1] + eta[1:]))[:, None, None], (nz, ny, nx))
    else:
        # Grid-less kernel tests: eta-coordinate proxy (p - p_top)/(p_sfc - p_top), clipped [0,1].
        p_top = jnp.maximum(p8w[-1], 1.0)
        p_sfc = jnp.maximum(p8w[0], p_top + 1.0)
        znu = jnp.clip((p - p_top) / (p_sfc - p_top), 0.0, 1.0)  # (nz, ny, nx)

    # W on (nz+1) interfaces: vertical velocity at faces (state.w is C-grid w).
    w_int = jnp.asarray(state.w, jnp.float64)
    if w_int.shape[0] == nz:  # mass-level w fallback: pad a top face
        w_int = jnp.concatenate([w_int, w_int[-1:]], axis=0)

    qv_flux = jnp.asarray(state.qv_flux, jnp.float64)  # kg kg^-1 m s^-1 (kinematic)
    rhosfc = jnp.asarray(state.rhosfc, jnp.float64)
    qfx_2d = qv_flux * rhosfc  # kg m^-2 s^-1
    xland_2d = jnp.asarray(state.xland, jnp.float64)

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)

    def _cols1(field3d):  # (nz+1, ny, nx) -> (ncol, nz+1)
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz + 1)

    T_c = _cols(T)
    qv_c = _cols(jnp.maximum(state.qv, 0.0))
    qc_c = _cols(jnp.maximum(state.qc, 0.0))
    qi_c = _cols(jnp.maximum(state.qi, 0.0))
    p_c = _cols(p)
    p8w_c = _cols1(p8w)
    dz_c = _cols(dz_full)
    rho_c = _cols(rho)
    pii_c = _cols(pii)
    u_c = _cols(_u_mass(state))
    v_c = _cols(_v_mass(state))
    w_c = _cols1(w_int)
    qvften_c = _cols(
        jnp.zeros_like(state.qv) if qvften is None else jnp.asarray(qvften, jnp.float64)
    )
    qvpblten_c = _cols(
        jnp.zeros_like(state.qv) if qvpblten is None else jnp.asarray(qvpblten, jnp.float64)
    )
    znu_c = _cols(znu)
    qfx_c = qfx_2d.reshape(ny * nx)
    xland_c = xland_2d.reshape(ny * nx)
    dt_f = float(dt)

    (rth, rqv, rqc, rqr, rqi, rqs, ru, rv, raincv) = tiedtke_tile_rows(
        T_c, qv_c, qc_c, qi_c, p_c, p8w_c, dz_c, rho_c, pii_c, u_c, v_c, w_c,
        qvften_c, qvpblten_c, qfx_c, xland_c, znu_c, dt_f, ny=ny, nx=nx, stepcu=int(stepcu))
    f_qc, f_qi = _cu_condensate_flags(mp_physics)
    rqc = rqc if f_qc else jnp.zeros_like(rqc)
    rqi = rqi if f_qi else jnp.zeros_like(rqi)

    def _back(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    if return_rates:
        return (dict(theta=_back(rth), qv=_back(rqv), qc=_back(rqc) if f_qc else None,
                     qi=_back(rqi) if f_qi else None, u=_back(ru), v=_back(rv)),
                raincv.reshape(ny, nx))

    # Momentum increment from RUCUTEN/RVCUTEN onto C-grid faces (A2C, like PBL).
    du_mass = dt_f * _back(ru)
    dv_mass = dt_f * _back(rv)
    u_new = _add_a2c_u_increment(state.u, du_mass).astype(_output_dtype(state, "u"))
    v_new = _add_a2c_v_increment(state.v, dv_mass).astype(_output_dtype(state, "v"))

    next_state = state.replace(
        u=u_new,
        v=v_new,
        theta=(state.theta + dt_f * _theta_m_tendency_from_dry(
            _back(rth), _back(rqv), state.theta, state.qv, state.theta.dtype
        )).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back(rqv)).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * _back(rqc)).astype(_output_dtype(state, "qc")),
        qr=(state.qr + dt_f * _back(rqr)).astype(_output_dtype(state, "qr")),
        qi=(state.qi + dt_f * _back(rqi)).astype(_output_dtype(state, "qi")),
        qs=(state.qs + dt_f * _back(rqs)).astype(_output_dtype(state, "qs")),
        rainc_acc=(
            jnp.asarray(state.rainc_acc, jnp.float64) + raincv.reshape(ny, nx)
        ).astype(_output_dtype(state, "rainc_acc")),
    )
    # WRF RAINCV of this call (module_surface_driver.F:1568/:3283 -> Noah-MP MP_RAINC).
    return (next_state, raincv.reshape(ny, nx)) if return_raincv else next_state


def ntiedtke_adapter(
    state: State,
    dt: float,
    grid=None,
    *,
    return_raincv: bool = False,
    stepcu: int = 1,
    qvften: jax.Array | None = None,
    thften: jax.Array | None = None,
    mp_physics: int | None = None,
    return_rates: bool = False,
) -> State:
    """cu=16 New-Tiedtke cumulus ``State -> State`` scan adapter.

    ``return_rates=True`` (F9): ``(rates, RAINCV)`` -- NTIEDTKE's phy_cu_ten set
    (module_physics_addtendc.F:1583-1618) uncoupled, for the RK source coupling.

    New-Tiedtke is a per-column kernel
    (``cumulus_ntiedtke_jax.ntiedtke_column_jax``, machine-precision-proven vs
    the fp64 WRF oracle savepoints); the adapter vmaps it over the ``(ny*nx)``
    grid columns. Like modified Tiedtke (cu=6) it carries NO persistent cumulus
    state, so this is a plain ``State -> State`` adapter.

    Tendencies are applied ``state += dt*tend``; ``RAINCV`` (mm/step)
    accumulates into ``rainc_acc``. ``stepcu=1``: the cumulus slot runs every
    dynamics step (the kernel multiplies dt by stepcu internally, mirroring
    WRF's cadence contract exactly as the cu=6 adapter does).

    WRF-driver coupling (module_cumulus_driver.F CASE(NTIEDTKESCHEME)):
      * ``QVFTEN`` = WRF RQVFTEN (advective + PBL moisture forcing): threaded
        by the operational runtime as flux-form qv-advection diagnostic + the
        PBL-slot qv increment. Zero when omitted (isolated kernel tests only).
      * ``THFTEN`` = WRF RTHFTEN (theta-space forcing): threaded as the
        accumulated non-convective physics theta forcing of this step
        (radiation + surface + PBL slots). NAMED COUPLING CAVEAT: WRF's
        RTHFTEN additionally contains the advective theta tendency; this
        port's dycore does not expose a step-entry theta-advection
        diagnostic, so that component is absent here (closure-modulating
        only -- the convective tendencies themselves are oracle-proven).
      * ``HFX``/``QFX`` from the B2 kinematic flux handles
        (``rhosfc*CP_DRY*theta_flux`` / ``rhosfc*qv_flux``), the same
        reconstruction the GF adapter uses.
      * ``DX`` from ``grid.projection.dx_m`` (the kernel's scale-aware
        closure factor); ``itimestep=2`` semantics (forcing active) -- WRF
        zeroes the forcing only on the very first model step.
    """

    nz, ny, nx = state.theta.shape
    # cumulus_driver.F:1421-1426 CU_NTIEDTKE_DRIVER(T3D=t_phy, PI3D=pi_phy, RHO3D=rho,
    # DZ8W, PCPS=p_hyd, P8W=p_hyd_w) from phy_prep (first_rk_step_part1.F:1379-1400).
    prep = _wrf_cumulus_prep(state, grid)
    rho, T, pii, dz_full, p, p8w = (prep[k] for k in ("rho", "T", "pii", "dz", "p", "p8w"))

    # W on (nz+1) interfaces (state.w is C-grid w; mass-level fallback pads top).
    w_int = jnp.asarray(state.w, jnp.float64)
    if w_int.shape[0] == nz:
        w_int = jnp.concatenate([w_int, w_int[-1:]], axis=0)

    rhosfc = jnp.asarray(state.rhosfc, jnp.float64)
    theta_flux = jnp.asarray(state.theta_flux, jnp.float64)  # K m s^-1
    qv_flux = jnp.asarray(state.qv_flux, jnp.float64)        # kg kg^-1 m s^-1
    hfx_2d = rhosfc * CP_DRY * theta_flux                    # W m^-2
    qfx_2d = rhosfc * qv_flux                                # kg m^-2 s^-1
    xland_2d = jnp.asarray(state.xland, jnp.float64)
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)

    def _cols1(field3d):  # (nz+1, ny, nx) -> (ncol, nz+1)
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz + 1)

    T_c = _cols(T)
    qv_c = _cols(jnp.maximum(state.qv, 0.0))
    qc_c = _cols(jnp.maximum(state.qc, 0.0))
    qi_c = _cols(jnp.maximum(state.qi, 0.0))
    p_c = _cols(p)
    p8w_c = _cols1(p8w)
    dz_c = _cols(dz_full)
    rho_c = _cols(rho)
    pii_c = _cols(pii)
    u_c = _cols(_u_mass(state))
    v_c = _cols(_v_mass(state))
    w_c = _cols1(w_int)
    qvften_c = _cols(
        jnp.zeros_like(state.qv) if qvften is None else jnp.asarray(qvften, jnp.float64)
    )
    thften_c = _cols(
        jnp.zeros_like(state.theta) if thften is None else jnp.asarray(thften, jnp.float64)
    )
    qfx_c = qfx_2d.reshape(ny * nx)
    hfx_c = hfx_2d.reshape(ny * nx)
    xland_c = xland_2d.reshape(ny * nx)
    dt_f = float(dt)

    def _one(T0, QV0, QC0, QI0, P0, P8W0, DZ0, RHO0, PI0, U0, V0, W0,
             QVF0, THF0, QFX0, HFX0, XL0):
        out = ntiedtke_column_jax(
            T0, QV0, QC0, QI0, P0, P8W0, DZ0, RHO0, PI0, U0, V0, W0,
            QVF0, THF0, QFX0, HFX0, XL0, dx, dt_f,
            stepcu=int(stepcu), itimestep=2,
        )
        return (out["RTHCUTEN"], out["RQVCUTEN"], out["RQCCUTEN"],
                out["RQICUTEN"], out["RUCUTEN"], out["RVCUTEN"], out["RAINCV"])

    (rth, rqv, rqc, rqi, ru, rv, raincv) = jax.vmap(_one)(
        T_c, qv_c, qc_c, qi_c, p_c, p8w_c, dz_c, rho_c, pii_c, u_c, v_c, w_c,
        qvften_c, thften_c, qfx_c, hfx_c, xland_c,
    )
    f_qc, f_qi = _cu_condensate_flags(mp_physics)
    rqc = rqc if f_qc else jnp.zeros_like(rqc)
    rqi = rqi if f_qi else jnp.zeros_like(rqi)

    def _back(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    if return_rates:
        return (dict(theta=_back(rth), qv=_back(rqv), qc=_back(rqc) if f_qc else None,
                     qi=_back(rqi) if f_qi else None, u=_back(ru), v=_back(rv)),
                raincv.reshape(ny, nx))

    du_mass = dt_f * _back(ru)
    dv_mass = dt_f * _back(rv)
    u_new = _add_a2c_u_increment(state.u, du_mass).astype(_output_dtype(state, "u"))
    v_new = _add_a2c_v_increment(state.v, dv_mass).astype(_output_dtype(state, "v"))

    next_state = state.replace(
        u=u_new,
        v=v_new,
        # Dry RTHCUTEN -> theta_m: conv_t_tendf_to_moist (module_first_rk_step_part2.F:1005-1016).
        theta=(state.theta + dt_f * _theta_m_tendency_from_dry(
            _back(rth), _back(rqv), state.theta, state.qv, state.theta.dtype
        )).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back(rqv)).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * _back(rqc)).astype(_output_dtype(state, "qc")),
        qi=(state.qi + dt_f * _back(rqi)).astype(_output_dtype(state, "qi")),
        rainc_acc=(
            jnp.asarray(state.rainc_acc, jnp.float64) + raincv.reshape(ny, nx)
        ).astype(_output_dtype(state, "rainc_acc")),
    )
    # WRF RAINCV of this call (module_surface_driver.F:1568/:3283 -> Noah-MP MP_RAINC).
    return (next_state, raincv.reshape(ny, nx)) if return_raincv else next_state


def _kpbl_bulk_richardson(thv_c: jax.Array, z_mass_c: jax.Array) -> jax.Array:
    """Diagnose the PBL-top mass level (1-based, GF convention) per column.

    GF's driver (``module_cu_gf_wrfdrv.F``) receives ``KPBL`` from the active PBL
    scheme. The operational scan does not thread a PBL-height leaf, so the cumulus
    slot reconstructs it the way every WRF PBL driver does: the lowest mass level
    whose bulk Richardson number relative to the surface level exceeds a critical
    value ``Ricr=0.25`` (YSU/ACM2 ``BRCR``). ``thv`` is virtual potential
    temperature; ``z_mass`` mass-level heights above ground. Returns a 1-based
    level index in ``[2, nz]`` (GF arrays are 1-based, surface = level 1), so the
    PDF/inversion logic sees a physically-located PBL top, not a fabricated const.

    Inputs are 0-based ``(ncol, nz)``; the +1 shift to GF's 1-based indexing is
    applied by the caller (which prepends the dummy level-0).
    """

    ricr = 0.25
    ncol, nz = thv_c.shape
    thv_sfc = thv_c[:, :1]  # (ncol, 1)
    dz = jnp.maximum(z_mass_c - z_mass_c[:, :1], 1.0)  # height above surface
    # Bulk Richardson vs surface (no shear term -> thermal-only BRN, WRF's
    # convective limit; the PBL-top estimate GF needs is robust to this).
    rib = (9.81 / jnp.maximum(thv_sfc, 1.0)) * (thv_c - thv_sfc) * dz / jnp.maximum(dz, 1.0)
    above = (rib > ricr) & (jnp.arange(nz)[None, :] >= 1)  # exclude surface level
    any_above = jnp.any(above, axis=1)
    first0 = jnp.argmax(above.astype(jnp.int32), axis=1)  # 0-based first crossing
    # 0-based -> GF 1-based (+1); default to level 2 when no crossing (shallow PBL).
    kpbl1 = jnp.where(any_above, first0 + 1, 1) + 1
    kpbl1 = jnp.clip(kpbl1, 2, nz).astype(jnp.int32)
    return kpbl1


def gf_adapter(state: State, dt: float, grid=None, *, ishallow_g3: int = 0,
               ichoice: int = 0, return_raincv: bool = False, mp_physics: int | None = None,
               return_rates: bool = False,
               rthften=None, rqvften=None, rthraten=None, rthblten=None, rqvblten=None,
               forcing_entry_state: State | None = None, periodic_x: bool = False,
               periodic_y: bool = False, kpbl=None):
    """cu=3 Grell-Freitas scale-aware cumulus ``State -> State`` scan adapter.

    ``ishallow_g3`` is WRF's namelist ``ishallow`` (module_first_rk_step_part1.F:1410 ->
    module_cumulus_driver.F:1305), Registry default 0: no CUP_gf_sh call and ``cutens=0``
    (module_cu_gf_wrfdrv.F:331, :504).  The namelist honour check refuses ishallow != 0.

    WRF GFDRV caller inputs (t-cfix-cu F18): deep forced sounding with the held
    RTHFTEN/RQVFTEN, RTHRATEN and this step's RTHBLTEN/RQVBLTEN (gf_wrfdrv:412-417;
    ``forcing_entry_state`` turns an Euler-applied PBL increment into its rate);
    ``PSUR = p8w(1)*.01`` with P8W = p_hyd_w (:385); ``omeg = -g*rho*w`` on the
    STAGGERED w (:471); KPBL = grid%kpbl (first_rk_step_part1.F:1397): the KPBL the active PBL
    published (``kpbl``, BL_GFS KPBL2D), else its PBLH placed with the MYNN GET_PBLH level rule
    (exact for MYNN only); GF's outputs are kept only on ids+4..ide-5 / jds+4..jde-5 unless
    periodic (gf_wrfdrv:264-277) -- the batched kernel still evaluates every column and the band
    is masked afterwards (no boundary-column work is saved).

    Grell-Freitas is a per-column kernel (``physics._gf_jax.gfdrv_column``, the
    GPU-batched jit/vmap port of WRF ``GFDRV -> cup_gf (deep) + cup_gf_sh
    (shallow)``); this adapter ``jax.vmap``s it over the ``(ny*nx)`` grid columns
    via ``gfdrv_batched`` -- the whole deep+shallow column physics (16-member
    closure ensemble + beta-PDF gamma + scale-aware ``sig``) runs inside ONE
    vmapped jit, no host transfer inside the column loop. Like modified-Tiedtke,
    GF carries NO persistent cumulus state, so this is a plain ``State -> State``
    adapter (no carry, threaded via ``CU_STATELESS_SCAN_ADAPTERS``).

    Tendencies are applied ``state += dt*tend``: ``RTHCUTEN`` is the THETA tendency
    (K s^-1; the kernel already divides the temperature tendency by Exner), and
    ``RQVCUTEN/RQCCUTEN/RQICUTEN`` are mixing-ratio tendencies (kg kg^-1 s^-1).
    ``RAINCV`` is the per-step convective precip (mm; the kernel returns
    ``pratec*dt``) and accumulates into ``rainc_acc`` (exactly the KF/Tiedtke
    accumulation convention). GF momentum tendencies are diagnostic in the WRF
    GFDRV path (not added to U/V there), so this adapter writes none -- matching
    the kernel's ``gfdrv_column`` output set (no RUCUTEN/RVCUTEN).

    The GF kernel uses 1-based length-(nz+1) column arrays (index 0 unused,
    level 1 = surface) -- the operational State is bottom-up ``(nz, ny, nx)``
    (index 0 = surface), so each column is mapped by prepending a dummy level-0.

    GF-specific inputs not in the frozen B2 contract are assembled here, pure
    ``jnp``, no host transfer:
      * ``HFX`` surface sensible-heat flux (W m^-2) = ``rho_sfc*cp*theta_flux``;
        ``QFX`` surface moisture flux (kg m^-2 s^-1) = ``rho_sfc*qv_flux`` (the
        kinematic B2 ``theta_flux``/``qv_flux`` handles -> GF flux units).
      * ``HT`` terrain height (m) = surface geopotential ``ph[0]/g``.
      * ``DX`` grid spacing (m) for the scale-aware factor (from the grid, like KF;
        defaults to 3 km when no grid is supplied).
      * ``KPBL`` PBL-top mass level, bulk-Richardson-diagnosed per column
        (``_kpbl_bulk_richardson``) -- the value the active PBL scheme would hand
        GF in WRF (no PBL-height leaf is threaded in this scan).
      * ``RTHBLTEN``/``RQVBLTEN`` PBL theta/qv forcing tendencies: zero in the
        per-slot scan (no separate PBL forcing tracked into the cumulus slot here;
        the PBL slot already applied its tendency to State the same step -- WRF
        folds this forcing via the "forced sounding", documented carry-over,
        IDENTICAL to the Tiedtke adapter's zero QVFTEN/QVPBLTEN treatment). With
        zero forcing the forced sounding collapses to the current sounding and GF
        triggers on the actual column state.
    """

    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    nz, ny, nx = state.theta.shape
    # cumulus_driver.F:1292-1298 GFDRV(T=t_phy, P=p_hyd, PI=pi_phy, RHO=rho, DZ8W, HT)
    # from phy_prep (first_rk_step_part1.F:1379-1400); g=9.81 heights.
    prep = _wrf_cumulus_prep(state, grid)
    rho, T, pii, dz_full, p_gf = (prep[k] for k in ("rho", "T", "pii", "dz", "p"))
    interface_z = prep["z_at_w"]  # (nz+1, ny, nx)
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])  # (nz, ny, nx)
    ht_2d = interface_z[0]  # surface geopotential height (m), (ny, nx)
    w_face = jnp.asarray(state.w, jnp.float64)[:nz]  # gf_wrfdrv:471 omega on staggered w(i,k,j)
    # gf_wrfdrv:385 PSUR = p8w(1)*.01 with P8W = p_hyd_w (phy_prep); None without metrics.
    psur_b = None if prep["psfc"] is None else jnp.asarray(prep["psfc"], jnp.float64).reshape(ny * nx)

    # Surface fluxes in GF units (W m^-2 / kg m^-2 s^-1) from kinematic handles.
    rhosfc = jnp.asarray(state.rhosfc, jnp.float64)
    theta_flux = jnp.asarray(state.theta_flux, jnp.float64)  # K m s^-1
    qv_flux = jnp.asarray(state.qv_flux, jnp.float64)        # kg kg^-1 m s^-1
    hfx_2d = rhosfc * CP_DRY * theta_flux                    # W m^-2
    qfx_2d = rhosfc * qv_flux                                # kg m^-2 s^-1
    xland_2d = jnp.asarray(state.xland, jnp.float64)

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)

    # grid%kpbl (first_rk_step_part1.F:1397): the 1-based KPBL the active PBL published this
    # step (``kpbl``, e.g. BL_GFS KPBL2D); otherwise its PBLH placed on the AGL interfaces
    # with the MYNN GET_PBLH rule (exact for MYNN).
    from gpuwrf.coupling.cuplus import mynn_kpbl
    kpbl_c = (mynn_kpbl(interface_z - interface_z[0][None], state.pblh, mass_levels=nz).reshape(ny * nx)
              if kpbl is None else jnp.asarray(kpbl, jnp.int32).reshape(ny * nx))

    # 0-based (ncol, nz) -> GF 1-based (ncol, nz+1): prepend a dummy level-0.
    def _cols1(field3d):
        c = _cols(field3d)  # (ncol, nz)
        return jnp.concatenate([jnp.zeros((ny * nx, 1), jnp.float64), c], axis=1)

    T_c = _cols1(T)
    qv_c = _cols1(jnp.maximum(state.qv, 0.0))
    p_c = _cols1(p_gf)
    pii_c = _cols1(pii)
    dz_c = _cols1(dz_full)
    rho_c = _cols1(rho)
    u_c = _cols1(_u_mass(state))
    v_c = _cols1(_v_mass(state))
    w_c = _cols1(w_face)
    dt_f = float(dt)
    kx = nz
    if forcing_entry_state is not None:
        # PBL applied as a state increment before this slot: its WRF RTHBLTEN/RQVBLTEN.
        rthblten = (_dry_theta_view(state).astype(jnp.float64)
                    - _dry_theta_view(forcing_entry_state).astype(jnp.float64)) / dt_f
        rqvblten = (jnp.asarray(state.qv, jnp.float64) - jnp.asarray(forcing_entry_state.qv, jnp.float64)) / dt_f
    forcing = lambda f: None if f is None else _cols1(jnp.asarray(f, jnp.float64))  # noqa: E731
    zero_c = jnp.zeros((ny * nx, nz + 1), jnp.float64)
    rthbl_c = zero_c if rthblten is None else forcing(rthblten)
    rqvbl_c = zero_c if rqvblten is None else forcing(rqvblten)
    wrf_forcing = {}
    if rthften is not None or rqvften is not None or rthraten is not None or psur_b is not None:
        wrf_forcing = dict(rthften1=forcing(rthften), rqvften1=forcing(rqvften),
                           rthraten1=forcing(rthraten), psur_pa=psur_b)

    dt_b = jnp.full((ny * nx,), dt_f, jnp.float64)
    dx_b = jnp.full((ny * nx,), dx, jnp.float64)
    hfx_b = hfx_2d.reshape(ny * nx)
    qfx_b = qfx_2d.reshape(ny * nx)
    xland_b = xland_2d.reshape(ny * nx)
    ht_b = ht_2d.reshape(ny * nx)

    out = gfdrv_batched(
        T_c, qv_c, p_c, pii_c, dz_c, rho_c, u_c, v_c, w_c, rthbl_c, rqvbl_c,
        kx, dt_b, dx_b, hfx_b, qfx_b, kpbl_c, xland_b, ht_b,
        ishallow_g3=int(ishallow_g3), ichoice=int(ichoice), **wrf_forcing,
    )

    def _back(field2d_1based):  # (ncol, nz+1) -> (nz, ny, nx); drop GF level-0
        c0 = field2d_1based[:, 1:]  # (ncol, nz)
        return jnp.moveaxis(c0.reshape(ny, nx, nz), -1, 0)

    rth = _back(out['RTHCUTEN'])   # K s^-1 (theta tendency)
    rqv = _back(out['RQVCUTEN'])   # kg kg^-1 s^-1
    rqc = _back(out['RQCCUTEN'])
    rqi = _back(out['RQICUTEN'])
    # GFDRV always splits; phy_cu_ten (module_physics_addtendc.F:1382-1395) drops absent species.
    f_qc, f_qi = _cu_condensate_flags(mp_physics)
    rqc = rqc if f_qc else jnp.zeros_like(rqc)
    rqi = rqi if f_qi else jnp.zeros_like(rqi)
    raincv = jnp.asarray(out['RAINCV'], jnp.float64).reshape(ny, nx)  # mm/step
    # gf_wrfdrv:264-277: columns within 4 points of a non-periodic boundary are never computed.
    owned = jnp.ones((ny, nx), bool)
    if not periodic_x:
        owned = owned & ((jnp.arange(nx) >= 4) & (jnp.arange(nx) <= nx - 5))[None, :]
    if not periodic_y:
        owned = owned & ((jnp.arange(ny) >= 4) & (jnp.arange(ny) <= ny - 5))[:, None]
    rth, rqv, rqc, rqi = (jnp.where(owned[None], f, jnp.zeros_like(f)) for f in (rth, rqv, rqc, rqi))
    raincv = jnp.where(owned, raincv, jnp.zeros_like(raincv))
    if return_rates:  # F9: GFSCHEME phy_cu_ten set (addtendc.F:1369-1412), uncoupled
        return (dict(theta=rth, qv=rqv, qc=rqc if f_qc else None, qi=rqi if f_qi else None), raincv)

    next_state = state.replace(
        # Dry RTHCUTEN -> theta_m: conv_t_tendf_to_moist (module_first_rk_step_part2.F:1005-1016).
        theta=(state.theta + dt_f * _theta_m_tendency_from_dry(
            rth, rqv, state.theta, state.qv, state.theta.dtype)).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * rqv).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * rqc).astype(_output_dtype(state, "qc")),
        qi=(state.qi + dt_f * rqi).astype(_output_dtype(state, "qi")),
        rainc_acc=(
            jnp.asarray(state.rainc_acc, jnp.float64) + raincv
        ).astype(_output_dtype(state, "rainc_acc")),
    )
    # WRF RAINCV of this call (module_surface_driver.F:1568/:3283 -> Noah-MP MP_RAINC).
    return (next_state, raincv) if return_raincv else next_state

def grell_cumulus_adapter(state: State, dt: float, grid=None, *, cu_physics: int,
                          forcing_entry_state: State | None = None, rthblten=None,
                          rqvblten=None, rthften=None, rqvften=None, rthraten=None,
                          gsw=None, periodic_x: bool = False,
                          periodic_y: bool = False, ishallow: int = 0, cugd_avedx: int = 1,
                          ichoice: int = 0, return_raincv: bool = False, mp_physics: int | None = None,
                          return_rates: bool = False):
    """cu=5 Grell-3D ensemble / cu=93 Grell-Devenyi ``State -> State`` scan adapter.

    ``return_rates=True`` (WRF RK coupling, F9) returns ``(rates, PRATEC*dt)`` instead:
    the uncoupled dry-theta RTHCUTEN / RQVCUTEN / RQCCUTEN / RQICUTEN that
    phy_cu_ten adds for GD/G3 (module_physics_addtendc.F:1369-1412) -- the caller
    couples them into the RK sources (coupling.cu_rk) and accumulates RAINC.

    Calls the line-faithful JAX port of WRF ``G3DRV`` + ``conv_grell_spread3d``
    (cu=5) or ``GRELLDRV`` (cu=93) in ``physics._grell_cup_jax`` on the whole
    ``(ny, nx)`` tile (G3 is not column-local: 3x3 neighbour omega ensemble and
    horizontal spreading/smoothing; both schemes act only on columns >= 4 points
    from a non-periodic boundary, exactly like WRF). No persistent carry.

    WRF cumulus_driver inputs (module_first_rk_step_part1.F:1379-1400), in the
    State dtype (WRF REAL under the release carry): T = t_phy (dry theta x
    pi_phy), RHO = phy_prep total density, W = staggered w_2, U/V = mass-point
    winds, Q = qv, P = p_hyd and P8W = p_hyd_w (PSUR = P8W(1)).  Closure forcing
    (module_cumulus_driver.F:866-883; GRELLDRV module_cu_gd.F:221-222):
    ``rthften``/``rqvften`` = WRF's held RTHFTEN/RQVFTEN (previous step's
    last-RK-stage advection, decoupled and dry; coupling.cu_forcing),
    ``rthraten`` = the held radiation rate, ``rthblten``/``rqvblten`` = this
    step's PBL rates; for a PBL still applied as a state increment the caller
    passes ``forcing_entry_state`` instead and the PBL rate is that increment/dt.
    Omitted terms are zero (WRF itimestep 1).  ``gsw`` (net surface shortwave) only feeds G3's
    night cap test (``gsw < 1``).  Tendencies: dry-theta RTHCUTEN/RQVCUTEN are
    converted to the theta_m tendency at the old state (conv_t_tendf_to_moist),
    RQCCUTEN/RQICUTEN update qc/qi, RAINC += PRATEC*dt (physics_addtendc).
    """

    from gpuwrf.physics import _grell_cup_jax as grell

    real = jnp.asarray(state.theta).dtype
    nz, ny, nx = state.theta.shape
    dt_f = float(dt)
    dx = float(grid.projection.dx_m) if grid is not None else 3000.0
    metrics = getattr(grid, "metrics", None)
    theta_dry = jnp.asarray(_dry_theta_view(state), real)
    pi_phy = (jnp.asarray(state.p, real) / jnp.asarray(1.e5, real)) ** jnp.asarray(2. / 7., real)
    t_phy = theta_dry * pi_phy
    if metrics is not None:
        rho = _wrf_phy_prep_rho_from_state(state, metrics, output_dtype=real)
    else:
        rho = jnp.asarray(_rho_from_state(state), real) * (1.0 + jnp.asarray(state.qv, real))
    rho = jnp.asarray(rho, real)
    qv = jnp.asarray(state.qv, real)
    if metrics is not None:
        # first_rk_step_part1.F:1380/1400: P=grid%p_hyd, P8W=grid%p_hyd_w (PSUR=p8w(1), g3.F:342, gd.F:233).
        p_hyd, p8w, _psfc = _wrf_hydrostatic_pressure_profiles_from_state(state, metrics, output_dtype=real)
        p8w = jnp.asarray(p8w, real)
    else:
        # No metrics (isolated tests): total mass-level pressure and a hydrostatic
        # extrapolation of the lowest level over half the first layer for PSUR.
        p_mass = jnp.asarray(state.p, real)
        ph = jnp.asarray(state.ph, real)
        g_ = jnp.asarray(9.81, real)
        dz_half = (0.5 * (ph[1] + ph[0]) - ph[0]) / g_
        tv0 = t_phy[0] * (1.0 + 0.608 * qv[0])
        psfc = p_mass[0] * jnp.exp(g_ * dz_half / (jnp.asarray(287.0, real) * tv0))
        p8w = jnp.concatenate([psfc[None], 0.5 * (p_mass[1:] + p_mass[:-1]), p_mass[-1:]], axis=0)
        p_hyd = p_mass
    zero3 = jnp.zeros_like(theta_dry)
    if forcing_entry_state is not None:
        # PBL applied as a state increment before this slot: its WRF RTHBLTEN/RQVBLTEN.
        if rthblten is not None or rqvblten is not None:
            raise ValueError("grell_cumulus_adapter: pass PBL rates OR forcing_entry_state, not both")
        rthbl = (theta_dry - jnp.asarray(_dry_theta_view(forcing_entry_state), real)) / dt_f
        rqvbl = (qv - jnp.asarray(forcing_entry_state.qv, real)) / dt_f
    else:
        rthbl = zero3 if rthblten is None else jnp.asarray(rthblten, real)
        rqvbl = zero3 if rqvblten is None else jnp.asarray(rqvblten, real)
    thften = zero3 if rthften is None else jnp.asarray(rthften, real)
    qvften = zero3 if rqvften is None else jnp.asarray(rqvften, real)
    rthra = zero3 if rthraten is None else jnp.asarray(rthraten, real)
    ht = jnp.asarray(state.ph, real)[0] / jnp.asarray(9.81, real)
    xland = jnp.asarray(state.xland, real)
    gsw2 = jnp.zeros((ny, nx), real) if gsw is None else jnp.asarray(gsw, real)
    common = dict(u=jnp.asarray(_u_mass(state), real), v=jnp.asarray(_v_mass(state), real),
                  w=jnp.asarray(state.w, real), t=t_phy, q=qv, p=p_hyd, pi=pi_phy, rho=rho,
                  p8w=p8w, rthften=thften, rqvften=qvften, rthraten=rthra, rthblten=rthbl,
                  rqvblten=rqvbl, ht=ht, xland=xland, gsw=gsw2, dt=dt_f,
                  periodic_x=bool(periodic_x), periodic_y=bool(periodic_y))
    if int(cu_physics) == 5:
        z_mass = 0.5 * (jnp.asarray(state.ph, real)[1:] + jnp.asarray(state.ph, real)[:-1]) / 9.81
        thv = theta_dry * (1.0 + 0.608 * jnp.maximum(qv, 0.0))
        cols = lambda f: jnp.moveaxis(f, 0, -1).reshape(ny * nx, nz)
        kpbl = _kpbl_bulk_richardson(cols(thv), cols(z_mass)).reshape(ny, nx)
        out = grell.g3drv_tile(**common, kpbl=kpbl, dx=dx, cugd_avedx=int(cugd_avedx),
                               ishallow=int(ishallow), ichoice=int(ichoice))
    elif int(cu_physics) == 93:
        out = grell.grelldrv_tile(**common)
    else:
        raise ValueError(f"grell_cumulus_adapter: cu_physics={cu_physics} is not 5 or 93")
    rth, rqv = out["RTHCUTEN"], out["RQVCUTEN"]
    rqc, rqi = out["RQCCUTEN"], out["RQICUTEN"]
    f_qc, f_qi = _cu_condensate_flags(mp_physics)
    if not f_qi:  # .NOT.F_QI: all detrained condensate stays in RQCCUTEN (g3.F:541-563, gd.F:340-360)
        rqc, rqi = rqc + rqi, jnp.zeros_like(rqi)
    if not f_qc:  # P_QC absent: phy_cu_ten drops RQCCUTEN
        rqc = jnp.zeros_like(rqc)
    if return_rates:
        return (dict(theta=rth, qv=rqv, qc=rqc if f_qc else None, qi=rqi if f_qi else None),
                dt_f * out["PRATEC"])
    theta_dtype = jnp.asarray(state.theta).dtype
    theta_next = (
        jnp.asarray(state.theta, theta_dtype)
        + dt_f * _theta_m_tendency_from_dry(rth, rqv, state.theta, state.qv, theta_dtype)
    ).astype(_output_dtype(state, "theta"))
    next_state = state.replace(
        theta=theta_next,
        qv=(state.qv + dt_f * rqv).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * rqc).astype(_output_dtype(state, "qc")),
        qi=(state.qi + dt_f * rqi).astype(_output_dtype(state, "qi")),
        rainc_acc=(
            jnp.asarray(state.rainc_acc) + dt_f * out["PRATEC"]
        ).astype(_output_dtype(state, "rainc_acc")),
    )
    # WRF G3/GD RAINCV = PRATEC*DT of this call (-> Noah-MP MP_RAINC).
    return (next_state, dt_f * out["PRATEC"]) if return_raincv else next_state


def g3_adapter(state: State, dt: float, grid=None, **kwargs) -> State:
    """cu=5 Grell-3D ensemble (WRF defaults ishallow=0, cugd_avedx=1, clos_choice=0)."""

    return grell_cumulus_adapter(state, dt, grid, cu_physics=5, **kwargs)


def gd_adapter(state: State, dt: float, grid=None, **kwargs) -> State:
    """cu=93 Grell-Devenyi ensemble."""

    return grell_cumulus_adapter(state, dt, grid, cu_physics=93, **kwargs)


def bmj_adapter(state: State, dt: float, cldefi, *, grid=None, return_raincv: bool = False,
                return_rates: bool = False):
    """cu=2 Betts-Miller-Janjic cumulus scan adapter.

    ``return_rates=True`` (F9): ``((rates, RAINCV), cldefi)`` with BMJSCHEME's
    phy_cu_ten set RTHCUTEN/RQVCUTEN (module_physics_addtendc.F:1198-1210) uncoupled.

    BMJ writes WRF ``RTHCUTEN``/``RQVCUTEN`` tendencies and deep-convective
    ``RAINCV``.  Its persistent WRF state member is ``CLDEFI`` (cloud
    efficiency), carried as a cumulus sibling tree rather than a dycore leaf.
    """
    _initialize_bmj()

    nz, ny, nx = state.theta.shape
    # cumulus_driver.F:994-1004 BMJDRV(T=t_phy, RHO=rho, DZ8W, PINT=p_hyd_w, PMID=p_hyd,
    # PI=pi_phy) from phy_prep; bmj.F:193 PSFC=PINT(LOWLYR).
    prep = _wrf_cumulus_prep(state, grid)
    rho, T, dz_full, p_bmj = (prep[k] for k in ("rho", "T", "dz", "p"))
    pii = _exner_columns(_mp_in(state.p, ny, nx, nz))

    def _cols(field3d):
        return jnp.moveaxis(field3d, 0, -1).reshape(ny * nx, nz)

    T_c = _cols(T)
    qv_c = _cols(state.qv)
    p_c = _cols(p_bmj)
    dz_c = _cols(dz_full)
    rho_c = _cols(rho)
    psfc_c = (None if prep["psfc"] is None
              else jnp.asarray(prep["psfc"], jnp.float64).reshape(ny * nx))
    cldefi_c = jnp.asarray(cldefi, jnp.float64).reshape(ny * nx)
    xland_c = jnp.asarray(state.xland, jnp.float64).reshape(ny * nx)

    def _one(T0, QV0, P0, DZQ, RHOE, PII, XLAND, CLDEFI, PSFC=None):
        res = step_bmj_column(
            T0,
            QV0,
            P0,
            DZQ,
            RHOE,
            PII,
            float(dt),
            stepcu=1,
            xland=XLAND,
            cldefi=CLDEFI,
            psfc=PSFC,
        )
        st = res.tendency.state_tendencies
        cc = res.carry.cumulus
        return (
            st["theta"],
            st["qv"],
            res.tendency.accumulator_increments["rainc_acc"],
            cc["cldefi"],
        )

    rth, rqv, raincv, cldefi_next_c = jax.vmap(_one)(
        T_c, qv_c, p_c, dz_c, rho_c, pii, xland_c, cldefi_c,
        *(() if psfc_c is None else (psfc_c,))
    )

    def _back(field2d):
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    if return_rates:
        return (dict(theta=_back(rth), qv=_back(rqv)), raincv.reshape(ny, nx)), cldefi_next_c.reshape(ny, nx)
    dt_f = float(dt)
    next_state = state.replace(
        # Dry RTHCUTEN -> theta_m: conv_t_tendf_to_moist (module_first_rk_step_part2.F:1005-1016).
        theta=(state.theta + dt_f * _theta_m_tendency_from_dry(
            _back(rth), _back(rqv), state.theta, state.qv, state.theta.dtype
        )).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back(rqv)).astype(_output_dtype(state, "qv")),
        rainc_acc=(
            jnp.asarray(state.rainc_acc, jnp.float64) + raincv.reshape(ny, nx)
        ).astype(_output_dtype(state, "rainc_acc")),
    )
    if return_raincv:  # WRF RAINCV of this call (-> Noah-MP MP_RAINC)
        return next_state, cldefi_next_c.reshape(ny, nx), raincv.reshape(ny, nx)
    return next_state, cldefi_next_c.reshape(ny, nx)

def initial_bmj_carry(state: State):
    """Seed BMJ ``CLDEFI`` carry with BMJINIT's default value."""
    _initialize_bmj()

    return initial_bmj_cldefi(state.theta.shape[1:])


# --- PBL adapters (bl_pbl_physics) --------------------------------------------
#
# YSU (1) / ACM2 (7) are the v0.6.0 ``jax.lax.scan``-traceable / vmap-batched PBL
# kernels (``physics.pbl_ysu.ysu_columns`` / ``physics.pbl_acm2.acm2_columns``).
# They occupy the same operational PBL slot as the v0.2.0 ``mynn_adapter`` and
# consume the surface-layer forcing the surface-layer slot wrote earlier in the
# WRF call chain. Both kernels need MORE surface inputs than the State's frozen B2
# kinematic-flux contract carries (YSU: hfx/qfx/br/psim/psih/u10/v10/znt; ACM2:
# hfx/qfx/pblh/wspd) -- the FROZEN B2 handles are kinematic fluxes only. The
# WRF-faithful surface-forcing assembler is the SAME revised-MM5 surface layer the
# operational scan already runs (``surface_layer_with_diagnostics``): it returns
# HFX/LH (W m^-2), psim/psih, the bulk Richardson number BR, U10/V10, ZNT and 1/L,
# so the PBL adapter re-derives the full per-cell forcing here -- no host transfer,
# fully traceable, exactly the inputs the savepoint-parity kernel was validated on.
#
# Momentum coupling mirrors the MYNN adapter (``_state_from_mynn_output``): form the
# A-grid PBL momentum INCREMENT on mass points and add it A2C-averaged onto the
# dynamics' ORIGINAL C-grid u/v faces (WRF ``add_a2c_u``/``add_a2c_v``), so the
# large-scale C-grid winds + the surface-w BC are preserved exactly. theta/qv live
# on the mass grid (read-back is identity) so they take the increment directly.

def _pbl_surface_forcing(state: State, grid, *, surface_forcing=None, use_resistances=False,
                         ysu_surface=None):
    """Re-derive the YSU/ACM2 surface forcing via the revised-MM5 surface layer.

    Returns the per-cell ``(ny, nx)`` diagnostics the PBL kernels consume, plus the
    column-major ``(ncol, nz)`` profile views (mass-point winds, lowest->top) and
    the interface pressure column. Pure-``jnp``, no host transfer.
    """

    if ysu_surface is not None:
        surface_forcing, use_resistances = ysu_surface, True
    if surface_forcing is not None and grid is not None:
        from gpuwrf.coupling.urban_surface_layer import UrbanSurfaceLayer
        from gpuwrf.coupling.urban_pbl_hook import urban_pbl_forcing
        sf = surface_forcing
        surface = UrbanSurfaceLayer(sf['br'], sf['fm'] if use_resistances else sf['psim'],
            sf['fh'] if use_resistances else sf['psih'], sf['wspd'], sf['u10'], sf['v10'])
        # Existing WRF phy_prep assembly: dry theta/T_PHY, hydrostatic P/P8W,
        # nonhydrostatic Exner, true PSFC, and the LSM's actual HFX/QFX/UST.
        with jax.enable_x64(False):
            f = urban_pbl_forcing(state, grid, surface)
            flat = lambda x: jnp.asarray(x, jnp.float32).reshape(f['ncol'])
            cols = lambda x: jnp.moveaxis(jnp.asarray(x,jnp.float32),0,-1).reshape(f['ncol'],f['nz'])
            z = jnp.asarray(state.ph,jnp.float32)/jnp.float32(9.81)
            f.update(tsk=flat(sf.get('tsk',state.t_skin)),
                gz1oz0=flat(sf['gz1oz0']) if 'gz1oz0' in sf else jnp.log(.5*f['dz_cols'][:,0]/f['znt']),
                z_cols=cols(.5*(z[:-1]+z[1:])),theta_m_cols=cols(state.theta),
                pblh=flat(state.pblh),rhosfc=jnp.asarray(state.rhosfc,jnp.float32),
                cpm=jnp.float32(1004.5)*(1+jnp.float32(.8)*jnp.asarray(state.qv[0],jnp.float32)))
        return f
    ysu_surface = surface_forcing
    nz, ny, nx = state.theta.shape
    ncol = ny * nx
    # YSU reads the SELECTED surface solve's FM/FHH, not the raw psi corrections
    # (module_pbl_driver.F:1227). Reuse that solve; a second MYNN surface solve
    # both changes the forcing and supplies PSI=0/0 at neutral stability.
    diag = (surface_layer_with_diagnostics(_surface_column_view(state))
            if ysu_surface is None else None)

    def _flat2d(field2d):  # (ny, nx) -> (ncol,)
        return jnp.asarray(field2d, jnp.float64).reshape(ncol)

    # WRF phy_prep staging (pbl_driver args, first_rk_step_part1.F:1130-1139):
    # dry TH_PHY/T_PHY, P_PHY=p_hyd, P8W=p_hyd_w, PSFC=p_hyd_w(1), total RHO,
    # dz8w/z with g=9.81; non-hydrostatic PI_PHY.
    prep = wrf_phy_prep(state, getattr(grid, "metrics", None))
    rho = _rho_from_state(state) if prep["rho"] is None else prep["rho"]
    T = prep["t_phy"]
    pii = _exner_columns(state.p)
    interface_z = prep["z_at_w"]  # (nz+1, ny, nx)
    dz = jnp.maximum(prep["dz8w"], 1.0)  # (nz, ny, nx)
    if prep["p_hyd"] is not None:
        p, p_int = prep["p_hyd"], prep["p_hyd_w"]
    else:
        # No metrics (isolated tests): mean-of-levels faces, zero-gradient edges.
        p = state.p.astype(jnp.float64)
        p_int_interior = 0.5 * (p[:-1] + p[1:])  # (nz-1, ny, nx)
        p_int = jnp.concatenate([p[:1], p_int_interior, p[-1:]], axis=0)  # (nz+1, ny, nx)

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(ncol, nz)

    rhosfc = jnp.asarray(state.rhosfc, jnp.float64)  # (ny, nx)
    cpm = CP_DRY * (1.0 + 0.84 * jnp.maximum(state.qv[0], 0.0))  # moist cp at k0
    hfx = jnp.asarray(diag.hfx if diag is not None else state.hfx, jnp.float64)
    lh = jnp.asarray(diag.lh if diag is not None else 2.5e6 * state.qfx, jnp.float64)
    XLV = 2.5e6
    qfx = lh / XLV  # kg m^-2 s^-1 (LH = XLV*QFX)
    u_mass = _u_mass(state)
    v_mass = _v_mass(state)
    z_mass = 0.5 * (interface_z[:-1] + interface_z[1:])  # (nz, ny, nx) full-level height
    tsk = jnp.asarray(state.t_skin, jnp.float64)  # (ny, nx)
    return {
        "ncol": ncol, "ny": ny, "nx": nx, "nz": nz,
        "u_mass": u_mass, "v_mass": v_mass,
        "u_cols": _cols(u_mass), "v_cols": _cols(v_mass),
        "theta_cols": _cols(prep["th_phy"]), "T_cols": _cols(T),
        # theta_m at PBL entry: _pbl_state_rates applies conv_t_tendf_to_moist.
        "theta_m_cols": _cols(state.theta),
        "qv_cols": _cols(jnp.maximum(state.qv, 0.0)),
        "p_cols": _cols(p), "pii_cols": _cols(pii), "rho_cols": _cols(rho),
        "dz_cols": _cols(dz), "z_cols": _cols(z_mass),
        **({} if prep["psfc"] is None else {"psfc": _flat2d(prep["psfc"])}),
        "tsk": _flat2d(tsk),
        "p_int_cols": jnp.moveaxis(p_int, 0, -1).reshape(ncol, nz + 1),
        # surface forcing (ncol,)
        "hfx": _flat2d(hfx), "qfx": _flat2d(qfx),
        "psim": _flat2d(diag.psim if diag is not None else ysu_surface["fm" if use_resistances else "psim"]),
        "psih": _flat2d(diag.psih if diag is not None else ysu_surface["fh" if use_resistances else "psih"]),
        "br": _flat2d(diag.br if diag is not None else ysu_surface["br"]),
        "ust": jnp.maximum(_flat2d(state.ustar), 1.0e-3),
        "znt": jnp.maximum(_flat2d(jnp.maximum(state.roughness_m, 1.0e-4)), 1.0e-7),
        "wspd": (jnp.maximum(jnp.sqrt(_flat2d(u_mass[0]) ** 2 + _flat2d(v_mass[0]) ** 2), 0.1)
                 if diag is not None else _flat2d(ysu_surface["wspd"])),
        "u10": _flat2d(diag.u10 if diag is not None else ysu_surface["u10"]),
        "v10": _flat2d(diag.v10 if diag is not None else ysu_surface["v10"]),
        "xland": _flat2d(state.xland),
        # ACM2 needs an INCOMING PBLH guess (pblh_initial) before it diagnoses its
        # own height. The revised-MM5 surface layer does not export a PBLH leaf and
        # State carries none, so seed with the surface-layer's own pblh assumption
        # (1000 m, sf_sfclayrev default) -- ACM2 overwrites it with its diagnosis.
        "pblh": jnp.full((ncol,), 1000.0, dtype=jnp.float64),
        "rhosfc": rhosfc, "cpm": cpm,
    }


def _apply_pbl_increment(state: State, dt: float, tend: dict, *, ny: int, nx: int, nz: int) -> State:
    """Apply batched PBL tendencies (``(ncol, nz)`` rates) WRF-faithfully.

    u/v: A2C-averaged mass-point increment onto the ORIGINAL C-grid faces (WRF
    ``add_a2c_u``/``add_a2c_v``); theta/qv: direct ``state += dt*tend`` on the mass
    grid. Writes at the live State dtype (fp32-defeat fix).
    """

    def _back3d(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(ny, nx, nz), -1, 0)

    dt_f = float(dt)
    du_mass = dt_f * _back3d(tend["u"])
    dv_mass = dt_f * _back3d(tend["v"])
    u_new = _add_a2c_u_increment(state.u, du_mass).astype(_output_dtype(state, "u"))
    v_new = _add_a2c_v_increment(state.v, dv_mass).astype(_output_dtype(state, "v"))
    return state.replace(
        u=u_new,
        v=v_new,
        theta=(state.theta + dt_f * _back3d(tend["theta"])).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back3d(tend["qv"])).astype(_output_dtype(state, "qv")),
    )


def _pbl_state_rates(rates, forcing):
    """Convert dry WRF theta rates for the State-only adapter; raw RK rates stay dry."""
    if 'theta_m_cols' not in forcing:
        return rates
    from gpuwrf.coupling.physics_couplers import _theta_m_tendency_from_dry
    return dict(rates, theta=_theta_m_tendency_from_dry(rates['theta'],rates['qv'],
        forcing['theta_m_cols'],forcing['qv_cols'],forcing['theta_m_cols'].dtype))


def _pbl_diagnostic_state(state, rates, selected):
    if selected is not None and 'pblh' in rates:
        state = state.replace(pblh=rates['pblh'].reshape(state.pblh.shape).astype(state.pblh.dtype))
    return state


def _apply_ysu_cloud_rates(result: State, state: State, dt: float, rates, *, ny: int, nx: int, nz: int) -> State:
    """RQCBLTEN/RQIBLTEN (module_bl_ysu.F:466-467) on the State-only path; the
    RK source path restores the entry qc/qi and couples the raw rates instead."""
    back = lambda value: jnp.moveaxis(value.reshape(ny, nx, nz), -1, 0)
    updates = {key: (getattr(state, key) + float(dt) * back(rates[key])).astype(_output_dtype(state, key))
               for key in ('qc', 'qi') if key in rates}
    return result.replace(**updates) if updates else result


def _apply_pbl_condensate(result: State, state: State, dt: float, rates: dict, f: dict) -> State:
    """State-only path: apply RQCBLTEN/RQIBLTEN (``(ncol, nz)``) as qc/qi increments.

    The RK source path (operational_mode sf1 branch) restores the entry qc/qi and
    couples the same raw rates once through moist_tendf instead.
    """
    back = lambda x: jnp.moveaxis(x.reshape(f["ny"], f["nx"], f["nz"]), -1, 0)  # noqa: E731
    return result.replace(**{k: (getattr(state, k) + float(dt) * back(rates[k])).astype(_output_dtype(state, k))
                             for k in ("qc", "qi") if k in rates})



def ysu_pbl_adapter(state: State, dt: float, grid=None, *, urban=None, _with_sources=False,
                    surface_forcing=None, rthraten=None, ysu_topdown_pblmix=1) -> State:
    """bl_pbl=1 YSU PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel).

    module_pbl_driver.F:1208-1251: QC3D/QI3D (zero fields without P_QC/P_QI give
    zero rates), RTHRATEN (held radiation theta rate), YSU_TOPDOWN_PBLMIX
    (Registry default 1), CTOPO/CTOPO2 = 1 (start_em.F:1581, topo_wind=0)."""

    ysu_kwargs = dict(rthraten=rthraten, ysu_topdown_pblmix=int(ysu_topdown_pblmix))
    if urban is not None:
        from gpuwrf.coupling.urban_pbl_hook import urban_pbl_forcing,urban_pbl_full_columns,replace_urban_tendencies
        f=urban_pbl_forcing(state,grid,urban[1].surface_layer)
        out=urban_pbl_full_columns(f,dt,1,ysu_kwargs=ysu_kwargs)
        out=replace_urban_tendencies(state,f,out,dt=dt,grid=grid,bundle=urban[0],carry=urban[1],scheme=1,
                                     ysu_kwargs=ysu_kwargs)
        result=_apply_pbl_increment(state,dt,out,ny=f['ny'],nx=f['nx'],nz=f['nz'])
        result=_apply_ysu_cloud_rates(result,state,dt,out,ny=f['ny'],nx=f['nx'],nz=f['nz'])
        result=result.replace(pblh=out['pblh'].reshape(f['ny'],f['nx']).astype(state.pblh.dtype))
        if state.cam_kvh3d is not None:
            from gpuwrf.physics.cammg.pbl import retain_columns
            result=retain_columns(result,exch_h=out['exch_h'])
        return (result,out) if _with_sources else result
    forcing_state = state
    if surface_forcing is None:
        # Direct adapter callers have no preceding surface driver. Use the YSU
        # revised-MM5 partner; operational callers pass the already-paid solve.
        forcing_state, surface_forcing = sfclay_revised_mm5_adapter(
            state, dt, grid, history=True)
    f = _pbl_surface_forcing(forcing_state, grid, surface_forcing=surface_forcing, use_resistances=True)
    cols = lambda field3d: jnp.moveaxis(field3d, 0, -1).reshape(f["ncol"], f["nz"])
    # Original YSU has REAL inputs/arithmetic, no DOUBLE island: honour that
    # ABI even when the surrounding legacy carry is wide.
    with jax.enable_x64(False):
        real = lambda value: jnp.asarray(value, jnp.float32)
        out = ysu_columns(
            real(f["u_cols"]), real(f["v_cols"]), real(f["T_cols"]), real(f["qv_cols"]), real(f["p_cols"]),
            real(f["p_int_cols"]), real(f["pii_cols"]), real(f["dz_cols"]),
            psfc=real(f['psfc'] if 'psfc' in f else f["p_cols"][:, 0]), znt=real(f["znt"]), ust=real(f["ust"]),
            hfx=real(f["hfx"]), qfx=real(f["qfx"]), wspd=real(f["wspd"]), br=real(f["br"]), psim=real(f["psim"]),
            psih=real(f["psih"]), dt=float(dt), xland=real(f["xland"]), u10=real(f["u10"]), v10=real(f["v10"]),
            qc=real(cols(forcing_state.qc)), qi=real(cols(forcing_state.qi)),
            rthraten=None if rthraten is None else real(cols(rthraten)),
            ysu_topdown_pblmix=int(ysu_topdown_pblmix),
        )
    result = _apply_pbl_increment(state, dt, _pbl_state_rates(out,f), ny=f["ny"], nx=f["nx"], nz=f["nz"])
    result = _apply_ysu_cloud_rates(result, state, dt, out, ny=f["ny"], nx=f["nx"], nz=f["nz"])
    result = _pbl_diagnostic_state(result,out,surface_forcing)
    if state.cam_kvh3d is not None:
        from gpuwrf.physics.cammg.pbl import retain_columns
        result=retain_columns(result,exch_h=out['exch_h'])
    return (result, out) if _with_sources else result


def acm2_pbl_adapter(state: State, dt: float, grid=None, *, surface_forcing=None, _with_sources=False,
                     xtime=60) -> State:
    """bl_pbl=7 ACM2 PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel)."""

    f = _pbl_surface_forcing(state, grid, surface_forcing=surface_forcing)
    mut = jnp.asarray(state.mu_total if hasattr(state, "mu_total") else state.mu, jnp.float64).reshape(f["ncol"])
    cols = lambda field3d: jnp.moveaxis(jnp.asarray(field3d, f["qv_cols"].dtype), 0, -1).reshape(f["ncol"], f["nz"])  # noqa: E731
    out = acm2_columns(
        f["u_cols"], f["v_cols"], f["theta_cols"], f["T_cols"], f["qv_cols"],
        f["rho_cols"], f["dz_cols"],
        pblh_initial=f["pblh"], ust=f["ust"], hfx=f["hfx"], qfx=f["qfx"],
        wspd=f["wspd"], mut=mut, dt=float(dt), xtime=xtime,
        # pbl_driver.F:1592 passes QC3D/QI3D: they drive EDDYX's moist Richardson
        # number (imvdif) and are mixed (RQCBLTEN/RQIBLTEN).
        qc=cols(state.qc), qi=cols(state.qi),
    )
    result = _apply_pbl_increment(state, dt, _pbl_state_rates(out,f), ny=f["ny"], nx=f["nx"], nz=f["nz"])
    back = lambda x: jnp.moveaxis(x.reshape(f["ny"], f["nx"], f["nz"]), -1, 0)  # noqa: E731
    result = result.replace(**{k: (getattr(state, k) + float(dt) * back(out[k])).astype(getattr(state, k).dtype)
                               for k in ("qc", "qi") if k in out})
    result = _pbl_diagnostic_state(result,out,surface_forcing)
    if state.cam_kvh3d is not None:
        from gpuwrf.physics.cammg.pbl import retain_columns
        result=retain_columns(result,exch_h=out['exch_h'])
    return (result,out) if _with_sources else result


def boulac_pbl_adapter(state: State, dt: float, grid=None, *, urban=None, _with_sources=False, surface_forcing=None) -> State:
    """bl_pbl=8 BouLac PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel)."""

    if urban is not None:
        from gpuwrf.coupling.urban_pbl_hook import urban_pbl_forcing,urban_pbl_full_columns,replace_urban_tendencies
        f=urban_pbl_forcing(state,grid,urban[1].surface_layer)
        out=urban_pbl_full_columns(f,dt,8)
        out=replace_urban_tendencies(state,f,out,dt=dt,grid=grid,bundle=urban[0],carry=urban[1],scheme=8)
        result=_apply_pbl_increment(state,dt,out,ny=f['ny'],nx=f['nx'],nz=f['nz'])
        back=lambda x:jnp.moveaxis(x.reshape(f['ny'],f['nx'],f['nz']),-1,0)
        result=result.replace(qc=(state.qc+float(dt)*back(out['qc'])).astype(state.qc.dtype),
            qke=back(out['tke']).astype(state.qke.dtype),
            pblh=out['pblh'].reshape(f['ny'],f['nx']).astype(state.pblh.dtype))
        if state.cam_kvh3d is not None:
            from gpuwrf.physics.cammg.pbl import retain_columns
            result=retain_columns(result,exch_h=out['exch_h'],tke=out['tke'])
        return (result,out) if _with_sources else result
    f = _pbl_surface_forcing(state, grid, surface_forcing=surface_forcing)

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(f["ncol"], f["nz"])

    out = boulac_columns(
        f["u_cols"],
        f["v_cols"],
        f["theta_cols"],
        f["qv_cols"],
        _cols(state.qc),
        f["rho_cols"],
        f["dz_cols"],
        jnp.maximum(_cols(state.qke), BOULAC_TKE_MIN),
        hfx=f["hfx"],
        qfx=f["qfx"],
        ust=f["ust"],
        dt=float(dt),
    )
    next_state = _apply_pbl_increment(state, dt, _pbl_state_rates(out,f), ny=f["ny"], nx=f["nx"], nz=f["nz"])

    def _back3d(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(f["ny"], f["nx"], f["nz"]), -1, 0)

    dt_f = float(dt)
    result = next_state.replace(
        qc=(state.qc + dt_f * _back3d(out["qc"])).astype(_output_dtype(state, "qc")),
        qke=_back3d(out["tke"]).astype(_output_dtype(state, "qke")),
    )
    result = _pbl_diagnostic_state(result,out,surface_forcing)
    if state.cam_kvh3d is not None:
        from gpuwrf.physics.cammg.pbl import retain_columns
        result=retain_columns(result,exch_h=out['exch_h'],tke=out['tke'])
    return (result, out) if _with_sources else result


def mrf_pbl_adapter(state: State, dt: float, grid=None, *, surface_forcing=None, _with_sources=False,
                    f_qi=True) -> State:
    """bl_pbl=99 MRF PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel).

    The MRF (Hong-Pan 1996) nonlocal-K PBL is YSU's predecessor; it consumes the
    SAME revised-MM5 surface-layer forcing the YSU adapter re-derives, plus the
    skin temperature ``tsk`` and the ``gz1oz0 = ln(za_1/znt)`` surface-layer log
    ratio that MRF reads. The momentum increment is applied A2C-faithfully like the
    other PBL adapters.
    """

    f = _pbl_surface_forcing(state, grid, surface_forcing=surface_forcing)
    znt = f["znt"]
    za1 = 0.5 * f["dz_cols"][:, 0]  # ~mid-height of the lowest mass level
    gz1oz0 = f['gz1oz0'] if 'gz1oz0' in f else jnp.log(jnp.maximum(za1, znt) / znt)
    cols = lambda field3d: jnp.moveaxis(jnp.asarray(field3d, f["qv_cols"].dtype), 0, -1).reshape(f["ncol"], f["nz"])  # noqa: E731
    out = mrf_columns(
        f["u_cols"], f["v_cols"], f["T_cols"], f["qv_cols"],
        # module_pbl_driver.F:1358-1359 passes QC3D=qc_curr, QI3D=qi_curr (FLAG_QI):
        # mixed (RQCBLTEN/RQIBLTEN) and they switch MRF2D's in-cloud Ri (:999-1000).
        cols(state.qc),
        f["p_cols"], f["pii_cols"], f["dz_cols"], f["z_cols"],
        psfc=f['psfc'] if 'psfc' in f else f["p_cols"][:, 0], znt=znt, ust=f["ust"], hfx=f["hfx"], qfx=f["qfx"],
        tsk=f["tsk"], gz1oz0=gz1oz0, wspd=f["wspd"], br=f["br"],
        psim=f["psim"], psih=f["psih"], xland=f["xland"], dt=float(dt),
        # FLAG_QI=f_qi: MPs without a WRF QI slot (Registry mp 0/1/3) mix no ice.
        qi=cols(state.qi) if f_qi else None,
    )
    # The MRF kernel emits WRF-named tendency RATES (rublten/.../rthblten with the
    # theta tendency already divided by exner, exactly the WRF wrapper convention);
    # map them to the shared (u/v/theta/qv/qc/qi) increment-rate contract.
    tend = {"u": out["rublten"], "v": out["rvblten"],
            "theta": out["rthblten"], "qv": out["rqvblten"],
            "qc": out["rqcblten"], **({"qi": out["rqiblten"]} if f_qi else {})}
    result = _apply_pbl_increment(state, dt, _pbl_state_rates(tend,f), ny=f["ny"], nx=f["nx"], nz=f["nz"])
    result = _apply_pbl_condensate(result, state, dt, tend, f)
    # MRF2D rewrites the driver's UST (INOUT, module_bl_mrf.F:774-781).
    result = result.replace(ustar=out["ust"].reshape(f["ny"], f["nx"]).astype(_output_dtype(state, "ustar")))
    result = _pbl_diagnostic_state(result,out,surface_forcing)
    return (result,tend) if _with_sources else result


def gfs_pbl_adapter(state: State, dt: float, grid=None, *, surface_forcing=None, _with_sources=False,
                    f_qi=True) -> State:
    """bl_pbl=3 GFS PBL ``State -> State`` scan adapter (jit/vmap-traceable kernel).

    The GFS (Hong-Pan lineage) nonlocal-K PBL is the operational NCEP first-order
    closure and an algorithmic ancestor of YSU; it consumes the SAME revised-MM5
    surface-layer forcing the YSU/MRF adapters re-derive, plus the ``gz1oz0 =
    ln(za_1/znt)`` surface-layer log ratio that GFS reads (with the surface stress
    re-derived inside the kernel as ``(karman*wspd/(gz1oz0-psim))^2``). The
    momentum increment is applied A2C-faithfully like the other PBL adapters.
    """

    f = _pbl_surface_forcing(state, grid, surface_forcing=surface_forcing)
    znt = f["znt"]
    za1 = 0.5 * f["dz_cols"][:, 0]  # ~mid-height of the lowest mass level
    gz1oz0 = f['gz1oz0'] if 'gz1oz0' in f else jnp.log(jnp.maximum(za1, znt) / znt)
    cols = lambda field3d: jnp.moveaxis(jnp.asarray(field3d, f["qv_cols"].dtype), 0, -1).reshape(f["ncol"], f["nz"])  # noqa: E731
    args = (f["u_cols"], f["v_cols"], f["T_cols"], f["qv_cols"],
            # module_pbl_driver.F:1408 passes QC3D=qc_curr, QI3D=qi_curr: MONINP tracers
            # 2 and 3 (NTRAC=3 with P_QI), returned as RQCBLTEN/RQIBLTEN.
            cols(state.qc), f["p_cols"], f["pii_cols"], f["dz_cols"], f["z_cols"])
    kwargs = dict(psfc=f['psfc'] if 'psfc' in f else f["p_cols"][:, 0], ust=f["ust"], hfx=f["hfx"],
                  qfx=f["qfx"], tsk=f["tsk"], gz1oz0=gz1oz0, psim=f["psim"], psih=f["psih"],
                  wspd=f["wspd"], br=f["br"],
                  # NTRAC=3 only when P_QI exists (Registry mp 0/1/3 have no QI slot).
                  qi=cols(state.qi) if f_qi else None)
    if surface_forcing is not None:
        out = gfs_bl_real_double(args, kwargs, float(dt))
    else:
        out = gfs_columns(*args, dt=float(dt), **kwargs)
    # The GFS kernel emits WRF-named tendency RATES (rublten/.../rthblten with the
    # theta tendency already divided by exner, the WRF BL_GFS wrapper convention);
    # map them to the shared (u/v/theta/qv/qc/qi) increment-rate contract.
    tend = {"u": out["rublten"], "v": out["rvblten"],
            "theta": out["rthblten"], "qv": out["rqvblten"],
            "qc": out["rqcblten"], **({"qi": out["rqiblten"]} if f_qi else {})}
    result = _apply_pbl_increment(state, dt, _pbl_state_rates(tend,f), ny=f["ny"], nx=f["nx"], nz=f["nz"])
    result = _apply_pbl_condensate(result, state, dt, tend, f)
    result = _pbl_diagnostic_state(result,out,surface_forcing)
    if surface_forcing is not None:
        # module_bl_gfs.F:464-466 INOUT grid stores after MONINP: UST=SQRT(STRESS) and
        # WSPD=SQRT(U3D(kts)**2+V3D(kts)**2)+1.E-9 (REAL); the next SF_GFS call reads UST.
        shape = state.ustar.shape
        written = {'ustar': out['ust'].reshape(shape).astype(state.ustar.dtype)}
        if getattr(state, 'sfc_wspd', None) is not None:
            u0 = jnp.asarray(f["u_cols"][:, 0], jnp.float32)
            v0 = jnp.asarray(f["v_cols"][:, 0], jnp.float32)
            written['sfc_wspd'] = (jnp.sqrt(u0 * u0 + v0 * v0) + jnp.float32(1.e-9)).reshape(shape).astype(
                state.sfc_wspd.dtype)
        result = result.replace(_cast=False, **written)
    # pbl_driver.F:1418 KPBL2D=kpbl -> grid%kpbl, read by the GF cumulus caller (P1:1397).
    # (both real GFS kernels return it; a stub without it leaves GF on its PBLH fallback)
    return (result,{**tend,**({"kpbl":out["kpbl"]} if "kpbl" in out else {})}) if _with_sources else result


def gfs_bl_real_double(args, kwargs, dt):
    """WRF BL_GFS call: REAL setup statements, kind_phys DOUBLE MONINP, REAL stores.

    module_bl_gfs.F:276-398 evaluates its pure-REAL right-hand sides (QV3D/(1.+QV3D),
    P3D*.001, PSFC*.00001, Z differences ...) in REAL before storing them into the
    REAL(kind_phys) work arrays; MONINP (:569-1298) runs in DOUBLE (module_gfs_machine.F:8)
    with WRF's default-REAL literal constants; tendencies/UST/PBL are stored back REAL.
    An explicit narrow fp64 island inside the REAL caller (S2: WRF DOUBLE).
    """
    return bl_gfs_wrf_columns(*args, dt=float(dt), **kwargs)


def shinhong_pbl_adapter(state: State, dt: float, grid=None, *, shinhong_tke_diag=0,
                        surface_forcing=None, _with_sources=False) -> State:
    """bl_pbl=11 Shin-Hong scale-aware PBL ``State -> State`` scan adapter."""

    f = _pbl_surface_forcing(state, grid, surface_forcing=surface_forcing, use_resistances=True)
    dx = 1000.0 if grid is None else float(grid.projection.dx_m)
    dy = 1000.0 if grid is None else float(grid.projection.dy_m)
    # WRF passes the same TKE_PBL that microphysics later consumes. CAM's
    # shared profile is authoritative, independently of transported QKE.
    tke_in = state.qke if state.cam_tke_pbl is None else state.cam_tke_pbl[:-1]
    cols = lambda field3d: jnp.moveaxis(field3d, 0, -1).reshape(f["ncol"], f["nz"])
    # module_pbl_driver.F:1301-1329 passes QC3D/QI3D (moist P_QC/P_QI, zero slot
    # without ice) and CORF=f: in-cloud Ri (:1049), mixlen cwm (:1565) and the
    # stable mixing length (:1993) consume them.
    corf = 1.0e-4 if grid is None else grid.metrics.f.reshape(-1)
    # Original Shin-Hong has REAL inputs/arithmetic, no DOUBLE island.
    # Honour that ABI even when the surrounding legacy carry is wide.
    with jax.enable_x64(False):
        real = lambda value: jnp.asarray(value, jnp.float32)
        out = shinhong_columns(
            real(f["u_cols"]),
            real(f["v_cols"]),
            real(f["T_cols"]),
            real(f["qv_cols"]),
            real(f["p_cols"]),
            real(f["p_int_cols"]),
            real(f["pii_cols"]),
            real(f["dz_cols"]),
            real(jnp.moveaxis(tke_in, 0, -1).reshape(f["ncol"], f["nz"])),
            psfc=real(f['psfc'] if 'psfc' in f else f["p_cols"][:, 0]),
            znt=real(f["znt"]),
            ust=real(f["ust"]),
            hfx=real(f["hfx"]),
            qfx=real(f["qfx"]),
            wspd=real(f["wspd"]),
            br=real(f["br"]),
            psim=real(f["psim"]),
            psih=real(f["psih"]),
            dt=real(float(dt)),
            xland=real(f["xland"]),
            u10=real(f["u10"]),
            v10=real(f["v10"]),
            dx=real(dx),
            dy=real(dy),
            shinhong_tke_diag=shinhong_tke_diag,
            qc=real(cols(state.qc)),
            qi=real(cols(state.qi)),
            corf=real(corf),
        )
    next_state = _apply_pbl_increment(state, dt, _pbl_state_rates(out,f), ny=f["ny"], nx=f["nx"], nz=f["nz"])

    def _back3d(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(f["ny"], f["nx"], f["nz"]), -1, 0)

    # module_bl_shinhong.F:1478-1612 only writes shared TKE for diag=1.
    tke_out = _back3d(out["tke"]) if int(shinhong_tke_diag)==1 else tke_in
    dt_f = float(dt)
    result = next_state.replace(
        qke=tke_out.astype(_output_dtype(state, "qke")),
        # RQCBLTEN/RQIBLTEN (module_bl_shinhong.F:253-256); the RK source path
        # restores the entry qc/qi and couples these raw rates instead.
        qc=(state.qc + dt_f * _back3d(out["qc"])).astype(_output_dtype(state, "qc")),
        qi=(state.qi + dt_f * _back3d(out["qi"])).astype(_output_dtype(state, "qi")),
    )
    result = _pbl_diagnostic_state(result,out,surface_forcing)
    if state.cam_kvh3d is not None:
        from gpuwrf.physics.cammg.pbl import retain_columns
        result=retain_columns(result,exch_h=out['exch_h'],
            tke=out['tke'] if int(shinhong_tke_diag)==1 else None)
    return (result,out) if _with_sources else result


def gbm_pbl_adapter(state: State, dt: float, grid=None, *, surface_forcing=None, _with_sources=False,
                    rthraten=None) -> State:
    """bl_pbl=12 GBM moist prognostic-TKE PBL ``State -> State`` scan adapter."""

    f = _pbl_surface_forcing(state, grid, surface_forcing=surface_forcing)
    znt = f["znt"]
    za1 = 0.5 * f["dz_cols"][:, 0]
    gz1oz0 = f['gz1oz0'] if 'gz1oz0' in f else jnp.log(jnp.maximum(za1, znt) / znt)

    def _cols(field3d):  # (nz, ny, nx) -> (ncol, nz)
        return jnp.moveaxis(field3d, 0, -1).reshape(f["ncol"], f["nz"])

    out = gbm_columns(
        f["u_cols"],
        f["v_cols"],
        f["T_cols"],
        f["qv_cols"],
        _cols(state.qc),
        f["p_cols"],
        f["pii_cols"],
        f["dz_cols"],
        jnp.maximum(_cols(state.qke), 0.001),
        psfc=f['psfc'] if 'psfc' in f else f["p_cols"][:, 0],
        znt=znt,
        ust=f["ust"],
        hfx=f["hfx"],
        qfx=f["qfx"],
        tsk=f["tsk"],
        gz1oz0=gz1oz0,
        wspd=f["wspd"],
        br=f["br"],
        psim=f["psim"],
        psih=f["psih"],
        dt=float(dt),
        xland=f["xland"],
        # module_pbl_driver.F GBMPBL(RTHRATEN=rthraten): held radiative heating (gbmpbl.F:243).
        rthraten=None if rthraten is None else _cols(jnp.asarray(rthraten, f["qv_cols"].dtype)),
    )
    next_state = _apply_pbl_increment(state, dt, _pbl_state_rates(out,f), ny=f["ny"], nx=f["nx"], nz=f["nz"])

    def _back3d(field2d):  # (ncol, nz) -> (nz, ny, nx)
        return jnp.moveaxis(field2d.reshape(f["ny"], f["nx"], f["nz"]), -1, 0)

    dt_f = float(dt)
    result = next_state.replace(
        qc=(state.qc + dt_f * _back3d(out["qc"])).astype(_output_dtype(state, "qc")),
        qke=_back3d(out["tke"]).astype(_output_dtype(state, "qke")),
        # GBM2D reconstructs UST (:539) and publishes it to the driver (:835).
        ustar=out["ust"].reshape(f["ny"], f["nx"]).astype(_output_dtype(state, "ustar")),
        exch_tke=_back3d(out["kethl"]).astype(jnp.float32),
    )
    result = _pbl_diagnostic_state(result,out,surface_forcing)
    if state.cam_kvh3d is not None:
        from gpuwrf.physics.cammg.pbl import retain_columns
        result=retain_columns(result,exch_h=out['kth'],tke=out['tke'])
    return (result,out) if _with_sources else result


def camuw_pbl_adapter(state: State, dt: float, grid=None, *, camuw: CamUwCarry | None = None,
                      rthraten=None, cldfra=None, return_tendencies=False, return_cumulus_inputs=False, cammg=False):
    """bl_pbl=9 CAM-UW (UW moist turbulence) ``(State, CamUwCarry) -> (State, CamUwCarry)``.

    Inputs follow WRF ``phy_prep``/``module_pbl_driver.F:2033``: mass-point winds, dry physics
    temperature/nonhydrostatic Exner, ``rho=(1+qv)/alt`` (metrics) or EOS,
    hydrostatic ``p_phy/p8w`` (first_rk_step_part1.F:1130–1132),
    ``z_at_w=ph/g`` with WRF g, ``HFX/QFX/UST`` from the surface driver (State ``hfx/qfx/ustar``:
    LSM-blended), ``CLDFRA`` = held radiation cloud fraction, ``QNC`` = 0 (only diffused, never
    returned by WRF). Carry = WRF ``KVM3D/KVH3D/TAURESX2D/TAURESY2D`` (REAL, zero at itimestep 1,
    which reproduces WRF's is_first_step branch). Tendencies are applied like WRF
    ``update_phy_ten`` (CAMUWPBLSCHEME): RU/RV (a2c), RTH (dry -> theta_m), RQV, RQC, RQI, RQNI.
    With ``return_tendencies=True``, return the unintegrated WRF rates alongside
    diagnostic State/carry; the operational RK consumer applies them at each stage.
    ``return_cumulus_inputs=True`` appends ``(TPERT, KPBL)`` to either return
    form: WRF REAL temperature excess (K), INTEGER one-based PBL-top index.

    Known approximation: ``rthraten`` is the held TOTAL radiative heating; WRF passes the LW-only
    RTHRATENLW (identical at night). PBLH is written to ``State.pblh`` (WRF PBLH2D).
    """

    nz, ny, nx = state.theta.shape
    ncol = ny * nx
    f32 = jnp.float32
    if camuw is None:
        camuw = initial_camuw_carry(state)

    def cols(a, n=nz):
        return jnp.moveaxis(jnp.asarray(a, f32), 0, -1).reshape(ncol, n)

    def flat(a):
        return jnp.asarray(a, f32).reshape(ncol)

    theta_dry = _dry_theta_view(state)
    p_nonhyd = jnp.asarray(state.p, jnp.float64)
    exner = (jnp.maximum(p_nonhyd, 1.0) / 1.0e5) ** (287.0 / 1004.5)  # WRF phy_prep pi_phy
    t_phy = jnp.asarray(theta_dry, jnp.float64) * exner
    z_face = jnp.asarray(state.ph, jnp.float64) / WRF_PHYSICS_G
    z_mass = 0.5 * (z_face[:-1] + z_face[1:])
    metrics = getattr(grid, "metrics", None)
    if metrics is not None:
        # WRF passes p_hyd/p_hyd_w here, independently of pi_phy/t_phy.
        # Reuse the existing single-column GPU hydrostatic kernel.
        p, p8w, _psfc = _wrf_hydrostatic_pressure_profiles_from_state(state, metrics, output_dtype=f32)
        rho = _wrf_phy_prep_rho_from_state(state, metrics, output_dtype=f32)
    else:  # analytic callers without eta weights
        p = p_nonhyd
        p8w = jnp.concatenate([p[:1], 0.5 * (p[:-1] + p[1:]), p[-1:]], axis=0)
        rho = _rho_from_state(state) * (1.0 + jnp.asarray(state.qv, jnp.float64))
    zeros3 = jnp.zeros((nz, ny, nx), f32)
    ni = state.Ni if state.Ni is not None else zeros3
    rth = zeros3 if rthraten is None else rthraten
    cf = state.cldfra_old_mp if cammg else zeros3 if cldfra is None else cldfra
    hfx = state.hfx if getattr(state, "hfx", None) is not None else jnp.zeros((ny, nx), f32)
    qfx = state.qfx if getattr(state, "qfx", None) is not None else jnp.zeros((ny, nx), f32)
    out = jax.vmap(lambda *a: camuw_column(*a, first_step=False, **({"cammg":True} if cammg else {})))(
        jnp.full((ncol,), dt, f32), cols(_u_mass(state)), cols(_v_mass(state)), cols(theta_dry), cols(rho),
        cols(state.qv), cols(state.qc), cols(state.qi), cols(state.Nc if cammg else zeros3), cols(ni), cols(p), cols(p8w, nz + 1),
        cols(z_mass), cols(z_face, nz + 1), cols(t_phy), cols(cf), cols(rth), cols(exner), cols(state.wsedl3d if cammg else zeros3),
        flat(hfx), flat(qfx), flat(state.ustar), flat(z_face[0]), cols(camuw.kvm3d, nz + 1),
        cols(camuw.kvh3d, nz + 1), flat(camuw.tauresx2d), flat(camuw.tauresy2d))

    def back(a, n=nz):
        return jnp.moveaxis(jnp.asarray(a).reshape(ny, nx, n), -1, 0)

    next_carry = CamUwCarry(
        kvm3d=back(out["kvm3d"], nz + 1), kvh3d=back(out["kvh3d"], nz + 1),
        tauresx2d=out["tauresx2d"].reshape(ny, nx), tauresy2d=out["tauresy2d"].reshape(ny, nx),
        tke_pbl=back(out["tke_pbl"], nz + 1))
    diagnostics = {}
    if cammg:
        diagnostics.update(cam_kvh3d=back(out["kvh3d"],nz+1),cam_tke_pbl=back(out["tke_pbl"],nz+1),
                           cam_turbtype3d=back(out["turbtype3d"],nz+1),cam_smaw3d=back(out["smaw3d"],nz+1))
    if getattr(state, "pblh", None) is not None:
        diagnostics["pblh"] = out["pblh"].reshape(ny, nx).astype(_output_dtype(state, "pblh"))
    cumulus_inputs = ()
    if return_cumulus_inputs:
        # CAMUWPBL driver:752–754, consumed by the subsequent cumulus driver.
        # These diagnostics are already computed; keep them out of persistent carry.
        cumulus_inputs = ((out["tpert"].reshape(ny, nx).astype(f32),
                           out["kpbl"].reshape(ny, nx).astype(jnp.int32)),)
    if return_tendencies:
        # WRF module_physics_addtendc.F:903–969: do not round rate*dt
        # into the prognostics before calculate_phy_tend/update_phy_ten.
        rates = {name: back(out[name]) for name in (
            "rublten", "rvblten", "rthblten", "rqvblten", "rqcblten", "rqiblten", "rqniblten")}
        return (state.replace(**diagnostics), next_carry, rates) + cumulus_inputs

    dt_f = float(dt)
    u_new = _add_a2c_u_increment(state.u, dt_f * back(out["rublten"])).astype(_output_dtype(state, "u"))
    v_new = _add_a2c_v_increment(state.v, dt_f * back(out["rvblten"])).astype(_output_dtype(state, "v"))
    rqv = back(out["rqvblten"])
    theta_dtype = jnp.asarray(state.theta).dtype
    theta_new = (jnp.asarray(state.theta, theta_dtype) + dt_f * _theta_m_tendency_from_dry(
        back(out["rthblten"]), rqv, state.theta, state.qv, theta_dtype)).astype(_output_dtype(state, "theta"))
    updates = dict(
        u=u_new, v=v_new, theta=theta_new,
        qv=(state.qv + dt_f * rqv).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * back(out["rqcblten"])).astype(_output_dtype(state, "qc")),
        qi=(state.qi + dt_f * back(out["rqiblten"])).astype(_output_dtype(state, "qi")),
    )
    if state.Ni is not None:
        updates["Ni"] = (state.Ni + dt_f * back(out["rqniblten"])).astype(_output_dtype(state, "Ni"))
    updates.update(diagnostics)
    return (state.replace(**updates), next_carry) + cumulus_inputs


# --- dispatch tables ----------------------------------------------------------

# Microphysics options whose State->State scan adapter is threaded into the GPU
# scan. mp=8 (Thompson) is the existing physics_couplers.thompson_adapter; mp=0 is
# passive (no microphysics). The rest map to the adapters above.
from gpuwrf.physics.cammg.adapter import cammg_adapter

MP_SCAN_ADAPTERS = {
    11: cammg_adapter,
    5: ferrier5_adapter,
    15: ferrier15_adapter,
    1: kessler_adapter,
    2: lin_adapter,
    3: wsm3_adapter,
    4: wsm5_adapter,
    6: wsm6_adapter,
    7: goddard4ice_adapter,
    38: thompson_hail_adapter,
    10: morrison_adapter,
    13: sbu_ylin_adapter,
    # v0.3.4 O1: Morrison-aerosol at the WRF stand-alone point aercu_opt=0.
    40: morrison_aero_adapter,
    14: wdm5_adapter,
    16: wdm6_adapter,
    # v0.17 WSM7 = WSM6 + separate precipitating hail (qh + hail_acc).
    24: wsm7_adapter,
    # v0.17 WDM7 = WDM6 double-moment + separate single-moment hail.
    26: wdm7_adapter,
    97: goddard_adapter,
}


def nssl_public_precipitation(precip: dict) -> dict:
    """NSSL_2MOM public slots (microphysics_driver.F:2223): all-phase RAINNCV plus
    SNOWNCV, GRAUPELNCV and HAILNCV for the land-surface caller."""
    zero = jnp.zeros_like(precip["snowncv"])
    return {"rain": precip["rain_liquid"], "snow": precip["snowncv"], "graupel": precip["grplncv"],
            "ice": zero, "hail": precip["hailncv"], "total": precip["rainncv"], "sr": precip["sr"]}


def nssl2mom_scan_adapter(state, dt: float, grid=None, *, return_precipitation=False):
    """mp=18 NSSL 2-moment registry adapter (non-first step). The operational scan calls
    ``physics.nssl2mom.adapter.nssl2mom_adapter`` directly with WRF's itimestep==1 flag."""
    from gpuwrf.physics.nssl2mom.adapter import nssl2mom_adapter

    if return_precipitation:
        updated, precip = nssl2mom_adapter(state, dt, grid, return_precipitation=True)
        return updated, nssl_public_precipitation(precip)
    return nssl2mom_adapter(state, dt, grid)


# v0.3.4 NSSL 2-moment (mp=18): physics.nssl2mom port, CPU-oracle-qualified.
MP_SCAN_ADAPTERS[18] = nssl2mom_scan_adapter
MP_SCAN_ADAPTERS[17] = nssl2mom_scan_adapter  # WRF 4.7.1 legacy alias, same qualified kernel.


def milbrandt2mom_scan_adapter(state, dt: float, grid=None):
    from gpuwrf.physics.milbrandt2mom.adapter import milbrandt_adapter
    return milbrandt_adapter(state, dt, grid)


MP_SCAN_ADAPTERS[9] = milbrandt2mom_scan_adapter
def _p3_scan_adapter(mp):
    def adapter(state, dt, grid=None):
        from gpuwrf.physics.p3.adapter import p3_adapter
        return p3_adapter(state,dt,grid,mp=mp)
    return adapter


for _p3_option in (50,51,52,53):
    MP_SCAN_ADAPTERS[_p3_option] = _p3_scan_adapter(_p3_option)


def jensen_scan_adapter(state,dt,grid=None):
    from gpuwrf.physics.jensen_ishmael.adapter import jensen_adapter
    return jensen_adapter(state,dt,grid)


MP_SCAN_ADAPTERS[55]=jensen_scan_adapter


def nssl_nohail_scan_adapter(state, dt: float, grid=None):
    """WRF legacy22: two-moment NSSL, predicted CCN/graupel density, no hail."""
    from gpuwrf.physics.nssl2mom.adapter import nssl2mom_adapter
    return nssl2mom_adapter(state, dt, grid, hail_enabled=False)


MP_SCAN_ADAPTERS[22] = nssl_nohail_scan_adapter


def nssl1mom19_scan_adapter(state,dt:float,grid=None):
    from gpuwrf.physics.nssl2mom.adapter import nssl1mom19_adapter
    return nssl1mom19_adapter(state,dt,grid)


def nssl1mom21_scan_adapter(state,dt:float,grid=None):
    from gpuwrf.physics.nssl2mom.adapter import nssl1mom21_adapter
    return nssl1mom21_adapter(state,dt,grid)


MP_SCAN_ADAPTERS[19]=nssl1mom19_scan_adapter
MP_SCAN_ADAPTERS[21]=nssl1mom21_scan_adapter

# Surface-layer options whose adapter is threaded (sf_sfclay=5 MYNN-sfclay is the
# existing surface_adapter; sf_sfclay=0 disables).
SFCLAY_SCAN_ADAPTERS = {
    1: sfclay_revised_mm5_adapter,
    3: gfs_sfclay_adapter,
    7: pleim_xiu_sfclay_adapter,
    91: sfclay_old_mm5_adapter,
}

# Cumulus options whose adapter is threaded (cu=0 = no cumulus). KF (1) is the
# carry-threaded ``(State, carry) -> (State, carry)`` adapter; Tiedtke (6) and
# Grell-Freitas (3) are GPU-batched (jit/vmap) ``State -> State`` adapters (no
# persistent carry). GF (3) is the v0.9.0 GPU-batched port of the scale-aware
# closure-ensemble kernel (physics._gf_jax.gfdrv_batched), savepoint-parity gated
# (proofs/v060/gf_gpubatch_savepoint_parity.json) and scan-wired here.
CU_SCAN_ADAPTERS = {
    1: kf_adapter,
    2: bmj_adapter,
    3: gf_adapter,
    6: tiedtke_adapter,
    # v0.23 F2: New-Tiedtke, machine-precision-proven column kernel.
    16: ntiedtke_adapter,
    # v0.3.4 o1-grell: Grell-3D / Grell-Devenyi (proofs/v034, CPU-oracle-qualified).
    5: g3_adapter,
    93: gd_adapter,
    # v0.3.4 o1-sas: scale-aware GFS SAS -- rates only; the held-rate cadence step is
    # routed explicitly in the cumulus slot (coupling.scalesas_adapter), like KF.
    4: scalesas_rates,
    94: sas94_rates,
    95: osas95_rates,
    99: kf99_rates,
    14: ksas14_rates,
    96: nsas96_rates,
}

# Cumulus options that carry NO persistent cumulus state (plain State->State, like
# the microphysics adapters). KF (1) and BMJ (2) are excluded -- KF threads
# (w0avg, nca) and BMJ threads CLDEFI. Tiedtke (6) and Grell-Freitas (3) are the
# GPU-batched stateless adapters.
CU_STATELESS_SCAN_ADAPTERS = {
    3: gf_adapter,
    6: tiedtke_adapter,
    16: ntiedtke_adapter,
    5: g3_adapter,
    93: gd_adapter,
}

# PBL options whose scan adapter is threaded (bl=5 MYNN is the existing
# physics_couplers.mynn_adapter; bl=0 disables). YSU(1)/ACM2(7)/BouLac(8) are
# v0.6.0 jax.lax.scan-traceable rewrites; MRF(99) is the v0.13 jit/vmap-traceable
# port of phys/module_bl_mrf.F (savepoint-parity gated, proofs/v013/mrf_oracle.py).
# CAM-UW(9) threads its own KVM/KVH/TAURES carry: operational_mode calls
# camuw_pbl_adapter explicitly (like MYJ), so it is deliberately absent here.
PBL_SCAN_ADAPTERS = {
    1: ysu_pbl_adapter,
    3: gfs_pbl_adapter,
    11: shinhong_pbl_adapter,
    12: gbm_pbl_adapter,
    7: acm2_pbl_adapter,
    8: boulac_pbl_adapter,
    99: mrf_pbl_adapter,
}


# --------------------------------------------------------------------------- #
# v0.3.4 (lane o1-nlbind): fp64 islands for MP/PBL adapters that cannot TRACE under the
# release REAL32 carry.  Their column kernels are fp64-savepoint-parity ports; fed
# mixed f32 State leaves and f64 metrics (dz from ph.astype(f64)) their internal
# scan/cond/while carries change dtype and tracing fails (Swiss RD11 CLI probe,
# proofs/o1_nlbind).  The island widens the State to f64 for the call and writes
# every leaf back at its carry dtype.  Without the REAL carry (theta not f32,
# e.g. the legacy fp64 path) it calls the adapter unchanged.  Not WRF REAL arithmetic: a
# documented opt-in precision island for non-release schemes.
# --------------------------------------------------------------------------- #
def _fp64_island(adapter):
    def island(state: State, dt: float, grid=None, **kwargs) -> State:
        if state.theta.dtype != jnp.float32:  # not the REAL carry: original program
            return adapter(state, dt, grid, **kwargs)
        from gpuwrf.contracts.cammg_fields import STATE_FIELDS as cammg_real_carry
        narrow = {
            name: value.dtype
            for name in state.__slots__
            if name not in cammg_real_carry
            and (value := getattr(state, name, None)) is not None
            and getattr(value, "dtype", None) == jnp.float32
        }
        wide = state.replace(_cast=False, **{n: getattr(state, n).astype(jnp.float64) for n in narrow})
        out = adapter(wide, dt, grid, **kwargs)
        side = ()
        if isinstance(out, tuple):  # (State, side outputs), e.g. return_precipitation=True
            out, *side = out
        # Leaves the scheme did not replace get their ORIGINAL object back: callers
        # detect what a scheme changed by identity (operational_mode spec-zone
        # restriction), and an untouched leaf must not round-trip.
        out = out.replace(_cast=False, **{
            n: getattr(state, n) if getattr(out, n) is getattr(wide, n) else getattr(out, n).astype(d)
            for n, d in narrow.items()})
        return (out, *side) if side else out

    island.__wrapped__ = adapter
    island.__name__ = f"fp64_island_{getattr(adapter, '__name__', 'adapter')}"
    return island


# O2 (o2-rootbdy): + 24 WSM7 -- reached the same _nislfv carry-dtype failure once its root qh
# boundary was routed (WSM6 sedimentation helpers; Swiss probe proofs/o2_rootbdy).
FP64_ISLAND_MP_CODES: tuple[int, ...] = (1, 2, 3, 4, 6, 10, 13, 14, 16, 24, 26, 97)
for _code in FP64_ISLAND_MP_CODES:
    MP_SCAN_ADAPTERS[_code] = _fp64_island(MP_SCAN_ADAPTERS[_code])
# Same failure class for three PBL column kernels (ACM2, BouLac _pbl_height scan, GBM).
FP64_ISLAND_PBL_CODES: tuple[int, ...] = (7, 8, 12)
def _pbl_surface_island(adapter):
    legacy = _fp64_island(adapter)
    def island(state,dt,grid=None,*,surface_forcing=None,_with_sources=False,**kwargs):
        if surface_forcing is None:
            return legacy(state,dt,grid,**kwargs)
        # Selected SF1 supplies consistently REAL phy_prep operands. Preserve
        # the legacy precision bridge for callers without this handoff.
        with jax.enable_x64(False):
            return adapter(state,dt,grid,surface_forcing=surface_forcing,_with_sources=_with_sources,**kwargs)
    island.__wrapped__ = adapter
    island.__name__ = legacy.__name__
    return island

for _code in FP64_ISLAND_PBL_CODES:
    PBL_SCAN_ADAPTERS[_code] = _pbl_surface_island(PBL_SCAN_ADAPTERS[_code])
del _code


__all__ = [
    "kessler_adapter",
    "lin_adapter",
    "sbu_ylin_adapter",
    "wsm3_adapter",
    "wsm5_adapter",
    "wsm6_adapter",
    "morrison_adapter",
    "morrison_aero_adapter",
    "wdm5_adapter",
    "wdm6_adapter",
    "sfclay_revised_mm5_adapter",
    "pleim_xiu_sfclay_adapter",
    "sfclay_old_mm5_adapter",
    "gfs_sfclay_adapter",
    "kf_adapter",
    "tiedtke_adapter",
    "gf_adapter",
    "grell_cumulus_adapter",
    "g3_adapter",
    "gd_adapter",
    "bmj_adapter",
    "ysu_pbl_adapter",
    "gfs_pbl_adapter",
    "shinhong_pbl_adapter",
    "gbm_pbl_adapter",
    "camuw_pbl_adapter",
    "acm2_pbl_adapter",
    "boulac_pbl_adapter",
    "initial_kf_carry",
    "initial_bmj_carry",
    "MP_SCAN_ADAPTERS",
    "SFCLAY_SCAN_ADAPTERS",
    "CU_SCAN_ADAPTERS",
    "CU_STATELESS_SCAN_ADAPTERS",
    "PBL_SCAN_ADAPTERS",
]


_BMJ_BINDINGS = ("initial_bmj_cldefi", "step_bmj_column")


def _initialize_bmj():
    if all(name in globals() for name in _BMJ_BINDINGS):
        return
    with jax.ensure_compile_time_eval():
        from gpuwrf.physics import cumulus_bmj
    for name in _BMJ_BINDINGS:
        globals().setdefault(name, getattr(cumulus_bmj, name))


def __getattr__(name):
    if name in _BMJ_BINDINGS:
        _initialize_bmj()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def ntu_scan_adapter(state, dt, grid=None):
    from gpuwrf.physics.ntu.adapter import ntu_adapter
    return ntu_adapter(state, dt, grid)


MP_SCAN_ADAPTERS[56] = ntu_scan_adapter
