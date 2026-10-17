"""Noah-MP <-> surface/PBL/dycore coupling adapter (Sprint S6a — INTEGRATION).

The handshake that plugs prognostic Noah-MP into the existing surface->PBL->dycore
chain. LAND-ONLY MASKED: ocean/lake columns keep the current prescribed-SST bulk
``surface_layer_with_diagnostics`` path VERBATIM (byte-for-byte unchanged).

This is a NEW module. ``runtime.operational_mode.surface_adapter`` will call it
once wired; operational_mode.py is NOT edited here (perf-sidecar worktree owns
it). See ADR-NOAHMP-INTERFACES.md §4.

Per physics step (frozen sequence):
  1. sfclay first (UNCHANGED): ``surface_layer_with_diagnostics(state)`` over ALL
     columns -> CH/CM/ustar/tau + water/lake HFX/LH/QFX/T2/Q2. opt_sfc=1: sfclay
     OWNS CH/CM and feeds them INTO Noah-MP (seeded into land_state.cm/ch).
  2. Noah-MP over land: ``noah_mp_step(land_state, forcing, static, dt)`` -> land
     HFX/LH/QFX/TSK/albedo/emiss/ZNT. Noah-MP RE-DERIVES the land-tile CH/CM in its
     own VEGE_FLUX/BARE_FLUX loop (AUTHORITATIVE over land; sfclay CH is NOT forced
     onto the land tile — ADR §4 / S1).
  3. Masked blend (the ONLY land/water flux switch):
       hfx = where(is_land, noahmp.hfx, sfclay.hfx); likewise lh/qfx/tsk/znt.
     Rebuild kinematic handles (theta_flux/qv_flux/fltv) from the BLENDED flux with
     the identical formulae as surface_layer.py:710-715.
  4. PBL bottom BC: the blended SurfaceFluxes is passed as ``surface=`` into the
     MYNN column (the frozen Gate-1 hand-off). mynn_pbl is NOT changed here.
  5. Write-back: return (state', land_state') with state' carrying blended
     t_skin/roughness_m via State.replace; land_state' threaded to next step.

Invariants: no in-loop host transfer; ocean path unchanged (the where(is_land,...)
selection is the sole switch); CH/CM from sfclay over water, Noah-MP-own over land;
dycore + microphysics + State.__slots__ untouched.
"""

from __future__ import annotations

from typing import Any
import os

import jax.numpy as jnp
from jax import lax,tree_util

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.physics.noahmp.types import NoahMPForcing
from gpuwrf.physics.noahmp.precision import real_dtype, real_scalar, real_tree; from gpuwrf.kernels.layout_pin import small_grid_firewall as _small_grid_firewall
from gpuwrf.physics.surface_constants import (
    CP_D,
    EP1,
    P0_PA,
    P608,
    R_D,
    R_D_OVER_CP,
)
from gpuwrf.physics.mynn_constants import mynn_constant as _mynn_constant
from gpuwrf.physics.surface_layer import (
    _mynn_surface_exner_kwargs,
    mynn_driver_surface_fluxes,
    mynn_fltv_wrf_enabled,
    surface_layer_with_diagnostics,
)

# R_v / R_d for the WRF moist-potential-temperature decoupling. MUST match the
# constant the dycore couples with (operational_mode._RVRD = 461.6/287.0) so the
# theta_m -> theta_dry inverse is exact; see assemble_noahmp_forcing.
RVOVRD = 461.6 / R_D


def _get(obj, name, default=None):
    """Attribute or dict-style lookup with a default (tolerant accessor)."""
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


def _surface(field):
    """Lowest model level of a column field, as the 2-D surface (ny, nx).

    Column fields are stored trailing-z ``(ny, nx, nz)`` (the surface-layer/State
    convention), so the lowest level is the last-axis index 0. A field already 2-D
    is returned unchanged.
    """
    a = jnp.asarray(field, dtype=real_dtype())
    if a.ndim >= 3:
        return a[..., 0]
    return a


def assemble_noahmp_forcing(
    state: Any,
    static: NoahMPStatic,
    radiation: Any,
    clock: Any,
    dt: float,
) -> NoahMPForcing:
    """Build the Noah-MP forcing pytree from device state (no host transfer).

    Pulls the atmosphere lowest level (sfctmp/sfcprs/uu/vv/qair/qc) from ``state``,
    radiation (soldn/lwdn/cosz) from ``radiation``, the microphysics precip
    partition from ``state``/``radiation``, and the clock (julian/yearlen) from
    ``clock``. Missing optional inputs default to physically inert values (no
    precip, neutral radiation) so the adapter is usable before the operational
    radiation/precip plumbing is finalised.
    """
    sfcprs = _surface(_get(state, "p", _get(state, "sfcprs")))
    psfc = _surface(_get(state, "psfc", sfcprs))
    uu = _surface(_get(state, "u"))
    vv = _surface(_get(state, "v"))
    # WRF carries QV3D as a water-vapour mixing ratio, but the Noah-MP driver
    # hands the land model specific humidity (module_sf_noahmpdrv.F):
    #
    #   Q_ML = QV3D / (1.0 + QV3D)
    #
    # Keep the unconverted mixing ratio for the moist-theta decoupling below;
    # that WRF expression is defined in terms of the prognostic QVAPOR value.
    # Do not clamp here: the pristine driver evaluates the source expression
    # directly, including for a representable negative numerical QVAPOR.
    qv_mixing_ratio = _surface(_get(state, "qv", _get(state, "qair")))
    qair = qv_mixing_ratio / (1.0 + qv_mixing_ratio)
    # lowest-level air temperature: prescribed t_air/sfctmp if present, else from
    # theta via the Exner function at the lowest-level pressure (matches sfclay).
    t_air = _get(state, "t_air", _get(state, "sfctmp", None))
    if t_air is not None:
        sfctmp = _surface(t_air)
    else:
        # ``state.theta`` is the WRF MOIST potential temperature
        # theta_m = theta_dry * (1 + R_v/R_d * q_v) (use_theta_m=1; the dycore
        # prognostic -- see operational_mode conv_t_tendf_to_moist, which divides
        # ``before.theta`` by the SAME (1 + _RVRD*qv) to recover dry theta). WRF
        # hands noahmplsm the DRY sensible temperature T3D = t_phy =
        # theta_dry*(p/p0)^kappa (module_sf_noahmpdrv.F:755), so decouple
        # theta_m -> theta_dry BEFORE the Exner conversion. Skipping it left the
        # lowest-level air temperature ~+4 K too warm (= the (1+R_v/R_d*q_v)
        # factor), biasing the whole Noah-MP land-tile surface energy balance.
        theta_m0 = _surface(_get(state, "theta"))
        theta_dry0 = theta_m0 / (1.0 + _mynn_constant("RVOVRD", real_dtype(), RVOVRD) * qv_mixing_ratio)
        sfctmp = theta_dry0 * (jnp.maximum(sfcprs, 1.0) / P0_PA) ** R_D_OVER_CP
    qc = _surface(_get(state, "qc", None)) if _get(state, "qc", None) is not None else jnp.zeros_like(qair)
    shape = sfctmp.shape

    def rad2d(name, default):
        v = _get(radiation, name, None)
        return _surface(v) if v is not None else jnp.full(shape, real_scalar(float(default)))

    soldn = jnp.maximum(rad2d("soldn", 0.0), 0.0)
    lwdn = rad2d("lwdn", 0.0)
    cosz = rad2d("cosz", 0.0)

    def prc2d(name):
        v = _get(state, name, None)
        if v is None:
            v = _get(radiation, name, None)
        return _surface(v) if v is not None else jnp.zeros(shape, dtype=real_dtype())

    zero = jnp.zeros(shape, dtype=real_dtype())
    # v0.20.0 #91 nested-fix: do NOT ``float()`` the clock scalars. #91 threads the
    # date-derived phenology clock (``noahmp_julian``/``noahmp_yearlen``) as TRACED
    # 0-D jnp arrays so the per-step HLO is date-INDEPENDENT and the persistent
    # compile cache HITS across forecast dates. ``float()`` on a traced array raises
    # ConcretizationTypeError -- it broke the all-7 nested ``_advance_chunk_fori``
    # megacompile (#91 verified single-domain only). The downstream phenology
    # (``noahmp/phenology.py``) is fully traced-compatible (only ``jnp`` ops on
    # ``julian``/``yearlen``) and re-casts to f64, so this dtype is not load-bearing.
    # No explicit dtype: a concrete Python float (legacy ``clock_base=None`` path)
    # keeps JAX's default precision -- byte-identical to the old
    # ``jnp.asarray(float(...))`` (f64 under x64 / f32 under x64-off) and warning-free
    # in fp32 mode; a traced array preserves its own dtype from ``build_clock_base``.
    julian = real_tree(jnp.asarray(_get(clock, "julian", 1.0)))
    yearlen = real_tree(jnp.asarray(_get(clock, "yearlen", 365.0)))
    zlvl_field = _get(state, "zlvl", None)
    if zlvl_field is not None:
        zlvl = _surface(zlvl_field)
    else:
        dz_field = _get(state, "dz", None)
        dz = _surface(dz_field) if dz_field is not None else jnp.full(shape, real_scalar(100.0))
        zlvl = 0.5 * dz

    return NoahMPForcing(
        sfctmp=sfctmp, sfcprs=sfcprs, psfc=psfc, uu=uu, vv=vv, qair=qair, qc=qc,
        soldn=soldn, lwdn=lwdn,
        prcpconv=prc2d("prcpconv"), prcpnonc=prc2d("prcpnonc"),
        prcpsnow=prc2d("prcpsnow"), prcpgrpl=prc2d("prcpgrpl"), prcphail=prc2d("prcphail"),
        cosz=cosz, zlvl=zlvl, julian=julian, yearlen=yearlen,
    )


def _mynn_pbl_surface_density(
    forcing: NoahMPForcing,
    qv_mixing_ratio: Any | None = None,
):
    """Reproduce MYNN's independent mean-tendency boundary density.

    Noah-MP receives specific humidity, whereas MYNN's WRF driver expression
    uses the atmospheric QVAPOR mixing ratio.  Operational callers pass that
    original state value explicitly.  The inverse conversion is retained only
    for focused/direct callers that have a forcing object but no State view.
    """

    from gpuwrf.physics.mynn_pbl import _env_bool
    dtype = jnp.float32 if _env_bool("GPUWRF_MYNN_FP32_COLUMNS", False) else jnp.float64
    qv = (
        jnp.asarray(forcing.qair, dtype=dtype)
        / (1.0 - jnp.asarray(forcing.qair, dtype=dtype))
        if qv_mixing_ratio is None
        else jnp.asarray(qv_mixing_ratio, dtype=dtype)
    )

    return (
        jnp.asarray(forcing.psfc, dtype=dtype)
        / (
            R_D
            * (
                jnp.asarray(forcing.sfctmp, dtype=dtype)
                + _mynn_constant("P608", dtype, P608) * qv
            )
        )
    )


def noahmp_surface_adapter(
    state: Any,
    land_state: NoahMPLandState,
    static: NoahMPStatic,
    radiation: Any = None,
    clock: Any = None,
    dt: float = 1.0,
    forcing: NoahMPForcing | None = None,
    energy_params: Any = None,
    rad_params: Any = None,
    first_timestep: Any = False,
    history: bool = False,
    land_history: bool = False,
    urban: Any = None,
    lake_bundle: Any = None,
    observation_aux: bool = False,
    return_cumulus_inputs: bool = False,
    surface_diagnostics: Any = None,
    spp_pattern: Any = None,
    urban_lead_seconds: Any = None,
    surface_is_mynn: bool = False,
    surface_cpm_factor: float = 0.0,
    mmf: Any = None,
    itimestep: Any = None,
    options: Any = None,
    irrigation: Any = None,
    carbon: Any = None,
    groundwater: Any = None,
    tile_drain: Any = None,
) -> tuple[Any, NoahMPLandState, SurfaceFluxes]:
    """Run the land-masked Noah-MP / sfclay blend for one physics step.

    Returns ``(state', land_state', blended_surface_fluxes)``. Ocean/lake columns
    take the sfclay branch unchanged (the ``where(is_land,...)`` selection is the
    sole land/water switch). ``forcing`` may be supplied pre-assembled; otherwise
    it is built from ``state``/``radiation``/``clock`` via
    :func:`assemble_noahmp_forcing`.

    ``energy_params``/``rad_params`` (S6b ACTIVATE) may be supplied pre-built so the
    operational scan never re-runs the (concrete-``nroot``) ``build_energy_params``
    inside jit; when None the driver builds them itself (the eager S6a gate path).

    ``history=True`` appends the WRF history surface fields this step produced
    (blended HFX/LH; T2/Q2 = land Noah-MP blend, ISWATER bulk-flux form; sfclay U10/V10).

    ``urban=(UrbanBundle, UrbanCarry)`` (sf_urban_physics=1) runs WRF ``noahmp_urban``
    on the Noah-MP outputs (coupling.urban_slucm_hook) and appends the advanced
    UrbanCarry as the LAST return element; ``None`` leaves this path unchanged.
    """
    # ---- 1. sfclay over ALL columns (UNCHANGED formulae). ``first_timestep``
    #         engages the WRF MYNN surface FIRST-CALL semantics (UST first guess,
    #         MOL=0, land QSFC, Li_etal_2010 z/L seed) on the Noah-MP path too;
    #         without it the blend ran the warm-call branch at step 1 while the
    #         standalone surface slot ran the fixed first-call branch. ----
    # WRF surface_driver feeds the PRE-Noah snow depth to SFCLAY_mynn.
    multilayer = urban is not None and getattr(urban[0], 'option', 1) in (2, 3)
    if multilayer:
        from gpuwrf.coupling.urban_surface_layer import revised_surface_layer
        diag, urban_pbl_resistances = revised_surface_layer(state, return_pbl_resistances=True)
    else:
        diag = surface_diagnostics if surface_diagnostics is not None else surface_layer_with_diagnostics(
            state, first_timestep=first_timestep, snowh=land_state.snowh,
            **({"with_chs": True} if urban is not None else {}),
            **({"spp_pattern": spp_pattern} if spp_pattern is not None else {}),
        )
    sf = diag.fluxes                          # SurfaceFluxes (kinematic)
    obs_aux = ()
    if observation_aux:
        from gpuwrf.coupling.obs_rk import surface_observation_aux
        obs_aux = (surface_observation_aux(state, diag),)
    cumulus_aux = (jnp.asarray(diag.zol, jnp.float32),) if return_cumulus_inputs else ()
    # WRF's surface driver supplies lowest-level RHO3D to sfclay.  That density
    # is therefore the authority for converting physical HFX/QFX to the
    # kinematic handles below (module_sf_mynn.F:322,1051-1052).
    flux_density = jnp.asarray(sf.rhosfc, dtype=real_dtype())

    # is_land mask (xland: 1 land / 2 water) — identical convention to sfclay.
    xland = _surface(_get(state, "xland", jnp.ones_like(flux_density)))
    is_land = (xland - 1.5) < 0.0

    # ---- 2. Noah-MP over land. sfclay SEEDS CH/CM into the land carry; Noah-MP
    #         RE-DERIVES the authoritative land-tile CH/CM internally (ADR §4). ----
    if forcing is None:
        forcing = assemble_noahmp_forcing(state, static, radiation, clock, dt)
    forcing = real_tree(forcing)
    # MYNN does *not* retain the surface driver's RHO3D at its mean-tendency
    # boundary.  It recomputes a second density exactly as
    #   psfc / (R_d * (tk(kts) + p608*qv(kts)))
    # in module_bl_mynnedmf.F90:3960.  Keep this distinct from ``flux_density``:
    # using either density for both roles creates a deterministic surface-drag
    # error on the authenticated d03 fixture.
    # The Noah-MP forcing now correctly carries specific humidity; MYNN's
    # independent density boundary still consumes WRF QVAPOR (mixing ratio).
    state_qv = _get(state, "qv", None)
    qv_mixing_ratio = None if state_qv is None else _surface(state_qv)
    pbl_density = flux_density if multilayer else _mynn_pbl_surface_density(forcing, qv_mixing_ratio)
    # seed sfclay CH/CM (opt_sfc=1 supplies the drag coeffs Noah-MP consumes).
    land_before_surface = land_state
    # noahmpdrv.F:817-818 passes CM=CMXY, CH=CHXY (Noah-MP's own previous values);
    # SFCDIF2 (opt_sfc=2) iterates from them, so a sfclay CH/CM (e.g. QNSESFC) must
    # not overwrite them. opt_sfc=1 SFCDIF1 recomputes both (seed is inert there).
    own_sfcdif = options is not None and int(getattr(options, "opt_sfc", 1)) == 2
    ch_seed = _surface(land_state.ch if own_sfcdif else _get(diag, "ch", land_state.ch))
    cm_seed = _surface(land_state.cm if own_sfcdif else _get(diag, "cm", land_state.cm))
    land_state = land_state.replace(ch=ch_seed, cm=cm_seed); land_state, forcing, diag, sf = _small_grid_firewall((land_state, forcing, diag, sf), flux_density.shape)

    if getattr(static,'glacier',None) is not None:
        if mmf is not None:
            raise ValueError('MMF glacier coupling is not qualified; use an ice-free domain')
        from gpuwrf.coupling.glacier_surface_hook import route_glacier_with_lakes
        tile_options={key:value for key,value in (('options',options),('irrigation',irrigation),
            ('carbon',carbon),('groundwater',groundwater),('tile_drain',tile_drain)) if value is not None}
        if tile_options:
            from gpuwrf.coupling.noahmp_option_tiles import option_glacier_step
            step_result=option_glacier_step(noah_mp_step,land_state,forcing,static,dt,
                energy_params=energy_params,rad_params=rad_params,history=land_history,
                lake_bundle=lake_bundle,frozen_land=land_before_surface,**tile_options)
        else:
            step_result = route_glacier_with_lakes(noah_mp_step,land_state,forcing,static,dt,
                energy_params=energy_params,rad_params=rad_params,history=land_history,
                lake_bundle=lake_bundle,frozen_land=land_before_surface)
    elif lake_bundle is not None:
        if mmf is not None:
            raise ValueError('MMF plus the compact lake coupling is not qualified')
        if carbon is not None:raise ValueError("Dynamic Noah-MP vegetation plus compact lake tiles is not qualified")
        if groundwater is not None:raise ValueError("Noah-MP groundwater plus compact lake tiles is not qualified")
        if tile_drain is not None:raise ValueError("Noah-MP tile drainage plus compact lake tiles is not qualified")
        if irrigation is not None:
            raise ValueError("Noah-MP irrigation plus compact lake tiles is not qualified")
        from gpuwrf.coupling.lake_surface_hook import noahmp_without_lakes
        from functools import partial
        land_kernel = noah_mp_step if options is None else partial(noah_mp_step, options=options)
        step_result = noahmp_without_lakes(land_kernel,land_state,forcing,static,dt,lake_bundle,
            energy_params=energy_params,rad_params=rad_params,history=land_history,frozen_land=land_before_surface)
    else:
        step_result = noah_mp_step(
            land_state, forcing, static, dt,
            energy_params=energy_params, rad_params=rad_params,
            **({"history": True} if land_history else {}),
            **({'mmf': mmf} if mmf is not None else {}),
            **({"options": options} if options is not None else {}),
            **({"irrigation": irrigation} if irrigation is not None else {}),
            **({"carbon":carbon} if carbon is not None else {}),
            **({"groundwater":groundwater} if groundwater is not None else {}),
            **({"tile_drain":tile_drain} if tile_drain is not None else {}),
        )
    if mmf is not None:*step_result,mmf=step_result
    if tile_drain is not None:*step_result,tile_out=step_result
    if groundwater is not None:*step_result,groundwater_out=step_result
    if carbon is not None:*step_result,carbon_out=step_result
    if irrigation is not None:*step_result,irrigation_out=step_result
    land_state_out,nm=step_result
    if mmf is not None:
        if itimestep is None:raise ValueError('MMF needs the absolute WRF itimestep for STEPWTD cadence')
        from gpuwrf.physics.noahmp.groundwater import advance_lateral
        mmf,land_state_out=advance_lateral(mmf,land_state_out,static,itimestep)
    mmf_aux=() if mmf is None else (mmf,)
    # module_sf_noahmpdrv.F:732 skips ocean columns. Retain the incoming
    # option carries there, just as the surface state blend does below.
    hold_water=lambda new,old:tree_util.tree_map(lambda a,b:jnp.where(is_land,a,b),new,old)
    if irrigation is not None:irrigation_out=hold_water(irrigation_out,irrigation[1])
    if carbon is not None:carbon_out=hold_water(carbon_out,carbon[1])
    if groundwater is not None:groundwater_out=hold_water(groundwater_out,groundwater)
    if tile_drain is not None:tile_out=jnp.where(is_land,tile_out,tile_drain)
    irrigation_aux=() if irrigation is None else (irrigation_out,)
    carbon_aux=() if carbon is None else (carbon_out,)
    groundwater_aux=() if groundwater is None else (groundwater_out,)
    tile_aux=() if tile_drain is None else (tile_out,)
    urban_carry_out = urban_step = None
    if urban is not None:
        # WRF noahmp_urban (after noahmplsm): blended TSK/HFX/QFX/LH/GRDFLX/ALBEDO/QSFC/UST
        # become the grid fields MYNN, radiation and the next Noah-MP call read.
        if getattr(urban[0], 'option', 1) in (2, 3):
            from gpuwrf.coupling.urban_multilayer_hook import apply_urban_multilayer
            nm, ust_urban, urban_carry_out, urban_step = apply_urban_multilayer(
                urban[0], urban[1], atmosphere=urban[2], forcing=forcing, nm=nm, ust=sf.ustar, dt=dt)
        else:
            from gpuwrf.coupling.urban_slucm_hook import apply_urban_slucm
            nm, ust_urban, urban_carry_out, urban_step = apply_urban_slucm(
                urban[0], urban[1], forcing=forcing, nm=nm, ust=sf.ustar, chs=diag.chs,
                chs2=diag.chs2, cqs2=diag.cqs2, qv_mixing_ratio=qv_mixing_ratio, dt=dt,
                **({"lead_seconds":urban_lead_seconds} if urban[0].requires_inputs else {}))
        sf = sf._replace(ustar=ust_urban)
        if multilayer:
            from gpuwrf.coupling.urban_surface_layer import UrbanSurfaceLayer
            idx=jnp.asarray(urban[0].indices,jnp.int32);shape=diag.u10.shape
            profile=urban[2][0]
            u10_bep=diag.u10.reshape(-1).at[idx].set(profile.u[:,0]).reshape(shape)
            v10_bep=diag.v10.reshape(-1).at[idx].set(profile.v[:,0]).reshape(shape)
            urban_carry_out = urban_carry_out._replace(surface_layer=UrbanSurfaceLayer(
                diag.br,urban_pbl_resistances[0],urban_pbl_resistances[1],sf.wspd,u10_bep,v10_bep))
        land_state_out = land_state_out.replace(
            t_skin=jnp.asarray(nm.tsk, land_state_out.t_skin.dtype),
            qsfc=jnp.asarray(nm.qsfc, land_state_out.qsfc.dtype),
            albedo=jnp.asarray(nm.albedo, land_state_out.albedo.dtype))
        if getattr(urban[0], 'option', 1) in (2, 3):
            land_state_out = land_state_out.replace(emiss=jnp.asarray(nm.emiss, land_state_out.emiss.dtype))

    # ---- 3. masked blend (land vs water). Water path = sfclay diagnostics. ----
    # rho*cpm with the WRF MYNN moist heat capacity (surface_layer.py): cpm =
    # CP_D*(1+0.84*qx) (module_sf_mynn.F:552). theta_flux = hfx/(rho*cpm) inverts the
    # MYNN-SL flux mapping, so the coefficient MUST match surface_layer.py (0.84).
    qx = jnp.maximum(_surface(_get(state, "qv", jnp.zeros_like(flux_density))), 0.0)
    rho_cpm = (flux_density * CP_D if surface_diagnostics is not None and not multilayer
               else flux_density * (CP_D * (1.0 + (0.8 if multilayer else 0.84) * qx)))
    if surface_cpm_factor:
        rho_cpm=flux_density*(CP_D*(1.+surface_cpm_factor*qx))

    # blended physical fluxes (W/m2 and kg/m2/s)
    hfx_water = jnp.asarray(diag.hfx, dtype=real_dtype())
    lh_water = jnp.asarray(diag.lh, dtype=real_dtype())
    qfx_water = flux_density * jnp.asarray(sf.qv_flux, dtype=real_dtype())
    znt_water = jnp.asarray(diag.znt, dtype=real_dtype())
    tsk_water = _surface(_get(state, "t_skin", jnp.asarray(nm.tsk)))

    hfx = jnp.where(is_land, jnp.asarray(nm.hfx), hfx_water)
    if urban is not None and getattr(urban[0],"ahe_opt",0) == 2:
        from gpuwrf.coupling.urban_inputs import ahe_at_lead
        # surface_driver:4588, AFTER urban and land/water blends; no FRC scaling.
        hfx_before_ahe = hfx  # ACHFX accumulation precedes distributed AHE (:3432).
        hfx = hfx+ahe_at_lead(urban[0],urban[1].inputs,urban_lead_seconds)
    qfx = jnp.where(is_land, jnp.asarray(nm.qfx), qfx_water)
    znt = jnp.where(is_land, jnp.asarray(nm.znt), znt_water)
    tsk = jnp.where(is_land, jnp.asarray(nm.tsk), tsk_water)
    # blended LH (diagnostic; the PBL bottom BC is rebuilt from HFX/QFX below).
    lh = jnp.where(is_land, jnp.asarray(nm.lh), lh_water)  # noqa: F841

    # ---- 4. rebuild kinematic handles from the BLENDED flux (surface_layer.py
    #         :710-715), so the MYNN bottom BC sees the land flux over land. ----
    thx = jnp.asarray(_thx(state, flux_density), dtype=real_dtype())
    theta_flux = hfx / jnp.maximum(rho_cpm, 1.0e-12)
    qv_flux = qfx / jnp.maximum(flux_density, 1.0e-12)
    ep1 = _mynn_constant("P608", real_dtype(), EP1)
    fltv = (1.0 + ep1 * qx) * theta_flux + ep1 * thx * qv_flux
    if (surface_diagnostics is None or surface_is_mynn) and mynn_fltv_wrf_enabled() and not multilayer:
        # WRF's MYNN derives flt/flqv/fltv itself from the blended HFX/QFX, TSK and the lowest-level
        # p_phy (module_bl_mynnedmf.F:859-876); same helper as the surface-layer producer.
        r = real_dtype()
        p1 = jnp.asarray(_surface(_get(state, "p", jnp.full_like(flux_density, 1.0e5))), dtype=r)
        mynn = mynn_driver_surface_fluxes(jnp.asarray(hfx, dtype=r), jnp.asarray(qfx, dtype=r), flux_density,
                                          qx, jnp.asarray(tsk, dtype=r), p1,
                                          **_mynn_surface_exner_kwargs(state, r))
        theta_flux, qv_flux, fltv = mynn.flt, mynn.flqv, mynn.fltv

    blended = SurfaceFluxes(
        ustar=sf.ustar,            # momentum: sfclay owns ustar/tau everywhere (opt_sfc=1)
        theta_flux=theta_flux,
        qv_flux=qv_flux,
        tau_u=sf.tau_u,
        tau_v=sf.tau_v,
        rhosfc=pbl_density,
        fltv=fltv,
        xland=sf.xland,            # carry land/sea mask through to MYNN mixing length
        wspd=sf.wspd,              # WRF Noah-MP does not overwrite SFCLAY WSPD.
    )

    # ---- 5. state write-back (blended t_skin / roughness_m). State.__slots__
    #         unchanged; qsfc written only if the State carries that slot. ----
    updates = {"t_skin": _broadcast_like(state, "t_skin", tsk),
               "roughness_m": _broadcast_like(state, "roughness_m", znt)}
    if getattr(state,"sms_br",None) is not None:
        updates.update(sms_br=_broadcast_like(state,"sms_br",diag.br),
                       sms_rmol=_broadcast_like(state,"sms_rmol",diag.rmol),
                       sms_wspd=_broadcast_like(state,"sms_wspd",diag.fluxes.wspd),
                       sms_ustm=_broadcast_like(state,"sms_ustm",diag.ustm))
    if hasattr(state, "qsfc"):
        # WRF grid%QSFC after the surface driver: Noah-MP QSFC1D over land
        # (module_sf_noahmpdrv.F:1244), the sfclay value over water.
        updates["qsfc"] = _broadcast_like(
            state, "qsfc", jnp.where(is_land, jnp.asarray(nm.qsfc), jnp.asarray(diag.qsfc)))
    # B39: grid%MOL (sfclay) and the BLENDED physical HFX/QFX (Noah-MP land,
    # sfclay water) are the next step's SFCLAY_mynn WSTAR / z/L-seed inputs.
    if hasattr(state, "mol"):
        updates["mol"] = _broadcast_like(state, "mol", jnp.asarray(diag.mol))
    if hasattr(state, "hfx"):
        updates["hfx"] = _broadcast_like(state, "hfx", hfx)
    if hasattr(state, "qfx"):
        updates["qfx"] = _broadcast_like(state, "qfx", qfx)
    state_out = state.replace(**updates) if hasattr(state, "replace") else state

    if history or land_history:
        # WRF Noah-MP path (module_surface_driver.F:3446-3472): ISWATER points take the
        # bulk-flux 2-m form with rho = PSFC/(R_d*TSK); land takes the Noah-MP FVEG blend.
        t2_water, q2_water = _wrf_water_2m(
            tsk_water, _get(state, "psfc", forcing.psfc), hfx_water, qfx_water,
            diag.qsfc, diag.chs2, diag.cqs2)
        iswater = jnp.asarray(static.ivgtyp, dtype=jnp.int32) == int(static.parameters.iswater)
        t2 = jnp.where(is_land, jnp.asarray(nm.t2),
                       jnp.where(iswater, t2_water, jnp.asarray(diag.t2, dtype=real_dtype())))
        q2 = jnp.where(is_land, jnp.asarray(nm.q2),
                       jnp.where(iswater, q2_water, jnp.asarray(diag.q2, dtype=real_dtype())))
        u10, v10 = diag.u10, diag.v10
        if urban_step is not None:
            if getattr(urban[0], 'option', 1) in (2, 3):
                from gpuwrf.coupling.urban_multilayer_hook import multilayer_history_overwrite as urban_history_overwrite
            else:
                from gpuwrf.coupling.urban_slucm_hook import urban_history_overwrite
            t2, q2, u10, v10 = urban_history_overwrite(t2, q2, u10, v10, urban_step,
                                                       _get(state, "psfc", forcing.psfc))
        q2 = _land_q2_postlsm(q2, _surface(_get(state, "qv")), is_land)
        fields = {"hfx": hfx, "lh": lh, "t2": t2, "q2": q2, "u10": u10, "v10": v10}
        if land_history and urban is not None and getattr(urban[0],"ahe_opt",0) == 2:
            fields["accumulator_hfx"] = hfx_before_ahe
        if lake_bundle is not None:
            fields.update(grdflx=jnp.where(is_land,nm.grdflx,0.),albedo=land_state_out.albedo)
        if land_history:
            fields["land_history"] = {name: jnp.where(is_land, value, 0)
                                      for name, value in nm.history.items()}
            fields["grdflx"] = jnp.where(is_land, nm.grdflx, 0)
        if urban is not None:
            return (state_out, land_state_out, blended, fields, urban_carry_out) + irrigation_aux + carbon_aux + groundwater_aux + tile_aux + mmf_aux + obs_aux + cumulus_aux
        return (state_out, land_state_out, blended, fields) + irrigation_aux + carbon_aux + groundwater_aux + tile_aux + mmf_aux + obs_aux + cumulus_aux
    if urban is not None:
        return (state_out, land_state_out, blended, urban_carry_out) + irrigation_aux + carbon_aux + groundwater_aux + tile_aux + mmf_aux + obs_aux + cumulus_aux
    return (state_out, land_state_out, blended) + irrigation_aux + carbon_aux + groundwater_aux + tile_aux + mmf_aux + obs_aux + cumulus_aux


def _land_q2_postlsm(q2, qv_mixing_ratio, is_land):
    """Original surface_driver post-LSM cap uses this call's QV_CURR(kts)."""
    if os.environ.get("GPUWRF_LAND_Q2_CAP_POSTLSM", "0") != "1":
        return q2
    limit = jnp.asarray(1.05, dtype=q2.dtype) * jnp.asarray(qv_mixing_ratio, dtype=q2.dtype)
    # The completed post-LSM cap leaves the surface/land producers outside its carry export.
    return lax.optimization_barrier(jnp.where(is_land, jnp.minimum(q2, limit), q2))


def _wrf_water_2m(tsk, psfc, hfx, qfx, qsfc, chs2, cqs2):
    """WRF Noah-MP-path 2-m T2/Q2 over ISWATER (module_surface_driver.F:3446-3458).

    Q2 = QSFC - QFX/(rho*CQS2) and T2 = TSK - HFX/(rho*CP*CHS2) with
    rho = PSFC/(R_d*TSK); CQS2/CHS2 below 1e-5 fall back to QSFC/TSK.
    """
    def r(x):
        return jnp.asarray(x, dtype=real_dtype())

    tsk, psfc, hfx, qfx, qsfc, chs2, cqs2 = map(r, (tsk, psfc, hfx, qfx, qsfc, chs2, cqs2))
    rho = psfc / (R_D * tsk)
    q2 = jnp.where(cqs2 < 1.0e-5, qsfc, qsfc - qfx / (rho * jnp.maximum(cqs2, 1.0e-5)))
    t2 = jnp.where(chs2 < 1.0e-5, tsk, tsk - hfx / (rho * CP_D * jnp.maximum(chs2, 1.0e-5)))
    return t2, q2


def _thx(state, rhosfc):
    """Lowest-level potential temperature (theta), as a 2-D field for fltv."""
    th = _get(state, "theta", None)
    if th is not None:
        return _surface(th)
    return jnp.full_like(rhosfc, 300.0)


def _broadcast_like(state, name, value):
    """Match a 2-D blended field to the State field's stored shape (t_skin/znt are
    2-D surface fields, so this is normally an identity / reshape)."""
    cur = _get(state, name, None)
    if cur is None:
        return value
    cur = jnp.asarray(cur)
    if cur.shape == value.shape:
        return value
    return value.reshape(cur.shape) if value.size == cur.size else jnp.broadcast_to(value, cur.shape)


__all__ = ["assemble_noahmp_forcing", "noahmp_surface_adapter"]
