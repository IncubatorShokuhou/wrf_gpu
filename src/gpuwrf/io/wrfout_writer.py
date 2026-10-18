"""Minimal WRF-compatible NetCDF wrfout writer for M7 output handoff."""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass, replace as dataclass_replace
from datetime import date, datetime
from math import cos, pi
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

from gpuwrf.contracts.moisture import total_moist_mass
from gpuwrf.contracts.physics_registry import MOIST_SPECIES, MOIST_WRFOUT_NAME

from gpuwrf.io.wrfout_theta_convention import parse_use_theta_m, thm_source
from gpuwrf.physics.surface_constants import CP_D, KARMAN, P0_PA, R_D_OVER_CP, XLV
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

P0_THETA_OFFSET_K = 300.0
# WRF rvovrd = R_v/R_d (share/module_model_constants.F:41). Operational
# State.theta is MOIST theta_m (use_theta_m=1); WRF wrfout variable ``T`` is
# the DRY perturbation theta, so the writer decouples
# theta_dry = theta_m / (1 + rvovrd*qv) and emits theta_m itself as ``THM``.
RVOVRD = 461.6 / 287.0
RVOVRD_REAL = np.float32(np.float32(461.6) / np.float32(287.0))
_WRITER_RVOVRD_REAL = os.environ.get("GPUWRF_WRITER_RVOVRD_REAL", "0") == "1"
CP_AIR_J_KG_K = CP_D
LV_J_KG = XLV
DATE_STR_LEN = 19
DEFAULT_SOIL_LAYERS = 4
DEFAULT_SNOW_LAYERS = 3
DEFAULT_SNSO_LAYERS = 7
DEFAULT_SEED_DIM = 8
_WRFOUT_HOST_ARRAYS: contextvars.ContextVar["_WrfoutHostArrays | None"] = (
    contextvars.ContextVar("_WRFOUT_HOST_ARRAYS", default=None)
)

WRFOUT_DOMAIN_AUTHORITY_SCHEMA = "gpuwrf.wrfout-domain-authority.v1"
_WRF_DOMAIN_RE = re.compile(r"d([0-9]{2})\Z")


@dataclass(frozen=True)
class WrfoutDomainAuthority:
    """Immutable binding from authenticated run-grid metadata to WRF ``GRID_ID``.

    ``domain`` is supplied independently by the integration scheduler.  The
    source grid is the domain metadata loaded from the already-authenticated run,
    while ``mass_*`` binds that authority to the actual output ``GridSpec``.  No
    output filename/path or field shape participates in selecting the ID.
    """

    schema: str
    domain: str
    grid_id: int
    mass_nx: int
    mass_ny: int
    mass_nz: int
    e_we: int
    e_sn: int
    e_vert: int
    authority_sha256: str


def _domain_authority_payload(
    *,
    domain: str,
    grid_id: int,
    mass_nx: int,
    mass_ny: int,
    mass_nz: int,
    e_we: int,
    e_sn: int,
    e_vert: int,
) -> dict[str, Any]:
    return {
        "schema": WRFOUT_DOMAIN_AUTHORITY_SCHEMA,
        "domain": domain,
        "grid_id": grid_id,
        "mass_nx": mass_nx,
        "mass_ny": mass_ny,
        "mass_nz": mass_nz,
        "e_we": e_we,
        "e_sn": e_sn,
        "e_vert": e_vert,
    }


def _domain_authority_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _strict_wrf_domain_id(domain: Any) -> int:
    if type(domain) is not str:  # bool/int coercion is not domain authority.
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_DOMAIN_TYPE")
    match = _WRF_DOMAIN_RE.fullmatch(domain)
    if match is None:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_DOMAIN_FORMAT")
    grid_id = int(match.group(1))
    if grid_id < 1:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_GRID_ID_RANGE")
    return grid_id


def bind_wrfout_domain_authority(
    domain: str,
    authenticated_grid: Any,
    output_grid: Any,
) -> WrfoutDomainAuthority:
    """Bind scheduler domain + authenticated run grid to an output grid.

    ``authenticated_grid`` must expose the exact domain ID and mass extents from
    run authority (``Gen2GridSpec`` in production).  There is deliberately no
    fallback to the target name, path, output arrays, or a default domain.
    """

    # Exact type is intentional: a duck-typed object can self-assert arbitrary
    # extents and is not the authenticated run-grid record.
    from gpuwrf.io.gen2_accessor import Gen2GridSpec

    if type(authenticated_grid) is not Gen2GridSpec:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_SOURCE_TYPE")
    grid_id = _strict_wrf_domain_id(domain)
    if authenticated_grid.id != domain:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_SOURCE_ID_MISMATCH")
    source_extents = (
        getattr(authenticated_grid, "mass_nx", None),
        getattr(authenticated_grid, "mass_ny", None),
        getattr(authenticated_grid, "mass_nz", None),
    )
    if any(type(value) is not int or value < 1 for value in source_extents):
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_SOURCE_EXTENTS")
    staggered_extents = (
        authenticated_grid.e_we,
        authenticated_grid.e_sn,
        authenticated_grid.e_vert,
    )
    if any(type(value) is not int or value < 2 for value in staggered_extents):
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_SOURCE_STAGGERED_EXTENTS")
    expected_staggered = tuple(value + 1 for value in source_extents)
    if staggered_extents != expected_staggered:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_SOURCE_EXTENT_RELATION")
    output_extents = _grid_extent(output_grid)
    if tuple(source_extents) != tuple(output_extents):
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_OUTPUT_GRID_MISMATCH")
    payload = _domain_authority_payload(
        domain=domain,
        grid_id=grid_id,
        mass_nx=source_extents[0],
        mass_ny=source_extents[1],
        mass_nz=source_extents[2],
        e_we=staggered_extents[0],
        e_sn=staggered_extents[1],
        e_vert=staggered_extents[2],
    )
    return WrfoutDomainAuthority(
        **payload, authority_sha256=_domain_authority_sha256(payload)
    )


def _validate_wrfout_domain_authority(
    authority: WrfoutDomainAuthority | None,
    *,
    expected_domain: str | None,
    output_grid: Any,
    dimensions: Mapping[str, int | None] | None = None,
) -> WrfoutDomainAuthority:
    if type(authority) is not WrfoutDomainAuthority:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_MISSING_OR_TYPE")
    expected_id = _strict_wrf_domain_id(expected_domain)
    if authority.schema != WRFOUT_DOMAIN_AUTHORITY_SCHEMA:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_SCHEMA")
    if authority.domain != expected_domain:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_DOMAIN_MISMATCH")
    if type(authority.grid_id) is not int or authority.grid_id != expected_id:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_GRID_ID_MISMATCH")
    extents = (authority.mass_nx, authority.mass_ny, authority.mass_nz)
    if any(type(value) is not int or value < 1 for value in extents):
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_EXTENTS")
    staggered = (authority.e_we, authority.e_sn, authority.e_vert)
    if any(type(value) is not int or value < 2 for value in staggered):
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_STAGGERED_EXTENTS")
    if staggered != tuple(value + 1 for value in extents):
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_EXTENT_RELATION")
    if extents != _grid_extent(output_grid):
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_OUTPUT_GRID_MISMATCH")
    if dimensions is not None:
        written_staggered = (
            dimensions.get("west_east_stag"),
            dimensions.get("south_north_stag"),
            dimensions.get("bottom_top_stag"),
        )
        if written_staggered != staggered:
            raise ValueError("WRFOUT_DOMAIN_AUTHORITY_OUTPUT_DIMENSION_MISMATCH")
    payload = _domain_authority_payload(
        domain=authority.domain,
        grid_id=authority.grid_id,
        mass_nx=authority.mass_nx,
        mass_ny=authority.mass_ny,
        mass_nz=authority.mass_nz,
        e_we=authority.e_we,
        e_sn=authority.e_sn,
        e_vert=authority.e_vert,
    )
    if authority.authority_sha256 != _domain_authority_sha256(payload):
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_SHA256")
    return authority


DOWNSTREAM_CRITICAL_VARIABLES: tuple[str, ...] = (
    "Times",
    "XTIME",
    "XLAT",
    "XLONG",
    "HGT",
    "LANDMASK",
    "LU_INDEX",
    "U10",
    "V10",
    "T2",
    "Q2",
    "PSFC",
    "RAINC",
    "RAINNC",
    "RAINSH",
    "SWDOWN",
    "GLW",
    "PBLH",
    "UST",
    "HFX",
    "LH",
    "TSK",
    "CLDFRA",
    "QCLOUD",
    "QICE",
    "QRAIN",
)

MINIMUM_WRFOUT_VARIABLES: tuple[str, ...] = (
    *DOWNSTREAM_CRITICAL_VARIABLES,
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
    "U",
    "V",
    "W",
    "T",
    "THM",
    "QVAPOR",
    "P",
    "PB",
    "PH",
    "PHB",
    "MU",
    "MUB",
)

# --- P0-5a operational completeness additions (closing the gap to WRF's field
# list for the Canary operational product). Every name below is a real WRF-ARW
# wrfout variable (verified against
# <DATA_ROOT>/canairy_meteo/runs/wrf_l3/.../wrfout_d02_*): names, units, dims,
# staggering, and descriptions are copied from the reference file -- no invented
# fields. They are populated ONLY from real model state / grid metrics /
# operational diagnostics; a field with no available source is left out (it does
# NOT silently emit zeros for a quantity we do not actually compute). See
# proofs/p0_5/FINDINGS.md for the field-by-field source/justification table.

# (a) Always-available: extra hydrometeors + number concentrations (Thompson
#     prognostic state) + the MYNN TKE. Source = prognostic State leaves.
MICROPHYSICS_EXTRA_VARIABLES: tuple[str, ...] = (
    "QSNOW",
    "QGRAUP",
    "QNICE",
    "QNRAIN",
    "QNSNOW",
    "QNGRAUPEL",
    "QNCLOUD",
    "QNCCN",
    # v0.17 ADR-032 graupel/hail substrate (State qh/Nh/qvolg/qvolh); emitted
    # only when the source leaves carry hail (optional-extra; a non-hail run
    # never gets a fabricated hail field).
    "QHAIL",
    "QNHAIL",
    "QVGRAUPEL",
    "QVHAIL",
    # v0.16 aerosol-aware Thompson (mp=28) prognostic aerosol numbers
    # (State nwfa/nifa); emitted only when the source leaves are present.
    "QNWFA",
    "QNIFA",
    "QIR", "QIB", "QICE2", "QNICE2", "QIR2", "QIB2", "QZI",
    "QKE",
    # Optional LES prognostic, distinct from the PBL's QKE/TKE_PBL.
    "TKE",
)

# (b) Grid-static coordinate / projection / map-factor / Coriolis fields. Source
#     = GridSpec.metrics + GridSpec.vertical (device-resident static arrays). WRF
#     repeats these every history frame; they let downstream tools reconstruct
#     the projection, eta coordinate, and Coriolis terms without a separate
#     geogrid/wrfinput read.
GRID_COORDINATE_VARIABLES: tuple[str, ...] = (
    "ZNU",
    "ZNW",
    "ZS",
    "DZS",
    "MAPFAC_M",
    "MAPFAC_U",
    "MAPFAC_V",
    "F",
    "E",
    "SINALPHA",
    "COSALPHA",
    "XLAND",
    "P_TOP",
)

# (b2) Extended grid-static vertical-coordinate + map-factor metric arrays.
#     v0.12.0 A1: PURE PAYLOAD of arrays ALREADY resident on ``DycoreMetrics`` /
#     the eta column -- no recompute, no physics. Vertical eta-spacing metrics
#     (DN/DNW/RDN/RDNW/FNM/FNP), the hybrid-eta C-coefficients (C1H..C4H,
#     C1F..C4F), the scalar extrapolation constants (CF1/CF2/CF3 already +
#     CFN/CFN1), the directional map factors (MAPFAC_MX/MY/UX/UY/VX/VY ==
#     msftx/msfty/msfux/msfuy/msfvx/msfvy), and the inverse grid lengths
#     (RDX/RDY) from the projection. Each is emitted ONLY when its source array
#     is present on the grid (dict/synthetic grids carry no metrics -> none are
#     written), so no field is ever fabricated. Names/dims/units/descriptions are
#     copied from the reference wrfout_d02 (375-var operational WRF file).
GRID_METRIC_EXTRA_VARIABLES: tuple[str, ...] = (
    "DN",
    "DNW",
    "RDN",
    "RDNW",
    "FNM",
    "FNP",
    "CFN",
    "CFN1",
    "CF1",
    "CF2",
    "CF3",
    "C1H",
    "C2H",
    "C3H",
    "C4H",
    "C1F",
    "C2F",
    "C3F",
    "C4F",
    "MAPFAC_MX",
    "MAPFAC_MY",
    "MAPFAC_UX",
    "MAPFAC_UY",
    "MAPFAC_VX",
    "MAPFAC_VY",
    "RDX",
    "RDY",
)

# (c) Accumulated precipitation partition (grid-scale snow/ice + graupel). Source
#     = State precip accumulators (coupling.physics_couplers writes snow_acc /
#     graupel_acc / ice_acc each microphysics step). WRF SNOWNC carries snow+ice.
PRECIP_PARTITION_VARIABLES: tuple[str, ...] = (
    "SNOWNC",
    "GRAUPELNC",
)

# (h) Trivially-derived diagnostics (v0.12.0 A2). WRF-FAITHFUL closed forms from
#     fields the writer already has -- NO physics, NO GPU. Each self-gates on the
#     availability of its inputs:
#       CLAT   = supplied computational latitude (initial nest feedback may
#                change it); XLAT is the legacy fallback when absent.
#       COSZEN = WRF radconst/calc_coszen cosine solar zenith from XLAT/XLONG +
#                the forecast clock (run_start + lead) -- the exact transcription
#                already in coupling.physics_couplers._compute_coszen.
#       SR     = frozen-precipitation fraction = solid_acc/(solid_acc + rain_acc),
#                from the State precip accumulators (snow+graupel+ice vs rain).
#       SNOWC  = snow-cover flag (1 where SWE > 0) from the land carry's bulk SWE.
DERIVED_DIAGNOSTIC_VARIABLES: tuple[str, ...] = (
    "CLAT",
    "COSZEN",
    "SR",
    "SNOWC",
)

# (d) Surface energy/moisture-budget fluxes routed from the operational
#     diagnostics map (QFX/GRDFLX) plus the 2-m potential temperature TH2.
#     QFX/GRDFLX appear only when the caller supplies them via ``diagnostics``
#     (see the operational_mode hook spec in proofs/p0_5/FINDINGS.md); TH2 is
#     always derivable from T2 + PSFC. No invented surface flux.
SURFACE_FLUX_EXTRA_VARIABLES: tuple[str, ...] = (
    "QFX",
    "GRDFLX",
    "TH2",
)

# (e) Prognostic Noah-MP soil/snow land columns + land diagnostics. Source = the
#     optional ``land_state`` (NoahMPLandState) handed to the writer; WRF-faithful
#     4-layer soil + bulk snow. Absent (not written) when no land carry is
#     supplied -- the writer never fabricates a soil profile.
LAND_SOIL_VARIABLES: tuple[str, ...] = (
    "TSLB",
    "SMOIS",
    "SH2O",
    "SNOW",
    "SNOWH",
    "CANWAT",
    "SFROFF",
    "UDROFF",
    "ALBEDO",
    "EMISS",
)

# (f) Noah-MP internal snow-layer + canopy diagnostics. Source = the optional
# ``NoahMPLandState`` handed to the writer; absent when no land carry exists.
# The four 3-D snow-column fields (TSNO/SNICE/SNLIQ/ZSNSO) plus the scalar
# snow/canopy prognostics ISNOW/SNEQVO/CANLIQ/CANICE are all genuine device-
# resident slots of the carry (contracts/noahmp_state.py) — each self-gates on a
# present source so a partial carry never fabricates a profile.
LAND_SNOW_DIAGNOSTIC_VARIABLES: tuple[str, ...] = (
    "TSNO",
    "SNICE",
    "SNLIQ",
    "ZSNSO",
    "ISNOW",
    "SNEQVO",
    "CANLIQ",
    "CANICE",
)

# (g) Restart seed arrays for WRF stochastic perturbation options. The Canary
# supported namelist has stochastic physics disabled, so these are emitted only
# when an explicit caller-provided source is present (for example diagnostics).
STOCHASTIC_SEED_VARIABLES: tuple[str, ...] = (
    "ISEEDARR_SPPT",
    "ISEEDARR_SKEBS",
    "ISEEDARRAY_SPP_CONV",
    "ISEEDARRAY_SPP_PBL",
    "ISEEDARRAY_SPP_LSM",
)

# (h) B1 (v0.12.0) RRTMG up/down all-sky radiation flux diagnostics. Source =
# the operational radiation diagnostics map (``M9Diagnostics`` -> wrfout via
# ``diagnostics``): the SW/LW surface (bottom-of-atmosphere) and TOA up/down
# flux slices the RRTMG column solvers already compute, plus the slope-normal
# surface SW flux (SWNORM) and the derived OLR (== LWUPT). All ADD-only (no
# physical fallback) so they appear only when the radiation diagnostics supply
# them, never fabricated. The WRF clear-sky ``...C`` flux vars are deliberately
# absent: this port runs no separate clear-sky radiative-transfer pass, so a
# clear-sky flux value cannot be produced without fabrication.
RADIATION_FLUX_DIAGNOSTIC_VARIABLES: tuple[str, ...] = (
    "SWDNB",
    "SWUPB",
    "LWDNB",
    "LWUPB",
    "SWDNT",
    "SWUPT",
    "LWDNT",
    "LWUPT",
    "OLR",
    "SWNORM",
)

# The full operational field set the writer KNOWS how to emit. Each frame writes
# the subset for which a real source is present (grid metrics + state are always
# present; diagnostics / land carry are optional), so a missing optional source
# cannot manufacture a field. ``write_prepared_wrfout`` iterates the prepared
# payload keys, not this tuple, so optional fields self-gate.
OPERATIONAL_WRFOUT_VARIABLES: tuple[str, ...] = (
    *MINIMUM_WRFOUT_VARIABLES,
    *MICROPHYSICS_EXTRA_VARIABLES,
    *GRID_COORDINATE_VARIABLES,
    *GRID_METRIC_EXTRA_VARIABLES,
    *PRECIP_PARTITION_VARIABLES,
    *SURFACE_FLUX_EXTRA_VARIABLES,
    *DERIVED_DIAGNOSTIC_VARIABLES,
    *LAND_SOIL_VARIABLES,
    *LAND_SNOW_DIAGNOSTIC_VARIABLES,
    *STOCHASTIC_SEED_VARIABLES,
    *RADIATION_FLUX_DIAGNOSTIC_VARIABLES,
)


# Coordinate / dimension variables that make a history frame self-describing
# (geometry + eta levels + staggered lat/lon). ``Times`` and ``XTIME`` are always
# written by ``_write_times`` / ``_write_xtime`` regardless; the rest go through
# the normal variable loop, so ``write_prepared_wrfout`` force-includes THESE in
# any ``variable_subset`` (a name still absent from the prepared payload is simply
# skipped -- the file never fabricates a coordinate). This guarantees a subset
# stream (e.g. the training stream) is a self-contained, stream-valid WRF file.
MANDATORY_WRFOUT_COORDINATES: frozenset[str] = frozenset(
    {"Times", "XTIME", "ZNU", "ZNW", "XLAT_U", "XLAT_V", "XLONG_U", "XLONG_V"}
)


# Lossless NetCDF4 (zlib) compression settings for SUBSET streams only (see
# write_prepared_wrfout). complevel 4 is a strong size/CPU tradeoff for float
# fields; the writer thread runs off the GPU step thread so the CPU cost is hidden.
_SUBSET_STREAM_COMPRESSION: dict[str, object] = {"zlib": True, "complevel": 4}


# A compact, training-ready variable subset for the B200 3km->1km nest dataset
# (issue #122). These 36 names are the model fields a downstream learner needs;
# the writer ALWAYS additionally emits MANDATORY_WRFOUT_COORDINATES so each frame
# is self-describing. Every name is a real entry of WRFOUT_VARIABLE_SPECS; a name
# whose source is absent from a given run is silently skipped (never fabricated).
# OPT-IN ONLY -- selected via the GPUWRF_TRAINING_OUTPUT_SUBSET env flag on the
# nest output path; the default numerical/variable payload is unchanged (the
# authenticated global GRID_ID metadata is common to every output mode).
MINIMAL_TRAINING_SET: tuple[str, ...] = (
    # 3D prognostic / diagnostic (16)
    "U", "V", "W", "T", "P", "PB", "PH", "PHB",
    "QVAPOR", "QCLOUD", "QICE", "QRAIN", "QSNOW", "QGRAUP", "CLDFRA", "QKE",
    # 2D surface (12)
    "T2", "Q2", "U10", "V10", "PSFC", "RAINNC",
    "SWDOWN", "GLW", "HFX", "LH", "PBLH", "TSK",
    # 2D cloud-validation (v0.20.2, output-only): OLR (TOA outgoing LW == cloud-top,
    # MSG-satellite-observable), RAINC (convective precip; completes RAINNC for the
    # 3 km cumulus parent), SWDNB (RRTMG instantaneous surface downwelling SW) (3)
    "OLR", "RAINC", "SWDNB",
    # static geography (5)
    "HGT", "XLAT", "XLONG", "LANDMASK", "LU_INDEX",
    # wind-rotation / map-factor (3)
    "SINALPHA", "COSALPHA", "MAPFAC_M",
)


# Opt-in WRF primary-history field set for the Canary WRFv4 configuration used as
# the v0.22 compatibility target. Order and names match the reference WRF
# ``wrfout_d02`` stream (375 variables including ``Times``), cross-checked against
# <USER_HOME>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON and the EM Registry
# include tree that contributes Noah-MP, stochastic, hybrid-coordinate, and mask
# history fields.
# The default writer still uses OPERATIONAL_WRFOUT_VARIABLES; this heavy list is
# activated only by ``full_variable_set=True`` on the writer/prepared payload.
FULL_WRFOUT_VARIABLES: tuple[str, ...] = (
    "Times", "XLAT", "XLONG", "LU_INDEX", "ZNU", "ZNW", "ZS", "DZS",
    "VAR_SSO", "BATHYMETRY_FLAG", "U", "V", "W", "PH", "PHB", "T",
    "THM", "HFX_FORCE", "LH_FORCE", "TSK_FORCE", "HFX_FORCE_TEND", "LH_FORCE_TEND", "TSK_FORCE_TEND", "MU",
    "MUB", "NEST_POS", "P", "PB", "FNM", "FNP", "RDNW", "RDN",
    "DNW", "DN", "CFN", "CFN1", "THIS_IS_AN_IDEAL_RUN", "P_HYD", "Q2", "T2",
    "TH2", "PSFC", "U10", "V10", "RDX", "RDY", "AREA2D", "DX2D",
    "RESM", "ZETATOP", "CF1", "CF2", "CF3", "ITIMESTEP", "XTIME", "QVAPOR",
    "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNICE", "QNRAIN", "SHDMAX",
    "SHDMIN", "SHDAVG", "SNOALB", "TSLB", "SMOIS", "SH2O", "SEAICE", "XICEM",
    "SFROFF", "UDROFF", "IVGTYP", "ISLTYP", "VEGFRA", "GRDFLX", "ACGRDFLX", "ACSNOM",
    "SNOW", "SNOWH", "CANWAT", "SSTSK", "WATER_DEPTH", "COSZEN", "LAI", "QKE",
    "MAXMF", "MAXWIDTH", "ZTOP_PLUME", "DTAUX3D", "DTAUY3D", "DUSFCG", "DVSFCG", "VAR",
    "CON", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3",
    "OL4", "TKE_PBL", "EL_PBL", "O3_GFS_DU", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "MAPFAC_MX",
    "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY", "MAPFAC_VX", "MF_VX_INV", "MAPFAC_VY", "F", "E",
    "SINALPHA", "COSALPHA", "HGT", "TSK", "P_TOP", "GOT_VAR_SSO", "T00", "P00",
    "TLP", "TISO", "TLP_STRAT", "P_STRAT", "MAX_MSFTX", "MAX_MSFTY", "RAINC", "RAINSH",
    "RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC", "CLDFRA", "SWDOWN", "GLW", "SWNORM",
    "ACSWUPT", "ACSWUPTC", "ACSWDNT", "ACSWDNTC", "ACSWUPB", "ACSWUPBC", "ACSWDNB", "ACSWDNBC",
    "ACLWUPT", "ACLWUPTC", "ACLWDNT", "ACLWDNTC", "ACLWUPB", "ACLWUPBC", "ACLWDNB", "ACLWDNBC",
    "SWUPT", "SWUPTC", "SWDNT", "SWDNTC", "SWUPB", "SWUPBC", "SWDNB", "SWDNBC",
    "LWUPT", "LWUPTC", "LWDNT", "LWDNTC", "LWUPB", "LWUPBC", "LWDNB", "LWDNBC",
    "OLR", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "ALBEDO", "CLAT", "ALBBCK",
    "EMISS", "NOAHRES", "TMN", "XLAND", "UST", "PBLH", "HFX", "QFX",
    "LH", "ACHFX", "ACLHF", "SNOWC", "SR", "SAVE_TOPO_FROM_REAL", "ISEEDARR_SPPT", "ISEEDARR_SKEBS",
    "ISEEDARR_RAND_PERTURB", "ISEEDARRAY_SPP_CONV", "ISEEDARRAY_SPP_PBL", "ISEEDARRAY_SPP_LSM", "ISNOW", "TV", "TG", "CANICE",
    "CANLIQ", "EAH", "TAH", "CM", "CH", "FWET", "SNEQVO", "ALBOLD",
    "QSNOWXY", "QRAINXY", "WSLAKE", "ZWT", "WA", "WT", "TSNO", "ZSNSO",
    "SNICE", "SNLIQ", "LFMASS", "RTMASS", "STMASS", "WOOD", "STBLCP", "FASTCP",
    "XSAI", "TAUSS", "T2V", "T2B", "Q2V", "Q2B", "TRAD", "NEE",
    "GPP", "NPP", "FVEG", "QIN", "RUNSF", "RUNSB", "ECAN", "EDIR",
    "ETRAN", "FSA", "FIRA", "APAR", "PSN", "SAV", "SAG", "RSSUN",
    "RSSHA", "BGAP", "WGAP", "TGV", "TGB", "CHV", "CHB", "SHG",
    "SHC", "SHB", "EVG", "EVB", "GHV", "GHB", "IRG", "IRC",
    "IRB", "TR", "EVC", "CHLEAF", "CHUC", "CHV2", "CHB2", "CHSTAR",
    "SMCWTD", "RECH", "QRFS", "QSPRINGS", "QSLAT", "ACINTS", "ACINTR", "ACDRIPR",
    "ACTHROR", "ACEVAC", "ACDEWC", "FORCTLSM", "FORCQLSM", "FORCPLSM", "FORCZLSM", "FORCWLSM",
    "ACRAINLSM", "ACRUNSB", "ACRUNSF", "ACECAN", "ACETRAN", "ACEDIR", "ACQLAT", "ACQRF",
    "ACETLSM", "ACSNOWLSM", "ACSUBC", "ACFROC", "ACFRZC", "ACMELTC", "ACSNBOT", "ACSNMELT",
    "ACPONDING", "ACSNSUB", "ACSNFRO", "ACRAINSNOW", "ACDRIPS", "ACTHROS", "ACSAGB", "ACIRB",
    "ACSHB", "ACEVB", "ACGHB", "ACPAHB", "ACSAGV", "ACIRG", "ACSHG", "ACEVG",
    "ACGHV", "ACPAHG", "ACSAV", "ACIRC", "ACSHC", "ACEVC", "ACTR", "ACPAHV",
    "ACSWDNLSM", "ACSWUPLSM", "ACLWDNLSM", "ACLWUPLSM", "ACSHFLSM", "ACLHFLSM", "ACGHFLSM", "ACPAHLSM",
    "ACCANHS", "SOILENERGY", "SNOWENERGY", "ACEFLXB", "GRAIN", "GDD", "CROPCAT", "PGS",
    "QTDRAIN", "IRNUMSI", "IRNUMMI", "IRNUMFI", "IRSIVOL", "IRMIVOL", "IRFIVOL", "IRELOSS",
    "IRRSPLH", "C1H", "C2H", "C1F", "C2F", "C3H", "C4H", "C3F",
    "C4F", "PCB", "PC", "LANDMASK", "LAKEMASK", "SST", "SST_INPUT",
)


@dataclass(frozen=True)
class WrfoutVariableSpec:
    """WRF variable schema metadata for the v0 minimum output subset."""

    name: str
    dimensions: tuple[str, ...]
    memory_order: str
    description: str
    units: str
    stagger: str = ""
    coordinates: str | None = None
    dtype: str = "f4"


def _spec(
    name: str,
    dimensions: tuple[str, ...],
    memory_order: str,
    description: str,
    units: str,
    *,
    stagger: str = "",
    coordinates: str | None = None,
    dtype: str = "f4",
) -> WrfoutVariableSpec:
    return WrfoutVariableSpec(
        name=name,
        dimensions=dimensions,
        memory_order=memory_order,
        description=description,
        units=units,
        stagger=stagger,
        coordinates=coordinates,
        dtype=dtype,
    )


XY = ("Time", "south_north", "west_east")
XYZ = ("Time", "bottom_top", "south_north", "west_east")
U_XYZ = ("Time", "bottom_top", "south_north", "west_east_stag")
V_XYZ = ("Time", "bottom_top", "south_north_stag", "west_east")
W_XYZ = ("Time", "bottom_top_stag", "south_north", "west_east")
Z_XYZ = ("Time", "bottom_top_stag", "south_north", "west_east")
U_XY = ("Time", "south_north", "west_east_stag")
V_XY = ("Time", "south_north_stag", "west_east")
# P0-5a additions: soil column (4-layer, Z-staggered) + 1-D eta level columns
# + scalar Time-only fields, matching WRF wrfout dimension layout exactly.
SOIL = ("Time", "soil_layers_stag", "south_north", "west_east")
SNOW = ("Time", "snow_layers_stag", "south_north", "west_east")
SNSO = ("Time", "snso_layers_stag", "south_north", "west_east")
SEED = ("Time", "seed_dim_stag")
Z_HALF = ("Time", "bottom_top")
Z_FULL = ("Time", "bottom_top_stag")
SOIL_1D = ("Time", "soil_layers_stag")
TIME_ONLY = ("Time",)
MAPFAC_U_XY = ("Time", "south_north", "west_east_stag")
MAPFAC_V_XY = ("Time", "south_north_stag", "west_east")

WRFOUT_VARIABLE_SPECS: dict[str, WrfoutVariableSpec] = {
    "XTIME": _spec(
        "XTIME",
        ("Time",),
        "0  ",
        "minutes since simulation start",
        "minutes since simulation start",
    ),
    "XLAT": _spec("XLAT", XY, "XY ", "LATITUDE, SOUTH IS NEGATIVE", "degree_north", coordinates="XLONG XLAT"),
    "XLONG": _spec("XLONG", XY, "XY ", "LONGITUDE, WEST IS NEGATIVE", "degree_east", coordinates="XLONG XLAT"),
    "XLAT_U": _spec(
        "XLAT_U",
        U_XY,
        "XY ",
        "LATITUDE, SOUTH IS NEGATIVE",
        "degree_north",
        stagger="X",
        coordinates="XLONG_U XLAT_U",
    ),
    "XLONG_U": _spec(
        "XLONG_U",
        U_XY,
        "XY ",
        "LONGITUDE, WEST IS NEGATIVE",
        "degree_east",
        stagger="X",
        coordinates="XLONG_U XLAT_U",
    ),
    "XLAT_V": _spec(
        "XLAT_V",
        V_XY,
        "XY ",
        "LATITUDE, SOUTH IS NEGATIVE",
        "degree_north",
        stagger="Y",
        coordinates="XLONG_V XLAT_V",
    ),
    "XLONG_V": _spec(
        "XLONG_V",
        V_XY,
        "XY ",
        "LONGITUDE, WEST IS NEGATIVE",
        "degree_east",
        stagger="Y",
        coordinates="XLONG_V XLAT_V",
    ),
    "HGT": _spec("HGT", XY, "XY ", "Terrain Height", "m", coordinates="XLONG XLAT XTIME"),
    "LANDMASK": _spec(
        "LANDMASK",
        XY,
        "XY ",
        "LAND MASK (1 FOR LAND, 0 FOR WATER)",
        "",
        coordinates="XLONG XLAT XTIME",
    ),
    "LU_INDEX": _spec("LU_INDEX", XY, "XY ", "LAND USE CATEGORY", "", coordinates="XLONG XLAT XTIME"),
    "U10": _spec("U10", XY, "XY ", "U at 10 M", "m s-1", coordinates="XLONG XLAT XTIME"),
    "V10": _spec("V10", XY, "XY ", "V at 10 M", "m s-1", coordinates="XLONG XLAT XTIME"),
    "U10E": _spec("U10E", XY, "XY ", "Special U at 10 M from MYJSFC", "m s-1", coordinates="XLONG XLAT XTIME"),
    "V10E": _spec("V10E", XY, "XY ", "Special V at 10 M from MYJSFC", "m s-1", coordinates="XLONG XLAT XTIME"),
    "T2": _spec("T2", XY, "XY ", "TEMP at 2 M", "K", coordinates="XLONG XLAT XTIME"),
    "Q2": _spec("Q2", XY, "XY ", "QV at 2 M", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "PSFC": _spec("PSFC", XY, "XY ", "SFC PRESSURE", "Pa", coordinates="XLONG XLAT XTIME"),
    "RAINC": _spec(
        "RAINC",
        XY,
        "XY ",
        "ACCUMULATED TOTAL CUMULUS PRECIPITATION",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "RAINNC": _spec(
        "RAINNC",
        XY,
        "XY ",
        "ACCUMULATED TOTAL GRID SCALE PRECIPITATION",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "RAINSH": _spec(
        "RAINSH",
        XY,
        "XY ",
        "ACCUMULATED SHALLOW CUMULUS PRECIPITATION",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWDOWN": _spec(
        "SWDOWN",
        XY,
        "XY ",
        "DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "GLW": _spec(
        "GLW",
        XY,
        "XY ",
        "DOWNWARD LONG WAVE FLUX AT GROUND SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "PBLH": _spec("PBLH", XY, "XY ", "PBL HEIGHT", "m", coordinates="XLONG XLAT XTIME"),
    "UST": _spec("UST", XY, "XY ", "U* IN SIMILARITY THEORY", "m s-1", coordinates="XLONG XLAT XTIME"),
    "HFX": _spec(
        "HFX",
        XY,
        "XY ",
        "UPWARD HEAT FLUX AT THE SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LH": _spec(
        "LH",
        XY,
        "XY ",
        "LATENT HEAT FLUX AT THE SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "TSK": _spec(
        "TSK",
        XY,
        "XY ",
        "SURFACE SKIN TEMPERATURE",
        "K",
        coordinates="XLONG XLAT XTIME",
    ),
    "CLDFRA": _spec("CLDFRA", XYZ, "XYZ", "CLOUD FRACTION", "", coordinates="XLONG XLAT XTIME"),
    "QC_BL": _spec("QC_BL", XYZ, "XYZ", "CLOUD WATER MIXING RATIO IN PBL schemes", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "CLDFRA_BL": _spec("CLDFRA_BL", XYZ, "XYZ", "CLOUD FRACTION pbl", "", coordinates="XLONG XLAT XTIME"),
    "DTAUX3D": _spec("DTAUX3D", XYZ, "XYZ", "LOCAL U GWDO STRESS", "m s-1", coordinates="XLONG XLAT XTIME"),
    "DTAUY3D": _spec("DTAUY3D", XYZ, "XYZ", "LOCAL V GWDO STRESS", "m s-1", coordinates="XLONG XLAT XTIME"),
    "QCLOUD": _spec(
        "QCLOUD",
        XYZ,
        "XYZ",
        "Cloud water mixing ratio",
        "kg kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "QICE": _spec("QICE", XYZ, "XYZ", "Ice mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QRAIN": _spec(
        "QRAIN",
        XYZ,
        "XYZ",
        "Rain water mixing ratio",
        "kg kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "U": _spec(
        "U",
        U_XYZ,
        "XYZ",
        "x-wind component",
        "m s-1",
        stagger="X",
        coordinates="XLONG_U XLAT_U XTIME",
    ),
    "V": _spec(
        "V",
        V_XYZ,
        "XYZ",
        "y-wind component",
        "m s-1",
        stagger="Y",
        coordinates="XLONG_V XLAT_V XTIME",
    ),
    "W": _spec(
        "W",
        W_XYZ,
        "XYZ",
        "z-wind component",
        "m s-1",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    "T": _spec(
        "T",
        XYZ,
        "XYZ",
        "perturbation potential temperature theta-t0",
        "K",
        coordinates="XLONG XLAT XTIME",
    ),
    "THM": _spec(
        "THM",
        XYZ,
        "XYZ",
        "either 1) pert moist pot temp=(1+Rv/Rd Qv)*(theta+T0)-T0; or 2) pert dry pot temp=theta; based on use_theta_m setting",
        "K",
        coordinates="XLONG XLAT XTIME",
    ),
    "QVAPOR": _spec(
        "QVAPOR",
        XYZ,
        "XYZ",
        "Water vapor mixing ratio",
        "kg kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "P": _spec("P", XYZ, "XYZ", "perturbation pressure", "Pa", coordinates="XLONG XLAT XTIME"),
    "PB": _spec("PB", XYZ, "XYZ", "BASE STATE PRESSURE", "Pa", coordinates="XLONG XLAT XTIME"),
    "PH": _spec(
        "PH",
        Z_XYZ,
        "XYZ",
        "perturbation geopotential",
        "m2 s-2",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    "PHB": _spec(
        "PHB",
        Z_XYZ,
        "XYZ",
        "base-state geopotential",
        "m2 s-2",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    "MU": _spec(
        "MU",
        XY,
        "XY ",
        "perturbation dry air mass in column",
        "Pa",
        coordinates="XLONG XLAT XTIME",
    ),
    "MUB": _spec(
        "MUB",
        XY,
        "XY ",
        "base state dry air mass in column",
        "Pa",
        coordinates="XLONG XLAT XTIME",
    ),
    # --- P0-5a (a) extra hydrometeors + number concentrations + MYNN TKE ---
    "QSNOW": _spec("QSNOW", XYZ, "XYZ", "Snow mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QGRAUP": _spec("QGRAUP", XYZ, "XYZ", "Graupel mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QNICE": _spec("QNICE", XYZ, "XYZ", "Ice Number concentration", "  kg-1", coordinates="XLONG XLAT XTIME"),
    "QNRAIN": _spec("QNRAIN", XYZ, "XYZ", "Rain Number concentration", "  kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QNSNOW": _spec("QNSNOW", XYZ, "XYZ", "Snow Number concentration", "  kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QNGRAUPEL": _spec(
        "QNGRAUPEL",
        XYZ,
        "XYZ",
        "Graupel Number concentration",
        "  kg(-1)",
        coordinates="XLONG XLAT XTIME",
    ),
    "QNCLOUD": _spec(
        "QNCLOUD",
        XYZ,
        "XYZ",
        "cloud water Number concentration",
        "  kg(-1)",
        coordinates="XLONG XLAT XTIME",
    ),
    "QNCCN": _spec("QNCCN", XYZ, "XYZ", "CCN Number concentration", "  kg(-1)", coordinates="XLONG XLAT XTIME"),
    # v0.17 ADR-032 graupel/hail substrate (WRF Registry hail-family
    # qh/qnh/qvolg/qvolh names/descriptions/units).
    "QHAIL": _spec("QHAIL", XYZ, "XYZ", "Hail mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QNHAIL": _spec("QNHAIL", XYZ, "XYZ", "Hail Number concentration", "  kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QVGRAUPEL": _spec("QVGRAUPEL", XYZ, "XYZ", "Graupel Particle Volume", "m(3) kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QVHAIL": _spec("QVHAIL", XYZ, "XYZ", "Hail Particle Volume", "m(3) kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QIR": _spec("QIR", XYZ, "XYZ", "Rime ice mass-1 mixing ratio", "kg kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QIB": _spec("QIB", XYZ, "XYZ", "Rime ice volume-1 mixing ratio", "m(3) kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QICE2": _spec("QICE2", XYZ, "XYZ", "Ice mixing ratio cat 2", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QNICE2": _spec("QNICE2", XYZ, "XYZ", "Ice Number concentration cat 2", "# kg-1", coordinates="XLONG XLAT XTIME"),
    "QIR2": _spec("QIR2", XYZ, "XYZ", "Rime ice mass-2 mixing ratio", "kg kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QIB2": _spec("QIB2", XYZ, "XYZ", "Rime ice volume-2 mixing ratio", "m(3) kg(-1)", coordinates="XLONG XLAT XTIME"),
    "QZI": _spec("QZI", XYZ, "XYZ", "Sq root of sixth moment ice mixing ratio times number mixing ratio", "m(3) kg(-1)", coordinates="XLONG XLAT XTIME"),
    # v0.16 aerosol-aware Thompson (mp=28) aerosol numbers (WRF Registry
    # thompsonaero qnwfa/qnifa names/descriptions).
    "QNWFA": _spec(
        "QNWFA",
        XYZ,
        "XYZ",
        "water-friendly aerosol number con",
        "  kg(-1)",
        coordinates="XLONG XLAT XTIME",
    ),
    "QNIFA": _spec(
        "QNIFA",
        XYZ,
        "XYZ",
        "ice-friendly aerosol number con",
        "  kg(-1)",
        coordinates="XLONG XLAT XTIME",
    ),
    "QKE": _spec("QKE", XYZ, "XYZ", "twice TKE from MYNN", "m2 s-2", coordinates="XLONG XLAT XTIME"),
    "TKE": _spec("TKE", XYZ, "XYZ", "LES turbulence kinetic energy (km_opt=2)", "m2 s-2", coordinates="XLONG XLAT XTIME"),
    # --- P0-5a (b) grid-static coordinate / map-factor / Coriolis fields ---
    "ZNU": _spec("ZNU", Z_HALF, "Z  ", "eta values on half (mass) levels", ""),
    "ZNW": _spec("ZNW", Z_FULL, "Z  ", "eta values on full (w) levels", "", stagger="Z"),
    # Static Noah/Noah-MP soil-layer geometry (init_soil_depth_2; the same arrays
    # WRF repeats every history frame). ZS = layer-center depths, DZS = layer
    # thicknesses; 1-D columns of length soil_layers_stag. No coordinates attr
    # (matches the reference wrfout, which omits it for these 1-D soil fields).
    "ZS": _spec("ZS", SOIL_1D, "Z  ", "DEPTHS OF CENTERS OF SOIL LAYERS", "m", stagger="Z"),
    "DZS": _spec("DZS", SOIL_1D, "Z  ", "THICKNESSES OF SOIL LAYERS", "m", stagger="Z"),
    "MAPFAC_M": _spec("MAPFAC_M", XY, "XY ", "Map scale factor on mass grid", "", coordinates="XLONG XLAT XTIME"),
    "MAPFAC_U": _spec(
        "MAPFAC_U", MAPFAC_U_XY, "XY ", "Map scale factor on u-grid", "",
        stagger="X", coordinates="XLONG_U XLAT_U XTIME",
    ),
    "MAPFAC_V": _spec(
        "MAPFAC_V", MAPFAC_V_XY, "XY ", "Map scale factor on v-grid", "",
        stagger="Y", coordinates="XLONG_V XLAT_V XTIME",
    ),
    "F": _spec("F", XY, "XY ", "Coriolis sine latitude term", "s-1", coordinates="XLONG XLAT XTIME"),
    "E": _spec("E", XY, "XY ", "Coriolis cosine latitude term", "s-1", coordinates="XLONG XLAT XTIME"),
    "SINALPHA": _spec("SINALPHA", XY, "XY ", "Local sine of map rotation", "", coordinates="XLONG XLAT XTIME"),
    "COSALPHA": _spec("COSALPHA", XY, "XY ", "Local cosine of map rotation", "", coordinates="XLONG XLAT XTIME"),
    "XLAND": _spec(
        "XLAND", XY, "XY ", "LAND MASK (1 FOR LAND, 2 FOR WATER)", "",
        coordinates="XLONG XLAT XTIME",
    ),
    "P_TOP": _spec("P_TOP", TIME_ONLY, "0  ", "PRESSURE TOP OF THE MODEL", "Pa"),
    # --- v0.12.0 A1 (b2) extended grid-static vertical/map-factor metrics ---
    # Vertical eta-spacing metrics (DycoreMetrics.dn/dnw/rdn/rdnw/fnm/fnp).
    "DN": _spec("DN", Z_HALF, "Z  ", "d(eta) values between half (mass) levels", ""),
    "DNW": _spec("DNW", Z_HALF, "Z  ", "d(eta) values between full (w) levels", ""),
    "RDN": _spec("RDN", Z_HALF, "Z  ", "inverse d(eta) values between half (mass) levels", ""),
    "RDNW": _spec("RDNW", Z_HALF, "Z  ", "inverse d(eta) values between full (w) levels", ""),
    "FNM": _spec("FNM", Z_HALF, "Z  ", "upper weight for vertical stretching", ""),
    "FNP": _spec("FNP", Z_HALF, "Z  ", "lower weight for vertical stretching", ""),
    # Scalar vertical extrapolation constants (DycoreMetrics.cf1/cf2/cf3 + the
    # WRF top-level cfn/cfn1 derived from dnw/dn).
    "CFN": _spec("CFN", TIME_ONLY, "0  ", "extrapolation constant", ""),
    "CFN1": _spec("CFN1", TIME_ONLY, "0  ", "extrapolation constant", ""),
    "CF1": _spec("CF1", TIME_ONLY, "0  ", "2nd order extrapolation constant", ""),
    "CF2": _spec("CF2", TIME_ONLY, "0  ", "2nd order extrapolation constant", ""),
    "CF3": _spec("CF3", TIME_ONLY, "0  ", "2nd order extrapolation constant", ""),
    # Hybrid-eta C-coefficients (DycoreMetrics.c1h..c4h / c1f..c4f).
    "C1H": _spec("C1H", Z_HALF, "Z  ", "half levels, c1h = d bf / d eta, using znw", "Dimensionless"),
    "C2H": _spec("C2H", Z_HALF, "Z  ", "half levels, c2h = (1-c1h)*(p0-pt)", "Pa"),
    "C3H": _spec("C3H", Z_HALF, "Z  ", "half levels, c3h = bh", "Dimensionless"),
    "C4H": _spec("C4H", Z_HALF, "Z  ", "half levels, c4h = (eta-bh)*(p0-pt), using znu", "Pa"),
    "C1F": _spec("C1F", Z_FULL, "Z  ", "full levels, c1f = d bf / d eta, using znu", "Dimensionless", stagger="Z"),
    "C2F": _spec("C2F", Z_FULL, "Z  ", "full levels, c2f = (1-c1f)*(p0-pt)", "Pa", stagger="Z"),
    "C3F": _spec("C3F", Z_FULL, "Z  ", "full levels, c3f = bf", "Dimensionless", stagger="Z"),
    "C4F": _spec("C4F", Z_FULL, "Z  ", "full levels, c4f = (eta-bf)*(p0-pt), using znw", "Pa", stagger="Z"),
    # Directional map-scale factors (DycoreMetrics.msftx/msfty/msfux/msfuy/msfvx/msfvy).
    "MAPFAC_MX": _spec(
        "MAPFAC_MX", XY, "XY ", "Map scale factor on mass grid, x direction", "",
        coordinates="XLONG XLAT XTIME",
    ),
    "MAPFAC_MY": _spec(
        "MAPFAC_MY", XY, "XY ", "Map scale factor on mass grid, y direction", "",
        coordinates="XLONG XLAT XTIME",
    ),
    "MAPFAC_UX": _spec(
        "MAPFAC_UX", MAPFAC_U_XY, "XY ", "Map scale factor on u-grid, x direction", "",
        stagger="X", coordinates="XLONG_U XLAT_U XTIME",
    ),
    "MAPFAC_UY": _spec(
        "MAPFAC_UY", MAPFAC_U_XY, "XY ", "Map scale factor on u-grid, y direction", "",
        stagger="X", coordinates="XLONG_U XLAT_U XTIME",
    ),
    "MAPFAC_VX": _spec(
        "MAPFAC_VX", MAPFAC_V_XY, "XY ", "Map scale factor on v-grid, x direction", "",
        stagger="Y", coordinates="XLONG_V XLAT_V XTIME",
    ),
    "MAPFAC_VY": _spec(
        "MAPFAC_VY", MAPFAC_V_XY, "XY ", "Map scale factor on v-grid, y direction", "",
        stagger="Y", coordinates="XLONG_V XLAT_V XTIME",
    ),
    # Inverse grid lengths (1/dx, 1/dy) from the projection.
    "RDX": _spec("RDX", TIME_ONLY, "0  ", "INVERSE X GRID LENGTH", "m-1"),
    "RDY": _spec("RDY", TIME_ONLY, "0  ", "INVERSE Y GRID LENGTH", "m-1"),
    # --- v0.12.0 A2 (h) trivially-derived diagnostics ---
    "CLAT": _spec(
        "CLAT", XY, "XY ", "COMPUTATIONAL GRID LATITUDE, SOUTH IS NEGATIVE", "degree_north",
        coordinates="XLONG XLAT XTIME",
    ),
    "COSZEN": _spec(
        "COSZEN", XY, "XY ", "COS of SOLAR ZENITH ANGLE", "dimensionless",
        coordinates="XLONG XLAT XTIME",
    ),
    "SR": _spec(
        "SR", XY, "XY ", "fraction of frozen precipitation", "-",
        coordinates="XLONG XLAT XTIME",
    ),
    "SNOWC": _spec(
        "SNOWC", XY, "XY ", "FLAG INDICATING SNOW COVERAGE (1 FOR SNOW COVER)", "",
        coordinates="XLONG XLAT XTIME",
    ),
    # --- P0-5a (c) accumulated precipitation partition ---
    "SNOWNC": _spec(
        "SNOWNC", XY, "XY ", "ACCUMULATED TOTAL GRID SCALE SNOW AND ICE", "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "GRAUPELNC": _spec(
        "GRAUPELNC", XY, "XY ", "ACCUMULATED TOTAL GRID SCALE GRAUPEL", "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    # v0.17 hail microphysics (WSM7/WDM7): accumulated grid-scale hail.
    "HAILNC": _spec(
        "HAILNC", XY, "XY ", "ACCUMULATED TOTAL GRID SCALE HAIL", "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    # --- P0-5a (d) surface energy/moisture-budget fluxes + 2-m theta ---
    "QFX": _spec(
        "QFX", XY, "XY ", "UPWARD MOISTURE FLUX AT THE SURFACE", "kg m-2 s-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "GRDFLX": _spec("GRDFLX", XY, "XY ", "GROUND HEAT FLUX", "W m-2", coordinates="XLONG XLAT XTIME"),
    "TH2": _spec("TH2", XY, "XY ", "POT TEMP at 2 M", "K", coordinates="XLONG XLAT XTIME"),
    # --- P0-5a (e) prognostic Noah-MP soil/snow land columns + land diagnostics ---
    "TSLB": _spec("TSLB", SOIL, "XYZ", "SOIL TEMPERATURE", "K", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SMOIS": _spec("SMOIS", SOIL, "XYZ", "SOIL MOISTURE", "m3 m-3", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SH2O": _spec("SH2O", SOIL, "XYZ", "SOIL LIQUID WATER", "m3 m-3", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SNOW": _spec("SNOW", XY, "XY ", "SNOW WATER EQUIVALENT", "kg m-2", coordinates="XLONG XLAT XTIME"),
    "SNOWH": _spec("SNOWH", XY, "XY ", "PHYSICAL SNOW DEPTH", "m", coordinates="XLONG XLAT XTIME"),
    "CANWAT": _spec("CANWAT", XY, "XY ", "CANOPY WATER", "kg m-2", coordinates="XLONG XLAT XTIME"),
    "SFROFF": _spec("SFROFF", XY, "XY ", "SURFACE RUNOFF", "mm", coordinates="XLONG XLAT XTIME"),
    "UDROFF": _spec("UDROFF", XY, "XY ", "UNDERGROUND RUNOFF", "mm", coordinates="XLONG XLAT XTIME"),
    "ALBEDO": _spec("ALBEDO", XY, "XY ", "ALBEDO", "-", coordinates="XLONG XLAT XTIME"),
    "EMISS": _spec("EMISS", XY, "XY ", "SURFACE EMISSIVITY", "", coordinates="XLONG XLAT XTIME"),
    "TSNO": _spec("TSNO", SNOW, "XYZ", "snow temperature", "K", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SNICE": _spec("SNICE", SNOW, "XYZ", "snow layer ice", "mm", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "SNLIQ": _spec("SNLIQ", SNOW, "XYZ", "snow layer liquid", "mm", stagger="Z", coordinates="XLONG XLAT XTIME"),
    "ZSNSO": _spec(
        "ZSNSO",
        SNSO,
        "XYZ",
        "layer-bottom depth from snow surf",
        "m",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    # WRF integer field (FieldType 106): active snow-layer count in {-2,-1,0}.
    # The reference wrfout labels its units "m3 m-3" (a known upstream WRF
    # Registry typo); we reproduce it byte-for-byte for schema conformance.
    "ISNOW": _spec(
        "ISNOW",
        XY,
        "XY ",
        "no. of snow layer",
        "m3 m-3",
        coordinates="XLONG XLAT XTIME",
        dtype="i4",
    ),
    "SNEQVO": _spec(
        "SNEQVO",
        XY,
        "XY ",
        "snow mass at last time step",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "CANLIQ": _spec(
        "CANLIQ",
        XY,
        "XY ",
        "intercepted liquid water",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "CANICE": _spec(
        "CANICE",
        XY,
        "XY ",
        "intercepted ice mass",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "ISEEDARR_SPPT": _spec(
        "ISEEDARR_SPPT",
        SEED,
        "Z  ",
        "Array to hold seed for restart, SPPT",
        "",
        stagger="Z",
        dtype="i4",
    ),
    "ISEEDARR_SKEBS": _spec(
        "ISEEDARR_SKEBS",
        SEED,
        "Z  ",
        "Array to hold seed for restart, SKEBS",
        "",
        stagger="Z",
        dtype="i4",
    ),
    "ISEEDARRAY_SPP_CONV": _spec(
        "ISEEDARRAY_SPP_CONV",
        SEED,
        "Z  ",
        "Array to hold seed for restart, RAND_PERT2",
        "",
        stagger="Z",
        dtype="i4",
    ),
    "ISEEDARRAY_SPP_PBL": _spec(
        "ISEEDARRAY_SPP_PBL",
        SEED,
        "Z  ",
        "Array to hold seed for restart, RAND_PERT3",
        "",
        stagger="Z",
        dtype="i4",
    ),
    "ISEEDARRAY_SPP_LSM": _spec(
        "ISEEDARRAY_SPP_LSM",
        SEED,
        "Z  ",
        "Array to hold seed for restart, RAND_PERT4",
        "",
        stagger="Z",
        dtype="i4",
    ),
    # --- B1 (v0.12.0) RRTMG up/down all-sky radiation flux diagnostics --------
    # Instantaneous surface (bottom-of-atmosphere) and top-of-atmosphere up/down
    # SW/LW fluxes + the slope-normal surface SW flux (SWNORM) + outgoing LW (OLR).
    # Schemas (dims/MemoryOrder/stagger/units/description/dtype) are copied verbatim
    # from the reference Gen2 wrfout_d02; every field maps to a flux slice the RRTMG
    # SW/LW column solvers already compute (surface == bottom interface, TOA == top
    # interface). OLR == LWUPT (WRF's TOA outgoing LW) and is derived in
    # ``prepare_wrfout_payload`` from the diagnostic LWUPT.
    #
    # v0.13.0: the WRF clear-sky ``...C`` flux vars (SWDNBC/SWUPBC/LWDNBC/LWUPBC
    # /SWDNTC/SWUPTC/LWDNTC/LWUPTC) ARE now specced/emitted -- the RRTMG SW/LW
    # solvers run the WRF second clear-sky (cloud-free) radiative-transfer pass
    # (``solve_rrtmg_*_column(..., with_clear_sky=True)``; WRF ``pbbcd/pbbcu``,
    # ``totdclfl/totuclfl``).  Still ADD-only: emitted only when the radiation
    # diagnostics supply them, never fabricated, and the all-sky fluxes above are
    # byte-identical with or without the clear-sky pass.
    "SWDNB": _spec(
        "SWDNB", XY, "XY ", "INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWUPB": _spec(
        "SWUPB", XY, "XY ", "INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWDNB": _spec(
        "LWDNB", XY, "XY ", "INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWUPB": _spec(
        "LWUPB", XY, "XY ", "INSTANTANEOUS UPWELLING LONGWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWDNT": _spec(
        "SWDNT", XY, "XY ", "INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWUPT": _spec(
        "SWUPT", XY, "XY ", "INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWDNT": _spec(
        "LWDNT", XY, "XY ", "INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWUPT": _spec(
        "LWUPT", XY, "XY ", "INSTANTANEOUS UPWELLING LONGWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "OLR": _spec(
        "OLR", XY, "XY ", "TOA OUTGOING LONG WAVE", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWNORM": _spec(
        "SWNORM", XY, "XY ", "NORMAL SHORT WAVE FLUX AT GROUND SURFACE (SLOPE-DEPENDENT)", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    # --- v0.13.0 RRTMG CLEAR-SKY (cloud-free) up/down radiation flux diagnostics.
    # Schemas copied verbatim from the reference Gen2 wrfout; each maps to a
    # clear-sky flux slice the RRTMG SW/LW clear-sky pass computes (WRF `pbbcd/
    # pbbcu`, `totdclfl/totuclfl`).  Bottom == surface interface, top == model-top
    # interface.  ADD-only (no fallback): emitted only when the radiation
    # diagnostics supply the clear-sky fluxes. ---
    "SWDNBC": _spec(
        "SWDNBC", XY, "XY ", "INSTANTANEOUS DOWNWELLING CLEAR SKY SHORTWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWUPBC": _spec(
        "SWUPBC", XY, "XY ", "INSTANTANEOUS UPWELLING CLEAR SKY SHORTWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWDNBC": _spec(
        "LWDNBC", XY, "XY ", "INSTANTANEOUS DOWNWELLING CLEAR SKY LONGWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWUPBC": _spec(
        "LWUPBC", XY, "XY ", "INSTANTANEOUS UPWELLING CLEAR SKY LONGWAVE FLUX AT BOTTOM", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWDNTC": _spec(
        "SWDNTC", XY, "XY ", "INSTANTANEOUS DOWNWELLING CLEAR SKY SHORTWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWUPTC": _spec(
        "SWUPTC", XY, "XY ", "INSTANTANEOUS UPWELLING CLEAR SKY SHORTWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWDNTC": _spec(
        "LWDNTC", XY, "XY ", "INSTANTANEOUS DOWNWELLING CLEAR SKY LONGWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LWUPTC": _spec(
        "LWUPTC", XY, "XY ", "INSTANTANEOUS UPWELLING CLEAR SKY LONGWAVE FLUX AT TOP", "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "CWM": _spec("CWM", XYZ, "XYZ", "Total condensate mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    **{name: _spec(name, XYZ, "XYZ", description, "micron", coordinates="XLONG XLAT XTIME")
       for name, description in (("RE_CLOUD_GSFC", "Cloud Water effective radius"),
                                 ("RE_RAIN_GSFC", "Rain Water effective radius"),
                                 ("RE_ICE_GSFC", "Cloud Ice effective radius"),
                                 ("RE_SNOW_GSFC", "Snow effective radius"),
                                 ("RE_GRAUPEL_GSFC", "Graupel Water effective radius"),
                                 ("RE_HAIL_GSFC", "Hail Water effective radius"))},
    "QRIMEF": _spec("QRIMEF", XYZ, "XYZ", "rime factor * qi", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QNWFA2D": _spec("QNWFA2D", XY, "XY ", "Surface aerosol number conc emission", "kg-1 s-1", coordinates="XLONG XLAT XTIME"),
    "QNIFA2D": _spec("QNIFA2D", XY, "XY ", "Surface dust number conc emission", "kg-1 s-1", coordinates="XLONG XLAT XTIME"),
    "PHYS_TOT": _spec("PHYS_TOT", XYZ, "XYZ", "TOTAL MICROPHYSICAL LATENT HEATING", "K s-1", coordinates="XLONG XLAT XTIME"),
    "F_ICE_PHY": _spec("F_ICE_PHY", XYZ, "XYZ", "FRACTION OF ICE", "", coordinates="XLONG XLAT XTIME"),
    "F_RAIN_PHY": _spec("F_RAIN_PHY", XYZ, "XYZ", "FRACTION OF RAIN ", "", coordinates="XLONG XLAT XTIME"),
    "F_RIMEF_PHY": _spec("F_RIMEF_PHY", XYZ, "XYZ", "MASS RATIO OF RIMED ICE ", "", coordinates="XLONG XLAT XTIME"),
}


_FULL_XYZ_GENERIC_NAMES: tuple[str, ...] = (
    "P_HYD", "DTAUX3D", "DTAUY3D",
)

# Registry.EM_COMMON RUC-only history slots. They are emitted only on sf=3;
# adding specs leaves every existing stream's variable list unchanged.
_RUC_EXTRA_HISTORY = ("SOILT1", "RHOSNF", "SNOWFALLAC", "ACRUNOFF")
for _name, _description, _units in (
    ("SOILT1", "TEMPERATURE INSIDE SNOW", "K"),
    ("RHOSNF", "DENSITY OF FROZEN PRECIP", "kg/m^3"),
    ("SNOWFALLAC", "RUN-TOTAL ACCUMULATED SNOWFALL [mm]", ""),
    ("ACRUNOFF", "ACCUMULATED RUNOFF", "kg m-2"),
):
    WRFOUT_VARIABLE_SPECS[_name] = _spec(_name, XY, "XY ", _description, _units,
                                         coordinates="XLONG XLAT XTIME")
_FULL_Z_XYZ_GENERIC_NAMES: tuple[str, ...] = (
    "TKE_PBL", "EL_PBL",
)
# MMF Registry restart fields can also be explicitly requested in history.
# Their specs alone add no variables to the existing default streams.
MMF_HISTORY_FIELDS = ('FDEPTH', 'AREA', 'EQZWT', 'RIVERBED', 'RIVERCOND', 'PEXP',
                      'SMOISEQ', 'RECHCLIM', 'ZWT', 'DEEPRECH', 'RECH', 'QLAT',
                      'QSLAT', 'QRF', 'QRFS', 'QSPRING', 'QSPRINGS', 'SMCWTD', 'WA', 'WT',
                      'ACRECH', 'ACQSPRING')
for _name, _description, _units in (
    ('FDEPTH', 'e-folding depth for transmissivity', 'm'),
    ('AREA', 'area of grid boxes', 'm2'), ('EQZWT', 'equilibrium water table depth', 'm'),
    ('RIVERBED', 'river bed depth', 'm'), ('RIVERCOND', 'river conductance', 'Kg m s-1'),
    ('PEXP', 'exponent for river conductance', 'Kg m s-1'),
    ('SMOISEQ', 'EQ. SOIL MOISTURE', 'm3 m-3'),
    ('RECHCLIM', 'equilibrium recharge rate', 'mm'),
    ('DEEPRECH', 'deep water table recharge', 'mm'),
    ('QLAT', 'instantaneous lateral flow', 'mm'), ('QRF', 'baseflow', 'm'),
    ('QSPRING', 'seeping water', 'm'),
    ('ACRECH', 'accumulated recharge', 'mm'), ('ACQSPRING', 'accumulated seeping water', 'mm'),
):
    WRFOUT_VARIABLE_SPECS[_name] = _spec(_name, SOIL if _name == 'SMOISEQ' else XY,
        'XYZ' if _name == 'SMOISEQ' else 'XY ', _description, _units,
        coordinates='XLONG XLAT XTIME')
_FULL_XY_GENERIC_NAMES: tuple[str, ...] = (
    "VAR_SSO", "NEST_POS", "AREA2D", "DX2D", "SHDMAX", "SHDMIN", "SHDAVG", "SNOALB",
    "SEAICE", "XICEM", "VEGFRA", "ACGRDFLX", "ACSNOM", "SSTSK", "WATER_DEPTH", "LAI",
    "MAXMF", "MAXWIDTH", "ZTOP_PLUME", "DUSFCG", "DVSFCG", "VAR", "CON", "OA1",
    "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4", "O3_GFS_DU",
    "ACSWUPT", "ACSWUPTC", "ACSWDNT", "ACSWDNTC", "ACSWUPB", "ACSWUPBC", "ACSWDNB", "ACSWDNBC",
    "ACLWUPT", "ACLWUPTC", "ACLWDNT", "ACLWDNTC", "ACLWUPB", "ACLWUPBC", "ACLWDNB", "ACLWDNBC",
    "ALBBCK", "NOAHRES", "TMN", "ACHFX", "ACLHF", "TV", "TG", "EAH",
    "TAH", "CM", "CH", "FWET", "ALBOLD", "QSNOWXY", "QRAINXY", "WSLAKE",
    "ZWT", "WA", "WT", "LFMASS", "RTMASS", "STMASS", "WOOD", "STBLCP",
    "FASTCP", "XSAI", "TAUSS", "T2V", "T2B", "Q2V", "Q2B", "TRAD",
    "NEE", "GPP", "NPP", "FVEG", "QIN", "RUNSF", "RUNSB", "ECAN",
    "EDIR", "ETRAN", "FSA", "FIRA", "APAR", "PSN", "SAV", "SAG",
    "RSSUN", "RSSHA", "BGAP", "WGAP", "TGV", "TGB", "CHV", "CHB",
    "SHG", "SHC", "SHB", "EVG", "EVB", "GHV", "GHB", "IRG",
    "IRC", "IRB", "TR", "EVC", "CHLEAF", "CHUC", "CHV2", "CHB2",
    "CHSTAR", "SMCWTD", "RECH", "QRFS", "QSPRINGS", "QSLAT", "ACINTS", "ACINTR",
    "ACDRIPR", "ACTHROR", "ACEVAC", "ACDEWC", "FORCTLSM", "FORCQLSM", "FORCPLSM", "FORCZLSM",
    "FORCWLSM", "ACRAINLSM", "ACRUNSB", "ACRUNSF", "ACECAN", "ACETRAN", "ACEDIR", "ACQLAT",
    "ACQRF", "ACETLSM", "ACSNOWLSM", "ACSUBC", "ACFROC", "ACFRZC", "ACMELTC", "ACSNBOT",
    "ACSNMELT", "ACPONDING", "ACSNSUB", "ACSNFRO", "ACRAINSNOW", "ACDRIPS", "ACTHROS", "ACSAGB",
    "ACIRB", "ACSHB", "ACEVB", "ACGHB", "ACPAHB", "ACSAGV", "ACIRG", "ACSHG",
    "ACEVG", "ACGHV", "ACPAHG", "ACSAV", "ACIRC", "ACSHC", "ACEVC", "ACTR",
    "ACPAHV", "ACSWDNLSM", "ACSWUPLSM", "ACLWDNLSM", "ACLWUPLSM", "ACSHFLSM", "ACLHFLSM", "ACGHFLSM",
    "ACPAHLSM", "ACCANHS", "SOILENERGY", "SNOWENERGY", "ACEFLXB", "GRAIN", "GDD", "QTDRAIN",
    "IRSIVOL", "IRMIVOL", "IRFIVOL", "IRELOSS", "IRRSPLH", "PCB", "PC", "LAKEMASK",
    "SST", "SST_INPUT",
)
_FULL_XY_INT_GENERIC_NAMES: tuple[str, ...] = (
    "IVGTYP", "ISLTYP", "CROPCAT", "PGS", "IRNUMSI", "IRNUMMI", "IRNUMFI",
)
_FULL_V_XY_GENERIC_NAMES: tuple[str, ...] = ("MF_VX_INV",)
_FULL_TIME_GENERIC_NAMES: tuple[str, ...] = (
    "HFX_FORCE", "LH_FORCE", "TSK_FORCE", "HFX_FORCE_TEND", "LH_FORCE_TEND", "TSK_FORCE_TEND",
    "RESM", "ZETATOP", "T00", "P00", "TLP", "TISO", "TLP_STRAT", "P_STRAT",
    "MAX_MSFTX", "MAX_MSFTY",
)
_FULL_TIME_INT_GENERIC_NAMES: tuple[str, ...] = (
    "BATHYMETRY_FLAG", "THIS_IS_AN_IDEAL_RUN", "ITIMESTEP", "GOT_VAR_SSO", "SAVE_TOPO_FROM_REAL",
)
_FULL_SEED_INT_GENERIC_NAMES: tuple[str, ...] = ("ISEEDARR_RAND_PERTURB",)


ONLINE_SURFACE_FIELDS = ("PREC_ACC_C","PREC_ACC_NC","SNOW_ACC_NC","WSPD10MAX","W_UP_MAX","W_DN_MAX",
                         "UP_HELI_MAX","W_MEAN","GRPL_MAX","REFD_MAX","HAIL_MAXK1","HAIL_MAX2D","POWER")
WRFOUT_VARIABLE_SPECS["POWER"] = _spec("POWER",XY,"XY ","Power generated by wind turbines","W",coordinates="XLONG XLAT XTIME")
SOLAR_SURFACE_FIELDS = ("SWDOWN", "SWDDIR", "SWDDNI", "SWDDIF", "GSW")
GODDARD_SURFACE_FIELDS = ('COD2D_OUT','CTOP2D_OUT')
# radiation_driver.F:2331-2334/2981-2984 and the retained Goddard
# diagnostics: native names for the SAME surface/TOA all-sky fluxes.
GODDARD_NATIVE_FLUXES = {
    'SLWDN':'LWDNB', 'SLWUP':'LWUPB', 'SSWDN':'SWDNB', 'SSWUP':'SWUPB',
    'TLWDN':'LWDNT', 'TLWUP':'LWUPT', 'TSWDN':'SWDNT', 'TSWUP':'SWUPT',
}
for _name in GODDARD_NATIVE_FLUXES:
    WRFOUT_VARIABLE_SPECS[_name]=_spec(_name,XY,'XY ',_name,'W m-2',coordinates='XLONG XLAT XTIME')
for _name,_description,_units in (
    ('COD2D_OUT','Column Cloud Optical Depth',' '),
    ('CTOP2D_OUT','Cloud-Top Pressure','mb'),
):
    WRFOUT_VARIABLE_SPECS[_name]=_spec(_name,XY,'XY ',_description,_units,coordinates='XLONG XLAT XTIME')
for _name, _description in (
    ("SWDDIR", "Shortwave surface downward direct irradiance"),
    ("SWDDNI", "Shortwave surface downward direct normal irradiance"),
    ("SWDDIF", "Shortwave surface downward diffuse irradiance"),
    ("GSW", "NET SHORT WAVE FLUX AT GROUND SURFACE"),
):
    WRFOUT_VARIABLE_SPECS[_name] = _spec(_name, XY, "XY ", _description, "W m-2", coordinates="XLONG XLAT XTIME")

for _name,_units,_description in (
    ("WSPD10MAX","m s-1","WIND SPD MAX 10 M"),
    ("W_UP_MAX","m s-1","MAX Z-WIND UPDRAFT"),
    ("W_DN_MAX","m s-1","MAX Z-WIND DOWNDRAFT"),
    ("UP_HELI_MAX","m2 s-2","MAX UPDRAFT HELICITY"),
    ("W_MEAN","m s-1","HOURLY MEAN Z-WIND"),
    ("GRPL_MAX","kg m-2","MAX COL INT GRAUPEL"),
    ("REFD_MAX","dbZ","MAX DERIVED RADAR REFL"),
    ("HAIL_MAXK1","m","MAX HAIL DIAMETER K=1"),
    ("HAIL_MAX2D","m","MAX HAIL DIAMETER ENTIRE COLUMN"),
):
    WRFOUT_VARIABLE_SPECS[_name]=_spec(_name,XY,"XY ",_description,_units,coordinates="XLONG XLAT XTIME")
WRFOUT_VARIABLE_SPECS["REFL_10CM"]=_spec("REFL_10CM",XYZ,"XYZ","Radar reflectivity (lamda = 10 cm)","dBZ",coordinates="XLONG XLAT XTIME")

for _bucket, _description in (
    ("PREC_ACC_C", "ACCUMULATED CUMULUS PRECIPITATION OVER prec_acc_dt PERIODS OF TIME"),
    ("PREC_ACC_NC", "ACCUMULATED GRID SCALE  PRECIPITATION OVER prec_acc_dt PERIODS OF TIME"),
    ("SNOW_ACC_NC", "ACCUMULATED SNOW WATER EQUIVALENT OVER prec_acc_dt PERIODS OF TIME"),
):
    WRFOUT_VARIABLE_SPECS[_bucket] = _spec(_bucket, XY, "XY ", _description, "mm", coordinates="XLONG XLAT XTIME")


def _register_full_wrfout_generic_specs() -> None:
    for name in _FULL_XYZ_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name, _spec(name, XYZ, "XYZ", name, "", coordinates="XLONG XLAT XTIME")
        )
    for name in _FULL_Z_XYZ_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name, _spec(name, Z_XYZ, "XYZ", name, "", stagger="Z", coordinates="XLONG XLAT XTIME")
        )
    for name in _FULL_XY_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name, _spec(name, XY, "XY ", name, "", coordinates="XLONG XLAT XTIME")
        )
    for name in _FULL_XY_INT_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name, _spec(name, XY, "XY ", name, "", coordinates="XLONG XLAT XTIME", dtype="i4")
        )
    for name in _FULL_V_XY_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(
            name,
            _spec(
                name,
                V_XY,
                "XY ",
                name,
                "",
                stagger="Y",
                coordinates="XLONG_V XLAT_V XTIME",
            ),
        )
    for name in _FULL_TIME_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(name, _spec(name, TIME_ONLY, "0  ", name, ""))
    for name in _FULL_TIME_INT_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(name, _spec(name, TIME_ONLY, "0  ", name, "", dtype="i4"))
    for name in _FULL_SEED_INT_GENERIC_NAMES:
        WRFOUT_VARIABLE_SPECS.setdefault(name, _spec(name, SEED, "Z  ", name, "", stagger="Z", dtype="i4"))


_register_full_wrfout_generic_specs()

# Registry.EM_COMMON EEPS/KEPS history reads the held physical fields. These
# extra variables are emitted only for bl=16/17; no output-time closure solve.
_EPSILON_HISTORY = {
    "PEK_PBL": ("qke","TKE from EEPS","m2 s-2"),
    "PEP_PBL": ("epsilon_pbl","TKE dissipation from EEPS","m2 s-3"),
    "TKE_PBL": ("qke","TKE from KEPS","m2 s-2"),
    "DISS_PBL": ("epsilon_pbl","TKE dissipation from KEPS","m2 s-3"),
    "TPE_PBL": ("tpe_pbl","TPE from KEPS","K2"),
    "PR_PBL": ("pr_pbl","Prandtl number from KEPS","-"),
}
for _name, (_leaf,_description,_units) in _EPSILON_HISTORY.items():
    if _name != "TKE_PBL":  # Existing non-epsilon full streams retain their spec.
        WRFOUT_VARIABLE_SPECS[_name] = _spec(_name,XYZ,"XYZ",_description,_units,coordinates="XLONG XLAT XTIME")


def _epsilon_history_fields(state,bl,shape,*,requested_names=None):
    names=("PEK_PBL","PEP_PBL") if int(bl)==16 else ("TKE_PBL","DISS_PBL","TPE_PBL","PR_PBL")
    fields={}
    for name in names:
        if requested_names is not None and name not in requested_names:
            continue
        leaf=_EPSILON_HISTORY[name][0]
        source=_lookup(state,leaf,None)
        if source is None: raise ValueError(f"epsilon PBL history missing held {leaf}")
        value=_coerce_array(name,source,shape,dtype=np.float32)
        if not np.isfinite(value).all(): raise ValueError(f"epsilon PBL history nonfinite {leaf}")
        if name == "TKE_PBL":
            # Registry.EM_COMMON:1230 is Z-staggered. KEPS:249-340 writes
            # kts:kte only; the unused top retains gen_allocs.c:235 zero init.
            padded=np.zeros((shape[0]+1,*shape[1:]),dtype=np.float32)
            padded[:-1]=value
            value=padded
        fields[name]=value
    return fields


# PBLs whose WRF shared tke_pbl (Registry.EM_COMMON:1230, Z, rh) the port holds
# in State.qke, with the factor from that leaf to WRF's TKE: MYJ/QNSE keep
# q2 = 2*TKE (myj_adapters.py:266, qnse_adapter.py:145; WRF TKE_MYJ=0.5*Q2K,
# module_bl_myjpbl.F:462, module_bl_qnsepbl.F); BouLac (TKE=tke_pbl,
# pbl_driver.F:2012), Shin-Hong and GBM (:2099) keep the TKE itself. CAM-UW
# (bl=9) holds it in its carry; the nested writer passes that as a diagnostic.
_QKE_TO_TKE_PBL = {2: 0.5, 4: 0.5, 8: 1.0, 11: 1.0, 12: 1.0}


def _pbl_tke_history(fields,state,shape,bl):
    """History TKE_PBL = the PBL's held shared array; the unused Z top keeps the
    allocation zero (each scheme writes kts:kte). Only the stream's existing slot
    is filled. Shin-Hong: shinhonginit seeds EPSQ2L/2 before lead-zero history
    (module_bl_shinhong.F:2270-2277), diag=1 rewrites it (:1601-1603); under
    CAM-MG its State.cam_tke_pbl (incl. top memory) is the shared array."""
    if "TKE_PBL" not in fields:
        return
    source=_lookup(state,"cam_tke_pbl",None) if bl == 11 else None
    if source is None:
        held=_lookup(state,"qke",None)
        if held is None: raise ValueError(f"bl={bl} TKE_PBL history missing held qke")
        source=np.zeros((shape[0]+1,*shape[1:]),dtype=np.float32)
        source[:-1]=np.float32(_QKE_TO_TKE_PBL[bl])*np.asarray(held,dtype=np.float32)
    value=_coerce_array("TKE_PBL",source,(shape[0]+1,*shape[1:]),dtype=np.float32)
    if not np.isfinite(value).all(): raise ValueError(f"bl={bl} TKE_PBL history nonfinite")
    fields["TKE_PBL"]=value

# Generic full-stream placeholders had blank units for active land/AC fields.
# These definitions are the real WRF inventory, independent of output values.
from gpuwrf.io.land_history_metadata import LAND_HISTORY_METADATA
for _land_name, (_land_description, _land_units) in LAND_HISTORY_METADATA.items():
    if _land_name in WRFOUT_VARIABLE_SPECS:
        WRFOUT_VARIABLE_SPECS[_land_name] = dataclass_replace(
            WRFOUT_VARIABLE_SPECS[_land_name], description=_land_description, units=_land_units,
        )


# SSiB Registry history inventory; no model/JAX import is added here.
from gpuwrf.io.ssib_history import SSIB_EXTRA_HISTORY,SSIB_STANDARD_HISTORY,SSIB_HISTORY_METADATA
for _ssib_name,(_ssib_description,_ssib_units,_ssib_stagger) in SSIB_HISTORY_METADATA.items():
    WRFOUT_VARIABLE_SPECS[_ssib_name]=_spec(_ssib_name,XY,"XY ",_ssib_description,_ssib_units,
        stagger=_ssib_stagger,coordinates="XLONG XLAT XTIME")


def write_wrfout_netcdf(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    path: str | Path,
    *,
    domain: str,
    domain_authority: WrfoutDomainAuthority,
    valid_time: datetime | date | str,
    lead_hours: float,
    run_start: datetime | date | str,
    diagnostics: Mapping[str, Any] | None = None,
    land_state: Any | None = None,
    variable_subset: tuple[str, ...] | frozenset[str] | None = None,
    include_mandatory_coords: bool = False,
    compress: bool = False,
    full_variable_set: bool = False,
) -> Path:
    """Write one WRF-style ``wrfout`` NetCDF file for the M7 minimum variable set.

    The function accepts plain Python/numpy objects as well as the project
    ``State``/``GridSpec`` objects. Device arrays, if passed after an operational
    run, are converted only at this output boundary.

    ``domain`` and ``domain_authority`` are mandatory and independent of the
    target path. They bind the emitted integer global ``GRID_ID`` to the
    authenticated run-grid metadata before any field is materialized.

    ``diagnostics`` optionally carries host-only output diagnostics/metadata.
    Static latitude/longitude payloads (``XLAT``/``XLONG`` and staggered variants)
    are selected from this map before the legacy State/projection lookup, so real
    WRF statics can be routed to output without adding JIT-visible state leaves.
    It also carries operational surface-layer diagnostics
    (e.g. the M9 surface map: ``T2``/``U10``/``V10``/``Q2``/``PSFC``/``SWDOWN``/
    ``GLW``/``PBLH``/``TSK`` plus the P0-5a additions ``QFX``/``GRDFLX``). When a
    name is present there, it OVERRIDES the state/default for that output field --
    the writer otherwise falls back to raw lowest-level fields which are
    physically wrong over terrain (e.g. raw level-1 wind/theta read far too
    strong/warm at a high summit). When ``diagnostics`` is ``None`` the numerical
    and variable payload is identical to the legacy path; only the separately
    required authenticated global ``GRID_ID`` metadata is added.

    ``land_state`` optionally carries the prognostic Noah-MP land carry
    (``NoahMPLandState``: 4-layer ``tslb``/``smois``/``sh2o``, bulk snow, canopy
    water, runoff, albedo, emissivity). When supplied, the WRF soil/snow land
    fields (``TSLB``/``SMOIS``/``SH2O``/``SNOW``/``SNOWH``/``CANWAT``/``SFROFF``/
    ``UDROFF``/``ALBEDO``/``EMISS``) are written; when ``None`` they are simply
    absent from the file (the writer never fabricates a soil profile).

    ``variable_subset`` optionally restricts the emitted variables to the named
    subset -- the compact training-output path (#122). ``include_mandatory_coords``
    additionally force-emits the geometry/eta/lat-lon coordinates and ``compress``
    applies lossless NetCDF4 compression; both default OFF. When ``variable_subset``
    is ``None`` (the default) the full uncompressed variable payload and IEEE
    values are unchanged; the global ``GRID_ID`` insertion is intentional.

    ``full_variable_set`` is the opt-in heavy WRF compatibility stream. When True
    the payload is expanded to :data:`FULL_WRFOUT_VARIABLES` (375 names including
    ``Times``), using real state/diagnostics where present and inactive defaults
    for WRF stream slots this port does not currently carry. Default False keeps
    the historical operational subset unchanged.
    """

    prepared = prepare_wrfout_payload(
        state,
        grid,
        namelist,
        path,
        domain=domain,
        domain_authority=domain_authority,
        valid_time=valid_time,
        lead_hours=lead_hours,
        run_start=run_start,
        diagnostics=diagnostics,
        land_state=land_state,
        variable_subset=variable_subset,
        include_mandatory_coords=include_mandatory_coords,
        full_variable_set=full_variable_set,
    )
    return write_prepared_wrfout(
        prepared,
        expected_domain=domain,
        expected_domain_authority=domain_authority,
        variable_subset=variable_subset,
        include_mandatory_coords=include_mandatory_coords,
        compress=compress,
    )


@dataclass(frozen=True)
class PreparedWrfout:
    """Fully host-materialized wrfout payload (v0.2.0 wall-clock win #3).

    Everything needed to write one wrfout NetCDF file with NO device-array or live
    model-state dependency: every field is already a host ``np.float32`` array
    (the device->host pull happened in :func:`prepare_wrfout_payload` while the
    GPU result was still resident). This object can therefore be handed to a
    background writer thread while the GPU advances the next forecast hour, with
    no risk of racing a donated/reused device buffer. The NetCDF arrays and
    metadata written are identical to the synchronous path -- only the wall-clock
    timing of the write changes.
    """

    target: Path
    dimensions: Mapping[str, int | None]
    fields: Mapping[str, np.ndarray]
    run_start_dt: datetime
    valid_dt: datetime
    lead_hours: float
    grid: Any
    namelist: Any
    domain: str
    domain_authority: WrfoutDomainAuthority
    full_variable_set: bool = False

    # Optional host-only restart journal; travels with asynchronous payloads.
    restart_publication: Any = None
    source_global_attrs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _WrfoutHostArrays:
    by_id: Mapping[int, Any]

    def host_value(self, value: Any) -> Any:
        return self.by_id.get(id(value), value)


@contextmanager
def _batched_wrfout_device_get(
    *,
    state: Any,
    grid: Any,
    diagnostics: Mapping[str, Any] | None,
    land_state: Any | None,
    requested_names: frozenset[str] | None = None,
    tracer_history: Any = None,
) -> Iterator[None]:
    """Materialize device leaves once for one wrfout payload build."""

    context = _build_wrfout_host_arrays(
        state,
        grid,
        diagnostics,
        land_state,
        requested_names=requested_names,
        tracer_history=tracer_history,
    )
    token = _WRFOUT_HOST_ARRAYS.set(context)
    try:
        yield
    finally:
        _WRFOUT_HOST_ARRAYS.reset(token)


def _build_wrfout_host_arrays(
    *sources: Any, requested_names: frozenset[str] | None = None, tracer_history: Any = None
) -> _WrfoutHostArrays:
    try:
        import jax
    except Exception:  # pragma: no cover - JAX is a project dependency.
        return _WrfoutHostArrays({})

    array_type = getattr(jax, "Array", ())
    if not array_type:
        return _WrfoutHostArrays({})

    leaves: list[Any] = []
    seen_objects: set[int] = set()
    seen_arrays: set[int] = set()

    def visit(value: Any) -> None:
        if value is None or isinstance(
            value, (str, bytes, bool, int, float, complex, date, datetime, Path)
        ):
            return
        if isinstance(value, array_type):
            value_id = id(value)
            if value_id not in seen_arrays:
                seen_arrays.add(value_id)
                leaves.append(value)
            return
        if isinstance(value, np.ndarray):
            return
        value_id = id(value)
        if value_id in seen_objects:
            return
        seen_objects.add(value_id)
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in dataclass_fields(value):
                visit(getattr(value, field.name, None))
            return
        namespace = getattr(value, "__dict__", None)
        if namespace:
            for item in namespace.values():
                visit(item)

    scan_sources = sources
    if requested_names is not None:
        state, grid, diagnostics, land_state = sources
        scan_sources = tuple(
            _requested_wrfout_source_values(
                state=state,
                grid=grid,
                diagnostics=diagnostics,
                land_state=land_state,
                requested_names=requested_names,
            )
        )
    for source in scan_sources:
        visit(source)
    if tracer_history is not None:
        visit(tracer_history.stack)
        visit(tracer_history.volumes)
        visit(tracer_history.finite)
    if not leaves:
        return _WrfoutHostArrays({})
    host_leaves = jax.device_get(tuple(leaves))
    return _WrfoutHostArrays(
        {id(device_value): host_value for device_value, host_value in zip(leaves, host_leaves)}
    )


def _host_materialized_value(value: Any) -> Any:
    context = _WRFOUT_HOST_ARRAYS.get()
    if context is None:
        return value
    return context.host_value(value)


from gpuwrf.io.lake_schema import specs as _lake_specs
WRFOUT_VARIABLE_SPECS.update(_lake_specs(_spec))
from gpuwrf.io.ocean_schema import FIELDS as OCEAN_FIELDS
WRFOUT_VARIABLE_SPECS.update({name.upper():_spec(name.upper(),XY,'XY ',
    'Ocean mixed layer '+name, 'K' if name in ('tml','t0ml','tmoml') else ('m2 s-1' if name in ('huml','hvml') else 'm')) for name in OCEAN_FIELDS})

from gpuwrf.io.temf_schema import TEMF_HISTORY
WRFOUT_VARIABLE_SPECS.update({name.upper():_spec(name.upper(),XYZ if volume else XY,'XYZ' if volume else 'XY ',description,units)
                             for name,description,units,volume in TEMF_HISTORY})
from gpuwrf.io.microphysics_history import (
    DIAGNOSTIC_METADATA as _MP_DIAGNOSTIC_METADATA,
    SCHEME_FIELDS as _MP_HISTORY_FIELDS,
    STATE_ALIASES as _MP_STATE_ALIASES,
    UNREPRESENTED_FIELDS as _MP_UNREPRESENTED_FIELDS,
)
WRFOUT_VARIABLE_SPECS.update({
    name: _spec(name, XYZ if volume else XY, "XYZ" if volume else "XY ",
                description, units, coordinates="XLONG XLAT XTIME")
    for name, (description, units, volume) in _MP_DIAGNOSTIC_METADATA.items()
})
from gpuwrf.io.stochastic_schema import history_specs as _stochastic_specs, SPECTRAL_FIELDS as _STOCH_SPECTRAL_FIELDS
WRFOUT_VARIABLE_SPECS.update(_stochastic_specs(_spec))
WRFOUT_VARIABLE_SPECS["HFX_FDDA"] = _spec("HFX_FDDA", XYZ, "XYZ", "PSEUDO RADIATIVE HEAT FLUX FROM TEMPERATURE FDDA", "W m-2", coordinates="XLONG XLAT XTIME")
# Restart's compatibility metadata remains its existing ABI. History-only
# corrections below must not silently alter authenticated restart formats.
RESTART_WRFOUT_VARIABLE_SPECS = dict(WRFOUT_VARIABLE_SPECS)
from gpuwrf.io.producer_history import FIELDS as _PRODUCER_HISTORY_FIELDS
from gpuwrf.io.wrf_variable_metadata import WRF_VARIABLE_METADATA as _PRODUCER_METADATA
for _name in _PRODUCER_HISTORY_FIELDS:
    _meta = _PRODUCER_METADATA[_name]
    _dims = (("Time", "num_stoch_levels", "south_north", "west_east")
             if _name.startswith("FIELD_") else XY if _meta["MemoryOrder"].strip() == "XY" else XYZ)
    WRFOUT_VARIABLE_SPECS[_name] = _spec(_name, _dims, _meta["MemoryOrder"],
        _meta["description"], _meta["units"], coordinates=_meta.get("coordinates"))
for _name, _description, _units, _volume in TEMF_HISTORY:
    RESTART_WRFOUT_VARIABLE_SPECS.pop(_name.upper(), None)
    RESTART_WRFOUT_VARIABLE_SPECS[_name] = _spec(
        _name, XYZ if _volume else XY, 'XYZ' if _volume else 'XY ', _description, _units)
from gpuwrf.io.wrf_variable_metadata import WRF_VARIABLE_METADATA as _WRF_METADATA
for _name, _metadata in _WRF_METADATA.items():
    if _name in WRFOUT_VARIABLE_SPECS:
        WRFOUT_VARIABLE_SPECS[_name] = dataclass_replace(
            WRFOUT_VARIABLE_SPECS[_name],
            memory_order=_metadata['MemoryOrder'], description=_metadata['description'],
            units=_metadata['units'], stagger=_metadata['stagger'],
            coordinates=_metadata.get('coordinates'),
        )
# WRF registry.stoch uses num_stoch_levels without the stagger suffix.
for _name, _specification in tuple(WRFOUT_VARIABLE_SPECS.items()):
    if 'num_stoch_levels_stag' in _specification.dimensions:
        WRFOUT_VARIABLE_SPECS[_name] = dataclass_replace(_specification, dimensions=tuple(
            'num_stoch_levels' if dim == 'num_stoch_levels_stag' else dim
            for dim in _specification.dimensions))


def _prepare_requested_names(
    variable_subset: tuple[str, ...] | frozenset[str] | None,
    *,
    include_mandatory_coords: bool,
) -> frozenset[str] | None:
    if variable_subset is None:
        return None
    requested = set(variable_subset)
    if include_mandatory_coords:
        requested.update(MANDATORY_WRFOUT_COORDINATES)
    requested.discard("Times")
    requested.discard("XTIME")
    return frozenset(requested)


_STATE_SOURCE_ALIASES: Mapping[str, tuple[str, ...]] = {
    **{name:(name,source) for name,source in GODDARD_NATIVE_FLUXES.items()},
    **_MP_STATE_ALIASES,
    "U": ("U", "u"),
    "V": ("V", "v"),
    "W": ("W", "w"),
    "T": ("theta", "THETA"),
    "THM": ("theta", "THETA"),
    "QVAPOR": ("QVAPOR", "qv", "qvapor"),
    "QCLOUD": ("QCLOUD", "qc", "qcloud"),
    "QICE": ("QICE", "qi", "qice"),
    "QRAIN": ("QRAIN", "qr", "qrain"),
    "QIR": ("QIR", "qir"), "QIB": ("QIB", "qib"),
    "QICE2": ("QICE2", "qi2"), "QNICE2": ("QNICE2", "Ni2"),
    "QIR2": ("QIR2", "qir2"), "QIB2": ("QIB2", "qib2"), "QZI": ("QZI", "qzi"),
    "QSNOW": ("QSNOW", "qs", "qsnow"),
    "QGRAUP": ("QGRAUP", "qg", "qgraup"),
    "QKE": ("QKE", "qke"),
    "TKE": ("TKE", "tke"),
    "CLDFRA": ("CLDFRA", "cldfra", "cloud_fraction"),
    "P": ("p_perturbation", "P"),
    "PB": ("pb", "p_base", "PB"),
    "PH": ("ph_perturbation", "PH"),
    "PHB": ("phb", "ph_base", "PHB"),
    "MU": ("mu_perturbation", "MU"),
    "MUB": ("mub", "mu_base", "MUB"),
    "U10": ("U10", "u10"),
    "V10": ("V10", "v10"),
    "T2": ("T2", "t2"),
    "Q2": ("Q2", "q2"),
    "PSFC": ("PSFC", "psfc"),
    "RAINC": ("RAINC", "rainc", "rainc_acc"),
    "RAINNC": ("RAINNC", "rainnc", "rain_acc", "thompson_rainnc", "ntu_rainnc"),
    "RAINSH": ("RAINSH", "rainsh"),
    "SWDOWN": ("SWDOWN", "swdown"),
    "GLW": ("GLW", "glw"),
    "PBLH": ("PBLH", "pblh"),
    "UST": ("UST", "ustar"),
    "HFX": ("HFX", "hfx"),
    "LH": ("LH", "lh"),
    "TSK": ("TSK", "tsk", "t_skin"),
    "HGT": ("HGT", "hgt", "terrain_height"),
    "LANDMASK": ("LANDMASK", "landmask"),
    "LU_INDEX": ("LU_INDEX", "lu_index", "ivgtyp"),
    "XLAT": ("XLAT", "xlat", "lat"),
    "XLONG": ("XLONG", "xlong", "lon"),
    "XLAT_U": ("XLAT_U", "xlat_u", "lat_u"),
    "XLONG_U": ("XLONG_U", "xlong_u", "lon_u"),
    "XLAT_V": ("XLAT_V", "xlat_v", "lat_v"),
    "XLONG_V": ("XLONG_V", "xlong_v", "lon_v"),
    "QNWFA2D": ("QNWFA2D", "nwfa2d"),
    "QNIFA2D": ("QNIFA2D", "nifa2d"),
    "CWM": ("CWM", "qt"),
    "SR": ("SR", "ferrier_sr", "goddard_sr", "thompson_sr", "ntu_sr"),
    "SNOWNC": ("SNOWNC", "snow_acc", "ice_acc", "thompson_snownc", "ntu_snownc"),
}


def _requested_wrfout_source_values(
    *,
    state: Any,
    grid: Any,
    diagnostics: Mapping[str, Any] | None,
    land_state: Any | None,
    requested_names: frozenset[str],
) -> list[Any]:
    del land_state
    values: list[Any] = []

    def add(value: Any) -> None:
        if value is not None:
            values.append(value)

    def add_aliases(source: Any, aliases: tuple[str, ...]) -> None:
        for alias in aliases:
            add(_lookup_raw(source, alias, None))

    names = _expand_requested_source_names(requested_names)
    # Health checks cover held online diagnostics even when a stream omits them.
    if diagnostics is not None:
        names.update(name for name in (*ONLINE_SURFACE_FIELDS,"REFL_10CM",*GODDARD_SURFACE_FIELDS) if name in diagnostics)
        if any(name in diagnostics for name in ("SWDDIR", "SWDDNI", "SWDDIF")):
            names.update(name for name in SOLAR_SURFACE_FIELDS if name in diagnostics)
    for name in names:
        aliases = ((_EPSILON_HISTORY[name][0],) if name in _EPSILON_HISTORY and
                   _lookup_raw(state,'epsilon_pbl',None) is not None else
                   _STATE_SOURCE_ALIASES.get(name, (name, name.lower())))
        add_aliases(state, aliases)
        add_aliases(diagnostics, aliases)

    # Total/base aliases used by _perturbation_base_pair fallbacks.
    if names & {"P", "PB", "PSFC", "T2", "TSK", "HFX", "LH"}:
        add_aliases(state, ("p_total", "p", "P_total"))
    if names & {"PH", "PHB", "PSFC", "HFX", "LH"}:
        add_aliases(state, ("ph_total", "ph", "PH_total"))
    if names & {"MU", "MUB", "PSFC"}:
        add_aliases(state, ("mu_total", "mu", "MU_total"))

    if names & {"HGT"}:
        add_aliases(grid, ("HGT", "hgt", "terrain_height"))
    if names & {"XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V"}:
        for coord in ("XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V"):
            add_aliases(diagnostics, _STATE_SOURCE_ALIASES[coord])
            add_aliases(state, _STATE_SOURCE_ALIASES[coord])

    if names & {"ZNU", "ZNW"}:
        add(_lookup_raw(grid, "eta_levels", None))

    metrics = _lookup_raw(grid, "metrics", None)
    metric_attrs = {
        "MAPFAC_M": ("msftx",),
        "MAPFAC_U": ("msfux",),
        "MAPFAC_V": ("msfvx",),
        "SINALPHA": ("sina",),
        "COSALPHA": ("cosa",),
        "F": ("f",),
        "E": ("e",),
    }
    if "PSFC" in names:
        metric_attrs.setdefault("PSFC", ())
        for attr in ("c1h", "c2h", "dnw", "p_top"):
            add(_lookup_raw(metrics, attr, None))
    for name, attrs in metric_attrs.items():
        if name in names:
            for attr in attrs:
                add(_lookup_raw(metrics, attr, None))
    return values


def _expand_requested_source_names(requested_names: frozenset[str]) -> set[str]:
    names = set(requested_names)
    if names & {"T", "THM", "T2", "TSK", "HFX", "LH"}:
        names.update({"T", "QVAPOR", "P", "PB"})
    if names & {"PSFC"}:
        names.update({"P", "PB", "PH", "PHB", "MU", "MUB"})
        names.update(MOIST_WRFOUT_NAME.values())
    if names & {"CLDFRA"}:
        names.update({"QCLOUD", "QICE", "QRAIN"})
    if names & {"U10", "HFX", "LH"}:
        names.add("U")
    if names & {"V10", "HFX", "LH"}:
        names.add("V")
    if names & {"HFX", "LH"}:
        names.update({"HGT", "LANDMASK", "TSK", "Q2", "UST"})
    if names & {"LU_INDEX"}:
        names.add("LANDMASK")
    if names & {"RAINNC"}:
        names.update({"QSNOW", "QGRAUP"})
    if names & {"OLR"}:
        names.add("LWUPT")
    return names


def prepare_wrfout_payload(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    path: str | Path,
    *,
    domain: str,
    domain_authority: WrfoutDomainAuthority,
    valid_time: datetime | date | str,
    lead_hours: float,
    run_start: datetime | date | str,
    diagnostics: Mapping[str, Any] | None = None,
    land_state: Any | None = None,
    variable_subset: tuple[str, ...] | frozenset[str] | None = None,
    include_mandatory_coords: bool = False,
    full_variable_set: bool = False,
    source_global_attrs: Mapping[str, Any] | None = None,
) -> PreparedWrfout:
    """Materialize all wrfout fields to host numpy (the device->host boundary).

    This is the part of :func:`write_wrfout_netcdf` that touches the live model
    state / device arrays. Calling it eagerly (before the GPU advances the next
    hour) lets the subsequent NetCDF write run on a background thread. ``grid`` and
    ``namelist`` are kept by reference for the global-attribute scalars only --
    those reads are pure host metadata (projection floats / dimensions), never
    device arrays, so they are safe to read on the writer thread.
    """

    target = Path(path)
    run_start_dt = _coerce_datetime(run_start)
    valid_dt = _coerce_datetime(valid_time)
    tracer_names = ()
    if getattr(state, "tracers", None) is not None:
        from gpuwrf.tracers.history import register_specs
        tracer_names = register_specs(namelist)
    nx, ny, nz = _grid_extent(grid)
    bound_authority = _validate_wrfout_domain_authority(
        domain_authority, expected_domain=domain, output_grid=grid
    )
    dimensions = _dimension_sizes(nx=nx, ny=ny, nz=nz, namelist=namelist,
        resolved_lsm=(source_global_attrs or {}).get('SF_SURFACE_PHYSICS'))
    requested_names = _prepare_requested_names(
        variable_subset,
        include_mandatory_coords=include_mandatory_coords,
    )
    tracer_history = None
    if tracer_names:
        from gpuwrf.tracers.history import select_history
        tracer_history = select_history(state, namelist, requested_names)
    # The single device->host boundary: materialize all reachable device leaves in
    # one jax.device_get call, then let the existing field builder coerce host
    # arrays in its historical order.
    with _batched_wrfout_device_get(
        state=state,
        grid=grid,
        diagnostics=diagnostics,
        land_state=land_state,
        requested_names=requested_names,
        tracer_history=tracer_history,
    ):
        if requested_names is None:
            fields = _build_output_fields(
                state, grid, namelist, dimensions,
                diagnostics=diagnostics, land_state=land_state,
                run_start=run_start_dt, lead_hours=float(lead_hours),
            )
        else:
            fields = _build_subset_output_fields(
                state, grid, namelist, dimensions,
                requested_names=requested_names,
                diagnostics=diagnostics, land_state=land_state,
                run_start=run_start_dt, lead_hours=float(lead_hours),
            )
        # Registry CLAT has u=(copy_fcnm): initial feedback can change the
        # parent's computational latitude while XLAT remains unchanged. Keep
        # the supplied producer value after the compatibility builders (whose
        # CLAT fallback is XLAT), for both full and selected history streams.
        if requested_names is None or "CLAT" in requested_names:
            clat = _optional_field_array(diagnostics, ("CLAT", "clat"), (ny, nx))
            if clat is None:
                clat = _optional_field_array(state, ("CLAT", "clat"), (ny, nx))
            if clat is not None:
                fields["CLAT"] = clat
        _add_microphysics_history_fields(
            fields, state=state, namelist=namelist, dimensions=dimensions,
            diagnostics=diagnostics, requested_names=requested_names,
        )
        if full_variable_set:
            _add_full_wrfout_fields(
                fields,
                state=state,
                grid=grid,
                namelist=namelist,
                dimensions=dimensions,
                diagnostics=diagnostics,
                land_state=land_state,
                run_start=run_start_dt,
                lead_hours=float(lead_hours),
                requested_names=requested_names,
            )
        for name in _MP_UNREPRESENTED_FIELDS.get(int(_lookup(namelist,"mp_physics",0)),()):
            fields.pop(name,None)
        from gpuwrf.io.producer_history import add_fields as _add_producer_history
        _add_producer_history(fields, state=state, namelist=namelist, diagnostics=diagnostics,
            dimensions=dimensions, requested_names=requested_names, lookup=_lookup,
            coerce=_coerce_array, specs=WRFOUT_VARIABLE_SPECS,
            shape_for_dimensions=_shape_for_dimensions)
        if _lookup(namelist,'bl_pbl_physics') in (16,17):
            fields.update(_epsilon_history_fields(
                state,_lookup(namelist,'bl_pbl_physics'),(nz,ny,nx),
                requested_names=requested_names,
            ))
        if _lookup(namelist,'bl_pbl_physics') in _QKE_TO_TKE_PBL:
            _pbl_tke_history(fields,state,(nz,ny,nx),int(_lookup(namelist,'bl_pbl_physics')))
        if tracer_names:
            from gpuwrf.tracers.history import history_fields
            fields.update(history_fields(state, namelist, requested_names, selection=tracer_history))
        # Explicit WRF land history wins over defaults and raw sibling-carry
        # values (notably the initialized water tile). Host-only output work.
        from gpuwrf.io.land_history import LAND_HISTORY_FIELDS
        if _lookup(namelist,'bl_pbl_physics') == 10:
            for name in (*(row[0] for row in TEMF_HISTORY),'TH2'):
                output_name = name.upper()
                if requested_names is not None and output_name not in requested_names:
                    continue
                if diagnostics is None or diagnostics.get(name) is None:
                    raise ValueError(f'TEMF history is missing held {name}; refusing invented output')
                spec=WRFOUT_VARIABLE_SPECS[output_name]
                fields[output_name]=_coerce_array(output_name,diagnostics[name],_shape_for_dimensions(spec.dimensions,dimensions),
                                          dtype=_numpy_dtype_for_spec(spec))
        for name in LAND_HISTORY_FIELDS:
            if diagnostics is None or diagnostics.get(name) is None:
                continue
            if requested_names is not None and name not in requested_names:
                continue
            spec = WRFOUT_VARIABLE_SPECS.get(name)
            if spec is None:  # Internal budget leaves need not belong to the history stream.
                continue
            fields[name] = _coerce_array(name, diagnostics[name],
                _shape_for_dimensions(spec.dimensions, dimensions), dtype=_numpy_dtype_for_spec(spec))
        if _lookup(namelist,"sf_surface_physics")==8:
            for name in (*SSIB_EXTRA_HISTORY,*SSIB_STANDARD_HISTORY,"TSLB","SMOIS","ZS","DZS"):
                if diagnostics is None or diagnostics.get(name) is None: continue
                if requested_names is not None and name not in requested_names: continue
                spec=WRFOUT_VARIABLE_SPECS[name]
                fields[name]=_coerce_array(name,diagnostics[name],_shape_for_dimensions(spec.dimensions,dimensions),
                                           dtype=_numpy_dtype_for_spec(spec))
        if _lookup(namelist, "sf_surface_physics") == 3:
            from gpuwrf.coupling.ruc_surface_hook import RUC_HISTORY_FIELDS
            for name in (*RUC_HISTORY_FIELDS, "ZS", "DZS"):
                if diagnostics is None or diagnostics.get(name) is None:
                    continue
                if requested_names is not None and name not in requested_names:
                    continue
                spec = WRFOUT_VARIABLE_SPECS[name]
                fields[name] = _coerce_array(name, diagnostics[name],
                    _shape_for_dimensions(spec.dimensions, dimensions), dtype=_numpy_dtype_for_spec(spec))
            if "T2" in fields and "PSFC" in fields and (requested_names is None or "TH2" in requested_names):
                # SFCDIAGS_RUCLSM's final TH2 uses WRF REAL at each operation.
                # Reconstruct this algebraic diagnostic from the HELD T2/PSFC;
                # an extra prognostic carry plane would duplicate these values.
                cp_ratio = np.float32(_lookup(namelist, "ruc_static").cfg.rovcp)
                fields["TH2"] = (np.asarray(fields["T2"], np.float32)
                                  * (np.float32(1.e5) / np.asarray(fields["PSFC"], np.float32)) ** cp_ratio)
        if diagnostics is not None and "TH2" in diagnostics and (
                float(lead_hours)==0.0 or diagnostics.get('_TWO_WAY_TH2',False)
                or int(_lookup(namelist, "sf_sfclay_physics", 5)) == 0):
            if requested_names is None or "TH2" in requested_names:
                fields["TH2"] = _coerce_array("TH2", diagnostics["TH2"], (ny, nx))
        # These are held model diagnostics, including WRF's pre-solve t0 values.
        # Apply them after the compatibility builder so a hydrometeor proxy or
        # cold-start turbulence seed cannot replace the requested history leaf.
        if diagnostics is not None:
            solar_names = (SOLAR_SURFACE_FIELDS if any(name in diagnostics for name in
                           ("SWDDIR", "SWDDNI", "SWDDIF")) else ())
            for name in (*ONLINE_SURFACE_FIELDS,"REFL_10CM", *solar_names,*GODDARD_SURFACE_FIELDS):
                value=diagnostics.get(name)
                if value is not None and not np.isfinite(_host_materialized_value(value)).all():
                    kind="precipitation" if name in ONLINE_SURFACE_FIELDS[:3] else "online diagnostic"
                    raise ValueError(f"nonfinite {kind} history field {name}")
            for name in ("CLDFRA", "QC_BL", "CLDFRA_BL", "QKE", "DTAUX3D", "DTAUY3D", "T", "REFL_10CM", "HFX_FDDA"):
                if (requested_names is None or name in requested_names) and name in diagnostics:
                    if name == "HFX_FDDA" and not np.isfinite(_host_materialized_value(diagnostics[name])).all():
                        raise ValueError("nonfinite FDDA history field HFX_FDDA")
                    fields[name] = _coerce_array(name, diagnostics[name], (nz, ny, nx))
            for name in ("DUSFCG", "DVSFCG", *GODDARD_SURFACE_FIELDS):
                if (requested_names is None or name in requested_names) and name in diagnostics:
                    fields[name] = _coerce_array(name, diagnostics[name], (ny, nx))
            if _lookup(namelist,'ra_lw_physics') == 5 and _lookup(namelist,'ra_sw_physics') == 5:
                for name,source in GODDARD_NATIVE_FLUXES.items():
                    if requested_names is not None and name not in requested_names:
                        continue
                    value=diagnostics.get(name,diagnostics.get(source))
                    if value is not None:
                        if not np.isfinite(_host_materialized_value(value)).all():
                            raise ValueError(f'nonfinite Goddard native flux {name}')
                        fields[name]=_coerce_array(name,value,(ny,nx))
            if _lookup(namelist,'stochastic_config') is not None:
                for name in _STOCH_SPECTRAL_FIELDS:
                    if name not in diagnostics or (requested_names is not None and name not in requested_names):
                        continue
                    value=_host_materialized_value(diagnostics[name])
                    if not np.isfinite(value).all():
                        raise ValueError(f'nonfinite stochastic spectral history {name}')
                    spec=WRFOUT_VARIABLE_SPECS[name]
                    fields[name]=_coerce_array(name,value,_shape_for_dimensions(spec.dimensions,dimensions))
    if _lookup(namelist,'sf_ocean_physics') == 1:
        for name in OCEAN_FIELDS:
            key=name.upper()
            if requested_names is None or key in requested_names:
                if diagnostics is None or key not in diagnostics:
                    raise ValueError(f'ocean history missing resident {key}')
                fields[key]=_coerce_array(key,diagnostics[key],(ny,nx))
    if _lookup(namelist, 'noahmp_mmf_enabled'):
        for name in MMF_HISTORY_FIELDS:
            if requested_names is not None and name not in requested_names:
                continue
            if diagnostics is None or diagnostics.get(name) is None:
                raise ValueError(f'MMF history missing held {name}; refusing invented output')
            spec = WRFOUT_VARIABLE_SPECS[name]
            fields[name] = _coerce_array(name, diagnostics[name],
                _shape_for_dimensions(spec.dimensions, dimensions), dtype=np.float32)
    if _lookup(namelist,'sf_lake_physics') == 1:
        from gpuwrf.io.lake_schema import LAKE_HISTORY_FIELDS
        bundle = _lookup(namelist,'lake_bundle')
        if bundle is None or diagnostics is None:
            raise ValueError('lake history requires lake_bundle and held lake fields')
        for name in LAKE_HISTORY_FIELDS:
            if requested_names is None or name in requested_names:
                if name not in diagnostics:
                    raise ValueError(f'lake history missing {name}')
                spec = WRFOUT_VARIABLE_SPECS[name]
                fields[name] = _coerce_array(name,diagnostics[name],_shape_for_dimensions(spec.dimensions,dimensions))
        for key in ('HFX','LH','GRDFLX','TSK','QFX','T2','TH2','Q2','ALBEDO'):
            if requested_names is not None and key not in requested_names:
                continue
            compact = diagnostics.get('_LAKE_'+key)
            if compact is not None:
                from gpuwrf.coupling.lake_surface_hook import scatter_columns
                old = fields.get(key,diagnostics.get(key,np.zeros((ny,nx),np.float32)))
                fields[key] = np.asarray(scatter_columns(old,compact,bundle),np.float32)
            elif key in diagnostics:
                fields[key] = _coerce_array(key,diagnostics[key],(ny,nx))
    if _lookup(namelist,"stochastic_config") is not None:
        from gpuwrf.io.stochastic_schema import HISTORY_FIELDS
        if diagnostics is None:
            raise ValueError("active stochastic history requires held patterns and seeds")
        for name in HISTORY_FIELDS:
            if name in _STOCH_SPECTRAL_FIELDS:
                continue  # already materialized INSIDE the single host batch
            if name in diagnostics and (requested_names is None or name in requested_names):
                spec = WRFOUT_VARIABLE_SPECS[name]
                fields[name] = _coerce_array(name,diagnostics[name],
                    _shape_for_dimensions(spec.dimensions,dimensions),dtype=_numpy_dtype_for_spec(spec))
    return PreparedWrfout(
        target=target,
        dimensions=dimensions,
        fields=fields,
        run_start_dt=run_start_dt,
        valid_dt=valid_dt,
        lead_hours=float(lead_hours),
        grid=grid,
        namelist=namelist,
        domain=domain,
        domain_authority=bound_authority,
        full_variable_set=bool(full_variable_set),
        source_global_attrs=dict(source_global_attrs) if source_global_attrs is not None else None,
    )


def write_prepared_wrfout(
    prepared: PreparedWrfout,
    *,
    expected_domain: str,
    expected_domain_authority: WrfoutDomainAuthority,
    variable_subset: tuple[str, ...] | frozenset[str] | None = None,
    target_override: Path | None = None,
    include_mandatory_coords: bool = False,
    compress: bool = False,
) -> Path:
    """Write a :class:`PreparedWrfout` to NetCDF. Pure host work; thread-safe.

    Contains NO device-array access, so it is safe to run on a background writer
    thread while the GPU advances. The arrays and metadata are identical to the synchronous
    :func:`write_wrfout_netcdf` path.

    ``variable_subset`` optionally restricts the emitted variables to the named
    subset -- a stream-generic hook for a secondary WRF ``auxhist`` history stream
    (e.g. a surface-only set). When ``None`` (the default) EVERY prepared field is
    written exactly as before, apart from required authenticated ``GRID_ID``
    metadata. The ``Times``/``XTIME`` time coordinates and the global attributes
    are ALWAYS written regardless of the subset (a stream-valid WRF history frame
    always carries its time stamp -- matching how WRF stamps every auxhist frame).
    A name in ``variable_subset`` that is absent from the prepared payload is
    simply skipped (the file never fabricates a field), and the canonical
    operational write order is preserved.

    ``include_mandatory_coords`` (#122 training output): additionally force-emit the
    geometry/eta/staggered-lat-lon coordinates in :data:`MANDATORY_WRFOUT_COORDINATES`
    so a subset frame is a self-contained, stream-valid WRF file. OFF by default so
    the existing auxhist surface stream keeps its exact prior variable set (a coord
    still absent from the payload is skipped -- never fabricated).

    ``compress`` (#122 training output): apply lossless NetCDF4 zlib compression to
    the written variables (shrinking the ~10 GB/day training target). OFF by default
    so existing callers keep the same variable encoding; compression is lossless
    so values read back are bit-identical regardless.

    ``expected_domain`` and ``expected_domain_authority`` are supplied separately
    by the scheduler/writer owner at this final boundary. They must exactly match
    the prepared binding, so replacing both fields inside ``PreparedWrfout`` does
    not replace the trusted expectation.

    ``target_override`` writes the same host payload to a different path without a
    second device->host pull -- used by the auxhist stream, which reuses the main
    stream's already-materialized :class:`PreparedWrfout`.
    """

    bound_authority = _validate_wrfout_domain_authority(
        expected_domain_authority,
        expected_domain=expected_domain,
        output_grid=prepared.grid,
        dimensions=prepared.dimensions,
    )
    if prepared.domain != expected_domain or prepared.domain_authority != bound_authority:
        raise ValueError("WRFOUT_PREPARED_AUTHORITY_SUBSTITUTION")
    target = Path(target_override) if target_override is not None else prepared.target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"WRFOUT_TARGET_EXISTS:{target}")
    subset = None if variable_subset is None else frozenset(variable_subset)
    if subset is not None and include_mandatory_coords:
        # Self-contained training frame: also emit the coordinate/dimension vars (a
        # coord still absent from the payload is skipped below, never fabricated).
        subset = subset | MANDATORY_WRFOUT_COORDINATES
    # Lossless NetCDF4 zlib compression, opt-in (#122 training stream). Default OFF
    # keeps every existing caller's variable encoding unchanged.
    compression = _SUBSET_STREAM_COMPRESSION if compress else None
    dimensions = prepared.dimensions
    temporary_fd, temporary = _create_wrfout_temporary(target)
    os.close(temporary_fd)
    try:
        with Dataset(temporary, "w", format="NETCDF4") as dataset:
            _create_dimensions(dataset, dimensions)
            _write_global_attrs(
                dataset, prepared.grid, prepared.namelist, dimensions,
                prepared.run_start_dt, prepared.valid_dt, bound_authority,
                use_theta_m=parse_use_theta_m(_lookup(prepared.namelist, "use_theta_m", 1)),
                source_global_attrs=prepared.source_global_attrs,
            )
            _write_times(dataset, prepared.valid_dt)
            if _lookup(prepared.namelist,'isftcflx',0):
                dataset.ISFTCFLX=np.int32(_lookup(prepared.namelist,'isftcflx'))
            if _lookup(prepared.namelist,'sf_ocean_physics') == 1:
                for key,default in (('sf_ocean_physics',0),('oml_hml0',50.),('oml_gamma',.14),('oml_relaxation_time',0.)):
                    dtype=np.int32 if key=='sf_ocean_physics' else np.float32
                    dataset.setncattr(key.upper(),dtype(_lookup(prepared.namelist,key,default)))
            # Write in the canonical operational order, but emit ONLY the fields
            # actually prepared. Optional sources self-gate; ``subset`` further
            # restricts to a stream's requested variables.
            if prepared.full_variable_set:
                write_order = FULL_WRFOUT_VARIABLES
            else:
                _write_xtime(dataset, prepared.run_start_dt, prepared.lead_hours)
                write_order = OPERATIONAL_WRFOUT_VARIABLES
            write_order = (*write_order, *(name for name in ("U10E", "V10E", "QC_BL", "CLDFRA_BL", *ONLINE_SURFACE_FIELDS, "REFL_10CM", *SOLAR_SURFACE_FIELDS, *GODDARD_SURFACE_FIELDS, *GODDARD_NATIVE_FLUXES,
                                         "CWM", "QRIMEF", "F_ICE_PHY", "F_RAIN_PHY", "F_RIMEF_PHY",
                                         "RE_CLOUD_GSFC", "RE_RAIN_GSFC", "RE_ICE_GSFC", "RE_SNOW_GSFC", "RE_GRAUPEL_GSFC", "RE_HAIL_GSFC", "PHYS_TOT", "QNWFA2D", "QNIFA2D")
                                         if name in prepared.fields and name not in write_order))
            write_order = (*write_order, *(name for name in _MP_HISTORY_FIELDS.get(
                int(_lookup(prepared.namelist, "mp_physics", 0)), ())
                if name in prepared.fields and name not in write_order))
            write_order = (*write_order, *(name for name in _PRODUCER_HISTORY_FIELDS
                if name in prepared.fields and name not in write_order))
            if getattr(prepared.namelist, "tracer_opt", 0) or getattr(prepared.namelist, "tracer_config", None) is not None:
                from gpuwrf.tracers.history import register_specs
                write_order += tuple(n for n in register_specs(prepared.namelist) if n in prepared.fields and n not in write_order)
            if _lookup(prepared.namelist,"sf_surface_physics")==8:
                write_order=(*write_order,*(name for name in (*SSIB_EXTRA_HISTORY,*SSIB_STANDARD_HISTORY,"TSLB","SMOIS","ZS","DZS") if name not in write_order))
            if _lookup(prepared.namelist, "sf_surface_physics") == 3:
                write_order = (*write_order, *(name for name in _RUC_EXTRA_HISTORY if name not in write_order))
            if _lookup(prepared.namelist,'bl_pbl_physics') == 10:
                write_order = (*write_order,*(name.upper() for name in (*(row[0] for row in TEMF_HISTORY),'TH2') if name.upper() not in write_order))
            if _lookup(prepared.namelist,'sf_ocean_physics') == 1:
                write_order=(*write_order,*(name.upper() for name in OCEAN_FIELDS if name.upper() not in write_order))
            if _lookup(prepared.namelist,'noahmp_mmf_enabled'):
                write_order=(*write_order,*(name for name in MMF_HISTORY_FIELDS
                    if name in prepared.fields and name not in write_order))
            if _lookup(prepared.namelist,'sf_lake_physics') == 1:
                from gpuwrf.io.lake_schema import LAKE_HISTORY_FIELDS
                write_order = (*write_order,*(name for name in (*LAKE_HISTORY_FIELDS,'GRDFLX','TH2','ALBEDO') if name not in write_order))
            if _lookup(prepared.namelist,"stochastic_config") is not None:
                from gpuwrf.io.stochastic_schema import HISTORY_FIELDS
                write_order = (*write_order,*(name for name in HISTORY_FIELDS if name not in write_order))
            if "HFX_FDDA" in prepared.fields and "HFX_FDDA" not in write_order:
                write_order = (*write_order, "HFX_FDDA")
            if _lookup(prepared.namelist,'bl_pbl_physics') in (16,17):
                write_order=(*write_order,*(name for name in _EPSILON_HISTORY if name in prepared.fields and name not in write_order))
            for name in write_order:
                if name == "Times":
                    continue
                if name == "XTIME":
                    if prepared.full_variable_set:
                        _write_xtime(dataset, prepared.run_start_dt, prepared.lead_hours)
                    continue
                if name not in prepared.fields:
                    continue
                if subset is not None and name not in subset:
                    continue
                spec = WRFOUT_VARIABLE_SPECS[name]
                _write_float_variable(
                    dataset, spec, prepared.fields[name], dimensions, compression
                )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if prepared.restart_publication is None:
            _publish_wrfout_noreplace(temporary, target)
        else:
            prepared.restart_publication(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _publish_wrfout_noreplace(temporary: Path, target: Path) -> None:
    """Atomically publish one completed frame without replacing any directory entry."""

    if temporary.parent != target.parent:
        raise ValueError("WRFOUT_PUBLICATION_DIRECTORY_MISMATCH")
    directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            os.link(
                temporary.name,
                target.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"WRFOUT_TARGET_EXISTS:{target}") from exc
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _create_wrfout_temporary(target: Path) -> tuple[int, Path]:
    """Create an unpredictable same-directory file exclusively, respecting umask."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for _ in range(128):
        temporary = target.parent / (
            f".{target.name}.grid-id-{secrets.token_hex(16)}.tmp"
        )
        try:
            return os.open(temporary, flags, 0o666), temporary
        except FileExistsError:
            continue
    raise FileExistsError("WRFOUT_TEMPORARY_NAME_EXHAUSTED")


def _dimension_sizes(*, nx: int, ny: int, nz: int, namelist: Mapping[str, Any] | Any | None,
                     history_schema: bool = True, resolved_lsm=None) -> dict[str, int | None]:
    soil_layers = int(_lookup(namelist, "soil_layers_stag", DEFAULT_SOIL_LAYERS))
    # WRF module_check_a_mundo.F:set_physics_rconfigs (3566-3590).
    # Omitted selector retains the compatibility writer's explicit dimensions.
    lsm = _lookup(namelist, "sf_surface_physics")
    if lsm is None:
        lsm=resolved_lsm  # operational disabled LSM uses None; raw WRF selector is0
    if history_schema and lsm in (0, 1, 2, 4, 5, 7, 8):
        soil_layers = {0: 5, 1: 5, 2: 4, 4: 4, 5: 10, 7: 2, 8: 3}[lsm]
    elif lsm == 8:
        soil_layers = 3
    if _lookup(namelist, "sf_surface_physics") == 3:
        static = _lookup(namelist, "ruc_static")
        if static is None:
            raise ValueError("RUC history requires ruc_static soil geometry")
        soil_layers = static.cfg.nzs
    snow_layers = int(_lookup(namelist, "snow_layers_stag", DEFAULT_SNOW_LAYERS))
    snso_layers = int(_lookup(namelist, "snso_layers_stag", DEFAULT_SNSO_LAYERS))
    seed_dim = int(_lookup(namelist, "seed_dim_stag", DEFAULT_SEED_DIM))
    result = {
        "Time": None,
        "DateStrLen": DATE_STR_LEN,
        "west_east": int(nx),
        "west_east_stag": int(nx) + 1,
        "south_north": int(ny),
        "south_north_stag": int(ny) + 1,
        "bottom_top": int(nz),
        "bottom_top_stag": int(nz) + 1,
        "soil_layers_stag": soil_layers,
        "snow_layers_stag": snow_layers,
        "snso_layers_stag": snso_layers,
        "seed_dim_stag": seed_dim,
    }
    if _lookup(namelist,'sf_lake_physics') == 1:
        from gpuwrf.io.lake_schema import LAKE_DIMENSIONS
        result.update(LAKE_DIMENSIONS)
    if _lookup(namelist,"stochastic_config") is not None:
        result["num_stoch_levels" if history_schema else "num_stoch_levels_stag"] = int(nz)+1
    return result


def _create_dimensions(dataset: Dataset, dimensions: Mapping[str, int | None]) -> None:
    for name in (
        "Time",
        "DateStrLen",
        "west_east",
        "west_east_stag",
        "south_north",
        "south_north_stag",
        "bottom_top",
        "bottom_top_stag",
        "soil_layers_stag",
        "snow_layers_stag",
        "snso_layers_stag",
        "seed_dim_stag",
        *(name for name in ('soil_levels_or_lake_levels_stag','snow_and_soil_levels_stag','interface_levels_stag') if name in dimensions),
        *(name for name in ("num_stoch_levels", "num_stoch_levels_stag") if name in dimensions),
    ):
        dataset.createDimension(name, dimensions[name])


def _write_times(dataset: Dataset, valid_time: datetime) -> None:
    times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
    encoded = _wrf_time_string(valid_time).encode("ascii")
    times[0, :] = np.frombuffer(encoded, dtype="S1")


def _write_xtime(dataset: Dataset, run_start: datetime, lead_hours: float) -> None:
    spec = WRFOUT_VARIABLE_SPECS["XTIME"]
    xtime = dataset.createVariable("XTIME", spec.dtype, spec.dimensions)
    xtime[:] = np.asarray([float(lead_hours) * 60.0], dtype=np.float32)
    _set_variable_attrs(xtime, spec)
    time_label = run_start.strftime("%Y-%m-%d %H:%M:%S")
    xtime.description = f"minutes since {time_label}"
    xtime.units = f"minutes since {time_label}"


def _write_float_variable(
    dataset: Dataset,
    spec: WrfoutVariableSpec,
    data: np.ndarray,
    dimensions: Mapping[str, int | None],
    compression: Mapping[str, object] | None = None,
) -> None:
    expected_shape = _shape_for_dimensions(spec.dimensions, dimensions)
    array = _coerce_array(spec.name, data, expected_shape, dtype=_numpy_dtype_for_spec(spec))
    # ``compression`` (zlib/complevel) is passed ONLY for subset streams; the
    # default full-output path leaves it None so the variable encoding is
    # unchanged for existing callers. Compression is lossless, so values read back
    # are bit-identical regardless.
    kwargs = dict(compression) if compression else {}
    variable = dataset.createVariable(spec.name, spec.dtype, spec.dimensions, **kwargs)
    _set_variable_attrs(variable, spec)
    variable[0, ...] = array


def _set_variable_attrs(variable: Any, spec: WrfoutVariableSpec) -> None:
    # WRF tags real fields FieldType=104 and integer fields FieldType=106
    # (module_io_int / Registry). Derive it from the spec dtype so integer
    # diagnostics (ISNOW, the stochastic seed arrays) carry the correct code.
    variable.FieldType = np.int32(106 if str(spec.dtype).startswith("i") else 104)
    variable.MemoryOrder = spec.memory_order
    variable.description = spec.description
    variable.units = spec.units
    variable.stagger = spec.stagger
    if spec.coordinates is not None:
        variable.coordinates = spec.coordinates


def _write_global_attrs(
    dataset: Dataset,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
    run_start: datetime,
    valid_time: datetime,
    domain_authority: WrfoutDomainAuthority | None = None,
    use_theta_m: int | None = None,
    source_global_attrs: Mapping[str, Any] | None = None,
) -> None:
    # ``None`` is retained only for the separate wrfrst compatibility writer,
    # which imports this common-attribute helper. Every wrfout entry point
    # validates and supplies a domain authority before opening its Dataset.
    if domain_authority is not None and "GRID_ID" in dataset.ncattrs():
        raise ValueError("WRFOUT_GRID_ID_DUPLICATE_OR_CONFLICT")
    projection = _lookup(grid, "projection")
    lat_0 = float(_lookup(projection, "lat_0", _lookup(namelist, "cen_lat", 0.0)))
    lon_0 = float(_lookup(projection, "lon_0", _lookup(namelist, "cen_lon", 0.0)))
    dx_m = float(_lookup(projection, "dx_m", _lookup(namelist, "dx", 0.0)))
    dy_m = float(_lookup(projection, "dy_m", _lookup(namelist, "dy", dx_m)))
    kind = str(_lookup(projection, "kind", "lambert")).lower()
    map_proj = {"lambert": 1, "polar": 2, "mercator": 3}.get(kind, 1)

    attrs: dict[str, Any] = {
        "TITLE": str(_lookup(namelist, "title", " OUTPUT FROM GPUWRF WRF-COMPATIBLE NETCDF WRITER")),
        "START_DATE": _wrf_time_string(run_start),
        "SIMULATION_START_DATE": _wrf_time_string(run_start),
        "WEST-EAST_GRID_DIMENSION": np.int32(int(dimensions["west_east_stag"])),
        "SOUTH-NORTH_GRID_DIMENSION": np.int32(int(dimensions["south_north_stag"])),
        "BOTTOM-TOP_GRID_DIMENSION": np.int32(int(dimensions["bottom_top_stag"])),
        "DX": np.float32(dx_m),
        "DY": np.float32(dy_m),
        "GRIDTYPE": "C",
        "MAP_PROJ": np.int32(map_proj),
        "CEN_LAT": np.float32(_lookup(namelist, "cen_lat", lat_0)),
        "CEN_LON": np.float32(_lookup(namelist, "cen_lon", lon_0)),
        "TRUELAT1": np.float32(_lookup(namelist, "truelat1", lat_0)),
        "TRUELAT2": np.float32(_lookup(namelist, "truelat2", lat_0)),
        "MOAD_CEN_LAT": np.float32(_lookup(namelist, "moad_cen_lat", lat_0)),
        "STAND_LON": np.float32(_lookup(namelist, "stand_lon", lon_0)),
        "GMT": np.float32(run_start.hour + run_start.minute / 60.0),
        "JULYR": np.int32(run_start.year),
        "JULDAY": np.int32(run_start.timetuple().tm_yday),
        "ISWATER": np.int32(_lookup(namelist, "iswater", 17)),
        "ISLAKE": np.int32(_lookup(namelist, "islake", 21)),
        "ISICE": np.int32(_lookup(namelist, "isice", 15)),
        "ISURBAN": np.int32(_lookup(namelist, "isurban", 13)),
        "ISOILWATER": np.int32(_lookup(namelist, "isoilwater", 14)),
        "Conventions": "WRF-ARW",
        "history": "Created by gpuwrf.io.wrfout_writer.write_wrfout_netcdf",
        "PARENT_ID": np.int32(_lookup(namelist, "parent_id", 1)),
        "I_PARENT_START": np.int32(_lookup(grid, "i_parent_start", _lookup(namelist, "i_parent_start", 1))),
        "J_PARENT_START": np.int32(_lookup(grid, "j_parent_start", _lookup(namelist, "j_parent_start", 1))),
        "PARENT_GRID_RATIO": np.int32(_lookup(grid, "parent_grid_ratio", _lookup(namelist, "parent_grid_ratio", 1))),
        "BUCKET_MM": np.float32(_lookup(namelist, "bucket_mm", -1.0)),
        "BUCKET_J": np.float32(_lookup(namelist, "bucket_j", -1.0)),
        "DT": np.float32(_lookup(namelist, "dt_s", _lookup(namelist, "time_step", 0.0))),
        "MMINLU": str(_lookup(namelist, "mminlu", "MODIFIED_IGBP_MODIS_NOAH")),
        "NUM_LAND_CAT": np.int32(_lookup(namelist, "num_land_cat", 21)),
    }
    if source_global_attrs is not None:
        # WRF writes input geometry plus resolved history controls. Preserve those
        # scalars instead of approximating projection/land-use metadata here.
        # The authenticated GRID_ID and GPU writer provenance remain writer-owned.
        attrs.update({name: value for name, value in source_global_attrs.items()
                      if name not in {"GRID_ID", "TITLE", "START_DATE", "SIMULATION_START_DATE",
                                      "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
                                      "BOTTOM-TOP_GRID_DIMENSION", "DX", "DY",
                                      "Conventions", "history"}})
    if use_theta_m is not None:
        # WRF share/output_wrf.F:679 writes the integer global USE_THETA_M (wrfout only here; restart is M3R's).
        attrs["USE_THETA_M"] = np.int32(parse_use_theta_m(use_theta_m))
    if _lookup(namelist,'sf_lake_physics') == 1:
        attrs['SF_LAKE_PHYSICS'] = np.int32(1)
    for name, value in attrs.items():
        dataset.setncattr(name, value)
    if domain_authority is not None:
        _set_wrfout_grid_id_attr(dataset, domain_authority)


def _set_wrfout_grid_id_attr(
    dataset: Dataset, domain_authority: WrfoutDomainAuthority
) -> None:
    """Insert the one writer-owned global attribute without overwrite semantics."""

    if type(domain_authority) is not WrfoutDomainAuthority:
        raise ValueError("WRFOUT_DOMAIN_AUTHORITY_MISSING_OR_TYPE")
    if "GRID_ID" in dataset.ncattrs():
        raise ValueError("WRFOUT_GRID_ID_DUPLICATE_OR_CONFLICT")
    dataset.setncattr("GRID_ID", np.int32(domain_authority.grid_id))


def _moist_hydrostatic_psfc(
    state: Any, grid: Any, shape_xyz: tuple[int, ...],
    mu_pert: np.ndarray, mu_base: np.ndarray, p_pert: np.ndarray,
) -> np.ndarray | None:
    """WRF phy_prep moist surface pressure for either direct writer route.

    PSFC = p_hyd_w(kts): surface_driver.F:1988, first_rk_step_part1.F:1400,
    big_step_utilities_em.F:4946-4958. All represented moist mass members
    contribute; particle numbers, volumes and rime partitions do not.
    Return None without usable hybrid metrics, preserving the caller's
    existing compatibility extrapolation. Reduction precision is unchanged.
    """
    _psfc_default = None
    _psfc_metrics = _lookup(grid, "metrics")
    if _psfc_metrics is not None:
        _c1h = _lookup(_psfc_metrics, "c1h")
        _c2h = _lookup(_psfc_metrics, "c2h")
        _dnw = _lookup(_psfc_metrics, "dnw")
        _p_top = _lookup(_psfc_metrics, "p_top")
        if all(v is not None for v in (_c1h, _c2h, _dnw, _p_top)):
            _c1h = np.asarray(_c1h, dtype=np.float64)
            _c2h = np.asarray(_c2h, dtype=np.float64)
            _dnw = np.asarray(_dnw, dtype=np.float64)
            _nz = shape_xyz[0]
            if _c1h.shape == (_nz,) and _c2h.shape == (_nz,) and _dnw.shape == (_nz,):
                # Normalize writer aliases and missing inactive core slots on
                # the host; the shared selector owns moist membership/order.
                # Scalar zeros avoid allocating missing core volume arrays.
                _moist = {}
                for _name, _history_name in MOIST_WRFOUT_NAME.items():
                    _value = _optional_field_array(
                        state, _STATE_SOURCE_ALIASES[_history_name], shape_xyz
                    )
                    _moist[_name] = (np.float32(0) if _value is None and
                                     _name in MOIST_SPECIES else _value)
                _qtot = total_moist_mass(SimpleNamespace(**_moist))
                _mut = mu_pert + mu_base
                _dp_dry = (
                    _c1h[:, None, None] * _mut[None, :, :] + _c2h[:, None, None]
                ) * (-_dnw[:, None, None])
                _terms = np.asarray((1.0 + _qtot) * _dp_dry, dtype=np.float64)
                # v0.20 fp32 INTEGRATION bit-identity fix (secondary source): the S4
                # merge unconditionally switched this PSFC column integral to Kahan
                # compensated summation, which rounds differently from the historical
                # plain .sum(axis=0) and broke fp64_default PSFC bit-identity. Kahan
                # is only needed for the perturbation-authoritative fp32 mode (its
                # accumulator is fp32-sensitive); gate on the perturbation storage
                # dtype so fp64_default re-emits the exact pre-S4 plain sum.
                _mixed_fp32_psfc = (
                    np.asarray(mu_pert).dtype == np.float32
                    or np.asarray(p_pert).dtype == np.float32
                )
                if _mixed_fp32_psfc:
                    _total = np.zeros_like(_terms[0], dtype=np.float64)
                    _comp = np.zeros_like(_total, dtype=np.float64)
                    for _k in range(_terms.shape[0]):
                        _y = _terms[_k] - _comp
                        _t = _total + _y
                        _comp = (_t - _total) - _y
                        _total = _t
                    _psfc_default = float(np.asarray(_p_top).reshape(-1)[0]) + (_total - _comp)
                else:
                    # fp64_default: historical plain vertical sum (byte-identical).
                    _psfc_default = (
                        float(np.asarray(_p_top).reshape(-1)[0]) + _terms.sum(axis=0)
                    )
    return _psfc_default


def _build_subset_output_fields(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
    *,
    requested_names: frozenset[str],
    diagnostics: Mapping[str, Any] | None = None,
    land_state: Any | None = None,
    run_start: datetime | None = None,
    lead_hours: float = 0.0,
) -> dict[str, np.ndarray]:
    del land_state, run_start, lead_hours
    shape_xy = _shape_for_dimensions(XY, dimensions)
    shape_xyz = _shape_for_dimensions(XYZ, dimensions)
    shape_u = _shape_for_dimensions(U_XYZ, dimensions)
    shape_v = _shape_for_dimensions(V_XYZ, dimensions)
    shape_w = _shape_for_dimensions(W_XYZ, dimensions)
    shape_u_xy = _shape_for_dimensions(U_XY, dimensions)
    shape_v_xy = _shape_for_dimensions(V_XY, dimensions)
    shape_z = _shape_for_dimensions(Z_XYZ, dimensions)

    fields: dict[str, np.ndarray] = {}
    requested = set(requested_names)

    def want(name: str) -> bool:
        return name in requested

    need_u = bool(requested & {"U", "U10", "HFX", "LH"})
    need_v = bool(requested & {"V", "V10", "HFX", "LH"})
    need_w = want("W")
    need_theta = bool(requested & {"T", "THM", "T2", "TSK", "HFX", "LH"})
    need_qv = bool(requested & {"T", "THM", "QVAPOR", "Q2", "T2", "TSK", "HFX", "LH"})
    need_qc = bool(requested & {"QCLOUD", "CLDFRA"})
    need_qi = bool(requested & {"QICE", "CLDFRA"})
    need_qr = bool(requested & {"QRAIN", "CLDFRA"})
    need_pressure = bool(requested & {"P", "PB", "T2", "TSK", "PSFC", "HFX", "LH"})
    need_geopotential = bool(requested & {"PH", "PHB", "PSFC", "HFX", "LH"})
    need_mu = want("PSFC")

    u = _field_array(state, ("U", "u"), shape_u) if need_u else None
    v = _field_array(state, ("V", "v"), shape_v) if need_v else None
    w = _field_array(state, ("W", "w"), shape_w) if need_w else None
    theta = (
        _field_array(state, ("theta", "THETA"), shape_xyz, default=300.0)
        if need_theta
        else None
    )
    qv = _field_array(state, ("QVAPOR", "qv", "qvapor"), shape_xyz) if need_qv else None
    qc = _field_array(state, ("QCLOUD", "qc", "qcloud"), shape_xyz) if need_qc else None
    qi = _field_array(state, ("QICE", "qi", "qice"), shape_xyz) if need_qi else None
    qr = _field_array(state, ("QRAIN", "qr", "qrain"), shape_xyz) if need_qr else None
    if theta is not None and qv is not None:
        theta_dry = _theta_dry_for_output(state, theta, qv)
    else:
        theta_dry = theta

    p_pert = p_base = None
    if need_pressure:
        p_pert, p_base = _perturbation_base_pair(
            state,
            total_names=("p_total", "p", "P_total"),
            perturbation_names=("p_perturbation", "P"),
            base_names=("pb", "p_base", "PB"),
            shape=shape_xyz,
            base_source=diagnostics,
        )
    ph_pert = ph_base = None
    if need_geopotential:
        ph_pert, ph_base = _perturbation_base_pair(
            state,
            total_names=("ph_total", "ph", "PH_total"),
            perturbation_names=("ph_perturbation", "PH"),
            base_names=("phb", "ph_base", "PHB"),
            shape=shape_z,
            base_source=diagnostics,
        )
    mu_pert = mu_base = None
    if need_mu:
        mu_pert, mu_base = _perturbation_base_pair(
            state,
            total_names=("mu_total", "mu", "MU_total"),
            perturbation_names=("mu_perturbation", "MU"),
            base_names=("mub", "mu_base", "MUB"),
            shape=shape_xy,
            base_source=diagnostics,
        )

    if want("U") and u is not None:
        fields["U"] = u
    if want("V") and v is not None:
        fields["V"] = v
    if want("W") and w is not None:
        fields["W"] = w
    if want("T") and theta_dry is not None:
        fields["T"] = theta_dry - P0_THETA_OFFSET_K
    if want("THM") and theta is not None:
        # WRF THM follows the prognostic: theta_m (use_theta_m=1) or dry theta (0); T is dry for both.
        fields["THM"] = thm_source(theta, theta_dry, parse_use_theta_m(_lookup(namelist, "use_theta_m", 1))) - P0_THETA_OFFSET_K
    if want("QVAPOR") and qv is not None:
        fields["QVAPOR"] = qv
    if want("QCLOUD") and qc is not None:
        fields["QCLOUD"] = qc
    if want("QICE") and qi is not None:
        fields["QICE"] = qi
    if want("QRAIN") and qr is not None:
        fields["QRAIN"] = qr
    for wrf_name, state_names in (
        ("QSNOW", ("QSNOW", "qs", "qsnow")),
        ("QGRAUP", ("QGRAUP", "qg", "qgraup")),
        ("QKE", ("QKE", "qke")),
        ("TKE", ("TKE", "tke")),
    ):
        # EEPS/KEPS store physical TKE in State.qke. Registry history uses
        # PEK_PBL/TKE_PBL; it must not be labelled as MYNN's twice-TKE QKE.
        if wrf_name == "QKE" and _lookup(namelist, "bl_pbl_physics", 5) in (16, 17):
            continue
        if want(wrf_name):
            value = _optional_field_array(state, state_names, shape_xyz)
            if value is not None:
                fields[wrf_name] = value
    if want("CLDFRA"):
        cldfra_default = None
        if qc is not None and qi is not None and qr is not None:
            cldfra_default = np.where((qc + qi + qr) > 1.0e-8, 1.0, 0.0)
        fields["CLDFRA"] = _field_array(
            state,
            ("CLDFRA", "cldfra", "cloud_fraction"),
            shape_xyz,
            default=0.0 if cldfra_default is None else cldfra_default,
        )

    if want("P") and p_pert is not None:
        fields["P"] = p_pert
    if want("PB") and p_base is not None:
        fields["PB"] = p_base
    if want("PH") and ph_pert is not None:
        fields["PH"] = ph_pert
    if want("PHB") and ph_base is not None:
        fields["PHB"] = ph_base
    if want("MU") and mu_pert is not None:
        fields["MU"] = mu_pert
    if want("MUB") and mu_base is not None:
        fields["MUB"] = mu_base

    if requested & {"XLAT", "XLONG"}:
        xlat, xlong = _latlon_fields(state, grid, namelist, shape_xy, diagnostics=diagnostics)
        if want("XLAT"):
            fields["XLAT"] = xlat
        if want("XLONG"):
            fields["XLONG"] = xlong
    if requested & {"XLAT_U", "XLONG_U"}:
        xlat_u, xlong_u = _latlon_fields(
            state, grid, namelist, shape_u_xy, suffix="_u", diagnostics=diagnostics
        )
        if want("XLAT_U"):
            fields["XLAT_U"] = xlat_u
        if want("XLONG_U"):
            fields["XLONG_U"] = xlong_u
    if requested & {"XLAT_V", "XLONG_V"}:
        xlat_v, xlong_v = _latlon_fields(
            state, grid, namelist, shape_v_xy, suffix="_v", diagnostics=diagnostics
        )
        if want("XLAT_V"):
            fields["XLAT_V"] = xlat_v
        if want("XLONG_V"):
            fields["XLONG_V"] = xlong_v

    terrain = None
    if requested & {"HGT", "HFX", "LH"}:
        terrain = _grid_or_state_array(state, grid, ("HGT", "hgt", "terrain_height"), shape_xy)
        if want("HGT"):
            fields["HGT"] = terrain
    landmask = None
    if requested & {"LANDMASK", "LU_INDEX", "HFX", "LH"}:
        landmask = _landmask(state, shape_xy)
        if want("LANDMASK"):
            fields["LANDMASK"] = landmask
    if want("LU_INDEX"):
        default_lu = 17.0 if landmask is None else np.where(landmask > 0.5, 2.0, 17.0)
        fields["LU_INDEX"] = _field_array(
            state, ("LU_INDEX", "lu_index", "ivgtyp"), shape_xy, default=default_lu
        )

    if want("U10") and u is not None:
        fields["U10"] = _field_array(state, ("U10", "u10"), shape_xy, default=_unstagger_x(u)[0])
    if want("V10") and v is not None:
        fields["V10"] = _field_array(state, ("V10", "v10"), shape_xy, default=_unstagger_y(v)[0])
    if want("Q2") and qv is not None:
        fields["Q2"] = _field_array(state, ("Q2", "q2"), shape_xy, default=qv[0])
    if want("T2") and theta_dry is not None and p_pert is not None and p_base is not None:
        t_default = theta_dry[0] * (np.maximum(p_pert[0] + p_base[0], 1.0) / P0_PA) ** R_D_OVER_CP
        fields["T2"] = _field_array(state, ("T2", "t2"), shape_xy, default=t_default)
    if want("TSK") and theta_dry is not None and p_pert is not None and p_base is not None:
        tsk_default = theta_dry[0] * (np.maximum(p_pert[0] + p_base[0], 1.0) / P0_PA) ** R_D_OVER_CP
        fields["TSK"] = _field_array(state, ("TSK", "tsk", "t_skin"), shape_xy, default=tsk_default)
    if want("PSFC"):
        psfc_default = _moist_hydrostatic_psfc(
            state, grid, shape_xyz, mu_pert, mu_base, p_pert
        )
        if psfc_default is None:
            p_total = p_pert + p_base
            phi = ph_pert + ph_base
            phi0 = phi[0]
            phi1 = 0.5 * (phi[0] + phi[1])
            phi2 = 0.5 * (phi[1] + phi[2])
            weight = (phi0 - phi2) / (phi1 - phi2)
            psfc_default = weight * p_total[0] + (1.0 - weight) * p_total[1]
        fields["PSFC"] = _field_array(state, ("PSFC", "psfc"), shape_xy, default=psfc_default)

    for name, aliases in (
        ("RAINC", ("RAINC", "rainc", "rainc_acc")),
        ("RAINSH", ("RAINSH", "rainsh")),
        ("SWDOWN", ("SWDOWN", "swdown")),
        ("GLW", ("GLW", "glw")),
        ("PBLH", ("PBLH", "pblh")),
        ("UST", ("UST", "ustar")),
        ("HFX", ("HFX", "hfx")),
        ("LH", ("LH", "lh")),
    ):
        if want(name):
            fields[name] = _field_array(state, aliases, shape_xy)
    if want("RAINNC"):
        rain_acc = _optional_field_array(state, ("rain_acc", "RAINNC"), shape_xy)
        snow_acc = _optional_field_array(state, ("snow_acc", "SNOWNC"), shape_xy)
        ice_acc = _optional_field_array(state, ("ice_acc",), shape_xy)
        graupel_acc = _optional_field_array(state, ("graupel_acc", "GRAUPELNC"), shape_xy)
        hail_acc = _optional_field_array(state, ("hail_acc", "HAILNC"), shape_xy)
        if any(a is not None for a in (snow_acc, graupel_acc, ice_acc, hail_acc)):
            def _zero_xy(a: np.ndarray | None) -> np.ndarray:
                return np.asarray(a, dtype=np.float64) if a is not None else np.zeros(shape_xy, dtype=np.float64)

            rainnc_total = (
                _zero_xy(rain_acc)
                + _zero_xy(snow_acc)
                + _zero_xy(graupel_acc)
                + _zero_xy(ice_acc)
                + _zero_xy(hail_acc)
            )
            fields["RAINNC"] = _coerce_array("RAINNC", rainnc_total, shape_xy)
        else:
            fields["RAINNC"] = _field_array(state, ("RAINNC", "rainnc", "rain_acc"), shape_xy)

    for name, leaf in (("PHYS_TOT", "phys_tot"), ("CWM", "qt"), ("QRIMEF", "qrimef"), ("F_ICE_PHY", "f_ice_phy"),
                       ("F_RAIN_PHY", "f_rain_phy"), ("F_RIMEF_PHY", "f_rimef_phy"),
                       *((name.upper(), name) for name in ("re_cloud_gsfc", "re_rain_gsfc", "re_ice_gsfc",
                                                           "re_snow_gsfc", "re_graupel_gsfc", "re_hail_gsfc"))):
        if want(name):
            value = _optional_field_array(state, (name, leaf), shape_xyz)
            if value is not None:
                fields[name] = value
    # mp38 holds the original overlapping REAL totals, whose per-step rounding
    # cannot be recovered by summing the independent component accumulators.
    for name, leaf in (("RAINNC", "thompson_rainnc"), ("SNOWNC", "thompson_snownc"), ("RAINNC", "ntu_rainnc"), ("SNOWNC", "ntu_snownc"), ("RAINNC", "cam_rainnc"), ("SNOWNC", "cam_snownc")):
        if want(name):
            value = _optional_field_array(state, (leaf,), shape_xy)
            if value is not None:
                fields[name] = value
    for name, leaf in (("QNWFA2D", "nwfa2d"), ("QNIFA2D", "nifa2d")):
        if want(name):
            value = _optional_field_array(state, (name, leaf), shape_xy)
            if value is not None:
                fields[name] = value
    if want("SR"):
        value = _optional_field_array(state, ("ferrier_sr", "goddard_sr", "thompson_sr", "ntu_sr", "cam_sr"), shape_xy)
        if value is not None:
            fields["SR"] = value
    if requested & {"ZNU", "ZNW", "MAPFAC_M", "SINALPHA", "COSALPHA"}:
        _add_subset_grid_fields(fields, grid, dimensions, requested_names=requested_names, shape_xy=shape_xy)

    _apply_subset_diagnostics(fields, diagnostics, requested_names=requested_names, shape_xy=shape_xy)

    ordered: dict[str, np.ndarray] = {}
    for name in OPERATIONAL_WRFOUT_VARIABLES:
        if name in fields and name in requested:
            ordered[name] = _materialized_dtype(name, fields[name])
    for name, value in fields.items():
        if name in requested and name not in ordered:
            ordered[name] = _materialized_dtype(name, value)
    return ordered


def _add_subset_grid_fields(
    fields: dict[str, np.ndarray],
    grid: Any,
    dimensions: Mapping[str, int | None],
    *,
    requested_names: frozenset[str],
    shape_xy: tuple[int, int],
) -> None:
    nz = int(dimensions["bottom_top"])
    eta = _lookup(grid, "eta_levels")
    if eta is not None and ({"ZNU", "ZNW"} & requested_names):
        eta_host = np.asarray(eta, dtype=np.float64)
        if eta_host.shape == (nz + 1,):
            if "ZNW" in requested_names:
                fields["ZNW"] = _coerce_array("ZNW", eta_host, (nz + 1,))
            if "ZNU" in requested_names:
                fields["ZNU"] = _coerce_array("ZNU", 0.5 * (eta_host[:-1] + eta_host[1:]), (nz,))

    metrics = _lookup(grid, "metrics")
    if metrics is None:
        return
    for wrf_name, attr in (
        ("MAPFAC_M", "msftx"),
        ("SINALPHA", "sina"),
        ("COSALPHA", "cosa"),
    ):
        if wrf_name not in requested_names:
            continue
        value = _lookup(metrics, attr)
        if value is not None:
            fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), shape_xy)


def _apply_subset_diagnostics(
    fields: dict[str, np.ndarray],
    diagnostics: Mapping[str, Any] | None,
    *,
    requested_names: frozenset[str],
    shape_xy: tuple[int, int],
) -> None:
    if diagnostics is None:
        return
    diagnostic_surface_fields = {
        "U10E", "V10E",
        *ONLINE_SURFACE_FIELDS,
        *SOLAR_SURFACE_FIELDS,
        *GODDARD_SURFACE_FIELDS,
        "PREC_ACC_C", "PREC_ACC_NC", "SNOW_ACC_NC",
        "T2", "U10", "V10", "Q2", "PSFC", "SWDOWN", "GLW", "PBLH", "UST",
        "HFX", "LH", "TSK", "QFX", "GRDFLX",
        "SWDNB", "SWUPB", "LWDNB", "LWUPB",
        "SWDNT", "SWUPT", "LWDNT", "LWUPT", "SWNORM",
        "SWDNBC", "SWUPBC", "LWDNBC", "LWUPBC",
        "SWDNTC", "SWUPTC", "LWDNTC", "LWUPTC",
        "OLR",
    }
    for name in requested_names & diagnostic_surface_fields:
        value = diagnostics.get(name)
        if name in ("PREC_ACC_C", "PREC_ACC_NC", "SNOW_ACC_NC") and value is not None and not np.isfinite(_host_materialized_value(value)).all():
            raise ValueError(f"nonfinite precipitation history field {name}")
        if value is not None:
            fields[name] = _coerce_array(name, value, shape_xy)
    if "OLR" in requested_names and "OLR" not in fields:
        lwupt = diagnostics.get("LWUPT")
        if lwupt is not None:
            fields["OLR"] = _coerce_array("OLR", lwupt, shape_xy)


def _build_output_fields(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
    *,
    diagnostics: Mapping[str, Any] | None = None,
    land_state: Any | None = None,
    run_start: datetime | None = None,
    lead_hours: float = 0.0,
) -> dict[str, np.ndarray]:
    shape_xy = _shape_for_dimensions(XY, dimensions)
    shape_xyz = _shape_for_dimensions(XYZ, dimensions)
    shape_u = _shape_for_dimensions(U_XYZ, dimensions)
    shape_v = _shape_for_dimensions(V_XYZ, dimensions)
    shape_w = _shape_for_dimensions(W_XYZ, dimensions)
    shape_u_xy = _shape_for_dimensions(U_XY, dimensions)
    shape_v_xy = _shape_for_dimensions(V_XY, dimensions)
    shape_z = _shape_for_dimensions(Z_XYZ, dimensions)
    shape_soil = _shape_for_dimensions(SOIL, dimensions)
    shape_snow = _shape_for_dimensions(SNOW, dimensions)
    shape_snso = _shape_for_dimensions(SNSO, dimensions)
    shape_seed = _shape_for_dimensions(SEED, dimensions)
    shape_mapu = _shape_for_dimensions(MAPFAC_U_XY, dimensions)
    shape_mapv = _shape_for_dimensions(MAPFAC_V_XY, dimensions)

    u = _field_array(state, ("U", "u"), shape_u)
    v = _field_array(state, ("V", "v"), shape_v)
    w = _field_array(state, ("W", "w"), shape_w)
    theta = _field_array(state, ("theta", "THETA"), shape_xyz, default=300.0)
    qv = _field_array(state, ("QVAPOR", "qv", "qvapor"), shape_xyz)
    # State.theta is moist theta_m (use_theta_m=1); WRF-compatible ``T`` and the
    # temperature-derived diagnostics below need the DRY theta view (identical
    # when qv = 0, e.g. synthetic/test states).
    theta_dry = _theta_dry_for_output(state, theta, qv)
    qc = _field_array(state, ("QCLOUD", "qc", "qcloud"), shape_xyz)
    qi = _field_array(state, ("QICE", "qi", "qice"), shape_xyz)
    qr = _field_array(state, ("QRAIN", "qr", "qrain"), shape_xyz)

    p_pert, p_base = _perturbation_base_pair(
        state,
        total_names=("p_total", "p", "P_total"),
        perturbation_names=("p_perturbation", "P"),
        base_names=("pb", "p_base", "PB"),
        shape=shape_xyz,
        base_source=diagnostics,
    )
    ph_pert, ph_base = _perturbation_base_pair(
        state,
        total_names=("ph_total", "ph", "PH_total"),
        perturbation_names=("ph_perturbation", "PH"),
        base_names=("phb", "ph_base", "PHB"),
        shape=shape_z,
        base_source=diagnostics,
    )
    mu_pert, mu_base = _perturbation_base_pair(
        state,
        total_names=("mu_total", "mu", "MU_total"),
        perturbation_names=("mu_perturbation", "MU"),
        base_names=("mub", "mu_base", "MUB"),
        shape=shape_xy,
        base_source=diagnostics,
    )
    _psfc_default = _moist_hydrostatic_psfc(
        state, grid, shape_xyz, mu_pert, mu_base, p_pert
    )
    if _psfc_default is None:
        _p_total = p_pert + p_base
        _phi = ph_pert + ph_base
        _phi0 = _phi[0]
        _phi1 = 0.5 * (_phi[0] + _phi[1])
        _phi2 = 0.5 * (_phi[1] + _phi[2])
        _w1 = (_phi0 - _phi2) / (_phi1 - _phi2)
        _psfc_default = _w1 * _p_total[0] + (1.0 - _w1) * _p_total[1]

    xlat, xlong = _latlon_fields(state, grid, namelist, shape_xy, diagnostics=diagnostics)
    xlat_u, xlong_u = _latlon_fields(
        state, grid, namelist, shape_u_xy, suffix="_u", diagnostics=diagnostics
    )
    xlat_v, xlong_v = _latlon_fields(
        state, grid, namelist, shape_v_xy, suffix="_v", diagnostics=diagnostics
    )

    terrain = _grid_or_state_array(state, grid, ("HGT", "hgt", "terrain_height"), shape_xy)
    landmask = _landmask(state, shape_xy)
    lu_index = _field_array(state, ("LU_INDEX", "lu_index", "ivgtyp"), shape_xy, default=np.where(landmask > 0.5, 2.0, 17.0))
    hfx = _optional_field_array(state, ("HFX", "hfx"), shape_xy)
    lh = _optional_field_array(state, ("LH", "lh"), shape_xy)
    if (hfx is None or lh is None) and isinstance(diagnostics, Mapping):
        # Operational HFX/LH override the fallback below anyway; skip its re-solve.
        if diagnostics.get("HFX") is not None and diagnostics.get("LH") is not None:
            hfx = diagnostics["HFX"] if hfx is None else hfx
            lh = diagnostics["LH"] if lh is None else lh
    if hfx is None or lh is None:
        surface_fluxes = _surface_flux_fallbacks(
            state=state,
            grid=grid,
            theta=theta_dry,
            qv=qv,
            p_total=p_pert + p_base,
            ph_total=ph_pert + ph_base,
            u=u,
            v=v,
            landmask=landmask,
            shape_xy=shape_xy,
        )
        if hfx is None:
            hfx = surface_fluxes["HFX"]
        if lh is None:
            lh = surface_fluxes["LH"]

    fields = {
        "XLAT": xlat,
        "XLONG": xlong,
        "XLAT_U": xlat_u,
        "XLONG_U": xlong_u,
        "XLAT_V": xlat_v,
        "XLONG_V": xlong_v,
        "HGT": terrain,
        "LANDMASK": landmask,
        "LU_INDEX": lu_index,
        "U": u,
        "V": v,
        "W": w,
        "T": theta_dry - P0_THETA_OFFSET_K,
        "THM": thm_source(theta, theta_dry, parse_use_theta_m(_lookup(namelist, "use_theta_m", 1))) - P0_THETA_OFFSET_K,
        "QVAPOR": qv,
        "P": p_pert,
        "PB": p_base,
        "PH": ph_pert,
        "PHB": ph_base,
        "MU": mu_pert,
        "MUB": mu_base,
        "QCLOUD": qc,
        "QICE": qi,
        "QRAIN": qr,
        "CLDFRA": _field_array(
            state,
            ("CLDFRA", "cldfra", "cloud_fraction"),
            shape_xyz,
            default=np.where((qc + qi + qr) > 1.0e-8, 1.0, 0.0),
        ),
        "U10": _field_array(state, ("U10", "u10"), shape_xy, default=_unstagger_x(u)[0]),
        "V10": _field_array(state, ("V10", "v10"), shape_xy, default=_unstagger_y(v)[0]),
        # T2/TSK fall back to lowest-level *actual* temperature (theta -> T via the
        # local Exner factor), NOT raw potential temperature: theta[0] over high
        # terrain (e.g. Teide ~3.4km, psfc~660hPa) reads ~+34K too warm when mislabeled
        # as a 2-m/skin temperature. Level-1 air T is a sane proxy when the real
        # surface diagnostic is absent. (The proper fix routes the operational
        # surface-layer T2/U10/V10 diagnostics into the writer state; see task.)
        "T2": _field_array(state, ("T2", "t2"), shape_xy, default=theta_dry[0] * (np.maximum(p_pert[0] + p_base[0], 1.0) / P0_PA) ** R_D_OVER_CP),
        "Q2": _field_array(state, ("Q2", "q2"), shape_xy, default=qv[0]),
        "PSFC": _field_array(state, ("PSFC", "psfc"), shape_xy, default=_psfc_default),
        "RAINC": _field_array(state, ("RAINC", "rainc", "rainc_acc"), shape_xy),
        "RAINNC": _field_array(state, ("RAINNC", "rainnc", "rain_acc"), shape_xy),
        "RAINSH": _field_array(state, ("RAINSH", "rainsh"), shape_xy),
        "SWDOWN": _field_array(state, ("SWDOWN", "swdown"), shape_xy),
        "GLW": _field_array(state, ("GLW", "glw"), shape_xy),
        "PBLH": _field_array(state, ("PBLH", "pblh"), shape_xy),
        "UST": _field_array(state, ("UST", "ustar"), shape_xy),
        "HFX": hfx,
        "LH": lh,
        "TSK": _field_array(state, ("TSK", "tsk", "t_skin"), shape_xy, default=theta_dry[0] * (np.maximum(p_pert[0] + p_base[0], 1.0) / P0_PA) ** R_D_OVER_CP),
    }

    # --- P0-5a (a) extra hydrometeors + number concentrations + MYNN TKE. ---
    # Source = prognostic State leaves (Thompson qs/qg/Ni/Nr, MYNN qke). These
    # leaves always exist in the operational State; when absent (synthetic/test
    # state) _optional_field_array returns None and the field is skipped, so a
    # reduced state never gets a fabricated hydrometeor.
    for wrf_name, state_names in (
        ("QSNOW", ("QSNOW", "qs", "qsnow")),
        ("QGRAUP", ("QGRAUP", "qg", "qgraup")),
        ("QNICE", ("QNICE", "Ni", "qnice")),
        ("QNRAIN", ("QNRAIN", "Nr", "qnrain")),
        ("QNSNOW", ("QNSNOW", "Ns", "qnsnow")),
        ("QNGRAUPEL", ("QNGRAUPEL", "Ng", "qngraupel")),
        ("QNCLOUD", ("QNCLOUD", "Nc", "qnc", "qnc_cloud")),
        ("QNCCN", ("QNCCN", "Nn", "qnn", "qccn")),
        # v0.17 ADR-032 graupel/hail substrate (optional; zero/absent in a
        # non-hail run, so a reduced/non-hail state never gets a fabricated
        # hail field).
        ("QHAIL", ("QHAIL", "qh", "qhail")),
        ("QNHAIL", ("QNHAIL", "Nh", "qnh", "qnhail")),
        ("QVGRAUPEL", ("QVGRAUPEL", "qvolg")),
        ("QVHAIL", ("QVHAIL", "qvolh")),
        # v0.16 aerosol-aware Thompson (mp=28) aerosol numbers (optional).
        ("QNWFA", ("QNWFA", "nwfa", "qnwfa")),
        ("QNIFA", ("QNIFA", "nifa", "qnifa")),
        ("QIR", ("QIR", "qir")), ("QIB", ("QIB", "qib")),
        ("QICE2", ("QICE2", "qi2")), ("QNICE2", ("QNICE2", "Ni2")),
        ("QIR2", ("QIR2", "qir2")), ("QIB2", ("QIB2", "qib2")), ("QZI", ("QZI", "qzi")),
        ("PHYS_TOT", ("PHYS_TOT", "phys_tot")),
        ("CWM", ("CWM", "qt")),
        ("QRIMEF", ("QRIMEF", "qrimef")),
        ("F_ICE_PHY", ("F_ICE_PHY", "f_ice_phy")),
        ("F_RAIN_PHY", ("F_RAIN_PHY", "f_rain_phy")),
        ("F_RIMEF_PHY", ("F_RIMEF_PHY", "f_rimef_phy")),
        *((name.upper(), (name.upper(), name)) for name in ("re_cloud_gsfc", "re_rain_gsfc", "re_ice_gsfc",
                                                         "re_snow_gsfc", "re_graupel_gsfc", "re_hail_gsfc")),
        ("QKE", ("QKE", "qke")),
        ("TKE", ("TKE", "tke")),
    ):
        if wrf_name == "QNCLOUD" and int(_lookup(namelist, "mp_physics", 0)) == 22:
            continue  # Original NSSL Registry names its Nc field QNDROP.
        if wrf_name in _MP_UNREPRESENTED_FIELDS.get(int(_lookup(namelist,"mp_physics",0)),()):
            continue
        if wrf_name == "QKE" and _lookup(namelist, "bl_pbl_physics", 5) in (16, 17):
            continue
        value = _optional_field_array(state, state_names, shape_xyz)
        if value is not None:
            fields[wrf_name] = value

    # --- P0-5a (c) accumulated precipitation partition. Source = the State
    # precip accumulators that coupling.physics_couplers advances each
    # microphysics step as DISJOINT channels (rain_acc=liquid, snow_acc, ice_acc,
    # graupel_acc). WRF's wrfout convention (module_mp_thompson.F:1298-1306) is:
    #   RAINNC    = rain + snow + graupel + ice   (TOTAL accumulated precip, all phases)
    #   SNOWNC    = snow + ice                     (frozen subset)
    #   GRAUPELNC = graupel
    # i.e. SNOWNC/GRAUPELNC are OVERLAPPING subsets of RAINNC, not extra channels.
    # The internal accumulators stay disjoint (conservation budget / SR / restart
    # rely on that); only the wrfout mapping folds them into WRF's total here.
    # Emitted only when present (microphysics active). ---
    rain_acc = _optional_field_array(state, ("rain_acc", "RAINNC"), shape_xy)
    snow_acc = _optional_field_array(state, ("snow_acc", "SNOWNC"), shape_xy)
    ice_acc = _optional_field_array(state, ("ice_acc",), shape_xy)
    graupel_acc = _optional_field_array(state, ("graupel_acc", "GRAUPELNC"), shape_xy)
    # v0.17 hail (WSM7/WDM7): HAILNC is a distinct accumulated channel folded into
    # the RAINNC all-phase total, mirroring graupel. Zero/absent in a non-hail run.
    hail_acc = _optional_field_array(state, ("hail_acc", "HAILNC"), shape_xy)
    if snow_acc is not None or ice_acc is not None:
        snowice = (snow_acc if snow_acc is not None else 0.0) + (
            ice_acc if ice_acc is not None else 0.0
        )
        fields["SNOWNC"] = _coerce_array("SNOWNC", snowice, shape_xy)
    if graupel_acc is not None:
        fields["GRAUPELNC"] = graupel_acc
    if hail_acc is not None:
        fields["HAILNC"] = hail_acc
    # WRF RAINNC is the all-phase total; overwrite the rain-only default written
    # above (RAINNC: rain_acc) when any frozen channel is present so SNOWNC never
    # exceeds RAINNC and the domain precip total matches CPU-WRF.
    if any(a is not None for a in (snow_acc, graupel_acc, ice_acc, hail_acc)):
        def _zero_xy(a: np.ndarray | None) -> np.ndarray:
            return np.asarray(a, dtype=np.float64) if a is not None else np.zeros(shape_xy, dtype=np.float64)

        rainnc_total = (
            _zero_xy(rain_acc)
            + _zero_xy(snow_acc)
            + _zero_xy(graupel_acc)
            + _zero_xy(ice_acc)
            + _zero_xy(hail_acc)
        )
        fields["RAINNC"] = _coerce_array("RAINNC", rainnc_total, shape_xy)

    for name, leaf in (("RAINNC", "thompson_rainnc"), ("SNOWNC", "thompson_snownc"), ("RAINNC", "ntu_rainnc"), ("SNOWNC", "ntu_snownc"), ("RAINNC", "cam_rainnc"), ("SNOWNC", "cam_snownc")):
        value = _optional_field_array(state, (leaf,), shape_xy)
        if value is not None:
            fields[name] = value

    # --- P0-5a (b) grid-static coordinate / map-factor / Coriolis fields. Source
    # = GridSpec.metrics + GridSpec.vertical (always present on a real GridSpec).
    # Skipped wholesale when the grid carries no metrics (dict/synthetic grid). ---
    _add_grid_coordinate_fields(
        fields, grid, dimensions,
        shape_xy=shape_xy, shape_mapu=shape_mapu, shape_mapv=shape_mapv,
        landmask=landmask,
    )

    # --- P0-5a (d) 2-m potential temperature TH2 = T2 * (P0/PSFC)^(Rd/cp). Always
    # derivable from the (possibly diagnostic-overridden) T2 + PSFC below; computed
    # after the diagnostics override so it tracks the operational 2-m fields. ---

    # --- P0-5a (e) prognostic Noah-MP soil/snow land columns + land diagnostics.
    # Source = the optional NoahMPLandState carry. Never fabricated. ---
    if land_state is not None:
        _add_land_soil_fields(
            fields,
            land_state,
            shape_soil=shape_soil,
            shape_snow=shape_snow,
            shape_snso=shape_snso,
            shape_xy=shape_xy,
        )

    # The set of surface-map names the diagnostics dict is allowed to write. Most
    # OVERRIDE a raw lowest-level fallback already in ``fields``; QFX/GRDFLX are
    # ADD-only operational fluxes (the writer has no physical fallback for them, so
    # they appear only when the operational coupler supplies them -- never
    # fabricated). All are mass-point (ny, nx).
    _DIAGNOSTIC_SURFACE_FIELDS = {
        "U10E", "V10E",
        *ONLINE_SURFACE_FIELDS,
        *SOLAR_SURFACE_FIELDS,
        *GODDARD_SURFACE_FIELDS,
        "PREC_ACC_C", "PREC_ACC_NC", "SNOW_ACC_NC",
        "T2", "U10", "V10", "Q2", "PSFC", "SWDOWN", "GLW", "PBLH", "UST",
        "HFX", "LH", "TSK", "QFX", "GRDFLX",
        # --- B1 (v0.12.0) RRTMG up/down all-sky surface + TOA flux diagnostics ---
        # ADD-only (no physical fallback): appear only when the operational
        # radiation diagnostics supply them, never fabricated. SWDNB == SWDOWN in
        # the no-slope config; SWNORM is the slope-normal surface SW flux. OLR is
        # derived below from LWUPT (== WRF's TOA outgoing LW).
        "SWDNB", "SWUPB", "LWDNB", "LWUPB",
        "SWDNT", "SWUPT", "LWDNT", "LWUPT", "SWNORM",
        # --- v0.13.0 RRTMG clear-sky (cloud-free) flux diagnostics. ADD-only:
        # appear only when the radiation diagnostics supply the WRF clear-sky pass
        # outputs (``with_clear_sky=True``). ---
        "SWDNBC", "SWUPBC", "LWDNBC", "LWUPBC",
        "SWDNTC", "SWUPTC", "LWDNTC", "LWUPTC",
    }
    if diagnostics is not None:
        # WRF stochastic-perturbation restart seed arrays are 1-D integer state.
        # They are not active in the supported Canary suite, but if a caller carries
        # real seed state, write it faithfully instead of leaving the KI-3 dimension
        # unsupported.
        for name in STOCHASTIC_SEED_VARIABLES:
            value = diagnostics.get(name) if isinstance(diagnostics, Mapping) else None
            if value is not None:
                fields[name] = _coerce_array(name, value, shape_seed, dtype=np.int32)

        # Operational surface-layer diagnostics OVERRIDE the raw lowest-level
        # fallbacks for the surface map. These are physically diagnosed 2-m / 10-m /
        # skin / surface fields (mass-point ``(ny, nx)``), not raw level-1 values.
        for name, value in diagnostics.items():
            if value is None or name not in _DIAGNOSTIC_SURFACE_FIELDS:
                continue
            if name in ("PREC_ACC_C", "PREC_ACC_NC", "SNOW_ACC_NC") and not np.isfinite(_host_materialized_value(value)).all():
                raise ValueError(f"nonfinite precipitation history field {name}")
            fields[name] = _coerce_array(name, value, shape_xy)

        # --- B1 (v0.12.0) OLR = WRF's TOA outgoing longwave == LWUPT (the
        # top-of-atmosphere upwelling LW flux). WRF carries OLR as a separate
        # history var but its value is identically the upward LW flux at the model
        # top; derived here from the diagnostic LWUPT so it is always consistent
        # and never an independent fabricated quantity. Emitted only when LWUPT is. ---
        if "LWUPT" in fields:
            fields["OLR"] = _coerce_array("OLR", np.asarray(fields["LWUPT"]), shape_xy)

    # --- P0-5a (d) TH2: 2-m potential temperature = T2 * (P0/PSFC)^(Rd/cp). Built
    # from the final (diagnostic-overridden when present) T2 + PSFC so it is the
    # operational 2-m theta, consistent with WRF's TH2 = T2/pi2 diagnostic. ---
    if "T2" in fields and "PSFC" in fields:
        psfc = np.maximum(np.asarray(fields["PSFC"], dtype=np.float64), 1.0)
        t2 = np.asarray(fields["T2"], dtype=np.float64)
        fields["TH2"] = _coerce_array("TH2", t2 * (P0_PA / psfc) ** R_D_OVER_CP, shape_xy)

    # --- v0.12.0 A2 trivially-derived diagnostics (no physics, no GPU). ---
    _add_derived_diagnostic_fields(
        fields,
        state=state,
        land_state=land_state,
        xlat=xlat,
        xlong=xlong,
        shape_xy=shape_xy,
        run_start=run_start,
        lead_hours=lead_hours,
    )

    return {name: _materialized_dtype(name, value) for name, value in fields.items()}


def _add_microphysics_history_fields(
    fields: dict[str, np.ndarray], *, state: Any, namelist: Any,
    dimensions: Mapping[str, int | None], diagnostics: Mapping[str, Any] | None,
    requested_names: frozenset[str] | None,
) -> None:
    """Transcribe existing species/held diagnostics; never re-solve physics."""
    if int(_lookup(namelist,"mp_physics",0))==11:
        shape=_shape_for_dimensions(WRFOUT_VARIABLE_SPECS["RAINNC"].dimensions,dimensions)
        for name,leaf in (("RAINNC","cam_rainnc"),("SNOWNC","cam_snownc"),("SR","cam_sr")):
            if requested_names is None or name in requested_names:
                value=_optional_field_array(state,(leaf,),shape)
                if value is not None:
                    if not np.isfinite(value).all(): raise ValueError(f"nonfinite CAM-MG history {name}")
                    fields[name]=value
    for name in _MP_HISTORY_FIELDS.get(int(_lookup(namelist, "mp_physics", 0)), ()):
        if requested_names is not None and name not in requested_names:
            continue
        spec = WRFOUT_VARIABLE_SPECS[name]
        shape = _shape_for_dimensions(spec.dimensions, dimensions)
        value = _optional_field_array(diagnostics, (name,), shape)
        if value is None:
            value = _optional_field_array(state, _MP_STATE_ALIASES.get(name, (name, name.lower())), shape)
        if value is None and int(_lookup(namelist, "mp_physics", 0)) == 14:
            raise ValueError(f"WDM5 history requires computed {name}; refusing invented output")
        if value is not None:
            if not np.isfinite(value).all():
                raise ValueError(f"nonfinite microphysics history field {name}")
            fields[name] = value


def _add_full_wrfout_fields(
    fields: dict[str, np.ndarray],
    *,
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
    diagnostics: Mapping[str, Any] | None,
    land_state: Any | None,
    run_start: datetime,
    lead_hours: float,
    requested_names: frozenset[str] | None = None,
) -> None:
    """Expand ``fields`` to the opt-in 375-variable WRF history stream.

    This is an output-compatibility layer, not a physics path. It first preserves
    all fields already mapped by the operational writer, then accepts explicit
    same-name payloads from diagnostics/state/land/grid/namelist, derives simple
    WRF algebraic statics, and finally fills unsupported inactive stream slots
    with the zero/default value WRF would carry for disabled diagnostics.
    """

    for name in FULL_WRFOUT_VARIABLES:
        if name in _MP_UNREPRESENTED_FIELDS.get(int(_lookup(namelist,"mp_physics",0)),()):
            continue
        if requested_names is not None and name not in requested_names:
            continue
        if name in {"Times", "XTIME"} or name in fields:
            continue
        spec = WRFOUT_VARIABLE_SPECS.get(name)
        if spec is None:
            raise KeyError(f"FULL_WRFOUT_VARIABLES contains {name} without a WRFOUT_VARIABLE_SPECS entry")
        shape = _shape_for_dimensions(spec.dimensions, dimensions)
        dtype = _numpy_dtype_for_spec(spec)
        value = _full_source_value(
            name,
            shape,
            dtype=dtype,
            diagnostics=diagnostics,
            state=state,
            land_state=land_state,
            grid=grid,
            namelist=namelist,
        )
        if value is None:
            value = _full_derived_value(
                name,
                fields,
                grid=grid,
                namelist=namelist,
                shape=shape,
                dtype=dtype,
                run_start=run_start,
                lead_hours=lead_hours,
            )
        if value is None:
            value = _full_default_value(name, shape, dtype=dtype)
        fields[name] = _coerce_array(name, value, shape, dtype=dtype)


def _full_source_value(
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: Any,
    diagnostics: Mapping[str, Any] | None,
    state: Any,
    land_state: Any | None,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
) -> np.ndarray | None:
    # WRF's disabled SST-skin work arrays remain zero. SST and land TSK are distinct.
    if name in {"SSTSK", "SST_INPUT"} and not bool(_lookup(namelist, "sst_skin", False)):
        return np.zeros(shape, dtype=dtype)
    aliases = _full_source_aliases(name)
    for source in (diagnostics, state, land_state, grid, namelist):
        if source is None:
            continue
        for alias in aliases:
            value = _lookup(source, alias, None)
            if value is None:
                continue
            # MYNN copies EL only over kts:kte. Registry stores nz+1 levels;
            # its unused top stays zero (CPU-WRF WN3 0227 and PROD D5).
            # This is padding, not interpolation: retain every computed level.
            if name == "EL_PBL" and np.asarray(value).shape == (shape[0] - 1, *shape[1:]):
                padded = np.zeros(shape, dtype=dtype)
                padded[:-1] = value
                value = padded
            return _coerce_array(name, value, shape, dtype=dtype)
    return None


def _full_source_aliases(name: str) -> tuple[str, ...]:
    lower = name.lower()
    aliases = [name, lower]
    # Common WRF Registry output-name aliases that differ from gpuwrf's carry names.
    extras = {
        "SEAICE": ("xice", "seaice"),
        "SST": ("sst", "t_sst"),
        "SST_INPUT": ("sst_input", "sst"),
        "SSTSK": ("sstsk", "t_skin"),
        "TMN": ("tmn", "deep_soil_temperature"),
        "LAKEMASK": ("lakemask",),
        "VAR_SSO": ("var_sso",),
        "NEST_POS": ("nest_pos",),
        "IVGTYP": ("ivgtyp", "lu_index"),
        "ISLTYP": ("isltyp", "soil_type"),
        "ITIMESTEP": ("itimestep", "step_index"),
        "THIS_IS_AN_IDEAL_RUN": ("this_is_an_ideal_run",),
        "GOT_VAR_SSO": ("got_var_sso",),
        "SAVE_TOPO_FROM_REAL": ("save_topo_from_real",),
        "ISEEDARR_RAND_PERTURB": ("iseedarr_rand_perturb",),
        "TKE_PBL": ("tke_pbl",),
        "EL_PBL": ("el_pbl",),
        "P_HYD": ("p_hyd",),
    }.get(name, ())
    aliases.extend(extras)
    return tuple(dict.fromkeys(aliases))


def _full_derived_value(
    name: str,
    fields: Mapping[str, np.ndarray],
    *,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    shape: tuple[int, ...],
    dtype: Any,
    run_start: datetime,
    lead_hours: float,
) -> np.ndarray | float | int | None:
    if name == "P_HYD" and "P" in fields and "PB" in fields:
        return np.asarray(fields["P"], dtype=np.float64) + np.asarray(fields["PB"], dtype=np.float64)
    if name == "AREA2D":
        return _full_area2d(fields, grid, shape)
    if name == "DX2D":
        # Active compute_2d_dx_area branch (physics_init.F:5683-5686).
        projection = _lookup(grid, "projection")
        return _lookup(projection, "dx_m", _lookup(grid, "dx", None))
    if name == "MF_VX_INV" and "MAPFAC_VX" in fields:
        return 1.0 / np.maximum(np.asarray(fields["MAPFAC_VX"], dtype=np.float64), 1.0e-12)
    if name in {"MAX_MSFTX", "MAX_MSFTY"}:
        # start_em only assigns these scalars for adaptive time stepping.
        key = "MAPFAC_MX" if name == "MAX_MSFTX" else "MAPFAC_MY"
        if bool(_lookup(namelist, "use_adaptive_time_step", False)) and key in fields:
            return float(np.nanmax(np.asarray(fields[key], dtype=np.float64)))
        return 0.0
    if name == "IVGTYP" and "LU_INDEX" in fields:
        return np.rint(np.asarray(fields["LU_INDEX"], dtype=np.float64)).astype(dtype)
    if name == "LAKEMASK":
        return np.zeros(shape, dtype=dtype)
    if name in {"SST", "SST_INPUT", "SSTSK", "TMN", "TG", "TGV", "TGB"} and "TSK" in fields:
        return np.asarray(fields["TSK"], dtype=np.float64)
    if name in {"ALBBCK", "ALBOLD"} and "ALBEDO" in fields:
        return np.asarray(fields["ALBEDO"], dtype=np.float64)
    if name == "TV" and "T2" in fields and "Q2" in fields:
        return np.asarray(fields["T2"], dtype=np.float64) * (
            1.0 + 0.608 * np.maximum(np.asarray(fields["Q2"], dtype=np.float64), 0.0)
        )
    if name in {"T2V", "T2B"} and "T2" in fields:
        return np.asarray(fields["T2"], dtype=np.float64)
    if name in {"Q2V", "Q2B"} and "Q2" in fields:
        return np.asarray(fields["Q2"], dtype=np.float64)
    if name == "ITIMESTEP":
        dt_s = _lookup(namelist, "dt_s", _lookup(namelist, "time_step", None))
        if dt_s not in (None, 0):
            return int(round(float(lead_hours) * 3600.0 / float(dt_s)))
        return int(round(float(lead_hours) * 60.0))
    if name == "THIS_IS_AN_IDEAL_RUN":
        return int(bool(_lookup(namelist, "this_is_an_ideal_run", False)))
    if name == "GOT_VAR_SSO":
        return int(bool(_lookup(namelist, "got_var_sso", False)))
    if name == "T00":
        return float(_lookup(namelist, "t00", 290.0))
    if name == "P00":
        return float(_lookup(namelist, "p00", P0_PA))
    if name == "TLP":
        return float(_lookup(namelist, "tlp", 50.0))
    if name == "TISO":
        return float(_lookup(namelist, "tiso", 200.0))
    if name == "P_STRAT":
        return float(_lookup(namelist, "p_strat", 0.0))
    if name == "TLP_STRAT":
        return float(_lookup(namelist, "tlp_strat", -11.0))
    if name == "ZETATOP":
        return float(_lookup(namelist, "zetatop", 0.0))
    if name == "RESM":
        return float(_lookup(namelist, "resm", 0.0))
    if name in {"HFX_FORCE", "LH_FORCE", "TSK_FORCE", "HFX_FORCE_TEND", "LH_FORCE_TEND", "TSK_FORCE_TEND"}:
        return 0.0
    if name == "SAVE_TOPO_FROM_REAL":
        return int(bool(_lookup(namelist, "save_topo_from_real", False)))
    if name == "BATHYMETRY_FLAG":
        return int(bool(_lookup(namelist, "bathymetry_flag", False)))
    del run_start  # currently only lead_hours contributes to derived full fields.
    return None


def _full_area2d(fields: Mapping[str, np.ndarray], grid: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    projection = _lookup(grid, "projection")
    dx_m = _lookup(projection, "dx_m", _lookup(grid, "dx", None))
    dy_m = _lookup(projection, "dy_m", _lookup(grid, "dy", dx_m))
    if dx_m is None or dy_m is None:
        return None
    area = float(dx_m) * float(dy_m)
    # WRF compiles out the map-factor branch (#if 0, physics_init.F:5675).
    return np.full(shape, area, dtype=np.float64)


def _full_default_value(name: str, shape: tuple[int, ...], *, dtype: Any) -> np.ndarray:
    del name
    return np.zeros(shape, dtype=dtype)


def _add_derived_diagnostic_fields(
    fields: dict[str, np.ndarray],
    *,
    state: Any,
    land_state: Any | None,
    xlat: np.ndarray,
    xlong: np.ndarray,
    shape_xy: tuple[int, int],
    run_start: datetime | None,
    lead_hours: float,
) -> None:
    """Populate the cheap WRF-faithful derived diagnostics (v0.12.0 A2).

    Each field self-gates on the availability of its inputs and uses a closed
    WRF-faithful form -- never a fabricated quantity:

    - ``CLAT`` uses ``XLAT`` as its compatibility fallback. The public payload
      builder preserves a supplied CLAT producer, including initial feedback.
    - ``COSZEN`` is WRF's ``radconst``/``calc_coszen`` cosine solar zenith angle
      (module_radiation_driver.F:3594-3666) evaluated from XLAT/XLONG and the
      forecast clock (``run_start`` + ``lead_hours``). Routed through the exact
      transcription in ``coupling.physics_couplers._compute_coszen``. Skipped
      (not fabricated) if the clock is absent.
    - ``SR`` is the WRF surface frozen-precipitation fraction
      = solid_acc / (solid_acc + liquid_acc), built from the State precip
      accumulators (snow+graupel+ice vs rain). Zero where no precip has fallen;
      skipped entirely when the State carries no accumulators (dry/synthetic).
    - ``SNOWC`` is WRF's snow-cover flag (1 where bulk SWE > 0) from the land
      carry's ``sneqv``; skipped when there is no land carry (never fabricated).
    """

    # Compatibility fallback; prepare_wrfout_payload retains supplied CLAT.
    fields["CLAT"] = _coerce_array("CLAT", np.asarray(xlat), shape_xy)

    # WRF lead-zero history precedes the first radiation call.
    if float(lead_hours) == 0.0:
        fields["COSZEN"] = np.zeros(shape_xy, dtype=np.float32)
    elif run_start is not None:
        try:
            from gpuwrf.coupling.physics_couplers import _compute_coszen

            coszen = _compute_coszen(
                np.asarray(xlat, dtype=np.float64),
                np.asarray(xlong, dtype=np.float64),
                run_start,
                lead_seconds=float(lead_hours) * 3600.0,
            )
            fields["COSZEN"] = _coerce_array("COSZEN", np.asarray(coszen), shape_xy)
        except Exception:  # noqa: BLE001 -- best-effort diagnostic; never blocks output.
            pass

    # SR = frozen-precip fraction from the State accumulators (mm). solid = snow +
    # graupel + ice; total = solid + rain. WRF SR is in [0,1]; zero where dry.
    rain_acc = _optional_field_array(state, ("rain_acc", "RAINNC"), shape_xy)
    snow_acc = _optional_field_array(state, ("snow_acc", "SNOWNC"), shape_xy)
    graupel_acc = _optional_field_array(state, ("graupel_acc", "GRAUPELNC"), shape_xy)
    ice_acc = _optional_field_array(state, ("ice_acc",), shape_xy)
    for name, leaf in (("QNWFA2D", "nwfa2d"), ("QNIFA2D", "nifa2d")):
        value = _optional_field_array(state, (name, leaf), shape_xy)
        if value is not None:
            fields[name] = value
    held_ferrier_sr = _optional_field_array(state, ("ferrier_sr", "goddard_sr", "thompson_sr", "ntu_sr", "cam_sr"), shape_xy)
    if held_ferrier_sr is not None:
        fields["SR"] = held_ferrier_sr
    elif any(a is not None for a in (rain_acc, snow_acc, graupel_acc, ice_acc)):
        def _z(a: np.ndarray | None) -> np.ndarray:
            return np.asarray(a, dtype=np.float64) if a is not None else np.zeros(shape_xy, dtype=np.float64)

        solid = _z(snow_acc) + _z(graupel_acc) + _z(ice_acc)
        total = solid + _z(rain_acc)
        sr = np.where(total > 0.0, solid / np.maximum(total, 1.0e-12), 0.0)
        fields["SR"] = _coerce_array("SR", sr, shape_xy)

    # SNOWC = snow-cover flag (1 where bulk SWE > 0) from the land carry SWE.
    sneqv = _lookup(land_state, "sneqv") if land_state is not None else None
    if sneqv is None and land_state is not None:
        sneqv = _lookup(land_state, "snow")
    if sneqv is not None:
        swe = np.asarray(sneqv, dtype=np.float64)
        fields["SNOWC"] = _coerce_array("SNOWC", np.where(swe > 0.0, 1.0, 0.0), shape_xy)


def _add_grid_coordinate_fields(
    fields: dict[str, np.ndarray],
    grid: Any,
    dimensions: Mapping[str, int | None],
    *,
    shape_xy: tuple[int, int],
    shape_mapu: tuple[int, int],
    shape_mapv: tuple[int, int],
    landmask: np.ndarray,
) -> None:
    """Populate the WRF grid-static coordinate / map-factor / Coriolis fields.

    Source = the real ``GridSpec`` (``eta_levels`` + ``DycoreMetrics``). Each
    field is added ONLY when its source array is actually present on the grid, so
    a dict-driven / synthetic grid with no metrics emits none of them (rather than
    fabricating unit map factors). All map/Coriolis arrays are
    ``DycoreMetrics`` fp64 device arrays; ``_coerce_array`` pulls them to host
    np.float32 at this output boundary.
    """

    nz = int(dimensions["bottom_top"])
    # ZNU/ZNW: eta on half (mass) / full (w) levels. WRF znw = full eta levels;
    # znu = mass-level midpoints. eta_levels is the authoritative (nz+1,) column.
    eta = _lookup(grid, "eta_levels")
    if eta is not None:
        eta_host = np.asarray(eta, dtype=np.float64)
        if eta_host.shape == (nz + 1,):
            fields["ZNW"] = _coerce_array("ZNW", eta_host, (nz + 1,))
            fields["ZNU"] = _coerce_array("ZNU", 0.5 * (eta_host[:-1] + eta_host[1:]), (nz,))

    # ZS/DZS: static Noah/Noah-MP soil-layer geometry. Pure constants of the soil
    # configuration (WRF init_soil_depth_2, module_soil_pre.F:1128-1151): DZS =
    # layer thicknesses, ZS = layer-center depths derived from DZS. WRF carries
    # them in every history frame so downstream tools can reconstruct the soil
    # column. They depend ONLY on the soil_layers_stag count, so they are always
    # emittable (no metrics/state source needed) -- the SAME arrays the real-init
    # + Noah-MP land hook consume. Computed in real(4) to bit-match the Fortran
    # init_soil_depth_2, whose fp32 cumulative sum rounds ZS(3)=0.70000005
    # (0x3f333334) -- an fp64 accumulation rounds the other way (0x3f333333).
    n_soil = int(dimensions["soil_layers_stag"])
    if n_soil == 4:
        f32 = np.float32
        dzs_host = np.array([0.1, 0.3, 0.6, 1.0], dtype=np.float32)
        zs_host = np.zeros(n_soil, dtype=np.float32)
        zs_host[0] = f32(0.5) * dzs_host[0]
        for _l in range(1, n_soil):
            zs_host[_l] = zs_host[_l - 1] + f32(0.5) * dzs_host[_l - 1] + f32(0.5) * dzs_host[_l]
        fields["ZS"] = _coerce_array("ZS", zs_host, (n_soil,))
        fields["DZS"] = _coerce_array("DZS", dzs_host, (n_soil,))

    metrics = _lookup(grid, "metrics")
    if metrics is not None:
        # Map-scale factors. WRF emits both the primary MAPFAC_M/U/V (the x-direction
        # factors the dycore/PGF consume) and the directional MAPFAC_{M,U,V}{X,Y}.
        # All map onto DycoreMetrics.msf{t,u,v}{x,y}; the primary M/U/V alias the
        # x-direction factors (msftx/msfux/msfvx), matching WRF for the conformal
        # Canary projection.
        for wrf_name, attr, shape in (
            ("MAPFAC_M", "msftx", shape_xy),
            ("MAPFAC_U", "msfux", shape_mapu),
            ("MAPFAC_V", "msfvx", shape_mapv),
            ("MAPFAC_MX", "msftx", shape_xy),
            ("MAPFAC_MY", "msfty", shape_xy),
            ("MAPFAC_UX", "msfux", shape_mapu),
            ("MAPFAC_UY", "msfuy", shape_mapu),
            ("MAPFAC_VX", "msfvx", shape_mapv),
            ("MAPFAC_VY", "msfvy", shape_mapv),
            ("F", "f", shape_xy),
            ("E", "e", shape_xy),
            ("SINALPHA", "sina", shape_xy),
            ("COSALPHA", "cosa", shape_xy),
        ):
            value = _lookup(metrics, attr)
            if value is not None:
                fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), shape)

        # v0.12.0 A1: vertical eta-spacing + hybrid-eta C-coefficient column metrics.
        # PURE PAYLOAD of arrays already resident on DycoreMetrics -- no recompute.
        for wrf_name, attr in (
            ("DN", "dn"),
            ("DNW", "dnw"),
            ("RDN", "rdn"),
            ("RDNW", "rdnw"),
            ("FNM", "fnm"),
            ("FNP", "fnp"),
            ("C1H", "c1h"),
            ("C2H", "c2h"),
            ("C3H", "c3h"),
            ("C4H", "c4h"),
        ):
            value = _lookup(metrics, attr)
            if value is not None and np.asarray(value).shape == (nz,):
                fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), (nz,))
        for wrf_name, attr in (
            ("C1F", "c1f"),
            ("C2F", "c2f"),
            ("C3F", "c3f"),
            ("C4F", "c4f"),
        ):
            value = _lookup(metrics, attr)
            if value is not None and np.asarray(value).shape == (nz + 1,):
                fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), (nz + 1,))

        # Scalar extrapolation constants. cf1/cf2/cf3 are stored directly; the WRF
        # top-of-model constants cfn/cfn1 are the standard dnw/dn ratios at the top
        # mass level (module_initialize_real.F:3757-3758, 0-indexed nz-1):
        #   cfn  = (0.5*dnw[nz-1] + dn[nz-1]) / dn[nz-1]
        #   cfn1 = -0.5*dnw[nz-1] / dn[nz-1]
        for wrf_name, attr in (("CF1", "cf1"), ("CF2", "cf2"), ("CF3", "cf3")):
            value = _lookup(metrics, attr)
            if value is not None:
                fields[wrf_name] = _coerce_array(wrf_name, float(np.asarray(value).reshape(-1)[0]), ())
        dn = _lookup(metrics, "dn")
        dnw = _lookup(metrics, "dnw")
        if dn is not None and dnw is not None:
            dn_top = float(np.asarray(dn).reshape(-1)[-1])
            dnw_top = float(np.asarray(dnw).reshape(-1)[-1])
            if dn_top != 0.0:
                fields["CFN"] = _coerce_array("CFN", (0.5 * dnw_top + dn_top) / dn_top, ())
                fields["CFN1"] = _coerce_array("CFN1", -0.5 * dnw_top / dn_top, ())

        p_top = _lookup(metrics, "p_top")
        if p_top is not None:
            fields["P_TOP"] = _coerce_array("P_TOP", float(np.asarray(p_top).reshape(-1)[0]), ())

    # Inverse grid lengths RDX = 1/dx, RDY = 1/dy from the projection (WRF scalars).
    projection = _lookup(grid, "projection")
    dx_m = _lookup(projection, "dx_m", _lookup(grid, "dx", None))
    dy_m = _lookup(projection, "dy_m", _lookup(grid, "dy", dx_m))
    if dx_m is not None and float(dx_m) != 0.0:
        fields["RDX"] = _coerce_array("RDX", 1.0 / float(dx_m), ())
    if dy_m is not None and float(dy_m) != 0.0:
        fields["RDY"] = _coerce_array("RDY", 1.0 / float(dy_m), ())

    # XLAND (1 land / 2 water) = WRF land/water flag. Derive from the landmask the
    # writer already resolved (1 land / 0 water) so it is always consistent.
    fields["XLAND"] = _coerce_array("XLAND", np.where(landmask > 0.5, 1.0, 2.0), shape_xy)


def _add_land_soil_fields(
    fields: dict[str, np.ndarray],
    land_state: Any,
    *,
    shape_soil: tuple[int, int, int],
    shape_snow: tuple[int, int, int],
    shape_snso: tuple[int, int, int],
    shape_xy: tuple[int, int],
) -> None:
    """Populate the prognostic Noah-MP soil/snow land + diagnostic fields.

    Source = the ``NoahMPLandState`` carry (4-layer ``tslb``/``smois``/``sh2o``,
    bulk ``sneqv``/``snowh``, intercepted canopy water, accumulated runoff, broadband
    ``albedo``/``emiss``). Each field is added ONLY when the carry exposes its
    source attribute, so a partial land carry never fabricates a soil profile.
    WRF mapping (module_sf_noahmpdrv.F): SNOW=SNEQV, SNOWH=SNOWH,
    CANWAT=CANLIQ+CANICE, SFROFF=SFCRUNOFF*1e3, UDROFF=UDRUNOFF*1e3 (m->mm).
    """

    for wrf_name, attr in (("TSLB", "tslb"), ("SMOIS", "smois"), ("SH2O", "sh2o")):
        value = _lookup(land_state, attr)
        if value is not None:
            fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), shape_soil)

    for wrf_name, attr, shape in (
        ("TSNO", "tsno", shape_snow),
        ("SNICE", "snice", shape_snow),
        ("SNLIQ", "snliq", shape_snow),
        ("ZSNSO", "zsnso", shape_snso),
    ):
        value = _lookup(land_state, attr)
        if value is not None:
            fields[wrf_name] = _coerce_array(wrf_name, np.asarray(value), shape)

    # Scalar (ny, nx) snow/canopy prognostics. ISNOW is the int32 active-layer
    # count (-2..0); SNEQVO is the prior-step SWE; CANLIQ/CANICE are the canopy
    # interception reservoirs (their sum is the CANWAT bulk written below).
    isnow = _lookup(land_state, "isnow")
    if isnow is not None:
        fields["ISNOW"] = _coerce_array("ISNOW", np.asarray(isnow), shape_xy, dtype=np.int32)
    sneqvo = _lookup(land_state, "sneqvo")
    if sneqvo is not None:
        fields["SNEQVO"] = _coerce_array("SNEQVO", np.asarray(sneqvo), shape_xy)

    sneqv = _lookup(land_state, "sneqv")
    if sneqv is not None:
        fields["SNOW"] = _coerce_array("SNOW", np.asarray(sneqv), shape_xy)
    snowh = _lookup(land_state, "snowh")
    if snowh is not None:
        fields["SNOWH"] = _coerce_array("SNOWH", np.asarray(snowh), shape_xy)
    canliq = _lookup(land_state, "canliq")
    canice = _lookup(land_state, "canice")
    if canliq is not None:
        fields["CANLIQ"] = _coerce_array("CANLIQ", np.asarray(canliq), shape_xy)
    if canice is not None:
        fields["CANICE"] = _coerce_array("CANICE", np.asarray(canice), shape_xy)
    if canliq is not None or canice is not None:
        canwat = (np.asarray(canliq) if canliq is not None else 0.0) + (
            np.asarray(canice) if canice is not None else 0.0
        )
        fields["CANWAT"] = _coerce_array("CANWAT", canwat, shape_xy)
    # WRF carries SFROFF/UDROFF in mm; the Noah-MP carry accumulates runoff in m.
    sfcrunoff = _lookup(land_state, "sfcrunoff")
    if sfcrunoff is not None:
        fields["SFROFF"] = _coerce_array("SFROFF", np.asarray(sfcrunoff) * 1.0e3, shape_xy)
    udrunoff = _lookup(land_state, "udrunoff")
    if udrunoff is not None:
        fields["UDROFF"] = _coerce_array("UDROFF", np.asarray(udrunoff) * 1.0e3, shape_xy)
    albedo = _lookup(land_state, "albedo")
    if albedo is not None:
        fields["ALBEDO"] = _coerce_array("ALBEDO", np.asarray(albedo), shape_xy)
    emiss = _lookup(land_state, "emiss")
    if emiss is not None:
        fields["EMISS"] = _coerce_array("EMISS", np.asarray(emiss), shape_xy)


def _surface_flux_fallbacks(
    *,
    state: Any,
    grid: Any,
    theta: np.ndarray,
    qv: np.ndarray,
    p_total: np.ndarray,
    ph_total: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    landmask: np.ndarray,
    shape_xy: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Recompute sfclay-style HFX/LH when direct WRF flux diagnostics are absent."""

    u_mass = _unstagger_x(u)
    v_mass = _unstagger_y(v)
    dz = np.maximum((ph_total[1:, :, :] - ph_total[:-1, :, :]) / 9.80665, 1.0)
    t_air0 = theta[0] * (np.maximum(p_total[0], 1.0) / P0_PA) ** R_D_OVER_CP
    t_skin = _field_array(state, ("TSK", "tsk", "t_skin"), shape_xy, default=t_air0)
    xland = _field_array(state, ("xland", "XLAND"), shape_xy, default=np.where(landmask > 0.5, 1.0, 2.0))
    soil_moisture = _optional_field_array(state, ("soil_moisture", "SMOIS"), shape_xy)
    mavail = _optional_field_array(state, ("mavail", "MAVAIL"), shape_xy)
    if mavail is None:
        mavail = soil_moisture if soil_moisture is not None else np.ones(shape_xy, dtype=np.float32)
    column_state = SimpleNamespace(
        u=np.moveaxis(u_mass, 0, -1),
        v=np.moveaxis(v_mass, 0, -1),
        theta=np.moveaxis(theta, 0, -1),
        qv=np.moveaxis(qv, 0, -1),
        p=np.moveaxis(p_total, 0, -1),
        dz=np.moveaxis(dz, 0, -1),
        t_skin=t_skin,
        soil_moisture=soil_moisture,
        xland=xland,
        lakemask=_field_array(state, ("lakemask", "LAKEMASK"), shape_xy),
        mavail=mavail,
        roughness_m=_optional_field_array(state, ("roughness_m", "ZNT"), shape_xy),
        ustar=_field_array(state, ("UST", "ustar"), shape_xy),
        dx_m=float(_lookup(_lookup(grid, "projection"), "dx_m", _lookup(grid, "dx", 3000.0))),
    )
    diag = surface_layer_with_diagnostics(column_state)
    # SurfaceLayerDiagnostics already carries the WRF-faithful sfclayrev surface
    # fluxes (hfx = flhc*(thgb-thx), lh = XLV*qfx; sf_sfclayrev.F90:856-878), both
    # W m^-2 positive-upward -- exactly the wrfout HFX/LH contract. Use them directly
    # instead of re-deriving HFX from an aerodynamic resistance: the old path read a
    # nonexistent diag.fh and raised AttributeError on the full operational/d03 path.
    hfx = np.asarray(diag.hfx, dtype=np.float64)
    lh = np.asarray(diag.lh, dtype=np.float64)
    return {"HFX": hfx.astype(np.float32), "LH": lh.astype(np.float32)}


def _perturbation_base_pair(
    state: Any,
    *,
    total_names: tuple[str, ...],
    perturbation_names: tuple[str, ...],
    base_names: tuple[str, ...],
    shape: tuple[int, ...],
    base_source: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    total = _optional_field_array(state, total_names, shape)
    perturbation = _optional_field_array(state, perturbation_names, shape)
    base = _optional_field_array(state, base_names, shape)
    if os.environ.get("GPUWRF_DYN_RK_FP32", "0") == "1" and base_source is not None:
        # Native RK retains the resolved initial WRF REAL base. Recovering it
        # from separately rounded totals and perturbations changes static fields.
        resolved_base = _optional_field_array(base_source, base_names, shape)
        if resolved_base is not None:
            base = resolved_base
    zeros = np.zeros(shape, dtype=np.float32)

    if total is None and perturbation is None and base is None:
        return zeros, zeros
    if total is None and perturbation is not None and base is not None:
        total = perturbation + base
    if perturbation is None and total is not None and base is not None:
        perturbation = total - base
    if base is None and total is not None and perturbation is not None:
        base = total - perturbation
    if perturbation is None and total is not None:
        perturbation = total
    if base is None:
        base = zeros
    return perturbation.astype(np.float32), base.astype(np.float32)


def _theta_dry_for_output(state: Any, theta: np.ndarray, qv: np.ndarray) -> np.ndarray:
    """Use WRF's per-operation REAL constant only for a real fp32 source carry.

    The ordinary writer coerces host arrays to fp32 before deriving fields, so
    inspect the original leaves to preserve the historical fp64-source path.
    WRF module_model_constants.F:41 folds rvovrd = r_v/r_d in REAL arithmetic.
    """
    if _WRITER_RVOVRD_REAL:
        sources = (
            next((v for name in names if (v := _lookup(state, name)) is not None), None)
            for names in (("theta", "THETA"), ("QVAPOR", "qv", "qvapor"))
        )
        if all(getattr(value, "dtype", None) == np.dtype(np.float32) for value in sources):
            return theta / (np.float32(1.0) + RVOVRD_REAL * np.maximum(qv, np.float32(0.0)))
    return theta / (1.0 + RVOVRD * np.maximum(qv, 0.0))


def _field_array(
    obj: Any,
    names: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    default: Any = 0.0,
) -> np.ndarray:
    value = _optional_field_array(obj, names, shape)
    if value is None:
        value = default
    return _coerce_array(names[0], value, shape)


def _optional_field_array(obj: Any, names: tuple[str, ...], shape: tuple[int, ...]) -> np.ndarray | None:
    for name in names:
        value = _lookup(obj, name, None)
        if value is not None:
            return _coerce_array(name, value, shape)
    return None


def _grid_or_state_array(
    state: Any,
    grid: Any,
    names: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    default: Any = 0.0,
) -> np.ndarray:
    value = _optional_field_array(state, names, shape)
    if value is None:
        value = _optional_field_array(grid, names, shape)
    if value is None:
        value = default
    return _coerce_array(names[0], value, shape)


def _landmask(state: Any, shape: tuple[int, ...]) -> np.ndarray:
    explicit = _optional_field_array(state, ("LANDMASK", "landmask"), shape)
    if explicit is not None:
        return explicit
    xland = _optional_field_array(state, ("xland", "XLAND"), shape)
    if xland is None:
        return np.ones(shape, dtype=np.float32)
    return np.where(xland <= 1.5, 1.0, 0.0).astype(np.float32)


def _latlon_fields(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    shape: tuple[int, int],
    *,
    suffix: str = "",
    diagnostics: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lat_names = ("XLAT" + suffix.upper(), "xlat" + suffix, "lat" + suffix)
    lon_names = ("XLONG" + suffix.upper(), "xlong" + suffix, "lon" + suffix)
    lat = _optional_field_array(diagnostics, lat_names, shape)
    lon = _optional_field_array(diagnostics, lon_names, shape)
    if lat is not None and lon is not None:
        return lat, lon
    lat = _optional_field_array(state, lat_names, shape)
    lon = _optional_field_array(state, lon_names, shape)
    if lat is not None and lon is not None:
        return lat, lon
    projection = _lookup(grid, "projection")
    lat_0 = float(_lookup(projection, "lat_0", _lookup(namelist, "cen_lat", 0.0)))
    lon_0 = float(_lookup(projection, "lon_0", _lookup(namelist, "cen_lon", 0.0)))
    dx_m = float(_lookup(projection, "dx_m", _lookup(namelist, "dx", 3000.0)))
    dy_m = float(_lookup(projection, "dy_m", _lookup(namelist, "dy", dx_m)))
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    x = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    lat_step = dy_m / 111_320.0
    lon_step = dx_m / max(111_320.0 * cos(lat_0 * pi / 180.0), 1.0)
    lat_grid = lat_0 + y[:, None] * lat_step
    lon_grid = lon_0 + x[None, :] * lon_step
    return (
        np.broadcast_to(lat_grid, shape).astype(np.float32),
        np.broadcast_to(lon_grid, shape).astype(np.float32),
    )


def _unstagger_x(u: np.ndarray) -> np.ndarray:
    return 0.5 * (u[..., :-1] + u[..., 1:])


def _unstagger_y(v: np.ndarray) -> np.ndarray:
    return 0.5 * (v[..., :-1, :] + v[..., 1:, :])


def _shape_for_dimensions(dimensions: tuple[str, ...], sizes: Mapping[str, int | None]) -> tuple[int, ...]:
    return tuple(int(sizes[name]) for name in dimensions if name != "Time")


def _materialized_dtype(name: str, value: Any) -> np.ndarray:
    spec = WRFOUT_VARIABLE_SPECS.get(name)
    if spec is not None and str(spec.dtype).startswith("i"):
        return np.asarray(value, dtype=np.int32)
    return np.asarray(value, dtype=np.float32)


def _numpy_dtype_for_spec(spec: WrfoutVariableSpec) -> Any:
    if str(spec.dtype).startswith("i"):
        return np.int32
    if spec.dtype == "f8":
        return np.float64
    return np.float32


def _coerce_array(name: str, value: Any, shape: tuple[int, ...], *, dtype: Any = np.float32) -> np.ndarray:
    array = np.asarray(_host_materialized_value(value), dtype=dtype)
    if array.shape == (1, *shape):
        array = array[0]
    if array.shape == shape:
        return array.astype(dtype, copy=False)
    if array.shape == ():
        return np.full(shape, array.item(), dtype=dtype)
    try:
        return np.broadcast_to(array, shape).astype(dtype)
    except ValueError as exc:
        raise ValueError(f"{name} shape {array.shape} cannot be written to WRF shape {shape}") from exc


def _grid_extent(grid: Any) -> tuple[int, int, int]:
    nx = _lookup(grid, "nx", None)
    ny = _lookup(grid, "ny", None)
    nz = _lookup(grid, "nz", None)
    projection = _lookup(grid, "projection")
    vertical = _lookup(grid, "vertical")
    if nx is None:
        nx = _lookup(projection, "nx", None)
    if ny is None:
        ny = _lookup(projection, "ny", None)
    if nz is None:
        nz = _lookup(vertical, "nz", None)
    missing = [name for name, value in {"nx": nx, "ny": ny, "nz": nz}.items() if value is None]
    if missing:
        raise ValueError(f"grid is missing {', '.join(missing)}")
    return int(nx), int(ny), int(nz)


def _lookup(obj: Any, name: str, default: Any = None) -> Any:
    return _host_materialized_value(_lookup_raw(obj, name, default))


def _lookup_raw(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _coerce_datetime(value: datetime | date | str) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip().replace("Z", "")
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _wrf_time_string(value: datetime) -> str:
    return value.strftime("%Y-%m-%d_%H:%M:%S")
