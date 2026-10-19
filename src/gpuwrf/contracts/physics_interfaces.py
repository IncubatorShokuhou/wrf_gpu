"""Frozen v0.6.0 physics adapter interfaces.

This module defines the typed, allocation-free contract that per-scheme lanes
must implement before they are wired into ``runtime.operational_mode``. It does
not run a scheme and does not import JAX. The actual arrays are intentionally
typed as ``Any`` because JAX, NumPy, and savepoint replay payloads all need to
flow through the same interface during oracle work.

The contract separates:

* ``PhysicsTendency``: rates, direct replacements, and accumulator increments
  returned by one scheme call.
* ``PhysicsCarry``: persistent WRF ``state:`` members that are not dycore
  ``State`` leaves, following the existing Noah-MP sibling-carry pattern.
* ``PhysicsStepSpec``: per-option ownership, WRF call slot, State reads/writes,
  carry reads/writes, diagnostics, and required oracle gate.
"""

from __future__ import annotations

from .ice_diagnostics import ICE_DIAGNOSTIC_LEAVES, ice_diagnostic_leaves

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .physics_registry import (
    ACCEPTED_BL_PBL_PHYSICS,
    ACCEPTED_CU_PHYSICS,
    ACCEPTED_MP_PHYSICS,
    ACCEPTED_RA_LW_PHYSICS,
    ACCEPTED_RA_SW_PHYSICS,
    ACCEPTED_SF_SFCLAY_PHYSICS,
    ACCEPTED_SF_SURFACE_PHYSICS,
    ACCUMULATORS,
    CUMULUS_CARRY_MEMBERS,
    CUMULUS_TENDENCY_MEMBERS,
    LAND_CARRY_MEMBERS,
    MOIST_SPECIES_ALL,
    NOAH_CLASSIC_NUM_SOIL_LAYERS,
    NUMBER_SPECIES,
    VOLUME_SPECIES,
    PBL_CARRY_MEMBERS,
    PBL_DIAGNOSTIC_MEMBERS,
    PHYSICS_REGISTRY_VERSION,
    assert_registry_consistent,
    state_leaves_for_mp,
)


PHYSICS_INTERFACE_VERSION = "v0.6.0-S0-frozen-2026-06-04-consolidation3-bmj2-extension"

ArrayLike = Any
_EMPTY: Mapping[str, Any] = MappingProxyType({})


def _empty_mapping() -> Mapping[str, Any]:
    return _EMPTY


# WRF ARW call slots for the physics drivers. The current port's operational
# loop bridges these slots in its own order until a scheme lane proves parity;
# savepoint oracle gates must use the WRF slot named here.
WRF_CALL_ORDER_SLOTS: tuple[str, ...] = (
    "first_rk_pre_radiation_driver",
    "first_rk_radiation_driver",
    "first_rk_surface_driver",
    "first_rk_pbl_driver",
    "first_rk_cumulus_driver",
    "first_rk_fdda_driver",
    "solve_em_microphysics_driver",
)

STATE_TENDENCY_KEYS: tuple[str, ...] = (
    *ICE_DIAGNOSTIC_LEAVES,
    "u",
    "v",
    "w",
    "theta",
    # MOIST_SPECIES_ALL = the v0.2.0..v0.16 six-class core PLUS the v0.17 hail
    # class qh; VOLUME_SPECIES (qvolg/qvolh) are the predicted-density volumes.
    # A hail microphysics scheme (WSM7/WDM7) replaces qh (and a future
    # predicted-density scheme the volumes) in its PhysicsTendency.
    *MOIST_SPECIES_ALL,
    *VOLUME_SPECIES,
    *NUMBER_SPECIES,
    "qke",
    "ustar",
    "theta_flux",
    "qv_flux",
    "tau_u",
    "tau_v",
    "rhosfc",
    "fltv",
    # Existing surface State diagnostics written by the TEMF surface layer.
    "hfx",
    "qfx",
    "qsfc",
    "t_skin",
    "mol",
    "roughness_m",
    "soil_moisture",
    "mavail",
    # Conditional P3 State members (not the shared legacy append-order union).
    "qir", "qib", "qi2", "Ni2", "qir2", "qib2", "qzi", "mp_th_old", "mp_qv_old",
    "re_cloud", "re_ice",
    # EEPS/KEPS keys append after main's existing interface inventory.
    "epsilon_pbl", "tpe_pbl", "pr_pbl", "pek_adv", "pep_adv",
    "tke_adv", "diss_adv", "tpe_adv",
    "roughness_m",
    "qt", "qrimef", "f_ice_phy", "f_rain_phy", "f_rimef_phy", "ferrier_sr",
    "acphysc", "acphyse", "acphysd", "acphyss", "acphysm", "acphysf",
    'ntu_qdcn', 'ntu_qtcn', 'ntu_qccn', 'ntu_qrcn', 'ntu_qnin', 'ntu_fi', 'ntu_fs', 'ntu_vi', 'ntu_vs', 'ntu_vg', 'ntu_ai', 'ntu_as', 'ntu_ag', 'ntu_ah', 'ntu_i3m', 'ntu_rainnc', 'ntu_snownc', 'ntu_sr',
    "re_cloud_gsfc", "re_rain_gsfc", "re_ice_gsfc", "re_snow_gsfc", "re_graupel_gsfc", "re_hail_gsfc", "goddard_sr", "phys_tot", "nwfa2d", "nifa2d", "thompson_sr", "thompson_rainnc", "thompson_snownc",
    "re_snow", "qvoli", "qaoli", "qvoli2", "qaoli2", "qi3", "Ni3", "qvoli3", "qaoli3",
    *state_leaves_for_mp(11)[9:],
)
ACCUMULATOR_UPDATE_KEYS: tuple[str, ...] = ACCUMULATORS


@dataclass(frozen=True)
class PhysicsTendency:
    """One scheme-call update payload.

    ``state_tendencies`` contains per-second tendencies keyed by State leaf.
    ``state_replacements`` contains post-scheme replacement arrays for schemes
    ported in WRF's in-place style. An adapter may use one style per leaf, never
    both for the same leaf in the same call. ``accumulator_increments`` contains
    per-call millimeter increments for precipitation accumulators.
    """

    state_tendencies: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    state_replacements: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    accumulator_increments: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    diagnostics: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)

    def validate_keys(self) -> None:
        """Fail if an adapter returns an unknown State/accumulator key."""

        allowed_state = set(STATE_TENDENCY_KEYS)
        overlap = set(self.state_tendencies).intersection(self.state_replacements)
        if overlap:
            raise ValueError(f"PhysicsTendency leaf returned as both tendency and replacement: {sorted(overlap)}")
        for key in self.state_tendencies:
            if key not in allowed_state:
                raise ValueError(f"unknown state_tendency key {key!r}")
        for key in self.state_replacements:
            if key not in allowed_state:
                raise ValueError(f"unknown state_replacement key {key!r}")
        for key in self.accumulator_increments:
            if key not in ACCUMULATOR_UPDATE_KEYS:
                raise ValueError(f"unknown accumulator increment key {key!r}")


@dataclass(frozen=True)
class PhysicsDiagnostics:
    """Optional diagnostic side channels grouped by physics family."""

    microphysics: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    surface_layer: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    pbl: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    cumulus: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    land_surface: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    radiation: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)


@dataclass(frozen=True)
class PhysicsCarry:
    """Persistent WRF ``state:`` members outside the dycore State contract."""

    microphysics: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    surface_layer: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    pbl: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    cumulus: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    land_surface: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)
    radiation: Mapping[str, ArrayLike] = field(default_factory=_empty_mapping)


@dataclass(frozen=True)
class PhysicsStepResult:
    """Adapter return object for one physics scheme call."""

    tendency: PhysicsTendency
    carry: PhysicsCarry = field(default_factory=PhysicsCarry)
    diagnostics: PhysicsDiagnostics = field(default_factory=PhysicsDiagnostics)


@dataclass(frozen=True)
class PhysicsStepSpec:
    """Frozen per-scheme adapter contract and file-ownership metadata."""

    family: str
    option: int
    name: str
    wrf_slot: str
    owner_module: str
    oracle: str
    reads_state: tuple[str, ...]
    writes_state: tuple[str, ...]
    reads_carry: tuple[str, ...] = ()
    writes_carry: tuple[str, ...] = ()
    returns_accumulators: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    notes: str = ""
    # ``variant`` disambiguates families where one namelist option maps to more
    # than one independent adapter sharing a kernel. Radiation is the only such
    # case in the v0.6.0 menu: ra_lw_physics=4 and ra_sw_physics=4 are distinct
    # namelist switches that both select the RRTMG column code, so the freeze
    # carries two specs under option 4 keyed by variant ``"lw"`` / ``"sw"``.
    variant: str = ""


def _mp_spec(
    option: int,
    name: str,
    owner: str,
    oracle: str,
    *,
    diagnostics: tuple[str, ...] = (),
    accumulators: tuple[str, ...] = ("rain_acc", "snow_acc", "graupel_acc", "ice_acc"),
    extra_state: tuple[str, ...] = (),
) -> PhysicsStepSpec:
    leaves = (*state_leaves_for_mp(option), *extra_state)
    return PhysicsStepSpec(
        family="microphysics",
        option=option,
        name=name,
        wrf_slot="solve_em_microphysics_driver",
        owner_module=owner,
        oracle=oracle,
        reads_state=("theta", "p", "pb", "ph", "mu", *leaves),
        writes_state=("theta", *leaves),
        returns_accumulators=accumulators,
        diagnostics=diagnostics,
    )


SCHEME_STEP_SPECS: tuple[PhysicsStepSpec, ...] = (
    PhysicsStepSpec(
        family="surface_nudging", option=1, name="PSU surface analysis nudging",
        wrf_slot="first_rk_fdda_driver",
        owner_module="src/gpuwrf/assimilation/fdda_surface.py",
        oracle="Pristine WRF FDDAGD/SFDDAGD savepoint + caller lifetime: proofs/s2_sfdda/savepoints/MANIFEST.json",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "phb"),
        writes_state=("u", "v", "theta", "qv"),
        reads_carry=("fdda", "pblh", "surface_layer.regime", "noahmp.znt"),
        writes_carry=("fdda.surface", "fdda.surface_cursor", "fdda.rundgdten", "fdda.rvndgdten",
                      "fdda.rthndgdten", "fdda.rqvndgdten", "fdda.hfx_fdda", "fdda.sda_hfx", "fdda.sda_qfx"),
        diagnostics=("HFX_FDDA",),
        notes="grid_sfdda=1, grid_fdda=1; MYNN 5 + Noah-MP 4. CPU-oracle-qualified; GPU pending. "
              "Surface rates precede 3D nudging; held rates coupled and decoupled once per own-step. "
              "Surface+spectral is WRF fatal; classic-Noah indirect soil updates refused.",
    ),
    PhysicsStepSpec(
        family="surface_nudging", option=2, name="FASDAS surface analysis nudging",
        wrf_slot="first_rk_fdda_driver",
        owner_module="src/gpuwrf/assimilation/fdda_surface.py",
        oracle="Pristine WRF FDDAGD/SFDDAGD savepoint + caller lifetime: proofs/s2_sfdda/savepoints/MANIFEST.json",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "phb"),
        writes_state=("u", "v", "theta", "qv"),
        reads_carry=("fdda", "pblh", "surface_layer.regime", "noahmp.znt"),
        writes_carry=("fdda.surface", "fdda.surface_cursor", "fdda.rundgdten", "fdda.rvndgdten",
                      "fdda.rthndgdten", "fdda.rqvndgdten", "fdda.hfx_fdda", "fdda.sda_hfx", "fdda.sda_qfx"),
        diagnostics=("HFX_FDDA",),
        notes="grid_sfdda=2, grid_fdda=1; MYNN 5 + Noah-MP 4. CPU-oracle-qualified; GPU pending. "
              "Surface rates precede 3D nudging; held rates coupled and decoupled once per own-step. "
              "Surface+spectral is WRF fatal; classic-Noah indirect soil updates refused.",
    ),
    PhysicsStepSpec(
        family='cumulus',option=7,name='CAM Zhang-McFarlane',wrf_slot='first_rk_cumulus_driver',
        owner_module='src/gpuwrf/coupling/cuplus.py',
        oracle='Pristine WRF 4.7.1 complete CAMZM_DRIVER savepoint/column gate: proofs/v040/cuplus/full_callers; literal REAL/DOUBLE boundary BYTE',
        reads_state=('u','v','theta','qv','qc','qi','p','pb','ph','mu','qke','pblh','hfx','qfx','ustar','mavail','t_skin','xland'),
        writes_state=('u','v','theta','qv','qc','qi','Ni'),
        reads_carry=('cuplus','camuw_pbl'),writes_carry=('cuplus',),returns_accumulators=('rainc_acc',),
        diagnostics=('RAINCV','PRATEC','CAPE_OUT','CUTOP','CUBOT','DLF_OUT','MU_OUT','MD_OUT','CMFMC','ENTROPY_FAILED','DLF_CLIPPED','VAPOR_FLOORED'),
        notes='Dedicated original mixed REAL/DOUBLE column; momentum/number rates at WRF RK slots, actual PBL TPERT/KPBL, resident held outputs. Experimental CPU-oracle-qualified; GPU/coupled-forecast qualification pending.',
    ),
    PhysicsStepSpec(
        family='cumulus',option=10,name='KF-CuP',wrf_slot='first_rk_cumulus_driver',
        owner_module='src/gpuwrf/coupling/cuplus.py',
        oracle='Pristine WRF 4.7.1 complete KF_cup_CPS savepoint/column gate: proofs/v040/cuplus/full_callers; literal REAL4 BYTE; radiation block proofs/v040/cuplus/fixtures.json',
        reads_state=('u','v','w','theta','qv','p','pb','ph','mu','pblh','hfx','qfx','ustar','t_skin','xland'),
        writes_state=('theta','qv','qc','qr','qi','qs'),
        reads_carry=('cuplus',),writes_carry=('cuplus',),returns_accumulators=('rainc_acc',),
        diagnostics=('RAINCV','PRATEC','NCA','CUTOP','CUBOT','CUPFLAG','ISHALL','CLDFRA_CUP','QC_IC_CUP','TAUCLOUD','MFUP_CUP','MFDN_CUP'),
        notes='Dedicated REAL CuP ensemble/shallow/deep body, original Registry PDF controls, cloud lifetime and saved/restored radiation condensate view. Experimental CPU-oracle-qualified; GPU/coupled-forecast qualification pending.',
    ),
    PhysicsStepSpec(
        family='cumulus',option=11,name='MSKF',wrf_slot='first_rk_cumulus_driver',
        owner_module='src/gpuwrf/coupling/cuplus.py',
        oracle='Pristine WRF 4.7.1 complete MSKF_CPS/default CMT savepoint/column gate: proofs/v040/cuplus/full_callers; literal REAL4 and original DOUBLE CMT BYTE',
        reads_state=('u','v','w','theta','qv','p','pb','ph','mu','pblh','hfx','qfx','ustar'),
        writes_state=('u','v','theta','qv','qc','qr','qi','qs'),
        reads_carry=('cuplus',),writes_carry=('cuplus',),returns_accumulators=('rainc_acc',),
        diagnostics=('RAINCV','PRATEC','NCA','CUTOP','CUBOT','CLDFRA_DP_KF','CLDFRA_SH_KF','QC_KF','QI_KF','TIMEC'),
        notes='Dedicated scale-aware KF and genuine DOUBLE CMT; actual surface ZOL; forced WRF cloud feedback. Calm singularity retained/reported and compiler excess disclosed in UPSTREAM_WRF_BUGS.md. Experimental CPU-oracle-qualified; GPU/coupled-forecast qualification pending.',
    ),
    _mp_spec(
        1,
        "Kessler warm rain",
        "src/gpuwrf/physics/microphysics_kessler.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:kessler",
    ),
    _mp_spec(
        2,
        "Purdue-Lin",
        "src/gpuwrf/physics/microphysics_lin.py",
        "v0.6.0 Purdue-Lin pristine-WRF savepoint parity gate at module_microphysics_driver.F:lin",
    ),
    _mp_spec(
        3,
        "WSM3 simple ice",
        "src/gpuwrf/physics/microphysics_wsm3.py",
        "v0.6.0 WSM3 pristine-WRF savepoint parity gate at module_microphysics_driver.F:wsm3",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        4,
        "WSM5",
        "src/gpuwrf/physics/microphysics_wsm5.py",
        "v0.6.0 WSM5 pristine-WRF savepoint parity gate at module_microphysics_driver.F:wsm5",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        6,
        "WSM6",
        "src/gpuwrf/physics/microphysics_wsm6.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:wsm6",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        8,
        "Thompson",
        "src/gpuwrf/physics/thompson_column.py",
        "operational / Tier-4 RMSE validated vs CPU-WRF corpus, NOT isolated-unmodified-WRF-savepoint-proven",
        diagnostics=("re_cloud", "re_ice", "re_snow", "ThompsonTendencySideChannel"),
    ),
    _mp_spec(
        10,
        "Morrison two-moment",
        "src/gpuwrf/physics/microphysics_morrison.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:morrison two moment",
    ),
    _mp_spec(
        13,
        "SBU-YLin",
        "src/gpuwrf/physics/microphysics_sbu_ylin.py",
        "v0.17 SBU-YLin pristine-WRF single-column savepoint parity gate at "
        "module_mp_sbu_ylin.F (proofs/v017/run_sbu_ylin_parity.py / "
        "savepoints_sbu_ylin)",
        diagnostics=("ri3d",),
    ),
    _mp_spec(
        14,
        "WDM5",
        "src/gpuwrf/physics/microphysics_wdm5.py",
        "v0.13 WDM5 pristine-WRF single-column savepoint parity gate at "
        "module_mp_wdm5.F (proofs/v013/t3_wdm5_oracle.py/.json)",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        16,
        "WDM6",
        "src/gpuwrf/physics/microphysics_wdm6.py",
        "M20 physics-oracle factory savepoint at module_microphysics_driver.F:wdm6",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
    ),
    _mp_spec(
        28,
        "Thompson aerosol-aware",
        "src/gpuwrf/physics/thompson_aero_column.py",
        "v0.16 aerosol-aware Thompson pristine-WRF grid savepoint parity gate at "
        "module_mp_thompson.F:mp_gt_driver (mp_physics=28; "
        "proofs/v016/thompson_aero_savepoint_parity.py/.json)",
    ),
    _mp_spec(
        97,
        "Goddard GCE",
        "src/gpuwrf/physics/microphysics_goddard.py",
        "v0.17 Goddard GCE (gsfcgcescheme) single-column savepoint parity gate vs "
        "unmodified phys/module_mp_gsfcgce.F (proofs/v090/run_goddard_parity.py, "
        "proofs/v090/goddard_mp_r2_savepoint_parity.json; 5/5 cases, ~machine "
        "precision vs the fp64 transparency oracle)",
    ),
    # v0.17 WSM7 = WSM6 + a separate precipitating hail class (qh leaf + hail_acc
    # surface accumulator). Savepoint-parity-proven against the unmodified
    # phys/module_mp_wsm7.F (proofs/v013/run_wsm7_parity.py, 6/6 PASS).
    _mp_spec(
        24,
        "WSM7",
        "src/gpuwrf/physics/microphysics_wsm7.py",
        "v0.17 WSM7 pristine-WRF single-column savepoint parity gate at "
        "module_mp_wsm7.F (proofs/v013/run_wsm7_parity.py/.json)",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
        accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc", "hail_acc"),
    ),
    # v0.17 WDM7 = WDM6 double-moment warm rain (Nc/Nr/Nn) + a separate
    # single-moment precipitating hail class (qh + hail_acc; no Nh).
    # Savepoint-parity-proven against the unmodified phys/module_mp_wdm7.F
    # (proofs/v013_wdm7/run_wdm7_parity.py, 6/6 PASS).
    _mp_spec(
        26,
        "WDM7",
        "src/gpuwrf/physics/microphysics_wdm7.py",
        "v0.17 WDM7 pristine-WRF single-column savepoint parity gate at "
        "module_mp_wdm7.F (proofs/v013_wdm7/run_wdm7_parity.py/.json)",
        diagnostics=("re_cloud", "re_ice", "re_snow"),
        accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc", "hail_acc"),
    ),
    _mp_spec(
        17,
        "NSSL legacy alias of 18 (WRF 4.7.1)",
        "src/gpuwrf/physics/microphysics_nssl2mom.py",
        "Pristine-WRF normalization (proofs/s2_mpmore/normalization.json) resolves mp=17 "
        "to the default mp=18 configuration; stage + end-to-end savepoint parity vs "
        "proofs/v034/f2_oracles/nssl_2mom (17 cases, fp64 <=1e-12, fp32 dual-reference band). "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending; root domains only.",
        accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc", "hail_acc"),
    ),
    _mp_spec(
        18,
        "NSSL 2-moment",
        "src/gpuwrf/physics/microphysics_nssl2mom.py",
        "v0.3.4 JAX port (gpuwrf.physics.nssl2mom) of unmodified phys/module_mp_nssl_2mom.F "
        "(default mp=18 config), stage + end-to-end savepoint parity vs the pristine-WRF oracle "
        "proofs/v034/f2_oracles/nssl_2mom (17 cases, fp64 <=1e-12, fp32 dual-reference band); "
        "CPU-oracle-qualified, GPU/coupled-forecast qualification pending; root domains only.",
        accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc", "hail_acc"),
    ),
    _mp_spec(
        22,
        "NSSL 2-moment without hail",
        "src/gpuwrf/physics/nssl2mom/adapter.py",
        "Pristine-WRF legacy22 caller and REAL4/REAL8 full-column savepoints "
        "(proofs/s2_mpmore/nssl22/STATUS22.json); frozen NSSL rounding bands "
        "and independently measured original22 input-ULP CCN controls. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending; root domains only.",
        extra_state=("qvolg",),
    ),
    _mp_spec(
        40,
        "Morrison aerosol-aware",
        "src/gpuwrf/physics/microphysics_morrison_aero.py",
        "v0.3.4 O1 scan-wired at aercu_opt=0 (WRF default; constant droplets, "
        "Nc = WRF qnc): single-column fp32+fp64 pristine-WRF oracle savepoints "
        "vs unmodified phys/module_mp_morr_two_moment_aero.F at aercu_opt=0 "
        "(proofs/v034/f2_oracles/morrison_aero_opt0) and aercu_opt=2 "
        "(proofs/v022/f2_oracles/morrison_aero). aercu_opt>0 needs cu_physics=11 "
        "(MSKF, not ported) and is refused by the namelist check. At aercu_opt=0 "
        "pristine WRF physics_init sets has_reqc/i/s=0 for mp=40: RRTMG uses its default "
        "radii here too. Morrison radius blending is only in the refused aercu_opt>0 MSKF path "
        "(proofs/v040/nssl_cumulus/radius_audit.json).",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=1,
        name="YSU",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_ysu.py",
        oracle="M20 physics-oracle factory savepoint at module_pbl_driver.F:YSU",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv"),
        diagnostics=("pblh",),
        notes="topo_wind 1/2: static start_em CTOPO/CTOPO2 (OperationalNamelist.topo_wind_static, "
        "io/topo_wind_static.py) scale the momentum surface drag; CTOPO2 < 1 blends U10/V10 "
        "(history) toward the lowest model wind (bl_ysu.F90:1309-1316, 1405-1408).",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=2,
        name="MYJ",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_myj.py",
        oracle="v0.13 operational savepoint parity vs unmodified WRF module_bl_myjpbl.F "
        "(proofs/v013/myj_janjic_oracle.py/.json; reuses v0.6.0 fp64 savepoints)",
        reads_state=("u", "v", "theta", "qv", "qke", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[2] + ("qnse",),
        writes_carry=PBL_CARRY_MEMBERS[2] + ("qnse",),
        diagnostics=("pblh", "kpbl", "mixht", "tke_pbl", "exch_h", "exch_m", "el_pbl"),
        notes="v0.13 OPERATIONAL (jit/vmap-traceable bl_myj.myj_columns, scan-wired via "
        "physics.myj_adapters.myj_pbl_adapter; host-NumPy reference in pbl_myj.py). "
        "Mandatory pair with sf_sfclay_physics=2; the TKE carry rides State.qke (q^2 "
        "convention). Pairing enforced by namelist_check and dispatcher resolution. "
        "t-cfix-sfc F12: reads THIS step's MYJSFC AKHS/AKMS/THZ0/QZ0/UZ0/VZ0/CT/UST/QSFC "
        "(held in the Janjic-family qnse carry), ELFLX = post-LSM LH, CHKLOWQ = 1 after "
        "an LSM, SNOW = Noah-MP SWE; returns THZ0/QZ0/QSFC (module_bl_myjpbl.F:540-563).",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=3,
        name="GFS",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_gfs.py",
        oracle="v0.17 operational savepoint parity vs unmodified WRF module_bl_gfs.F "
        "(BL_GFS -> MONINP -> TRIDI2/TRIDIN/TRIDIT; proofs/v017/gfs_oracle.py/.json, "
        "fp64 kind_phys-native ~1e-13, 6 regimes, NOT a self-compare)",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv"),
        diagnostics=("pblh", "kpbl"),
        notes="v0.17 OPERATIONAL (jit/vmap-traceable bl_gfs.gfs_columns, scan-wired via "
        "coupling.scan_adapters.gfs_pbl_adapter -> PBL_SCAN_ADAPTERS[3]). NCEP-GFS "
        "Hybrid-EDMF-ancestor nonlocal-K PBL (Hong-Pan lineage); WRF-mandatory pair with "
        "sf_sfclay_physics=3 (isfc=2) whose single SF_GFS solve supplies PSIM/PSIH/BR/WSPD/"
        "GZ1OZ0 (coupling.sfclay_selected); nonlocal-K so no prognostic carry.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=4,
        name="QNSE-EDMF",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_qnse.py",
        oracle="Pristine WRF 4.7.1 REAL4/REAL8 savepoints from module_bl_qnsepbl.F "
        "and module_bl_mfshconvpbl.F: proofs/s2_qnse/oracle.json, mf_oracle.json; "
        "literal REAL4 BYTE, compiled PBL independent input-ULP enclosure; "
        "EDMF compiled-spread excess disclosed in compiler_rounding.json.",
        reads_state=("u", "v", "theta", "qv", "qc", "qke", "p", "ph", "mu", "ustar", "hfx", "qfx", "qsfc", "xland"),
        writes_state=("u", "v", "theta", "qv", "qc", "qke", "qsfc"),
        reads_carry=("qnse",),
        writes_carry=("qnse",),
        diagnostics=("pblh", "el_pbl", "exch_h", "exch_m", "rc_mf"),
        notes="CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Mandatory sfclay=4 pair; coupling.qnse_adapter consumes the retained "
        "single surface solve and post-land HFX/LH. mfshconv=0/1 selects the "
        "ARW MFSHCONVPBL call before QNSEPBL; qke=2*TKE. Twelve REAL4 "
        "QnseCarry leaves and the surface history cache are required on restart. "
        "PBLH/EL are State diagnostics; optional full updraft diagnostics remain "
        "available through the column API and are not persistent carry.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=11,
        name="Shin-Hong",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_shinhong.py",
        oracle="v0.18 savepoint-derived JAX/vmap kernel checked against the v090 faithful host-NumPy "
        "reference port of unmodified phys/module_bl_shinhong.F; forecast-driving "
        "tendencies/EXCH_H/PBLH/KPBL/WSTAR/DELTA are parity-gated. TKE/EL are "
        "diagnostic-only here and retain explicit residuals vs the v090 PARTIAL "
        "reference (TKE rel ~=0.285, EL rel ~=0.013; fp32-sensitive upstream oracle)",
        reads_state=("u", "v", "theta", "qv", "qke", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[11],
        writes_carry=PBL_CARRY_MEMBERS[11],
        diagnostics=("pblh", "kpbl", "tke_pbl", "el_pbl", "exch_h"),
        notes="v0.18 OPERATIONAL dynamics-green: scale-aware (grid-dependent) YSU-family PBL "
        "(Shin-Hong 2015), scan-wired via coupling.scan_adapters.shinhong_pbl_adapter "
        "-> PBL_SCAN_ADAPTERS[11]. Consumes revised-MM5 surface forcing (sf_sfclay=1) "
        "and grid dx/dy for pu/pq/pthnl/pthl/ptke partition functions. Follow up by "
        "refining TKE/EL if a faithful pristine-WRF Shin-Hong TKE oracle is built. "
        "topo_wind 1/2: static start_em CTOPO/CTOPO2 (OperationalNamelist.topo_wind_static) "
        "scale the surface drag and blend U10/V10 (module_bl_shinhong.F:1387, 1469-1473).",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=12,
        name="GBM TKE",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_gbm.py",
        oracle="v0.18 fp64 parity-green vs single-column pristine-WRF savepoints "
        "from unmodified phys/module_bl_gbmpbl.F (proofs/v018/gbm_pbl12_jax_parity.json)",
        reads_state=("u", "v", "theta", "qv", "qc", "qke", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qc", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[12],
        writes_carry=PBL_CARRY_MEMBERS[12],
        diagnostics=("pblh", "kpbl", "tke_pbl", "el_pbl", "exch_tke"),
        notes="v0.18 OPERATIONAL: Grenier-Bretherton-McCaa moist prognostic-TKE PBL, "
        "scan-wired via coupling.scan_adapters.gbm_pbl_adapter -> PBL_SCAN_ADAPTERS[12]. "
        "The JAX/vmap kernel preserves GBMPBL's two-pass wrapper, advances qc/qke, and "
        "matches the fp64 pristine-WRF savepoint oracle at roundoff.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=5,
        name="MYNN",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/mynn_pbl.py",
        oracle="operational / Tier-4 RMSE validated vs CPU-WRF corpus, NOT isolated-unmodified-WRF-savepoint-proven",
        reads_state=("u", "v", "theta", "qv", "qke", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[5],
        writes_carry=PBL_CARRY_MEMBERS[5],
        diagnostics=("pblh", "tke_pbl", "sh3d", "sm3d", "tsq", "qsq", "cov", "el_pbl"),
    ),
    PhysicsStepSpec(
        family="pbl",
        option=7,
        name="ACM2",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_acm2.py",
        oracle="M20 physics-oracle factory savepoint at module_pbl_driver.F:ACM2",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv"),
        diagnostics=("pblh",),
    ),
    PhysicsStepSpec(
        family="pbl",
        option=8,
        name="BouLac",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_boulac.py",
        oracle="v0.6.0 BouLac WRF savepoint parity gate at module_bl_boulac.F",
        reads_state=("u", "v", "theta", "qv", "qc", "qke", "p", "pb", "ph", "mu", "ustar"),
        writes_state=("u", "v", "theta", "qv", "qc", "qke"),
        reads_carry=PBL_CARRY_MEMBERS[8],
        writes_carry=PBL_CARRY_MEMBERS[8],
        diagnostics=("pblh", "tke_pbl", "dlk", "exch_h", "exch_m"),
    ),
    PhysicsStepSpec(
        family="pbl",
        option=9,
        name="CAM-UW",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_camuw.py",
        oracle="v0.3.4 pristine-WRF CAM-UW column savepoint oracle proofs/v034/camuw_oracle "
        "(unmodified WRF objects, WRF CAM_INIT incl. esinti, 3 carried steps; 13 synthetic "
        "regimes + 13 Swiss CPU-WRF columns): every output <= 1-2 REAL ulp "
        "(tests/test_v034_camuw_oracle_parity.py).",
        reads_state=("u", "v", "theta", "qv", "qc", "qi", "Ni", "p", "ph", "mu", "ustar", "hfx", "qfx"),
        writes_state=("u", "v", "theta", "qv", "qc", "qi", "Ni"),
        reads_carry=PBL_CARRY_MEMBERS[9],
        writes_carry=PBL_CARRY_MEMBERS[9],
        diagnostics=PBL_DIAGNOSTIC_MEMBERS[9],
        notes="Operational (CPU-oracle-qualified, GPU-unqualified). Carry "
        "OperationalCarry.camuw_pbl = WRF KVM3D/KVH3D/TAURESX2D/TAURESY2D/TKE_PBL. "
        "RTHRATENLW approximated by the held total RTHRATEN (exact at night).",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=10,
        name="TEMF",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_temf.py",
        oracle="Unmodified WRF REAL savepoint gate phys/module_bl_temf.F; proofs/s2_temf/pbl_oracle.npz",
        reads_state=("u", "v", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux", "t_skin"),
        writes_state=("u", "v", "theta", "qv", "qc"),
        reads_carry=PBL_CARRY_MEMBERS[10],
        writes_carry=PBL_CARRY_MEMBERS[10],
        diagnostics=PBL_DIAGNOSTIC_MEMBERS[10],
        notes="TEMF paired with sfclay=10; CPU-oracle-qualified; GPU/coupled-forecast qualification pending.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=16,
        name="EEPS epsilon",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_eeps.py",
        oracle="Unmodified WRF 4.7.1 REAL4 full-column savepoint gate "
        "phys/module_bl_eepsilon.F; proofs/s2_keps/full_zero.json (30 carried records)",
        reads_state=("u", "v", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qc", "qi", "qke", "epsilon_pbl", "pek_adv", "pep_adv"),
        reads_carry=PBL_CARRY_MEMBERS[16],
        writes_carry=PBL_CARRY_MEMBERS[16],
        diagnostics=PBL_DIAGNOSTIC_MEMBERS[16],
        notes="EXPERIMENTAL: CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Requires sfclay=1 and full carry/history; State.qke holds physical TKE. "
        "Registry scalars transported; compiler-zero upstream undefined reads materially sensitive "
        "(proofs/s2_keps/UPSTREAM_WRF_BUGS.md).",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=17,
        name="KEPS k-epsilon",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/pbl_keps.py",
        oracle="Unmodified WRF 4.7.1 REAL4 full-column savepoint gate "
        "phys/module_bl_keps.F; proofs/s2_keps/full_zero.json (30 carried records)",
        reads_state=("u", "v", "theta", "qv", "qc", "p", "pb", "ph", "mu", "ustar", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv", "qc", "qke", "epsilon_pbl", "tpe_pbl", "pr_pbl", "tke_adv", "diss_adv", "tpe_adv"),
        reads_carry=PBL_CARRY_MEMBERS[17],
        writes_carry=PBL_CARRY_MEMBERS[17],
        diagnostics=PBL_DIAGNOSTIC_MEMBERS[17],
        notes="EXPERIMENTAL: CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Requires sfclay=1 and full carry/history; State.qke holds physical TKE with Z-staggered history. "
        "Registry scalars transported; compiler-zero upstream undefined reads materially sensitive "
        "(proofs/s2_keps/UPSTREAM_WRF_BUGS.md); strict 1x PD compiler-interval audit OPEN, frozen C05 passes.",
    ),
    PhysicsStepSpec(
        family="pbl",
        option=99,
        name="MRF",
        wrf_slot="first_rk_pbl_driver",
        owner_module="src/gpuwrf/physics/bl_mrf.py",
        oracle="v0.13 MRF WRF savepoint parity gate at phys/module_bl_mrf.F (proofs/v013/mrf_oracle.py)",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "mu", "ustar", "t_skin", "theta_flux", "qv_flux"),
        writes_state=("u", "v", "theta", "qv"),
        reads_carry=PBL_CARRY_MEMBERS[99],
        writes_carry=PBL_CARRY_MEMBERS[99],
        diagnostics=("pblh", "kpbl"),
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=1,
        name="sfclayrev",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/surface_layer_sfclayrev.py",
        oracle="M20 physics-oracle factory savepoint at module_surface_driver.F:sfclayrev",
        reads_state=("u", "v", "theta", "qv", "t_skin", "soil_moisture", "xland", "roughness_m"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"),
        diagnostics=("T2", "Q2", "U10", "V10", "PSFC", "HFX", "LH", "UST"),
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=2,
        name="Janjic Eta surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/sfclay_janjic.py",
        oracle="v0.13 operational savepoint parity vs unmodified WRF module_sf_myjsfc.F "
        "(proofs/v013/myj_janjic_oracle.py/.json; reuses v0.6.0 fp64 savepoints)",
        reads_state=("u", "v", "theta", "qv", "qc", "qke", "p", "ph", "mu", "t_skin", "xland", "roughness_m",
                     "ustar", "mavail", "qsfc"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv", "roughness_m",
                      "qsfc", "hfx", "qfx"),
        reads_carry=("qnse",),
        writes_carry=("qnse",),
        diagnostics=("T2", "Q2", "U10", "V10", "PSFC", "HFX", "QFX", "LH", "UST", "AKHS", "AKMS", "pblh"),
        notes="v0.13 OPERATIONAL (jit/vmap-batched via sf_myj.myjsfc_columns over the "
        "traceable sfclay_janjic kernel, scan-wired via physics.myj_adapters."
        "janjic_sfclay_adapter). Mandatory pair with bl_pbl_physics=2; runs FIRST in "
        "the WRF call chain and supplies the MYJ PBL coupling. Pairing enforced by "
        "namelist_check and dispatcher resolution. t-cfix-sfc F3/F12: ONE MYJSFC call per step "
        "(coupling.myj_surface) with carried THZ0/QZ0/UZ0/VZ0/AKHS/AKMS (NTSD==1 cold start) "
        "feeds Noah-MP, MYJPBL and history; pristine REAL4 oracle proofs/t_cfix_sfc.",
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=4,
        name="QNSE surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/sfclay_qnse.py",
        oracle="Pristine WRF 4.7.1 QNSESFC/QNSESFCINIT REAL4/REAL8 savepoints: "
        "proofs/s2_qnse/oracle.json and surface_tables.json; literal REAL4 BYTE "
        "and compiled independent input-ULP enclosure in enclosures.json.",
        reads_state=("u", "v", "theta", "qv", "qc", "qke", "p", "ph", "mu", "t_skin", "xland", "roughness_m", "ustar", "mavail", "qsfc", "hfx", "qfx"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv", "hfx", "qfx", "qsfc", "roughness_m"),
        reads_carry=("qnse",),
        writes_carry=("qnse",),
        diagnostics=("T2", "Q2", "U10", "V10", "PSFC", "HFX", "QFX", "LH", "UST", "AKHS", "AKMS", "pblh"),
        notes="CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Mandatory PBL=4 pair. Runs once before land physics; retained surface "
        "handles feed land/PBL and native history output without a second solve. "
        "QnseCarry preserves surface exchange memory; history diagnostics are "
        "required even when the global history cache flag is off.",
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=5,
        name="MYNN surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/surface_layer.py",
        oracle="operational / Tier-4 RMSE validated vs CPU-WRF corpus, NOT isolated-unmodified-WRF-savepoint-proven",
        reads_state=("u", "v", "theta", "qv", "t_skin", "soil_moisture", "xland", "roughness_m"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"),
        diagnostics=("T2", "Q2", "U10", "V10", "PSFC", "HFX", "LH", "UST"),
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=7,
        name="Pleim-Xiu surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/surface_layer_px.py",
        oracle="M20 physics-oracle factory savepoint at module_surface_driver.F:Pleim-Xiu sfclay",
        reads_state=("u", "v", "theta", "qv", "p", "ph", "mu", "t_skin", "xland", "lakemask", "mavail",
                     "roughness_m", "ustar", "mol", "qsfc", "hfx", "qfx", "pblh"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv", "roughness_m",
                      "mol", "qsfc", "hfx", "qfx"),
        diagnostics=("U10", "V10", "PSFC", "HFX", "LH", "UST"),
        notes="The operational call is coupling.sfclay_selected (t-cfix-sfc F3/F19): ONE call per step on WRF phy_prep operands with carried UST/ZNT/MOL/QSFC/HFX/QFX/PBLH memory, handed to Noah-MP/PBL/history; pristine REAL4 public-call oracle proofs/t_cfix_sfc. PXSFCLAY returns no T2/Q2.",
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=3,
        name="NCEP GFS surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/sfclay_gfs.py",
        oracle="v0.13 Tier-3 fp64 savepoint parity vs unmodified WRF module_sf_gfs.F "
        "(proofs/v013/t3_surface_lsm_oracle.py/.json; faithful fpvs table)",
        reads_state=("u", "v", "theta", "qv", "p", "ph", "mu", "t_skin", "xland", "lakemask", "mavail",
                     "roughness_m", "ustar", "mol", "qsfc", "hfx", "qfx", "pblh"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv", "roughness_m",
                      "mol", "qsfc", "hfx", "qfx"),
        diagnostics=("U10", "V10", "PSFC", "HFX", "QFX", "LH", "UST", "CHS", "CHS2", "CQS2"),
        notes="v0.13 OPERATIONAL (jit/vmap-batched sf_gfs_columns, scan-wired via "
        "coupling.scan_adapters.gfs_sfclay_adapter -> B2 kinematic flux handles). "
        "NCEP Monin-Obukhov bulk-Richardson exchange-coefficient solve; the GFS "
        "land/soil/canopy blocks are bypassed in standalone surface-layer mode. "
        "The operational call is coupling.sfclay_selected (t-cfix-sfc F3/F19): ONE call per step on WRF phy_prep operands with carried UST/ZNT/MOL/QSFC/HFX/QFX/PBLH memory, handed to Noah-MP/PBL/history; pristine REAL4 public-call oracle proofs/t_cfix_sfc.",
    ),
    PhysicsStepSpec(
        family='surface_layer',option=10,name='TEMF surface layer',wrf_slot='first_rk_surface_driver',
        owner_module='src/gpuwrf/physics/sf_temf.py',
        oracle='Unmodified WRF REAL savepoint gate phys/module_sf_temfsfclay.F; proofs/s2_temf/surface_oracle.npz',
        reads_state=('u','v','theta','qv','t_skin','mavail','roughness_m','hfx'),
        writes_state=('ustar','theta_flux','qv_flux','tau_u','tau_v','rhosfc','fltv','hfx','qfx','qsfc'),
        reads_carry=('temf',),writes_carry=('temf',),
        notes='Mandatory TEMF PBL=10 pair; CPU-oracle-qualified; GPU/coupled-forecast qualification pending.',
    ),
    PhysicsStepSpec(
        family="surface_layer",
        option=91,
        name="old MM5 surface layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/sfclay_old_mm5.py",
        oracle="v0.13 Tier-3 fp64 savepoint parity vs unmodified WRF module_sf_sfclay.F "
        "(proofs/v013/t3_surface_lsm_oracle.py/.json)",
        reads_state=("u", "v", "theta", "qv", "p", "ph", "mu", "t_skin", "xland", "lakemask", "mavail",
                     "roughness_m", "ustar", "mol", "qsfc", "hfx", "qfx", "pblh"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv", "roughness_m",
                      "mol", "qsfc", "hfx", "qfx"),
        diagnostics=("T2", "Q2", "TH2", "U10", "V10", "HFX", "QFX", "LH", "UST", "REGIME"),
        notes="v0.13 OPERATIONAL (jit/vmap-batched sfclay_old_mm5_columns, scan-wired via "
        "coupling.scan_adapters.sfclay_old_mm5_adapter -> B2 kinematic flux handles). "
        "Classic MM5 4-regime Monin-Obukhov surface layer (predecessor of sfclayrev=1). "
        "The operational call is coupling.sfclay_selected (t-cfix-sfc F3/F19): ONE call per step on WRF phy_prep operands with carried UST/ZNT/MOL/QSFC/HFX/QFX/PBLH memory, handed to Noah-MP/PBL/history; pristine REAL4 public-call oracle proofs/t_cfix_sfc.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=1,
        name="Kain-Fritsch",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_kf.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:KF",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("u", "v", "theta", "qv", "qc", "qr", "qi", "qs"),
        reads_carry=CUMULUS_CARRY_MEMBERS[1],
        writes_carry=CUMULUS_CARRY_MEMBERS[1],
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[1]),
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=2,
        name="Betts-Miller-Janjic",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_bmj.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:BMJ",
        reads_state=("theta", "qv", "p", "pb", "ph", "mu", "xland"),
        writes_state=("theta", "qv"),
        reads_carry=CUMULUS_CARRY_MEMBERS[2],
        writes_carry=CUMULUS_CARRY_MEMBERS[2],
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "cutop", "cubot", *CUMULUS_TENDENCY_MEMBERS[2]),
        notes="Frozen-contract extension for WRF cu_physics=2. BMJ is an adjustment scheme and returns only RTHCUTEN/RQVCUTEN plus cumulus precipitation.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=3,
        name="Grell-Freitas",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_grell_freitas.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:Grell-Freitas",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
        reads_carry=CUMULUS_CARRY_MEMBERS[3],
        writes_carry=CUMULUS_CARRY_MEMBERS[3],
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[3]),
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=6,
        name="Tiedtke",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_tiedtke.py",
        oracle="M20 physics-oracle factory savepoint at module_cumulus_driver.F:Tiedtke",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[6]),
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=4,
        name="Scale-aware GFS SAS",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_scalesas.py",
        oracle="pristine WRF CU_SCALESAS module savepoints REAL4 BYTE, proofs/v034/scalesas; "
        "Python API only because original WRF ARW check_a_mundo:665-684 refuses cu_physics=4",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "hbot", "htop", *CUMULUS_TENDENCY_MEMBERS[4]),
        notes="Module-oracle-qualified Python API; CLI remains refused exactly like WRF ARW. "
        "GPU/coupled-forecast qualification pending; ARW does not couple SAS momentum.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=16,
        name="New Tiedtke",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_ntiedtke_jax.py",
        oracle="v0.13 single-column fp64 pristine-WRF savepoints from unmodified "
        "phys/module_cu_ntiedtke.F + physics_mmm/cu_ntiedtke.F90 "
        "(proofs/v013/savepoints/cumulus/ntiedtke_case_*.json); v0.23 F2 faithful "
        "fp64 port at machine precision (RAINCV bit-identical, tendencies "
        "<=1e-15 abs; tests/test_ntiedtke_cumulus_oracle.py + "
        "tests/test_ntiedtke_jax_parity.py)",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[16]),
        notes="v0.23 F2 OPERATIONAL: NumPy reference (cumulus_ntiedtke) + traceable "
        "jit/vmap kernel (cumulus_ntiedtke_jax), scan-wired via "
        "coupling.scan_adapters.ntiedtke_adapter with WRF RQVFTEN (flux-form qv "
        "advection + PBL forcing) and RTHFTEN (accumulated physics theta forcing; "
        "the advective-theta component is a named coupling caveat).",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=94,
        name="2015 GFS SAS / HWRF",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_sas94.py",
        oracle="pristine WRF 4.7.1 CU_SAS savepoints REAL4 and promoted REAL8, deep and ARW shallow; "
        "proofs/v040/sas_family",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "hbot", "htop", *CUMULUS_TENDENCY_MEMBERS[94]),
        notes="CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "WRF first-step/STEPCU held rates; ARW phy_cu_ten omits SAS momentum.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=95,
        name="Previous GFS SAS / HWRF OSAS",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_osas95.py",
        oracle="pristine WRF 4.7.1 CU_OSAS savepoints with recorded original XKT2; REAL4 eight outputs BYTE "
        "455 columns at two cadences, proofs/v040/sas_family",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "hbot", "htop", *CUMULUS_TENDENCY_MEMBERS[95]),
        notes="Experimental; CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "WRF draws from the system clock; gpuwrf is deterministic by design. Restarted counter keys.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=96,
        name="Previous new GFS SAS / YSU NSAS",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_nsas96.py",
        oracle="pristine WRF 4.7.1 REAL4/REAL8 CU driver savepoints, proofs/v040/sas_family; "
        "Owner-accepted disclosed compiler/conditioning deviation; final D6 binding",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qi", "p", "pb", "ph", "mu"),
        writes_state=("u", "v", "theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", "hbot", "htop", *CUMULUS_TENDENCY_MEMBERS[96]),
        notes="CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Native REAL held scalar/momentum rates; WRF cadence, source-specific PBL diagnostics.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=5,
        name="Grell-3D ensemble",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_g3.py",
        oracle="v0.3.4 multi-column pristine-WRF tile savepoints (proofs/v034/savepoints/cumulus_grell) "
        "vs unmodified phys/module_cu_g3.F:G3DRV + conv_grell_spread3d (fp64 + WRF REAL builds)",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[5]),
        notes="v0.3.4 OPERATIONAL (CPU-oracle-qualified; GPU/coupled-forecast qualification "
        "pending): line-faithful JAX port scan-wired via CU_SCAN_ADAPTERS[5].",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=14,
        name="KIM Simplified Arakawa-Schubert",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_ksas14.py",
        oracle="pristine WRF 4.7.1 REAL4/REAL8 CU driver savepoints, proofs/v040/sas_family; "
        "Owner-accepted disclosed compiler/conditioning deviation; final D6 binding",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("u", "v", "theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", *CUMULUS_TENDENCY_MEMBERS[14]),
        notes="CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Native REAL held scalar/momentum rates; WRF cadence, source-specific PBL diagnostics.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=93,
        name="Grell-Devenyi ensemble",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_grell_devenyi.py",
        oracle="v0.3.4 multi-column pristine-WRF tile savepoints (proofs/v034/savepoints/cumulus_grell) "
        "vs unmodified phys/module_cu_gd.F:GRELLDRV (fp64 + WRF REAL builds)",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qi"),
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", *CUMULUS_TENDENCY_MEMBERS[93]),
        notes="v0.3.4 OPERATIONAL (CPU-oracle-qualified; GPU/coupled-forecast qualification "
        "pending): line-faithful JAX port scan-wired via CU_SCAN_ADAPTERS[93]; distinct WRF "
        "source path from cu=3 Grell-Freitas.",
    ),
    PhysicsStepSpec(
        family="cumulus",
        option=99,
        name="previous Kain-Fritsch",
        wrf_slot="first_rk_cumulus_driver",
        owner_module="src/gpuwrf/physics/cumulus_kf99.py",
        oracle="pristine WRF 4.7.1 KFCPS/KFPARA savepoints native REAL4 and promoted REAL8; "
        "ten-field promoted gate passes, compiled conditioning disclosed vs original FMA; proofs/v040/sas_family",
        reads_state=("u", "v", "w", "theta", "qv", "qc", "qr", "qi", "qs", "p", "pb", "ph", "mu"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs"),
        reads_carry=CUMULUS_CARRY_MEMBERS[99],
        writes_carry=CUMULUS_CARRY_MEMBERS[99],
        returns_accumulators=("rainc_acc",),
        diagnostics=("raincv", "pratec", *CUMULUS_TENDENCY_MEMBERS[99]),
        notes="Experimental distinct old KF; CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Source W0AVG/NCA, seven held rates and old clear threshold; final D6 binding.",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=1,
        name="thermal-diffusion slab LSM",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/lsm_slab.py",
        oracle="v0.13 Tier-3 fp64 savepoint parity vs unmodified WRF module_sf_slab.F "
        "(proofs/v013/t3_surface_lsm_oracle.py/.json)",
        reads_state=("t_skin", "mavail", "xland"),
        writes_state=("t_skin",),
        reads_carry=LAND_CARRY_MEMBERS[1],
        writes_carry=LAND_CARRY_MEMBERS[1],
        diagnostics=("TSK", "HFX", "QFX", "LH", "QSFC", "CAPG"),
        notes="5-layer Blackadar thermal-diffusion slab (SLAB1D). JAX-ported + fp64 "
        "oracle-validated (physics.lsm_slab.slab_columns) and v0.17 operationally "
        "scan-wired via coupling.slab_surface_hook.slab_surface_step: advances the "
        "5-layer TSLB soil-temperature land carry from GSW/GLW radiation forcing + an "
        "explicit TMN/THC/EMISS SlabStaticBundle (FLHC/FLQC recovered from the resident "
        "surface-layer kinematic flux handles).",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=2,
        name="Noah classic",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/noah_classic.py",
        oracle="M20 physics-oracle factory savepoint at module_surface_driver.F:Noah LSM",
        reads_state=("t_skin", "soil_moisture", "xland", "mavail", "roughness_m", "lu_index"),
        writes_state=("t_skin", "soil_moisture", "mavail"),
        reads_carry=LAND_CARRY_MEMBERS[2],
        writes_carry=LAND_CARRY_MEMBERS[2],
        diagnostics=("TSK", "GRDFLX", "QFX", "ALBEDO", "EMISS"),
        notes=f"Noah classic initializes a {NOAH_CLASSIC_NUM_SOIL_LAYERS}-layer land carry; "
        "do not reinterpret State.soil_moisture as a 4-layer field.",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=3,
        name="RUC LSM",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/ruclsm.py",
        oracle="v0.3.4 fp64+fp32 per-step savepoints vs unmodified WRF module_sf_ruclsm.F "
        "(LSMRUC driver, 31 regimes x 12 steps incl. snow/melt/frozen soil/mosaic; "
        "proofs/v034/oracle/ruclsm + savepoints/ruclsm/{fp64,fp32}/ruclsm_v2_*.json)",
        reads_state=("t_skin", "soil_moisture", "xland", "mavail", "roughness_m", "lu_index"),
        writes_state=("t_skin", "soil_moisture", "mavail"),
        reads_carry=LAND_CARRY_MEMBERS[3],
        writes_carry=LAND_CARRY_MEMBERS[3],
        diagnostics=("TSK", "HFX", "QFX", "LH", "GRDFLX", "SFCRUNOFF", "UDRUNOFF", "SNOW", "SNOWH"),
        notes="RUC multi-layer soil/snow LSM (sf_surface_physics=3). STATUS: implemented, "
        "CPU-oracle-qualified (faithful column-vectorised JAX port of LSMRUC/SOILVEGIN/SFCTMP/"
        "SOIL/SNOWSOIL/SOILTEMP/SNOWTEMP/SOILMOIST/SOILPROP/TRANSF/VILKA, fp64 <=1.3e-10 rel "
        "vs the pristine oracle); scan-wired via coupling.ruc_surface_hook. GPU/coupled-"
        "forecast qualification pending; sea-ice/lake points fail closed.",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=4,
        name="Noah-MP",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/noah_mp.py",
        oracle="existing Noah-MP WRF savepoint parity gate; rerun before mixed-suite integration",
        reads_state=("t_skin", "soil_moisture", "xland", "mavail", "roughness_m", "lu_index"),
        writes_state=("t_skin", "soil_moisture", "mavail"),
        reads_carry=LAND_CARRY_MEMBERS[4],
        writes_carry=LAND_CARRY_MEMBERS[4],
        diagnostics=("TSK", "GRDFLX", "QFX", "ALBEDO", "EMISS"),
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=7,
        name="Pleim-Xiu LSM",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/lsm_pleim_xiu.py",
        oracle="v0.17 fp64 savepoint parity vs unmodified WRF module_sf_pxlsm.F "
        "(SURFPX+QFLUX; proofs/v017/oracle/pxlsm + pxlsm_savepoint_parity_report.json, "
        "worst |jax-oracle| 4.4e-11 over 5 regimes)",
        reads_state=("t_skin", "soil_moisture", "xland", "ustar", "roughness_m", "theta_flux", "qv_flux"),
        writes_state=("t_skin",),
        reads_carry=LAND_CARRY_MEMBERS[7],
        writes_carry=LAND_CARRY_MEMBERS[7],
        diagnostics=("TSK", "HFX", "QFX", "LH", "GRDFLX", "CAPG", "T2", "Q2"),
        notes="Pleim-Xiu 2-layer ISBA LSM (SURFPX+QFLUX, NUDGEX=0). fp64-oracle-validated "
        "(physics.lsm_pleim_xiu.pxlsm_columns) and v0.17 operationally scan-wired via "
        "coupling.pleim_xiu_surface_hook.pleim_xiu_surface_step: advances the 2-layer ISBA "
        "carry (TG/T2/WG/W2/WR) from GSW/GLW radiation + an explicit PleimXiuStaticBundle "
        "(ISBA soil constants + vegetation fields); pairs with the PX surface layer "
        "(sf_sfclay_physics=7). RMOL recovered from the resident surface-layer flux handles.",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=8,
        name="SSiB LSM",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/lsm_ssib.py",
        oracle="proofs/s2_ssib: original WRF SSIB surface-caller savepoints, 400 REAL4 literal BYTE calls; "
        "compiled residuals disclosed, GPU/coupled-forecast qualification pending",
        reads_state=("u", "v", "theta", "qv", "p", "ph", "t_skin", "xland", "mavail", "roughness_m"),
        writes_state=("t_skin", "soil_moisture", "ustar", "hfx", "qfx", "theta_flux", "qv_flux", "fltv", "tau_u", "tau_v"),
        reads_carry=LAND_CARRY_MEMBERS[8],
        writes_carry=LAND_CARRY_MEMBERS[8],
        diagnostics=("TSK", "HFX", "QFX", "LH", "GRDFLX", "ALBEDO", "T2", "Q2"),
        notes="Experimental opt-in SSiB, three soil/four snow layers. Full INOUT carry and "
        "Registry history, pristine embedded DATA tables, revised-MM5/YSU, SW1/LW4. "
        "Direct State caller also updates roughness/WSPD (outside PhysicsTendency keys). "
        "Original UPDAT1 undefined read uses compiler-defined zero. See proofs/s2_ssib.",
    ),
    # Radiation (RRTMG, ra_lw/ra_sw option 4). Unlike the other families, the
    # radiation drivers write a HELD-RATE potential-temperature tendency
    # (WRF RTHRATEN) applied at every dynamics step over the radt interval; the
    # column adapter (coupling/physics_couplers.rrtmg_held_rate) emits a `theta`
    # state_tendency rather than an in-place replacement. SW and LW are distinct
    # namelist options but share one RRTMG column adapter; LW reads cloud
    # hydrometeors + qv + the LSM surface emissivity/skin temperature, SW adds
    # the solar-geometry/coszen inputs and surface albedo. Surface fluxes
    # (GLW/SWDOWN/GSW) are returned as radiation diagnostics consumed by the LSM.
    PhysicsStepSpec(
        family="radiation",
        option=4,
        name="RRTMG longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/rrtmg_lw.py",
        oracle="M20 physics-oracle factory savepoint at module_radiation_driver.F:RRTMG LW",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "LWUPB", "RTHRATENLW"),
        variant="lw",
        notes="ra_lw_physics=4. Held-rate RTHRATEN theta tendency; SW pairs with LW under "
        "RRTMG. cu_rad_feedback couples active cumulus cloud fraction into the column when on.",
    ),
    PhysicsStepSpec(
        family="radiation",
        option=4,
        name="RRTMG shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/rrtmg_sw.py",
        oracle="M20 physics-oracle factory savepoint at module_radiation_driver.F:RRTMG SW",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "SWUPB", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=4. Shares the RRTMG column adapter with LW; needs solar geometry "
        "(coszen/declination/equation-of-time) and surface albedo. Held-rate RTHRATEN.",
    ),
    # aer_opt=1 (v0.4.1 A3): the WRF Tegen climatology variant of RRTMG SW (same held-rate adapter).
    PhysicsStepSpec(
        family="radiation",
        option=4,
        name="RRTMG shortwave with Tegen aerosol climatology (aer_opt=1)",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/wrf_tegen_aerosol.py",
        oracle="Pristine WRF 4.7.1 savepoints: aerosol_in/aer_time_int/aer_p_int + RRTMG_SWRAD aer_opt=1 on 186 real WN3 0227 "
        "columns x 5 dates (proofs/w41_aerosol/ae01_*); unit chain bitwise",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        reads_carry=("aerod",),
        writes_carry=("aerod",),
        diagnostics=("SWDOWN", "GSW", "SWUPB", "SWDDIR", "COSZEN", "RTHRATENSW"),
        variant="sw_aer_opt1",
        notes="aer_opt=1 with ra_sw_physics=4: domain-1 aerosol_in at init, aer_time_int + aer_p_int on p_hyd/p_hyd_w "
        "at each radiation call, iaer=6 band optics; nests radiate with the parent's SINT-forced aerod (carry, "
        "nested runs only). CPU-oracle-qualified; GPU 24 h WN3 0227 3-nest forecast D6 PASS vs original CPU-WRF.",
    ),
    # aer_opt=3 (v0.4.1 A3): Thompson mp28 water/ice-friendly aerosol optics in RRTMG SW (same held-rate adapter).
    PhysicsStepSpec(
        family="radiation",
        option=4,
        name="RRTMG shortwave with Thompson water/ice-friendly aerosol optics (aer_opt=3)",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/wrf_aer_opt3.py",
        oracle="Pristine WRF 4.7.1 savepoints: gt_aod + calc_aerosol_rrtmg_sw + RRTMG_SWRAD aer_opt=3 on 186 real WN3 0227 "
        "columns with WRF-default QNWFA/QNIFA profiles (proofs/w41_aerosol/ae07_*)",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "nwfa", "nifa", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        reads_carry=("taod5502d",),
        writes_carry=("taod5502d",),
        diagnostics=("SWDOWN", "GSW", "SWUPB", "SWDDIR", "COSZEN", "RTHRATENSW", "TAOD5502D"),
        variant="sw_aer_opt3",
        notes="aer_opt=3 with mp_physics=28 and ra_sw_physics=4 (single domains, wif_input_opt=1): gt_aod on "
        "p_hyd/t_phy/QV/dz8w/QNWFA/QNIFA at each radiation call, rural-type RH spectral optics, iaer=10; held "
        "TAOD5502D. CPU-oracle-qualified; GPU 24 h Swiss mp28 forecast D6 PASS vs original CPU-WRF.",
    ),
    # Classic Dudhia shortwave (ra_sw_physics=1) -- Stephens-1984 broadband
    # SW. Self-contained column kernel (no external table file); reads cloud
    # hydrometeors + qv + solar geometry (coszen) + surface albedo, emits a
    # held-rate RTHRATEN theta tendency and the surface net SW flux GSW.
    PhysicsStepSpec(
        family="radiation",
        option=1,
        name="Dudhia shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_sw_dudhia.py",
        oracle="v0.6.0 physics-oracle factory savepoint at module_ra_sw.F:SWRAD (Dudhia SW)",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=1. Stephens-1984 broadband Dudhia SW; held-rate RTHRATEN theta "
        "tendency. No external lookup-table asset. Solar geometry via coszen + surface albedo. "
        "STATUS: isolated-WRF-savepoint parity-proven AND operational-scan-wired: "
        "OperationalNamelist.ra_sw_physics=1 routes the radiation slot in runtime.operational_mode "
        "to coupling.physics_couplers.dudhia_sw_theta_tendency (Dudhia SW) + rrtmg_lw_theta_tendency "
        "(RRTMG LW), composed as the held-rate RTHRATEN (WRF runs SW/LW drivers independently). "
        "Wired-coupler oracle proof: proofs/radiation/cdudhia_sw_oracle.py. The surface SWDOWN/flux "
        "history diagnostics remain RRTMG-derived (rrtmg_radiation_diagnostics).",
    ),
    # GSFC/Chou-Suarez shortwave (ra_sw_physics=2) -- v0.13 Tier-3 faithful port of
    # phys/module_ra_gsfcsw.F (Chou-Suarez 8-band SW with k-distribution + the 5x75
    # ozone climatology). Self-contained column kernel; held-rate RTHRATEN theta
    # tendency + surface net SW flux GSW.
    PhysicsStepSpec(
        family="radiation",
        option=2,
        name="GSFC/Chou-Suarez shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_sw_gsfc.py",
        oracle="pristine-WRF fp64 savepoint at module_ra_gsfcsw.F (proofs/v013/t3_radiation_oracle.py, PASS ~1e-12, 7 regimes, NOT self-compare)",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=2. GSFC/Chou-Suarez 8-band SW; held-rate RTHRATEN theta tendency. "
        "STATUS: pristine-WRF-savepoint parity-proven (fp64 ~1e-12) AND operational-scan-wired: "
        "OperationalNamelist.ra_sw_physics=2 routes the radiation slot in runtime.operational_mode "
        "to coupling.physics_couplers.gsfc_sw_theta_tendency (GSFC SW) + rrtmg_lw_theta_tendency "
        "(RRTMG LW), composed as the held-rate RTHRATEN. Surface SWDOWN/flux history diagnostics "
        "remain RRTMG-derived. v0.13 Tier-3 (kernel/coupler/oracle validated; full multi-step "
        "forecast gate is a follow-on).",
    ),
    # New Goddard shortwave: paired option5/5, distinct from standalone GSFC2.
    # CPU-qualified against the pristine public REAL4 wrapper, with original
    # cloud LUT/climatology and explicit DOUBLE core. Experimental admission
    # and upstream undefined-read disposition are stated in the notes below.
    PhysicsStepSpec(
        family="radiation",
        option=5,
        name="Goddard shortwave (new)",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_goddard_sw.py",
        oracle="v0.18 exact-driver real-WRF savepoint at module_radiation_driver.F -> "
        "module_ra_goddard.F:goddardrad (proofs/v018/savepoints/ra_tail_wrf/ra5_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_sw_physics=5 paired with ra_lw_physics=5; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="Experimental Goddard5/5 MP8/use_mp_re=1, one held paired refresh via coupling.goddard_radiation. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Public REAL4 wrapper, original DOUBLE solver; "
        "upstream uninitialized taux uses approved zero-init reference; material sensitivity disclosed in docs/UPSTREAM_WRF_BUGS.md.",
    ),
    # Classic RRTM longwave (ra_lw_physics=1) -- 16-band k-distribution from
    # AER, loaded from the RRTM_DATA asset. Reads cloud hydrometeors + qv +
    # interface T/p (t8w/p8w) + LSM surface emissivity/skin temperature; emits
    # a held-rate RTHRATEN theta tendency, surface downwelling GLW, and TOA OLR.
    PhysicsStepSpec(
        family="radiation",
        option=1,
        name="RRTM longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_lw_rrtm.py",
        oracle="v0.6.0 physics-oracle factory savepoint at module_ra_rrtm.F:RRTMLWRAD (classic RRTM LW)",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "RTHRATENLW"),
        variant="lw",
        notes="ra_lw_physics=1. AER classic RRTM 16-band LW; held-rate RTHRATEN. Loads the "
        "RRTM_DATA k-distribution asset and passes the proofs/v060/run_rrtm_lw_parity.py "
        "fp64 savepoint gate. STATUS: isolated-WRF-savepoint parity-proven AND "
        "operational-scan-wired: the host-NumPy reference kernel (physics.ra_lw_rrtm) was "
        "re-expressed as a JIT/vmap-traceable JAX column kernel (physics.ra_lw_rrtm_jax, "
        "reproducing the reference to fp64 round-off), and OperationalNamelist.ra_lw_physics=1 "
        "routes the radiation slot in runtime.operational_mode to "
        "coupling.physics_couplers.rrtm_lw_theta_tendency (classic RRTM LW) summed with the "
        "selected SW tendency (WRF runs SW/LW drivers independently). Wired-coupler oracle proof: "
        "proofs/radiation/rrtm_lw_oracle.py (pristine-WRF, NOT a self-compare). The surface GLW "
        "history diagnostic remains RRTMG-derived.",
    ),
    # GSFC/Goddard NUWRF longwave: original ten-band DOUBLE lwrad core behind
    # its REAL4 public wrapper, refreshed together with shortwave at cadence.
    PhysicsStepSpec(
        family="radiation",
        option=5,
        name="GSFC/Goddard NUWRF longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_goddard_lw.py",
        oracle="v0.18 exact-driver real-WRF savepoint at module_radiation_driver.F -> "
        "module_ra_goddard.F:goddardrad (proofs/v018/savepoints/ra_tail_wrf/ra5_wrf_real.json, "
        "generated by a physics-pristine, WRFGPU2_ORACLE-instrumented wrf.exe run with "
        "ra_lw_physics=5 paired with ra_sw_physics=5; NOT a self-compare). Prior v0.13 "
        "single-column lwrad oracle remains useful background evidence.",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "RTHRATENLW"),
        variant="lw",
        notes="Experimental Goddard5/5 MP8/use_mp_re=1, one held paired refresh via coupling.goddard_radiation. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Public REAL4 wrapper, original DOUBLE solver; "
        "upstream uninitialized taux uses approved zero-init reference; material sensitivity disclosed in docs/UPSTREAM_WRF_BUGS.md.",
    ),
    # GFDL-Eta: approved experimental CPU-only paired operational path.
    PhysicsStepSpec(
        family="radiation",option=99,name="GFDL-Eta LW",variant="lw",
        wrf_slot="first_rk_radiation_driver",owner_module="src/gpuwrf/physics/ra_gfdleta.py",
        oracle="pristine WRF REAL4 full-call savepoint gates proofs/s2_flggfdl/receipts; "
        "literal BYTE and disclosed CPU compiler spread; independent native routine reference",
        reads_state=("theta","qv","qc","qr","qi","qs","qg","p","pb","ph","phb","t_skin"),
        writes_state=("theta",),reads_carry=("radiation_diagnostics","gfdl_radiation"),
        writes_carry=("radiation_diagnostics","gfdl_radiation"),
        diagnostics=("GLW","SWDOWN","GSW","RTHRATENLW","RTHRATENSW"),
        notes="STATUS: IMPLEMENTED; EXPERIMENTAL CPU-ONLY paired 99/99, MP8/cu0/Noah-MP4. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "native GPU column kernel pending; GPU refused pre-JAX. "
        "Original WRF REAL4 carry, native DOUBLE islands only; one held-rate refresh; E78 restart.",
    ),
    PhysicsStepSpec(
        family="radiation",option=99,name="GFDL-Eta SW",variant="sw",
        wrf_slot="first_rk_radiation_driver",owner_module="src/gpuwrf/physics/ra_gfdleta.py",
        oracle="pristine WRF REAL4 full-call savepoint gates proofs/s2_flggfdl/receipts; "
        "literal BYTE and disclosed CPU compiler spread; independent native routine reference",
        reads_state=("theta","qv","qc","qr","qi","qs","qg","p","pb","ph","phb","t_skin"),
        writes_state=("theta",),reads_carry=("radiation_diagnostics","gfdl_radiation"),
        writes_carry=("radiation_diagnostics","gfdl_radiation"),
        diagnostics=("GLW","SWDOWN","GSW","RTHRATENLW","RTHRATENSW"),
        notes="STATUS: IMPLEMENTED; EXPERIMENTAL CPU-ONLY paired 99/99, MP8/cu0/Noah-MP4. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "native GPU column kernel pending; GPU refused pre-JAX. "
        "Original WRF REAL4 carry, native DOUBLE islands only; one held-rate refresh; E78 restart.",
    ),
    # CAM radiation pair (ra_lw_physics=3 / ra_sw_physics=3) -- v0.3.4 (lane o1-camrad) faithful
    # float64 JAX port of WRF's CAMRAD (phys/module_ra_cam.F + module_ra_cam_support.F, NCAR CAM 3.0
    # radiation): physics.ra_cam{,_common,_lw,_lw_abs,_sw}, scan-wired via coupling.cam_radiation
    # (held-rate RTHRATEN; CAM GLW / SWDOWN feed the land surface). WRF calls camrad separately for LW
    # (dolw) and SW (dosw), so LW and SW are selectable independently.
    PhysicsStepSpec(
        family="radiation",
        option=3,
        name="CAM longwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_cam.py",
        oracle="pristine WRF camrad true-caller column savepoint oracle CAM01 (proofs/cam_rad: unmodified V4.7.1 camradinit+camrad "
        "from libwrflib.a with the radiation_driver wiring, 210 real WN3 0227 + augmented columns, "
        "LW / held-absorptivity arms; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb", "t_skin"),
        writes_state=("theta",),
        diagnostics=("GLW", "OLR", "RTHRATENLW"),
        variant="lw",
        notes="ra_lw_physics=3. CAM 3.0 LW half of CAMRAD; STATUS: IMPLEMENTED (operational-scan-wired, "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending). Absorptivities recomputed every "
        "radiation call (WRF with cam_abs_freq_s <= radt).",
    ),
    PhysicsStepSpec(
        family="radiation",
        option=3,
        name="CAM shortwave",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_cam.py",
        oracle="pristine WRF camrad true-caller column savepoint oracle CAM01 (proofs/cam_rad: unmodified V4.7.1 camradinit+camrad "
        "from libwrflib.a with the radiation_driver wiring, 210 real WN3 0227 + augmented columns, "
        "SW arm day+night; NOT a self-compare).",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "p", "pb", "ph", "phb"),
        writes_state=("theta",),
        diagnostics=("SWDOWN", "GSW", "COSZEN", "RTHRATENSW"),
        variant="sw",
        notes="ra_sw_physics=3. CAM 3.0 SW half of CAMRAD; STATUS: IMPLEMENTED (operational-scan-wired, "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending). Uniform CAM aerosol climatology "
        "(aerosol_init) as in WRF.",
    ),
    # FLG: approved experimental CPU-only paired operational path.
    PhysicsStepSpec(
        family="radiation",option=7,name="FLG LW",variant="lw",
        wrf_slot="first_rk_radiation_driver",owner_module="src/gpuwrf/physics/ra_flg.py",
        oracle="pristine WRF REAL4 full-call savepoint gates proofs/s2_flggfdl/receipts; "
        "literal BYTE and disclosed CPU compiler spread; independent native routine reference",
        reads_state=("theta","qv","qc","qr","qi","qs","qg","p","pb","ph","phb","t_skin"),
        writes_state=("theta",),reads_carry=("radiation_diagnostics","gfdl_radiation"),
        writes_carry=("radiation_diagnostics","gfdl_radiation"),
        diagnostics=("GLW","SWDOWN","GSW","RTHRATENLW","RTHRATENSW"),
        notes="STATUS: IMPLEMENTED; EXPERIMENTAL CPU-ONLY paired 7/7, MP8/cu0/Noah-MP4. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "native GPU column kernel pending; GPU refused pre-JAX. "
        "Original WRF REAL4 carry, native DOUBLE islands only; one held-rate refresh; E78 restart.",
    ),
    PhysicsStepSpec(
        family="radiation",option=7,name="FLG SW",variant="sw",
        wrf_slot="first_rk_radiation_driver",owner_module="src/gpuwrf/physics/ra_flg.py",
        oracle="pristine WRF REAL4 full-call savepoint gates proofs/s2_flggfdl/receipts; "
        "literal BYTE and disclosed CPU compiler spread; independent native routine reference",
        reads_state=("theta","qv","qc","qr","qi","qs","qg","p","pb","ph","phb","t_skin"),
        writes_state=("theta",),reads_carry=("radiation_diagnostics","gfdl_radiation"),
        writes_carry=("radiation_diagnostics","gfdl_radiation"),
        diagnostics=("GLW","SWDOWN","GSW","RTHRATENLW","RTHRATENSW"),
        notes="STATUS: IMPLEMENTED; EXPERIMENTAL CPU-ONLY paired 7/7, MP8/cu0/Noah-MP4. "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "native GPU column kernel pending; GPU refused pre-JAX. "
        "Original WRF REAL4 carry, native DOUBLE islands only; one held-rate refresh; E78 restart.",
    ),
    # Held-Suarez idealized radiation (ra_lw_physics=31) -- v0.17 GPU-op endpoint.
    # A COMBINED LW+SW Newtonian relaxation toward an analytic radiative-equilibrium
    # temperature (phys/module_ra_hs.F:HSRAD; Held & Suarez 1994). Selected through
    # the LW slot; WRF makes NO separate shortwave call, so the operational dispatch
    # requires ra_sw_physics=0. Carries NO prognostic state and reads only T, p,
    # surface interface pressure, and latitude -> a true no-kernel-change port.
    PhysicsStepSpec(
        family="radiation",
        option=31,
        name="Held-Suarez idealized radiation",
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/ra_lw_hs.py",
        oracle="pristine-WRF fp64 single-column savepoint at module_ra_hs.F:HSRAD "
        "(proofs/v017/oracle/hs_build_and_run.sh + proofs/v017/held_suarez_lw_savepoint_parity.json; "
        "NOT a self-compare; fp64 worst rel 1.1e-14 across 7 latitude regimes).",
        reads_state=("theta", "p", "pb"),
        writes_state=("theta",),
        diagnostics=("RTHRATEN",),
        variant="lw",
        notes="ra_lw_physics=31. Held-Suarez idealized COMBINED LW+SW Newtonian relaxation "
        "(held-rate RTHRATEN theta tendency). STATUS: IMPLEMENTED + scan-wired "
        "(coupling.physics_couplers.held_suarez_theta_tendency, stateless State->RTHRATEN). "
        "HSRAD is the SOLE radiative call WRF makes for this scheme, so the operational scan "
        "REQUIRES ra_sw_physics=0 (a real SW selection fail-closes to avoid double-counting). "
        "No-kernel-change endpoint: no prognostic state, no solar geometry/moisture/cloud/ozone; "
        "savepoint-parity-proven against the unmodified WRF source at fp64.",
    ),
    _mp_spec(
        19,
        "NSSL one-moment with hail and predicted graupel density",
        "src/gpuwrf/physics/nssl2mom/adapter.py",
        "Original WRF4.7.1 legacy19 ipconc0 caller and REAL4/REAL8 active-column savepoints "
        "proofs/s2_mpmore/nssl1mom/fixtures/manifest.json; independently measured "
        "same-input original REAL4 FMA/noFMA spread at case16 QC(k1). "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending; single domains.",
        extra_state=("qvolg",),
        accumulators=("rain_acc","snow_acc","graupel_acc","ice_acc","hail_acc"),
    ),
    _mp_spec(
        21,
        "NSSL one-moment without hail",
        "src/gpuwrf/physics/nssl2mom/adapter.py",
        "Original WRF4.7.1 legacy21 ipconc0 caller and REAL4/REAL8 active-column savepoints "
        "proofs/s2_mpmore/nssl1mom/fixtures/manifest.json; independently measured "
        "same-input original REAL4 FMA/noFMA spread at case16 QC(k1). "
        "CPU-oracle-qualified; GPU/coupled-forecast qualification pending; single domains.",
    ),
    PhysicsStepSpec(
        family="land_surface",
        option=4,
        variant="opt_run5",
        name="Noah-MP Miguez-Macho–Fan groundwater",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/noahmp/groundwater.py",
        oracle="pristine WRF 4.7.1 REAL4 savepoint gates: GROUNDWATER_INIT/EQSMOISTURE, "
               "SOILWATER/SRT/SSTEP/SHALLOWWATERTABLE and WTABLE_mmf_noahmp; "
               "real WPS/real Swiss MMF geography (proofs/s2_mmf)",
        reads_state=(),
        writes_state=("soil_moisture",),
        reads_carry=("SMOIS", "SH2O", "SMCWTD", "ZWT", "SMOISEQ", "DEEPRECH", "RECH",
                     "FDEPTH", "AREA", "RIVERBED", "EQZWT", "RIVERCOND", "PEXP", "STEPWTD"),
        writes_carry=("SMOIS", "SH2O", "SMCWTD", "ZWT", "DEEPRECH", "RECH", "QLAT", "QRF",
                      "QSPRING", "QSLAT", "QRFS", "QSPRINGS"),
        diagnostics=("SMOISEQ", "SMCWTD", "ZWT", "DEEPRECH", "RECH", "QLAT", "QRF", "QSPRING",
                     "QSLAT", "QRFS", "QSPRINGS", "FDEPTH", "AREA", "RIVERBED", "EQZWT",
                     "RIVERCOND", "PEXP", "STEPWTD", "RECHCLIM", "ACRECH", "ACQSPRING"),
        notes="Opt-in opt_run=5 extension. WRF REAL4 throughout; no DOUBLE island. "
              "Column recharge deltas accumulate before the independent STEPWTD lateral-flow call. "
              "CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
              "Four complete restart formats; ACRECH/ACQSPRING invariant zero with NOAHMP_ACC_DT=0. "
              "Actual GPU performance/memory/E41 unmeasured.",
    ),
)

# Preserve the main contract order; append the opt-in MY2 interface.
SCHEME_STEP_SPECS += (
    _mp_spec(
        9, "Milbrandt-Yau two-moment", "src/gpuwrf/physics/milbrandt2mom/adapter.py",
        "Pristine WRF REAL4 savepoints: literal 3168-value BYTE, actual wrapper and "
        "independent input-ULP envelopes in proofs/s2_milbrandt; CPU-oracle-qualified, "
        "GPU/coupled-forecast qualification pending. Native single-domain only.",
        diagnostics=("REFL_10CM",),
        accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc", "hail_acc"),
    ),
)

# P3 keeps old theta/vapor in State, separately from transported particles.

SCHEME_STEP_SPECS += tuple(
    PhysicsStepSpec(
        family="microphysics", option=mp, name=name,
        wrf_slot="solve_em_microphysics_driver",
        owner_module="src/gpuwrf/physics/p3/adapter.py",
        oracle="Pristine WRF module_mp_p3.F wrapper REAL4 savepoints for mp50/51/52/53 "
        "(proofs/s2_p3/columns_manifest.json): literal no-FMA BYTE; compiled WRF "
        "contraction/cancellation bounds in proofs/s2_p3/COMPILED_BOUND.md.",
        reads_state=("theta", "p", "ph", "Nc", "rain_acc", "snow_acc",
                     *(leaf for leaf in state_leaves_for_mp(mp)
                       if leaf not in ("Nc", "re_cloud", "re_ice"))),
        writes_state=("theta", *state_leaves_for_mp(mp), *ice_diagnostic_leaves(mp)),
        returns_accumulators=("rain_acc", "snow_acc"),
        diagnostics=("re_cloud", "re_ice", *ice_diagnostic_leaves(mp)),
        notes="CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Experimental root domains only. REAL4 particle/history/radius carry; "
        "old theta/vapor are held physics history, never transported. Native "
        "outputs pruned to State replacements and precipitation; no per-step host work.",
    )
    for mp, name in ((50, "P3 one-category"), (51, "P3 one-category cloud number"),
                     (52, "P3 two-category"), (53, "P3 three-moment ice"))
)
SCHEME_STEP_SPECS += (
    PhysicsStepSpec(
        family="microphysics", option=55, name="Jensen-ISHMAEL three ice habits",
        wrf_slot="solve_em_microphysics_driver",
        owner_module="src/gpuwrf/physics/jensen_ishmael/adapter.py",
        oracle="Pristine WRF module_mp_jensen_ishmael.F savepoints: 34 REAL4/REAL8 public "
        "columns and evolved two-step columns; literal no-FMA BYTE. Independent "
        "original input-ULP/FMA controls in proofs/s2_mpmore/jensen55/column_controls.",
        reads_state=("theta", "p", "ph", "rain_acc", "snow_acc",
                     *(leaf for leaf in state_leaves_for_mp(55)
                       if leaf not in ("re_cloud", "re_ice"))),
        writes_state=("theta", *state_leaves_for_mp(55), *ice_diagnostic_leaves(55)),
        returns_accumulators=("rain_acc", "snow_acc"),
        diagnostics=("re_cloud", "re_ice", *ice_diagnostic_leaves(55)),
        notes="CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
        "Experimental single domains; three ice masses/numbers and six habit "
        "volume/aspect moments transported and restarted. One native program per "
        "column, outputs pruned to State replacements/precipitation. Compiled "
        "rounding excesses are disclosed deviations; D6 decides. WRF gammln is "
        "the sole DOUBLE island. Vm/DI/rho/phi retained at the WRF call time level; type/radar diagnostics omitted.",
    ),
)
# Additive v0.4.0 Noah-MP control variants. The released land_surface=4
# entry above remains the unqualified-variant lookup used by existing callers.
_NOAHMP_CONTROL_ROWS = (
    ("dveg", (1,2,3,5,6,7,8,9), "phenology.py", "phenology_oracle.json + phenology_dynamic_oracle.json + carbon_oracle.json", ("LAI","XSAI","FVEG","GPP","NPP","NEE")),
    ("opt_crs", (2,), "canopy_options.py", "canopy_oracle.json", ("RSSUN","RSSHA","PSN")),
    ("opt_btr", (2,3), "surface_options.py", "surface_oracle.json", ("BTRAN",)),
    ("opt_run", (1,2,4,6,7,8), "water_hydro.py", "groundwater_oracle.json + equilibrium_oracle.json + water_oracle.json + advanced_water_oracle.json", ("RUNSF","RUNSB","ZWT","WA","WT","QIN")),
    ("opt_sfc", (2,), "canopy_options.py", "canopy_oracle.json", ("HFX","QFX","T2","Q2")),
    ("opt_frz", (2,), "soil_thermo.py", "thermal_oracle.json", ("SH2O","SMOIS","SNOW")),
    ("opt_inf", (2,), "water_hydro.py", "water_oracle.json + equilibrium_oracle.json", ("SH2O","SMOIS","RUNSF","RUNSB")),
    ("opt_rad", (1,2), "energy_radiation.py", "gap_oracle.json", ("SAV","SAG","FSA","ALBEDO","BGAP","WGAP")),
    ("opt_alb", (1,), "energy_radiation.py", "snow_oracle.json", ("ALBEDO","TAUSS","ALBOLD")),
    ("opt_snf", (2,3,4,5), "precip_heat.py", "snow_oracle.json", ("RAINLSM","SNOWLSM","FPICE")),
    ("opt_tbot", (1,), "soil_thermo.py", "thermal_oracle.json", ("TSLB","TSNO")),
    ("opt_stc", (2,3), "energy.py", "thermal_oracle.json + surface_caps_oracle.json", ("TSLB","TSNO","TG","TGV","TGB")),
    ("opt_rsf", (2,3,4), "surface_options.py", "surface_oracle.json", ("QFX","LH","HFX")),
    ("opt_crop", (1,), "crop.py", "crop_oracle.json + crop_phenology_oracle.json", ("LAI","XSAI","GRAIN","GDD","PGS","GPP","NPP","NEE")),
    ("opt_soil", (2,3), "soil_options.py", "soil_oracle.json", ("TSLB","SH2O","SMOIS","HFX","QFX")),
    ("opt_pedo", (1,), "soil_options.py", "soil_oracle.json", ("TSLB","SH2O","SMOIS","HFX","QFX")),
    ("opt_infdv", (1,2,3), "runoff_options.py", "runoff_oracle.json + advanced_water_oracle.json", ("RUNSF","RUNSB","SH2O","SMOIS")),
    ("opt_tdrn", (1,2), "tile_drain.py", "advanced_water_oracle.json", ("QTDRAIN","ZWT","SH2O","SMOIS")),
    ("opt_irr", (1,2,3), "irrigation.py", "irrigation_oracle.json", ("IRNUMSI","IRNUMMI","IRNUMFI","IRSIVOL","IRMIVOL","IRFIVOL","IRELOSS","IRRSPLH")),
    ("opt_irrm", (0,1,2,3), "irrigation.py", "irrigation_oracle.json", ("IRSIVOL","IRMIVOL","IRFIVOL","IRELOSS","IRRSPLH")),
)
_NOAHMP_SURFACE_WRITES = ("ustar","theta_flux","qv_flux","tau_u","tau_v","rhosfc","fltv","t_skin")
_NOAHMP_CONTROL_STEP_SPECS = tuple(
    PhysicsStepSpec(
        family="land_surface",option=4,variant=f"noahmp.{control}={value}",
        name=f"Noah-MP {control}={value}",wrf_slot="first_rk_surface_driver",
        owner_module=f"src/gpuwrf/physics/noahmp/{module}",
        oracle=f"pristine WRF REAL4 savepoint oracle: proofs/u2_land/{oracle}",
        reads_state=("u","v","theta","qv","p_total","ph_total","t_skin","xland","ustar","hfx","qfx"),
        writes_state=_NOAHMP_SURFACE_WRITES,
        reads_carry=("noahmp_land","noahmp_rad") + (
            ("noahmp_carbon",) if (control=="dveg" and value in (2,5,6)) or control=="opt_crop" else
            ("noahmp_irrigation",) if control in ("opt_irr","opt_irrm") else
            ("noahmp_groundwater",) if control=="opt_run" and value==1 else
            ('noahmp_tile_drain',) if control=='opt_tdrn' else ()),
        writes_carry=("noahmp_land",) + (
            ("noahmp_carbon",) if (control=="dveg" and value in (2,5,6)) or control=="opt_crop" else
            ("noahmp_irrigation",) if control in ("opt_irr","opt_irrm") else
            ("noahmp_groundwater",) if control=="opt_run" and value==1 else
            ('noahmp_tile_drain',) if control=='opt_tdrn' else ()),
        diagnostics=diagnostics,
        notes="Opt-in native prognostic Noah-MP, WRF caller order and REAL4 state. "
              "Actual MPTABLE constants are restart identity. CPU original-source oracle qualification; "
              "small compiled deviations are disclosed in the u2-land precision receipts. "
              "GPU/coupled-original-WRF forecast qualification pending. "
              "Retained surface writeback also owns roughness_m/mol/hfx/qfx/qsfc/sfc_wspd. "
              + ("GROUNDWATER S_NODE retains its explicit WRF DOUBLE island; all carry fields remain REAL4. " if control=='opt_run' and value==1 else '')
              + ("EXPERIMENTAL: original DYNAMIC_VIC unassigned YD follows compiler-zero policy; sensitivity is material (UPSTREAM_WRF_BUGS.md). "
                 if (control=='opt_run' and value==8) or control=='opt_infdv' else ''),
    )
    for control,values,module,oracle,diagnostics in _NOAHMP_CONTROL_ROWS for value in values
) + (
    PhysicsStepSpec(
        family="land_surface",option=4,variant="sst_skin=1",name="WRF SST skin",
        wrf_slot="first_rk_surface_driver",owner_module="src/gpuwrf/coupling/sst_skin_hook.py",
        oracle="pristine WRF REAL4 savepoint oracle: proofs/u2_land/skin_oracle.json, 24 evolving skin steps",
        reads_state=("t_skin","xland","ustar","hfx","qfx","rhosfc"),writes_state=("t_skin",),
        reads_carry=("sst_skin","radiation_diagnostics","noahmp_land"),
        writes_carry=("sst_skin","noahmp_land"),diagnostics=("SST","SSTSK","DTW"),
        notes="After radiation/auxinput4, before surface layer/LSM. Old HFX/QFX/UST; current net SW. "
              "SSTSK/DTW cold seeds follow WRF allocator zero. Exact active full-carry restart required. "
              "RRTMG/Noah-MP/ice-free ocean scope; CPU original oracle qualified, GPU/D6 pending.",
    ),
)

# Optional surface-driver selectors have independent carry and admission gates.
SCHEME_STEP_SPECS += (
    PhysicsStepSpec(
        family="ocean", option=1, name="WRF one-dimensional ocean mixed layer",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/physics/ocean_mixed_layer.py",
        oracle="Unmodified WRF REAL savepoint phys/module_sf_oml.F; proofs/s2_oml/fixtures/oml.npz",
        reads_state=("u", "v", "xland", "hfx", "ustar", "t_skin"),
        writes_state=("t_skin",),
        reads_carry=("ocean", "history_diagnostics.lh", "radiation_diagnostics", "solar_interpolation", "noahmp_land"),
        writes_carry=("ocean", "noahmp_land.t_skin"),
        diagnostics=("TML", "T0ML", "HML", "H0ML", "HUML", "HVML", "TMOML"),
        notes="sf_ocean_physics=1 after LSM, before PBL; seven resident REAL fields. "
              "Requires Noah-MP=4, no lake/SST forcing. PWP=2 refused. "
              "CPU-oracle-qualified; GPU/coupled-forecast qualification pending.",
    ),
    *(PhysicsStepSpec(
        family="hurricane_flux", option=option, name=f"WRF isftcflx={option}",
        wrf_slot="first_rk_surface_driver",
        owner_module="src/gpuwrf/coupling/hurricane_surface_hook.py",
        oracle="Unmodified WRF REAL savepoint SFCLAYREV/SFCLAY/MYNNSFCLAY; "
               "proofs/s2_oml/fixtures/flux.json and cold_flux.json",
        reads_state=("u", "v", "theta", "qv", "p", "pb", "ph", "t_skin", "xland", "roughness_m", "ustar", "mol", "hfx", "qfx", "qsfc"),
        writes_state=("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv", "hfx", "qfx", "qsfc", "mol", "roughness_m"),
        reads_carry=("hurricane_ustm", "noahmp_land.snowh"),
        writes_carry=("hurricane_ustm", "history_diagnostics"),
        diagnostics=("T2", "Q2", "U10", "V10", "HFX", "LH", "UST", "USTM", "ZNT"),
        notes="isftcflx=1/2 only on sf_sfclay_physics=1/5/91; retained USTM, "
              "MYNN dissipative heating. CPU-oracle-qualified; GPU/coupled-forecast qualification pending.",
    ) for option in (1, 2)),
)

SCHEME_STEP_SPECS += tuple(PhysicsStepSpec(
    family='land_surface',option=4,variant=f'opt_gla={option}',name=f'Noah-MP glacier opt_gla={option}',
    wrf_slot='first_rk_surface_driver',owner_module='src/gpuwrf/physics/noahmp/glacier.py',
    oracle='Pristine WRF 4.7.1 module_sf_noahmp_glacier.F complete-call REAL4 savepoint gate; proofs/s2_glacier/oracle_manifest.json',
    reads_state=('u','v','theta','qv','p','ph','mu','t_skin'),writes_state=('t_skin',),
    reads_carry=('noahmp_land','glacier_qsnow','noahmp_rad'),writes_carry=('noahmp_land','glacier_qsnow'),
    diagnostics=('HFX','LH','QFX','GRDFLX','T2','Q2','QSNOWXY','RUNSF','RUNSB','SOILENERGY','SNOWENERGY'),
    notes='IVGTYP==ISICE uses the separate WRF glacier model. REAL4; held QSNOWXY required at restart. '
          'CPU-oracle-qualified; GPU/coupled-forecast qualification pending. Ice-free programs unchanged.')
    for option in (1,2))


# Append the four approved miscellaneous microphysics interfaces after the main inventory.
SCHEME_STEP_SPECS += (
    PhysicsStepSpec(
        "microphysics", 5, "Ferrier-Aligo", "solve_em_microphysics_driver",
        "src/gpuwrf/physics/ferrier/adapter.py",
        "Pristine WRF 4.7.1 FER_HIRES savepoint gate: Ferrier family 36 literal REAL columns BYTE; compiled deviations disclosed in proofs/v040/mpmisc/COMPILER_DEVIATIONS.md",
        reads_state=("theta", "qv", "qc", "qr", "qi", "p", "ph", "mu", "f_rimef_phy"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qt", "f_ice_phy", "f_rain_phy", "f_rimef_phy", "ferrier_sr"),
        returns_accumulators=("rain_acc",), diagnostics=("SR",),
        notes="Single domains, cu_physics=0; static WRF grid_id; CWM scalar transport, held fractions/SR; GPU/forecast pending.",
    ),
    PhysicsStepSpec(
        "microphysics", 15, "Ferrier-Aligo advected rime", "solve_em_microphysics_driver",
        "src/gpuwrf/physics/ferrier/adapter.py",
        "Pristine WRF 4.7.1 FER_HIRES_ADVECT savepoint gate: true QRIMEF caller and multi-step rime columns; literal BYTE, compiled deviations disclosed",
        reads_state=("theta", "qv", "qc", "qr", "qi", "p", "ph", "mu", "qrimef"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qrimef", "ferrier_sr"),
        returns_accumulators=("rain_acc",), diagnostics=("SR",),
        notes="Single domains, cu_physics=0; QRIMEF scalar transport/held SR; GPU/forecast pending.",
    ),
    PhysicsStepSpec(
        "microphysics", 7, "Goddard NUWRF four-ice", "solve_em_microphysics_driver",
        "src/gpuwrf/coupling/goddard4ice_adapter.py",
        "Pristine WRF 4.7.1 gsfcgce_4ice savepoint gate: 18 columns/36 groups literal BYTE; compiled prognostics in independent input-ULP/FMA bounds, diagnostic excesses disclosed",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh", "p", "ph", "mu", "w", "xland",
                     "acphysc", "acphyse", "acphysd", "acphyss", "acphysm", "acphysf"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
                      "acphysc", "acphyse", "acphysd", "acphyss", "acphysm", "acphysf",
                      "re_cloud_gsfc", "re_rain_gsfc", "re_ice_gsfc", "re_snow_gsfc", "re_graupel_gsfc", "re_hail_gsfc",
                      "re_cloud", "re_ice", "re_snow", "goddard_sr", "phys_tot"),
        returns_accumulators=("rain_acc", "snow_acc", "graupel_acc", "hail_acc"),
        diagnostics=("PHYS_TOT", "SR", "RE_CLOUD_GSFC", "RE_RAIN_GSFC", "RE_ICE_GSFC", "RE_SNOW_GSFC", "RE_GRAUPEL_GSFC", "RE_HAIL_GSFC"),
        notes="Single domains/non-Chem defaults; bottom W face, micron-to-meter radiation mediation, one-based accumulated-latent reset; native reduction cost and GPU/forecast qualification pending.",
    ),
    PhysicsStepSpec(
        "microphysics", 38, "Thompson variable-density hail and aerosols", "solve_em_microphysics_driver",
        "src/gpuwrf/coupling/thompson_hail_adapter.py",
        "Pristine WRF 4.7.1 module_mp_thompson.F true aerosol/hail savepoints: 11 columns/6160 physical values and 77 public surfaces literal BYTE; three evolving own-steps BYTE; independent compiled bounds/deviations in proofs/v040/mpmisc/thompson38/COMPILED_DEVIATIONS.md",
        reads_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Nc", "Ng", "qvolg",
                     "nwfa", "nifa", "nwfa2d", "nifa2d", "thompson_rainnc", "thompson_snownc", "p", "ph", "w"),
        writes_state=("theta", "qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Nc", "Ng", "qvolg", "nwfa", "nifa", "thompson_sr", "thompson_rainnc", "thompson_snownc"),
        returns_accumulators=("rain_acc", "snow_acc", "graupel_acc", "ice_acc"),
        diagnostics=("RAINNC", "SNOWNC", "GRAUPELNC", "SR", "QNWFA2D", "QNIFA2D"),
        notes="Experimental CPU-oracle-qualified with disclosed small compiled excesses. Single domains, wif_input_opt=1/aer_init_opt=0, zero black carbon, held surface emissions after physics and original REAL accumulated totals; no effective-radius flags. GPU/coupled-forecast qualification pending.",
    ),
)

SCHEME_STEP_SPECS += (
    PhysicsStepSpec(
        "microphysics", 56, "NTU multi-moment aerosols/ice", "solve_em_microphysics_driver",
        "src/gpuwrf/physics/ntu/adapter.py",
        "Pristine WRF module_mp_ntu.F REAL4/REAL8 public columns and scalar/activation/init savepoints; literal no-FMA BYTE (proofs/s2_ntu).",
        reads_state=("theta", "p", "ph", "w", *state_leaves_for_mp(56), "ntu_rainnc", "ntu_snownc",
                     "rain_acc", "snow_acc", "ice_acc", "graupel_acc", "hail_acc"),
        writes_state=("theta", *state_leaves_for_mp(56), "ntu_rainnc", "ntu_snownc", "ntu_sr"),
        returns_accumulators=("rain_acc", "snow_acc", "ice_acc", "graupel_acc", "hail_acc"),
        diagnostics=("RAINNC", "SNOWNC", "GRAUPELNC", "HAILNC", "SR"),
        notes="Experimental single domains. All 28 prognostics, original DOUBLE math/table islands, native column-local adaptive loops. Original compiler-zero initialization/scratch sensitivity disclosed. GPU/coupled-forecast qualification pending.",
    ),
)

SCHEME_STEP_SPECS += _NOAHMP_CONTROL_STEP_SPECS


SCHEME_STEP_SPECS += (PhysicsStepSpec(family="microphysics",option=11,name="CAM Morrison-Gettelman",
    wrf_slot="solve_em_microphysics_driver",owner_module="src/gpuwrf/physics/cammg/adapter.py",
    oracle="proofs/s2_cammg: original CAMMGMP savepoints, literal 48calls/all52publicfields BYTE; native compiled 48calls/all27modifiedoutputs within preregistered original FMA/input-ULP enclosure",
    reads_state=("theta","p","ph",*state_leaves_for_mp(11)),writes_state=("theta",*state_leaves_for_mp(11)),
    returns_accumulators=("rain_acc","snow_acc"),notes="CPU-oracle-qualified; GPU/coupled-forecast qualification pending. WRF DOUBLE CAM island, REAL4 public/held carry. No WRF mp11/bl9/ra3 pairing requirement; driver warns/zeros unsupported convection inputs."),)

# Radiation cloud-fraction options (namelist icloud, radiation_driver.F:1446-1506) select the
# CLDFRA/QC/QI view both RRTMG adapters consume; variant keys them under option 4 like the
# Noah-MP controls. No carry: held CLDFRA rides in the existing radiation diagnostics.
SCHEME_STEP_SPECS += tuple(
    PhysicsStepSpec(
        family="radiation", option=4, variant=f"icloud={value}", name=name,
        wrf_slot="first_rk_radiation_driver",
        owner_module="src/gpuwrf/physics/cloud_fraction.py",
        oracle=oracle,
        reads_state=("theta", "qv", "qc", "qi", "qs", "p", "pb", "ph", "phb", "xland"),
        writes_state=("theta",),
        diagnostics=("CLDFRA",),
        notes=notes,
    )
    for value, name, oracle, notes in (
        (3, "WRF icloud=3 cal_cldfra3 radiation cloud fraction",
         "verbatim pristine cal_cldfra3/find_cloudLayers/adjust_* + Thompson RSLF/RSIF savepoint oracle: "
         "proofs/w41_icloud/oracle_gate.py on fixtures/icloud3_columns_v1.npz",
         "icloud=3 replaces cal_cldfra1 and the icloud_bl merge; radiation-only qc/qi. CPU-oracle-qualified; "
         "GPU-qualified: WN3 0227 3-nest 24 h D6 PASS vs original CPU-WRF (proofs/w41_icloud)."),
        (2, "WRF icloud=2 cal_cldfra2 radiation cloud fraction",
         "verbatim pristine cal_cldfra2 savepoint (binary qc+qi>1e-6): tests/v041/test_icloud.py",
         "icloud=2 replaces cal_cldfra1 and the icloud_bl merge. CPU-oracle-qualified; GPU-qualified: WN3 0227 "
         "3-nest 24 h D6 PASS vs original CPU-WRF (proofs/w41_icloud2)."),
    )
)
# icloud=0: CLDFRA = 0 and RRTMG skips every condensate/radius input (IF ICLOUD .ne. 0), so radiation reads no
# hydrometeor; microphysics/MYNN clouds evolve unchanged (cloud-radiation decoupling switch).
SCHEME_STEP_SPECS += (PhysicsStepSpec(
    family="radiation", option=4, variant="icloud=0", name="WRF icloud=0 cloud-free RRTMG radiation",
    wrf_slot="first_rk_radiation_driver",
    owner_module="src/gpuwrf/coupling/physics_couplers.py",
    oracle="actual pristine radiation_driver ICLOUD=0 vs ICLOUD=1 savepoint on 16 real WN3 columns: "
           "proofs/w42_icloud0/fixtures/z01_radiation_driver_icloud.json (tests/v041/test_icloud0_oracle.py)",
    reads_state=("theta", "qv", "p", "pb", "ph", "phb", "xland"),
    writes_state=("theta",),
    diagnostics=("CLDFRA",),
    notes="icloud=0 runs no cloud-fraction branch (no icloud_bl merge); history CLDFRA = 0. CPU-oracle-qualified; "
          "GPU-qualified: WN3 0227 3-nest 24 h D6 PASS vs original CPU-WRF (proofs/w42_icloud0).",
),)

SCHEME_STEP_SPECS_BY_KEY: Mapping[tuple[str, int, str], PhysicsStepSpec] = {
    (spec.family, spec.option, spec.variant): spec for spec in SCHEME_STEP_SPECS
}


def scheme_step_spec(family: str, option: int, variant: str = "") -> PhysicsStepSpec:
    """Return a frozen step spec for a nonzero physics option.

    ``variant`` is only needed for radiation, where one option (4) selects both
    the RRTMG LW (``variant="lw"``) and SW (``variant="sw"``) adapters.
    """

    return SCHEME_STEP_SPECS_BY_KEY[(family, int(option), variant)]


def assert_interfaces_consistent() -> None:
    """Validate the interface freeze against the Registry freeze."""

    assert_registry_consistent()

    expected = {
        ("microphysics", opt, "") for opt in ACCEPTED_MP_PHYSICS if opt != 0
    } | {
        ("pbl", opt, "") for opt in ACCEPTED_BL_PBL_PHYSICS if opt != 0
    } | {
        ("surface_layer", opt, "") for opt in ACCEPTED_SF_SFCLAY_PHYSICS if opt != 0
    } | {
        ("cumulus", opt, "") for opt in ACCEPTED_CU_PHYSICS if opt != 0
    } | {
        ("land_surface", opt, "") for opt in ACCEPTED_SF_SURFACE_PHYSICS if opt != 0
    } | {
        ("radiation", opt, "lw") for opt in ACCEPTED_RA_LW_PHYSICS if opt != 0
    } | {
        ("radiation", opt, "sw") for opt in ACCEPTED_RA_SW_PHYSICS if opt != 0
    }
    expected |= {("ocean", 1, ""), ("hurricane_flux", 1, ""), ("hurricane_flux", 2, "")}
    actual = set(SCHEME_STEP_SPECS_BY_KEY)
    missing = expected - actual
    if missing:
        raise AssertionError(f"missing PhysicsStepSpec entries: {sorted(missing)}")

    slots = set(WRF_CALL_ORDER_SLOTS)
    allowed_writes = set(STATE_TENDENCY_KEYS)
    for spec in SCHEME_STEP_SPECS:
        if spec.wrf_slot not in slots:
            raise AssertionError(f"{spec.family}:{spec.option} has unknown WRF slot {spec.wrf_slot!r}")
        if not spec.owner_module.startswith("src/gpuwrf/"):
            raise AssertionError(f"{spec.family}:{spec.option} owner is not a source module")
        if "savepoint" not in spec.oracle.lower():
            raise AssertionError(f"{spec.family}:{spec.option} oracle must be a WRF savepoint gate")
        unknown_writes = set(spec.writes_state) - allowed_writes
        if unknown_writes:
            raise AssertionError(
                f"{spec.family}:{spec.option} writes unknown PhysicsTendency State keys: {sorted(unknown_writes)}"
            )

    sample = PhysicsTendency(
        state_tendencies={"theta": object()},
        accumulator_increments={"rainc_acc": object()},
    )
    sample.validate_keys()


__all__ = [
    "PHYSICS_INTERFACE_VERSION",
    "PHYSICS_REGISTRY_VERSION",
    "ArrayLike",
    "WRF_CALL_ORDER_SLOTS",
    "STATE_TENDENCY_KEYS",
    "ACCUMULATOR_UPDATE_KEYS",
    "PhysicsTendency",
    "PhysicsDiagnostics",
    "PhysicsCarry",
    "PhysicsStepResult",
    "PhysicsStepSpec",
    "SCHEME_STEP_SPECS",
    "SCHEME_STEP_SPECS_BY_KEY",
    "scheme_step_spec",
    "assert_interfaces_consistent",
]


if __name__ == "__main__":
    assert_interfaces_consistent()
    families = sorted({spec.family for spec in SCHEME_STEP_SPECS})
    print(
        "ok physics_interfaces consistent:",
        f"version={PHYSICS_INTERFACE_VERSION}",
        f"scheme_specs={len(SCHEME_STEP_SPECS)}",
        f"families={len(families)}({','.join(families)})",
        f"slots={len(WRF_CALL_ORDER_SLOTS)}",
    )
