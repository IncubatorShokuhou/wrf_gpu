"""Explicit Noah-MP W2 controls at WRF's TRANSFER_MP_PARAMETERS boundary.

No model arithmetic changes: existing consumers accept gathered cell/layer
fields. Soil values must replace both WATER's parameters and ENERGY's bundle;
HVT must replace both PHENOLOGY's parameter and ENERGY's bundle. Legacy Track P
controls stay separate so a request for its priors keeps its original meaning.
"""
from __future__ import annotations

import numpy as np

SOIL = frozenset({"smcmax", "dksat", "bexp", "psisat"})
VEGETATION = frozenset({"rsmin", "hvt", "hvb"})
SNOW = frozenset({"z0sno", "snow_emis"})

# ControlSpec-compatible tuples; no controls import (the registry imports us).
# Ranges are admission bounds, not a claim that all values are calibrated.
SPECS = {
    "smcmax": ("lsm", "soil", 1.e-6, 1., "m3 m-3", "SOILPARM MAXSMC (SMCMAX)",
               "ENERGY/PHASECHANGE/SOILWATER porosity; urban value fixed by WRF"),
    "dksat": ("lsm", "soil", 1.e-12, 1., "m s-1", "SOILPARM SATDK (DKSAT)",
              "SOILWATER conductivity and derived KDT; DWSAT remains independent"),
    "bexp": ("lsm", "soil", 1.e-6, 30., "1", "SOILPARM BB (BEXP)",
             "ENERGY/PHASECHANGE/SOILWATER Clapp-Hornberger exponent"),
    "psisat": ("lsm", "soil", 1.e-8, 100., "m", "SOILPARM SATPSI (PSISAT)",
               "ENERGY/PHASECHANGE saturated matric-potential magnitude (positive)"),
    "rsmin": ("lsm", "col", 1.e-6, 1.e5, "s m-1", "MPTABLE RS (RSMIN)",
              "CANRES/Jarvis minimum stomatal resistance (opt_crs=2 only)"),
    "hvt": ("lsm", "col", 0., 100., "m", "MPTABLE HVT(VEGTYP)",
            "PHENOLOGY snow burial, ENERGY displacement/wind, radiation geometry"),
    "hvb": ("lsm", "col", 0., 100., "m", "MPTABLE HVB(VEGTYP)",
            "PHENOLOGY snow burial and opt_rad=1 canopy geometry"),
    "z0sno": ("lsm", "col", 1.e-8, 1., "m", "MPTABLE Z0SNO",
              "ENERGY snow-covered ground momentum roughness"),
    "snow_emis": ("lsm", "col", .5, 1., "1", "MPTABLE SNOW_EMIS",
                  "ENERGY snow-covered ground emissivity"),
}


def inactive_reason(name, namelist):
    from gpuwrf.surface_params.controls import _options
    opt = _options(namelist)
    if not np.any(active_columns(namelist)):
        return f"{name}: this domain has no natural Noah-MP land columns"
    # Table-shaped groundwater/MMF and optional carry consumers need their own
    # qualification. Refuse those combinations rather than tune a partial path.
    if name in SOIL and int(opt("opt_run")) != 3:
        return f"{name}: W2 soil controls require opt_run=3 (Schaake)"
    if name in SOIL and int(opt("opt_soil")) != 1:
        return f"{name}: W2 SOILPARM controls require opt_soil=1"
    if name in SOIL and int(opt("opt_irr")) != 0:
        return f"{name}: W2 soil controls require opt_irr=0; irrigation keeps separate soil parameters"
    if name in SOIL and int(opt("opt_tdrn")) != 0:
        return f"{name}: W2 soil controls require opt_tdrn=0; tile drainage keeps precomputed KDT/FRZX"
    if name == "smcmax" and getattr(namelist, "noahmp_land", None) is None:
        return "smcmax: bound noahmp_land required for shared soil_porosity admission"
    if name == "rsmin" and int(opt("opt_crs")) != 2:
        return "rsmin: Jarvis CANRES requires opt_crs=2; Ball-Berry does not read RSMIN"
    return None


def target(name, namelist):
    """Gather exactly once, at the existing consumer's REAL precision."""
    p = namelist.noahmp_static.parameters
    ep = namelist.noahmp_energy_params
    if hasattr(ep, name):
        return getattr(ep, name)
    dtype = np.dtype(ep.z0mvt.dtype)
    table = np.asarray(getattr(p, name))
    category = namelist.noahmp_static.isltyp if name in SOIL else namelist.noahmp_static.ivgtyp
    indices = np.clip(np.asarray(category, np.int64), 0, len(table) - 1)
    field = table[indices].astype(dtype)
    return np.broadcast_to(field, (4,) + field.shape).copy() if name in SOIL else field


def table_base(name, namelist):
    """Unrounded table basis; urban SMCMAX reproduces WRF's post-gather override."""
    p = namelist.noahmp_static.parameters
    shape = tuple(namelist.noahmp_static.ivgtyp.shape)
    table = np.asarray(getattr(p, name), np.float64)
    if name in SNOW:
        return np.full(shape, float(table))
    category = namelist.noahmp_static.isltyp if name in SOIL else namelist.noahmp_static.ivgtyp
    field = table[np.clip(np.asarray(category, np.int64), 0, len(table) - 1)]
    if name == "smcmax":
        from gpuwrf.physics.noahmp.noahmp_driver import _urban_soil_on
        if _urban_soil_on():
            field = np.where(urban_columns(namelist), .45, field)
    return np.broadcast_to(field, (4,) + shape).copy() if name in SOIL else field


def apply(static, energy, controls):
    """Use the same traced operands in every consumer, with no value-dependent branch."""
    from gpuwrf.surface_params.controls import refuse_unqualified_controls
    refuse_unqualified_controls(controls)
    parameters, energy_fields = {}, {}
    for name, value in controls.items():
        if name in SOIL or name in VEGETATION:
            parameters[name] = value
        if hasattr(energy, name):
            energy_fields[name] = value
    if parameters:
        static = static.replace(parameters=static.parameters._replace(**parameters))
    if energy_fields:
        energy = energy._replace(**energy_fields)
    return static, energy


def scaled_value(name, namelist, value):
    """Scale natural land only; keep WRF's inactive and urban values exact."""
    value = np.where(active_columns(namelist), value, np.asarray(target(name, namelist)))
    if name == "smcmax":
        from gpuwrf.physics.noahmp.noahmp_driver import _urban_soil_on
        if _urban_soil_on():
            value = np.where(urban_columns(namelist), np.asarray(.45, value.dtype), value)
    return value


def urban_columns(namelist):
    """Host equivalent of WRF URBAN_FLAG: table urban and LCZ classes51..61."""
    static = namelist.noahmp_static
    v = np.asarray(static.ivgtyp)
    return (v == int(static.parameters.isurban)) | ((v >= 51) & (v <= 61))


def active_columns(namelist):
    """WRF natural land, excluding separately routed glacier/lake columns."""
    static = namelist.noahmp_static
    landmask = static.landmask if static.landmask is not None else np.asarray(static.xland) == 1
    mask = np.asarray(landmask) == 1
    if static.lakemask is not None:
        mask = mask & (np.asarray(static.lakemask) <= .5)
    glacier = getattr(static, "glacier", None)
    if glacier is not None:
        mask = mask.copy()
        mask.flat[list(glacier.indices)] = False
    return mask


def range_values(name, namelist, value):
    """Admit WRF's zero WATER table placeholders without allowing land zeros.

    Inactive cells retain the exact bound parameter bytes. This is host input
    admission, not a model-state mask or a change to timestep arithmetic.
    """
    active = np.broadcast_to(active_columns(namelist), value.shape)
    baseline = np.asarray(target(name, namelist))
    if value[~active].tobytes() != baseline[~active].tobytes():
        raise ValueError(f"{name}: inactive water/glacier/lake cells must retain their bound values")
    return value[active]


def validation_namelist(namelist, land_state):
    """Temporary host validation context; never publish it as the step namelist."""
    if land_state is None:
        return namelist
    from dataclasses import is_dataclass, replace
    from types import SimpleNamespace
    return (replace(namelist, noahmp_land=land_state) if is_dataclass(namelist)
            else SimpleNamespace(**{**vars(namelist), "noahmp_land": land_state}))


def validate(controls, namelist):
    """Coupled bounds and WRF urban invariants, checked on the host before tracing."""
    from gpuwrf.surface_params.controls import _target
    static = namelist.noahmp_static
    if "hvt" in controls or "hvb" in controls:
        hvt = np.asarray(controls.get("hvt", _target("hvt", namelist)))
        hvb = np.asarray(controls.get("hvb", _target("hvb", namelist)))
        if np.any(hvb > hvt):
            raise ValueError("canopy height requires 0 <= HVB <= HVT")
    if "smcmax" not in controls:
        return
    cap = np.asarray(controls["smcmax"])
    if np.any(cap < np.asarray(namelist.noahmp_energy_params.smcref)):
        raise ValueError("smcmax: porosity must be >= SMCREF (field capacity)")
    from gpuwrf.physics.noahmp.noahmp_driver import _urban_soil_on
    urban = urban_columns(namelist)
    if _urban_soil_on() and np.any(cap[:, urban] != np.asarray(.45, cap.dtype)):
        raise ValueError("smcmax: WRF fixes urban porosity to 0.45")
    land = getattr(namelist, "noahmp_land", None)
    if land is None:
        raise ValueError("smcmax: bound noahmp_land required for shared soil_porosity admission")
    from gpuwrf.io.wrfinput_consistency import validate_soil_porosity
    # The initialized glacier partition records the exact ISICE & XICE<=0
    # caller predicate. Other ice-class cells must retain the ordinary cap.
    seaice = np.ones(np.shape(static.ivgtyp), dtype=np.float32)
    glacier = getattr(static, "glacier", None)
    if glacier is not None:
        seaice.flat[list(glacier.indices)] = 0.
    if not validate_soil_porosity(land.smois, cap, static.landmask, static.ivgtyp,
                                  seaice=seaice, sh2o=land.sh2o):
        raise ValueError("soil_porosity: SMOIS must be in [0, controlled SMCMAX]")


def validate_retained_porosity(namelist, land_fields):
    """Host checkpoint/branch seam; never alter the production static holder.

    This checks only active controlled porosity. An unchanged atmospheric
    checkpoint is not reinterpreted as a new analysis. The branch owner calls
    this before its no-domain-edits early return and before device restoration.
    """
    controls = getattr(namelist, "surface_controls", None)
    if controls is None or "smcmax" not in controls:
        return None
    if land_fields is None or any(land_fields.get(name) is None for name in ("smois", "sh2o")):
        raise ValueError("controlled SMCMAX branch requires retained SMOIS and SH2O")
    from types import SimpleNamespace
    from gpuwrf.surface_params.controls import controls_digest, validate_controls
    land = SimpleNamespace(smois=land_fields["smois"], sh2o=land_fields["sh2o"])
    view = validation_namelist(namelist, land)
    validate_controls({"smcmax": controls["smcmax"]}, view)
    return {"passed": True, "check": "soil_porosity", "scope": "retained soil water versus controlled SMCMAX",
            "smcmax_digest": controls_digest({"smcmax": controls["smcmax"]})}


def validate_cold_start(controls, namelist, land_state=None):
    """Refuse changed coefficients whose frozen NOAHMP_INIT partition is stale.

    Identity controls keep the original partition. Forecast steps and restored
    checkpoints already carry SH2O and do not rerun NOAHMP_INIT.
    """
    selected = set(controls) & {"bexp", "psisat", "smcmax"}
    if not selected:
        return
    changes = {name: np.asarray(controls[name]) != np.asarray(target(name, namelist)) for name in selected}
    if not any(np.any(value) for value in changes.values()):
        return
    land = land_state if land_state is not None else getattr(namelist, "noahmp_land", None)
    if land is None or getattr(land, "tslb", None) is None:
        raise ValueError("W2 cold-start soil controls require initialized TSLB to validate NOAHMP_INIT SH2O")
    temperature = np.asarray(land.tslb, np.float64)
    if not np.isfinite(temperature).all():
        raise ValueError("W2 cold-start soil controls require finite TSLB")
    frozen = (temperature < 273.149) & np.broadcast_to(active_columns(namelist), temperature.shape)
    affected = sorted(name for name, value in changes.items() if np.any(frozen & value))
    if affected:
        raise ValueError("W2 changed " + ", ".join(affected) +
                         " is unsupported on initially frozen natural soil: NOAHMP_INIT SH2O "
                         "was partitioned with the original tables; identity controls remain allowed")
