"""Opt-in runtime scalar operands; static WRF selections remain in the namelist.

Families are separate optional namelist children. Their names/shapes/dtypes
select an executable; their values do not. Validation and digests are host-only.
"""
from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np

SCHEMA = "gpuwrf-runtime-parameters-v1"
FAMILIES = ("diffusion_parameters", "mynn_parameters", "thompson_parameters")
# Bounds reject invalid inputs; they never clamp an evolving model field.
DIFFUSION = {
    "c_s": (0., None, "1", "smag2d_km, Registry c_s; diff_opt=1/km_opt=4"),
    "diff_6th_factor": (0., 1., "1", "sixth_order_diffusion, Registry diff_6th_factor"),
    "dampcoef": (0., None, "s-1", "advance_w, Registry dampcoef (damp_opt=3)"),
    "zdamp": (0., None, "m", "advance_w, Registry zdamp (strictly positive)"),
    "w_alpha": (0., None, "m s-2", "w_damp, module_model_constants w_alpha"),
    "w_beta": (0., None, "1", "w_damp, module_model_constants w_beta"),
    "w_crit_cfl": (0., None, "1", "w_damp, Registry w_crit_cfl"),
}
W_DEFAULTS = {"w_alpha": .3, "w_beta": 1., "w_crit_cfl": 1.}
MYNN = {
    "sqfac": (3., "1", "module_bl_mynnedmf.F:305,3037; Kq = Sqfac * Km"),
    "alp1": (.23, "1", "module_bl_mynnedmf.F:1761; CASE(1) PBL-depth length coefficient"),
    "cphm_unst": (16., "1", "module_bl_mynnedmf.F:275,7724; PBL unstable phi_m coefficient"),
    "am_st": (6.1, "1", "module_bl_mynnedmf.F:7712; PBL stable phi_m coefficient"),
    "entrainment": (.33, "m s-1", "module_bl_mynnedmf.F:6136; EDMF entrainment numerator"),
}


def diffusion_value(namelist, name):
    """A device scalar when selected, otherwise the original Python constant."""
    controls = getattr(namelist, "diffusion_parameters", None)
    if controls is not None and name in controls:
        return controls[name]
    return W_DEFAULTS[name] if name in W_DEFAULTS else float(getattr(namelist, name))


def w_damping_kwargs(namelist):
    controls = getattr(namelist, "diffusion_parameters", None)
    if controls is None:
        return {}
    names = {"w_alpha": "w_alpha", "w_beta": "w_damp_on", "w_crit_cfl": "w_crit_cfl"}
    return {dest: controls[name] for name, dest in names.items() if name in controls}


def load_parameters(path):
    """Read the small per-domain JSON manifest; reject misspellings and empty work."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or set(payload) != {"schema", "domains"} or payload["schema"] != SCHEMA:
        raise ValueError(f"runtime parameters require schema={SCHEMA!r} and domains")
    domains = payload["domains"]
    if not isinstance(domains, dict) or not domains:
        raise ValueError("runtime parameters need a nonempty domains mapping")
    for domain, families in domains.items():
        if not isinstance(domain, str) or len(domain) != 3 or domain[0] != "d" or not domain[1:].isdigit() or int(domain[1:]) < 1:
            raise ValueError(f"invalid runtime-parameter domain {domain!r}")
        if not isinstance(families, dict) or not families or set(families) - set(FAMILIES):
            raise ValueError(f"{domain}: expected named parameter families {FAMILIES}")
        for family, values in families.items():
            if not isinstance(values, dict) or not values:
                raise ValueError(f"{domain}/{family}: expected a nonempty parameter mapping")
    check_wrf_domain_scope(domains); _refuse_unqualified_families(domains)
    return domains


def bind_diffusion_parameters(namelist, values):
    """Validate active native consumers before turning named values into REAL operands."""
    import os; _refuse_unqualified_families({"d01": {"diffusion_parameters": values}})
    import jax.numpy as jnp

    if not isinstance(values, dict) or not values:
        raise ValueError("diffusion_parameters must be a nonempty mapping")
    unknown = set(values) - set(DIFFUSION)
    if unknown:
        if unknown & {"khdif", "kvdif"}:
            raise ValueError("khdif/kvdif require constant-K km_opt=1, which the native driver does not support")
        if "c_k" in unknown:
            raise ValueError("runtime c_k for LES km_opt=2/5 is deferred pending separate native qualification")
        raise ValueError(f"unsupported diffusion parameters {sorted(unknown)}")
    if os.environ.get("GPUWRF_DYN_FP32", "0") != "1":
        raise ValueError("runtime diffusion/damping parameters require the native REAL dynamics")
    out = {}
    for name, value in sorted(values.items()):
        arr = np.asarray(value)
        if arr.shape or arr.dtype.kind not in "fiu" or not np.isfinite(arr):
            raise ValueError(f"{name}: expected a finite numeric scalar")
        lower, upper, _units, _source = DIFFUSION[name]
        if arr < lower or (upper is not None and arr > upper) or (name == "zdamp" and arr == 0):
            raise ValueError(f"{name}: value outside supported physical bounds")
        if name == "c_s":
            if (int(namelist.diff_opt), int(namelist.km_opt)) != (1, 4):
                raise ValueError("runtime c_s currently requires diff_opt=1/km_opt=4 (2-D Smagorinsky)")
        elif name == "diff_6th_factor":
            from gpuwrf.dynamics.explicit_diffusion import _NATIVE_DIFFUSION_FP32
            if not _NATIVE_DIFFUSION_FP32 or not int(namelist.diff_6th_opt):
                raise ValueError("diff_6th_factor needs active native sixth-order diffusion")
            from gpuwrf.runtime.operational_mode import _acoustic_lateral_bc_flags
            periodic, specified, nested = _acoustic_lateral_bc_flags(namelist)
            if periodic or not (specified or nested):
                raise ValueError("runtime sixth-order factor requires a specified/nested domain")
            if int(namelist.km_opt) in (2, 5):
                raise ValueError("runtime sixth-order factor is not wired to LES TKE transport")
            if int(getattr(namelist, "tracer_opt", 0)):
                raise ValueError("runtime sixth-order factor is not wired to passive tracer transport")
        elif name in ("dampcoef", "zdamp"):
            if int(namelist.damp_opt) != 3:
                raise ValueError(f"{name}: no active damp_opt=3 consumer")
        else:
            from gpuwrf.runtime.operational_mode import _W_DAMP_STAGE
            if int(namelist.w_damping) != 1 or not _W_DAMP_STAGE:
                raise ValueError(f"{name}: needs WRF stage w_damping=1")
        rounded = np.float32(arr)
        if not np.isfinite(rounded) or (name == "zdamp" and rounded <= 0):
            raise ValueError(f"{name}: not representable as a finite valid REAL scalar")
        out[name] = jnp.asarray(rounded)
    if "dampcoef" in out:
        # Preserve the OFF program's scalar-folding policy. In legacy mode its
        # Python dts*dampcoef is rounded once; tracing a REAL dampcoef instead
        # changes PROD RK1 by one ULP before any field operation. Bind the two
        # distinct stage strengths once on the host, outside the timestep loop.
        from gpuwrf.contracts.real_constfold import real_scalar_product
        dt = float(namelist.dt_s)
        stage_dts = (dt / 3., dt / float(namelist.acoustic_substeps))
        coefficient = np.float32(np.asarray(values["dampcoef"]))
        # All representations of the declared REAL default select its released
        # fold; other values use a REAL product like pristine WRF. Do not let
        # sub-REAL input bits change the bound physics or restart digest.
        if coefficient == np.float32(namelist.dampcoef):
            coefficient = float(namelist.dampcoef)
        out["dampmag_rk"] = jnp.asarray(
            [real_scalar_product(dts, coefficient) for dts in stage_dts], jnp.float32)
    return replace(namelist, diffusion_parameters=out)


def bind_domain_parameters(namelist, source, domain):
    domains = load_parameters(source) if not isinstance(source, dict) else check_wrf_domain_scope(source); _refuse_unqualified_families(domains)
    for family, values in domains.get(domain, {}).items():
        if family == "diffusion_parameters":
            namelist = bind_diffusion_parameters(namelist, values)
        elif family == "mynn_parameters":
            namelist = bind_mynn_parameters(namelist, values)
        elif family == "thompson_parameters":
            from gpuwrf.physics.thompson_parameters import bind_parameters
            namelist = bind_parameters(namelist, values)
        else:
            raise ValueError(f"unknown runtime parameter family {family!r}")
    return namelist


def bind_mynn_parameters(namelist, values):
    """Select measured WRF coefficients; broader closure bundles remain static."""
    import os; _refuse_unqualified_families({"d01": {"mynn_parameters": values}})
    import jax.numpy as jnp
    if not isinstance(values, dict) or not values or set(values) - set(MYNN):
        raise ValueError(f"mynn_parameters supports a nonempty subset of {tuple(MYNN)}")
    if int(namelist.bl_pbl_physics) != 5 or not namelist.run_physics:
        raise ValueError("runtime coefficients need the active MYNN bl_pbl_physics=5 path")
    if set(values) - {"sqfac"} and os.environ.get("GPUWRF_MYNN_FP32_COLUMNS", "0") != "1":
        raise ValueError("alp1/cphm_unst/am_st/entrainment require native FP32 MYNN columns")
    if "entrainment" in values:
        from gpuwrf.coupling.physics_couplers import _MYNN_EDMF
        if not _MYNN_EDMF or int(getattr(namelist, "bl_mynn_edmf", 1)) != 1:
            raise ValueError("entrainment needs active MYNN EDMF")
    out = {}
    for name, raw in sorted(values.items()):
        value = np.asarray(raw)
        if value.shape or value.dtype.kind not in "fiu" or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative numeric scalar")
        rounded = np.float32(value)
        if not np.isfinite(rounded):
            raise ValueError(f"{name} is not representable as finite REAL32")
        out[name] = jnp.asarray(rounded)
    return replace(namelist, mynn_parameters=out)


def mynn_kwargs(namelist):
    """Only the enabled family changes the MYNN adapter's argument structure."""
    controls = getattr(namelist, "mynn_parameters", None)
    if controls is None:
        return {}
    kwargs = {"sqfac": controls["sqfac"]} if "sqfac" in controls else {}
    coefficients = {name: value for name, value in controls.items() if name != "sqfac"}
    if coefficients:
        kwargs["mynn_coefficients"] = coefficients
    return kwargs


def restart_parameter_digests(namelist):
    """Family -> host digest for restart/branch identity; OFF adds no identity keys."""
    from gpuwrf.surface_params.controls import controls_digest
    get = namelist.get if isinstance(namelist, Mapping) else lambda name: getattr(namelist, name, None)
    return {name: controls_digest(get(name)) for name in FAMILIES if get(name) is not None}


def restart_static_namelist(namelist):
    """Use the branch owner's narrow, process-stable restart identity view."""
    from gpuwrf.runtime.restart_parameters import static_namelist
    return static_namelist(namelist)


# WRF scope of each name (pristine Registry.EM_COMMON / module PARAMETERs; critic FIX-5, w44-docs 2026-10-24):
# c_s, diff_6th_factor, dampcoef and zdamp are max_domains rconfig entries (one value per domain). w_crit_cfl is a
# count-1 rconfig, w_alpha/w_beta are module_model_constants PARAMETERs and the five MYNN coefficients are
# module_bl_mynnedmf PARAMETERs: one value for every domain. Differing per-domain values of those have no CPU-WRF
# equivalent and are refused.
PER_DOMAIN_IN_WRF = frozenset({"c_s", "diff_6th_factor", "dampcoef", "zdamp"})
WRF_GLOBAL = (frozenset(DIFFUSION) - PER_DOMAIN_IN_WRF) | frozenset(MYNN)


def check_wrf_domain_scope(domains, run_domains=None):
    """Refuse a WRF-global name unless every listed (and, if given, every run) domain sets the same value."""
    covered = set(domains) | set(run_domains or ())
    for name in sorted(WRF_GLOBAL):
        found = {domain: values[name] for domain, families in domains.items() if isinstance(families, dict)
                 for values in families.values() if isinstance(values, dict) and name in values}
        if not found:
            continue
        missing = sorted(covered - set(found))
        try:
            distinct = {np.float32(value).tobytes() for value in found.values()}
        except (TypeError, ValueError):
            raise ValueError(f"{name}: expected a finite numeric scalar") from None
        if missing or len(distinct) != 1:
            raise ValueError(f"{name} is one value for all domains in WRF; set the same value for every domain "
                             f"(missing {missing}, given {found})")
    return domains


# RC-only v0.4.4 policy. Extend this one set if another family fails qualification.
UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES = frozenset({"diffusion_parameters"})


def _refuse_unqualified_families(domains):
    blocked = sorted({family for families in domains.values() if isinstance(families, Mapping)
                      for family in families if family in UNAVAILABLE_RUNTIME_PARAMETER_FAMILIES})
    if blocked:
        raise ValueError(f"{', '.join(blocked)} not available in v0.4.4: failed the GPU qualification; see KNOWN_ISSUES")


def refuse_namelist_parameters(namelist):
    """Host admission before selected-family values can reach any model consumer."""
    selected = {name: getattr(namelist, name) for name in ("diffusion_parameters", "mynn_parameters")
                if getattr(namelist, name) is not None}
    _refuse_unqualified_families({"d01": selected})
