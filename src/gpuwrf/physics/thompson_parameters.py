"""Research extension: selected WRF Thompson constants as runtime operands.

WRF 4.7.1 requires recompilation to change these constants. This is not the
WRF ccn_conc namelist option, which has no Thompson mp=8 consumer.
"""
from __future__ import annotations

import os
import numpy as np

# name: (WRF default, units, definition/use in pristine module_mp_thompson.F)
PARAMETERS = {
    "nt_c": (100.e6, "m-3", "88: fixed mp8 droplet number; dynamic nu/gamma/freezing/radius dependencies"),
    "tno": (5.0, "m-3", "188/2625: Cooper ice-number prefactor"),
    "ato": (0.304, "K-1", "189/2625: Cooper temperature exponent"),
    "av_c": (31694600.0, "m-1 s-1", "163/3661: cloud-droplet fall-speed coefficient"),
    "av_i": (1493.9, "s-1", "161/3681/3686: ice mass/number fall-speed coefficient"),
    "av_r": (4854.0, "s-1", "143/786-788/801/3618/3626: rain fall law plus collection/evaporation initialization"),
    "autoconversion_threshold": (1.e-5, "kg m-3", "2180: source-literal rc guard, not a named WRF PARAMETER"),
    "autoconversion_rate": (.027, "1", "2186: source-literal zeta coefficient, not a factor or named WRF PARAMETER"),
}

# Initial research admission envelope, not WRF-declared bounds. These five
# coefficients can amplify fall-speed substeps or the Cooper exponential.
# All retained changed-WRF controls (2x default) remain admitted.
COEFFICIENT_BOUNDS = {
    name: (0.5 * PARAMETERS[name][0], 2.0 * PARAMETERS[name][0])
    for name in ("av_c", "av_i", "av_r", "tno", "ato")
}
EXPERIMENTAL_PARAMETERS = frozenset({
    "av_i", "av_r", "autoconversion_threshold", "autoconversion_rate",
})


def require_runtime_qualified(values):
    """Keep unqualified controls out of both public binding and the driver."""
    experimental = set(values) & EXPERIMENTAL_PARAMETERS
    if experimental:
        raise ValueError(f"Thompson controls {sorted(experimental)} are experimental and refused by the runtime binder/adapter until GPU/E41/cost qualification; low-level CPU qualification only")


def require_native():
    """Refuse paths for which these operands have not been wired/qualified."""
    required = ("GPUWRF_THOMPSON_NATIVE_REAL", "GPUWRF_THOMPSON_FULL_COLUMN",
                "GPUWRF_THOMPSON_MIXED_PHASE_WRF", "GPUWRF_THOMPSON_COLUMN_SED",
                "GPUWRF_THOMPSON_SED_FP32", "GPUWRF_THOMPSON_WRF_CONSTANTS")
    missing = [name for name in required if os.environ.get(name, "0") != "1"]
    if missing:
        raise ValueError("runtime Thompson parameters require " + ", ".join(missing))
    if os.environ.get("GPUWRF_THOMPSON_IMPLICIT_SED", "0") != "0":
        raise ValueError("runtime Thompson parameters require WRF explicit sedimentation")
    if os.environ.get("GPUWRF_MP_COLUMN_TILING", "0") != "0":
        raise ValueError("runtime Thompson parameters do not support optional column tiling")


def validate_parameters(values):
    """Host-only validation and REAL rounding; never clamp an evolving field."""
    import jax.numpy as jnp

    if not isinstance(values, dict) or not values:
        raise ValueError("thompson_parameters must be a nonempty mapping")
    unknown = set(values) - PARAMETERS.keys()
    if unknown:
        if "ccn_conc" in unknown:
            raise ValueError("ccn_conc has no Thompson consumer; it is not Nt_c")
        raise ValueError(f"unsupported Thompson parameters {sorted(unknown)}")
    require_native()
    result = {}
    for name, value in sorted(values.items()):
        scalar = np.asarray(value)
        zero_allowed = name == "autoconversion_rate"
        if scalar.shape or scalar.dtype.kind not in "fiu" or not np.isfinite(scalar) or scalar < 0 or (scalar == 0 and not zero_allowed):
            raise ValueError(f"{name} must be a finite {'nonnegative' if zero_allowed else 'positive'} numeric scalar")
        with np.errstate(over="ignore", under="ignore"):
            rounded = np.float32(scalar)
        if not np.isfinite(rounded) or rounded < 0 or (rounded == 0 and not (zero_allowed and scalar == 0)):
            raise ValueError(f"{name} must be representable as finite REAL32 without nonzero underflow")
        if name in COEFFICIENT_BOUNDS:
            lower, upper = COEFFICIENT_BOUNDS[name]
            if not np.float32(lower) <= rounded <= np.float32(upper):
                raise ValueError(f"{name} must be within {lower:g} to {upper:g} (0.5..2 times WRF default)")
        if name == "nt_c" and not 2. <= rounded <= np.float32(1999.e6):
            raise ValueError("nt_c support is 2..1999e6 m-3 (WRF cloud-number floor/cap; no NINT overflow)")
        if name == "autoconversion_threshold" and rounded < np.float32(1.e-12):
            raise ValueError("autoconversion_threshold must be >= WRF R1=1e-12 kg m-3; inactive rc=R1 has no initialized cloud slope")
        result[name] = jnp.asarray(rounded)
    return result


def bind_parameters(namelist, values):
    """Shared runtime-parameter registry entry, following w44-params' binder API."""
    from dataclasses import replace
    if int(namelist.mp_physics) != 8 or not namelist.run_physics:
        raise ValueError("thompson_parameters requires active mp_physics=8")
    if os.environ.get("GPUWRF_MICROPHYSICS_WRF_ORDER", "1") != "1":
        raise ValueError("thompson_parameters requires the WRF-ordered microphysics driver")
    checked = validate_parameters(values)
    require_runtime_qualified(checked)
    return replace(namelist, thompson_parameters=checked)
