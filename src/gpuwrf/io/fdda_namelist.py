"""``&fdda`` namelist binding for PSU grid (1) and spectral (2) nudging.

Pure Python (no JAX): the CLI runs :func:`fdda_namelist_errors` before the heavy
pipeline; the loaders call :func:`grid_fdda_options` per domain.

WRF semantics (pristine WRF v4):

* ``gfdda_interval_m``/``gfdda_end_h``/``gfdda_inname``/``io_form_gfdda`` are the
  auxinput10 stream (``dyn_em/namelist_remappings_em.h``), remapped AFTER
  ``check_nml_consistency`` zeroed the surface-FDDA ``sgfdda_*`` times of every
  domain with ``grid_sfdda = 0`` (``share/module_check_a_mundo.F:1620-1640,2796``)
  -> ``auxinput9_end_h = 0`` -> fddagd's ``end_fdda_hour_sfc = 0``.
* ``fddagd_driver`` fatal checks: ``gfdda_begin_*`` != 0, ``gfdda_end_{y,d,m,s}``
  != 0, ``gfdda_interval_{y,d,h,s}`` != 0; ``fddagd``: interval <= 0; ``fddagdinit``:
  negative ``guv``/``gt``/``gq``.
* ``STEPFG = max(nint(fgdt*60/dt), 1)`` (``phys/module_physics_init.F:4751``).

Port scope: ``grid_fdda = 1`` (PSU analysis) and 2 (spectral), plus surface modes
``grid_sfdda = 1/2`` with PSU analysis, MYNN and Noah-MP (``wrfsfdda``).
Classic-Noah FASDAS indirect soil updates and ``io_form_gfdda != 2`` are refused. Observation options are validated by
``obs_namelist`` independently, so grid and station nudging can coexist.
"""

from __future__ import annotations

from typing import Any, Mapping
import math

#: &fdda keys this port binds for grid_fdda=1, with their WRF Registry defaults.
BOUND_DEFAULTS: dict[str, Any] = {
    "grid_fdda": 0,
    "grid_sfdda": 0,
    "sgfdda_inname": "sgfdda_d<domain>",
    "sgfdda_interval_m": 0,
    "sgfdda_interval": 0,
    "sgfdda_end_h": 0,
    "io_form_sgfdda": 2,
    "guv_sfc": 0.0, "gt_sfc": 0.0, "gq_sfc": 0.0, "rinblw": 0.0,
    "gfdda_inname": "wrffdda_d<domain>",
    "gfdda_interval_m": 0,
    "gfdda_interval": 0,
    "gfdda_end_h": 0,
    "io_form_gfdda": 2,
    "fgdt": 0.0,
    "fgdtzero": 0,
    "if_no_pbl_nudging_uv": 0,
    "if_no_pbl_nudging_t": 0,
    "if_no_pbl_nudging_q": 0,
    "if_zfac_uv": 0,
    "k_zfac_uv": 0,
    "if_zfac_t": 0,
    "k_zfac_t": 0,
    "if_zfac_q": 0,
    "k_zfac_q": 0,
    "guv": 0.0,
    "gt": 0.0,
    "gq": 0.0,
    "if_ramping": 0,
    "dtramp_min": 0.0,
}

SPECTRAL_DEFAULTS: dict[str, Any] = {
    "if_no_pbl_nudging_ph": 0, "if_zfac_ph": 0, "k_zfac_ph": 0,
    "dk_zfac_uv": 1, "dk_zfac_t": 1, "dk_zfac_ph": 1, "dk_zfac_q": 1,
    "ktrop": 0, "gph": 0.0, "xwavenum": 0, "ywavenum": 0,
}

#: Keys only the spectral (grid_fdda=2), surface (grid_sfdda) or PX soil-nudging paths read;
#: WRF's psufddagd ignores them, so any value is honoured for grid_fdda=1.
IGNORED_BY_GRID_FDDA: frozenset[str] = frozenset({
    "if_no_pbl_nudging_ph", "if_zfac_ph", "k_zfac_ph", "dk_zfac_uv", "dk_zfac_t", "dk_zfac_ph", "dk_zfac_q",
    "ktrop", "gph", "xwavenum", "ywavenum",
    "sgfdda_inname", "sgfdda_interval_d", "sgfdda_interval_h", "sgfdda_interval_m", "sgfdda_interval_s",
    "sgfdda_interval_y", "sgfdda_interval", "sgfdda_begin_y", "sgfdda_begin_d", "sgfdda_begin_h",
    "sgfdda_begin_m", "sgfdda_begin_s", "sgfdda_end_y", "sgfdda_end_d", "sgfdda_end_h", "sgfdda_end_m",
    "sgfdda_end_s", "io_form_sgfdda", "guv_sfc", "gt_sfc", "gq_sfc", "rinblw",
    # observation-nudging family (validated separately when obs_nudge_opt=1)
    "max_obs", "fdda_start", "fdda_end",
})

#: WRF fddagd_driver fatal checks (must be 0 for grid_fdda=1).
WRF_FATAL_NONZERO: tuple[str, ...] = (
    "gfdda_begin_y", "gfdda_begin_d", "gfdda_begin_h", "gfdda_begin_m", "gfdda_begin_s",
    "gfdda_end_y", "gfdda_end_d", "gfdda_end_m", "gfdda_end_s",
    "gfdda_interval_y", "gfdda_interval_d", "gfdda_interval_h", "gfdda_interval_s",
)

#: Switches whose non-zero value selects an unported nudging family.
REFUSED_SWITCHES: dict[str, str] = {
    "pxlsm_soil_nudge": "Pleim-Xiu soil nudging requires the PX LSM and its unqualified indirect soil-nudging path; set pxlsm_soil_nudge=0",
}

_SINGLE_ENTRY = frozenset({"gfdda_inname", "io_form_gfdda", "sgfdda_inname", "io_form_sgfdda", "if_ramping", "dtramp_min", "ktrop"})


def _raw(wrf_namelist: Mapping[str, Any], key: str, index: int) -> Any:
    """Explicit value of domain ``index`` (0-based), None when that entry was not written.

    A Fortran namelist read assigns only the entries present: ``grid_fdda = 1,`` with
    max_dom=3 leaves grid_fdda(2:3) at the Registry default 0 (``guv = 3e-4,`` leaves
    guv(2:3) = 0.).  So an omitted tail entry -- or entries 2..n of a scalar -- is
    ABSENT, never the last written value.  ``_SINGLE_ENTRY`` keys have one value.
    """

    raw = (wrf_namelist.get("fdda") or {}).get(key)
    if raw is None:
        return None
    if key in _SINGLE_ENTRY:
        index = 0
    if isinstance(raw, (list, tuple)):
        return raw[index] if index < len(raw) else None
    return raw if index == 0 else None


def _value(wrf_namelist: Mapping[str, Any], key: str, index: int) -> Any:
    value = _raw(wrf_namelist, key, index)
    return BOUND_DEFAULTS.get(key, SPECTRAL_DEFAULTS.get(key, 0)) if value is None else value


def _index(domain: str) -> int:
    return max(int(str(domain)[1:]) - 1, 0)


def grid_fdda_code(wrf_namelist: Mapping[str, Any], domain: str) -> int:
    return int(_value(wrf_namelist, "grid_fdda", _index(domain)))


def fdda_domain_errors(wrf_namelist: Mapping[str, Any], domain: str, dt_s: float | None = None) -> list[str]:
    """WRF-fatal and unsupported &fdda selections of one domain (empty when grid_fdda=0 and clean)."""

    index = _index(domain)
    errors: list[str] = []
    from gpuwrf.io.obs_namelist import observation_namelist_errors
    errors.extend(observation_namelist_errors(wrf_namelist,domain))
    from gpuwrf.io.wrfsfdda import surface_errors
    errors.extend(surface_errors(wrf_namelist,domain))
    for key, why in REFUSED_SWITCHES.items():
        value = _raw(wrf_namelist, key, index)
        if value is not None and int(value) != 0:
            errors.append(f"&fdda {key}={value} ({domain}): {why}")
    code = grid_fdda_code(wrf_namelist, domain)
    if code == 0:
        return errors
    if code not in (1, 2):
        errors.append(f"&fdda grid_fdda={code} ({domain}): WRF fatal 'The fdda option does not exist'")
        return errors
    for key in WRF_FATAL_NONZERO:
        value = _raw(wrf_namelist, key, index)
        if value is not None and int(value) != 0:
            errors.append(f"&fdda {key}={value} ({domain}): WRF fddagd_driver is fatal on it "
                          "(analysis nudging starts at the model start; use gfdda_end_h / gfdda_interval_m)")
    interval = _interval_m(wrf_namelist, index)
    if interval <= 0:
        errors.append(f"&fdda gfdda_interval_m={interval} ({domain}): WRF fatal 'gfdda_interval_m must be > 0'")
    for key in (("guv", "gt", "gq", "gph") if code == 2 else ("guv", "gt", "gq")):
        coefficient = float(_value(wrf_namelist, key, index))
        if not math.isfinite(coefficient) or coefficient < 0.0:
            errors.append(f"&fdda {key} ({domain}): WRF fatal '{key} must be positive'")
    if int(_value(wrf_namelist, "io_form_gfdda", index)) != 2:
        errors.append(f"&fdda io_form_gfdda ({domain}): only NetCDF (2) wrffdda files are read")
    if int(_value(wrf_namelist, "if_ramping", index)) not in (0, 1):
        errors.append(f"&fdda if_ramping ({domain}): must be 0 or 1")
    switches = ("if_no_pbl_nudging_uv", "if_no_pbl_nudging_t", "if_no_pbl_nudging_q",
                "if_zfac_uv", "if_zfac_t", "if_zfac_q", "fgdtzero")
    if code == 2:
        switches += ("if_no_pbl_nudging_ph", "if_zfac_ph")
        for suffix in ("uv", "t", "ph", "q"):
            if any(int(_value(wrf_namelist, prefix + suffix, index)) == 1
                   for prefix in ("if_no_pbl_nudging_", "if_zfac_")):
                if int(_value(wrf_namelist, "dk_zfac_" + suffix, index)) < 1:
                    errors.append(f"&fdda dk_zfac_{suffix} ({domain}): must be greater or equal than 1")
        e_vert = (wrf_namelist.get("domains") or {}).get("e_vert")
        if isinstance(e_vert, (list, tuple)):
            e_vert = e_vert[index] if index < len(e_vert) else e_vert[-1]
        if e_vert is not None and int(_value(wrf_namelist, "ktrop", index)) > int(e_vert):
            errors.append(f"&fdda ktrop ({domain}): KTROP is set above the model top")
    for key in switches:
        if int(_value(wrf_namelist, key, index)) not in (0, 1):
            errors.append(f"&fdda {key} ({domain}): must be 0 or 1")
    pbl = _physics_int(wrf_namelist, "bl_pbl_physics", index, 5)  # omitted -> the pipelines' MYNN
    pbl_switches = tuple(k for k in switches if k.startswith("if_no_pbl_nudging_"))
    if pbl != 5 and any(int(_value(wrf_namelist, k, index)) == 1 for k in pbl_switches):
        errors.append(f"&fdda if_no_pbl_nudging_* ({domain}): the PBL-top weights need grid%PBLH, which the "
                      f"port's State carries only from MYNN (bl_pbl_physics=5), not {pbl}")
    return errors


def _physics_int(wrf_namelist: Mapping[str, Any], key: str, index: int, default: int) -> int:
    raw = (wrf_namelist.get("physics") or {}).get(key)
    if isinstance(raw, (list, tuple)):
        raw = (raw[index] if index < len(raw) else raw[-1]) if raw else None
    return default if raw is None else int(raw)


def _interval_m(wrf_namelist: Mapping[str, Any], index: int) -> int:
    interval = int(_value(wrf_namelist, "gfdda_interval_m", index))
    if interval == 0:  # set_timekeeping: interval_m .EQ. 0 -> auxinput10_interval
        interval = int(_value(wrf_namelist, "gfdda_interval", index))
    return interval


def stepfg(fgdt_min: float, dt_s: float) -> int:
    """``STEPFG = max(nint(FGDT*60./DT), 1)`` in REAL (phys/module_physics_init.F:4751)."""

    import struct

    def f32(x: float) -> float:
        return struct.unpack("f", struct.pack("f", float(x)))[0]

    value = f32(f32(f32(fgdt_min) * f32(60.0)) / f32(dt_s))
    rounded = int(abs(value) + 0.5) * (1 if value >= 0 else -1)  # Fortran NINT
    return max(rounded, 1)


def grid_fdda_options(wrf_namelist: Mapping[str, Any], domain: str, dt_s: float) -> dict[str, Any] | None:
    """Bound ``GridFddaOptions`` keyword values of ``domain`` (None when grid_fdda=0).

    Raises ``ValueError`` with every refusal when the selection is not runnable.
    """

    errors = fdda_domain_errors(wrf_namelist, domain, dt_s)
    if grid_fdda_code(wrf_namelist, domain) == 0:
        if errors:
            raise ValueError("; ".join(errors))
        return None
    if errors:
        raise ValueError("; ".join(errors))
    index = _index(domain)
    v = lambda key: _value(wrf_namelist, key, index)  # noqa: E731
    result = dict(
        guv=float(v("guv")), gt=float(v("gt")), gq=float(v("gq")),
        interval_m=_interval_m(wrf_namelist, index), end_h=int(v("gfdda_end_h")), end_h_sfc=0,
        stepfg=stepfg(float(v("fgdt")), float(dt_s)), fgdtzero=int(v("fgdtzero")),
        if_no_pbl_nudging_uv=int(v("if_no_pbl_nudging_uv")), if_no_pbl_nudging_t=int(v("if_no_pbl_nudging_t")),
        if_no_pbl_nudging_q=int(v("if_no_pbl_nudging_q")),
        if_zfac_uv=int(v("if_zfac_uv")), k_zfac_uv=int(v("k_zfac_uv")),
        if_zfac_t=int(v("if_zfac_t")), k_zfac_t=int(v("k_zfac_t")),
        if_zfac_q=int(v("if_zfac_q")), k_zfac_q=int(v("k_zfac_q")),
        if_ramping=int(v("if_ramping")), dtramp_min=float(v("dtramp_min")),
    )
    if grid_fdda_code(wrf_namelist, domain) == 2:
        result.update({k: float(v(k)) if k == "gph" else int(v(k)) for k in SPECTRAL_DEFAULTS})
    return result


def wrffdda_filename(wrf_namelist: Mapping[str, Any], domain: str) -> str:
    template = str(_value(wrf_namelist, "gfdda_inname", 0)).strip().strip("'\"")
    name = template.replace("<domain>", f"{_index(domain) + 1:02d}")
    if "<" in name:
        raise ValueError(f"{domain}: unresolved gfdda_inname {name!r}")
    return name


def fdda_namelist_errors(wrf_namelist: Mapping[str, Any], domains: tuple[str, ...]) -> list[str]:
    """Every &fdda refusal of the run (CLI pre-JAX check)."""

    errors: list[str] = []
    for domain in domains:
        errors.extend(fdda_domain_errors(wrf_namelist, domain))
    from gpuwrf.io.obs_namelist import DEFAULTS as OBS_DEFAULTS
    known = set(BOUND_DEFAULTS) | IGNORED_BY_GRID_FDDA | set(WRF_FATAL_NONZERO) | set(REFUSED_SWITCHES) | set(OBS_DEFAULTS)
    for key in sorted(wrf_namelist.get("fdda") or {}):
        if str(key).lower() not in known:
            errors.append(f"&fdda {key}: not a WRF v4 &fdda key the port knows; refusing rather than ignoring it")
    return errors


#: Output-only differences of a grid_fdda=1 run vs WRF history (forecast unchanged).
OUTPUT_GAP_WARNING = (
    "&fdda grid_fdda=1: output-only gap -- the global attributes "
    "FGDT/GUV/GT/GQ/IF_RAMPING/DTRAMP_MIN are not written "
    "(forecast unchanged)"
)


def fdda_namelist_warnings(wrf_namelist: Mapping[str, Any], domains: tuple[str, ...]) -> list[str]:
    """Non-fatal notes for the CLI (printed before the forecast)."""

    warnings = []
    if any(grid_fdda_code(wrf_namelist, d) == 1 for d in domains):
        warnings.append(OUTPUT_GAP_WARNING)
    if any(grid_fdda_code(wrf_namelist, d) == 2 for d in domains):
        warnings.append("&fdda grid_fdda=2: CPU-oracle-qualified; GPU/coupled-forecast qualification pending. "
                        "WRF FDDA global attributes are not written.")
    from gpuwrf.io.wrfsfdda import surface_code
    if any(surface_code(wrf_namelist,d) for d in domains):
        warnings.append('&fdda grid_sfdda=1/2: CPU-oracle-qualified; GPU/coupled-forecast qualification pending. '
                        'MYNN PBL/SFCLAY + Noah-MP only; classic Noah FASDAS soil nudging is refused. '
                        'SDA_HFX/SDA_QFX, QNORM, HFX_BOTH/QFX_BOTH diagnostics are not written.')
    return warnings
