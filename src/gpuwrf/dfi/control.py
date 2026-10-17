"""``&dfi_control`` binding, WRF admission rules and TDFI stage windows (pure Python).

WRF 4.7.1 semantics reproduced here (source lines in pristine WRF):

* ``dfi_opt``: 0 off, 1 DFL, 2 DDFI, 3 TDFI (main/module_wrf_top.F:2133-2222).  The port runs
  DFL (1), DDFI (2) and TDFI (3).
* restart => ``dfi_opt = 0`` (share/module_check_a_mundo.F:2349-2351); TEMF PBL refused (:2337).
* ``time_step_dfi = -1`` => ``time_step`` (:2181-2190).
* TDFI windows (share/set_timekeeping.F): backward stage from the forecast start to
  ``dfi_bckstop``; forward stage from ``dfi_bckstop + (start - dfi_bckstop)/2`` (integer seconds)
  to ``dfi_fwdstop``.  Each stage runs ``ceil(run/|dt|)`` steps (frame/module_domain.F
  domain_clockisstopsubtime), the filter span is the stage's run interval (optfil_driver).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

from gpuwrf.dfi.filter import DFI_TIME_DIM, SUPPORTED_NFILTER, optfil_nstep2

__all__ = ["DfiControl", "DfiNotSupported", "dfi_control_from_namelist", "dfi_refusals",
           "DFI_TDFI"]

DFI_NODFI, DFI_DFL, DFI_DDFI, DFI_TDFI = 0, 1, 2, 3
_TEMF_PBL = 10
_NOAHMP_LSM = 4

# WRF Registry.EM_COMMON:2225-2243 defaults.
_DEFAULTS: dict[str, Any] = {
    "dfi_opt": 0, "dfi_savehydmeteors": 0, "dfi_nfilter": 7, "dfi_write_filtered_input": True,
    "dfi_write_dfi_history": False, "dfi_cutoff_seconds": 3600, "dfi_time_dim": 1000,
    "dfi_fwdstop_year": 2004, "dfi_fwdstop_month": 3, "dfi_fwdstop_day": 13, "dfi_fwdstop_hour": 12,
    "dfi_fwdstop_minute": 0, "dfi_fwdstop_second": 0,
    "dfi_bckstop_year": 2004, "dfi_bckstop_month": 3, "dfi_bckstop_day": 14, "dfi_bckstop_hour": 12,
    "dfi_bckstop_minute": 0, "dfi_bckstop_second": 0,
}


class DfiNotSupported(ValueError):
    """A requested DFI configuration the port does not reproduce (refused pre-JAX)."""


def _section(nml: Mapping, name: str) -> Mapping:
    for key, value in nml.items():
        if str(key).lower() == name and isinstance(value, Mapping):
            return value
    return {}


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _get(nml: Mapping, section: str, key: str, default: Any) -> Any:
    sec = _section(nml, section)
    for k, v in sec.items():
        if str(k).lower() == key:
            v = _first(v)
            return default if v is None else v
    return default


def _all_values(nml: Mapping, section: str, key: str) -> list[Any]:
    sec = _section(nml, section)
    for k, v in sec.items():
        if str(k).lower() == key:
            return list(v) if isinstance(v, (list, tuple)) else [v]
    return []


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().strip(".").lower() in {"t", "true", "1"}
    return bool(value)


def _int(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip()
    return int(float(value))


def _stamp(nml: Mapping, prefix: str, section: str = "dfi_control", defaults=None) -> datetime:
    parts = []
    for unit in ("year", "month", "day", "hour", "minute", "second"):
        key = f"{prefix}_{unit}"
        default = (defaults or _DEFAULTS).get(key, 0)
        parts.append(_int(_get(nml, section, key, default)))
    return datetime(*parts)


@dataclass(frozen=True)
class DfiControl:
    """Resolved ``&dfi_control`` for the root domain."""

    dfi_opt: int
    nfilter: int
    cutoff_s: int
    write_filtered_input: bool
    write_dfi_history: bool
    savehydmeteors: int
    time_step_dfi: int
    time_step: float
    moist_adv_dfi_opt: int
    constant_bc: bool
    start: datetime
    bckstop: datetime
    fwdstop: datetime

    @property
    def active(self) -> bool:
        return self.dfi_opt != DFI_NODFI

    @property
    def dt_s(self) -> float:
        return float(self.time_step_dfi if self.time_step_dfi > 0 else self.time_step)

    @property
    def bck_run_s(self) -> int:
        if self.dfi_opt == DFI_DFL:
            return 0
        return int((self.start - self.bckstop).total_seconds())

    @property
    def fwd_start(self) -> datetime:
        if self.dfi_opt == DFI_DFL:
            return self.start
        if self.dfi_opt == DFI_DDFI:
            return self.bckstop
        # set_timekeeping.F:145-173: start = bckstop + (head start - head bckstop)/2, integer seconds.
        return self.bckstop + timedelta(seconds=self.bck_run_s // 2)

    @property
    def forecast_start(self) -> datetime:
        # set_timekeeping.F:91-120: DFL launches at the integer midpoint.
        if self.dfi_opt == DFI_DFL:
            return self.start + timedelta(seconds=self.fwd_run_s // 2)
        return self.start

    @property
    def fwd_run_s(self) -> int:
        return int((self.fwdstop - self.fwd_start).total_seconds())

    @property
    def n_bck(self) -> int:
        return int(math.ceil(self.bck_run_s / self.dt_s))

    @property
    def n_fwd(self) -> int:
        return int(math.ceil(self.fwd_run_s / self.dt_s))


def dfi_control_from_namelist(nml: Mapping) -> DfiControl:
    """Resolve the root-domain DFI controls with WRF Registry defaults."""

    g = lambda key: _get(nml, "dfi_control", key, _DEFAULTS[key])  # noqa: E731
    time_step = float(_get(nml, "domains", "time_step", 0) or 0)
    fract_num = _int(_get(nml, "domains", "time_step_fract_num", 0) or 0)
    fract_den = _int(_get(nml, "domains", "time_step_fract_den", 1) or 1)
    if fract_num:
        time_step = time_step + fract_num / max(fract_den, 1)
    time_step_dfi = _int(_get(nml, "domains", "time_step_dfi", -1))
    restart = _bool(_get(nml, "time_control", "restart", False))
    dfi_opt = _int(g("dfi_opt"))
    if restart:
        dfi_opt = DFI_NODFI  # module_check_a_mundo.F:2349-2351
    return DfiControl(
        dfi_opt=dfi_opt,
        nfilter=_int(g("dfi_nfilter")),
        cutoff_s=_int(g("dfi_cutoff_seconds")),
        write_filtered_input=_bool(g("dfi_write_filtered_input")),
        write_dfi_history=_bool(g("dfi_write_dfi_history")),
        savehydmeteors=_int(g("dfi_savehydmeteors")),
        time_step_dfi=time_step_dfi,
        time_step=time_step,
        moist_adv_dfi_opt=_int(_get(nml, "dynamics", "moist_adv_dfi_opt", 0)),
        constant_bc=_bool(_get(nml, "bdy_control", "constant_bc", False)),
        start=_stamp(nml, "start", section="time_control",
                     defaults={f"start_{u}": d for u, d in zip(
                         ("year", "month", "day", "hour", "minute", "second"), (1, 1, 1, 0, 0, 0))}),
        bckstop=_stamp(nml, "dfi_bckstop"),
        fwdstop=_stamp(nml, "dfi_fwdstop"),
    )


def dfi_refusals(nml: Mapping, *, max_dom: int | None = None, resume: bool = False) -> list[str]:
    """Return human-readable reasons the requested DFI cannot run faithfully (empty = OK).

    ``max_dom``/``resume`` are the CLI's resolved run shape; ``None`` skips that check.
    """

    ctl = dfi_control_from_namelist(nml)
    if not ctl.active or resume:
        return []  # WRF: restart (resume) shuts DFI off
    out: list[str] = []
    if ctl.dfi_opt not in (DFI_DFL, DFI_DDFI, DFI_TDFI):
        name = {DFI_DFL: "DFL (1)", DFI_DDFI: "DDFI (2)"}.get(ctl.dfi_opt, str(ctl.dfi_opt))
        out.append(f"dfi_opt={ctl.dfi_opt} ({name}) is not implemented; supported dfi_opt values are 0..3")
        return out
    if max_dom is not None and int(max_dom) > 1:
        out.append("DFI is implemented for single-domain native runs; nested (max_dom>1) DFI is not yet ported "
                   "(WRF supports simultaneous-start nests; the port lacks recursive filter phases and nest clocks)")
    if ctl.nfilter not in SUPPORTED_NFILTER:
        out.append(f"dfi_nfilter={ctl.nfilter} is not reproduced (supported 0..7; 8=RHOFIL indexes out of "
                   "bounds in WRF itself, -1 debug reads uninitialised weights)")
    if ctl.savehydmeteors != 0:
        out.append("dfi_savehydmeteors (dfi_radar)=1 (radar/hydrometeor-preserving DFI with rebalance_dfi) is "
                   "not implemented")
    if ctl.write_dfi_history:
        out.append("dfi_write_dfi_history=.true. (history during the DFI stages) is not implemented")
    if ctl.time_step_dfi > 0 and float(ctl.time_step_dfi) != float(ctl.time_step):
        out.append(f"time_step_dfi={ctl.time_step_dfi} differs from time_step={ctl.time_step:g}; only the "
                   "forecast time step is supported for the DFI stages")
    if ctl.moist_adv_dfi_opt != 0:
        out.append(f"moist_adv_dfi_opt={ctl.moist_adv_dfi_opt}: only the WRF default 0 is supported in the "
                   "backward stage")
    if any(_int(v) != 0 for v in _all_values(nml, "physics", "sst_update") if v is not None):
        out.append("sst_update=1 with DFI is not reproduced (WRF's forward stage switches it off only in the "
                   "namelist record; module_surface_driver still reads the grid copy)")
    if _bool(_get(nml, "bdy_control", "multi_bdy_files", False)):
        out.append("multi_bdy_files=.true. cannot be combined with DFI")
    for key in ("grid_fdda", "grid_sfdda", "obs_nudge_opt"):
        if any(_int(v) != 0 for v in _all_values(nml, "fdda", key) if v is not None):
            out.append(f"&fdda {key}!=0 (nudging) cannot be combined with DFI")
    if any(_int(v) == _TEMF_PBL for v in _all_values(nml, "physics", "bl_pbl_physics") if v is not None):
        out.append("DFI not available for bl_pbl_physics=10 (WRF check_a_mundo)")
    # Verified on Noah-MP / no urban / km_opt != 2 only; the rest would run but not like WRF.
    if any(_int(v) not in (0, _NOAHMP_LSM) for v in _all_values(nml, "physics", "sf_surface_physics")
           if v is not None):
        out.append("DFI land handling (dfi_array_reset snow/soil rule, NOAHMP_INIT at each stage start) is "
                   "implemented for Noah-MP (sf_surface_physics=4) only")
    if any(_int(v) != 0 for v in _all_values(nml, "physics", "sf_urban_physics") if v is not None):
        out.append("sf_urban_physics!=0 with DFI is not verified (urban canopy state through the DFI stages)")
    if any(_int(v) in (2, 5) for v in _all_values(nml, "dynamics", "km_opt") if v is not None):
        out.append("km_opt=2/5 with DFI is not implemented (WRF also filters the prognostic TKE, dfi_tke)")
    if ctl.dfi_opt == DFI_DFL:
        for key, default, supported in (("diff_opt_dfi", 0, (0, 1)), ("km_opt_dfi", 1, (1, 4))):
            value = _int(_get(nml, "dynamics", key, default))
            if value not in supported:
                out.append(f"DFL {key}={value} is not qualified (supported {supported})")
    if ctl.dt_s <= 0:
        out.append("DFI requires a positive time_step")
        return out
    if ctl.dfi_opt != DFI_DFL and ctl.bck_run_s <= 0:
        out.append(f"dfi_bckstop {ctl.bckstop:%Y-%m-%d_%H:%M:%S} must be before the forecast start "
                   f"{ctl.start:%Y-%m-%d_%H:%M:%S}")
        return out
    if ctl.fwd_run_s <= 0:
        out.append(f"dfi_fwdstop {ctl.fwdstop:%Y-%m-%d_%H:%M:%S} must be after the DFI forward start "
                   f"{ctl.fwd_start:%Y-%m-%d_%H:%M:%S}")
        return out
    if ctl.cutoff_s <= 0:
        out.append("dfi_cutoff_seconds must be positive")
    for label, run in (("backward", ctl.bck_run_s), ("forward", ctl.fwd_run_s)):
        if 2 * optfil_nstep2(run, ctl.dt_s) > DFI_TIME_DIM:
            out.append(f"the {label} DFI window ({run} s) needs more than dfi_time_dim={DFI_TIME_DIM} weights")
    interval = float(_get(nml, "time_control", "interval_seconds", 21600) or 21600)
    if ctl.n_fwd * ctl.dt_s - ctl.n_bck * ctl.dt_s > interval:
        out.append("the TDFI forward stage would run past the first lateral-boundary interval")
    return out
