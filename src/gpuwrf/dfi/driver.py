"""WRF twice-DFI (``dfi_opt=3``, TDFI) driver for the native single-domain pipeline.

Reproduces WRF 4.7.1's choreography (main/module_wrf_top.F wrf_dfi, share/dfi.F,
frame/module_integrate.F, dyn_em/solve_em.F, dyn_em/start_em.F) with the operational step:

1. **Backward stage** (``dfi_bck_init``): ``dt = -time_step`` from the forecast start to
   ``dfi_bckstop``; mp/ra/sf/sfclay/urban/pbl/cu/damp/sst/gwd physics off, ``km_opt=1``,
   ``diff_opt=0``, ``moist_adv_opt = moist_adv_dfi_opt`` (``nl_set_diff_6th_opt(0)`` is inert:
   solve_em reads the grid copy, so the 6th-order filter stays on); every odd-order advection
   stencil upwinds with ``sign(1,time_step)`` (gpuwrf.dynamics.time_direction); only QV is transported
   (``solve_em.F:310`` ``num_3d_m = P_QV``, no scalars) -- the other species are frozen and also
   excluded from the dynamics' moisture coupling, so they are zero in the stage's State and
   accumulated at their frozen values.  The lateral boundary is never re-read, so ``dtbc`` runs
   ``-n*|dt|`` and WRF extrapolates the first interval backwards (``bxs + dtbc*btxs``;
   ``constant_bc`` zeroes the tendency).
2. ``dfi_array_reset``: every accumulated field = sum(h*x)/sum(h) in REAL; moist and scalar
   species ``max(0, .)``.
3. **Forward stage** (``dfi_fwd_init``): ``dt = +time_step`` from
   ``dfi_bckstop + (start - dfi_bckstop)/2`` to ``dfi_fwdstop`` with the namelist physics but
   ``sst_update=0``; the domain clock (radiation, Noah-MP, CLWRF gases) is the real stage time,
   while ``med_latbound_in`` returns early so ``dtbc`` continues from the backward end.
   ``start_domain``/``phy_init`` restart the physics (MYNN QKE cold start, Noah-MP NOAHMP_INIT,
   radiation init) from the filtered fields -- realised by loading the stage through the
   validated native init chain from a stage ``wrfinput``.  ``Setup_Timekeeping`` makes the
   stage start the simulation start, so start_em also re-sets W from the surface kinematic
   condition (as at the forecast start, step 5).
4. ``dfi_array_reset`` after the forward stage, plus the land rule for ``dfi_savehydmeteors=0``:
   SNOW/SNOWH/SNOWC/CANWAT = max(stage end, initial); where SNOWC == 0 TSK/TSLB/SMOIS are
   restored to their initial values.
5. **Forecast** (``dfi_fst_init``): start_em re-derives AL/P from the filtered PH/MU/THM,
   re-initialises W from the surface kinematic condition (start of simulation,
   ``use_input_w=.false.``), MYNN cold-starts QKE and NOAHMP_INIT rebuilds the land carry from
   the reset fields (and the stage-end LAI).  The port writes exactly those inputs into
   ``wrfinput_initialized_d01`` (``dfi_write_filtered_input``) and runs the unchanged forecast
   from it.

In-memory WRF carry-overs that do not pass through ``wrfinput`` are not reproduced (history-only
AC*/runoff sums accumulated during the forward stage, the Noah-MP snow age TAUSS); see
``DFI_LIMITATIONS``.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import replace as dc_replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.dfi.control import DfiControl, dfi_control_from_namelist
from gpuwrf.dfi.filter import optfil_hcoeff

__all__ = ["DFI_LIMITATIONS", "MARKER", "apply_dfi_prepass", "run_tdfi"]

MARKER = "gpuwrf_dfi_applied.json"
DFI_LIMITATIONS = (
    "single domain (max_dom=1) native runs only",
    "forward-stage AC*/runoff history sums and the Noah-MP snow age (TAUSS) are not carried into the "
    "forecast (WRF keeps them in memory; they do not pass through wrfinput)",
    "the forecast t=0 QKE is the MYNN cold start (WRF re-initialises it at the first forecast step "
    "anyway, INITIALIZE_QKE=.TRUE.)",
)

_T0 = np.float32(300.0)
_RVOVRD = 461.6 / 287.0
# (State leaf, wrfinput variable) in WRF moist / scalar array order (Registry.EM_COMMON).
_MOIST = (("qv", "QVAPOR"), ("qc", "QCLOUD"), ("qr", "QRAIN"), ("qi", "QICE"), ("qs", "QSNOW"),
          ("qg", "QGRAUP"), ("qh", "QHAIL"))
_SCALAR = (("Ni", "QNICE"), ("Nr", "QNRAIN"), ("Ns", "QNSNOW"), ("Ng", "QNGRAUPEL"), ("Nh", "QNHAIL"),
           ("Nc", "QNCLOUD"), ("Nn", "QNCCN"), ("nwfa", "QNWFA"), ("nifa", "QNIFA"),
           ("qvolg", "QVGRAUPEL"), ("qvolh", "QVHAIL"))
# Boundary leaves derived from the run's initial condition (no wrfbdy record): kept per stage.
_IC_BDY_LEAVES = ("p_bdy", "pb_bdy", "phb_bdy", "mub_bdy")


def _log(report: dict, msg: str) -> None:
    report.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {msg}")


# --------------------------------------------------------------------------- #
# Accumulation (WRF dfi_accumulate / dfi_array_reset, REAL arithmetic)          #
# --------------------------------------------------------------------------- #
def _species(state) -> list[tuple[str, str, bool]]:
    """(leaf, wrf name, is_moist) for every present moist/scalar leaf."""

    out = []
    for leaf, name in _MOIST:
        if getattr(state, leaf, None) is not None:
            out.append((leaf, name, True))
    for leaf, name in _SCALAR:
        if getattr(state, leaf, None) is not None:
            out.append((leaf, name, False))
    return out


def _sample(state, frozen: dict | None = None) -> dict:
    import jax.numpy as jnp

    f32 = jnp.float32
    out = {
        "u": jnp.asarray(state.u).astype(f32),
        "v": jnp.asarray(state.v).astype(f32),
        "w": jnp.asarray(state.w).astype(f32),
        # WRF t_2 is the REAL theta_m perturbation; theta in [150, 600] K makes the
        # subtraction exact in binary32 (Sterbenz), as the prognostic storage is.
        "t_2": (jnp.asarray(state.theta).astype(f32) - f32(300.0)).astype(f32),
        "ph": jnp.asarray(state.ph_perturbation).astype(f32),
        "mu": jnp.asarray(state.mu_perturbation).astype(f32),
    }
    for leaf, _name, _moist in _species(state):
        value = (frozen or {}).get(leaf)
        out[leaf] = jnp.asarray(getattr(state, leaf) if value is None else value).astype(f32)
    return out


def _accumulate_fn():
    import jax

    @jax.jit
    def accumulate(acc, sample, h):
        # grid%dfi_x = grid%dfi_x + grid%x * hn  (share/dfi.F:26-53)
        return {k: acc[k] + sample[k] * h for k in acc}

    return accumulate


def _reset(acc: dict, tot: np.float32, species: list[tuple[str, str, bool]]) -> dict[str, np.ndarray]:
    """``dfi_array_reset``: x = acc/hcoeff_tot; moist/scalar = max(0, acc/tot) (share/dfi.F:598-618)."""

    out = {}
    clipped = {leaf for leaf, _n, _m in species}
    for key, value in acc.items():
        x = (np.asarray(value, dtype=np.float32) / np.float32(tot)).astype(np.float32)
        out[key] = np.maximum(np.float32(0.0), x) if key in clipped else x
    return out


# --------------------------------------------------------------------------- #
# Boundary records for the DFI stages                                           #
# --------------------------------------------------------------------------- #
def _shifted_boundary(state, *, shift_s: float, cadence_s: float, constant_bc: bool, ic_source=None) -> dict:
    """2-record leaves L with L(x) = B(x + shift_s), B = WRF's first-interval line.

    WRF evaluates the specified boundary as ``bxs + dtbc*btxs`` with the first record's tendency,
    for every ``dtbc`` (also negative: backward extrapolation).  The operational interpolation
    clamps the lead to ``[0, last record]``, so each DFI stage gets the line re-anchored at its
    own lead origin; ``constant_bc`` zeroes the tendency (``zero_bdytend``).  The record-constant
    leaves derived from the initial condition (``p_bdy``/``pb_bdy``/``phb_bdy``/``mub_bdy``) keep their
    values, taken from ``ic_source`` (the stage's own initial state; default ``state``), trimmed to the
    same two records so record-wise combinations stay shape-consistent.
    """

    import jax.numpy as jnp

    out = {}
    source = state if ic_source is None else ic_source
    for name in _IC_BDY_LEAVES:
        leaf = getattr(source, name, None)
        if leaf is not None and getattr(leaf, "ndim", 0) >= 2 and int(leaf.shape[0]) >= 2:
            out[name] = jnp.asarray(leaf)[:2]
    for name in state.__slots__:
        if not name.endswith("_bdy") or name in _IC_BDY_LEAVES:
            continue
        leaf = getattr(state, name)
        if leaf is None or getattr(leaf, "ndim", 0) < 2 or int(leaf.shape[0]) < 2:
            continue
        r0 = np.asarray(leaf[0], dtype=np.float64)
        r1 = np.asarray(leaf[1], dtype=np.float64)
        slope = np.zeros_like(r0) if constant_bc else (r1 - r0) / float(cadence_s)
        q0 = r0 + slope * float(shift_s)
        q1 = r0 + slope * (float(cadence_s) + float(shift_s))
        out[name] = jnp.asarray(np.stack([q0, q1]).astype(np.dtype(leaf.dtype)))
    return out


# --------------------------------------------------------------------------- #
# start_em first-trip pressure (dyn_em/start_em.F:776-850, hypsometric_opt=2)    #
# --------------------------------------------------------------------------- #
_START_EM_INPUTS = ("PB", "PHB", "MUB", "C3H", "C4H", "C3F", "C4F", "P_TOP", "P00", "T00", "TLP", "TISO",
                    "TLP_STRAT", "P_STRAT")


def start_em_pressure(*, ph, mu, t_2, base: dict, libm) -> dict[str, np.ndarray]:
    """AL/ALT/P/P_HYD that start_em's first trip re-derives from ph_1/mu_1/t_1 (REAL arithmetic).

    dyn_em/start_em.F:776-850 with hypsometric_opt=2 and use_theta_m=1 (qvf=1); ``alb`` is
    start_em's reference profile from PB (:281-287).  ``base`` holds the wrfinput variables
    ``_START_EM_INPUTS``; ``libm`` the WRF build's float32 ``log``/``pow``.
    """

    from gpuwrf.integration import d02_replay as dr

    f32 = np.float32
    one = lambda name: f32(np.asarray(base[name], dtype=np.float32).ravel()[0])  # noqa: E731
    pb, phb, mub = (np.asarray(base[n], dtype=np.float32) for n in ("PB", "PHB", "MUB"))
    c3h, c4h, c3f, c4f = (np.asarray(base[n], dtype=np.float32).ravel() for n in ("C3H", "C4H", "C3F", "C4F"))
    p_top = one("P_TOP")
    scalars = {"p_top": p_top, "p00": one("P00"), "t00": one("T00"), "a": one("TLP"), "tiso": one("TISO"),
               "a_strat": one("TLP_STRAT"), "p_strat": one("P_STRAT")}
    m = libm
    _t_init, alb = dr._wrf_start_domain_t_init_alb32(pb, scalars)
    ph = np.asarray(ph, f32)
    dph = (ph[1:] - ph[:-1] + phb[1:] - phb[:-1]).astype(f32)
    full_mu = (mub + np.asarray(mu, f32)).astype(f32)
    pfu = (c3f[1:, None, None] * full_mu[None] + c4f[1:, None, None] + p_top).astype(f32)
    pfd = (c3f[:-1, None, None] * full_mu[None] + c4f[:-1, None, None] + p_top).astype(f32)
    phm = (c3h[:, None, None] * full_mu[None] + c4h[:, None, None] + p_top).astype(f32)
    al = (dph / phm / m.log((pfd / pfu).astype(f32)) - alb).astype(f32)
    alt = (al + alb).astype(f32)
    theta_eos = (dr._WRF32_T0 + np.asarray(t_2, f32)).astype(f32)
    ratio = ((dr._WRF32_R_D * theta_eos) / (dr._WRF32_P1000MB * alt)).astype(f32)
    p = (dr._WRF32_P1000MB * m.pow(ratio, dr._WRF32_CPOVCV) - pb).astype(f32)
    return {"AL": al, "ALT": alt, "P": p, "P_HYD": (p + pb).astype(f32)}


def _start_em_pressure(run, domain: str, *, ph, mu, t_2) -> dict[str, np.ndarray]:
    from gpuwrf.integration import d02_replay as dr
    from gpuwrf.io.netcdf_lock import Dataset

    with Dataset(run.wrfinput_file(domain), "r") as ds:
        base = {name: np.asarray(ds.variables[name][0]) for name in _START_EM_INPUTS}
    return start_em_pressure(ph=ph, mu=mu, t_2=t_2, base=base, libm=dr._WRF_INIT_LIBM32)


# --------------------------------------------------------------------------- #
# Stage input directories                                                       #
# --------------------------------------------------------------------------- #
def _edit_namelist(text: str, edits: dict[str, Any]) -> str:
    for key, value in edits.items():
        pat = re.compile(rf"^(\s*){key}\s*=.*$", re.M | re.I)
        if pat.search(text):
            text = pat.sub(lambda mt, v=value, k=key: f"{mt.group(1)}{k} = {v},", text, count=1)
        else:
            group = "time_control" if key.startswith("start_") else "physics"
            text = re.sub(rf"(&{group}\s*\n)", lambda mt, k=key, v=value: mt.group(1) + f" {k} = {v},\n",
                          text, count=1, flags=re.I)
    return text


def _stamp(dt_: datetime) -> str:
    return dt_.strftime("%Y-%m-%d_%H:%M:%S")


def _write_stage_dir(src: Path, dst: Path, *, fields: dict[str, np.ndarray], when: datetime | None,
                     namelist_edits: dict[str, Any] | None, marker: dict | None = None) -> Path:
    """dst = symlinks to src (except wrfinput_d01 [copied + fields] and namelist.input [edited])."""

    from gpuwrf.io.netcdf_lock import Dataset

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for entry in src.iterdir():
        if entry.name in {"wrfinput_d01", "namelist.input", MARKER}:
            continue
        (dst / entry.name).symlink_to(entry.resolve())
    nml_text = (src / "namelist.input").read_text()
    if namelist_edits:
        nml_text = _edit_namelist(nml_text, namelist_edits)
    (dst / "namelist.input").write_text(nml_text)
    target = dst / "wrfinput_d01"
    shutil.copyfile(src / "wrfinput_d01", target)
    with Dataset(target, "r+") as ds:
        for name, value in fields.items():
            if name not in ds.variables:
                continue
            var = ds.variables[name]
            arr = np.asarray(value, dtype=var.dtype)
            var[0] = arr.reshape(var.shape[1:])
        if when is not None:
            label = _stamp(when)
            ds.variables["Times"][0] = np.array(list(label), dtype="S1")
            ds.setncattr("START_DATE", label)
    if marker is not None:
        (dst / MARKER).write_text(json.dumps(marker, indent=1, default=str) + "\n")
    return dst


def _kinematic_w(filt: dict[str, np.ndarray], bundle) -> np.ndarray:
    """W of a start_em at the simulation start (``use_input_w=.false.``): set_w_surface fill.

    WRF re-sets W at the forward-stage start too: ``dfi_fwd_init`` -> ``Setup_Timekeeping`` sets
    ``simulation_start`` to the stage start, so ``start_of_simulation`` is true in start_em and
    ``w_needs_to_be_set`` holds for any input W (start_em.F:1475-1530).  [M] CPU-WRF history at the
    TDFI forward start: W == W_sfc * znw**2 to 3.4e-8 relative.
    """

    from gpuwrf.integration.d02_replay import _wrf_set_w_surface

    w, _set, _w_in = _wrf_set_w_surface(np.zeros_like(filt["w"]), filt["u"], filt["v"],
                                        grid=bundle.grid, metrics=bundle.metrics)
    return np.asarray(w, dtype=np.float32)


def _prognostic_fields(run, domain: str, filt: dict[str, np.ndarray], species, *, bundle) -> dict:
    """wrfinput variables for a stage start (forward stage, forecast) from the filtered fields."""

    fields: dict[str, np.ndarray] = {
        "U": filt["u"], "V": filt["v"], "PH": filt["ph"], "MU": filt["mu"],
        "W": _kinematic_w(filt, bundle),
        "THM": filt["t_2"],
        # wrfinput T is the DRY perturbation theta the loader re-couples with QVAPOR.
        "T": ((filt["t_2"].astype(np.float64) + 300.0) / (1.0 + _RVOVRD * filt["qv"].astype(np.float64))
              - 300.0).astype(np.float32),
        "QKE": np.zeros_like(filt["t_2"]),  # MYNN cold start at the stage's first step
    }
    for leaf, name, _moist in species:
        fields[name] = filt[leaf]
    fields.update(_start_em_pressure(run, domain, ph=filt["ph"], mu=filt["mu"], t_2=filt["t_2"]))
    return fields


# --------------------------------------------------------------------------- #
# Land (dfi_save_arrays / dfi_array_reset, dfi_savehydmeteors=0)                #
# --------------------------------------------------------------------------- #
def _land(carry) -> dict[str, np.ndarray] | None:
    land = getattr(carry, "noahmp_land", None)
    if land is None:
        return None
    f = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    out = {
        "TSK": f(carry.state.t_skin), "TSLB": f(land.tslb), "SMOIS": f(land.smois),
        "SNOW": f(land.sneqv), "SNOWH": f(land.snowh), "CANWAT": f(land.canliq) + f(land.canice),
        "LAI": f(land.lai),
    }
    hist = getattr(carry, "land_history", None)
    snowc = None
    if hist is not None:
        try:
            snowc = f(hist["SNOWC"])
        except Exception:  # noqa: BLE001 - optional history container
            snowc = None
    out["SNOWC"] = snowc if snowc is not None else np.where(out["SNOW"] > 0.0, 1.0, 0.0).astype(np.float32)
    return out


def _land_reset(end: dict, saved: dict) -> dict[str, np.ndarray]:
    """share/dfi.F:654-690 (savehydmeteors=0)."""

    out = {k: np.maximum(end[k], saved[k]).astype(np.float32) for k in ("SNOW", "SNOWH", "SNOWC", "CANWAT")}
    bare = out["SNOWC"] == 0.0
    out["TSK"] = np.where(bare, saved["TSK"], end["TSK"]).astype(np.float32)
    out["TSLB"] = np.where(bare[None], saved["TSLB"], end["TSLB"]).astype(np.float32)
    out["SMOIS"] = np.where(bare[None], saved["SMOIS"], end["SMOIS"]).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                        #
# --------------------------------------------------------------------------- #
def _stage_loop(carry, namelist, *, n_steps: int, start_step: int, hcoeff: np.ndarray,
                frozen: dict | None, report: dict, label: str):
    import jax
    import jax.numpy as jnp

    from gpuwrf.runtime import operational_mode as om

    accumulate = _accumulate_fn()
    clock = om.build_clock_base(namelist)
    cadence = int(namelist.radiation_cadence_steps)
    species = _species(carry.state)
    sample = _sample(carry.state, frozen)
    h0 = np.float32(hcoeff[0])
    acc = accumulate({k: jnp.zeros_like(v) for k, v in sample.items()}, sample, jnp.float32(h0))
    tot = h0
    t0 = time.perf_counter()
    for n in range(1, n_steps + 1):
        carry = om._advance_chunk(carry, namelist, start_step + n - 1, clock, n_steps=1, cadence=cadence)
        hn = np.float32(hcoeff[n]) if n < len(hcoeff) else np.float32(0.0)
        acc = accumulate(acc, _sample(carry.state, frozen), jnp.float32(hn))
        tot = np.float32(tot + hn)  # grid%hcoeff_tot = grid%hcoeff_tot + hn (REAL)
        if n == 1:
            jax.block_until_ready(acc)
            _log(report, f"{label}: first step (incl. compile) {time.perf_counter() - t0:.1f} s")
    jax.block_until_ready(acc)
    report[f"{label}_steps"] = int(n_steps)
    report[f"{label}_wall_s"] = round(time.perf_counter() - t0, 2)
    report[f"{label}_hcoeff_tot"] = float(tot)
    acc = {k: np.asarray(v) for k, v in acc.items()}
    nonfinite = int(sum(int((~np.isfinite(v)).sum()) for v in acc.values()))
    report[f"{label}_nonfinite"] = nonfinite
    if nonfinite:
        raise FloatingPointError(f"DFI {label} stage produced {nonfinite} non-finite accumulated values")
    return carry, _reset(acc, tot, species), species


def run_tdfi(config, *, work_dir: Path, nml: dict | None = None, fst_dir: Path | None = None) -> tuple[Path, dict]:
    """Run TDFI for ``config`` (NestedPipelineConfig, max_dom=1); return (filtered input dir, report)."""

    import jax.numpy as jnp

    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.kernels.dyn_carry_fp32 import real_carry

    src = Path(config.input_dir)
    run = Gen2Run(src)
    nml = run.namelist if nml is None else nml
    ctl: DfiControl = dfi_control_from_namelist(nml)
    report: dict[str, Any] = {
        "schema": "gpuwrf.dfi.v1", "dfi_opt": ctl.dfi_opt, "dfi_nfilter": ctl.nfilter,
        "dfi_cutoff_seconds": ctl.cutoff_s, "start": _stamp(ctl.start), "dfi_bckstop": _stamp(ctl.bckstop),
        "dfi_fwdstop": _stamp(ctl.fwdstop), "fwd_start": _stamp(ctl.fwd_start), "dt_s": ctl.dt_s,
        "constant_bc": ctl.constant_bc, "limitations": list(DFI_LIMITATIONS),
    }
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    names = ("d01",)
    dt = float(ctl.dt_s)

    # ---- initial load (WRF wrf_init start_domain + phy_init) and dfi_save_arrays ----
    _h, bundles, _meta, _run_start, _dt, carries = pipeline._load_domains(config, names)
    bundle, carry0 = bundles["d01"], carries["d01"]
    ns = bundle.namelist
    if abs(float(ns.dt_s) - dt) > 1e-9:
        raise ValueError(f"DFI time step {dt} differs from the loaded domain dt {ns.dt_s}")
    saved_land = _land(carry0)
    if saved_land is not None:
        from gpuwrf.io.netcdf_lock import Dataset
        with Dataset(src / "wrfinput_d01", "r") as ds:
            if "SNOWC" in ds.variables:  # grid%SNOWC at the backward start is the input value
                saved_land["SNOWC"] = np.asarray(ds.variables["SNOWC"][0], dtype=np.float32)
    cadence_s = float(ns.boundary_config.update_cadence_s)
    report["boundary_cadence_s"] = cadence_s

    # ---- backward stage ----
    n_bck = ctl.n_bck
    state0 = carry0.state
    species = _species(state0)
    frozen = {leaf: getattr(state0, leaf) for leaf, _n, _m in species if leaf != "qv"}
    st_bck = state0.replace(**{leaf: jnp.zeros_like(getattr(state0, leaf)) for leaf in frozen})
    # solve_em.F:310: num_3d_m = P_QV -> no boundary or transport work for the frozen species.
    st_bck = st_bck.replace(**{f"{leaf}_bdy": jnp.zeros_like(getattr(state0, f"{leaf}_bdy")) for leaf in frozen
                               if getattr(state0, f"{leaf}_bdy", None) is not None})
    # lead x = (n_bck + 1 - n)*|dt| >= |dt| keeps every RK-stage lead (x + dt_rk) >= 0.
    shifted_b = _shifted_boundary(state0, shift_s=-(n_bck + 1) * dt, cadence_s=cadence_s,
                                  constant_bc=ctl.constant_bc)
    st_bck = st_bck.replace(**{k: v for k, v in shifted_b.items() if k[:-4] not in frozen})
    sound = pipeline._wrf_sound_steps(ns.grid, -dt, int(pipeline._domain_int(run, "dynamics",
                                                                            "time_step_sound", "d01", 0)))
    # dfi_bck_init's nl_set_* switches reach solve_em through config_flags EXCEPT diff_6th_opt and
    # diff_6th_factor, which solve_em passes from the grid copies (solve_em.F:899, :2315, ...) that
    # nl_set_diff_6th_opt never touches: WRF's backward stage keeps the namelist 6th-order filter.
    nl_bck = dc_replace(ns, dt_s=-dt, run_physics=False, diff_opt=0, km_opt=1, damp_opt=0,
                        moist_adv_opt=int(ctl.moist_adv_dfi_opt), lower_boundary=None,
                        use_noahmp=False, gwd_opt=0, cu_physics=0, acoustic_substeps=int(sound))
    carry_b = om._commit_to_operational_device(real_carry(om._initial_carry_for_run(st_bck, nl_bck)))
    h_bck = optfil_hcoeff(ctl.bck_run_s, -dt, ctl.cutoff_s, ctl.nfilter)
    report["bck_hcoeff"] = [float(x) for x in h_bck]
    report["bck_acoustic_substeps"] = int(sound)
    _log(report, f"backward stage: {n_bck} steps of {-dt:g} s")
    _carry_end, filt_b, _sp = _stage_loop(carry_b, nl_bck, n_steps=n_bck, start_step=-n_bck, hcoeff=h_bck,
                                          frozen=frozen, report=report, label="bck")
    del _carry_end, carry_b

    # ---- forward stage (start_domain/phy_init on the backward-filtered state) ----
    fwd_fields = _prognostic_fields(run, "d01", filt_b, species, bundle=bundle)
    fwd_dir = _write_stage_dir(src, work_dir / "fwd_input", fields=fwd_fields, when=ctl.fwd_start,
                               namelist_edits={
                                   "start_year": ctl.fwd_start.year, "start_month": ctl.fwd_start.month,
                                   "start_day": ctl.fwd_start.day, "start_hour": ctl.fwd_start.hour,
                                   "start_minute": ctl.fwd_start.minute, "start_second": ctl.fwd_start.second,
                                   "sst_update": 0, "dfi_opt": 0})
    cfg_fwd = dc_replace(config, input_dir=fwd_dir)
    _h, bundles_f, _meta, run_start_f, _dt, carries_f = pipeline._load_domains(cfg_fwd, names)
    report["fwd_clock_start"] = str(run_start_f)
    nl_fwd = bundles_f["d01"].namelist
    carry_f = carries_f["d01"]
    shifted = _shifted_boundary(state0, shift_s=-n_bck * dt, cadence_s=cadence_s, constant_bc=ctl.constant_bc,
                                ic_source=carry_f.state)
    shifted = {k: v.astype(getattr(carry_f.state, k).dtype) for k, v in shifted.items()}
    carry_f = om._commit_to_operational_device(carry_f.replace(state=carry_f.state.replace(**shifted)))
    h_fwd = optfil_hcoeff(ctl.fwd_run_s, dt, ctl.cutoff_s, ctl.nfilter)
    report["fwd_hcoeff"] = [float(x) for x in h_fwd]
    _log(report, f"forward stage: {ctl.n_fwd} steps of {dt:g} s from {_stamp(ctl.fwd_start)}")
    carry_end, filt_f, _sp = _stage_loop(carry_f, nl_fwd, n_steps=ctl.n_fwd, start_step=1, hcoeff=h_fwd,
                                         frozen=None, report=report, label="fwd")
    shutil.rmtree(fwd_dir, ignore_errors=True)

    # ---- forecast input (dfi_fst_init) ----
    # dfi_fst_init's start_domain is at the simulation start: W from the filtered winds.
    fst_fields = _prognostic_fields(run, "d01", filt_f, species, bundle=bundles_f["d01"])
    end_land = _land(carry_end)
    if end_land is not None and saved_land is not None:
        reset = _land_reset(end_land, saved_land)
        fst_fields.update(reset)
        fst_fields["CANWAT"] = np.zeros_like(reset["CANWAT"])  # NOAHMP_INIT: CANWAT = 0
        fst_fields["LAI"] = end_land["LAI"]
        report["land_reset_cells_restored"] = int((reset["SNOWC"] == 0.0).sum())
    diag = getattr(carry_end, "history_diagnostics", None)
    if diag is not None:
        for name, leaf in (("T2", "t2"), ("Q2", "q2"), ("U10", "u10"), ("V10", "v10"), ("PSFC", "psfc")):
            value = getattr(diag, leaf, None)
            if value is not None:
                fst_fields[name] = np.asarray(value, dtype=np.float32)
    marker = {k: v for k, v in report.items() if k not in {"bck_hcoeff", "fwd_hcoeff", "log"}}
    fst_dir = _write_stage_dir(src, Path(fst_dir) if fst_dir is not None else work_dir / "fst_input",
                               fields=fst_fields, when=None, namelist_edits=None, marker=marker)
    report["filtered_input_dir"] = str(fst_dir)
    return fst_dir, report


def _batch_size() -> int:
    import os

    raw = os.environ.get("GPUWRF_BATCH_ENSEMBLE", "").strip()
    return int(raw) if raw.isdigit() else 1


def apply_dfi_prepass(config) -> tuple[Any, dict | None]:
    """Run TDFI before a native forecast when ``&dfi_control`` asks for it.

    Returns ``(config, report)``: ``config.input_dir`` points at the filtered input
    (WRF ``wrfinput_initialized_d01`` semantics, persisted as ``<output_dir>/dfi_input``) when DFI
    ran, otherwise ``config`` is unchanged and ``report`` is None.  A resumed run never filters
    (WRF: restart => dfi_opt=0): it continues from the persisted filtered input, which the restart
    identity (input dir + wrfinput/namelist hashes) was bound to.  An input directory written by
    this pre-pass (``MARKER``) is not filtered twice.
    """

    from gpuwrf.dfi.control import dfi_refusals
    from gpuwrf.io.namelist_check import _coerce_config

    src = Path(config.input_dir)
    namelist_path = src / "namelist.input"
    if (src / MARKER).exists() or not namelist_path.exists():
        return _dfl_forecast_config(config, src), None
    nml = _coerce_config(namelist_path)
    ctl = dfi_control_from_namelist(nml)
    persisted = Path(config.output_dir) / "dfi_input"
    if not ctl.active:
        return config, None
    if config.resume_checkpoint is not None:
        if (persisted / MARKER).exists():
            return _dfl_forecast_config(dc_replace(config, input_dir=persisted), persisted), None
        return config, None
    reasons = dfi_refusals(nml, max_dom=int(config.max_dom))
    if _batch_size() != 1:
        reasons.append("GPUWRF_BATCH_ENSEMBLE batches cannot run DFI")
    if reasons:
        raise ValueError("&dfi_control: " + "; ".join(reasons))
    base = Path(config.scratch_dir) if config.scratch_dir is not None else Path(config.output_dir)
    work = base / "dfi_tdfi"
    t0 = time.perf_counter()
    if ctl.dfi_opt in (1, 2):
        from gpuwrf.dfi.driver12 import run_dfi12

        fst_dir, report = run_dfi12(config, work_dir=work, nml=nml, fst_dir=persisted)
    else:
        fst_dir, report = run_tdfi(config, work_dir=work, nml=nml, fst_dir=persisted)
    report["prepass_wall_s"] = round(time.perf_counter() - t0, 2)
    shutil.rmtree(work, ignore_errors=True)  # stage scratch (the forward-stage input copy)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    if ctl.write_filtered_input:
        target = Path(config.output_dir) / "wrfinput_initialized_d01"
        shutil.copyfile(fst_dir / "wrfinput_d01", target)
        report["wrfinput_initialized"] = str(target)
    Path(config.proof_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.proof_dir) / "dfi_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    return _dfl_forecast_config(dc_replace(config, input_dir=fst_dir), fst_dir), report


def _dfl_forecast_config(config, directory: Path):
    """DFL starts later while retaining the requested original forecast end, also on resume."""
    marker = directory / MARKER
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("dfi_opt") == 1:
            hours = float(config.hours) - float(saved["forecast_offset_s"]) / 3600
            if hours <= 0:
                raise ValueError("DFL forecast end must be after its forward-window midpoint")
            if saved.get("dt_s", 0) > 0:
                dt = float(saved["dt_s"])
                hours = math.ceil(hours * 3600 / dt) * dt / 3600
            return dc_replace(config, hours=hours)
    return config
