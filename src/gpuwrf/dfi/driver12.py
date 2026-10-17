"""WRF DFL/DDFI phase drivers; the previously qualified TDFI driver is retained.

DFL filters forward from the input time and launches at its integer midpoint.
DDFI integrates adiabatically backward, then filters forward from the UNFILTERED
endpoint. Shared numerical building blocks live in :mod:`gpuwrf.dfi.driver`.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from gpuwrf.dfi.control import DFI_DDFI, DFI_DFL, _get, dfi_control_from_namelist
from gpuwrf.dfi import driver as d
from gpuwrf.dfi.filter import optfil_hcoeff


def _clock_edits(when):
    return {f"start_{unit}": getattr(when, unit) for unit in
            ("year", "month", "day", "hour", "minute", "second")}


def _dfl_diffusion(nml):
    # Unlike DDFI/TDFI, DFL never calls bck_init to save the original options.
    # fwd_init still restores these Registry fields, defaulting to 0 and 1.
    return {key: int(_get(nml, "dynamics", f"{key}_dfi", default))
            for key, default in (("diff_opt", 0), ("km_opt", 1))}


def _write_dfl_stage_dir(src, dst, *, fields, when, namelist_edits):
    """Restore DFL diffusion in &dynamics, including omitted forecast keys.

    Keep the shared TDFI writer unchanged; its absent-key fallback is &physics.
    Other stage edits and already-present diffusion lines retain the same bytes.
    """
    edits = dict(namelist_edits)
    diffusion = {key: edits.pop(key) for key in ("diff_opt", "km_opt") if key in edits}
    stage = d._write_stage_dir(src, dst, fields=fields, when=when, namelist_edits=edits)
    text = (stage / "namelist.input").read_text()
    if not re.search(r"&dynamics\b", text, re.I):
        text += "\n&dynamics\n/\n"
    for key, value in diffusion.items():
        group = re.search(r"(&dynamics\b)(.*?)(^\s*/)", text, re.I | re.S | re.M)
        body = group.group(2)
        if re.search(rf"^\s*{key}\s*=", body, re.M | re.I):
            body = d._edit_namelist(body, {key: value})
        else:
            body = f"\n {key} = {value}," + body
        text = text[:group.start(2)] + body + text[group.end(2):]
    (stage / "namelist.input").write_text(text)
    return stage


def _dfl_preserves_w(nml):
    # set_timekeeping.F:225-237 only re-fetches the original start components
    # when a positive run length is supplied; an end-date-only run keeps midpoint
    # components and therefore start_of_simulation=true at the DFL launch.
    return any(float(_get(nml, "time_control", key, 0)) > 0
               for key in ("run_days", "run_hours", "run_minutes", "run_seconds"))


def _launch_fields(run, ctl, filt, species, bundle, *, preserve_filtered_w):
    fields = d._prognostic_fields(run, "d01", filt, species, bundle=bundle)
    if ctl.dfi_opt == DFI_DFL and preserve_filtered_w:
        # set_timekeeping.F:225-237 fetches the ORIGINAL start into start_year/...,
        # used by nl_set_simulation_start_* at :382. Thus the midpoint forecast
        # is NOT start_of_simulation and start_em.F:1475 does not re-set W.
        fields["W"] = np.asarray(filt["w"], np.float32)
    return fields


def _radar_field(path):
    """Complete WRF's initialized-file schema for non-radar DFI (Registry :907).

    dfi_tten_rad is zero-initialized (tools/gen_allocs.c + get_initial_data_value)
    and unused without WRF_DFI_RADAR. Nonzero radar inputs are outside this port.
    """
    from gpuwrf.io.netcdf_lock import Dataset
    with Dataset(path, "r+") as ds:
        if "RAD_TTEN_DFI" in ds.variables:
            if np.any(np.asarray(ds["RAD_TTEN_DFI"][:]) != 0):
                raise ValueError("Nonzero RAD_TTEN_DFI (radar DFI) is not implemented")
            return
        var = ds.createVariable("RAD_TTEN_DFI", "f4", ds["QVAPOR"].dimensions)
        var.setncatts(dict(FieldType=np.int32(104), MemoryOrder="XYZ", description="RADAR POT. TEMP. TENDENCY",
                          units="K s-1", stagger="", coordinates="XLONG XLAT XTIME"))
        var[:] = np.zeros(ds["QVAPOR"].shape, np.float32)


def _backward_endpoint(carry0, ns, ctl, run, pipeline, report):
    import jax
    import jax.numpy as jnp
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.kernels.dyn_carry_fp32 import real_carry

    state0 = carry0.state
    species = d._species(state0)
    frozen = {leaf: getattr(state0, leaf) for leaf, _n, _m in species if leaf != "qv"}
    st = state0.replace(**{leaf: jnp.zeros_like(value) for leaf, value in frozen.items()})
    st = st.replace(**{f"{leaf}_bdy": jnp.zeros_like(getattr(state0, f"{leaf}_bdy"))
                      for leaf in frozen if getattr(state0, f"{leaf}_bdy", None) is not None})
    shifted = d._shifted_boundary(state0, shift_s=-(ctl.n_bck + 1) * ctl.dt_s,
                                  cadence_s=ns.boundary_config.update_cadence_s,
                                  constant_bc=ctl.constant_bc)
    st = st.replace(**{k: v for k, v in shifted.items() if k[:-4] not in frozen})
    sound = pipeline._wrf_sound_steps(ns.grid, -ctl.dt_s,
                                     int(pipeline._domain_int(run, "dynamics", "time_step_sound", "d01", 0)))
    nl = replace(ns, dt_s=-ctl.dt_s, run_physics=False, diff_opt=0, km_opt=1, damp_opt=0,
                 moist_adv_opt=ctl.moist_adv_dfi_opt, lower_boundary=None, use_noahmp=False,
                 gwd_opt=0, cu_physics=0, acoustic_substeps=int(sound))
    carry = om._commit_to_operational_device(real_carry(om._initial_carry_for_run(st, nl)))
    clock = om.build_clock_base(nl)
    t0 = time.perf_counter()
    for n in range(ctl.n_bck):
        carry = om._advance_chunk(carry, nl, n - ctl.n_bck, clock, n_steps=1,
                                  cadence=int(nl.radiation_cadence_steps))
    jax.block_until_ready(carry)
    # DDFI has no dfi_array_reset here: NO weights, division, or species floor.
    endpoint = {k: np.asarray(v) for k, v in d._sample(carry.state, frozen).items()}
    nonfinite = sum(int((~np.isfinite(v)).sum()) for v in endpoint.values())
    report.update(bck_steps=ctl.n_bck, bck_wall_s=round(time.perf_counter() - t0, 2),
                  bck_nonfinite=nonfinite, bck_filtered=False, bck_acoustic_substeps=int(sound))
    if nonfinite:
        raise FloatingPointError(f"DFI backward endpoint has {nonfinite} non-finite values")
    return endpoint, species


def rebase_dfl_boundary(path: Path, *, offset_s: int, forecast_s: float, interval_s: int) -> int:
    """Re-anchor coupled WRF strips at the DFL launch time, retaining every knot.

    The native loader indexes equal-duration records relative to wrfinput time.
    A gcd grid includes every original interval boundary and the shifted launch;
    resampling each coupled strip with its original BT tendency preserves WRF's
    piecewise linear boundary and does not extrapolate across an interval change.
    Only records required by this forecast are written. No JAX/device work.
    """
    from datetime import datetime, timedelta
    from gpuwrf.io.netcdf_lock import Dataset

    stride = math.gcd(int(interval_s), int(offset_s))
    count = int(math.ceil(float(forecast_s) / stride))
    if count > 4096:
        raise ValueError("DFL boundary re-anchoring needs more than 4096 records; use a midpoint "
                         "with a larger common divisor with interval_seconds")
    original = path.resolve()
    tmp = path.with_name("wrfbdy_d01.rebased")
    with Dataset(original, "r") as src, Dataset(tmp, "w", format=src.data_model) as dst:
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if name == "Time" else len(dim))
        nsrc = len(src.dimensions["Time"])
        if offset_s + count * stride > nsrc * interval_s:
            raise ValueError("DFL forecast exceeds its original lateral-boundary coverage")
        times = np.asarray(src.variables["Times"][:])
        first = datetime.strptime(b"".join(times[0].tolist()).decode(), "%Y-%m-%d_%H:%M:%S")
        for name, var in src.variables.items():
            attrs = {k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"}
            kw = {"fill_value": var.getncattr("_FillValue")} if "_FillValue" in var.ncattrs() else {}
            out = dst.createVariable(name, var.datatype, var.dimensions, **kw)
            out.setncatts(attrs)
            if not var.dimensions or var.dimensions[0] != "Time":
                out[:] = var[:]
                continue
            for k in range(count):
                lead = offset_s + k * stride
                record, advance = divmod(lead, interval_s)
                if name == "Times" or name.startswith("md___") and "bdytime" in name:
                    delta = stride if "nextbdytime" in name else 0
                    stamp = (first + timedelta(seconds=lead + delta)).strftime("%Y-%m-%d_%H:%M:%S")
                    out[k] = np.array(list(stamp), dtype="S1")
                elif "_B" in name and "_BT" not in name:
                    tend_name = name.replace("_B", "_BT", 1)
                    if tend_name in src.variables:
                        # WRF REAL bxs + dtbc*btxs, in the COUPLED representation.
                        out[k] = (np.asarray(var[record], np.float32) + np.float32(advance) *
                                  np.asarray(src.variables[tend_name][record], np.float32)).astype(np.float32)
                    else:
                        out[k] = var[record]
                else:
                    out[k] = var[record]
    path.unlink()  # stage symlink, never the original boundary
    tmp.rename(path)
    return stride


def run_dfi12(config, *, work_dir: Path, nml=None, fst_dir: Path | None = None):
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.runtime import operational_mode as om

    src = Path(config.input_dir)
    run = Gen2Run(src)
    nml = run.namelist if nml is None else nml
    ctl = dfi_control_from_namelist(nml)
    if ctl.dfi_opt not in (DFI_DFL, DFI_DDFI):
        raise ValueError("run_dfi12 requires dfi_opt=1 or 2")
    with Dataset(src / "wrfinput_d01", "r") as ds:
        if "RAD_TTEN_DFI" in ds.variables and np.any(np.asarray(ds["RAD_TTEN_DFI"][:]) != 0):
            raise ValueError("Nonzero RAD_TTEN_DFI (radar DFI) is not implemented")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report = dict(schema="gpuwrf.dfi.v1", dfi_opt=ctl.dfi_opt, dfi_nfilter=ctl.nfilter,
                  dfi_cutoff_seconds=ctl.cutoff_s, start=d._stamp(ctl.start),
                  dfi_bckstop=d._stamp(ctl.bckstop), dfi_fwdstop=d._stamp(ctl.fwdstop),
                  fwd_start=d._stamp(ctl.fwd_start), forecast_start=d._stamp(ctl.forecast_start),
                  dt_s=ctl.dt_s, constant_bc=ctl.constant_bc, limitations=list(d.DFI_LIMITATIONS))
    preserve_w = ctl.dfi_opt == DFI_DFL and _dfl_preserves_w(nml)
    report["dfl_preserve_filtered_w"] = preserve_w
    offset = int((ctl.forecast_start - ctl.start).total_seconds())
    forecast_s = float(config.hours) * 3600 - offset
    if forecast_s <= 0:
        raise ValueError("DFL forecast end must follow its midpoint")
    if ctl.dfi_opt == DFI_DFL:
        # WRF domain_clockisstopsubtime stops at the first step reaching/passing
        # the original end; integer midpoint seconds need not align with dt.
        forecast_s = math.ceil(forecast_s / ctl.dt_s) * ctl.dt_s
    report["forecast_offset_s"] = offset
    report["forecast_hours"] = forecast_s / 3600
    _h, bundles, _meta, _start, _dt, carries = pipeline._load_domains(config, ("d01",))
    bundle, carry0 = bundles["d01"], carries["d01"]
    ns, state0 = bundle.namelist, carry0.state
    if abs(ns.dt_s - ctl.dt_s) > 1e-9:
        raise ValueError("DFI time step differs from loaded domain time step")
    saved_land = d._land(carry0)
    if saved_land is not None:
        with Dataset(src / "wrfinput_d01", "r") as ds:
            if "SNOWC" in ds.variables:
                saved_land["SNOWC"] = np.asarray(ds.variables["SNOWC"][0], np.float32)
    edits = {**_clock_edits(ctl.fwd_start), "sst_update": 0, "dfi_opt": 0}
    if ctl.dfi_opt == DFI_DDFI:
        endpoint, species = _backward_endpoint(carry0, ns, ctl, run, pipeline, report)
        fields = d._prognostic_fields(run, "d01", endpoint, species, bundle=bundle)
    else:
        species, fields = d._species(state0), {}
        edits.update(_dfl_diffusion(nml))
        report.update(bck_steps=0, bck_filtered=False, dfl_diffusion=_dfl_diffusion(nml))
    write_stage = _write_dfl_stage_dir if ctl.dfi_opt == DFI_DFL else d._write_stage_dir
    fwd_dir = write_stage(src, work_dir / "fwd_input", fields=fields, when=ctl.fwd_start,
                                namelist_edits=edits)
    _h, bundles_f, _meta, start_f, _dt, carries_f = pipeline._load_domains(replace(config, input_dir=fwd_dir), ("d01",))
    report["fwd_clock_start"] = str(start_f)
    carry_f = carries_f["d01"]
    shifted = d._shifted_boundary(state0, shift_s=-ctl.n_bck * ctl.dt_s,
                                  cadence_s=ns.boundary_config.update_cadence_s,
                                  constant_bc=ctl.constant_bc, ic_source=carry_f.state)
    shifted = {k: v.astype(getattr(carry_f.state, k).dtype) for k, v in shifted.items()}
    carry_f = om._commit_to_operational_device(carry_f.replace(state=carry_f.state.replace(**shifted)))
    h = optfil_hcoeff(ctl.fwd_run_s, ctl.dt_s, ctl.cutoff_s, ctl.nfilter)
    report["fwd_hcoeff"] = [float(x) for x in h]
    carry_end, filt, _sp = d._stage_loop(carry_f, bundles_f["d01"].namelist, n_steps=ctl.n_fwd,
                                       start_step=1, hcoeff=h, frozen=None, report=report, label="fwd")
    shutil.rmtree(fwd_dir)
    fst_fields = _launch_fields(run, ctl, filt, species, bundles_f["d01"], preserve_filtered_w=preserve_w)
    end_land = d._land(carry_end)
    if end_land is not None and saved_land is not None:
        reset = d._land_reset(end_land, saved_land)
        fst_fields.update(reset)
        fst_fields["CANWAT"] = np.zeros_like(reset["CANWAT"])
        fst_fields["LAI"] = end_land["LAI"]
        report["land_reset_cells_restored"] = int((reset["SNOWC"] == 0).sum())
    diag = getattr(carry_end, "history_diagnostics", None)
    if diag is not None:
        for name, leaf in (("T2", "t2"), ("Q2", "q2"), ("U10", "u10"), ("V10", "v10"), ("PSFC", "psfc")):
            value = getattr(diag, leaf, None)
            if value is not None:
                fst_fields[name] = np.asarray(value, np.float32)
    fst_edits = {**_clock_edits(ctl.forecast_start), **_dfl_diffusion(nml)} if ctl.dfi_opt == DFI_DFL else None
    fst = write_stage(src, Path(fst_dir) if fst_dir is not None else work_dir / "fst_input",
                             fields=fst_fields, when=ctl.forecast_start, namelist_edits=fst_edits)
    _radar_field(fst / "wrfinput_d01")
    if offset:
        stride = rebase_dfl_boundary(fst / "wrfbdy_d01", offset_s=offset, forecast_s=forecast_s,
                                    interval_s=int(ns.boundary_config.update_cadence_s))
        text = d._edit_namelist((fst / "namelist.input").read_text(), {"interval_seconds": stride})
        (fst / "namelist.input").write_text(text)
        report["forecast_boundary_cadence_s"] = stride
    marker = {k: v for k, v in report.items() if k not in {"fwd_hcoeff", "log"}}
    (fst / d.MARKER).write_text(json.dumps(marker, indent=1) + "\n")
    report["filtered_input_dir"] = str(fst)
    return fst, report
