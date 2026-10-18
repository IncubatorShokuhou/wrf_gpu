"""Time-expanded (lagged) ensembles centred on a control run (lane a1-io; DESIGN_v2 §6).

Member (r, k) = x_ctrl(t_a) + scale·δ_{r,k},  δ_{r,k} = x_r(t_a+k) − mean_k' x_r(t_a+k'),
for every frame field (prognostic AND diagnostic, so H(x) stays consistent with the state), runs r and
offsets k (default ±1…±5 h). Each run's deviations have zero mean, so the ensemble mean is the
control frame (to float64 round-off) and passive fields come from the control frame. Static fields
(PB, PHB, …) get zero deviation. TES is a NONCAUSAL implementation fixture (it uses frames after t_a):
never skill evidence. :func:`perturbation_spectrum` reports the singular spectrum / effective rank.

Files: hourly wrfout at hh:00:00 / hh:00:18 / hh:00:36 (dt 54 s). The t_a frames must match exactly;
offset frames are picked by hour and their actual ``Times`` are recorded (|Δ| < 60 s).
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .state_io import (
    FRAME_VARIABLES,
    EnsembleFrames,
    assert_same_grid,
    frame_path,
    parse_wrf_time,
    read_frame,
    read_frame_index,
    sha256,
    wrf_time,
)

DEFAULT_OFFSETS_H = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
NONNEGATIVE = ("QVAPOR", "Q2")
OFFSET_TOLERANCE_S = 59.0
NONCAUSAL_LABEL = "TES: noncausal implementation fixture (uses frames after t_a); never skill evidence"


def hourly_wrfout(run_dir, domain: str, when) -> Path:
    """The single ``wrfout_<domain>_YYYY-MM-DD_HH:00:ss`` file of hour ``when`` in ``run_dir``."""
    hour = parse_wrf_time(when).strftime("%Y-%m-%d_%H")
    hits = sorted(Path(run_dir).glob(f"wrfout_{domain}_{hour}:00:*"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{run_dir}: need exactly one wrfout_{domain}_{hour}:00:* file, found {len(hits)}")
    return hits[0]


def time_expanded_members(runs=None, domain: str = "d01", t_a=None, offsets_h: Sequence[int] = DEFAULT_OFFSETS_H,
                          scale: float = 1.0, *, control_run=None, names: Iterable[str] = FRAME_VARIABLES,
                          clip_negative: bool = False, dtype=np.float64) -> EnsembleFrames:
    """Lagged TES ensemble valid at ``t_a``: N = len(runs)·len(offsets_h) members around the control.

    ``runs``: one run directory or a list; ``control_run`` defaults to ``runs[0]``. All runs and the
    control must share the grid (exact XLAT/XLONG/HGT/ZNU/ZNW + DX/DY), else refused. Negative
    QVAPOR/Q2 are always COUNTED per member (``meta['negative']``); ``clip_negative=True`` sets them
    to 0 and records that. float64 by default (d01, N = 20, 15 default fields ≈ 0.55 GB).
    """
    if runs is None or t_a is None:
        raise ValueError("runs and t_a are required")
    if isinstance(runs, (str, Path)):
        runs = [runs]
    runs = [Path(r) for r in runs]
    control_run = Path(control_run) if control_run is not None else runs[0]
    offsets_h = tuple(int(k) for k in offsets_h)
    if not runs or not offsets_h or 0 in offsets_h or len(set(offsets_h)) != len(offsets_h):
        raise ValueError("need ≥1 run and distinct non-zero offsets")
    names = tuple(names)
    t_a = wrf_time(t_a)
    control_file = frame_path(control_run, domain, t_a)
    control = {n: v.astype(np.float64) for n, v in read_frame(control_file, t_a, names).items()}
    hashes: dict[str, str] = {}

    def digest(path: Path) -> str:
        if str(path) not in hashes:
            hashes[str(path)] = sha256(path)
        return hashes[str(path)]

    fields = {n: np.empty((len(runs) * len(offsets_h),) + control[n].shape, dtype=dtype) for n in names}
    members, ids, m = [], [], 0
    for r, run in enumerate(runs):
        assert_same_grid(control_file, frame_path(run, domain, t_a))
        frames, records = [], []
        for k in offsets_h:
            nominal = parse_wrf_time(t_a) + timedelta(hours=k)
            path = hourly_wrfout(run, domain, nominal)
            index, actual = read_frame_index(path, nominal, tolerance_s=OFFSET_TOLERANCE_S)
            frames.append(read_frame(path, actual, names))
            records.append({"run_index": r, "run_dir": str(run), "offset_h": k, "file": str(path),
                            "frame_index": index, "sha256": digest(path), "nominal_time": wrf_time(nominal),
                            "actual_time": actual, "time_offset_s": (parse_wrf_time(actual) - nominal).total_seconds()})
        mean = {n: np.mean([f[n].astype(np.float64) for f in frames], axis=0) for n in names}
        for f, rec in zip(frames, records):
            for n in names:
                fields[n][m] = control[n] + scale * (f[n].astype(np.float64) - mean[n])
            members.append(rec)
            ids.append(f"r{r}{rec['offset_h']:+d}h")
            m += 1
        del frames
    negative = {}
    for n in NONNEGATIVE:
        if n in fields:
            counts = [int(np.count_nonzero(fields[n][i] < 0)) for i in range(m)]
            negative[n] = {"per_member": counts, "total": int(sum(counts)), "clipped": bool(clip_negative)}
            if clip_negative:
                np.maximum(fields[n], 0, out=fields[n])
    mean_dev = {n: float(np.abs(fields[n].astype(np.float64).mean(axis=0) - control[n]).max()) for n in names}
    meta = {"kind": "time_expanded", "label": NONCAUSAL_LABEL, "domain": domain, "t_a": t_a,
            "offsets_h": list(offsets_h), "scale": float(scale), "runs": [str(r) for r in runs],
            "control_run": str(control_run), "control_file": str(control_file),
            "control_sha256": digest(control_file), "names": list(names), "dtype": np.dtype(dtype).name,
            "negative": negative, "mean_minus_control_max_abs": mean_dev}
    return EnsembleFrames(fields, member_ids=ids, valid_time=t_a, members=members, meta=meta,
                          times=(t_a,), control_file=control_file)


def perturbation_spectrum(frames: Mapping[str, np.ndarray], names: Iterable[str] | None = None) -> dict:
    """Singular spectrum of the centred member deviations (each field scaled by its own rms deviation,
    so variables are comparable) via the N×N Gram matrix. Effective rank = exp(entropy of σ²/Σσ²);
    participation ratio = (Σσ²)²/Σσ⁴. A rank ≤ N−1 ensemble has at most N−1 nonzero σ."""
    names = list(names) if names is not None else list(frames)
    n = next(iter(frames.values())).shape[0]
    gram = np.zeros((n, n))
    used = {}
    for name in names:
        x = np.asarray(frames[name], np.float64).reshape(n, -1)
        d = x - x.mean(axis=0)
        rms = float(np.sqrt(np.mean(d ** 2)))
        used[name] = rms
        if rms > 0:
            d = d / (rms * np.sqrt(d.shape[1]))
            gram += d @ d.T
    lam = np.clip(np.linalg.eigvalsh(gram)[::-1], 0.0, None)
    p = lam / lam.sum() if lam.sum() > 0 else lam
    nz = p[p > 0]
    return {"singular_values": [float(v) for v in np.sqrt(lam)], "energy_fraction": [float(v) for v in p],
            "effective_rank_entropy": float(np.exp(-(nz * np.log(nz)).sum())) if nz.size else 0.0,
            "participation_ratio": float(1.0 / (p ** 2).sum()) if lam.sum() > 0 else 0.0,
            "field_rms_deviation": used, "n_members": n, "fields": names}


# --------------------------------------------------------------------------- perturbed-IC member wrfinputs

IC_NAMES = ("U", "V", "T", "QVAPOR")
_STAGGER = {"U": "x", "V": "y"}


def select_archive_states(run_dirs, domain: str, t_a, n_members: int, *, max_age_days: float = 45,
                          exclude_runs=()) -> tuple[list[dict], list[dict]]:
    """Causal same-UTC-hour archive states for ``t_a`` (lead module ``archive_ensemble.candidate_frames``).

    One state per valid time (overlapping runs → the shortest lead, i.e. the latest issue), runs in
    ``exclude_runs`` dropped (the control run's own t_a frame IS the control state), then the
    ``n_members`` most recent valid times. Returns (used, dropped) records; refuses if fewer remain.
    """
    from datetime import datetime, timezone
    from . import archive_ensemble as ae

    from . import alisios_guard

    t = parse_wrf_time(t_a).replace(tzinfo=timezone.utc)
    excluded = {str(Path(r).resolve()) for r in exclude_runs}
    by_valid, dropped = {}, []
    candidates = ae.candidate_frames(run_dirs, domain, t, max_age_days=max_age_days)
    # ALISIOS rule (manager 2026-10-17 01:25Z): guard-DENY days are dropped BEFORE anything is opened (scan only).
    guarded = [c for c in candidates if alisios_guard.is_alisios_data(c[0])]
    if guarded:
        allowed, verdicts = alisios_guard.drop_deny_days([(c, c[1].strftime("%Y%m%d")) for c in guarded])
        allowed = set(map(id, allowed))
        for c in guarded:
            if id(c) not in allowed:
                dropped.append({"file": str(c[0]), "run": c[2], "valid": wrf_time(c[1]),
                                "reason": f"ALISIOS guard {verdicts.get(c[1].strftime('%Y%m%d'))} (not opened)"})
        candidates = [c for c in candidates if not alisios_guard.is_alisios_data(c[0]) or id(c) in allowed]
    for path, stamp, run in candidates:
        match = ae._RUN_RE.search(run)
        issued = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        rec = {"file": str(path), "run": run, "valid": wrf_time(stamp), "issued": wrf_time(issued),
               "lead_h": (stamp - issued).total_seconds() / 3600.0}
        if str(Path(run).resolve()) in excluded:
            dropped.append({**rec, "reason": "control run (its t_a frame is the control state)"})
            continue
        prev = by_valid.get(rec["valid"])
        if prev is None or rec["lead_h"] < prev["lead_h"]:
            if prev is not None:
                dropped.append({**prev, "reason": "duplicate valid time, longer lead"})
            by_valid[rec["valid"]] = rec
        else:
            dropped.append({**rec, "reason": "duplicate valid time, longer lead"})
    ordered = sorted(by_valid.values(), key=lambda r: r["valid"], reverse=True)
    if len(ordered) < n_members:
        raise ValueError(f"only {len(ordered)} distinct causal archive states for {n_members} members")
    for rec in ordered[n_members:]:
        dropped.append({**rec, "reason": f"older than the {n_members} most recent valid times"})
    return sorted(ordered[:n_members], key=lambda r: r["valid"]), dropped


def run_days_from_name(run_dir, n_days: int = 4) -> list[str]:
    """ALISIOS WN3 runs ``wg_YYYYMMDD_18z`` start the next day 00Z for 72 h → valid days issue+1 … issue+4.
    Derived from the NAME so that no case file is opened outside a guarded session."""
    from datetime import datetime, timedelta
    from . import archive_ensemble as ae

    match = ae._RUN_RE.search(str(run_dir))
    if not match:
        raise ValueError(f"{run_dir}: not an ALISIOS wg_YYYYMMDD_HHz run directory")
    issued = datetime.strptime(match.group(1), "%Y%m%d")
    return [(issued + timedelta(days=k)).strftime("%Y%m%d") for k in range(1, n_days + 1)]


def case_read_set(run_dir) -> dict:
    """Files ``cycle_case.write_cycle_case`` opens in a source case dir (namelist, forcing, static tables) → run days."""
    run_dir = Path(run_dir)
    days = run_days_from_name(run_dir)
    names = ["namelist.input", "wrfbdy_d01", "wrflowinp_d01", "wrfinput_d01"]
    static = [p.name for p in sorted(run_dir.iterdir()) if p.is_file() and
              (p.suffix.lower() in (".tbl", ".dat", ".formatted", ".asc") or
               p.name.startswith(("RRTM_DATA", "RRTMG_", "CLWRF", "ETAMPNEW", "CAM_ABS_DATA", "CAM_AEROPT_DATA")))]
    return {str((run_dir / n).resolve()): days for n in names + static if (run_dir / n).exists()}


def perturbed_ic_read_plan(control, run_dirs, *, template_wrfinput=None, n_members: int = 15, domain: str = "d01",
                           max_age_days: float = 45, exclude_runs=None, source_case_dir=None,
                           return_states: bool = False) -> tuple:
    """(allowed {path: [days]}, issue_days[, used states, dropped]) for ``alisios_guard.guarded_session`` around
    :func:`perturbed_ic_members` (+ ``write_cycle_case`` when ``source_case_dir`` is given). Scan/listing only; no
    ALISIOS file is opened. Pass the returned states as ``perturbed_ic_members(archive_states=…)``: the guard scan
    reads ALISIOS's own blocklist, which the read spy refuses inside an active session."""
    control = Path(control).resolve()
    if control.name.startswith(f"wrfout_{domain}_"):  # exact-time frame: time from the WRF file name
        t_a = wrf_time(control.name[len(f"wrfout_{domain}_"):])
    else:  # a run's own wrfinput: valid at the run start (issue day + 1, 00Z)
        d = run_days_from_name(control.parent)[0]
        t_a = f"{d[:4]}-{d[4:6]}-{d[6:]}_00:00:00"
    exclude_runs = [control.parent] if exclude_runs is None else list(exclude_runs)
    used, dropped = select_archive_states(run_dirs, domain, t_a, n_members, max_age_days=max_age_days,
                                          exclude_runs=exclude_runs)
    allowed = {str(control): [parse_wrf_time(t_a).strftime("%Y%m%d")]}
    if template_wrfinput is not None:
        allowed[str(Path(template_wrfinput).resolve())] = run_days_from_name(Path(template_wrfinput).parent)[:1]
    for rec in used:
        allowed[str(Path(rec["file"]).resolve())] = [parse_wrf_time(rec["valid"]).strftime("%Y%m%d")]
    if source_case_dir is not None:
        allowed.update({p: d for p, d in case_read_set(source_case_dir).items() if p not in allowed})
    from . import archive_ensemble as ae
    issues = sorted({m.group(1) for p in allowed if (m := ae._RUN_RE.search(p))})
    return (allowed, issues, used, dropped) if return_states else (allowed, issues)


def ic_member_fields(control: Mapping[str, np.ndarray], states: Sequence[Mapping[str, np.ndarray]], scale: float,
                     *, names: Iterable[str] = IC_NAMES, spec_bdy_width: int = 5, ramp: int = 5, weights=None,
                     untapered: Iterable[str] = ()) -> dict:
    """(N, …) float64 member fields: control + taper·(archive member − control), archive member from the lead's
    ``archive_ensemble`` formulas (additive deviation about the archive mean; LOG_FIELDS log-space, positive,
    re-centred multiplicatively). The taper (0 in the specified/relaxation zone, cosine ramp) keeps the shared
    wrfbdy consistent; the ensemble mean stays the control (to round-off).

    ``weights`` (w41-da-ens, opt-in): (N, M) Gaussian combination weights (``spread.gaussian_combinations``) —
    member i = control + scale·Σ_m w_im (state_m − mean) (log space for LOG_FIELDS, ``perturb.recentre``), so N
    may exceed the number M of archive states (rank ≤ M − 1) and MU may be perturbed. ``untapered`` names skip the
    taper (only with matched per-member perturbed LBCs, ``lbc_ens``)."""
    from . import archive_ensemble as ae

    if weights is not None:
        from .perturb import combination_perturbations, recentre
        from .state_io import boundary_taper
        names = tuple(names)
        pool = {n: np.stack([np.asarray(s[n], np.float64) for s in states]) for n in names}
        dev, _ = combination_perturbations(pool, int(np.shape(weights)[0]), seed=None, names=names, weights=weights)
        taper = {}
        for name in names:
            if name in tuple(untapered):
                continue
            ref = np.asarray(control[name])
            ny, nx = ref.shape[-2] - (_STAGGER.get(name) == "y"), ref.shape[-1] - (_STAGGER.get(name) == "x")
            taper[name] = boundary_taper(ny, nx, spec_bdy_width, ramp, _STAGGER.get(name))
        return recentre({n: np.asarray(control[n], np.float64) for n in names}, dev, scale, taper=taper)

    out = {}
    for name in names:
        ref = np.asarray(control[name], np.float64)
        stack = np.stack([np.asarray(s[name], np.float64) for s in states])
        if name in ae.LOG_FIELDS:
            member = ae._log_members(ref, stack, scale)
        else:
            member = ref[None] + scale * (stack - stack.mean(axis=0)[None])
        ny, nx = ref.shape[-2] - (_STAGGER.get(name) == "y"), ref.shape[-1] - (_STAGGER.get(name) == "x")
        from .state_io import boundary_taper
        w = boundary_taper(ny, nx, spec_bdy_width, ramp, _STAGGER.get(name))
        out[name] = ref[None] + w * (member - ref[None])
    return out


def perturbed_ic_members(control, run_dirs, out_dir, *, template_wrfinput=None, scale: float = 0.3,
                         n_members: int = 15, domain: str = "d01", names: Iterable[str] = IC_NAMES,
                         max_age_days: float = 45, exclude_runs=None, spec_bdy_width: int | None = None, ramp: int = 5,
                         write_control: bool = True, archive_states=None, combination=None,
                         balanced: bool = False) -> dict:
    """Write N perturbed-IC member wrfinputs (+ a zero-perturbation control) valid at the control's time.

    member = control + tapered archive same-phase deviation (U, V, T, QVAPOR; MU frozen) → P, PH, P_HYD, AL by
    :func:`state_io.rebalance_increment` → :func:`state_io.write_analysis_wrfinput`.
    * cold (``template_wrfinput`` None): ``control`` is a run's own wrfinput_d01 (run start), used as template AND
      frame, NO land cycle (the port's/WRF's cold Noah-MP init, like the control); cases need land='cold'.
    * warm (``template_wrfinput`` = the run's wrfinput_d01): ``control`` is an exact-time wrfout frame; the
      writer retimes the template, copies the frame state and cycles the land from the frame (marker).
    ``exclude_runs`` defaults to the control's run directory. ``archive_states`` = (used, dropped) from
    :func:`perturbed_ic_read_plan` (select BEFORE entering a guarded session); None → select here (scan).
    ``spec_bdy_width`` None = the case width (``analysis.resolve_spec_bdy_width``: the control's namelist; an
    explicit value must agree with it), so the tapered increment never touches the specified + relaxation zone the
    shared wrfbdy owns. Returns the manifest (also written).

    w41-da-ens opt-ins (defaults unchanged): ``combination={'n': N, 'seed': s}`` draws N members as seeded Gaussian
    combinations of the ``n_members`` archive states (N may exceed the archive pool; weights recorded in the
    manifest); ``balanced=True`` adds the column mass MU to the perturbed fields (tapered: the shared wrfbdy keeps
    MU) and rebalances P/PH with ``perturb.rebalance_with_mu`` (WRF real.exe recursion incl. ΔMU) instead of the
    fixed-MU ``rebalance_increment``.
    """
    import json
    from datetime import datetime, timezone
    from . import archive_ensemble as ae
    from .state_io import (Dataset as _Dataset, frame_times, read_frame, read_geometry, rebalance_increment,
                           rebalance_report, write_analysis_wrfinput)

    control_file = Path(control).resolve()
    template = Path(template_wrfinput).resolve() if template_wrfinput is not None else control_file
    warm = template != control_file
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"{out_dir}: new output directory required")
    names = tuple(names)
    with _Dataset(str(control_file)) as ds:
        times = frame_times(ds)
        has_al = "AL" in ds.variables
    if len(times) != 1:
        raise ValueError("control file must hold exactly one time")
    t_a = times[0]
    exclude_runs = [control_file.parent] if exclude_runs is None else list(exclude_runs)
    if archive_states is None:
        used, dropped = select_archive_states(run_dirs, domain, t_a, n_members, max_age_days=max_age_days,
                                              exclude_runs=exclude_runs)
    else:
        used, dropped = [dict(u) for u in archive_states[0]], list(archive_states[1])
        if len(used) != n_members:
            raise ValueError(f"archive_states holds {len(used)} states for {n_members} members")
    pert_names = names + (("MU",) if balanced and "MU" not in names else ())
    control = {n: v for n, v in read_frame(control_file, t_a, pert_names + ae.GRID_IDENTITY).items()}
    states = []
    for rec in used:
        fields, valid = ae.read_fields(rec["file"], pert_names + ae.GRID_IDENTITY)
        for g in ae.GRID_IDENTITY:
            if not np.array_equal(fields[g], control[g]):
                raise ValueError(f"{rec['file']}: grid identity differs in {g}")
        rec["actual_time"] = valid
        rec["sha256"] = sha256(rec["file"])
        states.append(fields)
    weights = None
    if combination is not None:
        from .spread import gaussian_combinations
        weights = gaussian_combinations(int(combination["n"]), len(states), int(combination["seed"]))
    from .analysis import resolve_spec_bdy_width
    geo = read_geometry(control_file, base_wrfinput=template)
    spec_bdy_width, width_source = resolve_spec_bdy_width(geo, spec_bdy_width)
    members = ic_member_fields(control, states, scale, names=pert_names, spec_bdy_width=spec_bdy_width, ramp=ramp,
                               weights=weights)
    n_out = n_members if weights is None else int(weights.shape[0])
    del states
    bg = read_frame(control_file, t_a, names + ("MU", "P", "PH", "P_HYD") + (("AL",) if has_al else ()))
    out_dir.mkdir(parents=True)
    records = []
    plan = [("ctrl", None)] if write_control else []
    plan += [(f"m{k:02d}", k) for k in range(n_out)]
    for label, k in plan:
        an = dict(bg)
        if k is not None:
            an.update({n: members[n][k].astype(bg[n].dtype) for n in pert_names})
        if balanced:
            from .perturb import rebalance_with_mu
            reb = rebalance_with_mu(bg, an, geo)
        else:
            reb = rebalance_increment(bg, an, geo)
        fields = {**{n: an[n] for n in pert_names}, **{n: reb[n] for n in ("P", "PH", "P_HYD", "AL") if n in reb}}
        target = out_dir / label / f"wrfinput_{domain}"
        source_state = (used[k] if weights is None else {"combination_row": k}) if k is not None else None
        rec = write_analysis_wrfinput(template, control_file, t_a, fields, target, land_cycle=warm,
                                      extra_provenance={"kind": "perturbed-IC member" if k is not None else
                                                        "zero-perturbation control", "member": label,
                                                        "archive_state": source_state, "scale": scale,
                                                        "balanced": bool(balanced)})
        report = rebalance_report(bg, an, reb, geo)
        stats = {n: {"rms": float(np.sqrt(np.mean((an[n].astype(np.float64) - bg[n]) ** 2))),
                     "max_abs": float(np.abs(an[n].astype(np.float64) - bg[n]).max())} for n in pert_names}
        records.append({"member": label, "archive_state": source_state,
                        "wrfinput": str(target), "sha256": rec["output_sha256"], "bytes": target.stat().st_size,
                        "increment": stats, "rebalance": report, "n_changed": {n: v["n_changed"] for n, v in
                                                                              rec["analysis_fields"].items()}})
    manifest = {"schema": 1, "kind": "perturbed-IC members (archive same-phase deviations)", "t_a": t_a,
                "mode": "warm (frame + template, land cycled)" if warm else "cold (run-start wrfinput)",
                "control": str(control_file), "control_sha256": sha256(control_file),
                "template_wrfinput": str(template), "template_sha256": sha256(template),
                "scale": scale, "n_members": n_out, "names": list(pert_names),
                "mu": "perturbed (tapered) + rebalance_with_mu" if balanced else "frozen",
                "combination": None if weights is None else {"n": n_out, "pool": n_members,
                                                              "seed": int(combination["seed"]),
                                                              "weights": weights.tolist()},
                "qvapor": "log-space archive deviations (archive_ensemble._log_members)",
                "taper": {"spec_bdy_width": spec_bdy_width, "spec_bdy_width_source": width_source, "ramp": ramp,
                          "applied_to": "complete IC increment"},
                "land": ("cycled from the control frame (GPUWRF_LAND_CYCLE marker; write_cycle_case land='cycled')"
                         if warm else "unchanged, no GPUWRF_LAND_CYCLE marker (cold Noah-MP init; land='cold')"),
                "archive_states": used, "archive_dropped": dropped, "members": records,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    (out_dir / "members_manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest
