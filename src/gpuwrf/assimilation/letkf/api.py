"""Library entry points of ``gpuwrf da`` (used by the CLI and by cycle drivers).

``analyse_case(config, out_dir)`` runs one offline LETKF analysis time: plan the reads → governance
session → background ensemble → observations → inflation calibration → one LETKF analysis per configured
experiment (receipt each) → optional analysis-mean wrfinput (control + mean increment, rebalanced, moisture
gated) and per-member cycle cases. Host-only (NumPy/SciPy/netCDF4): no JAX import, no GPU. Outputs are new
files under ``out_dir`` (which must not exist); inputs are never modified.

Hooks owned by the ensemble lane are picked up when present: ``inflation.calibrate(frames, geo, obs, ensemble,
inflation) -> (frames, info)`` replaces the built-in calibration and ``inflation.analysis_hooks(section) ->
{maintenance, prior_inflation}`` supplies posterior maintenance / per-point prior inflation.
"""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import config as dacfg
from .analysis import KIND_NAMES, LetkfParams, analyse, resolve_spec_bdy_width

KINDS = {name: code for code, name in KIND_NAMES.items()}
SUMMARY_SCHEMA = "gpuwrf.da.analysis_summary.v1"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=1, default=float, allow_nan=False) + "\n")


# ------------------------------------------------------------------------------------------- ensemble

def load_ensemble(cfg, plan):
    """Background ensemble at t_a -> (frames {name: (N, ...)}, geo, meta)."""
    from . import archive_ensemble, state_io

    data = cfg.data
    ens = data["ensemble"]
    t_a = cfg.t_a
    names = tuple(ens["names"])
    if ens["kind"] == "members":
        paths = plan.paths("member")
        frames_obj = state_io.read_member_frames(paths, t_a, names)
        frames = {k: np.asarray(v) for k, v in dict(frames_obj).items()}
        meta = {"kind": "members (flow-dependent forecast ensemble)", "n_members": len(paths),
                "members": [str(p) for p in paths], "member_records": frames_obj.members}
        geometry_file = data.get("control_file", paths[0])
    else:
        picks = []
        for e in plan.entries:
            if e.role == "archive":
                stamp = state_io.parse_wrf_time(Path(e.path).name.split("_", 2)[2])
                picks.append((Path(e.path), stamp, e.meta.get("run", "")))
        frames, meta = archive_ensemble.same_phase_members(
            data["control_file"], [], data["domain"], t_a, names, max_age_days=ens.get("max_age_days"),
            max_members=ens.get("max_members"), picks=picks)
        geometry_file = data["control_file"]
    geo = state_io.read_geometry(geometry_file, base_wrfinput=data.get("template_wrfinput"))
    meta = {**dict(meta), "geometry_file": str(geometry_file)}
    if "control_file" in data:
        meta["control_file"] = data["control_file"]
    return frames, geo, meta


def calibrate(frames, geo, obs, cfg):
    """Prior scaling/inflation -> (frames, info with optional ``inflation_rho``)."""
    from . import inflation as infl

    ens = cfg.data["ensemble"]
    if hasattr(infl, "calibrate"):
        return infl.calibrate(frames, geo, obs, ens, cfg.data.get("inflation"))
    return calibrate_scale(frames, geo, obs, ens)


def calibrate_scale(frames, geo, obs, spec):
    """Fixed float, or 'innovation': s with E[d²] = σo² + s² var(Hx) over accepted assimilable obs (A1 probe)."""
    from . import archive_ensemble
    from .analysis import obs_sigma, quality_control
    from .operators import model_equivalents

    scale = spec.get("scale", 1.0)
    if spec.get("kind") == "members":
        # A real forecast ensemble keeps its members' own frames as the background (member cycling compares
        # against them; tapered zones must stay bitwise). The innovation-calibrated s becomes multiplicative prior
        # inflation rho = s^2 inside the LETKF transform instead of a frame rescale.
        if scale == "innovation":
            _, info = calibrate_scale(frames, geo, obs, {**spec, "kind": "same_phase"})
            return frames, {**info, "method": "innovation->prior inflation", "inflation_rho": info["scale"] ** 2}
        if scale == "innovation_debiased":
            from .inflation import calibrate_debiased
            return calibrate_debiased(frames, geo, obs, spec)
        return frames, {"scale": 1.0, "method": "none (members)", "inflation_rho": float(spec.get("inflation_rho", 1.0))}
    if scale == "innovation_bands":
        return _calibrate_bands(frames, geo, obs, spec)
    if scale != "innovation":
        return archive_ensemble.rescale(frames, float(scale)), {"scale": float(scale), "method": "fixed"}
    hx = np.asarray(model_equivalents(obs, frames, geo), dtype=np.float64)
    ok, _ = quality_control(obs, hx, 1.0e9)               # finite/valid only; no gross check before calibration
    kinds = np.asarray(obs.kind)[ok]
    d = np.asarray(obs.value, float)[ok] - hx[:, ok].mean(axis=0)
    var = hx[:, ok].var(axis=0, ddof=1)
    s, diag = archive_ensemble.innovation_scale(d, obs_sigma(obs)[ok], var)
    by_kind = {}
    for k in np.unique(kinds):
        sel = kinds == k
        by_kind[KIND_NAMES.get(int(k), str(k))] = archive_ensemble.innovation_scale(
            d[sel], obs_sigma(obs)[ok][sel], var[sel])[0]
    return archive_ensemble.rescale(frames, s), {"scale": s, "method": "innovation", **diag, "by_kind": by_kind}


def _calibrate_bands(frames, geo, obs, spec):
    """Per pressure band s from innovations (sonde obs in the band; surface obs join the lowest band)."""
    from . import archive_ensemble
    from .analysis import obs_sigma, quality_control
    from .operators import model_equivalents

    edges = spec.get("band_edges_pa", [85000.0, 50000.0])
    hx = np.asarray(model_equivalents(obs, frames, geo), dtype=np.float64)
    ok, _ = quality_control(obs, hx, 1.0e9)
    kind = np.asarray(obs.kind)
    p = np.asarray(obs.p_pa, dtype=np.float64)
    band = np.full(len(kind), -1)
    band[(kind == 2)] = 0
    lev = kind == 1
    band[lev] = np.digitize(-p[lev], [-e for e in edges])
    d = np.asarray(obs.value, float) - hx.mean(axis=0)
    var = hx.var(axis=0, ddof=1)
    s_band, diag = [], []
    for b in range(len(edges) + 1):
        sel = ok & (band == b)
        s_b, info = archive_ensemble.innovation_scale(d[sel], obs_sigma(obs)[sel], var[sel]) if sel.any() else (0.0, {"n": 0})
        s_band.append(s_b)
        diag.append(info)
    pmean = (np.asarray(frames["P"], float) + np.asarray(frames["PB"], float)).mean(axis=(0, 2, 3))
    level_band = np.digitize(-pmean, [-e for e in edges])
    scale_k = np.asarray(s_band)[level_band]
    return archive_ensemble.rescale_levels(frames, scale_k, surface_scale=s_band[0]), {
        "method": "innovation_bands", "band_edges_pa": edges, "scale_by_band": s_band, "band_diag": diag,
        "scale_by_level": scale_k.round(4).tolist(), "scale": float(np.mean(scale_k))}


def column_profile(frames, analysis, geo, lat, lon):
    """Background/analysis mean profiles at the column nearest (lat, lon): z AGL (m), T (K), QV, p."""
    from .analysis import _nearest_columns

    j, i = (int(x[0]) for x in _nearest_columns(geo, np.atleast_1d(lat), np.atleast_1d(lon)))
    out = {"j": j, "i": i}
    for tag, src in (("b", frames), ("a", analysis)):
        ph = (np.asarray(src["PH"], float) + np.asarray(src["PHB"], float)).mean(0)[:, j, i] / 9.81
        z = 0.5 * (ph[1:] + ph[:-1]) - float(np.asarray(geo["HGT"])[j, i])
        p = (np.asarray(src["P"], float) + np.asarray(src["PB"], float)).mean(0)[:, j, i]
        theta = np.asarray(src["T"], float).mean(0)[:, j, i] + 300.0
        out[f"z_agl_{tag}"] = z.round(1).tolist()
        out[f"t_{tag}"] = (theta * (p / 1.0e5) ** (287.0 / 1004.5)).round(3).tolist()
        out[f"qv_{tag}"] = np.asarray(src["QVAPOR"], float).mean(0)[:, j, i].round(7).tolist()
        out[f"p_{tag}"] = p.round(1).tolist()
    return out


# ------------------------------------------------------------------------------------------- analysis

def _params(base_cfg: dict, overrides: dict) -> LetkfParams:
    known = {f.name for f in fields(LetkfParams)}
    merged = {**base_cfg, **overrides}
    merged = {k: (tuple(v) if isinstance(v, list) else v) for k, v in merged.items() if k in known}
    return LetkfParams(**merged)


def _hooks(cfg):
    from . import inflation as infl

    hooks = {}
    if hasattr(infl, "analysis_hooks") and cfg.data.get("inflation") is not None:
        hooks.update(infl.analysis_hooks(cfg.data["inflation"]))
    if "localisation" in cfg.data:
        hooks["var_loc"] = cfg.data["localisation"]["variables"]
    return hooks


def _relabel(obj, labels):
    """Staged input paths -> their original paths in receipts (exact string matches only)."""
    if not labels:
        return obj
    if isinstance(obj, str):
        return labels.get(obj, obj)
    if isinstance(obj, dict):
        return {k: _relabel(v, labels) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_relabel(v, labels) for v in obj]
    return obj


def _boundary_width(geo, params, labels=None):
    """The case's specified + relaxation zone width (its namelist); refuses a contradiction or an unknown width."""
    try:
        width, source = resolve_spec_bdy_width(geo, params.spec_bdy_width)
    except ValueError as exc:
        message = str(exc)
        for staged, original in (labels or {}).items():        # name the original input, not a governance stage
            message = message.replace(staged, original)
        raise dacfg.ConfigError(f"params.spec_bdy_width: {message}") from None
    if source.startswith("WRF Registry default"):
        raise dacfg.ConfigError("params.spec_bdy_width: the case boundary width is unknown (no namelist.input next "
                                "to the control file / first member) - give params.spec_bdy_width")
    return width, source


def _write_mean(cfg, out_dir, frames, analysis, geo, policy, receipt, meta, land="cycled", labels=None,
                width=None):
    """Deterministic analysis-mean wrfinput: control + mean increment, moisture policy + gate, rebalance."""
    from . import moisture as mst, state_io

    data = cfg.data
    control_file = data.get("control_file") or meta.get("control_file")
    control = state_io.read_frame(control_file, cfg.t_a, ("U", "V", "T", "QVAPOR", "MU", "P", "PH", "P_HYD"))
    names = tuple(n for n in ("U", "V", "T", "QVAPOR") if n in analysis)
    mean_counts = {}
    fields_ = state_io.analysis_mean_fields(control, frames, analysis, geo, names=names, moisture=policy,
                                            counters=mean_counts)
    receipt["mean_moisture"] = mean_counts
    if not mean_counts.get("admissible", True):
        receipt["analysis_wrfinput"] = {"written": False, "refusal_reasons": mean_counts["refusal_reasons"]}
        return None
    from .member_cycle import spec_zone_changes
    zone = spec_zone_changes(control, {k: v for k, v in fields_.items() if k in control}, width)
    if any(zone.values()):        # the shared boundary files own this zone: the mean must leave it untouched
        receipt["analysis_wrfinput"] = {"written": False,
                                        "refusal_reasons": [f"mean changes the specified/relaxation zone {zone}"]}
        return None
    written = mst.check_written("QVAPOR", fields_["QVAPOR"], control["QVAPOR"]) if "QVAPOR" in fields_ else None
    if written and (written["negative"] or written["nonfinite"]):
        receipt["analysis_wrfinput"] = {"written": False, "refusal_reasons": [f"stored mean QVAPOR {written}"]}
        return None
    target = Path(out_dir) / f"wrfinput_{data['domain']}"
    record = state_io.write_analysis_wrfinput(data["template_wrfinput"], control_file, cfg.t_a, fields_, target,
                                              land_cycle=land == "cycled", source_labels=labels,
                                              extra_provenance={"config_sha256": cfg.sha256, "kind": "analysis mean"})
    receipt["analysis_wrfinput"] = {"written": True, "path": str(target), "sha256": record["output_sha256"],
                                    "stored_qvapor": written}
    return target


def _run(cfg, out_dir, plan, labels=None):
    from . import moisture as mst, readers

    data = cfg.data
    t0 = time.perf_counter()
    frames, geo, meta = load_ensemble(cfg, plan)
    t_ens = time.perf_counter() - t0
    obs, obs_info = readers.read_sources(data["obs"], cfg.t_a, geo, frames)
    background = frames
    kind = np.asarray(obs.kind)
    hooks = _hooks(cfg)
    summary = {"schema": SUMMARY_SCHEMA, "config_sha256": cfg.sha256, "t_a": data["t_a"],
               "ensemble_meta": {k: v for k, v in meta.items() if isinstance(v, (str, int, float, list, dict))},
               "load_ensemble_s": t_ens, "n_obs": int(len(kind)), "obs": obs_info, "obs_counts": obs.counts(),
               "experiments": {}}
    role_withheld = np.asarray(obs.role) == "withhold"        # rows a source marked as verification-only
    for exp in data["experiments"]:
        kinds = np.isin(kind, [KINDS[k] for k in exp["assimilate"]])
        use = kinds & ~role_withheld
        withhold = (np.isin(kind, [KINDS[k] for k in exp["withhold"]]) | (kinds & role_withheld)) & ~use
        # prior scaling/inflation is calibrated on THIS experiment's assimilated observations only: withheld
        # (verification) observations must not tune the analysis they verify
        t_cal = time.perf_counter()
        assimilated = obs.subset(use)
        frames, scale_info = calibrate(background, geo, assimilated, cfg)
        # an inflation plug-in may return bias-corrected assimilated observations (same rows, new values)
        corrected = scale_info.pop("obs", None)
        if corrected is not None:
            if len(corrected) != len(assimilated):
                raise ValueError("calibration returned a different number of observations")
            assimilated = corrected
        scale_info = {**scale_info, "calibration_obs": int(use.sum()), "calibration_s": time.perf_counter() - t_cal,
                      "obs_corrected": corrected is not None}
        base = dict(data["params"])
        if "inflation_rho" in scale_info:
            base["inflation"] = float(scale_info["inflation_rho"])
        params = _params(base, exp["params"])
        policy = mst.MoisturePolicy(**data["moisture"])
        width, _ = _boundary_width(geo, params, labels)
        params = replace(params, negative_qv=policy.negative, supersaturation=policy.supersaturation,
                         spec_bdy_width=width)
        from .obs import concat
        sub = concat([assimilated, obs.subset(withhold & ~use)], meta=obs.meta)
        withheld = np.r_[np.zeros(len(assimilated), bool), np.ones(int((withhold & ~use).sum()), bool)]
        kwargs = dict(hooks)
        fourd = data["ensemble"].get("fourd")
        if fourd:
            from .fourd import hx_4d
            runs = sorted({e.meta["run"] for e in plan.entries if e.role == "fourd"})
            res = hx_4d(sub, runs, cfg.t_a, float(fourd["window_hours"]), geo)
            kwargs.update(hx_b=res.hx, at_analysis_time=np.abs(np.asarray(sub.dt_s, float)) < 1.0)
        analysis, receipt = analyse(frames, geo, sub, params, withheld=withheld, moisture=policy, **kwargs)
        receipt["experiment"] = exp
        receipt["calibration"] = scale_info
        if exp.get("profile_site"):
            site = exp["profile_site"]
            receipt["profile"] = column_profile(frames, analysis, geo, site["lat"], site["lon"])
        if exp["write_mean"]:
            if receipt["admissible"]:
                _write_mean(cfg, out_dir, frames, analysis, geo, policy, receipt, meta, land=exp["land"], labels=labels,
                            width=width)
            else:
                receipt["analysis_wrfinput"] = {"written": False, "refusal_reasons": receipt["refusal_reasons"]}
        if exp.get("write_members"):
            if receipt["admissible"]:
                from .member_cycle import write_member_cycle
                spec = exp["write_members"]
                members = meta.get("members") or []
                manifest = write_member_cycle(
                    member_wrfouts=members, analysis_member_fields=analysis, valid_time=cfg.t_a,
                    template_wrfinput=data["template_wrfinput"], source_case_dir=spec["source_case_dir"],
                    hours=int(spec["hours"]), out_root=Path(out_dir) / f"members_{exp['name']}",
                    member_ids=[f"m{i:02d}" for i in range(len(members))], source_labels=labels,
                    spec_bdy_width=width,
                    extra_provenance={"config_sha256": cfg.sha256, "experiment": exp["name"]})
                receipt["member_cycle_manifest"] = {"path": str(manifest), "sha256": _sha256(manifest)}
            else:
                receipt["member_cycle_manifest"] = {"written": False, "refusal_reasons": receipt["refusal_reasons"]}
        path = Path(out_dir) / f"receipt_{exp['name']}.json"
        _json(path, _relabel(receipt, labels))
        summary["experiments"][exp["name"]] = {
            "receipt": str(path), "admissible": receipt["admissible"], "refusal_reasons": receipt["refusal_reasons"],
            "assimilated": receipt["assimilated"], "withheld_ok": receipt["withheld_ok"],
            "analysis_wrfinput": receipt.get("analysis_wrfinput"),
            "member_cycle_manifest": receipt.get("member_cycle_manifest")}
    summary["wall_s"] = time.perf_counter() - t0
    return _relabel(summary, labels)


def prepare(config, *, receipt_dir=None):
    """Validate + plan + refine (no payload opened) -> (cfg, policy, plan)."""
    from . import governance

    cfg = config if isinstance(config, dacfg.DaConfig) else dacfg.validate(config)
    policy = governance.make_policy(cfg.data["governance"], receipt_dir=receipt_dir)
    plan = policy.refine(governance.plan_reads(cfg), cfg)
    plan = governance.limit_archive(plan, cfg.data["ensemble"].get("max_members"))
    return cfg, policy, plan


def analyse_case(config, out_dir):
    """Run one analysis time; returns the summary dict (also ``out_dir/summary.json``)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir}: output directory must not exist")
    out_dir.mkdir(parents=True)
    cfg, policy, plan = prepare(config, receipt_dir=out_dir / "governance")
    _json(out_dir / "config.normalised.json", {"sha256": cfg.sha256, "source": cfg.source, "config": cfg.data})
    _json(out_dir / "read_plan.json", plan.to_dict())
    with policy.session(plan, out_dir / "governance", output_dir=out_dir) as session:
        localize = getattr(session, "localize", None)          # staged inputs (ALISIOS reader facade)
        run_cfg, run_plan = localize(cfg, plan) if localize else (cfg, plan)
        labels = {e.path: e.meta["original"] for e in run_plan.entries if "original" in e.meta}
        summary = _run(run_cfg, out_dir, run_plan, labels)
    summary["governance"] = session.receipt
    _json(out_dir / "summary.json", summary)
    return summary
