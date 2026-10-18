"""One LETKF analysis step on an ensemble of WRF frames (lane a1-da, DESIGN §6.1 steps 3-8).

``frames``: dict wrfout-name -> (N, …) member arrays at the analysis time.
``geo``:    dict of 2-D/1-D geometry (XLAT, XLONG, HGT, COSALPHA, SINALPHA, …).
``obs``:    ``letkf.obs.ObsBatch``.

Steps: H(xb) → QC (finite, qc flag, gross check) → local tables on the coarse
analysis grid → LETKF transforms → interpolated update of the analysis
variables → posterior maintenance (default RTPS) → boundary taper → counted
moisture policy and water-budget gate (``moisture``) → incremental hydrostatic
rebalance → H(xa) → O-B/O-A statistics for assimilated AND withheld
observations. Nothing here writes files; see ``state_io``.

Extension hooks (all default to the plain LETKF; the defaults are bitwise the earlier probe path except
the moisture default, which is now the counted clip policy instead of a binary refusal):

* ``prior_inflation``: multiplicative prior inflation per analysis point, an array broadcastable to the
  coarse analysis grid (nkc, njc, nic) or ``callable(agrid) -> array``; replaces ``params.inflation``.
* ``maintenance``: ``callable(name, xb, xa) -> xa`` posterior spread maintenance on (N, ...) float64
  members; default ``core.rtps(xb, xa, params.rtps_alpha)``. For the ``log`` moisture policy QVAPOR is
  passed in log space.
* ``var_loc``: variable localisation ``{state_var: {obs_key: factor in [0, 1]}}`` with ``obs_key`` =
  ``"*"``, a kind (``"sonde"``) or ``"kind:VAR"`` (``"surface:T2"``); the most specific key wins, missing
  keys mean 1. State variables with the same factor vector share one set of weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time

import numpy as np

from . import grid as lgrid
from .core import gc_halfwidth, rtps

KIND_NAMES = {1: "sonde", 2: "surface", 3: "cloud"}
CLOUD = 3                                              # obs.CLOUD
STAGGER = {"U": "x", "V": "y"}


@dataclass(frozen=True)
class LetkfParams:
    # Length convention: L is a Gaussian-equivalent scale; the GC taper uses half-width c = sqrt(10/3) L and is
    # zero beyond 2c = 3.65 L. E.g. loc_v_lnp_low = 0.04 -> support |dlnp| < 0.146 (about 1.2 km near the surface).
    loc_h_km: float = 100.0
    loc_v_lnp: float = 0.1             # vertical L above p_low_pa
    loc_v_lnp_low: float = 0.04        # vertical L below p_low_pa (must not smear across the trade-wind inversion)
    p_low_pa: float = 70000.0
    loc_v_sfc_lnp: float = 0.04        # surface observations, measured from the surface ln p
    stride_h: int = 2
    stride_v: int = 1                  # native levels (critique M1)
    pmax: int = 400
    inflation: float = 1.0             # multiplicative prior inflation
    rtps_alpha: float = 0.8
    gross_k: float = 4.0
    # WRF specified + relaxation zone width; None = the case's own namelist value (``geo['SPEC_BDY_WIDTH']``,
    # WRF Registry default 5 when the case has no namelist). An explicit value must equal the case value.
    spec_bdy_width: int | None = None
    taper_ramp: int = 5
    variables: tuple = ("U", "V", "T", "QVAPOR")     # MU frozen in the first probe (critique M4/B5)
    # 2-D diagnostics transformed with the lowest-level weights ONLY to report a linear posterior estimate of
    # surface H(x) ('posterior_linear'); they are not a re-diagnosed analysis and never satisfy a validation gate.
    diagnostic_variables: tuple = ("T2", "Q2", "U10", "V10", "PSFC")
    # Moisture policy names (``moisture.MoisturePolicy``; pass ``moisture=`` to analyse for non-default limits):
    # 'clip' (default): negative qv -> 0, counted, admissible within the water budget; 'refuse': any negative qv
    # makes the analysis inadmissible; 'log': qv analysed as ln(qv + floor).
    negative_qv: str = "clip"
    # 'keep' only: analysed supersaturation is left for the model to condense ('cap'/'to_cloud' are refused in this
    # build: they would act after the boundary taper and change condensate the fixed-QV rebalance and the writers
    # do not carry; see moisture.SUPERSATURATION_POLICIES)
    supersaturation: str = "keep"
    rebalance: bool = True
    workers: int = 4


def _nearest_columns(geo, lat, lon):
    xyz = lgrid.lonlat_to_xyz_km(np.asarray(geo["XLAT"]), np.asarray(geo["XLONG"])).reshape(-1, 3)
    from scipy.spatial import cKDTree
    _, flat = cKDTree(xyz).query(lgrid.lonlat_to_xyz_km(lat, lon))
    return np.unravel_index(flat, np.shape(geo["XLAT"]))


def observation_coordinates(obs, frames, geo, params: LetkfParams):
    """xyz (km), ln p, horizontal and vertical GC half-widths per observation."""
    kind = np.asarray(obs.kind)
    n = len(kind)
    xyz = lgrid.lonlat_to_xyz_km(obs.lat, obs.lon)
    lnp = np.full(n, np.nan)
    cv = np.full(n, np.inf)
    level = kind == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        lnp[level] = np.log(np.asarray(obs.p_pa, dtype=np.float64)[level])
    low = level & (lnp > np.log(params.p_low_pa))
    cv[level] = gc_halfwidth(params.loc_v_lnp)
    cv[low] = gc_halfwidth(params.loc_v_lnp_low)
    surface = kind == 2
    if surface.any():
        j, i = _nearest_columns(geo, np.asarray(obs.lat)[surface], np.asarray(obs.lon)[surface])
        psfc = np.asarray(frames["PSFC"], dtype=np.float64).mean(axis=0)
        lnp[surface] = np.log(psfc[j, i])
        cv[surface] = gc_halfwidth(params.loc_v_sfc_lnp)
    lv_scale = np.asarray(getattr(obs, "lv_scale", np.ones(n)), dtype=np.float64)
    lh_scale = np.asarray(getattr(obs, "lh_scale", np.ones(n)), dtype=np.float64)
    cv = np.where(np.isfinite(cv), cv * lv_scale, cv)
    ch = gc_halfwidth(params.loc_h_km) * lh_scale
    return xyz, lnp, ch, cv


def obs_sigma(obs):
    """Observation error standard deviation (DESIGN_v2 field ``sigma``; ``error`` accepted as alias)."""
    return np.asarray(obs.sigma if hasattr(obs, "sigma") else obs.error, dtype=np.float64)


def quality_control(obs, hx, gross_k):
    """Return (ok mask, counters). qc != 0 means rejected upstream."""
    value = np.asarray(obs.value, dtype=np.float64)
    error = obs_sigma(obs)
    finite_hx = np.isfinite(hx).all(axis=0)
    upstream = np.asarray(obs.qc) == 0
    valid = np.isfinite(value) & np.isfinite(error) & (error > 0)
    base = finite_hx & upstream & valid
    with np.errstate(invalid="ignore"):
        dep = np.abs(value - hx.mean(axis=0))
        limit = gross_k * np.sqrt(error ** 2 + hx.var(axis=0, ddof=1))
    gross = base & ~(dep <= limit)
    ok = base & ~gross
    counters = {"n_obs": int(len(value)), "rejected_upstream_qc": int((~upstream).sum()),
                "rejected_invalid_value": int((upstream & ~valid).sum()),
                "rejected_hx_nonfinite": int((upstream & valid & ~finite_hx).sum()),
                "rejected_gross": int(gross.sum()), "accepted": int(ok.sum())}
    return ok, counters


def _stats(obs, hx_b, hx_a, mask):
    out = {}
    kind, var = np.asarray(obs.kind), np.asarray(obs.var)
    value = np.asarray(obs.value, dtype=np.float64)
    for k in np.unique(kind[mask]):
        for v in np.unique(var[mask & (kind == k)]):
            sel = mask & (kind == k) & (var == v)
            omb = value[sel] - hx_b[:, sel].mean(axis=0)
            oma = value[sel] - hx_a[:, sel].mean(axis=0)
            out[f"{KIND_NAMES.get(int(k), int(k))}:{int(v)}"] = {
                "n": int(sel.sum()),
                "omb_mean": float(omb.mean()), "omb_rms": float(np.sqrt(np.mean(omb ** 2))),
                "oma_mean": float(oma.mean()), "oma_rms": float(np.sqrt(np.mean(oma ** 2))),
                "spread_b": float(np.sqrt(np.mean(hx_b[:, sel].var(axis=0, ddof=1)))),
                "spread_a": float(np.sqrt(np.mean(hx_a[:, sel].var(axis=0, ddof=1)))),
                "err_rms": float(np.sqrt(np.mean(obs_sigma(obs)[sel] ** 2)))}
    return out


def _obs_key_names(obs):
    from .obs import VAR_NAMES
    kind, var = np.asarray(obs.kind), np.asarray(obs.var)
    kinds = [KIND_NAMES.get(int(k), str(int(k))) for k in kind]
    return kinds, [f"{k}:{VAR_NAMES.get(int(v), str(int(v)))}" for k, v in zip(kinds, var)]


def variable_groups(names, obs, sel, var_loc):
    """[(state names, factor (n_sel,) or None)] — one group with ``None`` when ``var_loc`` is empty."""
    if not var_loc:
        return [(tuple(names), None)]
    unknown = set(var_loc) - set(names)
    if unknown:
        raise ValueError(f"var_loc names state variables that are not analysed: {sorted(unknown)}")
    kinds, keys = _obs_key_names(obs)
    kinds = [kinds[i] for i in sel]
    keys = [keys[i] for i in sel]
    groups = {}
    for name in names:
        table = var_loc.get(name, {})
        for key, value in table.items():
            if not (isinstance(value, (int, float)) and 0.0 <= value <= 1.0):
                raise ValueError(f"var_loc[{name!r}][{key!r}] must be a number in [0, 1]")
        factor = tuple(float(table.get(key, table.get(kind, table.get("*", 1.0)))) for kind, key in zip(kinds, keys))
        groups.setdefault(factor, []).append(name)
    return [(tuple(group), None if all(f == 1.0 for f in factor) else np.asarray(factor, np.float64))
            for factor, group in groups.items()]


def _prior_inflation(prior_inflation, agrid, params):
    if prior_inflation is None:
        return params.inflation
    value = prior_inflation(agrid) if callable(prior_inflation) else prior_inflation
    value = np.broadcast_to(np.asarray(value, np.float64), agrid.shape)
    if not (np.isfinite(value).all() and (value > 0).all()):
        raise ValueError("prior_inflation must be finite and > 0")
    return value.reshape(-1)


def resolve_spec_bdy_width(geo, explicit=None) -> tuple[int, str]:
    """(width, source) of the WRF specified + relaxation zone the analysis must leave untouched: the case's own
    ``spec_bdy_width`` (read from its namelist into ``geo``); an explicit value must agree with it."""
    case = geo.get("SPEC_BDY_WIDTH")
    if case is not None:
        how = "case namelist" + (" (key unset: WRF Registry default 5)" if geo.get("SPEC_BDY_WIDTH_unset") else "")
        if explicit is not None and int(explicit) != int(case):
            raise ValueError(f"spec_bdy_width = {explicit} contradicts the case's spec_bdy_width = {case} "
                             f"({how}, {geo.get('SPEC_BDY_WIDTH_source')}); the boundary zone must match the case")
        return int(case), how
    if explicit is not None:
        return int(explicit), "explicit parameter (no case namelist)"
    return 5, "WRF Registry default 5 (no case namelist, no explicit value)"


def analyse(frames, geo, obs, params: LetkfParams = LetkfParams(), *, withheld=None, model_equivalents=None,
            hx_b=None, at_analysis_time=None, moisture=None, maintenance=None, var_loc=None, prior_inflation=None):
    """Return (analysis frames dict, receipt dict).

    4D-LETKF: pass ``hx_b`` (N, n_obs) computed on each member's trajectory at the obs times (``fourd.hx_4d``) and
    ``at_analysis_time`` (n_obs bool). O−A is then reported only for obs valid at the analysis time; earlier-window
    obs get O−B statistics only (``stats_*_earlier_window``). ``moisture`` = ``moisture.MoisturePolicy`` (default:
    from ``params.negative_qv``/``params.supersaturation`` with the module's declared limits); see the module
    docstring for the other hooks."""
    from . import moisture as mst

    policy = mst.MoisturePolicy.from_params(params) if moisture is None else moisture
    maintenance_label = "rtps(params.rtps_alpha)" if maintenance is None else getattr(maintenance, "__name__", "custom")
    if maintenance is None:
        def maintenance(name, xb, xa):
            return rtps(xb, xa, params.rtps_alpha)
    timings = {}
    t0 = time.perf_counter()
    if model_equivalents is None:
        from .operators import model_equivalents
    frames = {k: np.asarray(v) for k, v in frames.items()}
    n_members = next(iter(frames.values())).shape[0]
    hx_b = np.asarray(model_equivalents(obs, frames, geo) if hx_b is None else hx_b, dtype=np.float64)
    if hx_b.shape != (n_members, len(obs.value)):
        raise ValueError(f"model_equivalents returned {hx_b.shape}, expected {(n_members, len(obs.value))}")
    ok, counters = quality_control(obs, hx_b, params.gross_k)
    withheld = np.zeros(len(ok), bool) if withheld is None else np.asarray(withheld, bool)
    # never assimilated, whatever the caller passes: rows a source marked 'withhold', and cloud rows (no cloud
    # analysis in this build; their O-B/O-A stay as diagnostics)
    n_obs = len(ok)
    role_withheld = np.asarray(getattr(obs, "role", np.full(n_obs, "assimilate", object))) == "withhold"
    cloud = np.asarray(getattr(obs, "kind", np.zeros(n_obs, int))) == CLOUD
    counters["role_withheld"], counters["cloud_not_assimilated"] = int(role_withheld.sum()), int(cloud.sum())
    withheld = withheld | role_withheld | cloud
    use = ok & ~withheld
    timings["hx_qc_s"] = time.perf_counter() - t0

    width, width_source = resolve_spec_bdy_width(geo, params.spec_bdy_width)
    t1 = time.perf_counter()
    xyz, lnp, ch, cv = observation_coordinates(obs, frames, geo, params)
    full_p = np.asarray(frames["P"], dtype=np.float64) + np.asarray(frames["PB"], dtype=np.float64)
    agrid = lgrid.AnalysisGrid.from_mass_grid(geo["XLAT"], geo["XLONG"], np.log(full_p.mean(axis=0)),
                                              stride_h=params.stride_h, stride_v=params.stride_v)
    sel = np.flatnonzero(use)
    idx, rho_loc, n_trunc = lgrid.local_obs_table(agrid.xyz, agrid.lnp, xyz[sel], lnp[sel], ch[sel], cv[sel],
                                                  pmax=params.pmax, workers=params.workers)
    timings["localisation_s"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    names = [name for name in params.variables + params.diagnostic_variables if name in frames]
    groups = variable_groups(names, obs, sel, var_loc or {})
    inflation = _prior_inflation(prior_inflation, agrid, params)
    weights_by_name = {}
    for group, factor in groups:
        rho_g = rho_loc if factor is None else rho_loc * factor[idx]
        weights = lgrid.analysis_weights(hx_b[:, sel], np.asarray(obs.value)[sel], obs_sigma(obs)[sel],
                                         idx, rho_g, inflation=inflation)
        weights = weights.reshape(agrid.shape + (n_members, n_members))
        weights_by_name.update({name: weights for name in group})
    timings["weights_s"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    from .state_io import boundary_taper
    analysis = dict(frames)
    counts = {}
    raw_qv = clipped_qv = None
    for name in names:
        xb = frames[name].astype(np.float64)
        weights = weights_by_name[name]
        stagger = STAGGER.get(name)
        if name == "QVAPOR" and policy.negative == "log":
            zb = mst.to_log(xb, policy.log_floor_kgkg)
            za = maintenance(name, zb, lgrid.transform_field(zb, weights, agrid, stagger=stagger))
            xa = mst.from_log(za, policy.log_floor_kgkg)
        else:
            xa = maintenance(name, xb, lgrid.transform_field(xb, weights, agrid, stagger=stagger))
        ny, nx = xb.shape[-2:]
        taper = np.asarray(boundary_taper(ny - (stagger == "y"), nx - (stagger == "x"),
                                          spec_bdy_width=width, ramp=params.taper_ramp,
                                          stagger=stagger), dtype=np.float64)
        xa = xb + taper * (xa - xb)
        if name in ("QVAPOR", "Q2"):
            counts[f"{name}_nonfinite"] = int((~np.isfinite(xa)).sum())
            xa, raw, neg = mst.negative_step(xa, policy, xb)
            counts[f"{name}_negative"] = neg["negative_cells"]
            counts[f"{name}_negative_min"] = neg["negative_min_kgkg"]
            counts[f"{name}_background_negative"] = neg["background_negative_cells"]
            if "clipped_cells" in neg:
                counts[f"{name}_clipped"] = neg["clipped_cells"]
            if name == "QVAPOR":
                raw_qv, clipped_qv = raw, xa          # float64: the budget must not see the storage rounding
        increment = xa - xb
        counts[f"{name}_increment_absmax"] = float(np.abs(increment).max())
        counts[f"{name}_mean_increment_rms"] = float(np.sqrt(np.mean(increment.mean(axis=0) ** 2)))
        analysis[name] = xa.astype(frames[name].dtype)
    timings["update_s"] = time.perf_counter() - t3

    mass, mass_label = None, None
    if "QVAPOR" in names:
        mass, mass_label = mst.layer_dry_mass(geo, frames.get("MU"), full_p)
    moist = {"policy": policy.as_dict(), "dry_mass": mass_label}
    if "QVAPOR" in names:
        moist["negative"] = {"cells": counts["QVAPOR_negative"], "min_kgkg": counts["QVAPOR_negative_min"],
                             "nonfinite_cells": counts["QVAPOR_nonfinite"],
                             "background_negative_cells_kept": counts["QVAPOR_background_negative"]}
        moist["budget"] = (None if policy.negative == "refuse"
                           else mst.water_budget(raw_qv, clipped_qv, mass, geo))
    if "QVAPOR" in names:                     # 'keep': counted, never changed (cap/to_cloud are refused)
        sat = mst.supersaturation_counts(analysis["QVAPOR"], analysis.get("T", frames["T"]), full_p)
        moist["supersaturation"] = sat
        counts.update({"supersaturated_cells": sat["supersaturated_cells"],
                       "supersaturation_excess_max_kgkg": sat["excess_max_kgkg"]})
    reasons = []
    if "QVAPOR" in names:
        gate_ok, gate_reasons = mst.gate(policy, {"negative_cells": counts["QVAPOR_negative"],
                                                  "nonfinite_cells": counts["QVAPOR_nonfinite"]}, moist["budget"])
        reasons += gate_reasons
    moist["admissible"] = not reasons

    t4 = time.perf_counter()
    admissible = not reasons
    counts["rebalance"] = "skipped (not admissible)" if (params.rebalance and not admissible) else (
        "applied" if params.rebalance else "off")
    if params.rebalance and admissible:
        from .state_io import rebalance_increment
        rebalanced = {name: np.array(frames[name], copy=True) for name in ("P", "PH", "AL", "P_HYD") if name in frames}
        try:
            for m in range(n_members):
                bg = {k: v[m] for k, v in frames.items()}
                an = {k: v[m] for k, v in analysis.items()}
                for name, value in rebalance_increment(bg, an, geo).items():
                    if name in rebalanced:
                        rebalanced[name][m] = np.asarray(value, dtype=rebalanced[name].dtype)
        except ValueError as exc:
            reasons.append(f"rebalance refused member {m}: {exc}")
            admissible = False
            counts["rebalance"] = "refused"
        else:
            analysis.update(rebalanced)
    timings["rebalance_s"] = time.perf_counter() - t4
    if "QVAPOR" in names:
        written = mst.check_written("QVAPOR", analysis["QVAPOR"], frames["QVAPOR"])
        moist["written"] = written
        if admissible and (written["negative"] or written["nonfinite"]):
            reasons.append(f"stored QVAPOR not positive/finite: {written}")
            admissible = False

    hx_a = np.asarray(model_equivalents(obs, analysis, geo), dtype=np.float64)
    now = np.ones(len(obs.value), bool) if at_analysis_time is None else np.asarray(at_analysis_time, bool)
    receipt = {
        "admissible": bool(admissible), "refusal_reasons": reasons,
        "oma_semantics": {"sonde": "H_of_analysis (analysed T/QVAPOR/U/V + rebalanced P/PH)",
                          "surface": "posterior_linear (T2/Q2/U10/V10/PSFC transformed by lowest-level weights; "
                                     "not re-diagnosed; diagnostic only)",
                          "cloud": "not assimilated"},
        "params": asdict(params),
        "spec_bdy_width": {"value": width, "source": width_source, "namelist": geo.get("SPEC_BDY_WIDTH_source")},
        "n_members": int(n_members), "qc": counters,
        "assimilated": int(use.sum()), "withheld_ok": int((ok & withheld).sum()),
        "analysis_grid": {"shape": list(agrid.shape), "points": agrid.n_points},
        "local_obs": {"pmax": params.pmax, "truncated_points": int(n_trunc),
                      "mean_used": float((rho_loc > 0).sum(axis=1).mean()) if len(rho_loc) else 0.0,
                      "max_used": int((rho_loc > 0).sum(axis=1).max()) if len(rho_loc) else 0},
        "hooks": {"prior_inflation": "params.inflation" if prior_inflation is None else "per-point",
                  "maintenance": maintenance_label,
                  "var_loc_groups": [list(g) for g, _ in groups]},
        "moisture": moist,
        "counters": counts, "timings_s": timings,
        "stats_assimilated": _stats(obs, hx_b, hx_a, use & now),
        "stats_withheld": _stats(obs, hx_b, hx_a, ok & withheld & now),
    }
    if not now.all():
        receipt["stats_assimilated_earlier_window"] = _stats(obs, hx_b, hx_b, use & ~now)
        receipt["stats_withheld_earlier_window"] = _stats(obs, hx_b, hx_b, ok & withheld & ~now)
        receipt["fourd"] = {"obs_total": int(len(now)), "obs_at_analysis_time": int(now.sum()),
                            "note": "earlier-window O−A not computed (needs analysed trajectories); 'oma' there = O−B"}
    return analysis, receipt
