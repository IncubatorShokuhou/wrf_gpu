"""Bias-aware (debiased) innovation-based inflation for the LETKF — OPT-IN (lane a1-io, task BA; R3 items 1–2).

R3 measured that the surface innovations are dominated by obs − model BIAS (T2 −1.9 K, 10-m winds +1.3…3.2 m/s)
which the E[d²]-based calibration turns into a large global ρ, and that the real spread deficit is variable- and
height-dependent (T 1–2 km ×3, QV ≈ 1, T2 none). Two options, selected by the config ``ensemble`` block
(``kind: members``) through ``cli.calibrate_scale``:

* ``scale: "innovation_debiased"`` — per (kind, var) group g: d′ = d − mean_g(d) (bias removed), then
  ρ_g = (E[d′²] − E[σo²]) / E[var H(x)] in the group's own units (Desroziers-type consistency E[d′²] = σo² + ρσb²,
  bias removed as in Li et al. 2009). ρ per kind (sonde/surface) and pooled = obs-count-weighted means of ρ_g,
  clamped to [rho_min, rho_max] (default [1, 10]). The pooled ρ goes into the LETKF transform (``inflation_rho``),
  frames unchanged — the drop-in replacement of ``"innovation"``.
* ``per_band: true`` (with the scale above) — STATE-space inflation per variable × height band: obs map to state
  variables (sonde T→T, QV→QVAPOR, U→U, V→V; surface T2→T, Q2→QVAPOR, U10→U, V10→V, all surface obs in 0–1 km) and
  to the band of their height AGL (0–1, 1–2, 2–5, >5 km; sonde heights from p in the member-mean column). ρ_vb from
  the debiased group (n ≥ n_min, default 6), else the variable's all-band ρ, else the pooled ρ; clamped. Member
  deviations of state variable v at levels in band b become mean + √ρ_vb·(x − mean); T2/Q2/U10/V10 take the 0–1 km
  factor of T/QVAPOR/U/V so the surface operators see a consistent prior; P/PH/MU/PSFC are not inflated. The factor
  is ramped with the analysis boundary taper (1 in the specified zone, cells there stay BITWISE). Mapping: for a
  linear H that senses only (v, b), inflating those deviations by √ρ multiplies var(H(x)) by ρ, i.e. reproduces the
  obs-space ρ of that group; obs that mix bands/variables see the weighted mixture. ``inflation_rho`` = 1 (already
  applied); RTPS then relaxes toward the INFLATED prior spread (R3 item 4). QVAPOR/Q2 are inflated in LOG space
  (log deviations × √ρ, re-centred multiplicatively: positive by construction, ensemble mean kept — archive_ensemble
  convention; the linear variance grows by ≈ ρ for small relative spread); a negative count is still reported.
* Pooling (per kind, all obs, per band) averages the CLAMPED group values weighted by obs count: a group with
  E[d′²] < E[σo²] (raw ρ < 1: its σo is over-estimated) contributes ρ = 1 instead of dragging the pool negative.
  Groups with n < n_min (default 6) are reported but NOT pooled (after removing the mean, n = 2 leaves one degree of
  freedom: 0228 surface V10 raw ρ 69). Per (state var, band): every reliable obs group with obs in the band
  contributes its own band estimate (n ≥ n_min) or its all-band value; a band without obs takes the variable's
  groups, else the pooled value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BANDS = ((0.0, 1000.0), (1000.0, 2000.0), (2000.0, 5000.0), (5000.0, np.inf))
STATE_OF_OBS = {(1, 1): "T", (1, 2): "QVAPOR", (1, 3): "U", (1, 4): "V",
                (2, 11): "T", (2, 12): "QVAPOR", (2, 13): "U", (2, 14): "V"}
DIAG_OF_STATE = {"T": "T2", "QVAPOR": "Q2", "U": "U10", "V": "V10"}
KIND_NAMES = {1: "sonde", 2: "surface", 3: "cloud"}
VAR_NAMES = {1: "T", 2: "QV", 3: "U", 4: "V", 11: "T2", 12: "Q2", 13: "U10", 14: "V10"}
G = 9.81


def group_rho(d, sigma_o, var_hx) -> float:
    """(E[d²] − E[σo²]) / E[var H(x)] for one homogeneous group (d already debiased)."""
    v = float(np.mean(var_hx))
    return (float(np.mean(np.square(d))) - float(np.mean(np.square(sigma_o)))) / v if v > 0 else np.nan


def _clamp(x, lo, hi):
    return float(np.clip(x, lo, hi)) if np.isfinite(x) else float(lo)


def _band_index(z_agl) -> np.ndarray:
    z = np.asarray(z_agl, np.float64)
    idx = np.full(z.shape, len(BANDS) - 1, dtype=np.int64)
    for b, (lo, hi) in reversed(list(enumerate(BANDS))):
        idx = np.where((z >= lo) & (z < hi), b, idx)
    return idx


def obs_height_agl(obs, frames, geo) -> np.ndarray:
    """Height AGL of each obs: surface obs 0; sonde from p in the member-mean nearest column (ln p interpolation)."""
    from .analysis import _nearest_columns
    kind = np.asarray(obs.kind)
    h = np.zeros(len(kind), np.float64)
    sel = np.flatnonzero(np.isfinite(np.asarray(obs.p_pa, np.float64)) & (kind == 1))
    if sel.size:
        j, i = _nearest_columns(geo, np.asarray(obs.lat)[sel], np.asarray(obs.lon)[sel])
        p = (np.asarray(frames["P"], np.float64) + np.asarray(frames["PB"], np.float64)).mean(axis=0)
        z = (np.asarray(frames["PH"], np.float64) + np.asarray(frames["PHB"], np.float64)).mean(axis=0) / G
        zm = 0.5 * (z[1:] + z[:-1]) - np.asarray(geo["HGT"], np.float64)[None]
        for n, jj, ii in zip(sel, j, i):
            h[n] = np.interp(-np.log(obs.p_pa[n]), -np.log(p[:, jj, ii]), zm[:, jj, ii])
    return h


def debiased_table(obs, hx, ok, frames=None, geo=None, *, rho_min=1.0, rho_max=10.0, n_min=6, per_band=False):
    """Per-group debiased ρ (raw and clamped), per kind, pooled; optionally per (state var, band)."""
    from .analysis import obs_sigma
    kind, var = np.asarray(obs.kind), np.asarray(obs.var)
    sig = np.asarray(obs_sigma(obs), np.float64)
    d = np.asarray(obs.value, np.float64) - hx.mean(axis=0)
    vh = hx.var(axis=0, ddof=1)
    dd = np.full_like(d, np.nan)
    groups = {}
    for k, v in sorted({(int(a), int(b)) for a, b in zip(kind[ok], var[ok])}):
        sel = ok & (kind == k) & (var == v)
        bias = float(d[sel].mean())
        dd[sel] = d[sel] - bias
        raw = group_rho(dd[sel], sig[sel], vh[sel])
        groups[f"{KIND_NAMES.get(k, k)}:{VAR_NAMES.get(v, v)}"] = {
            "n": int(sel.sum()), "bias": bias, "rho_raw": raw, "rho": _clamp(raw, rho_min, rho_max),
            "rho_with_bias_raw": group_rho(d[sel], sig[sel], vh[sel]),
            "sigma_o_rms": float(np.sqrt(np.mean(np.square(sig[sel])))),
            "debiased_omb_rms": float(np.sqrt(np.mean(np.square(dd[sel])))),
            "spread_b_rms": float(np.sqrt(np.mean(vh[sel]))), "reliable": bool(sel.sum() >= n_min)}

    def pooled(names):  # count-weighted mean of the CLAMPED values of the RELIABLE groups (n ≥ n_min)
        rows = [groups[g] for g in names if groups[g]["reliable"]]
        n = sum(r["n"] for r in rows)
        rho = sum(r["n"] * r["rho"] for r in rows) / n if n else float(rho_min)
        return {"n": int(n), "rho": _clamp(rho, rho_min, rho_max), "groups_used": [g for g in names if groups[g]["reliable"]],
                "rho_raw_weighted": (sum(r["n"] * r["rho_raw"] for r in rows if np.isfinite(r["rho_raw"])) / n) if n else None}
    by_kind = {kn: pooled([g for g in groups if g.startswith(kn + ":")]) for kn in sorted({g.split(":")[0] for g in groups})}
    out = {"groups": groups, "by_kind": by_kind, "pooled": pooled(list(groups)),
           "rho_min": rho_min, "rho_max": rho_max, "n_min": n_min}
    if per_band:
        h = obs_height_agl(obs, frames, geo)
        band = _band_index(h)
        gname = np.array([f"{KIND_NAMES.get(int(a), a)}:{VAR_NAMES.get(int(b), b)}" for a, b in zip(kind, var)])
        state = np.array([STATE_OF_OBS.get((int(a), int(b)), "") for a, b in zip(kind, var)])
        table = {}
        for sv in ("T", "QVAPOR", "U", "V"):
            gs = sorted({g for g in gname[ok & (state == sv)] if groups[g]["reliable"]})
            for b, (lo, hi) in enumerate(BANDS):
                num = den = 0.0
                parts = {}
                for g in gs:  # (obs group × band): own estimate if n ≥ n_min, else the group's all-band value
                    sel = ok & (gname == g) & (band == b) & np.isfinite(dd)
                    n = int(sel.sum())
                    if n == 0:
                        continue
                    raw = group_rho(dd[sel], sig[sel], vh[sel]) if n >= n_min else np.nan
                    r = _clamp(raw, rho_min, rho_max) if np.isfinite(raw) else groups[g]["rho"]
                    parts[g] = {"n": n, "rho": r, "source": "group×band" if np.isfinite(raw) else "group"}
                    num += n * r
                    den += n
                if den:
                    rho, src = num / den, "obs in band"
                else:  # no obs in this band: the variable's groups (all bands), else the pooled value
                    rows = [groups[g] for g in gs]
                    nn = sum(x["n"] for x in rows)
                    rho, src = ((sum(x["n"] * x["rho"] for x in rows) / nn, "variable groups (no obs in band)")
                                if nn else (out["pooled"]["rho"], "pooled"))
                table[f"{sv}|{lo/1000:.0f}-{'inf' if not np.isfinite(hi) else f'{hi/1000:.0f}'}km"] = {
                    "n": int(den), "rho": _clamp(rho, rho_min, rho_max), "source": src, "parts": parts}
        out["per_band"] = table
    return out


REBALANCE_KEYS = ("T", "QVAPOR", "MU", "P", "PH")


def inflate_state(frames, geo, table, *, spec_bdy_width=None, ramp=5, rebalance=True):
    """Apply the per-(var, band) ρ to member deviations (state space). Returns (new frames, counters).

    ``rebalance`` (default): the T/QVAPOR inflation increment of every member is carried into its P/PH (+ AL,
    P_HYD) with the same fixed-MU incremental rebalance the analysis applies to its own increment
    (``state_io.rebalance_increment`` against the RAW member). Without it the inflated prior keeps the raw
    member's P/PH and the analysis — which rebalances only analysis-minus-prior — writes members whose mass
    fields miss the inflation increment (critic cr-w41-da-ens P1: up to 794 m²/s² PH at ρ = 4). Frames without
    T/QVAPOR/MU/P/PH (synthetic tests) skip it, recorded in the counters.

    The protected boundary zone is the CASE ``spec_bdy_width`` (``analysis.resolve_spec_bdy_width``: geo
    SPEC_BDY_WIDTH from the case namelist; an explicit ``spec_bdy_width`` must agree with it), the same zone the
    analysis and the member writer keep untouched."""
    from .analysis import resolve_spec_bdy_width
    from .state_io import boundary_taper, rebalance_increment
    raw = frames
    frames = dict(frames)
    spec_bdy_width, width_source = resolve_spec_bdy_width(geo or {}, spec_bdy_width)
    z = (np.asarray(frames["PH"], np.float64) + np.asarray(frames["PHB"], np.float64)).mean(axis=0) / G
    zm = 0.5 * (z[1:] + z[:-1]) - np.asarray(geo["HGT"], np.float64)[None]
    band = _band_index(zm)  # (nz, ny, nx) mass levels
    ny, nx = band.shape[-2:]
    counters = {"spec_bdy_width": spec_bdy_width, "spec_bdy_width_source": width_source}

    def factor_field(sv, shape, stagger=None):
        rho = np.array([table[k]["rho"] for k in table if k.startswith(sv + "|")])  # in BANDS order
        f_mass = np.sqrt(rho)[band]
        if stagger == "x":
            f = np.concatenate((f_mass[..., :1], 0.5 * (f_mass[..., 1:] + f_mass[..., :-1]), f_mass[..., -1:]), axis=-1)
        elif stagger == "y":
            f = np.concatenate((f_mass[:, :1], 0.5 * (f_mass[:, 1:] + f_mass[:, :-1]), f_mass[:, -1:]), axis=-2)
        else:
            f = f_mass
        w = boundary_taper(ny, nx, spec_bdy_width, ramp, stagger)
        return 1.0 + (f - 1.0) * w[None], w

    def inflate(name, fac, w):
        x = np.asarray(frames[name])
        x64 = x.astype(np.float64)
        mean = x64.mean(axis=0)
        if name in ("QVAPOR", "Q2"):  # log space: positive, ensemble mean kept (archive_ensemble convention)
            from .archive_ensemble import _log_members
            new = _log_members(mean, x64, fac)
            counters[f"{name}_negative_after_inflation"] = int((new < 0).sum())
        else:
            new = mean[None] + fac[None] * (x64 - mean[None])
        out = new.astype(x.dtype)
        keep = np.broadcast_to(w == 0, x.shape[-2:])  # specified zone: original bytes
        frames[name] = np.where(keep, x, out)

    for sv, stagger in (("T", None), ("QVAPOR", None), ("U", "x"), ("V", "y")):
        if sv not in frames:
            continue
        fac, w = factor_field(sv, frames[sv].shape[1:], stagger)
        inflate(sv, fac, w)
        diag = DIAG_OF_STATE[sv]
        if diag in frames:
            rho0 = table[f"{sv}|0-1km"]["rho"]
            w2 = boundary_taper(ny, nx, spec_bdy_width, ramp)
            inflate(diag, 1.0 + (np.sqrt(rho0) - 1.0) * w2, w2)
    if not rebalance:
        counters["rebalance"] = "off"
    elif not all(k in raw for k in REBALANCE_KEYS):
        counters["rebalance"] = "not applicable (frames without " + "/".join(k for k in REBALANCE_KEYS if k not in raw) + ")"
    else:
        names = [k for k in ("P", "PH", "AL", "P_HYD") if k in raw]
        new = {k: np.array(raw[k], copy=True) for k in names}
        for m in range(np.asarray(raw["T"]).shape[0]):
            bg = {k: v[m] for k, v in raw.items()}
            an = {k: v[m] for k, v in frames.items()}
            for k, value in rebalance_increment(bg, an, geo).items():   # ValueError propagates (no silent skip)
                if k in new:
                    new[k][m] = np.asarray(value, dtype=new[k].dtype)
        frames.update(new)
        counters["rebalance"] = "applied (" + "/".join(names) + ")"
    return frames, counters


def calibrate_debiased(frames, geo, obs, spec):
    """``cli.calibrate_scale`` hook for ``scale: "innovation_debiased"`` (kind members). Returns (frames, info)."""
    from .analysis import quality_control
    from .operators import model_equivalents

    hx = np.asarray(model_equivalents(obs, frames, geo), dtype=np.float64)
    ok, _ = quality_control(obs, hx, 1.0e9)  # finite/valid only, as the "innovation" calibration
    per_band = bool(spec.get("per_band", False))
    table = debiased_table(obs, hx, ok, frames, geo, rho_min=float(spec.get("rho_min", 1.0)),
                           rho_max=float(spec.get("rho_max", 10.0)), n_min=int(spec.get("n_min", 6)),
                           per_band=per_band)
    info = {"method": "innovation_debiased" + ("+per_band(state)" if per_band else ""), "n": int(ok.sum()),
            "debiased": table, "by_kind": {k: v["rho"] for k, v in table["by_kind"].items()}}
    if per_band:
        frames, counters = inflate_state(frames, geo, table["per_band"])
        info.update({"inflation_rho": 1.0, "state_inflation_counters": counters,
                     "scale": float(np.sqrt(table["pooled"]["rho"]))})
    else:
        info.update({"inflation_rho": table["pooled"]["rho"], "scale": float(np.sqrt(table["pooled"]["rho"]))})
    return frames, info


# --------------------------------------------------------------------------- adaptive (cycled) inflation + obs bias
# Lane w41-da-ens (v0.4.1, R3 items 1-2 + BA "next steps"): the per-cycle debiased band table above is
# under-sampled (1-4 superobbed sonde obs per band) and hard-clamped. The adaptive version treats every
# (obs group x band) estimate as a noisy measurement of the needed inflation of (state variable, band) and fuses it
# with the previous cycle's value (Miyoshi 2011, QJRMS 137, 1813-1827, Gaussian update; Li et al. 2009 for the
# innovation statistic):
#   rho_o = (sum d'^2 / m - E[so^2]) / E[var H(xb)]   d' = debiased innovation, raw (uninflated) prior spread,
#                                                      m = n - 1 when the bias is this cycle's group mean, else n
#   v_o   = 2/n_eff * ((rho_b*sb^2 + so^2) / sb^2)^2  sampling variance of rho_o for Gaussian d'; n_eff = m scaled
#                                                      to the number of distinct profiles (station|site|launch):
#                                                      the levels of one sonde are not independent samples
#   rho_a = (rho_b/v_b + sum_k rho_o,k/v_o,k) / (1/v_b + sum_k 1/v_o,k),  v_a = 1/(1/v_b + sum_k 1/v_o,k)
# Sparse or low-signal groups (surface T2: sb 0.065 K vs so 1.75 K) get a huge v_o and leave the prior almost
# unchanged — shrinkage replaces the hard [1, 10] clamp (the clamp stays as a safety bound). Between cycles the
# estimate persists with a random-walk variance q (prior of the next cycle: rho_b = rho_a, v_b = v_a + q).

def _band_key(sv, b):
    lo, hi = BANDS[b]
    return f"{sv}|{lo/1000:.0f}-{'inf' if not np.isfinite(hi) else f'{hi/1000:.0f}'}km"


BAND_KEYS = tuple(_band_key(sv, b) for sv in ("T", "QVAPOR", "U", "V") for b in range(len(BANDS)))


class ObsBias:
    """Online observation-minus-background bias per group (scalar Kalman filter per key) — OPT-IN.

    key = 'kind:VAR' (``by='group'``) or 'kind:VAR:station' (``by='station'``, station/site id; elevation-aware
    because every station keeps its own offset). Update with this cycle's innovations d = y - mean H(xb):
      v_b <- v + q;  K = v_b / (v_b + e);  b <- b + K (mean(d) - b);  v <- (1 - K) v_b
      e = var_members(mean_key H(x)) + s_d^2 / n
    The first term is the ensemble's own estimate of the forecast error that is COHERENT over the key's
    observations (a domain-wide flow error looks exactly like a group bias within one cycle — Swiss twin OSSE,
    cycle 0: an unbiased V10 group got a -2.1 m/s "bias" without it); s_d^2 (floor so^2) is the sample variance of d.
    The prior variance and the per-cycle random walk are relative to the group's obs error variance: v0 = var0·so²,
    q = q·so² (unit-free across T/Q/wind). ``coherent=False`` drops the coherent term (the first estimator version,
    lane L96 table v1; also the mutant control). ``n_min`` defaults to 2 obs per key and cycle for groups, 1 for
    stations. :meth:`correct` subtracts b from the observation VALUES of
    the kinds in ``apply_kinds`` (default: surface only), so the analysis is not pulled toward a systematic
    obs-model offset (R3: T2 -1.88 K, 10-m winds +1.3...3.2 m/s). Semantics: the systematic part is treated as an
    observation/representativeness bias; the model bias itself is NOT corrected by this option.
    """

    def __init__(self, groups=None, *, by="group", apply_kinds=(2,), var0=0.25, q=0.01, n_min=None, coherent=True):
        if by not in ("group", "station"):
            raise ValueError("ObsBias.by must be 'group' or 'station'")
        self.groups = dict(groups or {})
        self.by, self.apply_kinds = by, tuple(int(k) for k in apply_kinds)
        self.var0, self.q, self.coherent = float(var0), float(q), bool(coherent)
        self.n_min = int(n_min) if n_min is not None else (2 if by == "group" else 1)

    def keys(self, obs) -> np.ndarray:
        kind, var = np.asarray(obs.kind), np.asarray(obs.var)
        base = [f"{KIND_NAMES.get(int(k), int(k))}:{VAR_NAMES.get(int(v), int(v))}" for k, v in zip(kind, var)]
        if self.by == "station":
            sid = [str(s) or str(t) for s, t in zip(np.asarray(obs.station), np.asarray(obs.site))]
            base = [f"{b}:{s}" for b, s in zip(base, sid)]
        return np.array(base, dtype=object)

    def update(self, obs, hx, ok) -> dict:
        from .analysis import obs_sigma
        d = np.asarray(obs.value, np.float64) - np.asarray(hx, np.float64).mean(axis=0)
        sig = np.asarray(obs_sigma(obs), np.float64)
        keys = self.keys(obs)
        receipt = {}
        for key in sorted(set(keys[ok])):
            sel = ok & (keys == key) & np.isfinite(d)
            n = int(sel.sum())
            if n < self.n_min:
                continue
            so2 = float(np.mean(sig[sel] ** 2))
            row = self.groups.get(key, {"bias": 0.0, "var": self.var0 * so2, "n_total": 0, "cycles": 0})
            mean = float(d[sel].mean())
            s2 = max(float(d[sel].var(ddof=1)) if n > 1 else 0.0, so2)
            coherent = float(np.asarray(hx, np.float64)[:, sel].mean(axis=1).var(ddof=1)) if self.coherent else 0.0
            vb = float(row["var"]) + self.q * so2
            gain = vb / (vb + coherent + s2 / n)
            new = {"bias": float(row["bias"]) + gain * (mean - float(row["bias"])), "var": (1.0 - gain) * vb,
                   "n_total": int(row["n_total"]) + n, "cycles": int(row["cycles"]) + 1}
            receipt[key] = {"n": n, "sample_mean": mean, "gain": gain, "coherent_fc_var": coherent,
                            "bias_before": float(row["bias"]),
                            "bias": new["bias"], "sd": float(np.sqrt(new["var"]))}
            self.groups[key] = new
        return receipt

    def bias_of(self, obs) -> np.ndarray:
        keys = self.keys(obs)
        return np.array([float(self.groups.get(k, {}).get("bias", 0.0)) for k in keys], np.float64)

    def correct(self, obs):
        """Return (obs with debiased VALUES for ``apply_kinds``, receipt). Other kinds keep their bytes."""
        import dataclasses
        b = self.bias_of(obs)
        apply = np.isin(np.asarray(obs.kind), self.apply_kinds) & (b != 0.0)
        value = np.asarray(obs.value, np.float64).copy()
        value[apply] -= b[apply]
        if dataclasses.is_dataclass(obs):
            out = dataclasses.replace(obs, value=value)
        else:  # duck-typed batches (tests, probes): shallow copy with the new values
            import copy
            out = copy.copy(obs)
            out.value = value
        keys = self.keys(obs)
        return out, {"applied": int(apply.sum()), "apply_kinds": list(self.apply_kinds), "by": self.by,
                     "bias_by_key": {k: float(self.groups[k]["bias"]) for k in sorted(set(keys[apply]))}}

    def to_dict(self) -> dict:
        return {"schema": "w41.obs_bias.v2", "by": self.by, "apply_kinds": list(self.apply_kinds), "var0": self.var0,
                "q": self.q, "n_min": self.n_min, "coherent": self.coherent, "groups": self.groups}

    @classmethod
    def from_dict(cls, d) -> "ObsBias":
        if d.get("schema") != "w41.obs_bias.v2":   # v1 (absolute priors, no coherent term): re-initialise
            raise ValueError(f"unknown ObsBias schema {d.get('schema')!r} (expected w41.obs_bias.v2)")
        return cls(d["groups"], by=d["by"], apply_kinds=d["apply_kinds"], var0=d["var0"], q=d["q"], n_min=d["n_min"],
                   coherent=d["coherent"])


def _profile_ids(obs):
    """Per-obs profile id (station|site|launch) or None when the batch carries no station/site ids."""
    if not hasattr(obs, "station"):
        return None
    n = len(np.asarray(obs.value))
    cols = [np.asarray(getattr(obs, f, np.full(n, "")), object) for f in ("station", "site", "launch")]
    if not any(str(a) or str(b) for a, b in zip(cols[0], cols[1])):
        return None
    return np.array(["|".join(str(c[i]) for c in cols) for i in range(n)], object)


class AdaptiveInflation:
    """Cycled per-(state variable, height band) multiplicative prior inflation (state space) — OPT-IN.

    ``rho``/``var`` map BAND_KEYS ('T|1-2km', …) to the current estimate and its variance. :meth:`update` returns a
    BA-format band table ({key: {'rho': ...}}) for :func:`inflate_state`; the transform then runs with ρ = 1 and
    RTPS relaxes toward the inflated prior (the written members carry the inflation, R3 item 4).
    """

    def __init__(self, rho=None, var=None, *, rho0=1.0, var0=1.0, q=0.04, rho_min=1.0, rho_max=10.0, n_min=2):
        self.rho = {k: float(rho0) for k in BAND_KEYS} | dict(rho or {})
        self.var = {k: float(var0) for k in BAND_KEYS} | dict(var or {})
        self.q, self.rho_min, self.rho_max, self.n_min = float(q), float(rho_min), float(rho_max), int(n_min)
        self.cycles = 0

    def prior(self) -> tuple[dict, dict]:
        """(ρ_b, v_b) for this cycle: the persisted estimate, variance inflated by the random-walk q."""
        return dict(self.rho), {k: v + (self.q if self.cycles else 0.0) for k, v in self.var.items()}

    def update(self, obs, hx, ok, frames, geo, *, bias: np.ndarray | None = None) -> dict:
        """Fuse this cycle's debiased innovation statistics. ``hx`` from the RAW prior members; ``bias`` (n_obs,)
        an external estimate (e.g. :class:`ObsBias`), else each (kind, var) group's own mean is removed."""
        from .analysis import obs_sigma
        kind, var = np.asarray(obs.kind), np.asarray(obs.var)
        sig = np.asarray(obs_sigma(obs), np.float64)
        hx = np.asarray(hx, np.float64)
        d = np.asarray(obs.value, np.float64) - hx.mean(axis=0)
        vh = hx.var(axis=0, ddof=1)
        gname = np.array([f"{KIND_NAMES.get(int(a), a)}:{VAR_NAMES.get(int(b), b)}" for a, b in zip(kind, var)])
        state = np.array([STATE_OF_OBS.get((int(a), int(b)), "") for a, b in zip(kind, var)])
        dd = np.full_like(d, np.nan)
        lost_dof = {}
        for g in sorted(set(gname[ok])):
            sel = ok & (gname == g) & np.isfinite(d)
            if bias is None:
                dd[sel] = d[sel] - d[sel].mean()
                lost_dof[g] = 1
            else:
                dd[sel] = d[sel] - np.asarray(bias, np.float64)[sel]
                lost_dof[g] = 0
        band = _band_index(obs_height_agl(obs, frames, geo))
        sid = _profile_ids(obs)
        rho_b, var_b = self.prior()
        table, estimates = {}, {}
        for sv in ("T", "QVAPOR", "U", "V"):
            for b in range(len(BANDS)):
                key = _band_key(sv, b)
                prec, num, parts = 1.0 / var_b[key], rho_b[key] / var_b[key], {}
                for g in sorted(set(gname[ok & (state == sv)])):
                    sel = ok & (gname == g) & (band == b) & np.isfinite(dd) & (vh > 0)
                    n = int(sel.sum())
                    n_grp = int((ok & (gname == g)).sum())
                    m_eff = n - lost_dof.get(g, 0) * n / max(n_grp, 1)       # removed group mean: its dof share
                    # the levels of one profile are not independent samples of d': the sampling variance of the
                    # estimate uses the number of distinct profiles (station+launch) in the band, when ids exist
                    n_eff = m_eff if sid is None else len(set(sid[sel])) * m_eff / max(n, 1)
                    if n < self.n_min or n_eff <= 0.5:
                        continue
                    sb2, so2 = float(np.mean(vh[sel])), float(np.mean(sig[sel] ** 2))
                    rho_o = (float(np.sum(dd[sel] ** 2)) / m_eff - so2) / sb2   # unbiased 2nd moment of d'
                    v_o = 2.0 / n_eff * ((rho_b[key] * sb2 + so2) / sb2) ** 2
                    prec += 1.0 / v_o
                    num += rho_o / v_o
                    parts[g] = {"n": n, "n_eff": n_eff, "rho_obs": rho_o, "sd_obs": float(np.sqrt(v_o)), "sb": float(np.sqrt(sb2)),
                                "so": float(np.sqrt(so2))}
                rho_a = num / prec
                clamped = float(np.clip(rho_a, self.rho_min, self.rho_max))
                estimates[key] = {"rho_prior": rho_b[key], "sd_prior": float(np.sqrt(var_b[key])),
                                  "rho_posterior_raw": float(rho_a), "rho": clamped, "sd": float(np.sqrt(1.0 / prec)),
                                  "parts": parts}
                self.rho[key], self.var[key] = clamped, 1.0 / prec
                table[key] = {"rho": clamped, "n": int(sum(p["n"] for p in parts.values())),
                              "source": "adaptive (prior + obs)" if parts else "adaptive (prior only)"}
        self.cycles += 1
        return {"table": table, "estimates": estimates, "cycle": self.cycles,
                "debias": "external bias" if bias is not None else "group mean (m = n - 1)",
                "n_eff": "distinct profiles" if sid is not None else "obs count"}

    def to_dict(self) -> dict:
        return {"schema": "w41.adaptive_inflation.v1", "rho": self.rho, "var": self.var, "q": self.q,
                "rho_min": self.rho_min, "rho_max": self.rho_max, "n_min": self.n_min, "cycles": self.cycles}

    @classmethod
    def from_dict(cls, d) -> "AdaptiveInflation":
        if d.get("schema") != "w41.adaptive_inflation.v1":
            raise ValueError("unknown AdaptiveInflation schema")
        out = cls(d["rho"], d["var"], q=d["q"], rho_min=d["rho_min"], rho_max=d["rho_max"], n_min=d["n_min"])
        out.cycles = int(d["cycles"])
        return out


# --------------------------------------------------------------------------- product integration (da-core API)
# Config section ``inflation`` (gpuwrf.da.v1, delegated by config.py) — every key optional; absent section = probe path:
#   {"method": "probe" | "adaptive",
#    "adaptive": {"state_in", "state_out", "rho0", "var0", "q", "rho_min", "rho_max", "n_min", "spec_bdy_width", "ramp"},
#    "obs_bias": {"by": "group"|"station", "apply_kinds": ["surface"], "correct": true, "state_in", "state_out",
#                 "var0", "q"},
#    "maintenance": {"method": "rtps"|"rtpp"|"multiplicative"|"none", "alpha": 0.8, "target": "raw"|"inflated",
#                    "rho": 1.0 | "calibrated" (the transform rho of this analysis' probe calibration), "factor": 1.0},
#    "additive": {"pool_npz": path, "amplitude": float | {name: float}, "seed": int}}
_KIND_CODES = {"sonde": 1, "surface": 2, "cloud": 3}


def _num(where, value, lo=None, hi=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{where}: must be a finite number")
    if integer and int(value) != value:
        raise ValueError(f"{where}: must be an integer")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f"{where}: must lie in [{lo}, {hi}]")
    return int(value) if integer else float(value)


def _only(where, value, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{where}: must be an object")
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError(f"{where}: unknown keys {sorted(extra)}")
    return value


def _opt_path(where, value, base_dir=None):
    """Absolute path string (or None). A relative path is resolved against ``base_dir`` (the config file's
    directory) and refused without one — never against the process CWD."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: must be a path string or null")
    path = Path(value).expanduser()
    if not path.is_absolute():
        if base_dir is None:
            raise ValueError(f"{where}: must be an absolute path (relative paths are not resolved against the "
                             f"working directory), got {value!r}")
        path = Path(base_dir) / path
    return str(path)


def validate_config_section(section, *, base_dir=None) -> dict:
    """Strict, normalised ``inflation`` config section (raises ValueError; config.py wraps it). File paths must be
    absolute, or relative to ``base_dir`` when the caller passes the config file's directory."""
    sec = _only("inflation", section, {"method", "adaptive", "obs_bias", "maintenance", "additive"})
    method = sec.get("method", "probe")
    if method not in ("probe", "adaptive"):
        raise ValueError("inflation.method: must be 'probe' or 'adaptive'")
    out = {"method": method}
    if method == "adaptive" or "adaptive" in sec:
        a = _only("inflation.adaptive", sec.get("adaptive", {}),
                  {"state_in", "state_out", "rho0", "var0", "q", "rho_min", "rho_max", "n_min", "spec_bdy_width", "ramp"})
        ad = {"state_in": _opt_path("inflation.adaptive.state_in", a.get("state_in"), base_dir),
              "state_out": _opt_path("inflation.adaptive.state_out", a.get("state_out"), base_dir),
              "rho0": _num("inflation.adaptive.rho0", a.get("rho0", 1.0), lo=0.1),
              "var0": _num("inflation.adaptive.var0", a.get("var0", 1.0), lo=1e-6),
              "q": _num("inflation.adaptive.q", a.get("q", 0.04), lo=0.0),
              "rho_min": _num("inflation.adaptive.rho_min", a.get("rho_min", 1.0), lo=0.1),
              "rho_max": _num("inflation.adaptive.rho_max", a.get("rho_max", 10.0), lo=1.0),
              "n_min": _num("inflation.adaptive.n_min", a.get("n_min", 2), lo=2, integer=True),
              "spec_bdy_width": (None if a.get("spec_bdy_width") is None else   # None: the case width
                                 _num("inflation.adaptive.spec_bdy_width", a["spec_bdy_width"], lo=0, integer=True)),
              "ramp": _num("inflation.adaptive.ramp", a.get("ramp", 5), lo=0, integer=True)}
        if ad["rho_min"] > ad["rho_max"]:
            raise ValueError("inflation.adaptive: rho_min > rho_max")
        out["adaptive"] = ad
    if "obs_bias" in sec:
        b = _only("inflation.obs_bias", sec["obs_bias"], {"by", "apply_kinds", "correct", "state_in", "state_out",
                                                           "var0", "q"})
        by = b.get("by", "group")
        if by not in ("group", "station"):
            raise ValueError("inflation.obs_bias.by: must be 'group' or 'station'")
        kinds = b.get("apply_kinds", ["surface"])
        if not isinstance(kinds, list) or any(k not in _KIND_CODES for k in kinds):
            raise ValueError("inflation.obs_bias.apply_kinds: list of 'sonde'|'surface'|'cloud'")
        correct = b.get("correct", True)
        if not isinstance(correct, bool):
            raise ValueError("inflation.obs_bias.correct: must be true/false")
        out["obs_bias"] = {"by": by, "apply_kinds": list(kinds), "correct": correct,
                           "state_in": _opt_path("inflation.obs_bias.state_in", b.get("state_in"), base_dir),
                           "state_out": _opt_path("inflation.obs_bias.state_out", b.get("state_out"), base_dir),
                           "var0": _num("inflation.obs_bias.var0", b.get("var0", 0.25), lo=1e-9),
                           "q": _num("inflation.obs_bias.q", b.get("q", 0.01), lo=0.0)}
        if method != "adaptive":
            raise ValueError("inflation.obs_bias needs method 'adaptive'")
    if "maintenance" in sec:
        m = _only("inflation.maintenance", sec["maintenance"], {"method", "alpha", "target", "rho", "factor"})
        meth = m.get("method", "rtps")
        if meth not in ("rtps", "rtpp", "multiplicative", "none"):
            raise ValueError("inflation.maintenance.method: rtps|rtpp|multiplicative|none")
        target = m.get("target", "raw")
        if target not in ("raw", "inflated"):
            raise ValueError("inflation.maintenance.target: raw|inflated")
        rho = m.get("rho", 1.0)
        rho = rho if rho == "calibrated" else _num("inflation.maintenance.rho", rho, lo=0.0)
        out["maintenance"] = {"method": meth, "alpha": _num("inflation.maintenance.alpha", m.get("alpha", 0.8), 0, 1),
                              "target": target, "rho": rho,
                              "factor": _num("inflation.maintenance.factor", m.get("factor", 1.0), lo=0.0)}
    if "additive" in sec:
        a = _only("inflation.additive", sec["additive"], {"pool_npz", "amplitude", "seed"})
        if "pool_npz" not in a or "seed" not in a:
            raise ValueError("inflation.additive: needs pool_npz and seed")
        amp = a.get("amplitude", 0.2)
        amp = ({str(k): _num(f"inflation.additive.amplitude.{k}", v, lo=0.0) for k, v in amp.items()}
               if isinstance(amp, dict) else _num("inflation.additive.amplitude", amp, lo=0.0))
        out["additive"] = {"pool_npz": _opt_path("inflation.additive.pool_npz", a["pool_npz"], base_dir), "amplitude": amp,
                           "seed": _num("inflation.additive.seed", a["seed"], lo=0, integer=True)}
    return out


def read_plan(section) -> list:
    """Files the ``inflation`` section reads (state/bias JSON inputs, additive pool) — for governance planning."""
    if not section:
        return []
    paths = [section.get("adaptive", {}).get("state_in"), section.get("obs_bias", {}).get("state_in"),
             section.get("additive", {}).get("pool_npz")]
    return [p for p in paths if p]


_CALIBRATED = {"rho": None}   # transform rho of the latest calibrate() in this process (maintenance rho 'calibrated')


def calibrate(frames, geo, obs, ensemble_section, inflation_section):
    """da-core hook: (frames, info). Absent section or method 'probe' → the probe ``api.calibrate_scale`` (bitwise
    unchanged). 'adaptive': H(xb) from the RAW prior → ObsBias update (+ obs correction, returned in info['obs'])
    → AdaptiveInflation Gaussian update with the debiased innovations → state-space inflation per (variable,
    band) → info['inflation_rho'] = 1 (the analysis' RTPS then relaxes toward the inflated prior)."""
    import json

    _CALIBRATED["rho"] = None
    if not inflation_section or inflation_section.get("method", "probe") == "probe":
        from .api import calibrate_scale
        frames, info = calibrate_scale(frames, geo, obs, ensemble_section)
        _CALIBRATED["rho"] = info.get("inflation_rho")
        return frames, info
    scale = (ensemble_section or {}).get("scale", 1.0)
    if (ensemble_section or {}).get("kind") != "members" or scale not in (1.0, 1):
        raise ValueError("inflation.method 'adaptive' needs ensemble kind 'members' with scale 1 (no probe rescaling)")
    from .analysis import quality_control
    from .operators import model_equivalents

    hx = np.asarray(model_equivalents(obs, frames, geo), dtype=np.float64)
    ok, _ = quality_control(obs, hx, 1.0e9)
    info = {"method": "adaptive (Miyoshi-2011 Gaussian update, debiased, state var x band)", "n": int(ok.sum())}
    bias_vec, used_obs = None, obs
    bsec = inflation_section.get("obs_bias")
    if bsec:
        src = bsec.get("state_in")
        ob = (ObsBias.from_dict(json.loads(Path(src).read_text())) if src else
              ObsBias(by=bsec["by"], apply_kinds=[_KIND_CODES[k] for k in bsec["apply_kinds"]], var0=bsec["var0"],
                      q=bsec["q"]))
        info["bias_update"] = ob.update(obs, hx, ok)
        bias_vec = ob.bias_of(obs)
        if bsec.get("correct", True):
            used_obs, info["bias_correction"] = ob.correct(obs)
            info["obs"] = used_obs
        info["bias_state"] = ob.to_dict()
        if bsec.get("state_out"):
            Path(bsec["state_out"]).write_text(json.dumps(ob.to_dict(), indent=1) + "\n")
    a = inflation_section["adaptive"]
    ai = (AdaptiveInflation.from_dict(json.loads(Path(a["state_in"]).read_text())) if a.get("state_in") else
          AdaptiveInflation(rho0=a["rho0"], var0=a["var0"], q=a["q"], rho_min=a["rho_min"], rho_max=a["rho_max"],
                            n_min=a["n_min"]))
    upd = ai.update(obs, hx, ok, frames, geo, bias=bias_vec)
    frames, counters = inflate_state(frames, geo, upd["table"], spec_bdy_width=a["spec_bdy_width"], ramp=a["ramp"])
    info.update({"inflation_rho": 1.0, "adaptive": upd, "state": ai.to_dict(), "state_inflation_counters": counters,
                 "scale": float(np.sqrt(np.mean([r["rho"] for r in upd["table"].values()])))})
    if a.get("state_out"):
        Path(a["state_out"]).write_text(json.dumps(ai.to_dict(), indent=1) + "\n")
    return frames, info


def analysis_hooks(section) -> dict:
    """da-core hook: {'maintenance': callable(name, xb, xa) -> xa} from the ``maintenance``/``additive`` keys
    (``spread.make_maintenance``); per-point prior inflation is applied in state space by :func:`calibrate`."""
    from .spread import load_pool_npz, make_maintenance
    if not section or ("maintenance" not in section and "additive" not in section):
        return {}
    add = section.get("additive")
    pool = load_pool_npz(add["pool_npz"]) if add else None
    spec = dict(section.get("maintenance") or {})
    if spec.get("rho") == "calibrated":   # resolved at call time: api builds the hooks before calibrate() runs
        def calibrated_rho():
            if _CALIBRATED["rho"] is None:
                raise ValueError("maintenance.rho 'calibrated' needs a calibrate() with a transform inflation_rho")
            return float(_CALIBRATED["rho"])
        spec["rho"] = calibrated_rho
    return {"maintenance": make_maintenance(spec, pool=pool,
                                            amplitude=None if add is None else add["amplitude"],
                                            seed=None if add is None else add["seed"])}


def chain_section(section, previous_dir, cycle_dir) -> dict:
    """Per-cycle ``inflation`` section for a cycle loop: state_in ← ``previous_dir``/{adaptive_state,obs_bias}.json
    (None for the first cycle or when absent), state_out ← ``cycle_dir``/… (absolute paths). Returns a NEW validated
    section."""
    if not section:
        return section
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in section.items()}
    for key, name in (("adaptive", "adaptive_state.json"), ("obs_bias", "obs_bias.json")):
        if key in out:
            prev = None if previous_dir is None else Path(previous_dir).resolve() / name
            out[key]["state_in"] = str(prev) if prev is not None and prev.is_file() else None
            out[key]["state_out"] = str(Path(cycle_dir).resolve() / name)
    return validate_config_section(out)
