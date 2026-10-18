"""AE01 port runner: RRTMG SW (the selected SW path of this process) on the pristine aer_opt=1 oracle columns.

usage: python sw_oracle_runner.py OUT_JSON [--aer3] VARIANT[:ARMS] ...   e.g.  port:all  no_aer:0,10

Default: the AE01 aer_opt=1 oracle (tegen_sw_0227_v1).  ``--aer3``: the AE07 aer_opt=3 oracle
(thompson_aer_sw_0227_v1; the port builds the band optics from the fixture's p_hyd/t/qv/dz8w/QNWFA/QNIFA with
physics/wrf_aer_opt3.py, exactly as the coupler does) -- the JSON then also holds the optics unit errors.

The release path needs GPUWRF_FAST_DEFAULTS=1 in the environment (REAL entry + fused band-sums kernel, CPU
interpret); without it the legacy XLA branch runs.  Per variant and arm the JSON holds the worst frozen-bound ratio
per output vs the oracle arm, and the aerosol EFFECT (aerosol arm minus its aer_opt=0 partner arm, both port) vs
WRF's effect.  Variants: ``port`` (as shipped) and genuine mutants of the aer_opt=1 code (E202):

  no_aer               the aerosol leaf is dropped (aer_opt=1 silently off)
  ssa_one              band optics single-scattering albedo -> 1 (ecaer_band_optics)
  asym_zero            band optics asymmetry -> 0
  kernel_scaled_direct fused kernel emits the delta-scaled direct beam instead of WRF's ztdbt_nodel
  kernel_no_delta      fused kernel skips the aerosol delta scaling (zf = 0)
  xla_scaled_direct    XLA branch emits the delta-scaled direct beam
  aer3_no_floor        (--aer3) gt_aod without WRF's MAX(1e-3) per-layer floor
  aer3_surface_rh      (--aer3) spectral Lagrange tables evaluated at the lowest-layer RH for every layer
"""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from gpuwrf.kernels import rad_sw_band_sums_aer  # noqa: E402
from gpuwrf.physics import rrtmg_sw as R  # noqa: E402
from gpuwrf.physics import wrf_tegen_aerosol as T  # noqa: E402
from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time  # noqa: E402

from gpuwrf.physics import wrf_aer_opt3 as A3  # noqa: E402

from tegen_fixture import AER3, AER3_ENS, FLUX_BOUND, HEATING_BOUND, RE01, SW, bound_ratio, load  # noqa: E402

RRSW_SCON = 1368.22
KERNEL_MUTANTS = {"kernel_scaled_direct": ("dn * bn", "dn * b"), "kernel_no_delta": ("f = gc * gc", "f = gc * zero")}
CALLS = {"one": 0, "all": 0}


def _spy():
    one, all_ = rad_sw_band_sums_aer.sw_band_flux_sums_aer, rad_sw_band_sums_aer.sw_allband_flux_sums_aer

    def spy_one(*a, **k):
        CALLS["one"] += 1
        return one(*a, **k)

    def spy_all(*a, **k):
        CALLS["all"] += 1
        return all_(*a, **k)
    rad_sw_band_sums_aer.sw_band_flux_sums_aer, rad_sw_band_sums_aer.sw_allband_flux_sums_aer = spy_one, spy_all


def _apply(variant):
    """Installs one mutant (returns an undo callable); 'port' and 'no_aer' change no code."""

    if variant in KERNEL_MUTANTS:
        old, new = KERNEL_MUTANTS[variant]
        source = Path(rad_sw_band_sums_aer.__file__).read_text()
        assert source.count(old) == 1, variant
        mod = types.ModuleType("rad_sw_band_sums_aer_mutant")
        exec(compile(source.replace(old, new), "rad_sw_band_sums_aer_mutant.py", "exec"), mod.__dict__)
        saved = (rad_sw_band_sums_aer.sw_band_flux_sums_aer, rad_sw_band_sums_aer.sw_allband_flux_sums_aer)
        rad_sw_band_sums_aer.sw_band_flux_sums_aer = mod.sw_band_flux_sums_aer
        rad_sw_band_sums_aer.sw_allband_flux_sums_aer = mod.sw_allband_flux_sums_aer

        def undo():
            rad_sw_band_sums_aer.sw_band_flux_sums_aer, rad_sw_band_sums_aer.sw_allband_flux_sums_aer = saved
        return undo
    if variant in ("ssa_one", "asym_zero"):
        original = T.ecaer_band_optics

        def mutant(ecaer):
            tau, ssa, asy = original(ecaer)
            return (tau, jnp.ones_like(ssa), asy) if variant == "ssa_one" else (tau, ssa, jnp.zeros_like(asy))
        T.ecaer_band_optics = mutant
        return lambda: setattr(T, "ecaer_band_optics", original)
    if variant == "aer3_no_floor":
        source = Path(A3.__file__).read_text()
        old = "jnp.maximum(F(1.0e-3), aod_wfa + aod_ifa)"
        assert source.count(old) == 1
        mod = types.ModuleType("wrf_aer_opt3_mutant")
        exec(compile(source.replace(old, "aod_wfa + aod_ifa"), "wrf_aer_opt3_mutant.py", "exec"), mod.__dict__)
        original = A3.wrf_gt_aod
        A3.wrf_gt_aod = mod.wrf_gt_aod
        return lambda: setattr(A3, "wrf_gt_aod", original)
    if variant == "aer3_surface_rh":
        original = A3.wrf_relative_humidity

        def mutant(*a):
            rh = original(*a)
            return jnp.broadcast_to(rh[..., :1], rh.shape)
        A3.wrf_relative_humidity = mutant
        return lambda: setattr(A3, "wrf_relative_humidity", original)
    if variant == "xla_scaled_direct":
        original = R._sw_clear_aerosol_optics

        def mutant(*a):
            _tco, tau, omega, asym = original(*a)
            return tau, tau, omega, asym
        R._sw_clear_aerosol_optics = mutant
        return lambda: setattr(R, "_sw_clear_aerosol_optics", original)
    assert variant in ("port", "no_aer"), variant
    return lambda: None


def aer3_optics(fx, ae, arm):
    """The port's aer_opt=3 band optics on the fixture's operands (coupler inputs: p_hyd, t_phy, QV, QNWFA, QNIFA, dz8w)."""

    nz = fx["in_t"].shape[1]
    scale = np.float32(ae["arm_scale"][arm])
    tau, ssa, asy, taod = A3.wrf_aer_opt3_band_optics(
        jnp.asarray(fx["in_p_hyd"], jnp.float32), jnp.asarray(fx["in_t"], jnp.float32),
        jnp.asarray(fx["in_qv"], jnp.float32), jnp.asarray(ae["in_qnwfa"] * scale), jnp.asarray(ae["in_qnifa"] * scale),
        jnp.asarray(fx["in_dz8w"][:, :nz], jnp.float32))
    return tau, ssa, asy, taod


def _solve(fx, ae, arm, drop_aerosol):
    nz = fx["in_t"].shape[1]
    a = lambda k: jnp.asarray(np.asarray(fx[k], np.float64))  # noqa: E731
    ghg = clwrf_ssp245_gases_for_time(datetime(2026, 1, 1) + timedelta(days=float(ae["julian"][0, arm])))
    extra = {}
    if ae["arm_aer_opt"][arm] == 1 and not drop_aerosol:
        extra["aerosol_ecaer"] = jnp.asarray(np.asarray(ae["aerod"][:, arm], np.float32))
    elif ae["arm_aer_opt"][arm] == 3 and not drop_aerosol:
        tau, ssa, asy, _ = aer3_optics(fx, ae, arm)
        extra["aerosol_bands"] = jnp.stack((tau, ssa, asy), axis=-1)
    st = R.RRTMGSWColumnState(
        T=a("in_t"), p=a("in_p_hyd"), qv=a("in_qv"), qc=a("in_qc_rad"), qi=a("in_qi_rad"), qs=a("in_qs"), qg=a("in_qg"),
        cloud_fraction=a("in_cldfra"), dz=a("in_dz8w")[:, :nz], rho=a("in_rho"), pressure_interfaces=a("in_p_hyd_w"),
        temperature_interfaces=a("in_t8w"), ozone_vmr=jnp.asarray(np.asarray(ae["o3"][:, arm], np.float64)),
        co2_vmr=ghg.co2_vmr, n2o_vmr=ghg.n2o_vmr, ch4_vmr=ghg.ch4_vmr, surface_albedo=a("in_albedo"),
        coszen=jnp.asarray(np.asarray(ae["coszen"][:, arm], np.float64)),
        solar_source_scale=jnp.asarray(np.asarray(ae["solcon"][:, arm], np.float64) / RRSW_SCON),
        re_cloud=a("col_re_cloud"), re_ice=a("col_re_ice"), re_snow=a("col_re_snow"), xland=a("in_xland"), **extra)
    sw = R.solve_rrtmg_sw_column(st, with_clear_sky=True)
    return {"swdnb": np.asarray(sw.surface_down), "swupt": np.asarray(sw.toa_up), "swddir": np.asarray(sw.surface_direct),
            "swdnbc": np.asarray(sw.clear_flux_down)[:, 0], "swuptc": np.asarray(sw.clear_flux_up)[:, -1],
            "swdn": np.asarray(sw.flux_down), "swup": np.asarray(sw.flux_up),
            "hsw": np.asarray(sw.heating_rate) / fx["in_pi"].astype(np.float64)}


def _envelope(ens, key, arm):
    """AE09 cells of ``arm`` where WRF's own 1-ulp ensemble leaves the frozen bound: (index, lo, hi) or None."""

    if ens is None or f"{key}_cells" not in ens:
        return None
    cells = ens[f"{key}_cells"]
    sel = cells[:, 1] == arm
    if not sel.any():
        return None
    return tuple(np.delete(cells[sel], 1, axis=1).T), ens[f"{key}_lo"][sel], ens[f"{key}_hi"][sel]


def _metrics(port, ae, arm, cols, ens=None):
    """Worst frozen-bound ratio per output vs WRF member 0; in the (few) AE09 cells where WRF's own REAL result is not
    reproducible at the bound under 1-ulp input perturbations, the ratio is the distance outside WRF's ensemble
    envelope instead (same bound at the nearer edge).  Also returns the plain member-0 ratios and the cell count."""

    out, raw, n_env = {}, {}, 0
    for key, value in port.items():
        ref = ae[key][:, arm]
        if key in ("swdn", "swup"):
            ref = ref[:, : value.shape[1]]  # surface .. model-top interfaces (RE01 LEVEL_SLICE)
        bound = HEATING_BOUND if key == "hsw" else FLUX_BOUND
        ratio = bound_ratio(ref, value, bound)
        raw[key] = float(ratio[cols].max())
        env = _envelope(ens, key, arm)
        if env is not None:
            idx, lo, hi = env
            v = np.asarray(value, np.float64)[idx]
            edge = np.where(v < lo, np.abs(lo), np.abs(hi))
            r_env = (np.maximum(lo - v, 0.0) + np.maximum(v - hi, 0.0)) / (bound[0] + bound[1] * edge)
            ratio = np.array(ratio, copy=True)
            ratio[idx] = np.where(np.isfinite(v), r_env, np.inf)  # E200
            n_env += int(np.asarray(cols)[idx[0]].sum()) if np.asarray(cols).dtype == bool else len(lo)
        out[key] = float(ratio[cols].max())
    return out, raw, n_env


def _partners(ae):
    """aerosol arm -> its aer_opt=0 arm with the same julian (AE01: x10 arm -> arm 0)."""

    out = {}
    for arm in range(len(ae["arm_aer_opt"])):
        if ae["arm_aer_opt"][arm] == 0:
            continue
        same = [o for o in range(len(ae["arm_aer_opt"]))
                if ae["arm_aer_opt"][o] == 0 and ae["arm_julian"][o] == ae["arm_julian"][arm]]
        if same:
            out[arm] = same[0]
    return out


def _optics_errors(fx, ae, arm):
    tau, ssa, asy, taod = (np.asarray(x, np.float64) for x in aer3_optics(fx, ae, arm))
    ref = {k: ae[k][:, arm].astype(np.float64) for k in ("taod5503d", "tauaer", "ssaaer", "asyaer")}
    rel = lambda got, want: float(np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-30)))  # noqa: E731
    return {"taod5503d_max_rel": rel(taod, ref["taod5503d"]), "tauaer_max_rel": rel(tau, ref["tauaer"]),
            "ssaaer_max_abs": float(np.abs(ssa - ref["ssaaer"]).max()),
            "asyaer_max_abs": float(np.abs(asy - ref["asyaer"]).max()),
            "taod5502d_max_rel": rel(taod.sum(-1), ae["taod5502d"][:, arm].astype(np.float64)),
            "finite": bool(all(np.isfinite(x).all() for x in (tau, ssa, asy, taod)))}


def main(path, specs):
    aer3 = "--aer3" in specs
    specs = [x for x in specs if x != "--aer3"]
    fx, ae = load(RE01), load(AER3 if aer3 else SW)
    ens = load(AER3_ENS) if aer3 else None
    if "o3" not in ae:
        # AE07 does not re-dump o3rad: it is the same pristine oznini/ozn_time_int/ozn_p_int output as AE01's arm of
        # the same julian on the same columns (asserted via the bitwise-identical radiation-call coszen/solcon).
        ae01 = load(SW)
        o3 = []
        for arm, jul in enumerate(ae["arm_julian"]):
            src = int(np.flatnonzero(ae01["arm_julian"] == jul)[0])
            assert np.array_equal(ae01["coszen"][:, src], ae["coszen"][:, arm])
            assert np.array_equal(ae01["solcon"][:, src], ae["solcon"][:, arm])
            o3.append(ae01["o3"][:, src])
        ae["o3"] = np.stack(o3, axis=1)
    narm = len(ae["arm_aer_opt"])
    partner = _partners(ae)
    _spy()
    import gpuwrf
    result = {"real_entry": bool(R._real_entry()), "band_sums": int(R._sw_band_sums_mode() or 0),
              "platform": jax.devices()[0].platform, "gpuwrf": gpuwrf.__file__, "oracle": "AE07" if aer3 else "AE01",
              "variants": {}}
    for spec in specs:
        variant, _, arms = spec.partition(":")
        arms = list(range(narm)) if arms in ("", "all") else [int(x) for x in arms.split(",")]
        undo = _apply(variant)
        jax.clear_caches()
        CALLS.update(one=0, all=0)
        try:
            port = {arm: _solve(fx, ae, arm, variant == "no_aer") for arm in arms}
        finally:
            undo()
            jax.clear_caches()
        rows = {}
        clear = fx["category"] == "clear"
        for arm in arms:
            bound, raw, n_env = _metrics(port[arm], ae, arm, np.ones(clear.shape, bool), ens)
            row = {"bound": bound, "bound_raw": raw, "envelope_cells": n_env,
                   "bound_clear": _metrics(port[arm], ae, arm, clear, ens)[0]}
            off = partner.get(arm)
            if off is not None and off in port:
                day = ae["coszen"][:, arm] > 0.05
                for tag, cols in (("effect", day), ("effect_clear", day & clear)):
                    row[tag] = {}
                    for k in ("swdnb", "swddir", "swupt", "swdnbc", "swuptc"):
                        wrf = (ae[k][:, arm] - ae[k][:, off]).astype(np.float64)
                        err = np.abs((port[arm][k] - port[off][k]) - wrf)
                        err = np.where(np.isfinite(err), err, np.inf)  # E200: non-finite is never inside
                        row[tag][k] = {"wrf_median": float(np.median(wrf[cols])), "max_err": float(err[cols].max()),
                                       # the tier-1 flux bound applied to the effect itself (abs 1 W/m2 + 5 %)
                                       "max_ratio": float((err / (FLUX_BOUND[0] + FLUX_BOUND[1] * np.abs(wrf)))[cols].max())}
            if aer3 and ae["arm_aer_opt"][arm] == 3:
                undo = _apply(variant)
                try:
                    row["optics"] = _optics_errors(fx, ae, arm)
                finally:
                    undo()
            rows[str(arm)] = row
        result["variants"][variant] = {"arms": rows, "aer_kernel_calls": dict(CALLS)}
        Path(path).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({v: {a: r["bound"] for a, r in d["arms"].items()} for v, d in result["variants"].items()}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
