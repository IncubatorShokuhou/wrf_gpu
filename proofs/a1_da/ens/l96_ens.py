"""Lorenz-96 OSSE for the R3 ensemble-design fixes (lane w41-da-ens), using the PRODUCT classes.

40 variables, F = 8, RK4 dt 0.01, every variable observed every 0.05 (sigma_o = 1), N = 20, GC R-localisation
L = 3 (letkf.core). Imperfect model: the truth receives additive Gaussian model error each cycle, LARGER on the odd
variables (q_odd 0.6, q_even 0.15 per cycle) — the two halves need different inflation (R3: T 1-2 km x3 vs T2 none).
Observations of the odd half carry a +0.8 systematic bias (R3: station bias T2 -1.9 K, 10-m wind +1.3...3.2 m/s).
Groups: odd obs are encoded as surface T2 (state group 'T|0-1km'), even as surface U10 ('U|0-1km'), so
``inflation.AdaptiveInflation`` / ``ObsBias`` run unchanged.

Configs (same truth, obs and initial ensemble per seed):
  R3      : global rho from E[d^2] (bias included) each cycle, in the transform; RTPS 0.8 toward the RAW prior.
  R3nb    : as R3 with the obs bias removed by ObsBias (isolates the inflation design).
  NEW     : ObsBias correction + AdaptiveInflation per group (state space) + RTPS 0.3 (toward the inflated prior)
            + additive inflation a = 0.02 from a climatological pool (seeded). alpha/additive frozen from the declared
            development sweep (``sweep`` mode, results/L96_SWEEP.json, other seeds 11/12, 1000 cycles).
  NEWnoadd: NEW without additive inflation.
Metrics over the last 3/4 of the cycles: analysis RMSE vs truth (all / odd / even), spread, spread/RMSE; the first
10 cycles' spread (spread maintained over >= 3 cycles); rho trajectories. Run: JAX_PLATFORMS=cpu python l96_ens.py out.json
"""
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from gpuwrf.assimilation.letkf import core, inflation, spread  # noqa: E402

NV, F, DT, STEPS = 40, 8.0, 0.01, 5
ODD = np.arange(NV) % 2 == 1
Q = np.where(ODD, 0.6, 0.15)
OBS_BIAS = np.where(ODD, 0.8, 0.0)
# L96_OBSBIAS=v1 reproduces results/L96_ENS.json (absolute priors var0 4 / q 0.05 = 4·so² / 0.05·so² at so = 1, no
# coherent term); default v2 = the current ObsBias defaults (results/L96_ENS_v2.json)
OBSBIAS = os.environ.get("L96_OBSBIAS", "v2")
# evaluation seeds: 1,2,3 (L96_ENS*.json; also seen by the first, pre-sweep pass) or held-out ones (L96_SEEDS=21,22,23)
SEEDS = tuple(int(x) for x in os.environ.get("L96_SEEDS", "1,2,3").split(","))


def rhs(x):
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + F


def rk4(x, n):
    for _ in range(n):
        k1 = rhs(x)
        k2 = rhs(x + 0.5 * DT * k1)
        k3 = rhs(x + 0.5 * DT * k2)
        k4 = rhs(x + DT * k3)
        x = x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def obs_batch(values):
    var = np.where(ODD, 11, 13)
    return types.SimpleNamespace(kind=np.full(NV, 2), var=var, value=values, error=np.ones(NV), lat=np.zeros(NV),
                                 lon=np.zeros(NV), p_pa=np.full(NV, np.nan), qc=np.zeros(NV, int),
                                 station=np.array([f"x{j}" for j in range(NV)], object), site=np.array([""] * NV, object))


def run(config, *, seed=1, cycles=1500, n=20, loc=3.0, alpha=0.8, additive=0.15, rho_min=1.0):
    rng = np.random.default_rng(seed)
    truth = rk4(8.0 + rng.normal(size=NV), 2000)
    xs = rk4(truth + rng.normal(size=(n, NV)), 1000)
    clim = rk4(truth + rng.normal(size=(200, NV)), 3000)            # climatological pool for additive inflation
    pool = spread.pool_from_states({"X": clim})["X"]
    dist = np.abs(np.arange(NV)[:, None] - np.arange(NV)[None, :])
    dist = np.minimum(dist, NV - dist)
    rho_loc = core.gaspari_cohn(dist / core.gc_halfwidth(loc))
    ai = inflation.AdaptiveInflation(rho_max=20.0, rho_min=rho_min)
    ob = (inflation.ObsBias(apply_kinds=(2,), var0=4.0, q=0.05, coherent=False) if OBSBIAS == "v1" else
          inflation.ObsBias(apply_kinds=(2,)))
    rmse, rmse_odd, rmse_even, spr, rho_hist = [], [], [], [], []
    model_rng = np.random.default_rng(seed + 1000)
    for c in range(cycles):
        truth = rk4(truth, STEPS) + Q * model_rng.standard_normal(NV)
        xs = rk4(xs, STEPS)
        yo = truth + OBS_BIAS + rng.standard_normal(NV)
        hx = xs.copy()
        obs = obs_batch(yo)
        ok = np.ones(NV, bool)
        xb = xs
        rho_t = 1.0
        if config in ("R3", "R3nb"):
            if config == "R3nb":
                ob.update(obs, hx, ok)
                obs, _ = ob.correct(obs)
            d = obs.value - hx.mean(0)
            rho_t = float(np.clip((np.mean(d ** 2) - 1.0) / np.mean(hx.var(0, ddof=1)), 1.0, 20.0))
            rho_hist.append(rho_t)
        else:
            ob.update(obs, hx, ok)
            bias = ob.bias_of(obs)
            obs, _ = ob.correct(obs)
            ai.update(obs_batch(yo), hx, ok, None, None, bias=bias)
            r_odd, r_even = ai.rho["T|0-1km"], ai.rho["U|0-1km"]
            fac = np.sqrt(np.where(ODD, r_odd, r_even))
            xb = xs.mean(0) + fac * (xs - xs.mean(0))                 # state-space inflation per group
            hx = xb.copy()
            rho_hist.append([r_odd, r_even])
        ybar = hx.mean(0)
        ypert = np.broadcast_to((hx - ybar).T, (NV, NV, n))
        d = np.broadcast_to(obs.value - ybar, (NV, NV))
        _, _, t = core.letkf_transform(ypert, d, rho_loc, rho=rho_t)
        xa = core.apply_transform(xb, t)
        xa = spread.rtps(xb, xa, alpha)          # xb is the inflated prior for NEW, the raw prior for R3 (rho in T)
        if config == "NEW":
            xa, _ = spread.additive_inflation({"X": xa}, {"X": pool}, additive, seed=seed * 100000 + c)
            xa = xa["X"]
        xs = xa
        err = xs.mean(0) - truth
        rmse.append(float(np.sqrt(np.mean(err ** 2))))
        rmse_odd.append(float(np.sqrt(np.mean(err[ODD] ** 2))))
        rmse_even.append(float(np.sqrt(np.mean(err[~ODD] ** 2))))
        spr.append(float(np.sqrt(np.mean(xs.var(0, ddof=1)))))
        if not np.isfinite(rmse[-1]) or rmse[-1] > 50:
            break
    tail = slice(len(rmse) // 4, None)
    out = {"config": config, "seed": seed, "cycles_run": len(rmse), "n": n, "rtps_alpha": alpha,
           "additive": additive if config == "NEW" else 0.0, "rho_min": rho_min,
           "rmse_tail": float(np.mean(rmse[tail])), "rmse_odd_tail": float(np.mean(rmse_odd[tail])),
           "rmse_even_tail": float(np.mean(rmse_even[tail])), "spread_tail": float(np.mean(spr[tail])),
           "spread_first10": spr[:10], "rmse_first10": rmse[:10]}
    out["spread_over_rmse"] = out["spread_tail"] / out["rmse_tail"]
    if config in ("R3", "R3nb"):
        out["rho_tail_mean"] = float(np.mean(rho_hist[len(rho_hist) // 4:]))
    else:
        rh = np.array(rho_hist)
        out["rho_tail_mean"] = {"odd(T)": float(rh[len(rh) // 4:, 0].mean()), "even(U)": float(rh[len(rh) // 4:, 1].mean())}
        out["obs_bias_estimate"] = {k: v["bias"] for k, v in ob.groups.items()}
    return out


def sweep(path):
    """Development sweep of the adaptive design (declared tuning, 2 seeds x 1000 cycles; not the final numbers)."""
    rows = []
    for alpha in (0.0, 0.3, 0.5, 0.8):
        for add in (0.0, 0.02, 0.05):
            for rmin in (1.0, 0.5):
                rs = [run("NEW", seed=s, cycles=1000, alpha=alpha, additive=add, rho_min=rmin) for s in (11, 12)]
                rows.append({"alpha": alpha, "additive": add, "rho_min": rmin,
                             "rmse": float(np.mean([r["rmse_tail"] for r in rs])),
                             "spread_over_rmse": float(np.mean([r["spread_over_rmse"] for r in rs]))})
                print(rows[-1], flush=True)
    base = [run(c, seed=s, cycles=1000) for s in (11, 12) for c in ("R3", "R3nb")]
    Path(path).write_text(json.dumps({"sweep": rows, "baselines": base}, indent=1) + "\n")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "sweep":
    sweep(sys.argv[2])
    sys.exit(0)

if __name__ == "__main__":
    t0 = time.time()
    def one(cfg, s):
        if cfg.startswith("NEW"):
            return run(cfg, seed=s, alpha=0.3, additive=0.02)
        return run(cfg, seed=s)
    runs = [one(cfg, s) for s in SEEDS for cfg in ("R3", "R3nb", "NEWnoadd", "NEW")]
    agg = {}
    for cfg in ("R3", "R3nb", "NEWnoadd", "NEW"):
        rows = [r for r in runs if r["config"] == cfg]
        agg[cfg] = {k: float(np.mean([r[k] for r in rows])) for k in ("rmse_tail", "rmse_odd_tail", "rmse_even_tail",
                                                                       "spread_tail", "spread_over_rmse")}
    gate = {"NEW_spread_over_rmse_in_[0.7,1.3]_all_seeds": all(0.7 <= r["spread_over_rmse"] <= 1.3
                                                                for r in runs if r["config"] == "NEW"),
            "NEW_rmse_below_R3_all_seeds": all(
                next(r for r in runs if r["config"] == "NEW" and r["seed"] == s)["rmse_tail"]
                < next(r for r in runs if r["config"] == "R3" and r["seed"] == s)["rmse_tail"] for s in SEEDS),
            # >= 3 cycles: after the initial climatological transient the spread never collapses below half the
            # actual error and never exceeds twice it (cycles 3..10 of every seed)
            "NEW_spread_maintained_cycles_3_10": all(
                0.5 <= sp / er <= 2.0 for r in runs if r["config"] == "NEW"
                for sp, er in zip(r["spread_first10"][3:], r["rmse_first10"][3:]))}
    out = {"experiment": "w41-da-ens L96 OSSE (imperfect model, obs bias, two groups)", "wall_s": time.time() - t0,
           "obs_bias_estimator": OBSBIAS, "seeds": list(SEEDS),
           "aggregate_mean_over_seeds": agg, "gate": gate, "runs": runs}
    text = json.dumps(out, indent=1)
    print(json.dumps({"aggregate": agg, "gate": gate, "wall_s": out["wall_s"]}, indent=1))
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text + "\n")
