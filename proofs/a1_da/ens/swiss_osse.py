"""Real-model identical-twin OSSE for the R3 ensemble-design fixes (lane w41-da-ens) — CPU-WRF members, Swiss d01.

Case: the shipped NON-ALISIOS example ``examples/switzerland_d01`` (42x42x44, dx 3 km, dt 18 s, 2023-01-15 00Z,
wrfbdy every 3 h); original CPU-WRF (canairy install_gen2_dmpar ``wrf``, 1 MPI rank per member, statics/tables
linked). Pool trajectory = the original CPU-WRF 24-h reference run of that case (release-docs RD11/cpu, hourly).

Perturbations (both arms, same pool and weights): member i = control + s * sum_m z_im (x_ref(t + lag_m) - mean_m),
lags 0..9 h, z ~ N(0,1)/sqrt(M-1) seeded; truth = the same construction with independent weights, its OWN perturbed
LBC, MU included and rebalanced (perturb.rebalance_with_mu). Obs from the truth run: 2 synthetic sondes (T, QV, U, V
on every 2nd model level below ~250 hPa) + 30 surface stations (T2 with a -1.0 K and U10 with a +0.8 m/s
station bias, Q2, V10), Gaussian errors, seeded per cycle. 3 analyses (03, 06, 09Z) + forecast to 12Z (OSSE_ANALYSES).

Arms:
  A0 'R3 baseline'  : IC U/V/T/QV only, MU frozen, tapered; shared wrfbdy; global innovation rho (E[d^2], bias in)
                      in the transform; RTPS 0.8 to the RAW prior; no additive; no variable localisation.
  A1 'new design'   : IC incl. MU (rebalanced), U/V/T/QV untapered + matched perturbed wrfbdy per member (lbc_ens);
                      adaptive debiased var x band rho (state space, persisted) + online surface obs bias
                      correction; RTPS (alpha 0.3) toward the inflated prior; additive a=0.1 (lagged pool npz at t_a);
                      variable localisation 'surface_split'.
  A1b (POST-HOC)    : A1 with estimator v2 (ObsBias coherent-variance gain, profile n_eff); reuses A1's cycle-0
                      forecast (identical inputs); added after the cycle-0/1 diagnosis, not pre-registered.
  A2 (optional)     : A1 with the shared wrfbdy (isolates the LBC effect).
Metrics (interior >= 8 rows from the edge): spread and ensemble-mean RMSE vs truth per variable x band, prior /
posterior / written / every forecast hour; rho tables; obs-space OmB/OmA; negative QV counts.

Run (CPU only, lane cores; JAX never imported):
  python proofs/a1_da/ens/swiss_osse.py truth [tag seed]      (F1: truth_f1 778)
  python proofs/a1_da/ens/swiss_osse.py arm A0|A1|A1b|F1|A2
  python proofs/a1_da/ens/swiss_osse.py balance        (B0/B1/B2 first-hour spread retention, N = 10)
"""
from __future__ import annotations

import concurrent.futures as cf
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from gpuwrf.assimilation.letkf import analysis as lan  # noqa: E402
from gpuwrf.assimilation.letkf import inflation, lbc_ens, perturb, spread, state_io, varloc  # noqa: E402
from gpuwrf.assimilation.letkf.cycle_case import _edit_namelist, _forcing_plan, _slice_netcdf  # noqa: E402
from gpuwrf.assimilation.letkf.obs import ObsBatch  # noqa: E402
from gpuwrf.assimilation.letkf.operators import model_equivalents  # noqa: E402

CASE = REPO / "examples" / "switzerland_d01"
REF = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
ROOT = Path(os.environ.get("OSSE_ROOT", "<DATA_ROOT>/wrf_gpu2/scratch/w41-da-ens/osse"))
STATIC = Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-da-ens/smoke1")
MPIRUN = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/mpirun"
CORES = [int(c) for c in os.environ.get("OSSE_CORES", "8,9,12,13,24,25,28,29").split(",")]
FMT = "%Y-%m-%d_%H:%M:%S"
T0 = datetime(2023, 1, 15, 0)
N = int(os.environ.get("OSSE_N", "20"))
CYCLE_H = 3
N_ANALYSES = int(os.environ.get("OSSE_ANALYSES", "3"))
LAGS_H = tuple(range(10))
SCALE = 0.5
INTERIOR = 8
STATE = ("U", "V", "T", "QVAPOR", "MU")
LBC_NAMES = ("U", "V", "T", "QVAPOR")
SURF_DIAG = ("T2", "Q2", "U10", "V10")
SIGMA = {1: 1.0, 2: 5.0e-4, 3: 2.0, 4: 2.0, 11: 1.0, 12: 5.0e-4, 13: 1.5, 14: 1.5}
STATION_BIAS = {11: -1.0, 13: 0.8}
SEED_W, SEED_TRUTH, SEED_NET = 20261017, 777, 4242
END = T0 + timedelta(hours=CYCLE_H * (N_ANALYSES + 1))


def wt(t: datetime) -> str:
    return t.strftime(FMT)


_REF_CACHE: dict = {}


def ref_state(t: datetime, names=STATE + SURF_DIAG) -> dict:
    key = (wt(t), tuple(names))
    if key not in _REF_CACHE:
        _REF_CACHE[key] = state_io.read_frame(REF / f"wrfout_d01_{wt(t)}", wt(t), names)
    return _REF_CACHE[key]


def lagged_pool(t: datetime, names) -> dict:
    states = [ref_state(t + timedelta(hours=h), names) for h in LAGS_H]
    return {n: np.stack([np.asarray(s[n], np.float64) for s in states]) for n in names}


def weights(seed, n):
    return spread.gaussian_combinations(n, len(LAGS_H), seed)


def tapers(ny, nx):
    return spread.taper_fields(("U", "V", "T", "QVAPOR", "MU"), ny, nx, spec_bdy_width=5, ramp=5)


def control_at(t):  # ensemble's prior-mean guess trajectory = the reference run
    return {k: np.asarray(v, np.float64) for k, v in ref_state(t).items()}


# --------------------------------------------------------------------------- member construction

def build_members(z, *, with_mu, untapered, geo, ny, nx, centre=True):
    """IC fields at T0 for every row of z (dict name -> (n, ...)) incl. rebalanced P/PH (and MU)."""
    bg = state_io.read_frame(CASE / "wrfinput_d01", wt(T0), ("U", "V", "T", "QVAPOR", "MU", "P", "PH", "P_HYD"))
    names = STATE if with_mu else LBC_NAMES
    pool = lagged_pool(T0, names)
    dev, _ = perturb.combination_perturbations(pool, z.shape[0], seed=None, names=names, weights=z, centre=centre)
    tap = tapers(ny, nx)
    taper = {"MU": tap["MU"]} if untapered else {n: tap[n] for n in names}
    ctrl = {n: np.asarray(bg[n], np.float64) for n in names}
    mem = perturb.recentre(ctrl, dev, SCALE, taper=taper, centre=centre)
    out = []
    for i in range(z.shape[0]):
        an = {**bg, **{n: mem[n][i] for n in names}}
        if with_mu:
            reb = perturb.rebalance_with_mu(bg, an, geo)
        else:
            reb = state_io.rebalance_increment(bg, an, geo)
        fields = {n: np.asarray(an[n]).astype(np.asarray(bg[n]).dtype) for n in names}
        fields.update({k: reb[k] for k in ("P", "PH", "P_HYD") if k in reb})
        if with_mu:
            fields["MU"] = reb["MU"]
        out.append(fields)
    return out, dev


def lbc_records(z, times, perturb_until, centre=True):
    """{label: {time: deviation dict}} from lagged pool combinations at each record (untapered)."""
    per_time = {}
    for t in times:
        tt = datetime.strptime(t, FMT)
        if tt > perturb_until:
            continue
        pool = lagged_pool(tt, LBC_NAMES)
        dev, _ = perturb.combination_perturbations(pool, z.shape[0], seed=None, names=LBC_NAMES, weights=z,
                                                   centre=centre)
        ctrl = {n: control_at(tt)[n] for n in LBC_NAMES}
        mem = perturb.recentre(ctrl, dev, SCALE, centre=centre)
        per_time[t] = {n: mem[n] - ctrl[n][None] for n in LBC_NAMES}
    out = {}
    for i in range(z.shape[0]):
        out[f"m{i:02d}"] = {t: ({n: per_time[t][n][i] for n in LBC_NAMES} if t in per_time else {}) for t in times}
    return out


# --------------------------------------------------------------------------- WRF runs

def make_rundir(d: Path, wrfinput: Path, wrfbdy: Path, start: datetime, hours: int):
    d.mkdir(parents=True)
    end = start + timedelta(hours=hours)
    updates = {("time_control", "run_days"): 0, ("time_control", "run_hours"): hours}
    for prefix, stamp in (("start", start), ("end", end)):
        for part in ("year", "month", "day", "hour", "minute", "second"):
            updates[("time_control", f"{prefix}_{part}")] = getattr(stamp, part)
    (d / "namelist.input").write_text(_edit_namelist((CASE / "namelist.input").read_text(), updates))
    shutil.copyfile(wrfinput, d / "wrfinput_d01")
    plan = _forcing_plan(Path(wrfbdy), start, end, 10800.0, boundary=True)
    _slice_netcdf(Path(wrfbdy), d / "wrfbdy_d01", plan["records"], start)
    for f in STATIC.iterdir():
        if f.name.startswith(("wrfout", "rsl.", "namelist", "wrfinput", "wrfbdy", "run.log")):
            continue
        (d / f.name).symlink_to(f.resolve() if f.is_symlink() else f)


def run_wrf(d: Path, core: int, timeout=1800):
    t0 = time.time()
    with open(d / "run.log", "w") as log:
        proc = subprocess.Popen(["taskset", "-c", str(core), "nice", "-n", "19", MPIRUN, "--oversubscribe",
                                 "--bind-to", "none", "-np", "1", "./wrf.exe"], cwd=d, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = -9
    ok = rc == 0 and "SUCCESS COMPLETE WRF" in (d / "rsl.out.0000").read_text(errors="ignore")
    if not ok:
        raise RuntimeError(f"WRF failed in {d} rc={rc}")
    for f in d.glob("rsl.*"):
        if f.name != "rsl.out.0000":
            f.unlink()
    return time.time() - t0


def run_many(dirs):
    walls = {}
    with cf.ThreadPoolExecutor(len(CORES)) as ex:
        futs = {}
        free = list(CORES)
        pending = list(dirs)
        while pending or futs:
            while pending and free:
                d = pending.pop(0)
                core = free.pop(0)
                futs[ex.submit(run_wrf, d, core)] = (d, core)
            done, _ = cf.wait(futs, return_when=cf.FIRST_COMPLETED)
            for f in done:
                d, core = futs.pop(f)
                walls[d.name] = f.result()
                free.append(core)
    return walls


# --------------------------------------------------------------------------- obs network

def geometry():
    return state_io.read_geometry(CASE / "wrfinput_d01", base_wrfinput=CASE / "wrfinput_d01")


def network(geo):
    rng = np.random.default_rng(SEED_NET)
    ny, nx = geo["XLAT"].shape
    land = np.asarray(geo.get("LANDMASK", np.ones((ny, nx))))
    cand = [(j, i) for j in range(7, ny - 7) for i in range(7, nx - 7) if land[j, i] == 1]
    pick = rng.choice(len(cand), 30, replace=False)
    stations = [cand[k] for k in pick]
    sondes = [(ny // 2, nx // 2), (ny // 3, 2 * nx // 3)]
    return {"stations": stations, "sondes": sondes}


def make_obs(truth_frame, geo, net, cycle, seed_offset=0):
    rng = np.random.default_rng(SEED_NET + 100 * cycle + seed_offset)
    rows = {k: [] for k in ("var", "lat", "lon", "p_pa", "height_m", "station")}
    p = np.asarray(truth_frame["P"], np.float64) + np.asarray(truth_frame["PB"], np.float64)
    for s, (j, i) in enumerate(net["sondes"]):
        for k in range(1, p.shape[0], 2):
            if p[k, j, i] < 25000.0:
                break
            for v in (1, 2, 3, 4):
                rows["var"].append(v); rows["lat"].append(geo["XLAT"][j, i]); rows["lon"].append(geo["XLONG"][j, i])
                rows["p_pa"].append(p[k, j, i]); rows["height_m"].append(np.nan); rows["station"].append(f"sonde{s}")
    for s, (j, i) in enumerate(net["stations"]):
        for v in (11, 12, 13, 14):
            rows["var"].append(v); rows["lat"].append(geo["XLAT"][j, i]); rows["lon"].append(geo["XLONG"][j, i])
            rows["p_pa"].append(np.nan); rows["height_m"].append(float(geo["HGT"][j, i]))
            rows["station"].append(f"st{s:02d}")
    var = np.array(rows["var"])
    template = ObsBatch.from_columns(var=var, value=np.zeros(len(var)), lat=np.array(rows["lat"]),
                                     lon=np.array(rows["lon"]), p_pa=np.array(rows["p_pa"]),
                                     height_m=np.array(rows["height_m"]), station=np.array(rows["station"], object),
                                     error=np.array([SIGMA[int(v)] for v in var]))
    one = {k: np.asarray(v)[None] for k, v in truth_frame.items()}
    hx = model_equivalents(template, one, geo)[0]
    noise = rng.standard_normal(len(var)) * template.error
    bias = np.array([STATION_BIAS.get(int(v), 0.0) for v in var])
    value = hx + bias + noise
    ok = np.isfinite(value)
    obs = ObsBatch.from_columns(var=var[ok], value=value[ok], lat=template.lat[ok], lon=template.lon[ok],
                                p_pa=template.p_pa[ok], height_m=template.height_m[ok],
                                station=template.station[ok], error=template.error[ok])
    return obs


# --------------------------------------------------------------------------- metrics

def stack(frames_list):
    return {n: np.stack([f[n] for f in frames_list]) for n in frames_list[0]}


def interior_mask(ny, nx):
    j, i = np.mgrid[0:ny, 0:nx]
    return np.minimum(np.minimum(j, ny - 1 - j), np.minimum(i, nx - 1 - i)) >= INTERIOR


def rmse_by_band(frames, truth, geo):
    band = spread.band_index(frames, geo)
    inner = interior_mask(*band.shape[1:])
    out = {}
    for name in ("T", "QVAPOR", "U", "V"):
        err = spread._mass(name, frames[name]).mean(0) - spread._mass(name, np.asarray(truth[name])[None])[0]
        if name == "QVAPOR":
            err = err * 1000.0
        out[name] = {}
        for b, (lo, hi) in enumerate(spread.BANDS):
            sel = (band == b) & inner[None]
            key = f"{lo/1000:.0f}-{'inf' if not np.isfinite(hi) else f'{hi/1000:.0f}'}km"
            out[name][key] = float(np.sqrt(np.mean(err[sel] ** 2))) if sel.any() else None
    for name in SURF_DIAG:
        if name in frames and name in truth:
            e = np.asarray(frames[name], np.float64).mean(0) - np.asarray(truth[name], np.float64)
            out[name] = float(np.sqrt(np.mean((e * (1000.0 if name == "Q2" else 1.0))[inner] ** 2)))
    return out


def summary(frames, truth, geo):
    s = spread.spread_by_band(frames, geo, interior=INTERIOR)
    r = rmse_by_band(frames, truth, geo)
    ratio = {}
    for name in ("T", "QVAPOR", "U", "V"):
        ratio[name] = {k: (s[name][k] / r[name][k] if s[name].get(k) and r[name].get(k) else None) for k in s[name]}
    return {"spread": s, "rmse": r, "spread_over_rmse": ratio}


# --------------------------------------------------------------------------- truth

def truth(tag="truth", seed=SEED_TRUTH):
    out = ROOT / tag
    if (out / "DONE").exists():
        return
    geo = geometry()
    ny, nx = geo["XLAT"].shape
    z = weights(seed, 21)[20:21]                # independent of the member weights (other seed)
    fields, dev = build_members(z, with_mu=True, untapered=True, geo=geo, ny=ny, nx=nx, centre=False)
    work = out / "inputs"
    work.mkdir(parents=True, exist_ok=True)
    ic = work / "wrfinput_d01"
    state_io.write_analysis_wrfinput(CASE / "wrfinput_d01", CASE / "wrfinput_d01", wt(T0), fields[0], ic,
                                     land_cycle=False, extra_provenance={"kind": "OSSE truth IC"})
    times = lbc_ens.record_times(CASE / "wrfbdy_d01")
    recs = lbc_records(z, times, END, centre=False)
    man = lbc_ens.write_member_wrfbdys(CASE / "wrfbdy_d01", ic, {"truth": recs["m00"]}, work / "lbc")
    make_rundir(out / "run", ic, Path(man["members"]["truth"]["output"]), T0, CYCLE_H * (N_ANALYSES + 1))
    wall = run_wrf(out / "run", CORES[0], timeout=9000)
    (out / "DONE").write_text(json.dumps({"wall_s": wall, "lbc": man["members"]["truth"]}, default=str))


def truth_frame(t, tag="truth"):
    names = state_io.FRAME_VARIABLES
    return state_io.read_frame(ROOT / tag / "run" / f"wrfout_d01_{wt(t)}", wt(t), names)


# --------------------------------------------------------------------------- arms

ARMS = {
    "A0": {"with_mu": False, "untapered": False, "perturbed_lbc": False, "adaptive": False, "additive": 0.0,
           "var_loc": None, "rtps_alpha": 0.8},
    "A1": {"with_mu": True, "untapered": True, "perturbed_lbc": True, "adaptive": True,
           "additive": float(os.environ.get("OSSE_ADDITIVE", "0.1")), "var_loc": "surface_split",
           "rtps_alpha": float(os.environ.get("OSSE_ALPHA", "0.3"))},
    "A2": {"with_mu": True, "untapered": False, "perturbed_lbc": False, "adaptive": True,
           "additive": float(os.environ.get("OSSE_ADDITIVE", "0.1")), "var_loc": "surface_split",
           "rtps_alpha": float(os.environ.get("OSSE_ALPHA", "0.3"))},
}
# POST-HOC (added after the A1 cycle-0/1 diagnosis, not pre-registered): A1's configuration with the second
# estimator version — ObsBias gain with the ensemble's coherent forecast-error variance and sigma_o-relative priors,
# AdaptiveInflation sampling count = distinct profiles. Cycle 0's forecast is A1's (identical ICs/LBCs/engine, a
# deterministic single-rank WRF): its members and hourly metrics are reused, the analyses from 03Z on are new.
ARMS["A1b"] = {**ARMS["A1"], "reuse_c0_forecast": "A1", "post_hoc": True,
               "estimator": "v2: coherent-variance ObsBias (var0 0.25 so2, q 0.01 so2), profile n_eff"}
# PROSPECTIVE final-code arm (registered in MAILBOX before it ran; critic cr-w41-da-ens item 4): the final code
# (estimator v2 + rebalanced state inflation), a NEW truth (seed 778, never run before) and NEW observation-noise
# seeds; same A1 configuration. Its 03Z prior ensemble is A1's (identical member inputs, deterministic engine: the
# member perturbation method is not tuned on the truth), re-scored against the new truth; analyses from 03Z on new.
ARMS["F1"] = {**ARMS["A1"], "reuse_c0_forecast": "A1", "truth": "truth_f1", "seed_truth": 778,
              "obs_seed_offset": 50000, "prospective": True,
              "estimator": "v2 + rebalanced state inflation (final code)"}
PARAMS = dict(loc_h_km=30.0, stride_h=1, stride_v=1, pmax=400, negative_qv="clip",
              diagnostic_variables=("T2", "Q2", "U10", "V10", "PSFC"), workers=2)


def run_arm(name):
    cfg = ARMS[name]
    base = ROOT / name
    base.mkdir(parents=True, exist_ok=True)
    report_path = base / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"arm": name, "config": cfg,
                                                                                "cycles": []}
    geo = geometry()
    ny, nx = geo["XLAT"].shape
    net = network(geo)
    z = weights(SEED_W, N)
    labels = [f"m{i:02d}" for i in range(N)]
    times = lbc_ens.record_times(CASE / "wrfbdy_d01")
    # --- initial members + boundaries
    reuse = ROOT / cfg["reuse_c0_forecast"] if cfg.get("reuse_c0_forecast") else None
    ttag, obs_off = cfg.get("truth", "truth"), int(cfg.get("obs_seed_offset", 0))
    if ttag != "truth":
        truth(ttag, int(cfg["seed_truth"]))
    ic_dir = reuse / "c0_inputs" if reuse else base / "c0_inputs"
    if reuse and not (ic_dir / "DONE").exists():
        raise RuntimeError(f"{name}: {ic_dir} not ready")
    if not (ic_dir / "DONE").exists():
        shutil.rmtree(ic_dir, ignore_errors=True)
        ic_dir.mkdir(parents=True)
        fields, dev = build_members(z, with_mu=cfg["with_mu"], untapered=cfg["untapered"], geo=geo, ny=ny, nx=nx)
        for lab, f in zip(labels, fields):
            state_io.write_analysis_wrfinput(CASE / "wrfinput_d01", CASE / "wrfinput_d01", wt(T0), f,
                                             ic_dir / lab / "wrfinput_d01", land_cycle=False,
                                             extra_provenance={"kind": f"OSSE {name} IC", "member": lab})
        if cfg["perturbed_lbc"]:
            recs = lbc_records(z, times, END)
            # IC/LBC consistency in the specified ring: written IC deviation vs record-0 boundary deviation
            ctrl_ic = state_io.read_frame(CASE / "wrfinput_d01", wt(T0), LBC_NAMES)
            cons = 0.0
            for lab in labels:
                mic = state_io.read_frame(ic_dir / lab / "wrfinput_d01", wt(T0), LBC_NAMES)
                icdev = {n: np.asarray(mic[n], np.float64) - np.asarray(ctrl_ic[n], np.float64) for n in LBC_NAMES}
                cons = max(cons, max(lbc_ens.ic_consistency(icdev, recs[lab][times[0]], 5).values()))
            man = lbc_ens.write_member_wrfbdys(CASE / "wrfbdy_d01", CASE / "wrfinput_d01", recs, ic_dir / "lbc")
            report["lbc"] = {"ic_record0_ring_max_abs_dev_diff": cons,
                             "max_state_ulp": max(max(r["physical_encoding_max_state_ulp"].values())
                                                  for r in man["members"].values())}
        report["ic_balance"] = perturb.balance_diagnostics(dev, geo)
        (ic_dir / "DONE").write_text("ok")
    bdy = {lab: (ic_dir / "lbc" / lab / "wrfbdy_d01" if cfg["perturbed_lbc"] else CASE / "wrfbdy_d01")
           for lab in labels}
    ics = {lab: ic_dir / lab / "wrfinput_d01" for lab in labels}
    state_in = bias_in = None
    for c in range(N_ANALYSES + 1):
        if c < len(report["cycles"]) and report["cycles"][c].get("done"):
            ics = {lab: Path(p) for lab, p in report["cycles"][c]["next_ics"].items()}
            state_in, bias_in = report["cycles"][c].get("state_out"), report["cycles"][c].get("bias_out")
            continue
        start = T0 + timedelta(hours=CYCLE_H * c)
        t_a = start + timedelta(hours=CYCLE_H)
        cdir = base / f"c{c}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir(parents=True)
        fcdir = cdir
        if reuse and c == 0:
            src = json.loads((reuse / "report.json").read_text())["cycles"][0]
            fcdir = reuse / "c0"
            row = {"cycle": c, "start": wt(start), "t_a": wt(t_a), "forecast_reused_from": str(fcdir),
                   **{k: src[k] for k in ("forecast_wall_s", "member_wall_s_mean", "forecast")}}
            if ttag != "truth":   # other truth: only the kept t_a frames can be re-scored
                fr = stack([state_io.read_frame(fcdir / lab / f"wrfout_d01_{wt(t_a)}", wt(t_a),
                                                state_io.FRAME_VARIABLES) for lab in labels])
                row["forecast"] = {f"+{CYCLE_H}h": summary(fr, truth_frame(t_a, ttag), geo)}
        else:
            dirs = []
            for lab in labels:
                make_rundir(cdir / lab, ics[lab], bdy[lab], start, CYCLE_H)
                dirs.append(cdir / lab)
            t_run = time.time()
            walls = run_many(dirs)
            row = {"cycle": c, "start": wt(start), "t_a": wt(t_a), "forecast_wall_s": time.time() - t_run,
                   "member_wall_s_mean": float(np.mean(list(walls.values())))}
            hourly = {}
            for h in range(0, CYCLE_H + 1):
                th = start + timedelta(hours=h)
                fr = stack([state_io.read_frame(cdir / lab / f"wrfout_d01_{wt(th)}", wt(th), state_io.FRAME_VARIABLES)
                            for lab in labels])
                hourly[f"+{h}h"] = summary(fr, truth_frame(th, ttag), geo)
            row["forecast"] = hourly
        if c == N_ANALYSES:
            row["done"] = True
            report["cycles"][c:] = [row]
            report_path.write_text(json.dumps(report, indent=1, default=float) + "\n")
            break
        # --- analysis
        paths = [fcdir / lab / f"wrfout_d01_{wt(t_a)}" for lab in labels]
        frames = stack([state_io.read_frame(p, wt(t_a), state_io.FRAME_VARIABLES) for p in paths])
        tf = truth_frame(t_a, ttag)
        obs = make_obs(tf, geo, net, c, obs_off)
        t_an = time.time()
        params = lan.LetkfParams(**PARAMS, rtps_alpha=cfg["rtps_alpha"])
        hooks = {}
        if cfg["adaptive"]:
            # the product path: inflation.validate_config_section -> calibrate -> analysis_hooks (da-core API)
            raw = {"method": "adaptive", "adaptive": {"state_in": state_in, "state_out": str(cdir / "adaptive_state.json")},
                   "obs_bias": {"by": "group", "apply_kinds": ["surface"], "state_in": bias_in,
                                "state_out": str(cdir / "obs_bias.json")},
                   "maintenance": {"method": "rtps", "alpha": cfg["rtps_alpha"]}}
            if cfg["additive"]:
                pool_path = cdir / "additive_pool.npz"
                spread.write_pool_npz(pool_path, spread.pool_from_states(lagged_pool(t_a, LBC_NAMES), LBC_NAMES),
                                      {"source": "RD11 CPU-WRF lagged states t_a+0..9h", "t_a": wt(t_a)})
                raw["additive"] = {"pool_npz": str(pool_path), "amplitude": cfg["additive"], "seed": 1000 + c}
            sec = inflation.validate_config_section(raw)
            prior_frames, info = inflation.calibrate(frames, geo, obs, {"kind": "members", "scale": 1.0}, sec)
            obs_used = info.pop("obs", obs)
            hooks = inflation.analysis_hooks(sec)
            row["inflation"] = {"table": info["adaptive"]["table"], "estimates": info["adaptive"]["estimates"],
                                "bias_update": info.get("bias_update"), "bias_correction": info.get("bias_correction")}
            state_in, bias_in = sec["adaptive"]["state_out"], sec["obs_bias"]["state_out"]
            row["state_out"], row["bias_out"] = state_in, bias_in
        else:
            from gpuwrf.assimilation.letkf.api import calibrate_scale
            prior_frames, info = calibrate_scale(frames, geo, obs, {"kind": "members", "scale": "innovation"})
            obs_used = obs
            params = lan.LetkfParams(**PARAMS, rtps_alpha=cfg["rtps_alpha"], inflation=float(info["inflation_rho"]))
            row["inflation"] = {"rho": float(info["inflation_rho"]), "by_kind": info.get("by_kind")}
        if cfg["var_loc"]:
            hooks["var_loc"] = varloc.as_var_loc(cfg["var_loc"], ("U", "V", "T", "QVAPOR") + PARAMS["diagnostic_variables"])
        analysis, receipt = lan.analyse(prior_frames, geo, obs_used, params, **hooks)
        row["analysis_wall_s"] = time.time() - t_an
        import resource
        row["peak_rss_mb_process"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        row["prior_raw"] = summary(frames, tf, geo)
        row["prior_inflated"] = summary(prior_frames, tf, geo) if prior_frames is not frames else None
        row["posterior"] = summary(analysis, tf, geo)
        row["receipt"] = {k: receipt.get(k) for k in ("admissible", "refusal_reasons", "qc", "assimilated", "counters",
                                                      "stats_assimilated", "local_obs", "hooks", "moisture")}
        # --- write next ICs from each member's own background frame
        nxt = {}
        for i, lab in enumerate(labels):
            fields = {n: np.asarray(analysis[n][i]) for n in ("U", "V", "T", "QVAPOR", "P", "PH")}
            out = cdir / "next" / lab / "wrfinput_d01"
            state_io.write_analysis_wrfinput(CASE / "wrfinput_d01", paths[i], wt(t_a), fields, out, land_cycle=False,
                                             extra_provenance={"kind": f"OSSE {name} analysis", "member": lab})
            nxt[lab] = str(out)
        written = stack([state_io.read_frame(Path(nxt[lab]), wt(t_a), ("U", "V", "T", "QVAPOR", "PH", "PHB"))
                         for lab in labels])
        row["written"] = summary(written, tf, geo)
        row["next_ics"] = nxt
        row["done"] = True
        ics = {lab: Path(p) for lab, p in nxt.items()}
        report["cycles"][c:] = [row]
        report_path.write_text(json.dumps(report, indent=1, default=float) + "\n")
        # keep the analysis-time frames only (disk)
        for lab in labels:
            for f in (cdir / lab).glob("wrfout_d01_*"):
                if f.name != f"wrfout_d01_{wt(t_a)}":
                    f.unlink()
        print(f"{name} cycle {c} done: forecast {row['forecast_wall_s']:.0f}s analysis {row['analysis_wall_s']:.1f}s",
              flush=True)
    return report


# --------------------------------------------------------------------------- balance experiment (R3 item 5)

def balance_experiment(n=int(os.environ.get("OSSE_BAL_N", "10")), hours=1):
    """First-hour spread retention of three IC constructions with the SAME combination weights, all tapered, shared
    wrfbdy (only the balance differs): B0 frozen MU (R3 archive style, fixed-MU rebalance), B1 MU included +
    rebalance_with_mu, B2 spun-up: B1 members forecast 3 h, deviations at 03Z recentred on the reference 03Z state
    (model-balanced). Spread by band at +0 h (as written) and +1 h; retention = +1h/+0h."""
    base = ROOT / "balance"
    base.mkdir(parents=True, exist_ok=True)
    geo = geometry()
    ny, nx = geo["XLAT"].shape
    z = weights(SEED_W + 1, n)
    labels = [f"m{i:02d}" for i in range(n)]
    out = {"n": n, "hours": hours}
    t3 = T0 + timedelta(hours=3)
    for arm, with_mu in (("B0", False), ("B1", True)):
        d = base / arm
        if not (d / "DONE").exists():
            shutil.rmtree(d, ignore_errors=True)
            fields, dev = build_members(z, with_mu=with_mu, untapered=False, geo=geo, ny=ny, nx=nx)
            dirs = []
            for lab, f in zip(labels, fields):
                state_io.write_analysis_wrfinput(CASE / "wrfinput_d01", CASE / "wrfinput_d01", wt(T0), f,
                                                 d / "ic" / lab / "wrfinput_d01", land_cycle=False)
                make_rundir(d / "run" / lab, d / "ic" / lab / "wrfinput_d01", CASE / "wrfbdy_d01", T0,
                            3 if arm == "B1" else hours)
                dirs.append(d / "run" / lab)
            run_many(dirs)
            (d / "DONE").write_text("ok")
    # B2: spun-up deviations of the B1 forecasts at 03Z, recentred on the reference 03Z frame
    d = base / "B2"
    if not (d / "DONE").exists():
        shutil.rmtree(d, ignore_errors=True)
        mem = stack([state_io.read_frame(base / "B1" / "run" / lab / f"wrfout_d01_{wt(t3)}", wt(t3), STATE)
                     for lab in labels])
        dev = perturb.forecast_deviations(mem, STATE)
        ref = state_io.read_frame(REF / f"wrfout_d01_{wt(t3)}", wt(t3), ("U", "V", "T", "QVAPOR", "MU", "P", "PH",
                                                                         "P_HYD"))
        tap = tapers(ny, nx)
        rec = perturb.recentre({k: np.asarray(ref[k], np.float64) for k in STATE}, dev, 1.0,
                               taper={k: tap[k] for k in STATE})
        dirs = []
        for i, lab in enumerate(labels):
            an = {**ref, **{k: rec[k][i] for k in STATE}}
            reb = perturb.rebalance_with_mu(ref, an, geo)
            f = {k: np.asarray(an[k]).astype(np.asarray(ref[k]).dtype) for k in ("U", "V", "T", "QVAPOR")}
            f.update({k: reb[k] for k in ("P", "PH", "P_HYD", "MU")})
            state_io.write_analysis_wrfinput(CASE / "wrfinput_d01", REF / f"wrfout_d01_{wt(t3)}", wt(t3), f,
                                             d / "ic" / lab / "wrfinput_d01", land_cycle=False)
            make_rundir(d / "run" / lab, d / "ic" / lab / "wrfinput_d01", CASE / "wrfbdy_d01", t3, hours)
            dirs.append(d / "run" / lab)
        run_many(dirs)
        (d / "DONE").write_text("ok")
    for arm, t_start in (("B0", T0), ("B1", T0), ("B2", t3)):
        rows = {}
        for h in range(0, hours + 1):
            th = t_start + timedelta(hours=h)
            fr = stack([state_io.read_frame(base / arm / "run" / lab / f"wrfout_d01_{wt(th)}", wt(th),
                                            ("U", "V", "T", "QVAPOR", "PH", "PHB", "T2", "U10", "V10")) for lab in labels])
            rows[f"+{h}h"] = spread.spread_by_band(fr, geo, interior=INTERIOR)
        dev_ic = {k: v - v.mean(0) for k, v in stack([state_io.read_frame(base / arm / "ic" / lab / "wrfinput_d01",
                                                                          wt(t_start), ("U", "V", "MU"))
                                                       for lab in labels]).items()}
        out[arm] = {"spread": rows, "retention_1h": spread.survival(rows["+0h"], rows[f"+{hours}h"]),
                    "ic_balance": perturb.balance_diagnostics(dev_ic, geo)}
    (base / "BALANCE.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


if __name__ == "__main__":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    what = sys.argv[1]
    if what == "truth":
        truth(*([sys.argv[2], int(sys.argv[3])] if len(sys.argv) > 3 else []))
    elif what == "arm":
        run_arm(sys.argv[2])
    elif what == "balance":
        balance_experiment()
    else:
        raise SystemExit(__doc__)
