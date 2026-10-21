"""CW01: pristine WRF acoustic small step (module_small_step_em.F calc_coef_w, advance_uv, advance_mu_t, advance_w incl.
damp_opt=3 w-Rayleigh, calc_p_rho; b-core pristine_oracle.py REAL4 binary32 adapter) vs the RELEASE native acoustic kernels
(kernels/dyn_acoustic_fp32 calc_coef_fp32 + acoustic_substep_fp32, fast-default flags W2PASS=3/MASS_BLOCK/UV_MASKED/W_RECUR_SL)
on a REAL strongly convective 1 km tile: #148 CPU-reference d06 τ24 -> τ25 (|W| up to 19.5 m/s, rain 81 mm/h).
Stage operands: the b-core build_savepoints recipe unchanged (real_state.load_real_snapshot on a minimal wrfout-format run
dir written from the guarded-extracted frames; stage_advanced_state hourly increment scaled to dt; small_step_prep_wrf,
calc_p_rho_wrf, operational _acoustic_core_state_from_prep for pg_buoy_w/rhs_ph/cqw) with dt 6 s, dx 1 km, NESTED config,
w_damping 1 / damp_opt 3 / dampcoef .2 / zdamp 5000 / epssm .5 (namelist). verify_pristine protocol: w_damping=0 in the
substep (w_damp lives in rk_tendency, CFL census CD-W: inactive), WRF stage trips 1/2/4 at dts = dt/3, dt/4, dt/4.
Gate (FROZEN, verify_pristine Tier-P): rms <= 2e-5 max(1,rms_ref) + 2 rms(E), max <= 2e-4 max(1,max_ref) + 2 max(E),
E = pristine fp-contract fast vs off; coefficient gate compare(); 0 guard events.
usage: acoustic_oracle.py <frame_a.npz> <frame_b.npz> <out.json>"""
import json, os, sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/v025"), str(ROOT / "tests/v025/b_core")]
if os.environ.get("W43_PLATFORM", "cpu") == "cpu":
    import cpu_guard  # noqa: F401  real_state's loader requires it before jax/gpuwrf
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
import numpy as np
import netCDF4
import jax, jax.numpy as jnp
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv")
RUN = LANE / "acoustic_run"
PLATFORM = os.environ.get("W43_PLATFORM", "cpu")
assert jax.devices()[0].platform == PLATFORM


def write_wrfout(npz, path, times):
    z = np.load(npz)
    nz, ny, nx = z["T"].shape
    dims = {"Time": None, "DateStrLen": 19, "bottom_top": nz, "bottom_top_stag": nz + 1, "south_north": ny,
            "south_north_stag": ny + 1, "west_east": nx, "west_east_stag": nx + 1, "soil_layers_stag": 4}
    names3 = {(nz, ny, nx): ("bottom_top", "south_north", "west_east"), (nz + 1, ny, nx): ("bottom_top_stag", "south_north", "west_east"),
              (nz, ny, nx + 1): ("bottom_top", "south_north", "west_east_stag"), (nz, ny + 1, nx): ("bottom_top", "south_north_stag", "west_east"),
              (4, ny, nx): ("soil_layers_stag", "south_north", "west_east")}
    names2 = {(ny, nx): ("south_north", "west_east"), (ny, nx + 1): ("south_north", "west_east_stag"), (ny + 1, nx): ("south_north_stag", "west_east")}
    with netCDF4.Dataset(path, "w") as nc:
        for d, n in dims.items():
            nc.createDimension(d, n)
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0, :] = np.frombuffer(times.encode(), dtype="S1")
        for k in z.files:
            if k.startswith("attr_"):
                nc.setncattr(k[5:], z[k].item() if z[k].ndim == 0 else z[k])
                continue
            a = z[k]
            if a.dtype.kind not in "fi":
                continue
            if a.ndim == 3:
                dd = names3.get(a.shape)
            elif a.ndim == 2:
                dd = names2.get(a.shape)
            elif a.ndim == 1:
                dd = ("bottom_top",) if a.shape[0] == nz else ("bottom_top_stag",) if a.shape[0] == nz + 1 else ("soil_layers_stag",) if a.shape[0] == 4 else None
            else:
                dd = ()
            if dd is None:
                continue
            v = nc.createVariable(k, a.dtype, ("Time",) + dd)
            v[0] = a


def make_run(fa, fb):
    RUN.mkdir(parents=True, exist_ok=True)
    za = np.load(fa)
    nz, ny, nx = za["T"].shape
    write_wrfout(fa, RUN / "wrfout_d01_2026-10-19_18:00:00", "2026-10-19_18:00:00")
    write_wrfout(fb, RUN / "wrfout_d01_2026-10-19_19:00:00", "2026-10-19_19:00:00")
    (RUN / "namelist.input").write_text(f"""&time_control
 start_year = 2026, start_month = 10, start_day = 19, start_hour = 18,
 end_year = 2026, end_month = 10, end_day = 19, end_hour = 19,
 history_interval = 60, frames_per_outfile = 1,
/
&domains
 time_step = 6, max_dom = 1, e_we = {nx + 1}, e_sn = {ny + 1}, e_vert = {nz + 1},
 dx = {float(za['attr_DX'])}, dy = {float(za['attr_DY'])}, p_top_requested = {float(za['P_TOP'])},
/
&physics
 mp_physics = 8, ra_lw_physics = 4, ra_sw_physics = 4, radt = 30, sf_sfclay_physics = 5, sf_surface_physics = 4,
 bl_pbl_physics = 5, cu_physics = 0,
/
&dynamics
 hybrid_opt = 2, w_damping = 1, diff_opt = 1, km_opt = 4, diff_6th_opt = 2, diff_6th_factor = 0.12, damp_opt = 3,
 zdamp = 5000., dampcoef = 0.2, epssm = 0.5, non_hydrostatic = .true., moist_adv_opt = 1, scalar_adv_opt = 1,
/
&bdy_control
 specified = .false., nested = .true.,
/
""")


def build_state(dt, dx):
    from real_state import load_real_snapshot, stage_advanced_state
    from gpuwrf.contracts.state import Tendencies
    from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG
    from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig
    from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
    from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
    from gpuwrf.runtime.operational_mode import _acoustic_core_state_from_prep
    before = load_real_snapshot(RUN, domain="d01", wrfout_name="wrfout_d01_2026-10-19_18:00:00")
    snap = load_real_snapshot(RUN, domain="d01", wrfout_name="wrfout_d01_2026-10-19_19:00:00")
    advanced, pair = stage_advanced_state(before, snap, dt)
    prep = small_step_prep_wrf(advanced, 2, dt, metrics=snap.metrics, reference_state=snap.state, base_state=snap.base_state)
    pressure = calc_p_rho_wrf(prep)
    grid = SimpleNamespace(projection=SimpleNamespace(dx_m=dx, dy_m=dx))
    nl = SimpleNamespace(grid=grid, metrics=snap.metrics, dt_s=dt, top_lid=False, run_boundary=False, h_sca_adv_order=5, w_damping=1,
                         boundary_config=DEFAULT_BOUNDARY_CONFIG)
    z = jnp.zeros_like
    tend = Tendencies(u=z(prep.u_work), v=z(prep.v_work), w=z(prep.w_work), theta=z(prep.theta_work), ph=z(prep.ph_work),
                      mu=z(prep.mu_work), qv=z(prep.theta_work), p=z(prep.theta_work))
    carry = SimpleNamespace(ww=prep.ww_save, mudf=z(prep.mu_work))
    state = _acoustic_core_state_from_prep(carry, prep, pressure, nl, tend).replace(mu_work=prep.mu_work)
    cfg = AcousticCoreConfig(dt=dt / 4., dx=dx, dy=dx, epssm=float(snap.config["epssm"]), dt_full=dt, periodic_x=False,
                             specified=False, nested=True, w_damping=0, damp_opt=3, dampcoef=.2, zdamp=5000.)
    f32 = {k: (jnp.asarray(v, jnp.float32) if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating) else v)
           for k, v in state.to_dict().items() if v is not None}
    return state.replace(**f32), cfg, pair


def verify(state, cfg, steps, dts, interpret):
    from bench_acoustic import compare
    from pristine_oracle import Oracle, build
    from verify_pristine import FIELDS
    from gpuwrf.kernels.dyn_acoustic_fp32 import acoustic_substep_fp32, calc_coef_fp32
    cfg = replace(cfg, dt=dts)
    libs = {}
    for name, kw in (("reference", {}), ("fast", dict(contraction="fast", native=True)), ("off", dict(contraction="off", native=True))):
        p = LANE / "acoustic_pristine" / name / "oracle.so"
        libs[name] = p if p.exists() else build(p.parent, **kw)
    o, fast, off = (Oracle(libs[n], state, cfg) for n in ("reference", "fast", "off"))
    for x in (o, fast, off):
        x.run(1)
    pcfg = replace(cfg, epssm=0.1) if os.environ.get("W43_MUTANT") == "epssm" else cfg  # CW02 port-only mutant
    co = jax.jit(lambda s: calc_coef_fp32(s, pcfg, interpret=interpret))(state)
    coefficients = {n: compare(o.field(n)[1:], v[1:]) for n, v in co.items()}
    fn = jax.jit(lambda s: acoustic_substep_fp32(s, coefficients=co, cfg=pcfg, interpret=interpret, return_guard_events=True))
    res = {"dts": dts, "coefficients": coefficients, "steps": {}, "guard_events": {}}
    for step in range(1, steps + 1):
        for ph in range(2, 6):
            for x in (o, fast, off):
                x.run(ph)
        state, events = fn(state)
        errs = {}
        for name, ref in FIELDS.items():
            e = compare(o.field(ref), getattr(state, name))
            d = fast.field(ref).astype(np.float64) - off.field(ref).astype(np.float64)
            cr, cm = float(np.sqrt(np.mean(d * d))), float(np.abs(d).max())
            e.update(frozen_passed=e["passed"], tier_p_rms_limit=e["rms_limit"] + 2 * cr, tier_p_max_limit=e["max_limit"] + 2 * cm)
            e["passed"] = bool(e["finite"] and e["rms"] <= e["tier_p_rms_limit"] and e["max_abs"] <= e["tier_p_max_limit"])
            errs[name] = e
        res["steps"][str(step)] = errs
        res["guard_events"][str(step)] = int(np.asarray(events).sum())
    res["max_w_ref"] = float(np.abs(o.field("w")).max())
    res["passed"] = all(v["passed"] for v in coefficients.values()) and all(
        v["passed"] for e in res["steps"].values() for v in e.values()) and all(v == 0 for v in res["guard_events"].values())
    return res


def save_fixture(state, cfg, pair, path):
    from dataclasses import asdict
    np.savez(path, **{"state_" + k: np.asarray(v) for k, v in state.to_dict().items() if v is not None})
    Path(str(path) + ".json").write_text(json.dumps(dict(config=asdict(cfg), pair=str(pair)[:2000]), default=str))


def load_fixture(path):
    from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, AcousticCoreState
    z = np.load(path)
    meta = json.loads(Path(str(path) + ".json").read_text())
    state = AcousticCoreState(**{k[6:]: jnp.asarray(z[k], z[k].dtype) for k in z.files})
    return state, AcousticCoreConfig(**meta["config"]), meta["pair"]


def main(fa, fb, out):
    # usage variants: <frame_a> <frame_b> <out.json> [--save-fixture F] | --from-fixture F <out.json> (GPU arm CG08)
    if fa == "--from-fixture":
        state, cfg, pair = load_fixture(fb)
        dt = float(cfg.dt_full)
    else:
        dt, dx = 6.0, float(np.load(fa)["attr_DX"])
        make_run(fa, fb)
        state, cfg, pair = build_state(dt, dx)
        if "--save-fixture" in sys.argv:
            save_fixture(state, cfg, pair, sys.argv[sys.argv.index("--save-fixture") + 1])
    print("state ready", state.theta.shape, "max|w|", float(jnp.abs(state.w).max()), flush=True)
    interpret = PLATFORM == "cpu"
    stages = [verify(state, cfg, n, d, interpret) for n, d in ((1, dt / 3.), (2, dt / 4.), (4, dt / 4.))]
    res = dict(protocol="WRF RK trips 1/2/4 (sound steps 4), separate stage savepoint arms", stage_pair=str(pair)[:500],
               stages=stages, passed=all(s["passed"] for s in stages))
    Path(out).write_text(json.dumps(res, indent=1, default=str))
    for i, s in enumerate(stages):
        bad = {f"{st}:{k}": (round(v["max_abs"], 6), round(v["tier_p_max_limit"], 6)) for st, e in s["steps"].items() for k, v in e.items() if not v["passed"]}
        worst = max(((v["max_abs"] / v["tier_p_max_limit"]), f"{st}:{k}") for st, e in s["steps"].items() for k, v in e.items())
        print("stage", i + 1, "passed", s["passed"], "guards", s["guard_events"], "worst ratio %.3f %s" % worst, "fails", bad, flush=True)
    print("PASSED", res["passed"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
