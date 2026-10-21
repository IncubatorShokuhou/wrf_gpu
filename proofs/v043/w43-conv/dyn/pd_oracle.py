"""CA01: pristine WRF advect_scalar_pd (moist_adv_opt=1 positive-definite 5th/3rd-order scalar advection; b-core
tests/v025/b_core/pd_scalar_oracle.py, REAL4, specified) vs the RELEASE native kernel kernels/dyn_pd_fp32
.advect_scalar_pd_fp32 (GPUWRF_DYN_PD_FP32 + SPECIES_LOOP from fast defaults; interpret on CPU) on WHOLE real convective
CPU-reference frames, all moist species + Nr/Ni stacked. Operands (same to both): q = qold = history species (PD at the
last RK stage), mut = MU+MUB, mub = MUB, muold = MU; ru = (c1h*muu+c2h) U / msfuy, rv = (c1h*muv+c2h) V / msfvx
(couple_momentum), muu/muv = calc_mu_uv edge averages; rom = ww from the PRISTINE calc_ww_cp (b-diff pristine_dyn_real,
module_em.F:163 wiring). dt = frame dt (stage-3 dt).
Gate (FROZEN, b-core BC42 / b-carry C05): interior ring>=1, max|d| <= max(8 max E, 1e-5 max|ref|), rms <= max(8 rms E,
1e-6 max|ref|), E = |pristine(fp-contract fast) - pristine(off)|; limiter ACTIVE: cells where pristine PD differs from
the pristine plain advect_scalar (b-core scalar_oracle) by > 1e-6 max|ref| must be > 0 (as C05).
usage: pd_oracle.py <frame.npz> [...] --out out.json"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/v025/b_core"), str(ROOT / "tests/v025/b_diff")]
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
import numpy as np
import jax, jax.numpy as jnp
import pd_scalar_oracle as PD
import scalar_oracle as SO
import pristine_dyn_real as DR
from gpuwrf.kernels.dyn_pd_fp32 import advect_scalar_pd_fp32
PLATFORM = os.environ.get("W43_PLATFORM", "cpu")
assert jax.devices()[0].platform == PLATFORM, jax.devices()
assert os.environ.get("GPUWRF_DYN_PD_FP32") == "1"
LANE = Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv")
SPECIES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "QNRAIN", "QNICE")


def operands(z):
    g = lambda v: np.asarray(z[v], np.float32)  # noqa: E731
    U, V, MU, MUB = g("U"), g("V"), g("MU"), g("MUB")
    nz, ny, nx = g("T").shape
    mut = MU + MUB
    mp = np.pad(mut, 1, mode="edge")
    muu = 0.5 * (mp[1:-1, :-1] + mp[1:-1, 1:])          # (ny, nx+1) calc_mu_uv
    muv = 0.5 * (mp[:-1, 1:-1] + mp[1:, 1:-1])          # (ny+1, nx)
    c1h, c2h = g("C1H")[:, None, None], g("C2H")[:, None, None]
    ru = (c1h * muu + c2h) * U / g("MAPFAC_UY")[None]
    rv = (c1h * muv + c2h) * V / g("MAPFAC_VX")[None]
    lib = DR.build(LANE / "dynreal_pristine")
    o = DR.Oracle(lib, (nz, ny, nx))
    for n, v in (("u", U), ("v", V), ("mu", MU), ("mub", MUB), ("mut", mut), ("msftx", g("MAPFAC_MX")), ("msfty", g("MAPFAC_MY")),
                 ("msfux", g("MAPFAC_UX")), ("msfuy", g("MAPFAC_UY")), ("msfvx", g("MAPFAC_VX")), ("msfvx_inv", 1.0 / g("MAPFAC_VX")),
                 ("msfvy", g("MAPFAC_VY")), ("c1h", g("C1H")), ("c2h", g("C2H")), ("dnw", g("DNW")),
                 ("rdx", np.float32(1.0 / float(z["attr_DX"]))), ("rdy", np.float32(1.0 / float(z["attr_DY"])))):
        o.set(n, v)
    o.run("calc_ww_cp")
    rom = o.field("ww", (nz + 1, ny, nx))
    return dict(mut=mut, mub=MUB, muold=MU, ru=ru, rv=rv, rom=rom, mx=g("MAPFAC_MX"), my=g("MAPFAC_MY"), ux=g("MAPFAC_UX"),
                uy=g("MAPFAC_UY"), vx=g("MAPFAC_VX"), vy=g("MAPFAC_VY"), fzm=g("FNM"), fzp=g("FNP"), rdzw=g("RDNW"),
                c1=g("C1H"), c2=g("C2H"))


def frame(path, libs):
    z = np.load(path)
    dt, rdx, rdy = float(z["attr_DT"]), 1.0 / float(z["attr_DX"]), 1.0 / float(z["attr_DY"])
    op = operands(z)
    q = np.stack([np.asarray(z[s], np.float32) for s in SPECIES])
    f = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
    got = advect_scalar_pd_fp32(f(q), f(q), f(op["ru"]), f(op["rv"]), f(op["rom"]), f(op["mut"]), f(op["muold"]),
                                f(op["c1"]), f(op["c2"]), f(op["mx"]), f(op["my"]), f(op["rdzw"]), f(op["fzm"]), f(op["fzp"]),
                                rdx=rdx, rdy=rdy, dt=dt, mub=f(op["mub"]), interpret=PLATFORM == "cpu",
                                block=16384 if PLATFORM == "cpu" else 256)
    got = np.asarray(jax.block_until_ready(got), np.float64)
    rows = {"dt": dt, "max_rom": float(np.abs(op["rom"]).max())}
    for i, sp in enumerate(SPECIES):
        arrays = dict(q=q[i], qold=q[i], ru=op["ru"], rv=op["rv"], rom=op["rom"], mut=op["mut"], mub=op["mub"], muold=op["muold"],
                      mx=op["mx"], my=op["my"], ux=op["ux"], uy=op["uy"], vx=op["vx"], vy=op["vy"], fzm=op["fzm"], fzp=op["fzp"],
                      rdzw=op["rdzw"], c1=op["c1"], c2=op["c2"])
        off = PD.evaluate(libs["off"], arrays, rdx, rdy, dt).astype(np.float64)
        fast = PD.evaluate(libs["fast"], arrays, rdx, rdy, dt).astype(np.float64)
        sl = (slice(None), slice(1, -1), slice(1, -1))
        d, env = (got[i] - off)[sl], (fast - off)[sl]
        refmax = float(np.abs(off[sl]).max()) or 1e-30
        mb = max(8 * float(np.abs(env).max()), 1e-5 * refmax)
        rb = max(8 * float(np.sqrt(np.mean(env ** 2))), 1e-6 * refmax)
        plain = SO.evaluate(libs["plain"], dict(q=arrays["q"], ru=arrays["ru"], rv=arrays["rv"], rom=arrays["rom"], mu=arrays["mut"],
                            map=arrays["mx"], fzm=arrays["fzm"], fzp=arrays["fzp"], rdzw=arrays["rdzw"], c1=arrays["c1"],
                            c2=arrays["c2"]), rdx).astype(np.float64)
        limited = int(np.count_nonzero(np.abs(off - plain)[sl] > 1e-6 * refmax))
        stress = int(np.count_nonzero((np.abs(dt * off) > 0.5 * np.abs(q[i])) & (q[i] > 0)))
        r = dict(ref_max=refmax, max_abs=float(np.abs(d).max()), rms=float(np.sqrt(np.mean(d * d))), max_bound=mb, rms_bound=rb,
                 env_max=float(np.abs(env).max()), q_max=float(q[i].max()), strong_update_cells=stress, limiter_cells=limited,
                 neg_after=int(np.count_nonzero(q[i][sl] + dt * got[i][sl] < -1e-12 * max(q[i].max(), 1e-30))),
                 finite=bool(np.isfinite(got[i]).all()))
        # ABSENT species (q == 0 in the whole frame; added after the d04 tau62 ice-free frame, E154 census, no bound change):
        # both sides must be EXACTLY zero; it is reported, not counted as coverage. Present species keep the frozen gate.
        r["absent"] = bool(not np.any(q[i]))
        if r["absent"]:
            r["passed"] = bool(r["finite"] and r["max_abs"] == 0.0 and float(np.abs(off).max()) == 0.0)
        else:
            r["passed"] = bool(r["finite"] and r["max_abs"] <= mb and r["rms"] <= rb and limited > 0)
        rows[sp] = r
    return rows


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".npz")]
    dst = Path(sys.argv[sys.argv.index("--out") + 1])
    libs = {c: PD.build(LANE / f"pd_pristine_{c}", contraction=c) for c in ("off", "fast")}
    libs["plain"] = SO.build(LANE / "scalar_pristine_off", contraction="off")
    res = {}
    for p in paths:
        res[os.path.basename(p)] = r = frame(p, libs)
        print(os.path.basename(p), {s: (r[s]["passed"], "%.1e/%.1e" % (r[s]["max_abs"], r[s]["max_bound"]), r[s]["limiter_cells"]) for s in SPECIES}, flush=True)
    res["ALL_PASS"] = all(r[s]["passed"] for k, r in res.items() for s in SPECIES)
    res["COVERAGE_species_present_and_limited_frames"] = {s: sum(1 for k, r in res.items() if isinstance(r, dict) and not r[s].get("absent")
                                                              and r[s]["limiter_cells"] > 0 and r[s]["passed"]) for s in SPECIES}
    dst.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
