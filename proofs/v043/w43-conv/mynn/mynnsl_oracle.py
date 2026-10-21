"""CS01: pristine WRF MYNN surface layer (module_sf_mynn.F byte-identical, REAL*4, proofs/v090 driver with B39b WRF
constants; SFCLAY1D_mynn, isfflx 1, isftcflx 0, iz0tlnd 0, warm step itimestep 2) vs the RELEASE-default product
surface_layer_with_diagnostics (first_timestep False) on REAL convective / cold-pool / gust columns of the #148
CPU-reference frames -- the operator behind U10/V10/T2 (the #148 d04 U10 residual). Operands fed IDENTICALLY to both
(tests/v025/b_phys/mynn_sl_oracle_gate.py mapping): level-1/2 u v (mass averages), t1d = (T+300)*pi_phy, qv, p = P_HYD (caller p_hyd), dz8w,
rho (hypso2 log alt, caller_ops.py), mavail 1, xland, tsk, psfc, znt 0.1 land (water: Charnock inside), warm UST/HFX/QFX/PBLH from history, MOL
from an oracle pre-pass (MOL is not in history; same as mynn_sl_oracle_gate), land QSFC = Q2/(1+Q2), water QSFC recomputed (-1); qcg/snowh 0.
Kinds: coldpool, core, downdraft, gust (top-|V10| cells), env -- as CM01 plus gust.
Gate (FROZEN v090 THR, tests/test_v090_mynnsl_oracle_parity.py): relative max per field (ust .01 hfx .02 lh .03 qsfc .01
br/zol/mol/rmol .05 psim/psih .03 u10/v10 .01 t2/th2 .001 regime 0 znt 1e-3), absolute escape 1e-6.
usage: mynnsl_oracle.py <frames_dir> <out.json>"""
import glob, json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/v025/b_phys")]
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caller_ops import caller_fields
import jax, jax.numpy as jnp
from mynn_sl_oracle_gate import run_oracle, THR
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
PLATFORM = os.environ.get("W43_PLATFORM", "cpu")
assert jax.devices()[0].platform == PLATFORM, jax.devices()
G, RD, CP = np.float32(9.81), np.float32(287.0), np.float32(7.0 * 287.0 / 2.0)
NPICK = 16
rng = np.random.default_rng(20261022)
MUTANT = os.environ.get("W43_MUTANT", "")  # CS02 deletion mutants (port only): mol0 | nopblh
BUILD = Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv/mynnsl_pristine")


def build():
    if not (BUILD / "mynn_oracle").exists():
        subprocess.run(["bash", str(ROOT / "proofs/v090/build_oracle.sh")], check=True,
                       env={**os.environ, "OUT_DIR": str(BUILD), "ORACLE_CPUS": "8,9,12"}, capture_output=True)
    return BUILD / "mynn_oracle"


def columns(path):
    z = np.load(path)
    g = lambda v: np.asarray(z[v], np.float32)  # noqa: E731
    T, qv, qr, W, U, V = g("T"), g("QVAPOR"), g("QRAIN"), g("W"), g("U"), g("V")
    th = T + np.float32(300.0)
    cf = caller_fields(z)                      # WRF caller: p = p_hyd, pi_phy dynamic, rho from hypso2 alt
    p, exner, rho = cf["p"], cf["exner"], cf["rho"]
    dz = np.diff((g("PH") + g("PHB")) / G, axis=0)
    um, vm = 0.5 * (U[:, :, 1:] + U[:, :, :-1]), 0.5 * (V[:, 1:] + V[:, :-1])
    xl = g("XLAND")
    anom = np.zeros_like(th[0])
    for c in np.unique(xl):
        m = xl == c
        anom[m] = th[0][m] - np.median(th[0][m])
    v10 = np.hypot(g("U10"), g("V10"))
    kinds = {"coldpool": (anom <= -1.5) & (qr[:3].max(0) > 1e-5), "core": qr.max(0) >= 2e-3,
             "downdraft": W[:15].min(0) <= -2.0, "gust": v10 >= np.quantile(v10, 0.98), "env": qr.max(0) < 1e-8}
    picks = []
    for kind, m in kinds.items():
        yx = np.argwhere(m)
        if len(yx):
            picks += [(kind, y, x) for y, x in yx[rng.choice(len(yx), min(NPICK, len(yx)), replace=False)]]
    ys, xs = np.array([q[1] for q in picks]), np.array([q[2] for q in picks])
    two = lambda a: np.ascontiguousarray(a[:2, ys, xs].T).astype(np.float64)  # noqa: E731
    one = lambda a: np.asarray(a[ys, xs], np.float64)  # noqa: E731
    q2 = one(g("Q2"))
    land = one(xl) < 1.5
    view = dict(u=two(um), v=two(vm), theta=two(th), qv=two(qv), p=two(p), dz=two(dz), rho=two(rho), t_air=two(th * exner),
                exner=two(exner), mavail=np.ones(len(ys)), xland=one(xl), t_skin=one(g("TSK")), psfc=one(g("PSFC")),
                roughness_m=np.full(len(ys), 0.1), ustar=one(g("UST")))
    extras = dict(mol=np.zeros(len(ys)), hfx=one(g("HFX")), qfx=one(g("QFX")), pblh=one(g("PBLH")),
                  qsfc=np.where(land, q2 / (1.0 + q2), -1.0))
    cols = dict(u=view["u"][:, 0], v=view["v"][:, 0], t1d=view["t_air"][:, 0], qv=np.maximum(view["qv"][:, 0], 0.0),
                p1d=view["p"][:, 0], dz8w=view["dz"][:, 0], rho=view["rho"][:, 0], u1d2=view["u"][:, 1], v1d2=view["v"][:, 1],
                dz2w=view["dz"][:, 1], mavail=view["mavail"], xland=view["xland"], tsk=view["t_skin"], psfcpa=view["psfc"],
                qcg=np.zeros(len(ys)), snowh=np.zeros(len(ys)), znt=view["roughness_m"], ust=view["ustar"], **extras)
    meta = [dict(kind=k, y=int(y), x=int(x), th0_anom=float(anom[y, x]), v10=float(v10[y, x])) for k, y, x in picks]
    return view, extras, cols, meta, float(z["attr_DX"])


def main(frames, dst):
    exe = build()
    res, ok_all = {}, True
    for f in sorted(glob.glob(os.path.join(frames, "*.npz"))):
        view, extras, cols, meta, dx = columns(f)
        pre = run_oracle(exe, cols, dx, itimestep=2, real4=True)
        cols["mol"] = extras["mol"] = pre["mol"].astype(np.float32).astype(np.float64)
        wrf = run_oracle(exe, cols, dx, itimestep=2, real4=True)
        # trailing-z column view (1, n, 2) / (1, n): surface_layer._surface needs ndim >= 3 for level fields
        port_in = {**view, **extras}
        if MUTANT == "mol0":
            port_in["mol"] = np.zeros_like(port_in["mol"])
        elif MUTANT == "nopblh":
            port_in.pop("pblh")
        ns = SimpleNamespace(**{k: jnp.asarray(v.astype(np.float32))[None] for k, v in port_in.items()}, dx_m=dx)
        d = surface_layer_with_diagnostics(ns, first_timestep=False)
        fl = lambda a: np.asarray(a, np.float64).reshape(-1)  # noqa: E731
        port = dict(ust=fl(d.fluxes.ustar), mol=fl(d.mol), rmol=fl(d.rmol), zol=fl(d.zol), regime=fl(d.regime),
                    psim=fl(d.psim), psih=fl(d.psih), br=fl(d.br), hfx=fl(d.hfx), lh=fl(d.lh), qsfc=fl(d.qsfc),
                    u10=fl(d.u10), v10=fl(d.v10), th2=fl(d.th2), t2=fl(d.t2), znt=fl(d.znt))
        inputs_ok = np.all([np.isfinite(np.asarray(v, np.float64)).all(axis=-1) if np.ndim(v) > 1 else np.isfinite(np.asarray(v, np.float64)) for v in cols.values()], axis=0)
        kinds = np.array([m["kind"] for m in meta])
        r = {"n": len(meta), "dx": dx, "kinds": {k: int((kinds == k).sum()) for k in set(kinds.tolist())},
             "census": {"max_v10": float(max(m["v10"] for m in meta)), "min_th0_anom": float(min(m["th0_anom"] for m in meta)),
                        "hfx_range": [float(cols["hfx"].min()), float(cols["hfx"].max())],
                        "regimes_wrf": sorted(set(np.round(wrf["regime"]).astype(int).tolist()))}}
        for k, thr in THR.items():
            a, b = port[k], wrf[k]
            absd = np.abs(a - b)
            rel = absd / np.maximum(np.abs(b), 1e-12)
            bad = ((rel > thr) & (absd >= 1e-6)) | ~np.isfinite(a) | ~np.isfinite(b) | ~inputs_ok  # E200
            ok_all &= not bad.any()
            r[k] = {"relmax": float(rel.max()), "absmax": float(absd.max()), "n_fail": int(bad.sum()),
                    "fail_kinds": sorted(set(kinds[bad].tolist()))}
        res[os.path.basename(f)] = r
        print(os.path.basename(f), r["kinds"], {k: (round(r[k]["relmax"], 5), r[k]["n_fail"]) for k in THR}, flush=True)
    res["ALL_PASS_FROZEN"] = bool(ok_all)
    Path(dst).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
