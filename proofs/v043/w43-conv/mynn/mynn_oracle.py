"""CM01: pristine WRF 4.7.1 PUBLIC mynnedmf_driver (unmodified objects; F20 caller tests/v025/t_cfix_mynn/f20_oracle.py,
module_pbl_driver wiring, Registry defaults, initflag 0, tsq/qsq/cov 0) vs the RELEASE-default product MYNN column kernel
(f20_port.column_inputs: bottom-face w, driver flux limits; P.step_mynn_pbl_column_with_pblh, edmf) on REAL convective /
COLD-POOL columns of the #148 CPU-reference frames. Inputs caller-faithful from history (phy_prep): u,v mass averages of
U/V, w = W faces, th = T + 300 (dry), p = P_HYD (WRF caller grid%p_hyd), exner = pi_phy = ((P+PB)/1e5)^(R_d/cp), t3d = th*exner, rho = (1+qv)/alt with alt from the hypsometric_opt=2 log form (caller_ops.py; AL/ALB are not in history),
dz = diff((PH+PHB)/g), qke = QKE, el_pbl = EL_PBL; 2-D XLAND TSK PSFC UST CH HFX QFX; QSFC=qv(k=1) and ZNT=0.1 as F20 (dead inside mynnedmf); WSPD (not in
history) = max(|V(k=1)|, 0.1) -- same operand on both sides.
Kinds: coldpool (lowest-level th >= 1.5 K below the frame median of its XLAND class AND rain qr>1e-5 in k<3),
core (max qr >= 2 g/kg), downdraft (min W(k<15) <= -2 m/s), env (random rain-free).
Gate (FROZEN, b-phys mynn_wrf_gates / tier2_mynn INDEPENDENT_TOLERANCE_ABS): max |d tendency| <= 1e-3 (u v th qv qc qi).
Pre-registered report-only (no frozen bound exists): per-field normalized max error (max|d|/max|WRF|), pblh / qke / exch_h /
maxmf differences.
usage: mynn_oracle.py <frames_dir> <out.json>"""
import glob, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/v025/t_cfix_mynn")]
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from caller_ops import caller_fields
import jax
import f20_oracle as O
import f20_port as F
from gpuwrf.physics import mynn_pbl as P
PLATFORM = os.environ.get("W43_PLATFORM", "cpu")
assert jax.devices()[0].platform == PLATFORM and os.environ.get("GPUWRF_MYNN_FP32_COLUMNS") == "1", jax.devices()
if PLATFORM == "gpu":
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
G, RD, CP = np.float32(9.81), np.float32(287.0), np.float32(7.0 * 287.0 / 2.0)
NPICK = 12
rng = np.random.default_rng(20261021)


def columns(path):
    z = np.load(path)
    g = lambda v: np.asarray(z[v], np.float32)  # noqa: E731
    T, qv, qr, W = g("T"), g("QVAPOR"), g("QRAIN"), g("W")
    U, V = g("U"), g("V")
    th = T + np.float32(300.0)
    cf = caller_fields(z)                      # WRF caller: p = p_hyd, pi_phy dynamic, rho from hypso2 alt
    p, exner, rho = cf["p"], cf["exner"], cf["rho"]
    dz = np.diff((g("PH") + g("PHB")) / G, axis=0)
    xl = g("XLAND")
    th0 = th[0]
    anom = np.zeros_like(th0)
    for c in np.unique(xl):
        m = xl == c
        anom[m] = th0[m] - np.median(th0[m])
    kinds = {"coldpool": (anom <= -1.5) & (qr[:3].max(0) > 1e-5), "core": qr.max(0) >= 2e-3,
             "downdraft": W[:15].min(0) <= -2.0, "env": qr.max(0) < 1e-8}
    picks = []
    for kind, m in kinds.items():
        yx = np.argwhere(m)
        if len(yx):
            picks += [(kind, y, x) for y, x in yx[rng.choice(len(yx), min(NPICK, len(yx)), replace=False)]]
    ys, xs = np.array([p_[1] for p_ in picks]), np.array([p_[2] for p_ in picks])
    col = lambda a: np.ascontiguousarray(a[:, ys, xs].T)  # noqa: E731
    cols = {"u": col(0.5 * (U[:, :, 1:] + U[:, :, :-1])), "v": col(0.5 * (V[:, 1:] + V[:, :-1])), "w": col(W),
            "th": col(th), "t3d": col(th * exner), "p": col(p), "exner": col(exner), "rho": col(rho), "dz": col(dz),
            "qv": col(qv), "qc": col(g("QCLOUD")), "qi": col(g("QICE")), "qs": col(g("QSNOW")), "qni": col(g("QNICE")),
            "qke": col(g("QKE")), "el_pbl": col(g("EL_PBL")[:-1] if g("EL_PBL").shape[0] == T.shape[0] + 1 else g("EL_PBL"))}
    for k2, v in (("xland", "XLAND"), ("tsk", "TSK"), ("psfc", "PSFC"), ("ust", "UST"), ("ch", "CH"),
                  ("hfx", "HFX"), ("qfx", "QFX")):
        cols[k2] = np.ascontiguousarray(g(v)[ys, xs])
    cols["qsfc"] = cols["qv"][:, 0].copy()
    cols["znt"] = np.full_like(cols["tsk"], np.float32(0.1))
    cols["wspd"] = np.maximum(np.hypot(cols["u"][:, 0], cols["v"][:, 0]), np.float32(0.1)).astype(np.float32)
    meta = [dict(kind=k, y=int(y), x=int(x), th0_anom=float(anom[y, x])) for k, y, x in picks]
    return cols, meta, float(z["attr_DT"]), float(z["attr_DX"])


def port_full(cols, dt, dx):
    state, surface, _ = F.column_inputs(cols)
    out, pblh = P.step_mynn_pbl_column_with_pblh(state, float(dt), surface=surface, edmf=True, dx=float(dx))
    t = lambda a, b: (np.asarray(a, np.float64) - np.asarray(b, np.float64)) / dt  # noqa: E731
    return {"rthblten": t(out.theta, state.theta), "rqvblten": t(out.qv, state.qv), "rublten": t(out.u, state.u),
            "rvblten": t(out.v, state.v), "rqcblten": t(out.qc, state.qc), "rqiblten": t(out.qi, state.qi),
            "pblh": np.asarray(pblh), "maxmf": np.asarray(out.maxmf), "qke": 2.0 * np.asarray(out.tke, np.float64)}


def main(frames, dst):
    exe = O.build(Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv/mynn_pristine"))
    res, all_ok = {}, True
    for f in sorted(glob.glob(os.path.join(frames, "*.npz"))):
        cols, meta, dt, dx = columns(f)
        wrf, _ = O.run(exe, Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv/mynn_work") / os.path.basename(f)[:-4], cols, dt, dx)
        port = port_full(cols, dt, dx)
        inputs_ok = all(np.isfinite(np.asarray(v, np.float64)).all() for v in cols.values())
        kinds = np.array([m["kind"] for m in meta])
        r = {"n": len(meta), "dt": dt, "dx": dx, "kinds": {k: int((kinds == k).sum()) for k in set(kinds.tolist())},
             "census": {"max_qke": float(cols["qke"].max()), "min_th0_anom": float(min(m["th0_anom"] for m in meta)),
                        "max_hfx": float(cols["hfx"].max()), "min_hfx": float(cols["hfx"].min()),
                        "max_wspd": float(cols["wspd"].max())}}
        for name in ("rthblten", "rqvblten", "rublten", "rvblten", "rqcblten", "rqiblten"):
            d = np.abs(port[name] - wrf[name])
            ok = bool(np.isfinite(port[name]).all() and np.isfinite(wrf[name]).all() and inputs_ok and d.max() <= 1e-3)  # E200
            all_ok &= ok
            per_kind = {k: float(d[kinds == k].max() / max(np.abs(wrf[name][kinds == k]).max(), 1e-30)) for k in set(kinds.tolist())}
            r[name] = {"max_abs": float(d.max()), "max_wrf": float(np.abs(wrf[name]).max()), "pass_frozen_1e-3": ok,
                       "normalized_max": float(d.max() / max(np.abs(wrf[name]).max(), 1e-30)), "normalized_by_kind": per_kind}
        for name in ("pblh", "maxmf", "qke"):
            ref = wrf["qke_out" if name == "qke" else name]
            d = np.abs(port[name] - ref)
            r[name] = {"max_abs": float(d.max()), "max_wrf": float(np.abs(ref).max()), "report_only": True}
        res[os.path.basename(f)] = r
        print(os.path.basename(f), {k: (round(v["normalized_max"], 4), v["pass_frozen_1e-3"]) for k, v in r.items() if isinstance(v, dict) and "pass_frozen_1e-3" in v},
              "pblh", round(r["pblh"]["max_abs"], 1), flush=True)
    res["ALL_PASS_FROZEN"] = all_ok
    Path(dst).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
