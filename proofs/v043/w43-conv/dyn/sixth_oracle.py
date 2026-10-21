"""CD01: pristine WRF sixth_order_diffusion (diff_6th_opt=2 monotonic, factor .12, rk_tendency caller args; b-diff
tests/v025/b_diff/pristine_diffusion.py, REAL4 ctypes) vs the PRODUCT sixth-order (release defaults: legacy entry
explicit_diffusion.wrf_sixth_order_* with _NATIVE_DIFFUSION_FP32 -> kernels/dyn_diff_fp32.sixth_order_fp32, interpret Pallas
on CPU) on whole REAL convective CPU-reference frames (d06/d04 1 km, d02 3 km). Fields u, v, w, theta (history T) and moist
scalars qv qc qr qi qs qg Nr. Gate (b-core bench_acoustic.compare, frozen): rms <= 2e-5 max(1, rms_ref), max <= 2e-4 max(1,
max_ref). PRE-REGISTERED for mixing ratios (kg/kg makes the max(1,.) floor vacuous): scalars are fed in g/kg (x1e3; the
operator is degree-1 homogeneous incl. the sign-based limiter) AND scored with the relative gate rms <= 2e-5 rms_ref,
max <= 2e-4 max_ref ('rel'); Nr fed as-is (#/kg ~1e3..1e5). Census per field: max |field|, max |tendency|, limiter-active
fraction (monotonic vs non-monotonic pristine tendencies differ).
usage: sixth_oracle.py <frame.npz> [<frame.npz> ...] --out out.json"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/v025/b_diff"), str(ROOT / "tests/v025/b_core")]
from gpuwrf._fast_defaults import apply_fast_path_defaults
STATUS = apply_fast_path_defaults()
import jax, jax.numpy as jnp
import numpy as np
from gpuwrf.dynamics import explicit_diffusion as diff
from pristine_diffusion import build, run, MAPS
from bench_acoustic import compare
PLATFORM = os.environ.get("W43_PLATFORM", "cpu")
assert jax.devices()[0].platform == PLATFORM, jax.devices()
if PLATFORM == "gpu":
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
assert diff._NATIVE_DIFFUSION_FP32, "release default must select the native fp32 sixth order"
SCAL = {"QVAPOR": 1e3, "QCLOUD": 1e3, "QRAIN": 1e3, "QICE": 1e3, "QSNOW": 1e3, "QGRAUP": 1e3, "QNRAIN": 1.0}
WRFMAP = {"msftx": "MAPFAC_MX", "msfty": "MAPFAC_MY", "msfux": "MAPFAC_UX", "msfuy": "MAPFAC_UY", "msfvx": "MAPFAC_VX",
          "msfvy": "MAPFAC_VY"}


def rel(ref, got):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    e = got - ref
    rms, mx = float(np.sqrt(np.mean(e ** 2))), float(np.abs(e).max())
    rr, mr = float(np.sqrt(np.mean(ref ** 2))), float(np.abs(ref).max())
    return dict(rms=rms, max_abs=mx, rms_ref=rr, max_ref=mr, passed=bool(np.isfinite(got).all() and rms <= 2e-5 * rr and mx <= 2e-4 * mr))


def frame(path, oracle):
    z = np.load(path)
    dt = float(z["attr_DT"]) if "attr_DT" in z else {"d02": 18.0}.get(os.path.basename(path)[:3], 6.0)
    f = lambda v: np.asarray(z[v], np.float32)  # noqa: E731
    mut = f("MU") + f("MUB")
    c1h, c2h, c1f, c2f = f("C1H"), f("C2H"), f("C1F"), f("C2F")
    maps = {m: f(WRFMAP[m]) for m in MAPS}
    J = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
    out = {"dt": dt, "shape": list(f("T").shape)}
    for mono in (True, False):
        u, v, w = diff.wrf_sixth_order_uvw_tendf(J(f("U")), J(f("V")), J(f("W")), J(mut), c1h=J(c1h), c2h=J(c2h), c1f=J(c1f),
                                                  c2f=J(c2f), msfux=J(maps["msfux"]), msfuy=J(maps["msfuy"]),
                                                  msfvx=J(maps["msfvx"]), msfvy=J(maps["msfvy"]), msftx=J(maps["msftx"]),
                                                  msfty=J(maps["msfty"]), dt=dt, diff_6th_factor=.12, monotonic=mono)
        port = {"U": u, "V": v, "W": w}
        port["T"] = diff.wrf_sixth_order_scalar_tendf(J(f("T")), J(mut), c1=J(c1h), c2=J(c2h), msftx=J(maps["msftx"]),
                                                       msfty=J(maps["msfty"]), dt=dt, diff_6th_factor=.12, monotonic=mono,
                                                       specified_or_nested=True)
        for q, s in SCAL.items():
            port[q] = diff.wrf_sixth_order_scalar_tendf(J(f(q) * np.float32(s)), J(mut), c1=J(c1h), c2=J(c2h),
                                                         msftx=J(maps["msftx"]), msfty=J(maps["msfty"]), dt=dt,
                                                         diff_6th_factor=.12, monotonic=mono, specified_or_nested=True)
        for name, val in port.items():
            kind = {"U": "u", "V": "v", "W": "w"}.get(name, "m")
            c1, c2 = (c1f, c2f) if kind == "w" else (c1h, c2h)
            fld = f(name) * np.float32(SCAL.get(name, 1.0))
            ref = run(oracle, fld, mut, c1, c2, maps, name=kind, dt=dt, factor=.12, monotonic=mono)
            got = np.asarray(val, np.float64)
            r = {"abs": compare(ref, got), "rel": rel(ref, got), "max_field": float(np.abs(fld).max())}
            out[f"{name}_{'mono' if mono else 'nomono'}"] = r
            if mono:
                out.setdefault("_ref_mono", {})[name] = ref
            else:
                rm = out["_ref_mono"][name]
                out[f"{name}_limiter_active_frac"] = float((np.abs(rm - ref) > 1e-6 * max(np.abs(ref).max(), 1e-30)).mean())
    out.pop("_ref_mono")
    return out


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".npz")]
    dst = Path(sys.argv[sys.argv.index("--out") + 1])
    oracle = build(Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv/sixth_pristine"))
    res = {"env_status": {k: STATUS["applied"].get(k) for k in STATUS["applied"] if "DIFF" in k or "REAL" in k}}
    for p in paths:
        res[os.path.basename(p)] = r = frame(p, oracle)
        print(os.path.basename(p), {k: (v["abs"]["passed"], v["rel"]["passed"]) for k, v in r.items() if isinstance(v, dict)}, flush=True)
    dst.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
