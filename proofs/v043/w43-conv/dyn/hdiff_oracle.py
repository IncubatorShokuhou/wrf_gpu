"""CD02: pristine WRF horizontal_diffusion / horizontal_diffusion_3dmp (b-diff pristine_diffusion.py, REAL4, rk_tendency
caller args) vs the PRODUCT diff_opt=1 operators (release: explicit_diffusion.wrf_nested_horizontal_diffusion_momentum_tendency
and horizontal_diffusion_coord_scalar_tendency(nonperiodic_owned) with the native fp32 adapter) on REAL convective frames,
driven by the PRISTINE convective Kh of CK01 (xkmh, ring zeroed as smag2d_km owns ids+1:ide-2; scalars 3*xkmh =
xkmh/prandtl). Fields u v w, theta (history T, base 0 both sides), qv qr qg (g/kg) and Nr.
Gate: frozen bench_acoustic.compare (abs) AND the CD01 pre-registered relative form (rms<=2e-5 rms_ref, max<=2e-4 max_ref).
usage: hdiff_oracle.py <frame.npz> [...] --out out.json"""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/v025/b_diff"), str(ROOT / "tests/v025/b_core"), str(HERE)]
import kh_oracle as KO   # applies fast defaults first
import numpy as np
import jax, jax.numpy as jnp
from gpuwrf.dynamics import explicit_diffusion as diff
from pristine_diffusion import build, run, MAPS
from bench_acoustic import compare
from sixth_oracle import rel, WRFMAP
assert diff._NATIVE_DIFFUSION_FP32
SCAL = {"QVAPOR": 1e3, "QRAIN": 1e3, "QGRAUP": 1e3, "QNRAIN": 1.0}


def frame(path, lib_kh, oracle):
    z = np.load(path)
    g = lambda v: np.asarray(z[v], np.float32)  # noqa: E731
    dx = float(z["attr_DX"])
    kh = KO.pristine(lib_kh, z)["xkmh"].astype(np.float32)
    kh[:, 0, :] = kh[:, -1, :] = kh[:, :, 0] = kh[:, :, -1] = 0.0
    mut = g("MU") + g("MUB")
    maps = {m: g(WRFMAP[m]) for m in MAPS}
    c1h, c2h, c1f, c2f = g("C1H"), g("C2H"), g("C1F"), g("C2F")
    J = lambda a: jnp.asarray(a, jnp.float32)  # noqa: E731
    kw = dict(c1h=J(c1h), c2h=J(c2h), c1f=J(c1f), c2f=J(c2f), msftx=J(maps["msftx"]), msfty=J(maps["msfty"]), msfux=J(maps["msfux"]),
              msfuy=J(maps["msfuy"]), msfvx=J(maps["msfvx"]), msfvy=J(maps["msfvy"]), dx_m=dx, dy_m=dx)
    u, v, w = diff.wrf_nested_horizontal_diffusion_momentum_tendency(J(g("U")), J(g("V")), J(g("W")), J(kh), J(mut), **kw)
    mh = J(c1h)[:, None, None] * J(mut)[None] + J(c2h)[:, None, None]
    port = {"U": u, "V": v, "W": w}
    for name, s in (("T", 1.0), *SCAL.items()):
        port[name] = diff.horizontal_diffusion_coord_scalar_tendency(J(g(name) * np.float32(s)), J(np.float32(3) * kh), mh, dx_m=dx,
                                                                      dy_m=dx, base_3d=J(np.zeros_like(g("T"))), msftx=kw["msftx"],
                                                                      msfty=kw["msfty"], msfux=kw["msfux"], msfuy=kw["msfuy"],
                                                                      msfvx=kw["msfvx"], msfvy=kw["msfvy"], nonperiodic_owned=True)
    out = {"dx": dx, "max_kh": float(kh.max())}
    for name, val in port.items():
        kind = {"U": "u", "V": "v", "W": "w"}.get(name, "m")
        c1, c2 = (c1f, c2f) if kind == "w" else (c1h, c2h)
        fld = g(name) * np.float32(SCAL.get(name, 1.0))
        ref = run(oracle, fld, mut, c1, c2, maps, name=kind, dt=1. / dx, factor=1. / dx, monotonic=True,
                  kh=kh * (np.float32(3) if kind == "m" else np.float32(1)), base=np.zeros_like(fld) if kind == "m" else None)
        got = np.asarray(val, np.float64)
        out[name] = {"abs": compare(ref, got), "rel": rel(ref, got)}
    return out


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".npz")]
    dst = Path(sys.argv[sys.argv.index("--out") + 1])
    lib_kh = KO.build()
    oracle = build(Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv/hdiff_pristine"))
    res = {}
    for p in paths:
        res[os.path.basename(p)] = r = frame(p, lib_kh, oracle)
        print(os.path.basename(p), {k: (v["abs"]["passed"], v["rel"]["passed"], "%.1e" % (v["rel"]["max_abs"] / max(v["rel"]["max_ref"], 1e-30)))
                                    for k, v in r.items() if isinstance(v, dict)}, flush=True)
    res["ALL_PASS"] = all(v["abs"]["passed"] and v["rel"]["passed"] for f, r in res.items() for v in r.values() if isinstance(v, dict))
    dst.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
