"""CT02: one pristine mp_gt_driver step (pristine WRF 4.7.1 module_mp_thompson mp8, TH08/LL01 column driver rebuilt in
<USER_HOME>/wrf_gpu2_lanes/w43-conv/thompson_drv; tables computed by thompson_init in that cwd) vs the product native
REAL Thompson full column (RELEASE fast-path defaults incl. MIXED_PHASE_WRF + WRF_CONSTANTS, interpret Pallas on CPU)
on the SAME real convective inputs. Gate = tests/v025/b_thompson/test_mixed_phase_wrf_oracle.py increment rule.
usage: run_oracle.py <columns.npz> <out.npz> [--mutant NAME]"""
import os, subprocess, sys
from pathlib import Path
import numpy as np
ORDER_IN = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz8w")
ORDER_OUT = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th")
DRV = Path("<USER_HOME>/wrf_gpu2_lanes/w43-conv/thompson_drv")
EXE = DRV / "thompson_column_driver.exe"


def run_wrf(z, idx, dt, tag):
    ncol, nlev = len(idx), z["in_qv"].shape[1]
    fin, fout = DRV / f"in_{tag}.bin", DRV / f"out_{tag}.bin"
    with open(fin, "wb") as f:
        f.write(np.array([ncol, nlev], ">i4").tobytes()); f.write(np.array([dt], ">f4").tobytes())
        for v in ORDER_IN:
            f.write(np.asarray(z[f"in_{v}"][idx].T, ">f4").tobytes(order="C"))
    subprocess.run([str(EXE), str(fin), str(fout)], cwd=DRV, check=True, capture_output=True)
    raw = np.fromfile(fout, ">f4"); out, o = {}, 0
    for v in ORDER_OUT:
        out[v] = raw[o:o + ncol * nlev].reshape(nlev, ncol).T.astype(np.float32); o += ncol * nlev
    for v in ("rainncv", "snowncv", "graupelncv", "sr"):
        out[v] = raw[o:o + ncol].astype(np.float32); o += ncol
    fin.unlink(); fout.unlink()
    return out


def run_port(z, idx, dt, interpret=True, platform="cpu"):
    import jax, jax.numpy as jnp
    assert jax.devices()[0].platform == platform, jax.devices()
    from gpuwrf.physics import thompson_column as tc
    from gpuwrf.kernels import phys_thompson_full as full
    jax.clear_caches()
    f = lambda v: jnp.asarray(z[f"in_{v}"][idx], jnp.float32)  # noqa: E731
    T, p, qv = f("th") * f("pii"), f("p"), f("qv")
    zero = jnp.zeros_like(qv)
    s = tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"), Ni=f("ni"), Nr=f("nr"),
                               Ns=zero, Ng=zero, T=T, p=p, rho=tc.density_from_pressure_temperature(p, T, qv),
                               dz=f("dz8w"), w=f("w"))
    out, ppt = full.full_column(s, float(dt), interpret=interpret)
    g = lambda a: np.asarray(a, np.float32)  # noqa: E731
    pii = np.asarray(z["in_pii"][idx], np.float32)
    return {"qv": g(out.qv), "qc": g(out.qc), "qr": g(out.qr), "qi": g(out.qi), "qs": g(out.qs), "qg": g(out.qg),
            "ni": g(out.Ni), "nr": g(out.Nr), "th": g(out.T) / pii, "rainncv": g(ppt["rain"]), "snowncv": g(ppt["snow"]),
            "graupelncv": g(ppt["graupel"]), "iceppt": g(ppt["ice"])}


def main():
    from gpuwrf._fast_defaults import apply_fast_path_defaults
    st = apply_fast_path_defaults()
    z = dict(np.load(sys.argv[1]))
    res = {"env_applied": np.array(sorted(f"{k}={v}" for k, v in st["applied"].items() if "THOMPSON" in k))}
    for dt in sorted(set(z["dt"].tolist())):
        idx = np.flatnonzero(z["dt"] == dt)
        w = run_wrf(z, idx, dt, f"dt{int(dt)}")
        p = run_port(z, idx, dt)
        for k, v in w.items():
            res.setdefault(f"wrf_{k}", np.zeros((len(z["dt"]),) + v.shape[1:], np.float32))[idx] = v
        for k, v in p.items():
            res.setdefault(f"port_{k}", np.zeros((len(z["dt"]),) + v.shape[1:], np.float32))[idx] = v
        print("dt", dt, len(idx), flush=True)
    np.savez_compressed(sys.argv[2], **res)


if __name__ == "__main__":
    main()
