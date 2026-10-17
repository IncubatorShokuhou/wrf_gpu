"""t-wsmdz W01: real-column WSM5/WSM6 kernel and caller-input A/B vs pristine WRF.

Inputs are reconstructed from ORIGINAL CPU-WRF history (c05 WSM5 / c06 WSM6)
exactly as WRF's post-RK ``moist_physics_prep_em`` forms them
(module_big_step_utilities_em.F:5480/5488 z_at_w/dz8w with REAL g=9.81,
:5509 th_phy, :5542 rho=1/(al+alb) with al+alb from calc_p_rho_phi
hypsometric_opt=2, :5543 pii, :5545 p). History states are realistic operands,
not the exact pre-MP state of a WRF step (no trajectory claim).

Arms (all REAL32 operands, one MP call, dt=18 s):
  P  pristine WRF wsm5/wsm6 (production flags) on WRF inputs
  J  JAX port kernel (CPU, fp32) on the SAME WRF inputs        -> kernel A/B
  Q  pristine WRF on the PORT's current caller inputs (dz with g=9.80665 in f64,
     rho = Thompson-formula 0.622p/(287.04 T (qv+0.622)))       -> caller sensitivity
Usage: python w01_columns.py <case_dir> <scheme 4|6> <out.json> <hour> [<hour>...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

assert os.environ.get("JAX_PLATFORMS") == "cpu", "CPU-only evidence (E50)"
import netCDF4 as nc  # noqa: E402

DRIVER = Path(os.environ.get("WSM_COLS", "<USER_HOME>/wrf_gpu2_lanes/t-wsmdz/build/wsm_cols"))
WORK = Path(os.environ.get("WSM_WORK", "<USER_HOME>/wrf_gpu2_lanes/t-wsmdz/w01"))
F4 = np.float32


def wrf_inputs(path: Path):
    d = nc.Dataset(path)
    v = lambda n: np.asarray(d.variables[n][0], dtype=F4)  # noqa: E731
    thm = v("THM") + F4(300.0)
    qv = v("QVAPOR")
    rvovrd = F4(F4(461.6) / F4(287.0))
    th = thm / (F4(1.0) + rvovrd * qv)                       # :5509
    p_phy = v("P") + v("PB")                                  # :5545
    rcp = F4(F4(287.0) / F4(7.0 * 287.0 / 2.0))
    pii = (p_phy / F4(1.0e5)) ** rcp                          # :5543
    ph, phb = v("PH"), v("PHB")
    z_at_w = (ph + phb) / F4(9.81)                            # :5480
    dz = z_at_w[1:] - z_at_w[:-1]                             # :5488
    mut = v("MU") + v("MUB")
    c3f, c4f = np.asarray(d.variables["C3F"][0], F4), np.asarray(d.variables["C4F"][0], F4)
    c3h, c4h = np.asarray(d.variables["C3H"][0], F4), np.asarray(d.variables["C4H"][0], F4)
    ptop = F4(np.asarray(d.variables["P_TOP"][0]))
    pfu = c3f[1:, None, None] * mut[None] + c4f[1:, None, None] + ptop
    pfd = c3f[:-1, None, None] * mut[None] + c4f[:-1, None, None] + ptop
    phm = c3h[:, None, None] * mut[None] + c4h[:, None, None] + ptop
    alt = (ph[1:] - ph[:-1] + phb[1:] - phb[:-1]) / phm / np.log(pfd / pfu)  # calc_p_rho_phi hypso 2
    rho = F4(1.0) / alt                                       # :5542
    q = {n: v(k) for n, k in (("qc", "QCLOUD"), ("qr", "QRAIN"), ("qi", "QICE"), ("qs", "QSNOW"))}
    q["qg"] = v("QGRAUP") if "QGRAUP" in d.variables else np.zeros_like(qv)
    # port caller inputs (scan_adapters wsm5/wsm6_adapter on current main)
    zf64 = (ph.astype(np.float64) + phb.astype(np.float64)) / 9.80665
    dz_port = np.maximum(zf64[1:] - zf64[:-1], 1.0).astype(F4)
    t_dry = th * ((np.maximum(p_phy, F4(1.0)) / F4(1.0e5)) ** rcp)
    rho_port = (F4(0.622) * p_phy / (F4(287.04) * t_dry * (qv + F4(0.622)))).astype(F4)
    d.close()
    cols = lambda a: np.ascontiguousarray(np.moveaxis(a, 0, -1).reshape(-1, a.shape[0]))  # noqa: E731
    base = dict(th=th, qv=qv, pii=pii, p=p_phy, **q)
    return ({k: cols(a) for k, a in base.items()},
            dict(den=cols(rho), delz=cols(dz)), dict(den=cols(rho_port), delz=cols(dz_port)))


NAMES = ("th", "qv", "qc", "qr", "qi", "qs", "qg")


def run_pristine(scheme, x, geo, dt, tag):
    WORK.mkdir(parents=True, exist_ok=True)
    ncol, nk = x["th"].shape
    fin, fout = WORK / f"{tag}.in", WORK / f"{tag}.out"
    with open(fin, "wb") as f:
        np.asarray([scheme, ncol, nk], np.int32).tofile(f)
        np.asarray([dt], F4).tofile(f)
        for n in NAMES:
            np.asfortranarray(x[n], F4).T.tofile(f)  # Fortran (ncol,nk) column-major
        for n in ("den", "pii", "p", "delz"):
            a = geo[n] if n in geo else x[n]
            np.asarray(a, F4).T.tofile(f)
    subprocess.run([str(DRIVER), str(fin), str(fout)], check=True)
    raw = np.fromfile(fout, F4)
    out, o = {}, 0
    for n in NAMES:
        out[n] = raw[o:o + ncol * nk].reshape(nk, ncol).T; o += ncol * nk
    for n in ("rainncv", "snowncv", "graupelncv", "sr"):
        out[n] = raw[o:o + ncol]; o += ncol
    fin.unlink(); fout.unlink()
    return out


def run_port(scheme, x, geo, dt):
    import jax
    import jax.numpy as jnp
    a = lambda n: jnp.asarray(geo[n] if n in geo else x[n], jnp.float32)  # noqa: E731
    if scheme == 4:
        from gpuwrf.physics.microphysics_wsm5 import wsm5_physics_tendency as f
        tend = f(a("th"), a("qv"), a("qc"), a("qr"), a("qi"), a("qs"), a("pii"), a("den"), a("p"), a("delz"), dt)
    else:
        from gpuwrf.physics.microphysics_wsm6 import wsm6_physics_tendency as f
        tend = f(a("th"), a("qv"), a("qc"), a("qr"), a("qi"), a("qs"), a("qg"), a("pii"), a("den"), a("p"), a("delz"), dt)
    jax.block_until_ready(tend.state_replacements)
    out = {("th" if k == "theta" else k): np.asarray(vv) for k, vv in tend.state_replacements.items()}
    acc = tend.accumulator_increments
    out["rainncv"] = np.asarray(acc["rain_acc"])
    out["snowncv"] = np.asarray(acc["snow_acc"])
    out["graupelncv"] = np.asarray(acc.get("graupel_acc", np.zeros_like(acc["rain_acc"])))
    out["dtypes"] = {k: str(np.asarray(vv).dtype) for k, vv in tend.state_replacements.items()}
    return out


def compare(a, b, x):
    res = {}
    for n in NAMES + ("rainncv", "snowncv", "graupelncv"):
        if n not in a or n not in b:
            continue
        u, w = np.asarray(a[n], np.float64), np.asarray(b[n], np.float64)
        if not (np.isfinite(u).all() and np.isfinite(w).all()):
            res[n] = {"nonfinite": True}; continue
        d = w - u
        r = {"max_abs": float(np.abs(d).max()), "rms": float(np.sqrt((d ** 2).mean())),
             "ref_max": float(np.abs(u).max()), "mean_diff": float(d.mean())}
        if n in NAMES:  # tendency-relative: compare increments, not totals
            inc = u - np.asarray(x[n], np.float64)
            r["ref_increment_rms"] = float(np.sqrt((inc ** 2).mean()))
            r["rel_to_increment"] = r["rms"] / max(r["ref_increment_rms"], 1e-30)
        else:
            r["ref_sum"] = float(u.sum()); r["test_sum"] = float(w.sum())
            r["rel_sum"] = (r["test_sum"] - r["ref_sum"]) / max(abs(r["ref_sum"]), 1e-30)
            r["n_ref_pos"] = int((u > 0).sum())
            r["n_cols_rel_gt_1e-3"] = int((np.abs(d) > 1e-3 * np.maximum(np.abs(u), 1e-6)).sum())
        res[n] = r
    return res


def main():
    case, scheme, out_json = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    hours = [int(h) for h in sys.argv[4:]]
    dt = 18.0
    report = {"schema": "t-wsmdz-w01-v1", "case": str(case), "scheme": scheme, "dt": dt,
              "driver": str(DRIVER), "hours": {}}
    for h in hours:
        f = sorted((case / "cpu").glob("wrfout_d01_*"))[h]
        x, geo_wrf, geo_port = wrf_inputs(f)
        t0 = time.time()
        P = run_pristine(scheme, x, geo_wrf, dt, f"h{h}_P")
        Q = run_pristine(scheme, x, geo_port, dt, f"h{h}_Q")
        J = run_port(scheme, x, geo_wrf, dt)
        report["hours"][str(h)] = {
            "file": f.name, "ncol": int(x["th"].shape[0]), "nk": int(x["th"].shape[1]),
            "dz_rel_port_vs_wrf": float(np.median(geo_port["delz"] / geo_wrf["delz"])),
            "rho_rel_port_vs_wrf": {"median": float(np.median(geo_port["den"] / geo_wrf["den"])),
                                    "min": float((geo_port["den"] / geo_wrf["den"]).min()),
                                    "max": float((geo_port["den"] / geo_wrf["den"]).max())},
            "kernel_J_vs_P": compare(P, J, x),
            "caller_Q_vs_P": compare(P, Q, x),
            "port_dtypes": J["dtypes"], "elapsed_s": time.time() - t0,
        }
        print(h, json.dumps({k: report["hours"][str(h)][k] for k in ("dz_rel_port_vs_wrf", "rho_rel_port_vs_wrf")}))
        for arm in ("kernel_J_vs_P", "caller_Q_vs_P"):
            r = report["hours"][str(h)][arm]
            print(" ", arm, {n: (round(r[n].get("rel_sum", r[n].get("rel_to_increment", 0)), 6), r[n]["max_abs"])
                             for n in r if "max_abs" in r[n]})
    out_json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
