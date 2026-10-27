"""o5-n333 cloudpdf oracle: port mym_condensation vs PRISTINE WRF 4.7.1 mym_condensation.

CPU only (JAX_PLATFORMS=cpu).  Inputs are real CPU-WRF restart columns (WRF state incl. the
previous-step MYNN TSQ/QSQ/COV/SH3D/EL_PBL the driver copies in, module_bl_mynnedmf_driver.F90
:420-428), wired like mynn_bl_driver :826-833 (sqx = qx/(1+qv), sqw = sqv+sqc+sqi, thl1).

usage: python cloudpdf_oracle.py ORACLE_EXE OUT_JSON LABEL RESTART [RESTART ...]
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

assert os.environ.get("JAX_PLATFORMS") == "cpu", "run with JAX_PLATFORMS=cpu"

F4 = np.float32
R_D, R_V = F4(287.0), F4(461.6)
CP = F4(7.0) * R_D / F4(2.0)
XLVCP = F4(2.5e6) / CP
XLSCP = (F4(2.5e6) + F4(3.50e5)) / CP   # common :78 xlscp=(xlv+xlf)/cp, xlf = 3.50E5
RCP = R_D / CP


def load_restart_columns(paths):
    import netCDF4

    cols = {k: [] for k in ("dz", "zw", "p", "exner", "th", "thl", "qw", "qv", "qc", "qi", "qs",
                            "tsq", "qsq", "cov", "sh", "el", "pblh", "hfx", "xland", "dx")}
    meta = []
    for path in paths:
        d = netCDF4.Dataset(path)
        g = lambda n: np.asarray(d.variables[n][0], dtype=np.float32)  # noqa: E731
        qv = np.maximum(g("QVAPOR"), F4(0.0))
        t = g("T_2" if "T_2" in d.variables else "T") + F4(300.0)
        if int(d.getncattr("USE_THETA_M")) == 1:
            t = t / (F4(1.0) + (R_V / R_D) * qv)
        p = g("P") + g("PB")
        ph = (g("PH_2" if "PH_2" in d.variables else "PH") + g("PHB")) / F4(9.81)
        dz = ph[1:] - ph[:-1]
        ex = (p / F4(1.0e5)) ** RCP
        qc, qi, qs = (np.maximum(g(n), F4(0.0)) for n in ("QCLOUD", "QICE", "QSNOW"))
        sqv, sqc, sqi, sqs = (x / (F4(1.0) + qv) for x in (qv, qc, qi, qs))
        thl = t - XLVCP / ex * sqc - XLSCP / ex * sqi
        fields = dict(dz=dz, p=p, exner=ex, th=t, thl=thl, qw=sqv + sqc + sqi, qv=sqv, qc=sqc,
                      qi=sqi, qs=sqs, tsq=g("TSQ"), qsq=g("QSQ"), cov=g("COV"), sh=g("SH3D"),
                      el=g("EL_PBL")[: dz.shape[0]])  # Z-staggered; driver copies kts:kte (B64)
        nz, ny, nx = dz.shape
        for name, arr in fields.items():
            cols[name].append(arr.reshape(nz, ny * nx).T)
        zw = np.concatenate((np.zeros((1, ny * nx), F4), np.cumsum(dz.reshape(nz, -1), axis=0, dtype=F4)))
        cols["zw"].append(zw.T)
        for name, var in (("pblh", "PBLH"), ("hfx", "HFX"), ("xland", "XLAND")):
            cols[name].append(g(var).reshape(-1))
        cols["dx"].append(np.full(ny * nx, F4(d.getncattr("DX"))))
        meta.append({"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                     "times": str(netCDF4.chartostring(d.variables["Times"][0])), "nx": nx, "ny": ny,
                     "nz": nz, "use_theta_m": int(d.getncattr("USE_THETA_M")),
                     "bl_mynn_cloudpdf": int(d.getncattr("BL_MYNN_CLOUDPDF")) if "BL_MYNN_CLOUDPDF" in d.ncattrs() else None})
        d.close()
    return {k: np.ascontiguousarray(np.concatenate(v, axis=0), dtype=F4) for k, v in cols.items()}, meta


def numpy_literal_sigma(c, cloudpdf):
    """Independent REAL4 literal sigma (NumPy IEEE ops, no FMA), CASE(0) DOUBLE island in float64.

    Follows module_bl_mynnedmf.F90 :3480-3509 / :3544-3566 operation by operation, incl. the
    esat_blend Horner order and Fortran left-to-right association.
    """
    f4 = np.float32
    J = [f4(v) for v in (0.611583699e03, 0.444606896e02, 0.143177157e01, 0.264224321e-1, 0.299291081e-3,
                          0.203154182e-5, 0.702620698e-8, 0.379534310e-11, -0.321582393e-13)]
    K = [f4(v) for v in (0.609868993e03, 0.499320233e02, 0.184672631e01, 0.402737184e-1, 0.565392987e-3,
                          0.521693933e-5, 0.307839583e-7, 0.105785160e-9, 0.161444444e-12)]
    t0c, tice, ep2 = f4(273.15), f4(240.0), R_D / R_V
    ep3, xlv = f4(1.0) - ep2, f4(2.5e6)
    nz = c["dz"].shape[1]
    th, ex, p = c["th"][:, :-1], c["exner"][:, :-1], c["p"][:, :-1]
    t = th * ex
    xc = np.maximum(f4(-80.0), t - t0c)

    def horner(co):
        return co[0] + xc * (co[1] + xc * (co[2] + xc * (co[3] + xc * (co[4] + xc * (co[5] + xc * (co[6] + xc * (co[7] + xc * co[8])))))))

    esl, esi = horner(J), horner(K)
    chi = ((t0c - f4(6.0)) - t) / ((t0c - f4(6.0)) - tice)
    esat = np.where(t >= t0c - f4(6.0), esl, np.where(t <= tice, esi, (f4(1.0) - chi) * esl + chi * esi))
    qsl = ep2 * esat / np.maximum(f4(1.0e-4), p - ep3 * esat)
    dqsl = qsl * ep2 * xlv / (R_D * t ** 2)
    alp = f4(1.0) / (f4(1.0) + dqsl * XLVCP)
    bet = dqsl * ex
    if cloudpdf == 0:
        t3 = np.maximum(c["tsq"][:, :-1].astype(np.float64), 0.0)
        r3 = np.maximum(c["qsq"][:, :-1].astype(np.float64), 0.0)
        c3 = c["cov"][:, :-1].astype(np.float64)
        c3 = np.copysign(np.minimum(np.abs(c3), np.sqrt(t3 * r3)), c3)
        r3 = r3 + (bet ** 2).astype(np.float64) * t3 - (f4(2.0) * bet).astype(np.float64) * c3
        return np.sqrt(np.maximum(r3, 1.0e-10)).astype(f4)
    thl, qw, dz, el, sh = (c[n] for n in ("thl", "qw", "dz", "el", "sh"))
    km1 = np.concatenate(([0], np.arange(nz - 2)))
    dzk = np.where(np.arange(nz - 1) == 0, f4(0.5) * dz[:, :-1], dz[:, :-1])
    dth = f4(0.5) * (thl[:, 1:] + thl[:, :-1]) - f4(0.5) * (thl[:, :-1] + thl[:, km1])
    dqw = f4(0.5) * (qw[:, 1:] + qw[:, :-1]) - f4(0.5) * (qw[:, :-1] + qw[:, km1])
    var = (alp ** 2 * np.maximum(el[:, :-1] ** 2, f4(0.1)) * f4(15.0) * np.maximum(sh[:, :-1], f4(0.03))) / f4(4.0) \
        * (dqw / dzk - bet * (dth / dzk)) ** 2
    return np.sqrt(np.maximum(var, f4(1.0e-10)))


def run_oracle(exe, c, cloudpdf):
    ncol, nz = c["dz"].shape
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR_ORACLE")) as tmp:
        inp, out = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        record = np.concatenate(
            [np.stack([c["dx"], c["xland"], c["pblh"], c["hfx"]], axis=1)]
            + [c[name] for name in ("dz", "zw", "p", "exner", "thl", "qw", "qv", "qc", "qi", "qs",
                                    "tsq", "qsq", "cov", "sh", "el", "th")], axis=1).astype(F4)
        with open(inp, "wb") as f:
            np.asarray([ncol, nz, cloudpdf], np.int32).tofile(f)
            record.tofile(f)
        subprocess.run([exe, str(inp), str(out)], check=True)
        raw = np.fromfile(out, dtype=F4).reshape(ncol, 6, nz)
    return {"qc_bl": raw[:, 0], "qi_bl": raw[:, 1], "cldfra_bl": raw[:, 2], "sgm": raw[:, 3]}


def run_port(c, cloudpdf, real32):
    import jax
    import jax.numpy as jnp
    from gpuwrf.physics import mynn_sgs_cloud as S

    dtype = jnp.float32 if real32 else jnp.float64
    with jax.enable_x64(not real32):
        a = {k: jnp.asarray(v, dtype) for k, v in c.items()}
        extra = {}
        if abs(cloudpdf) == 2:
            qc, qi, cf = S.mym_condensation_cloudpdf2(
                theta=a["th"], p=a["p"], exner=a["exner"], dz=a["dz"], qw=a["qw"], qc=a["qc"],
                qi=a["qi"], qs=a["qs"], qsq=a["qsq"], pblh=a["pblh"])
            if cloudpdf < 0:
                qc, qi, cf = (x.at[..., :-1].set(0.0) for x in (qc, qi, cf))
        else:
            kw = dict(theta=a["th"], p=a["p"], exner=a["exner"], dz=a["dz"], thl=a["thl"],
                      qw=a["qw"], tsq=a["tsq"], qsq=a["qsq"], cov=a["cov"], el=a["el"], sh=a["sh"])
            qc, qi, cf = S.mym_condensation_cloudpdf01(cloudpdf, **kw)
            _qc, _qi, _cf, sgm = S._cloudpdf01_columns(abs(cloudpdf), **kw)
            t = a["th"] * a["exner"]
            qsl, alp, _bet = S._alp_bet(t, a["p"], a["exner"])
            extra = {"sgm": np.asarray(sgm, np.float64), "alp": np.asarray(alp[..., :-1], np.float64),
                     "q1": np.asarray((a["qw"][..., :-1] - qsl[..., :-1]) / sgm, np.float64)}
        return {"qc_bl": np.asarray(qc, np.float64), "qi_bl": np.asarray(qi, np.float64),
                "cldfra_bl": np.asarray(cf, np.float64), **extra}


# Pre-registered gate (2026-10-24, before the d02/d03/day sets were scored).
# WRF's qll = cld*q1 + eq1 cancels for q1 << 0, so a 1-ulp erf/exp difference between glibc
# (gfortran REAL4) and XLA is amplified.  sigma is formed BEFORE erf and is a WRF output argument:
# it must agree to <= 2 REAL ulps.  Outputs: |dcld| <= ERF_ULP_TOL + 1e-6 cld,
# |dq| <= alp*sgm*(|q1|+1)*ERF_ULP_TOL + 1e-5 |q|.  Cells whose oracle value is a REAL subnormal
# (< 2**-126) and port value 0 are reported separately (XLA:CPU flushes subnormals, E117).
ERF_ULP_TOL = 2.4e-7
SGM_REL_TOL = 2.4e-7  # 2 REAL ulps; a 1e-6 relaxation was tried and withdrawn: the misses were an r_d*t*t association bug (fixed)
TINY = float(np.finfo(np.float32).tiny)


def compare(port, ref, cloudpdf):
    out = {}
    if "sgm" in port:
        rs, ps = ref["sgm"][:, :-1].astype(np.float64), port["sgm"]
        rel = np.abs(ps - rs) / np.maximum(np.abs(rs), 1e-30)
        out["sgm"] = {"max_rel": float(rel.max()), "exact_cells": int((ps == rs).sum()),
                      "cells": int(rs.size), "violations": int((rel > SGM_REL_TOL).sum())}
    for name in ("qc_bl", "qi_bl", "cldfra_bl"):
        x, r = port[name], ref[name].astype(np.float64)
        finite = bool(np.isfinite(x).all() and np.isfinite(r).all())
        diff = np.abs(x - r)
        subnormal = (np.abs(r) < TINY) & (r != 0) & (x == 0)
        if cloudpdf in (0, 1):
            if name == "cldfra_bl":
                tol = ERF_ULP_TOL + 1e-6 * np.abs(r)
            else:
                body = port["alp"] * port["sgm"] * (np.abs(port["q1"]) + 1.0) * ERF_ULP_TOL
                tol = np.concatenate((body, np.zeros_like(body[:, :1])), axis=1) + 1e-5 * np.abs(r)
        else:
            tol = np.zeros_like(r)  # CASE 2 / negative options: bitwise
        viol = (diff > tol) & ~subnormal
        out[name] = {"finite": finite, "ref_nonzero_cells": int((r != 0).sum()), "ref_max": float(np.max(r)),
                     "max_abs": float(diff.max()), "exact_cells": int((diff == 0).sum()),
                     "cells": int(diff.size), "subnormal_ftz_cells": int(subnormal.sum()),
                     "violations": int(viol.sum()),
                     "max_diff_over_tol": float(np.max(np.where(tol > 0, diff / np.where(tol > 0, tol, 1), np.where(diff > 0, np.inf, 0.0))))}
    out["PASS"] = bool(all(v.get("finite", True) for v in out.values() if isinstance(v, dict))
                       and all(v["violations"] == 0 for v in out.values() if isinstance(v, dict)))
    return out


def main():
    exe, out_json, label, *paths = sys.argv[1:]
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    c, meta = load_restart_columns(paths)
    ncol, nz = c["dz"].shape
    sample = {"columns": ncol, "nz": nz,
              "cloudy_cols_resolved_qc>1e-6": int((c["qc"] > 1e-6).any(axis=1).sum()),
              "icy_cols_qi>1e-8": int((c["qi"] > 1e-8).any(axis=1).sum()),
              "tsq_max": float(c["tsq"].max()), "qsq_max": float(c["qsq"].max()),
              "cov_minmax": [float(c["cov"].min()), float(c["cov"].max())],
              "sh_max": float(c["sh"].max()), "el_max": float(c["el"].max()),
              "t_min": float((c["th"] * c["exner"]).min())}
    result = {"label": label, "inputs": meta, "sample": sample,
              "oracle_exe_sha256": hashlib.sha256(Path(exe).read_bytes()).hexdigest(), "cases": {}}
    for cloudpdf in (0, 1, -1, 2, -2):
        ref = run_oracle(exe, c, cloudpdf)
        case = {"oracle_nonzero_cldfra_cells": int((ref["cldfra_bl"] > 0).sum()),
                **({"numpy_literal_sigma_exact_cells": int((numpy_literal_sigma(c, abs(cloudpdf)) == ref["sgm"][:, :-1]).sum()),
                    "sigma_cells": int(ref["sgm"][:, :-1].size)} if abs(cloudpdf) in (0, 1) else {}),
                "oracle_cldfra_ge_0.5_cells": int((ref["cldfra_bl"] >= 0.5).sum()),
                "oracle_qi_bl_nonzero_cells": int((ref["qi_bl"] > 0).sum())}
        for arm, real32 in (("native_real32", True), ("legacy_f64", False)):
            case[arm] = compare(run_port(c, cloudpdf, real32), ref, cloudpdf)
        result["cases"][str(cloudpdf)] = case
        print(label, cloudpdf, json.dumps({a: {f: (v if not isinstance(v, dict) else
                                                   (v.get("violations"), v.get("max_rel", v.get("max_abs"))))
                                               for f, v in case[a].items()} for a in ("native_real32", "legacy_f64")}))
    Path(out_json).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
