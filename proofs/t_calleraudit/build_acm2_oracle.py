"""Pristine WRF 4.7.1 ACMPBL public-call oracle on REAL cloudy columns (lane t-calleraudit, F6).

No JAX. Builds the UNMODIFIED phys/module_bl_acm.F (lane-dir copy, E104) + acm2_oracle_wrapper.f90:
  zero : -O0 -ffp-contract=off -finit-real=zero  (reference)
  inf  : -O0 -ffp-contract=off -finit-real=inf   (uninitialised-read sensitivity)
  fma  : -O2 -mfma -ffp-contract=fast -finit-real=zero (WRF's own optimiser/FMA spread)
Columns: original CPU-WRF c05 (mp4 WSM5 / bl7 ACM2 / sf1, Swiss 2023-01-15) at 06/12/15/18 UTC,
every frame: 6 columns with the most cloud water+ice in the lowest 15 levels + 2 random.
Writes <lane>/acm2_oracle/oracle.npz + manifest.json.
usage: python build_acm2_oracle.py <lane_dir>
"""
from pathlib import Path
import ctypes, hashlib, json, subprocess, sys
import numpy as np
import netCDF4

HERE = Path(__file__).resolve().parent
LANE = Path(sys.argv[1])
BUILD = LANE / "acm2_oracle"
PRISTINE = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_acm.F")
WRAPPER = HERE / "acm2_oracle_wrapper.f90"
FC = "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
C05 = Path("<USER_HOME>/wrf_gpu2_lanes/v-prep/c05/cpu")
FRAMES = ("2023-01-15_06:00:00", "2023-01-15_12:00:00", "2023-01-15_15:00:00", "2023-01-15_18:00:00")
VARIANTS = {"zero": ["-O0", "-ffp-contract=off", "-finit-real=zero"],
            "inf": ["-O0", "-ffp-contract=off", "-finit-real=inf"],
            "fma": ["-O2", "-mfma", "-ffp-contract=fast", "-finit-real=zero"]}
F = np.float32
R_D, R_V, G = F(287.), F(461.6), F(9.81)
CP = F(7.) * R_D / F(2.)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def build(name, flags):
    out = BUILD / name
    out.mkdir(parents=True, exist_ok=True)
    so = out / "oracle.so"
    subprocess.run([FC, "-shared", "-fPIC", *flags, "-cpp", "-ffree-form", "-ffree-line-length-none", str(BUILD / "module_bl_acm.F"),
                    str(WRAPPER), "-o", str(so)], cwd=out, check=True, timeout=300)
    fn = ctypes.CDLL(str(so)).probe
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags="F_CONTIGUOUS")
    fn.argtypes = [ctypes.c_int, arr, arr, arr, arr]
    return fn, dict(flags=flags, sha256=sha(so))


def columns():
    rng = np.random.default_rng(20261015)
    for frame in FRAMES:
        with netCDF4.Dataset(C05 / f"wrfout_d01_{frame}") as d:
            v = {k: np.asarray(d.variables[k][0], np.float64) for k in (
                "U", "V", "T", "P", "PB", "PH", "PHB", "QVAPOR", "QCLOUD", "QICE", "PSFC", "UST", "HFX",
                "QFX", "TSK", "PBLH", "MU", "MUB", "U10", "V10")}
        nz, ny, nx = v["T"].shape
        cloud = (v["QCLOUD"][:15] + v["QICE"][:15]).max(axis=0)
        order = np.argsort(cloud.ravel())[::-1]
        picks = [divmod(int(i), nx) for i in order[:6]] + [tuple(rng.integers([1, 1], [ny - 1, nx - 1])) for _ in range(2)]
        for j, i in picks:
            pp = v["P"][:, j, i] + v["PB"][:, j, i]
            zw = (v["PH"][:, j, i] + v["PHB"][:, j, i]) / 9.81
            qv = np.maximum(v["QVAPOR"][:, j, i], 0.)
            th = v["T"][:, j, i] + 300.
            t = th * (pp / 1.e5) ** (287. / 1004.5)
            rho = pp / (287. * t * (1. + 461.6 / 287. * qv)) * (1. + qv)
            prof = np.stack([.5 * (v["U"][:, j, i] + v["U"][:, j, i + 1]), .5 * (v["V"][:, j, i] + v["V"][:, j + 1, i]),
                             pp, np.diff(zw), th, t, qv, np.maximum(v["QCLOUD"][:, j, i], 0.),
                             np.maximum(v["QICE"][:, j, i], 0.), rho], axis=1)
            wspd = max(float(np.hypot(v["U10"][j, i], v["V10"][j, i])), .1)
            sfc = np.array([max(v["UST"][j, i], .01), v["HFX"][j, i], v["QFX"][j, i], v["TSK"][j, i], v["PSFC"][j, i],
                            v["PBLH"][j, i], 0., np.log(.5 * (zw[1] - zw[0]) / .1), wspd, 0.,
                            v["MU"][j, i] + v["MUB"][j, i], 0., 18., 100., R_V / R_D - F(1.), G, R_D / CP, R_D, CP])
            yield dict(frame=frame, j=int(j), i=int(i), cloud=float(cloud[j, i])), \
                np.asfortranarray(prof, np.float32), np.asarray(sfc, np.float32)


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "module_bl_acm.F").write_bytes(PRISTINE.read_bytes())
    assert sha(BUILD / "module_bl_acm.F") == sha(PRISTINE)
    fns, meta = {}, {}
    for name, flags in VARIANTS.items():
        fns[name], meta[name] = build(name, flags)
    rows = list(columns())
    nz = rows[0][1].shape[0]
    out = {name: np.zeros((len(rows), nz + 1, 8), np.float32) for name in VARIANTS}
    osfc = {name: np.zeros((len(rows), 5), np.float32) for name in VARIANTS}
    for n, (_, prof, sfc) in enumerate(rows):
        for name, fn in fns.items():
            o = np.zeros((nz + 1, 8), np.float32, order="F"); s = np.zeros(5, np.float32)
            fn(nz, prof, sfc, o, s)
            out[name][n], osfc[name][n] = o, s
    assert all(np.isfinite(x).all() for x in (*out.values(), *osfc.values()))
    np.savez_compressed(BUILD / "oracle.npz", prof=np.stack([r[1] for r in rows]), sfc=np.stack([r[2] for r in rows]),
                        **{f"out_{k}": v for k, v in out.items()}, **{f"osfc_{k}": v for k, v in osfc.items()})
    manifest = dict(source=str(PRISTINE), source_sha256=sha(PRISTINE), wrapper_sha256=sha(WRAPPER),
                    builds=meta, columns=[r[0] for r in rows],
                    zero_vs_inf_identical=bool(all(np.array_equal(out["zero"], out["inf"]) for _ in [0])),
                    max_cloud_input=float(max(r[0]["cloud"] for r in rows)),
                    rqc_nonzero=int((np.abs(out["zero"][:, :, 4]) > 0).sum()),
                    rqi_nonzero=int((np.abs(out["zero"][:, :, 5]) > 0).sum()))
    (BUILD / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("zero_vs_inf_identical", "max_cloud_input", "rqc_nonzero", "rqi_nonzero")}))


if __name__ == "__main__":
    main()
