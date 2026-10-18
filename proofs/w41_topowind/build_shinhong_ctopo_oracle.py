"""Pristine WRF 4.7.1 Shin-Hong public call with REAL start_em CTOPO/CTOPO2 (lane w41-topowind).

No JAX. Reuses the t-shinhong oracle (unmodified module_bl_shinhong.F lane copy +
shinhong_oracle_wrapper.f90, -O0 no-FMA zero build, FMA build) and its c07 (Swiss, mp9,
Shin-Hong) column construction, at the t-shinhong picks plus cells in every topo_wind=1
CTOPO regime (log, valley blend, ramp with CTOPO2 in (0,1), zero), CTOPO/CTOPO2 from
the byte-exact start_em port with the lane geogrid VAR_SSO. Modes 0 (Registry default)
and 1. WRF's own +-1 ULP input envelope (48 perturbations, CTOPO/CTOPO2/XLAND/dt/dx exact).
Writes tests/fixtures/w41_topowind/shinhong_ctopo_oracle.npz(.json).
usage: python build_shinhong_ctopo_oracle.py <lane_dir>
"""
from pathlib import Path
import hashlib, json, shutil, sys, time

import numpy as np
import netCDF4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LANE = Path(sys.argv[1])
(LANE / "oracle").mkdir(parents=True, exist_ok=True)
PRISTINE_SH = Path("<USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_shinhong.F")
if not (LANE / "oracle/module_bl_shinhong.F").exists():
    shutil.copy2(PRISTINE_SH, LANE / "oracle/module_bl_shinhong.F")
sys.argv = [sys.argv[0], str(LANE)]
sys.path[:0] = [str(REPO / "proofs/t_shinhong"), str(REPO / "src")]
import build_oracle as so  # noqa: E402
from gpuwrf.io.topo_wind_static import lap_hgt, start_em_ctopo  # noqa: E402

SWISS_VAR_SSO = Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-topowind/geogrid_swiss/geo_em.d01.nc")
NPERT = 48
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctopo_fields():
    with netCDF4.Dataset(so.C07.parent / "input/wrfinput_d01") as w:
        ht, xl = (np.asarray(w[k][0], np.float32) for k in ("HGT", "XLAND"))
    with netCDF4.Dataset(SWISS_VAR_SSO) as g:
        vs = np.asarray(g["VAR_SSO"][0], np.float32)
    ct, ct2 = start_em_ctopo(1, ht, xl, var_sso=vs)
    return ct, ct2, lap_hgt(ht)


def regime_picks(ct, ct2, lap):
    groups = dict(ctopo_max=ct == ct.max(), blend=(lap <= -10) & (lap >= -20),
                  ramp=(ct2 > 0) & (ct2 < 1), zero=(ct == 0) & (ct2 == 0))
    picks = {}
    for tag, mask in groups.items():
        idx = np.argwhere(mask[2:-2, 2:-2]) + 2
        if idx.size:
            picks[tag] = tuple(int(x) for x in idx[len(idx) // 2])
    return picks


def columns_at(frame, picks):
    """t-shinhong real_columns body (same construction) for explicit (j, i) picks."""
    G, R_D, CP = so.G, so.R_D, so.CP
    with netCDF4.Dataset(so.C07 / f"wrfout_d01_{frame}") as d:
        v = {k: np.asarray(d.variables[k][0], np.float64) for k in (
            "U", "V", "T", "P", "PB", "PH", "PHB", "QVAPOR", "QCLOUD", "QICE", "PSFC", "UST", "HFX",
            "QFX", "XLAND", "U10", "V10", "F", "PBLH", "TSK")}
    nz = v["T"].shape[0]
    for tag, (j, i) in picks.items():
        p = (v["P"] + v["PB"])[:, j, i]
        zw = (v["PH"] + v["PHB"])[:, j, i] / G
        dz = np.diff(zw)
        u = .5 * (v["U"][:, j, i] + v["U"][:, j, i + 1]); vv = .5 * (v["V"][:, j, i] + v["V"][:, j + 1, i])
        pi = (p / 1.e5) ** (R_D / CP); th = v["T"][:, j, i] + 300.
        pw = np.empty(nz + 1); pw[0] = v["PSFC"][j, i]
        zm = .5 * (zw[:-1] + zw[1:])
        wgt = (zw[1:-1] - zm[:-1]) / (zm[1:] - zm[:-1])
        pw[1:-1] = p[:-1] + wgt * (p[1:] - p[:-1]); pw[-1] = p[-1] - .5 * (p[-2] - p[-1])
        prof = np.zeros((nz + 1, 11), np.float32, order="F")
        for col, val in enumerate((u, vv, th * pi, v["QVAPOR"][:, j, i], p, None, pi, dz,
                                   np.full(nz, .005), v["QCLOUD"][:, j, i], v["QICE"][:, j, i])):
            if val is None:
                prof[:, col] = pw
            else:
                prof[:-1, col] = val; prof[-1, col] = val[-1]
        land = v["XLAND"][j, i] < 1.5
        znt = .1 if land else 2.e-4
        za = .5 * dz[0]
        wspd = max(np.hypot(u[0], vv[0]), .1)
        ths = v["TSK"][j, i] * (1.e5 / v["PSFC"][j, i]) ** (R_D / CP)
        br = float(np.clip(G * za * (th[0] - ths) / (th[0] * wspd * wspd), -4., .5))
        fm = np.log((za + znt) / znt); fh = np.log((za + znt) / max(.1 * znt, 1.e-4))
        s = np.array([v["PSFC"][j, i], znt, max(v["UST"][j, i], .01), v["HFX"][j, i], v["QFX"][j, i],
                      wspd, br, fm, fh, 18., v["XLAND"][j, i], v["U10"][j, i], v["V10"][j, i], 3000., 3000.,
                      v["F"][j, i], 1., 1., v["PBLH"][j, i]], np.float32)
        yield dict(source="c07", frame=frame, pick=tag, j=int(j), i=int(i)), prof, s


def main():
    t0 = time.monotonic()
    fns, builds = {}, {}
    for name in ("zero", "fma"):
        fns[name], builds[name] = so.build(name, so.VARIANTS[name])
    ct, ct2, lap = ctopo_fields()
    cols = [(m, p, s) for m, p, s, _ in so.real_columns()]
    for frame in ("2023-01-15_06:00:00", "2023-01-15_12:00:00"):
        cols += list(columns_at(frame, regime_picks(ct, ct2, lap)))
    rows, profs, surfs, modes, outs = [], [], [], [], {"zero": [], "fma": []}
    for meta, prof, surf in cols:
        surf = surf.copy()
        surf[16], surf[17] = ct[meta["j"], meta["i"]], ct2[meta["j"], meta["i"]]
        n = prof.shape[0] - 1
        for mode in (0, 1):
            for name, fn in fns.items():
                res = np.zeros((n + 1, 10), np.float32, order="F")
                fn(n, mode, prof.copy(order="F"), surf.copy(), res)
                outs[name].append(res)
            rows.append(dict(meta, mode=mode, nz=n, ctopo=float(surf[16]), ctopo2=float(surf[17]),
                             lap_hgt=float(lap[meta["j"], meta["i"]])))
            profs.append(prof); surfs.append(surf)
            modes.append(mode)
    zero, fma = np.stack(outs["zero"]), np.stack(outs["fma"])
    rng = np.random.default_rng(20261017)
    lower, upper = np.minimum(zero, fma), np.maximum(zero, fma)
    keep = [9, 10, 13, 14, 15, 16, 17, 18]          # dt, xland, dx, dy, corf, ctopo, ctopo2, hpbl_in exact
    for r, (p0, s0) in enumerate(zip(profs, surfs)):
        n = rows[r]["nz"]
        for _ in range(NPERT):
            step = rng.integers(-1, 2, size=p0.shape)
            p = np.where(step > 0, np.nextafter(p0, np.float32(np.inf)),
                         np.where(step < 0, np.nextafter(p0, np.float32(-np.inf)), p0)).astype(np.float32)
            p[:, 9:11] = np.where(p0[:, 9:11] == 0, 0, p[:, 9:11])
            ss = rng.integers(-1, 2, size=s0.shape)
            s = np.where(ss > 0, np.nextafter(s0, np.float32(np.inf)),
                         np.where(ss < 0, np.nextafter(s0, np.float32(-np.inf)), s0)).astype(np.float32)
            s[keep] = s0[keep]
            res = np.zeros((n + 1, 10), np.float32, order="F")
            fns["zero"](n, modes[r], np.asfortranarray(p), s, res)
            if np.isfinite(res).all():
                lower[r] = np.minimum(lower[r], res)
                upper[r] = np.maximum(upper[r], res)
    out = REPO / "tests/fixtures/w41_topowind"
    path = out / "shinhong_ctopo_oracle.npz"
    np.savez_compressed(path, profiles=np.stack(profs), surface=np.stack(surfs), nz=np.array([r["nz"] for r in rows]),
                        mode=np.array(modes), wrf=zero, lower=lower, upper=upper)
    s = np.stack(surfs)
    census = dict(rows=len(rows), ctopo_gt1=int((s[:, 16] > 1).sum()), ctopo_lt1=int((s[:, 16] < 1).sum()),
                  ctopo_zero=int((s[:, 16] == 0).sum()), ctopo2_partial=int(((s[:, 17] > 0) & (s[:, 17] < 1)).sum()),
                  ctopo2_zero=int((s[:, 17] == 0).sum()),
                  u10_blended=int(np.count_nonzero(zero[:, 4, 9] != s[:, 11])))
    meta = dict(fixture_sha256=sha(path), source="WRF 4.7.1 phys/module_bl_shinhong.F (pristine, unmodified lane copy)",
                source_sha256=sha(LANE / "oracle/module_bl_shinhong.F"), wrapper="proofs/t_shinhong/shinhong_oracle_wrapper.f90",
                wrapper_sha256=sha(so.WRAPPER), reference_build=builds["zero"], fma_build=builds["fma"],
                var_sso=str(SWISS_VAR_SSO), var_sso_sha256=sha(SWISS_VAR_SSO), envelope=dict(npert=NPERT),
                census=census, rows=rows, elapsed_s=time.monotonic() - t0,
                layout=dict(surface="psfc znt ust hfx qfx wspd br psim psih dt xland u10 v10 dx dy corf ctopo ctopo2 hpbl_in",
                            wrf="tke el exch_h rublten rvblten rthblten rqvblten rqcblten rqiblten scalars(hpbl,kpbl,wstar,delta,u10,v10,regime)"))
    (out / "shinhong_ctopo_oracle.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("fixture_sha256", "census", "elapsed_s")}, indent=1))


if __name__ == "__main__":
    main()
