"""Pristine WRF 4.7.1 YSU public call with REAL start_em CTOPO/CTOPO2 (lane w41-topowind).

No JAX. Reuses the t-ysuqc oracle (unmodified module_bl_ysu.F + bl_ysu.F90 lane copies,
ysu_oracle_wrapper.f90, -O0 no-FMA zero build + FMA build) and its real CPU-WRF column
construction, but every column carries the CTOPO/CTOPO2 WRF's start_em would give that
cell (proofs/w41_topowind/build_ctopo_oracle.py, byte-exact):
  c02/c07 (Swiss RD11 grid, 3 km Alps): topo_wind=1 from the lane geogrid VAR_SSO
    (NCAR varsso_10m) + terrain Laplacian; extra picks in every CTOPO regime (max log,
    valley blend, ramp with CTOPO2 in (0,1), CTOPO=CTOPO2=0);
  c40 (9 km) / c85 (3 km marine): topo_wind=2 from wrfinput VAR (capped + uncapped vfac).
Both ysu_topdown_pblmix modes. Then WRF's own +-1 ULP input envelope (CTOPO/CTOPO2,
XLAND, currents, dt kept exact) and the tracked fixture
tests/fixtures/w41_topowind/ysu_ctopo_oracle.npz(.json).
usage: python build_ysu_ctopo_oracle.py <lane_dir>
"""
from pathlib import Path
import hashlib, json, sys, time

import numpy as np
import netCDF4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LANE = Path(sys.argv[1])
sys.argv = [sys.argv[0], str(LANE)]          # t_ysuqc.build_oracle reads its lane from argv
sys.path[:0] = [str(REPO / "proofs/t_ysuqc"), str(REPO / "src")]
import build_oracle as yo  # noqa: E402
from gpuwrf.io.topo_wind_static import lap_hgt, start_em_ctopo  # noqa: E402

SWISS_VAR_SSO = Path("<DATA_ROOT>/wrf_gpu2/scratch/w41-topowind/geogrid_swiss/geo_em.d01.nc")
# Real WN3 0227 columns (ALISIOS CPU-WRF history, Thompson -> QICE): d02 3 km dt 18, d03 1 km dt 6,
# topo_wind=1 CTOPO from WRF's own grid%ht (history HGT) + lane geogrid VAR_SSO.
WN3 = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
WN3_CASES = {"wn3_d02": ("d02", 18.0, ("2026-02-28_06:00:00", "2026-02-28_14:00:00")),
             "wn3_d03": ("d03", 6.0, ("2026-02-28_06:00:00", "2026-02-28_14:00:00"))}
NPERT = 96
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctopo_fields(case):
    if case in WN3_CASES:
        d = WN3_CASES[case][0]
        geo = Path(f"<DATA_ROOT>/wrf_gpu2/scratch/w41-topowind/geogrid_wn3/geo_em.{d}.nc")
        with netCDF4.Dataset(WN3 / f"wrfout_{d}_2026-02-28_00:00:00") as h, netCDF4.Dataset(WN3 / f"wrfinput_{d}") as w, \
                netCDF4.Dataset(geo) as g:
            ht, xl = np.asarray(h["HGT"][0], np.float32), np.asarray(w["XLAND"][0], np.float32)
            vs = np.asarray(g["VAR_SSO"][0], np.float32)
        ct, ct2 = start_em_ctopo(1, ht, xl, var_sso=vs)
        return 1, ct, ct2, lap_hgt(ht), dict(var_sso=str(geo), var_sso_sha256=sha(geo))
    root = yo.CASES[case][0]
    with netCDF4.Dataset(root / "wrfinput_d01") as w:
        ht, xl, var = (np.asarray(w[k][0], np.float32) for k in ("HGT", "XLAND", "VAR"))
    if case in ("c02", "c07"):
        with netCDF4.Dataset(SWISS_VAR_SSO) as g:
            vs = np.asarray(g["VAR_SSO"][0], np.float32)
        ct, ct2 = start_em_ctopo(1, ht, xl, var_sso=vs)
        return 1, ct, ct2, lap_hgt(ht), dict(var_sso=str(SWISS_VAR_SSO), var_sso_sha256=sha(SWISS_VAR_SSO))
    ct, ct2 = start_em_ctopo(2, ht, xl, var2d=var)
    return 2, ct, ct2, lap_hgt(ht), {}


def regime_picks(topo_wind, ct, ct2, lap, var, xl):
    land = xl < 1.5
    if topo_wind == 1:
        groups = dict(ctopo_max=(ct == ct.max()), blend=(lap <= -10) & (lap >= -20) & land,
                      ramp=(ct2 > 0) & (ct2 < 1), zero=(ct == 0) & (ct2 == 0))
    else:
        raw = (var * np.float32(0.4)) / np.float32(200.0) + np.float32(1.175)
        groups = dict(vfac_capped=land & (raw > np.float32(1.575)), vfac_uncapped=land & (raw <= np.float32(1.575)),
                      ctopo_max=land & (ct == ct[land].max()))
    picks = {}
    for tag, mask in groups.items():
        idx = np.argwhere(mask[2:-2, 2:-2]) + 2        # interior, like the t-ysuqc random picks
        if idx.size:
            picks[tag] = tuple(int(x) for x in idx[len(idx) // 2])
    return picks


def columns():
    base = list(yo.real_columns())
    extra_frames = {"c02": ("2023-01-15_06:00:00", "2023-01-15_15:00:00"), "c07": ("2023-01-15_12:00:00",),
                    "c40": ("2026-10-13_06:00:00",), "c85": ("2026-10-13_12:00:00",)}
    fields = {case: ctopo_fields(case) for case in (*yo.CASES, *WN3_CASES)}
    rows = []
    for meta, prof, surf in base:
        tw, ct, ct2, lap, prov = fields[meta["source"]]
        j, i = meta["j"], meta["i"]
        surf = surf.copy(); surf[14], surf[15] = ct[j, i], ct2[j, i]
        rows.append((dict(meta, topo_wind=tw), prof, surf))
    # Extra cells in every CTOPO regime: rebuild the t-ysuqc column at chosen (j, i).
    for case, (d, _dt, frames) in WN3_CASES.items():
        tw, ct, ct2, lap, _ = fields[case]
        with netCDF4.Dataset(WN3 / f"wrfinput_{d}") as w:
            var, xl = np.asarray(w["VAR"][0], np.float32), np.asarray(w["XLAND"][0], np.float32)
        picks = regime_picks(tw, ct, ct2, lap, var, xl)
        land = np.argwhere(xl[2:-2, 2:-2] < 1.5) + 2
        picks["land_random"] = tuple(int(x) for x in land[len(land) // 3])
        for frame in frames:
            for meta, prof, surf in _replay(case, frame, picks):
                j, i = meta["j"], meta["i"]
                surf = surf.copy(); surf[14], surf[15] = ct[j, i], ct2[j, i]
                rows.append((dict(meta, topo_wind=tw, lap_hgt=float(lap[j, i])), prof, surf))
    for case, frames in extra_frames.items():
        tw, ct, ct2, lap, _ = fields[case]
        root = yo.CASES[case][0]
        with netCDF4.Dataset(root / "wrfinput_d01") as w:
            var, xl = np.asarray(w["VAR"][0], np.float32), np.asarray(w["XLAND"][0], np.float32)
        picks = regime_picks(tw, ct, ct2, lap, var, xl)
        for frame in frames:
            for meta, prof, surf in _replay(case, frame, picks):
                j, i = meta["j"], meta["i"]
                surf = surf.copy(); surf[14], surf[15] = ct[j, i], ct2[j, i]
                rows.append((dict(meta, topo_wind=tw, lap_hgt=float(lap[j, i])), prof, surf))
    return rows, {c: f[4] for c, f in fields.items()}


def _replay(case, frame, picks):
    if case in WN3_CASES:
        dom, dt, _ = WN3_CASES[case]
        root, has_qi = WN3, True
    else:
        (root, dt, has_qi, _), dom = yo.CASES[case], "d01"
    G, R_D, CP = yo.G, yo.R_D, yo.CP
    with netCDF4.Dataset(root / f"wrfout_{dom}_{frame}") as d:
        v = {k: np.asarray(d.variables[k][0], np.float64) for k in (
            "U", "V", "T", "P", "PB", "PH", "PHB", "QVAPOR", "QCLOUD", "PSFC", "UST", "HFX",
            "QFX", "XLAND", "U10", "V10", "PBLH", "TSK", "XLONG")}
        v["QICE"] = np.asarray(d.variables["QICE"][0], np.float64) if has_qi else np.zeros_like(v["QCLOUD"])
    nz = v["T"].shape[0]
    hour = int(frame[11:13])
    for tag, (j, i) in picks.items():
        solar_hour = (hour + v["XLONG"][j, i] / 15.) % 24
        daytime = 7. < solar_hour < 17.
        p = (v["P"] + v["PB"])[:, j, i]
        zw = (v["PH"] + v["PHB"])[:, j, i] / G
        dz = np.diff(zw)
        u = .5 * (v["U"][:, j, i] + v["U"][:, j, i + 1]); vv = .5 * (v["V"][:, j, i] + v["V"][:, j + 1, i])
        pi = (p / 1.e5) ** (R_D / CP); th = v["T"][:, j, i] + 300.
        pw = np.empty(nz + 1); pw[0] = v["PSFC"][j, i]
        zm = .5 * (zw[:-1] + zw[1:])
        wgt = (zw[1:-1] - zm[:-1]) / (zm[1:] - zm[:-1])
        pw[1:-1] = p[:-1] + wgt * (p[1:] - p[:-1]); pw[-1] = p[-1] - .5 * (p[-2] - p[-1])
        qc = v["QCLOUD"][:, j, i]; qi = v["QICE"][:, j, i]
        prof = np.zeros((nz + 1, 11), np.float32, order="F")
        for col, val in enumerate((u, vv, th * pi, v["QVAPOR"][:, j, i], qc, qi, p, None, pi, dz,
                                   yo.rthraten_profile(qc, qi, daytime))):
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
        s = np.array([v["PSFC"][j, i], znt, max(v["UST"][j, i], .01), fm, fh, v["XLAND"][j, i],
                      v["HFX"][j, i], v["QFX"][j, i], wspd, br, v["U10"][j, i], v["V10"][j, i],
                      0., 0., 1., 1., dt], np.float32)
        yield dict(source=case, frame=frame, pick=tag, j=int(j), i=int(i), flag_qi=has_qi,
                   daytime=bool(daytime)), prof, s


def main():
    t0 = time.monotonic()
    fns = {}
    builds = {}
    for name in ("zero", "fma"):
        fns[name], builds[name] = yo.build(name, yo.VARIANTS[name])
    rows, prov = columns()
    metas, profs, surfs, outs = [], [], [], {"zero": [], "fma": []}
    for meta, prof, surf in rows:
        n = prof.shape[0] - 1
        for topdown in (1, 0):
            for name, fn in fns.items():
                res = np.zeros((n + 1, 9), np.float32, order="F")
                fn(n, topdown, 1, int(meta["flag_qi"]), prof.copy(order="F"), surf.copy(), res)
                outs[name].append(res)
            metas.append(dict(meta, topdown=topdown, nz=n, ctopo=float(surf[14]), ctopo2=float(surf[15])))
            profs.append(prof); surfs.append(surf.copy())
    zero, fma = np.stack(outs["zero"]), np.stack(outs["fma"])
    # WRF's own +-1 ULP input envelope (t_ysuqc/ulp_envelope.py rule; CTOPO/CTOPO2 exact).
    rng = np.random.default_rng(20261017)
    lower, upper = np.minimum(zero, fma), np.maximum(zero, fma)
    fn = fns["zero"]
    for row, (prof0, surf0) in enumerate(zip(profs, surfs)):
        n = metas[row]["nz"]
        for _ in range(NPERT):
            step = rng.integers(-1, 2, size=prof0.shape)
            prof = np.where(step > 0, np.nextafter(prof0, np.float32(np.inf)),
                            np.where(step < 0, np.nextafter(prof0, np.float32(-np.inf)), prof0)).astype(np.float32)
            prof[:, 4:6] = np.where(prof0[:, 4:6] == 0, 0, prof[:, 4:6])
            sstep = rng.integers(-1, 2, size=surf0.shape)
            surf = np.where(sstep > 0, np.nextafter(surf0, np.float32(np.inf)),
                            np.where(sstep < 0, np.nextafter(surf0, np.float32(-np.inf)), surf0)).astype(np.float32)
            surf[[5, 12, 13, 14, 15, 16]] = surf0[[5, 12, 13, 14, 15, 16]]
            res = np.zeros((n + 1, 9), np.float32, order="F")
            fn(n, metas[row]["topdown"], 1, int(metas[row]["flag_qi"]), np.asfortranarray(prof), surf, res)
            if np.isfinite(res).all():
                lower[row] = np.minimum(lower[row], res)
                upper[row] = np.maximum(upper[row], res)
    out = REPO / "tests/fixtures/w41_topowind"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ysu_ctopo_oracle.npz"
    surf_arr = np.stack(surfs)
    np.savez_compressed(path, profiles=np.stack(profs), surface=surf_arr, nz=np.array([m["nz"] for m in metas]),
                        topdown=np.array([m["topdown"] for m in metas]),
                        flag_qi=np.array([m["flag_qi"] for m in metas]), wrf=zero, lower=lower, upper=upper)
    ct, ct2 = surf_arr[:, 14], surf_arr[:, 15]
    census = dict(rows=len(metas), columns=len(metas) // 2,
                  ctopo_gt1=int((ct > 1).sum() // 2), ctopo_lt1=int((ct < 1).sum() // 2),
                  ctopo_zero=int((ct == 0).sum() // 2), ctopo2_partial=int(((ct2 > 0) & (ct2 < 1)).sum() // 2),
                  ctopo2_zero=int((ct2 == 0).sum() // 2), topo_wind1=sum(m["topo_wind"] == 1 for m in metas) // 2,
                  topo_wind2=sum(m["topo_wind"] == 2 for m in metas) // 2,
                  wn3_columns=sum(m["source"].startswith("wn3") for m in metas) // 2,
                  u10_blended=int(np.count_nonzero(zero[:, 4, 8] != surf_arr[:, 10]) // 2),
                  ctopo_drag_sensitive=None)
    meta = dict(fixture_sha256=sha(path), source=yo.SOURCES, wrapper="proofs/t_ysuqc/ysu_oracle_wrapper.f90",
                wrapper_sha256=sha(yo.WRAPPER), reference_build=builds["zero"], fma_build=builds["fma"],
                ctopo_provenance=prov, envelope=dict(npert=NPERT, seed=20261017), census=census,
                rows=metas, elapsed_s=time.monotonic() - t0,
                layout=dict(profiles="u v t qv qc qi p p8w(n+1) pi dz8w rthraten",
                            surface="psfc znt ust psim(fm) psih(fhh) xland hfx qfx wspd br u10 v10 uoce voce ctopo ctopo2 dt",
                            wrf="rublten rvblten rthblten rqvblten rqcblten rqiblten exch_h exch_m scalars(hpbl,kpbl,wstar,delta,u10,v10)"),
                rule="literal path BYTE with exp/tanh through glibc (q rates below 1e-30 absolute, E117); compiled path "
                     "inside WRF +-1 ULP input envelope widened by 2 REAL4 output ULPs (same rule as t-ysuqc)")
    (out / "ysu_ctopo_oracle.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("fixture_sha256", "census", "elapsed_s")}, indent=1))


def _load_alisios_guard():
    """alisios_guard lives on the DA lanes (w41-da-core) until merged; stdlib-only module -> load by file path."""
    import importlib.util, os
    path = Path(os.environ.get("GPUWRF_ALISIOS_GUARD_SRC", "<USER_HOME>/src/wrf_gpu2_wt/w41-da-core/src")) / \
        "gpuwrf/assimilation/letkf/alisios_guard.py"
    spec = importlib.util.spec_from_file_location("w41_alisios_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guarded_main():
    """WN3 reads are ALISIOS CPU-WRF output -> only inside alisios_guard (scan -> admit -> READ_SPY), wave V041 rule."""
    guarded_session = _load_alisios_guard().guarded_session
    allowed = {WN3 / f"wrfinput_{d}": "20260227" for d in ("d02", "d03")}
    for d, _dt, frames in WN3_CASES.values():
        for frame in ("2026-02-28_00:00:00", *frames):
            allowed[WN3 / f"wrfout_{d}_{frame}"] = "20260228"
    with guarded_session(allowed, ["20260227"], LANE / "guard_ysu",
                         purpose="w41-topowind YSU+CTOPO oracle: real WN3 0227 d02/d03 columns (CPU-WRF history)") as s:
        main()
    print(json.dumps(dict(guard_receipt=str(LANE / "guard_ysu"), opened=len(s.observed), refused=len(s.refused))))


if __name__ == "__main__":
    guarded_main()
