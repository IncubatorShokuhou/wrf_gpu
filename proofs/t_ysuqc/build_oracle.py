"""Pristine WRF 4.7.1 YSU (bl_pbl_physics=1) public-call oracle on real columns.

Lane t-ysuqc. No JAX. Builds the UNMODIFIED phys/module_bl_ysu.F +
phys/physics_mmm/bl_ysu.F90 + phys/ccpp_kind_types.F (lane-dir copies, E104)
with proofs/t_ysuqc/ysu_oracle_wrapper.f90 in three variants:
  zero : -O0 -ffp-contract=off -finit-real=zero   (policy reference)
  inf  : -O0 -ffp-contract=off -finit-real=inf    (uninitialised-read sensitivity)
  fma  : -O2 -mfma -ffp-contract=fast -finit-real=zero (FMA/optimiser spread)
Columns: real 44-level columns of original CPU-WRF composite runs: bl=1 c02
(Swiss 2023-01-15, mp1 Kessler: qc, no qi -> FLAG_QI false) and c40 (2026-10-13,
mp8, cloud-free), plus c07 (Swiss, mp9, cloud ice) and c85 (mp8, marine cloud);
picks max/min HFX, max low cloud water, max low / column cloud ice, max PBLH,
max cloud at the PBL top, 2 random.
RTHRATEN is not in wrfout: a deterministic real-structured profile (clear-sky
-2e-5 K/s, cloud-top -4e-4 K/s, daytime in-cloud +1e-5 K/s) exercises the
top-down radsum branch with both signs. Modes: ysu_topdown_pblmix 1 (Registry
default, the composites) and 0. Writes <lane>/oracle.npz + oracle_manifest.json.
usage: python build_oracle.py <lane_dir>
"""
from pathlib import Path
import ctypes, hashlib, json, re, shutil, subprocess, sys, time
import numpy as np
import netCDF4

HERE = Path(__file__).resolve().parent
LANE = Path(sys.argv[1])
BUILD = LANE / 'oracle'
PRISTINE = Path('<USER_HOME>/src/wrf_pristine/WRF/phys')
SOURCES = {
    'ccpp_kind_types.F': '0cc7b2d71d8a93ff1f2b401324de8fd1ee70ba598f83d89254d6dbe8c31d0e4b',
    'physics_mmm/bl_ysu.F90': 'b2c9f11f92f967b053948b42a7c5dbaf1b7f72ddb9d35948e96c32a5d91092d2',
    'module_bl_ysu.F': '46ad9d21ecc88461ec3fc06086f9cb25fdccf9fc0f9b8f122a0cc2ff7c92a662',
}
WRAPPER = HERE / 'ysu_oracle_wrapper.f90'
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
CASES = {
    'c02': (Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c02/cpu'), 18.0, False,
            ('2023-01-15_00:00:00', '2023-01-15_06:00:00', '2023-01-15_12:00:00', '2023-01-15_15:00:00')),
    'c40': (Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c40/cpu'), 50.0, True,
            ('2026-10-13_00:00:00', '2026-10-13_06:00:00', '2026-10-13_12:00:00', '2026-10-13_18:00:00')),
    # Census (E154): c40 is cloud- and ice-free; c07 (Swiss mp9) carries cloud ice
    # (QICE up to 1.1e-4), c85 (mp8, 2026-10-13) marine cloud water.
    'c07': (Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c07/cpu'), 18.0, True,
            ('2023-01-15_06:00:00', '2023-01-15_12:00:00', '2023-01-15_18:00:00')),
    'c85': (Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c85/cpu'), 15.0, True,
            ('2026-10-13_06:00:00', '2026-10-13_12:00:00')),
}
VARIANTS = {
    'zero': ['-O0', '-ffp-contract=off', '-finit-real=zero', '-finit-integer=0', '-finit-logical=false'],
    'inf': ['-O0', '-ffp-contract=off', '-finit-real=inf', '-finit-integer=0', '-finit-logical=false'],
    'fma': ['-O2', '-mfma', '-ffp-contract=fast', '-finit-real=zero', '-finit-integer=0', '-finit-logical=false'],
}
R_D, CP, G = 287.0, 1004.5, 9.81
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(name, flags):
    out = BUILD / name
    out.mkdir(parents=True, exist_ok=True)
    srcs = []
    for rel, digest in SOURCES.items():
        dst = BUILD / Path(rel).name
        if not dst.exists():
            shutil.copy2(PRISTINE / rel, dst)
        assert sha(dst) == sha(PRISTINE / rel) == digest, rel
        srcs.append(str(dst))
    so = out / 'oracle.so'
    subprocess.run([FC, '-shared', '-fPIC', '-cpp', *flags, '-ffree-form', '-ffree-line-length-none', *srcs,
                    str(WRAPPER), '-o', str(so)], cwd=out, check=True, timeout=300)
    asm = subprocess.run(['objdump', '-d', str(so)], capture_output=True, text=True, check=True).stdout
    fn = ctypes.CDLL(str(so)).probe
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')
    fn.argtypes = [ctypes.c_int] * 4 + [arr, arr, arr]
    return fn, dict(flags=flags, sha256=sha(so), fma_instructions=len(re.findall(r'\svfn?m(?:add|sub)\d', asm)))


def rthraten_profile(qc, qi, daytime):
    cld = (qc + qi) > 1.e-5
    top = cld & ~np.r_[cld[1:], False]
    r = np.full(qc.shape, -2.e-5)
    if daytime:
        r[cld] = 1.e-5
    r[top] = -4.e-4
    return r


def real_columns():
    rng = np.random.default_rng(20261015)
    for case, (root, dt, has_qi, frames) in CASES.items():
        for frame in frames:
            with netCDF4.Dataset(root / f'wrfout_d01_{frame}') as d:
                v = {k: np.asarray(d.variables[k][0], np.float64) for k in (
                    'U', 'V', 'T', 'P', 'PB', 'PH', 'PHB', 'QVAPOR', 'QCLOUD', 'PSFC', 'UST', 'HFX',
                    'QFX', 'XLAND', 'U10', 'V10', 'PBLH', 'TSK', 'XLONG')}
                v['QICE'] = (np.asarray(d.variables['QICE'][0], np.float64) if has_qi
                             else np.zeros_like(v['QCLOUD']))
            nz, ny, nx = v['T'].shape
            zw_all = (v['PH'] + v['PHB']) / G
            zw_all = zw_all - zw_all[:1]
            cloud = v['QCLOUD'] + v['QICE']
            kpbl_like = np.clip(np.argmax(zw_all[1:] > v['PBLH'][None], axis=0), 0, nz - 2)
            at_top = np.take_along_axis(cloud, np.minimum(kpbl_like + 1, nz - 1)[None], 0)[0]
            pick = dict(max_hfx=np.unravel_index(np.argmax(v['HFX']), (ny, nx)),
                        min_hfx=np.unravel_index(np.argmin(v['HFX']), (ny, nx)),
                        max_qc_low=np.unravel_index(np.argmax(v['QCLOUD'][:15].sum(0)), (ny, nx)),
                        max_pblh=np.unravel_index(np.argmax(v['PBLH']), (ny, nx)),
                        max_cloud_pbltop=np.unravel_index(np.argmax(at_top), (ny, nx)))
            if has_qi:
                pick['max_qi_low'] = np.unravel_index(np.argmax(v['QICE'][:20].sum(0)), (ny, nx))
                pick['max_qi_col'] = np.unravel_index(np.argmax(v['QICE'].sum(0)), (ny, nx))
            for r in range(2):
                pick[f'random{r}'] = (int(rng.integers(2, ny - 2)), int(rng.integers(2, nx - 2)))
            hour = int(frame[11:13])
            for tag, (j, i) in pick.items():
                j, i = int(j), int(i)
                solar_hour = (hour + v['XLONG'][j, i] / 15.) % 24
                daytime = 7. < solar_hour < 17.
                p = (v['P'] + v['PB'])[:, j, i]
                zw = (v['PH'] + v['PHB'])[:, j, i] / G
                dz = np.diff(zw)
                u = .5 * (v['U'][:, j, i] + v['U'][:, j, i + 1]); vv = .5 * (v['V'][:, j, i] + v['V'][:, j + 1, i])
                pi = (p / 1.e5) ** (R_D / CP); th = v['T'][:, j, i] + 300.
                # phy_prep-like interface pressures: height-weighted interior, PSFC bottom.
                pw = np.empty(nz + 1); pw[0] = v['PSFC'][j, i]
                zm = .5 * (zw[:-1] + zw[1:])
                wgt = (zw[1:-1] - zm[:-1]) / (zm[1:] - zm[:-1])
                pw[1:-1] = p[:-1] + wgt * (p[1:] - p[:-1]); pw[-1] = p[-1] - .5 * (p[-2] - p[-1])
                qc = v['QCLOUD'][:, j, i]; qi = v['QICE'][:, j, i]
                prof = np.zeros((nz + 1, 11), np.float32, order='F')
                for col, val in enumerate((u, vv, th * pi, v['QVAPOR'][:, j, i], qc, qi, p, None, pi, dz,
                                           rthraten_profile(qc, qi, daytime))):
                    if val is None:
                        prof[:, col] = pw
                    else:
                        prof[:-1, col] = val; prof[-1, col] = val[-1]
                land = v['XLAND'][j, i] < 1.5
                znt = .1 if land else 2.e-4
                za = .5 * dz[0]
                wspd = max(np.hypot(u[0], vv[0]), .1)
                ths = v['TSK'][j, i] * (1.e5 / v['PSFC'][j, i]) ** (R_D / CP)
                br = float(np.clip(G * za * (th[0] - ths) / (th[0] * wspd * wspd), -4., .5))
                fm = np.log((za + znt) / znt); fh = np.log((za + znt) / max(.1 * znt, 1.e-4))
                s = np.array([v['PSFC'][j, i], znt, max(v['UST'][j, i], .01), fm, fh, v['XLAND'][j, i],
                              v['HFX'][j, i], v['QFX'][j, i], wspd, br, v['U10'][j, i], v['V10'][j, i],
                              0., 0., 1., 1., dt], np.float32)
                yield dict(source=case, frame=frame, pick=tag, j=j, i=i, flag_qi=has_qi,
                           daytime=bool(daytime)), prof, s


def main():
    t0 = time.monotonic()
    fns, builds = {}, {}
    for name, flags in VARIANTS.items():
        fns[name], builds[name] = build(name, flags)
    rows, profiles, surfaces, outs = [], [], [], {k: [] for k in VARIANTS}
    for meta, prof, surf in real_columns():
        n = prof.shape[0] - 1
        for topdown in (1, 0):
            for name, fn in fns.items():
                res = np.zeros((n + 1, 9), np.float32, order='F')
                fn(n, topdown, 1, int(meta['flag_qi']), prof.copy(order='F'), surf.copy(), res)
                outs[name].append(res)
            rows.append(dict(meta, topdown=topdown, nz=n))
            profiles.append(prof); surfaces.append(surf.copy())
    zero, inf, fma = (np.stack(outs[k]) for k in ('zero', 'inf', 'fma'))
    np.savez(LANE / 'oracle.npz', profiles=np.stack(profiles), surface=np.stack(surfaces),
             nz=np.array([r['nz'] for r in rows]), topdown=np.array([r['topdown'] for r in rows]),
             flag_qi=np.array([r['flag_qi'] for r in rows]), out_zero=zero, out_inf=inf, out_fma=fma)
    manifest = dict(source_sha256={k: sha(BUILD / Path(k).name) for k in SOURCES}, wrapper_sha256=sha(WRAPPER),
                    builds=builds, rows=rows,
                    init_sensitivity_byte_identical=bool(zero.tobytes() == inf.tobytes()),
                    init_sensitivity_max_abs=float(np.nanmax(np.abs(zero - inf))),
                    nonfinite={k: int(np.count_nonzero(~np.isfinite(np.stack(v)))) for k, v in outs.items()},
                    kpbl_topdown_differs=int(sum(1 for a in range(0, len(rows), 2)
                                                 if zero[a, 1, 8] != zero[a + 1, 1, 8])),
                    oracle_npz_sha256=sha(LANE / 'oracle.npz'), elapsed_s=time.monotonic() - t0, GPU=False)
    (LANE / 'oracle_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
