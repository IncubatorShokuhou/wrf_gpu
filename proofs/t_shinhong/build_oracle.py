"""Pristine WRF 4.7.1 Shin-Hong public-call oracle on real-structured columns.

Lane t-shinhong. No JAX. Builds the UNMODIFIED module_bl_shinhong.F (lane-dir
copy, E104) + proofs/t_shinhong/shinhong_oracle_wrapper.f90 in three variants:
  zero : -O0 -ffp-contract=off -finit-real=zero   (policy reference)
  inf  : -O0 -ffp-contract=off -finit-real=inf    (uninitialised-read sensitivity)
  fma  : -O2 -mfma -ffp-contract=fast -finit-real=zero (FMA/optimiser spread)
Columns: 4 synthetic 16-level regimes of the predecessor (CM78-CM89, qc=qi=0)
plus real 44-level columns of the original CPU-WRF c07 composite run
(bl_pbl_physics=11, mp=9, Swiss 2023-01-15) incl. cloud water/ice and the
grid Coriolis parameter F. Modes: 0, 1 cold (TKE=.005), 1 warm (TKE from the
zero build's own cold diagnostic call). Writes <out>/oracle.npz + manifest.
usage: python build_oracle.py <lane_dir>
"""
from pathlib import Path
import ctypes, hashlib, json, re, subprocess, sys, time
import numpy as np
import netCDF4

HERE = Path(__file__).resolve().parent
LANE = Path(sys.argv[1])
BUILD = LANE / 'oracle'
SOURCE = BUILD / 'module_bl_shinhong.F'
PRISTINE = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_shinhong.F')
WRAPPER = HERE / 'shinhong_oracle_wrapper.f90'
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
C07 = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c07/cpu')
FRAMES = ('2023-01-15_00:00:00', '2023-01-15_06:00:00', '2023-01-15_12:00:00', '2023-01-15_15:00:00')
VARIANTS = {
    'zero': ['-O0', '-ffp-contract=off', '-finit-real=zero'],
    'inf': ['-O0', '-ffp-contract=off', '-finit-real=inf'],
    'fma': ['-O2', '-mfma', '-ffp-contract=fast', '-finit-real=zero'],
}
R_D, R_V, CP, G = 287.0, 461.6, 1004.5, 9.81
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert sha(SOURCE) == sha(PRISTINE) == '99f44dbeb5e586b96be14424b8ab27c9986ffbd81f007f41fb8528d8ea466d56'


def build(name, flags):
    out = BUILD / name
    out.mkdir(exist_ok=True)
    so = out / 'oracle.so'
    subprocess.run([FC, '-shared', '-fPIC', *flags, '-ffree-form', '-ffree-line-length-none', str(SOURCE),
                    str(WRAPPER), '-o', str(so)], cwd=out, check=True, timeout=120)
    asm = subprocess.run(['objdump', '-d', str(so)], capture_output=True, text=True, check=True).stdout
    fn = ctypes.CDLL(str(so)).probe
    arr = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')
    fn.argtypes = [ctypes.c_int, ctypes.c_int, arr, arr, arr]
    return fn, dict(flags=flags, sha256=sha(so), fma_instructions=len(re.findall(r'\svfn?m(?:add|sub)\d', asm)))


def synthetic():
    """The predecessor's 4 regimes (proofs: s2-cammg CM78 shinhong_probe.py)."""
    n = 16
    for case in range(4):
        dz = np.full(n, 80. if case != 3 else 200.); z = np.cumsum(dz) - dz / 2
        pressure = 100000 * np.exp(-z / 8500); exner = (pressure / 1.e5) ** (287 / 1004.5)
        theta = 290 + (.004 if case == 0 else .0004) * z + np.maximum(z - 900, 0) * .005
        u = 4 + z * .002; v = 1 + z * .001
        tke = np.full(n, .005) if case < 2 else .1 + .25 * np.exp(-z / 1000)
        p = np.zeros((n + 1, 11), np.float32, order='F')
        for j, vv in enumerate((u, v, theta * exner, np.full(n, .005), pressure,
                                np.r_[100000, .5 * (pressure[:-1] + pressure[1:])], exner, dz, tke)):
            p[:-1, j] = vv; p[-1, j] = vv[-1]
        p[-1, 5] = pressure[-1] - .5 * (pressure[-2] - pressure[-1])
        hfx = (-30, 150, 350, 100)[case]; br = (.1, -.05, -.1, -.03)[case]
        s = np.array([100000, .1, .4, hfx, 1.e-4, np.hypot(u[0], v[0]), br, 4., 4., 18., 1., u[0], v[0],
                      3000 if case < 3 else 500, 3000 if case < 3 else 500, 1.e-4, 1., 1., 0.], np.float32)
        yield dict(source='synthetic', case=case), p, s, case >= 2


def real_columns():
    rng = np.random.default_rng(20261015)
    for frame in FRAMES:
        with netCDF4.Dataset(C07 / f'wrfout_d01_{frame}') as d:
            v = {k: np.asarray(d.variables[k][0], np.float64) for k in (
                'U', 'V', 'T', 'P', 'PB', 'PH', 'PHB', 'QVAPOR', 'QCLOUD', 'QICE', 'PSFC', 'UST', 'HFX',
                'QFX', 'XLAND', 'U10', 'V10', 'F', 'PBLH', 'TSK')}
        nz, ny, nx = v['T'].shape
        pick = {}
        pick['max_hfx'] = np.unravel_index(np.argmax(v['HFX']), (ny, nx))
        pick['min_hfx'] = np.unravel_index(np.argmin(v['HFX']), (ny, nx))
        pick['max_qc_low'] = np.unravel_index(np.argmax(v['QCLOUD'][:15].sum(0)), (ny, nx))
        pick['max_qi_low'] = np.unravel_index(np.argmax(v['QICE'][:20].sum(0)), (ny, nx))
        pick['max_pblh'] = np.unravel_index(np.argmax(v['PBLH']), (ny, nx))
        for r in range(3):
            pick[f'random{r}'] = (int(rng.integers(2, ny - 2)), int(rng.integers(2, nx - 2)))
        for tag, (j, i) in pick.items():
            j, i = int(j), int(i)
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
            prof = np.zeros((nz + 1, 11), np.float32, order='F')
            for col, val in enumerate((u, vv, th * pi, v['QVAPOR'][:, j, i], p, None, pi, dz,
                                       np.full(nz, .005), v['QCLOUD'][:, j, i], v['QICE'][:, j, i])):
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
            s = np.array([v['PSFC'][j, i], znt, max(v['UST'][j, i], .01), v['HFX'][j, i], v['QFX'][j, i],
                          wspd, br, fm, fh, 18., v['XLAND'][j, i], v['U10'][j, i], v['V10'][j, i], 3000., 3000.,
                          v['F'][j, i], 1., 1., v['PBLH'][j, i]], np.float32)
            yield dict(source='c07', frame=frame, pick=tag, j=j, i=i), prof, s, False


def main():
    t0 = time.monotonic()
    fns, builds = {}, {}
    for name, flags in VARIANTS.items():
        fns[name], builds[name] = build(name, flags)
    rows, profiles, surfaces, modes, outs = [], [], [], [], {k: [] for k in VARIANTS}
    for meta, prof, surf, _ in list(synthetic()) + list(real_columns()):
        n = prof.shape[0] - 1
        cold = None
        for mode, label in ((0, 'mode0'), (1, 'mode1_cold'), (1, 'mode1_warm')):
            p = prof.copy(order='F')
            if label == 'mode1_warm':
                p[:-1, 8] = cold[:-1, 0]; p[-1, 8] = cold[-2, 0]
            for name, fn in fns.items():
                res = np.zeros((n + 1, 10), np.float32, order='F')
                fn(n, mode, p, surf.copy(), res)
                outs[name].append(res)
            if label == 'mode1_cold':
                cold = outs['zero'][-1]
            rows.append(dict(meta, mode=mode, label=label, nz=n))
            profiles.append(p); surfaces.append(surf.copy())
    nmax = max(p.shape[0] for p in profiles)
    pad = lambda a: np.pad(a, ((0, nmax - a.shape[0]), (0, 0)))
    np.savez(LANE / 'oracle.npz', profiles=np.stack([pad(p) for p in profiles]),
             surface=np.stack(surfaces), nz=np.array([r['nz'] for r in rows]),
             mode=np.array([r['mode'] for r in rows]),
             **{f'out_{k}': np.stack([pad(o) for o in v]) for k, v in outs.items()})
    zero = np.stack([pad(o) for o in outs['zero']]); inf = np.stack([pad(o) for o in outs['inf']])
    fma = np.stack([pad(o) for o in outs['fma']])
    manifest = dict(source_sha256=sha(SOURCE), wrapper_sha256=sha(WRAPPER), builds=builds, rows=rows,
                    init_sensitivity_byte_identical=bool(zero.tobytes() == inf.tobytes()),
                    init_sensitivity_max_abs=float(np.nanmax(np.abs(zero - inf))),
                    fma_vs_zero_tke_max_rel=float(np.max(np.abs(fma[:, :, 0] - zero[:, :, 0]) / np.maximum(np.abs(zero[:, :, 0]), 1e-8))),
                    nonfinite={k: int(np.count_nonzero(~np.isfinite(np.stack([pad(o) for o in v])))) for k, v in outs.items()},
                    oracle_npz_sha256=sha(LANE / 'oracle.npz'), elapsed_s=time.monotonic() - t0, GPU=False)
    (LANE / 'oracle_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main()
