"""Pristine-WRF BL_GFS oracle on real columns of the ORIGINAL CPU-WRF c04 run (lane t-cfix-sfc, no JAX).

c04 = Swiss 2023-01-15, sf_sfclay_physics=3 / bl_pbl_physics=3: 44-level profiles (U V T P
PB PH PHB QVAPOR QCLOUD QICE) at 7 frames, 8 picks each (HFX/UST extremes + random), plus a
calm and a warm-skin variant of two picks per frame. c04 has no QICE (mp=3): the P_QI arm
uses a constructed ice tracer on the T<258 K levels.  The surface forcing is the WRF chain
itself: the UNMODIFIED SF_GFS oracle (sf3 zero build) is called on each column's lowest
level and its UST/PSIM/PSIH/GZ1OZ0/BR/WSPD/HFX/QFX feed the UNMODIFIED BL_GFS public call
(no LSM between, as surface_driver -> pbl_driver with sf_surface_physics=0), DT=dtbl=18 s,
with and without the QI tracer (P_QI).  Builds zero/inf/fma; +-1 ULP input envelope.
usage: python make_oracle_blgfs.py <lane_dir>   -> <lane>/oracle/oracle_blgfs.npz
"""
from pathlib import Path
import ctypes, hashlib, json, sys
import numpy as np
import netCDF4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_oracle import IN as SF_IN, OUT as SF_OUT  # noqa: E402

LANE = Path(sys.argv[1])
CPU = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c04/cpu')
FRAMES = [f'2023-01-15_{h:02d}:00:00' for h in (3, 6, 9, 12, 15, 18, 21)]
R_D, CP, G = 287.0, 1004.5, 9.81
PROF = 'u v th t qv qc qi p pi dz z'.split()
SFC = 'psfc ust pbl psim psih hfx qfx tsk gz1oz0 wspd br'.split()
TEND = 'rublten rvblten rthblten rqvblten rqcblten rqiblten'.split()
DIAG = 'ust pbl wspd kpbl'.split()
DT = 18.0
DRAWS = 64


def lib(path, argtypes):
    fn = ctypes.CDLL(str(path)).probe
    fn.argtypes = argtypes
    return fn


A = np.ctypeslib.ndpointer(dtype=np.float32, flags='F_CONTIGUOUS')


def sf_gfs(fn, inp):
    out = np.zeros((inp.shape[0], len(SF_OUT)), np.float32, order='F')
    fn(inp.shape[0], 2, np.asfortranarray(inp), out)
    return out


def columns(sf):
    rng = np.random.default_rng(20261016)
    profs, sfcs, labels = [], [], []
    for frame in FRAMES:
        with netCDF4.Dataset(CPU / f'wrfout_d01_{frame}') as d:
            v = {k: np.asarray(d.variables[k][0], np.float64) for k in (
                'U', 'V', 'T', 'P', 'PB', 'PH', 'PHB', 'QVAPOR', 'QCLOUD', 'PSFC', 'TSK', 'UST', 'HFX',
                'PBLH', 'XLAND')}
            # c04 is mp=3 (no QICE slot): the P_QI arm gets an ice tracer from the cold cloud
            # water (T < 258 K) so the third MONINP tracer is exercised on real structure.
            t_k = (v['T'] + 300.) * ((v['P'] + v['PB']) / 1.e5) ** (R_D / CP)
            v['QICE'] = np.where(t_k < 258., v['QCLOUD'] + 2.e-6, 0.)
        nz, ny, nx = v['T'].shape
        pick = {'max_hfx': np.argmax(v['HFX']), 'min_hfx': np.argmin(v['HFX']),
                'max_ust': np.argmax(v['UST']), 'min_ust': np.argmin(v['UST'])}
        for r in range(4):
            pick[f'random{r}'] = int(rng.integers(0, ny * nx))
        for tag, flat in pick.items():
            j, i = np.unravel_index(int(flat), (ny, nx))
            p = (v['P'] + v['PB'])[:, j, i]
            zw = (v['PH'] + v['PHB'])[:, j, i] / G
            th = v['T'][:, j, i] + 300.
            pi = (p / 1.e5) ** (R_D / CP)
            u = .5 * (v['U'][:, j, i] + v['U'][:, j, i + 1]); vv = .5 * (v['V'][:, j, i] + v['V'][:, j + 1, i])
            base = np.zeros((nz + 1, len(PROF)))
            for c, val in enumerate((u, vv, th, th * pi, v['QVAPOR'][:, j, i], v['QCLOUD'][:, j, i],
                                     v['QICE'][:, j, i], p, pi, np.diff(zw), .5 * (zw[:-1] + zw[1:]))):
                base[:-1, c] = val; base[-1, c] = val[-1]
            variants = [('profile', base, v['TSK'][j, i])]
            if tag in ('random0', 'max_hfx'):
                calm = base.copy(); calm[:, 0] *= .05; calm[:, 1] *= .05
                variants.append(('calm', calm, v['TSK'][j, i]))
                variants.append(('warm_skin', base, base[0, 3] + 6.))
            for surface, prof, tsk in variants:
                profs.append(prof)
                sfcs.append(dict(psfc=v['PSFC'][j, i], tsk=tsk, pbl=v['PBLH'][j, i], ust=max(v['UST'][j, i], .05),
                                 znt=float(rng.choice([.05, .1, .3]))))
                labels.append(dict(frame=frame, pick=tag, j=int(j), i=int(i), surface=surface))
    prof = np.array(profs, np.float32)
    # WRF chain: SF_GFS (pristine) on the lowest level supplies the BL_GFS surface operands.
    sf_in = np.zeros((prof.shape[0], len(SF_IN)), np.float32)
    for k, src in (('u', 'u'), ('v', 'v'), ('t', 't'), ('th', 'th'), ('qv', 'qv'), ('p', 'p'), ('dz', 'dz')):
        sf_in[:, SF_IN.index(k)] = prof[:, 0, PROF.index(src)]
    for k in ('psfc', 'tsk', 'ust', 'znt', 'pblh'):
        sf_in[:, SF_IN.index(k)] = [s[k if k != 'pblh' else 'pbl'] for s in sfcs]
    sf_in[:, SF_IN.index('xland')] = 1.; sf_in[:, SF_IN.index('mavail')] = 1.; sf_in[:, SF_IN.index('dx')] = 3000.
    so = sf_gfs(sf, sf_in)
    sfc = np.zeros((prof.shape[0], len(SFC)), np.float32)
    sfc[:, SFC.index('psfc')] = sf_in[:, SF_IN.index('psfc')]
    sfc[:, SFC.index('tsk')] = sf_in[:, SF_IN.index('tsk')]
    sfc[:, SFC.index('pbl')] = sf_in[:, SF_IN.index('pblh')]
    for k in ('ust', 'psim', 'psih', 'hfx', 'qfx', 'gz1oz0', 'wspd', 'br'):
        sfc[:, SFC.index(k)] = so[:, SF_OUT.index(k)]
    return prof, sfc, labels


def call(fn, prof, sfc, has_qi):
    n, nz = prof.shape[0], prof.shape[1] - 1
    tend = np.zeros((n, nz, len(TEND)), np.float32, order='F')
    diag = np.zeros((n, len(DIAG)), np.float32, order='F')
    fn(n, nz, has_qi, DT, np.asfortranarray(prof), np.asfortranarray(sfc), tend, diag)
    return tend, diag


def nudge(a, rng):
    step = rng.integers(-1, 2, size=a.shape)
    return np.where(step > 0, np.nextafter(a, np.float32(np.inf)),
                    np.where(step < 0, np.nextafter(a, np.float32(-np.inf)), a)).astype(np.float32)


def main():
    sf = lib(LANE / 'oracle' / 'sf3' / 'zero' / 'oracle.so', [ctypes.c_int, ctypes.c_int, A, A])
    prof, sfc, labels = columns(sf)
    args = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, A, A, A, A]
    fns = {v: lib(LANE / 'oracle' / 'pbl3' / v / 'oracle.so', args) for v in ('zero', 'inf', 'fma')}
    result = {'prof': prof, 'sfc': sfc}
    rng = np.random.default_rng(20261017)
    for has_qi in (1, 0):
        for v, fn in fns.items():
            result[f'q{has_qi}_{v}_tend'], result[f'q{has_qi}_{v}_diag'] = call(fn, prof, sfc, has_qi)
        tends = [result[f'q{has_qi}_zero_tend'], result[f'q{has_qi}_fma_tend']]
        diags = [result[f'q{has_qi}_zero_diag'], result[f'q{has_qi}_fma_diag']]
        for _ in range(DRAWS):
            t, d = call(fns['zero'], nudge(prof, rng), nudge(sfc, rng), has_qi)
            tends.append(t); diags.append(d)
        result[f'q{has_qi}_tend_lower'], result[f'q{has_qi}_tend_upper'] = np.min(tends, 0), np.max(tends, 0)
        result[f'q{has_qi}_diag_lower'], result[f'q{has_qi}_diag_upper'] = np.min(diags, 0), np.max(diags, 0)
        result[f'q{has_qi}_init_byte_identical'] = np.array(
            np.array_equal(result[f'q{has_qi}_zero_tend'], result[f'q{has_qi}_inf_tend']))
    path = LANE / 'oracle' / 'oracle_blgfs.npz'
    np.savez_compressed(path, draws=DRAWS, dt=DT, **result)
    meta = dict(prof=PROF, sfc=SFC, tend=TEND, diag=DIAG, labels=labels, dt=DT, draws=DRAWS,
                npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (LANE / 'oracle' / 'oracle_blgfs.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(prof.shape, {k: bool(result[k]) for k in result if k.endswith('init_byte_identical')},
          'finite', bool(np.isfinite(result['q1_zero_tend']).all()), meta['npz_sha256'])


if __name__ == '__main__':
    main()
